"""Reconcile ``trades`` rows that were logged before their broker fill arrived.

``IBKRClient.place_market_order`` polls a bounded window (30 s) for the fill and
returns ``status="Working"`` with ``fill_price=None`` when the order is still
open at the end of it. A market SELL placed at the 09:30 ET open routinely
takes longer than that (the opening cross; IB's regulatory price cap on the
first prints) — measured 2026-08-31, 2026-09-10 and 2026-09-14, each a REDUCE
that filled 40–80 s after the window closed. The daemon then logged the row
with the *estimate* it had (``price_at_order``) standing in for the fill and
nothing ever revisited the row: the broker's executions arrived on the IB
session as ``execDetails`` events and were written only to the daemon log.

The downstream cost is in the EOD attribution. ``compute_rotation_realized``
prices the day's rotation sleeve from those rows, so a REDUCE whose real fill
sat $3.33 below its estimate (PBF, 775 shares, 2026-09-14) put $2,581 into the
unattributed residual, and the residual bounds gate — correctly — failed the
postclose pipeline (``alpha-engine-config-I10800``).

This module closes the loop on three surfaces, all reading the SAME broker
execution record:

* **Daemon tick** — ``reconcile_unfilled_trades(conn, run_date,
  fills_by_order_from_ib(ib))`` after each poll, so a row is corrected within
  one poll interval of its fill arriving (``ib.fills()`` holds every execution
  the session has seen).
* **Snapshot capture** — the postclose ``CaptureSnapshot`` stage is the EOD
  backstop for a daemon that died between the fill and its next tick. Its
  fresh IB session connects on a DIFFERENT clientId than the daemon, and the
  executions ``ib_insync`` syncs on connect are that client's own — measured
  on the 2026-09-14 replay: ``patched=0``, both open rows "no execution on the
  session" while their executions sat in the daemon log. So the backstop reads
  the session first and then the box's own daemon log
  (``reconcile_from_daemon_log_file`` over ``LOCAL_DAEMON_LOG_PATH``), which
  carries every execution the daemon's session saw.
* **Daemon log replay** — ``parse_daemon_log_executions`` reads the
  ``execDetails`` / ``commissionReport`` lines the IB wrapper logs, for a day
  whose IB session is gone (the log is backed up to S3 every EOD, so it is the
  durable execution record). ``python -m executor.fill_reconciliation --date D
  --from-daemon-log PATH`` is the repair path for 2026-09-14 and any prior day.

A row that no source can resolve stays non-terminal and is reported by
``unresolved_trades`` so the EOD run can name it in ``data_warnings`` — the
attribution then falls back to the arrival price *visibly*, never silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import boto3

logger = logging.getLogger(__name__)

# Written by `eod_reconcile` at the end of every session (the literal lives at
# its call site; this is the only other reader). The log is the durable
# execution record for a day whose IB session is long gone.
DAEMON_LOG_KEY_TEMPLATE = "trades/logs/{run_date}/daemon.log"

# The daemon's live log on the trading box — the file `eod_reconcile` uploads
# to DAEMON_LOG_KEY_TEMPLATE. Read in place by the snapshot-capture backstop,
# which runs before that upload.
LOCAL_DAEMON_LOG_PATH = "/var/log/daemon.log"

# One marker per swept session. Its purpose is NEGATIVE caching: a row that
# can never resolve — an order genuinely cancelled before any fill — would
# otherwise re-download a ~100 MB log on every EOD run, forever. The marker
# records what the sweep found so the next run can skip the date and a human
# can see why it was left alone.
SWEEP_MARKER_KEY_TEMPLATE = "trades/fill_reconciliation/{run_date}.json"

# Per-run cap on archived logs pulled from S3. The daemon logs run 35-100 MB
# each; a backlog drains over consecutive sessions instead of making one EOD
# run pay for all of it.
DEFAULT_SWEEP_MAX_DATES = 3

# Statuses ``IBKRClient.place_market_order`` / ``place_bracket_with_stop`` emit
# for an order that may still fill (or fill further) after the row was written.
# ``Rejected`` is terminal and ``Filled`` is complete; neither is revisited.
NON_TERMINAL_STATUSES: tuple[str, ...] = ("Working", "PartialFill", "Timeout")

_SELL_SIDES = {"SLD", "SELL"}
_BUY_SIDES = {"BOT", "BUY"}


# ─────────────────────────────────────────────────────────────────────────────
# Execution sources — every source yields {order_id: [execution, ...]} where an
# execution is {"symbol", "side", "shares", "price", "time", "commission",
# "exec_id"}; ``commission`` is None when the broker reported none.
# ─────────────────────────────────────────────────────────────────────────────


def fills_by_order_from_ib(ib) -> dict[int, list[dict[str, Any]]]:
    """Group an ``ib_insync.IB`` session's ``fills()`` by IB order id.

    ``ib.fills()`` is the in-memory execution log of the session: every fill
    for an order placed on this connection, plus — because ``ib_insync``
    issues ``reqExecutions`` during connect — every execution of the current
    trading day when the session was opened after the fills happened (the
    snapshot-capture case). No request is made here; this reads state the
    session already holds, so it cannot stall the way a live ``reqExecutions``
    can (see ``IBKRClient._connect``).
    """
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fill in ib.fills():
        ex = fill.execution
        rep = getattr(fill, "commissionReport", None)
        commission = None
        if rep is not None and getattr(rep, "commission", None) not in (None, ""):
            # ib_insync leaves a placeholder CommissionReport (execId="") on a
            # fill whose report has not arrived yet; its commission reads 0.0
            # but is UNKNOWN, not zero — same distinction fill_commission_usd
            # draws (alpha-engine-config-I8188).
            commission = abs(float(rep.commission)) if getattr(rep, "execId", "") else None
        out[int(ex.orderId)].append(
            {
                "symbol": getattr(fill.contract, "symbol", None),
                "side": ex.side,
                "shares": float(ex.shares),
                "price": float(ex.price),
                "time": ex.time.isoformat() if isinstance(ex.time, datetime) else (ex.time or None),
                "commission": commission,
                "exec_id": ex.execId,
            }
        )
    return dict(out)


# The IB wrapper logs each execution twice — once as the bare ``Execution``
# repr (carries orderId/shares/price/time) and once wrapped in ``Fill(contract=
# Stock(... symbol='PBF' ...), execution=Execution(...))`` (carries the symbol).
# The commission arrives later on its own ``commissionReport`` line keyed by
# execId. All three are ``repr`` output of ib_insync dataclasses, so the field
# order is stable across the versions the fleet pins.
#
# ``datetime.__repr__`` drops a zero second (and a zero microsecond), so an
# execution at 13:32:00 logs as ``datetime.datetime(2026, 9, 14, 13, 32,
# tzinfo=...)``. Seconds are therefore optional: requiring them silently
# dropped PBF's 75-share 13:32:00 execution on 2026-09-14 and left the row a
# PartialFill of 700/775.
_EXEC_RE = re.compile(
    r"Execution\(execId='(?P<exec_id>[^']+)', time=datetime\.datetime\("
    r"(?P<y>\d+), (?P<mo>\d+), (?P<d>\d+), (?P<h>\d+), (?P<mi>\d+)(?:, (?P<s>\d+))?"
    r"(?:, (?P<us>\d+))?(?:, tzinfo=[^)]*)?\), acctNumber='[^']*', exchange='[^']*', "
    r"side='(?P<side>[A-Z]+)', shares=(?P<shares>[\d.]+), price=(?P<price>[\d.]+), "
    r"permId=\d+, clientId=\d+, orderId=(?P<order_id>\d+)"
)
_FILL_SYMBOL_RE = re.compile(r"Fill\(contract=Stock\([^)]*symbol='(?P<symbol>[A-Z.\-]+)'")
_COMMISSION_RE = re.compile(
    r"CommissionReport\(execId='(?P<exec_id>[^']+)', commission=(?P<commission>-?[\d.]+)"
)


def parse_daemon_log_executions(lines: Iterable[str]) -> dict[int, list[dict[str, Any]]]:
    """Rebuild ``{order_id: [execution, ...]}`` from daemon log lines.

    Accepts the raw JSON-lines the daemon writes (the ``msg`` field is what is
    matched, but matching the whole line is equivalent and tolerates a plain-
    text log). Executions are de-duplicated by ``exec_id`` — the same execution
    appears on both the bare and the ``Fill(...)`` line.
    """
    by_exec: dict[str, dict[str, Any]] = {}
    symbol_by_exec: dict[str, str] = {}
    commission_by_exec: dict[str, float] = {}
    for line in lines:
        m = _EXEC_RE.search(line)
        if m:
            exec_id = m["exec_id"]
            if exec_id not in by_exec:
                ts = datetime(
                    int(m["y"]), int(m["mo"]), int(m["d"]),
                    int(m["h"]), int(m["mi"]), int(m["s"] or 0),
                )
                by_exec[exec_id] = {
                    "exec_id": exec_id,
                    "order_id": int(m["order_id"]),
                    "side": m["side"],
                    "shares": float(m["shares"]),
                    "price": float(m["price"]),
                    "time": ts.isoformat() + "+00:00",
                    "symbol": None,
                    "commission": None,
                }
            sm = _FILL_SYMBOL_RE.search(line)
            if sm:
                symbol_by_exec[exec_id] = sm["symbol"]
            continue
        cm = _COMMISSION_RE.search(line)
        if cm:
            commission_by_exec[cm["exec_id"]] = abs(float(cm["commission"]))
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for exec_id, ex in by_exec.items():
        ex["symbol"] = symbol_by_exec.get(exec_id)
        ex["commission"] = commission_by_exec.get(exec_id)
        out[ex.pop("order_id")].append(ex)
    return dict(out)


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation
# ─────────────────────────────────────────────────────────────────────────────


def unresolved_trades(conn: sqlite3.Connection, run_date: str) -> list[dict[str, Any]]:
    """Rows for ``run_date`` whose status still admits a later fill."""
    placeholders = ",".join("?" for _ in NON_TERMINAL_STATUSES)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM trades WHERE date=? AND status IN ({placeholders}) ORDER BY created_at",  # noqa: S608 — placeholders are "?" marks, values bound below
            (run_date, *NON_TERMINAL_STATUSES),
        ).fetchall()
    finally:
        conn.row_factory = None
    return [dict(r) for r in rows]


def _side_matches(action: str, side: str | None) -> bool:
    if not side:
        return True
    action = (action or "").upper()
    if action in ("ENTER", "BUY", "COVER"):
        return side in _BUY_SIDES
    return side in _SELL_SIDES


def _summarise(executions: list[dict[str, Any]]) -> dict[str, Any]:
    qty = sum(e["shares"] for e in executions)
    cost = sum(e["shares"] * e["price"] for e in executions)
    reported = [e["commission"] for e in executions if e.get("commission") is not None]
    times = [e["time"] for e in executions if e.get("time")]
    return {
        "filled_shares": int(round(qty)),
        "fill_price": round(cost / qty, 4) if qty > 0 else None,
        # Commission is a sum over executions that REPORTED one; a fill with
        # no report leaves the total a lower bound, which is still a measured
        # figure — None only when no execution reported at all.
        "commission_usd": round(sum(reported), 6) if reported else None,
        "fill_time": max(times) if times else None,
        "n_executions": len(executions),
    }


def _roundtrip_fields(
    conn: sqlite3.Connection, row: Mapping[str, Any], fill_price: float, filled_shares: int
) -> dict[str, Any]:
    """Recompute the realized columns a sell row carries from its entry.

    They were computed at log time from the ESTIMATE, so they are wrong by the
    same amount the fill was. ``spy_return_during_hold`` does not depend on the
    fill and is kept; alpha is re-derived from it.
    """
    entry_id = row.get("entry_trade_id")
    if not entry_id:
        return {}
    entry = conn.execute(
        "SELECT fill_price FROM trades WHERE trade_id=?", (entry_id,)
    ).fetchone()
    entry_fill = entry[0] if entry else None
    if not entry_fill:
        return {}
    entry_fill = float(entry_fill)
    rpnl = (fill_price - entry_fill) * filled_shares
    rpct = (fill_price / entry_fill - 1.0) * 100.0
    spy_ret = row.get("spy_return_during_hold")
    ralpha = (rpct - float(spy_ret)) if spy_ret is not None else None
    return {
        "realized_pnl": rpnl,
        "realized_return_pct": rpct,
        "realized_alpha_pct": ralpha,
    }


def reconcile_unfilled_trades(
    conn: sqlite3.Connection,
    run_date: str,
    fills_by_order: Mapping[int, list[dict[str, Any]]],
    *,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Patch every non-terminal ``trades`` row for ``run_date`` that the broker
    record can resolve.

    Returns ``{"patched": [...], "unresolved": [...]}``. A patch carries the
    before/after of every column it changed so the caller can log it as the
    correction record it is. Idempotent: a row already terminal is not read,
    and a second run over the same fills changes nothing.
    """
    patched: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in unresolved_trades(conn, run_date):
        order_id = row.get("ib_order_id")
        executions = list(fills_by_order.get(int(order_id), [])) if order_id not in (None, "") else []
        executions = [
            e for e in executions
            if _side_matches(row["action"], e.get("side"))
            and (not e.get("symbol") or e["symbol"] == row["ticker"])
        ]
        if not executions:
            unresolved.append(row)
            continue
        summary = _summarise(executions)
        if summary["fill_price"] is None:
            unresolved.append(row)
            continue
        requested = int(row.get("shares") or 0)
        new_status = "Filled" if summary["filled_shares"] >= requested else "PartialFill"
        patch: dict[str, Any] = {
            "fill_price": summary["fill_price"],
            "filled_shares": summary["filled_shares"],
            "fill_time": summary["fill_time"],
            "status": new_status,
        }
        if summary["commission_usd"] is not None:
            patch["commission_usd"] = summary["commission_usd"]
        if (row.get("action") or "").upper() not in ("ENTER", "BUY", "COVER"):
            patch.update(_roundtrip_fields(conn, row, summary["fill_price"], summary["filled_shares"]))
        before = {k: row.get(k) for k in patch}
        changed = {k: v for k, v in patch.items() if before.get(k) != v}
        record = {
            "trade_id": row["trade_id"],
            "ticker": row["ticker"],
            "action": row["action"],
            "ib_order_id": order_id,
            "n_executions": summary["n_executions"],
            "before": before,
            "after": patch,
        }
        logger.info(
            "fill reconciled: %s %s %s order=%s %s→%s fill_price %s→%s filled_shares %s→%s%s",
            row["action"], row.get("shares"), row["ticker"], order_id,
            row.get("status"), new_status, row.get("fill_price"), patch["fill_price"],
            row.get("filled_shares"), patch["filled_shares"],
            " [dry-run]" if dry_run else "",
        )
        if changed and not dry_run:
            sets = ", ".join(f"{k}=?" for k in changed)
            conn.execute(
                f"UPDATE trades SET {sets} WHERE trade_id=?",  # noqa: S608 — column names come from this module's own patch dict, values bound
                (*changed.values(), row["trade_id"]),
            )
            conn.commit()
        patched.append(record)
    return {"patched": patched, "unresolved": unresolved}


def reconcile_from_ib(conn: sqlite3.Connection, run_date: str, ib) -> dict[str, list[dict[str, Any]]]:
    """``reconcile_unfilled_trades`` over the live session's fills — skips the
    IB read entirely when there is nothing to resolve, so the per-tick call in
    the daemon costs one SQL query on a normal day."""
    if not unresolved_trades(conn, run_date):
        return {"patched": [], "unresolved": []}
    return reconcile_unfilled_trades(conn, run_date, fills_by_order_from_ib(ib))


def reconcile_from_daemon_log_file(
    conn: sqlite3.Connection, run_date: str, path: str
) -> dict[str, list[dict[str, Any]]]:
    """``reconcile_unfilled_trades`` over a daemon log file on disk, restricted
    to executions timestamped on ``run_date``.

    The live log on the box spans more than one session, and IB order ids are
    per-client counters rather than globally unique, so an execution from
    another day must not be able to resolve today's row. Raises
    ``FileNotFoundError`` when the log is absent — the caller decides whether
    that is fatal. Skips the read entirely when nothing is open."""
    if not unresolved_trades(conn, run_date):
        return {"patched": [], "unresolved": []}
    with open(path, encoding="utf-8", errors="replace") as fh:
        fills = parse_daemon_log_executions(fh)
    same_day = {
        order_id: kept
        for order_id, executions in fills.items()
        if (kept := [e for e in executions if (e.get("time") or "")[:10] == run_date])
    }
    return reconcile_unfilled_trades(conn, run_date, same_day)


# ─────────────────────────────────────────────────────────────────────────────
# Historical sweep — repair sessions whose rows were never reconciled
# ─────────────────────────────────────────────────────────────────────────────


def unresolved_session_dates(conn: sqlite3.Connection, *, before: str) -> list[str]:
    """Ascending distinct ``trades.date`` values before ``before`` holding a
    non-terminal row. ``before`` is exclusive so the current session, which the
    live passes own, is never swept from the archive."""
    placeholders = ",".join("?" for _ in NON_TERMINAL_STATUSES)
    rows = conn.execute(
        f"SELECT DISTINCT date FROM trades WHERE date < ? AND status IN ({placeholders}) "  # noqa: S608 — placeholders are "?" marks, values bound below
        "ORDER BY date",
        (before, *NON_TERMINAL_STATUSES),
    ).fetchall()
    return [r[0] for r in rows]


def _s3(region: str):
    return boto3.client("s3", region_name=region)


def _sweep_marker(bucket: str, run_date: str, region: str) -> dict | None:
    """The prior sweep's record for ``run_date``, or None if never swept."""
    try:
        body = _s3(region).get_object(
            Bucket=bucket, Key=SWEEP_MARKER_KEY_TEMPLATE.format(run_date=run_date)
        )["Body"].read()
        return json.loads(body)
    except Exception:  # noqa: BLE001 — absent or unreadable marker means "not swept yet", which is the safe reading: it costs one re-download, never a missed repair
        return None


def _write_sweep_marker(bucket: str, run_date: str, region: str, record: dict) -> None:
    try:
        _s3(region).put_object(
            Bucket=bucket,
            Key=SWEEP_MARKER_KEY_TEMPLATE.format(run_date=run_date),
            Body=json.dumps(record, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as err:  # noqa: BLE001 — (a) the negative-cache write failed; (b) the repair itself already succeeded and is committed; (c) recorded at WARNING here, and the only cost is that the next run re-reads this date's log
        logger.warning(
            "fill reconciliation: sweep marker write failed for %s (%s) — "
            "the repair stands; the next sweep will re-read this date's log",
            run_date, err,
        )


def _archived_log_lines(bucket: str, run_date: str, region: str):
    """Stream the archived daemon log for ``run_date``, or None when absent.

    Streamed rather than read whole: these logs reach ~100 MB and this runs on
    the trading box alongside the snapshot capture.
    """
    key = DAEMON_LOG_KEY_TEMPLATE.format(run_date=run_date)
    try:
        body = _s3(region).get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as err:  # noqa: BLE001 — a missing archived log is a fact about that date, reported by the caller as `unavailable`, never an error that aborts the sweep
        logger.info(
            "fill reconciliation: no archived daemon log at s3://%s/%s (%s)",
            bucket, key, err.__class__.__name__,
        )
        return None
    return (line.decode("utf-8", errors="replace") for line in body.iter_lines())


def sweep_unresolved_sessions(
    conn: sqlite3.Connection,
    *,
    bucket: str,
    before: str,
    region: str = "us-east-1",
    max_dates: int = DEFAULT_SWEEP_MAX_DATES,
    force: bool = False,
) -> dict[str, Any]:
    """Repair prior sessions whose rows were never reconciled, from S3 logs.

    The live passes (daemon tick, snapshot capture) only ever see the session
    they run in, so every row written before this module existed is
    unreachable by them. This closes that back-catalogue from the archived
    daemon logs, which are the durable execution record.

    Returns ``{"changed": [...], "swept": [...], "unavailable": [...],
    "skipped": [...], "still_unresolved": {date: n}}``. ``changed`` is the
    dates whose rows actually moved — the caller re-derives those days'
    ``eod_pnl`` rows, because this function repairs the trade ledger and
    deliberately does not decide what is downstream of it.
    """
    result: dict[str, Any] = {
        "changed": [], "swept": [], "unavailable": [], "skipped": [], "still_unresolved": {},
    }
    dates = unresolved_session_dates(conn, before=before)
    if not dates:
        return result
    logger.info(
        "fill reconciliation sweep: %d prior session(s) hold an unresolved row: %s",
        len(dates), ", ".join(dates),
    )
    for run_date in dates:
        if len(result["swept"]) >= max_dates:
            logger.info(
                "fill reconciliation sweep: stopping at the %d-date cap; %d date(s) remain "
                "and will be picked up by the next run",
                max_dates, len(dates) - len(result["swept"]) - len(result["skipped"]),
            )
            break
        if not force and _sweep_marker(bucket, run_date, region) is not None:
            result["skipped"].append(run_date)
            continue
        lines = _archived_log_lines(bucket, run_date, region)
        if lines is None:
            result["unavailable"].append(run_date)
            continue
        outcome = reconcile_unfilled_trades(
            conn, run_date, parse_daemon_log_executions(lines)
        )
        result["swept"].append(run_date)
        if outcome["patched"]:
            result["changed"].append(run_date)
        if outcome["unresolved"]:
            result["still_unresolved"][run_date] = len(outcome["unresolved"])
            for row in outcome["unresolved"]:
                logger.warning(
                    "fill reconciliation sweep: %s %s %s %s (order %s) has no execution in "
                    "the archived log — the order never filled, or the log predates it",
                    run_date, row["action"], row["shares"], row["ticker"], row.get("ib_order_id"),
                )
        _write_sweep_marker(bucket, run_date, region, {
            "run_date": run_date,
            "swept_at": datetime.now(UTC).isoformat(),
            "patched": [
                {"ticker": r["ticker"], "action": r["action"], "ib_order_id": r["ib_order_id"],
                 "fill_price": r["after"].get("fill_price"),
                 "was": r["before"].get("fill_price")}
                for r in outcome["patched"]
            ],
            "unresolved": [
                {"ticker": r["ticker"], "action": r["action"], "shares": r["shares"],
                 "ib_order_id": r.get("ib_order_id"), "status": r.get("status")}
                for r in outcome["unresolved"]
            ],
        })
    logger.info(
        "fill reconciliation sweep: swept=%d changed=%s unavailable=%s skipped=%d",
        len(result["swept"]), result["changed"] or "none",
        result["unavailable"] or "none", len(result["skipped"]),
    )
    return result


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--date", help="trades.date session to reconcile (YYYY-MM-DD)")
    p.add_argument("--from-daemon-log", help="path to that day's daemon log")
    p.add_argument("--sweep", action="store_true",
                   help="repair every prior session with an unresolved row, from the S3 log archive")
    p.add_argument("--before", default=None,
                   help="sweep sessions strictly before this date (default: today's trading day)")
    p.add_argument("--max-dates", type=int, default=DEFAULT_SWEEP_MAX_DATES,
                   help="cap on archived logs pulled in one sweep")
    p.add_argument("--force", action="store_true",
                   help="re-sweep dates that already carry a sweep marker")
    p.add_argument("--db", default=None, help="trades.db path (default: config db_path)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.sweep and not (args.date and args.from_daemon_log):
        p.error("either --sweep, or both --date and --from-daemon-log, are required")

    from executor.trade_logger import init_db

    config = None
    db_path = args.db
    if db_path is None or args.sweep:
        from executor.config_loader import load_config

        config = load_config()
        db_path = db_path or config["db_path"]
    conn = init_db(db_path)

    if args.sweep:
        from nousergon_lib.dates import now_dual

        before = args.before or now_dual().trading_day
        sweep = sweep_unresolved_sessions(
            conn, bucket=config["trades_bucket"], before=before,
            region=config.get("aws_region", "us-east-1"),
            max_dates=args.max_dates, force=args.force,
        )
        for line in (
            f"swept:       {', '.join(sweep['swept']) or 'none'}",
            f"changed:     {', '.join(sweep['changed']) or 'none'}",
            f"unavailable: {', '.join(sweep['unavailable']) or 'none'}",
            f"skipped:     {len(sweep['skipped'])} already-swept date(s)",
        ):
            print(line)
        if sweep["changed"]:
            print("\nRe-derive those sessions' eod_pnl rows with:")
            for d in sweep["changed"]:
                print(f"  python -c \"from executor.eod_reconcile import run; "
                      f"run(run_date='{d}', send_email=False, run_audit=False)\"")
        return 1 if sweep["still_unresolved"] or sweep["unavailable"] else 0

    with open(args.from_daemon_log, encoding="utf-8", errors="replace") as fh:
        fills = parse_daemon_log_executions(fh)
    result = reconcile_unfilled_trades(conn, args.date, fills, dry_run=args.dry_run)
    for rec in result["patched"]:
        print(
            f"{rec['action']} {rec['ticker']} order={rec['ib_order_id']}: "
            f"{rec['before']} -> {rec['after']}"
        )
    for row in result["unresolved"]:
        print(
            f"UNRESOLVED {row['action']} {row['shares']} {row['ticker']} "
            f"order={row.get('ib_order_id')} status={row.get('status')}"
        )
    print(f"patched={len(result['patched'])} unresolved={len(result['unresolved'])}"
          f"{' [dry-run]' if args.dry_run else ''}")
    return 1 if result["unresolved"] else 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(_main())
