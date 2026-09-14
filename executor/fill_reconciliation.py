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
* **Snapshot capture** — the postclose ``CaptureSnapshot`` stage opens a fresh
  IB session; ``ib_insync`` syncs the day's executions on connect, so the same
  call is the EOD backstop for a daemon that died between the fill and its
  next tick.
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
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

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
_EXEC_RE = re.compile(
    r"Execution\(execId='(?P<exec_id>[^']+)', time=datetime\.datetime\("
    r"(?P<y>\d+), (?P<mo>\d+), (?P<d>\d+), (?P<h>\d+), (?P<mi>\d+), (?P<s>\d+)"
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
                    int(m["h"]), int(m["mi"]), int(m["s"]),
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


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--date", required=True, help="trades.date session to reconcile (YYYY-MM-DD)")
    p.add_argument("--from-daemon-log", required=True, help="path to that day's daemon log")
    p.add_argument("--db", default=None, help="trades.db path (default: config db_path)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    from executor.trade_logger import init_db

    db_path = args.db
    if db_path is None:
        from executor.config_loader import load_config

        db_path = load_config()["db_path"]
    with open(args.from_daemon_log, encoding="utf-8", errors="replace") as fh:
        fills = parse_daemon_log_executions(fh)
    conn = init_db(db_path)
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
