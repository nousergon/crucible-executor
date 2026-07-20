"""backfill_eod_pnl — reconstruct a MISSING daily eod_pnl row by ledger synthesis.

When a weekday/EOD Step Function is skipped, the EOD reconcile never runs for
that trading day and no `eod_pnl` row (NAV / daily return / positions
snapshot) is written. The next day's reconcile then has no adjacent prior-day
NAV baseline and the headline daily return/alpha span multiple sessions (the
2026-06-24 → RGEN +14.92% class; config#1228/#1229).

The same-day recovery path (a live IB snapshot via `snapshot_capturer`) only
works while the trading box is up — `snapshot_capturer` hard-refuses a past
date because IB's account state is now-as-of, and the TWS/Gateway API has no
historical-account endpoint. So a day discovered LATE (box gone) cannot be
recovered from IBKR at all.

This tool recovers it WITHOUT IBKR, from data we already store durably:
  * positions(D)  — ANCHORED on the prior day's RECONCILED `positions_snapshot`
                    (broker-book-aligned) + ONLY that day's fills/share-deltas,
                    so error never compounds across the replay (config#1281).
                    A free-running full-ledger replay (the legacy
                    `replay_positions`) drifts from the broker book — closed/
                    partial/fractional positions don't fully net — and produced
                    a badly-wrong NAV (config#1276: 20 synthesized positions /
                    $1.49M vs the real 7 / ~$0.98M on 2026-06-24). It survives
                    only as the missing-anchor cold-start fallback (flagged).
  * closes(D)     — ArcticDB universe/macro closes (gapless after the
                    market-data auto-heal, config#1228),
  * NAV/cash(D)   — rolled forward from the prior `eod_pnl` row:
                    cash(D) = cash(D-1) + D's trade cash-flows;
                    NAV(D)  = cash(D) + Σ shares(D)·close_D.

A reconciliation guard refuses to write (flags the gap) when the synthesized
position count diverges materially from the prior reconciled snapshot's, rather
than persisting a wrong NAV.

It synthesizes a `trades/snapshots/{D}.json` byte-compatible with what
`snapshot_capturer` writes (marked ``synthesized: true`` for audit) and then
runs the canonical `eod_reconcile.run(D)` — which re-prices every position
from ArcticDB itself, recomputes the NAV reconciliation, and writes the row.
Any reconstruction error (unmodelled fee, dividend, FX) surfaces in the
reconcile's existing `unattributed` residual rather than silently.

For the common late-miss — a halt day on which the box never traded — the
synthesis is EXACT: no trades → cash unchanged, positions = prior day's, and
NAV is just the prior book re-marked at D's closes.

Usage:
    python -m executor.backfill_eod_pnl --date 2026-06-24
    python -m executor.backfill_eod_pnl --date 2026-06-24 --dry-run
    python -m executor.backfill_eod_pnl --date 2026-06-24 --force   # overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import UTC, datetime

import boto3

from executor.config_loader import load_config
from executor.trade_logger import init_db

logger = logging.getLogger(__name__)

# Actions that change a position's share count. ENTER adds; EXIT closes the
# remaining position; REDUCE trims. (Mirrors the executor's order vocabulary.)
_BUY_ACTIONS = ("ENTER",)
_SELL_ACTIONS = ("EXIT", "REDUCE")


# If the anchored position count diverges from the prior reconciled snapshot's
# by MORE than this fraction (and by more than the absolute floor below), the
# synthesis is presumed broken and refuses to write — we'd rather flag a gap
# than persist a wrong NAV (the config#1276 / 2026-06-24 failure mode: 20
# synthesized positions vs the real 7). A small drift (a single fully-exited or
# newly-opened name on the day) is normal and allowed.
_DIVERGENCE_FRACTION = 0.5
_DIVERGENCE_ABS_FLOOR = 3


def replay_positions(conn: sqlite3.Connection, as_of_date: str) -> dict[str, int]:
    """DEPRECATED free-running anchor (config#1281). Net shares held per ticker
    at the close of ``as_of_date`` by **cumulatively netting the entire `trades`
    ledger** (all fills with date <= as_of_date).

    This drifts from the broker's actual book — closed/partial/fractional
    positions don't fully net across a long ledger — so the synthesized NAV can
    be badly wrong (config#1276: 20 synthesized positions / NAV $1.49M vs the
    real 7 positions / ~$0.98M on 2026-06-24). It is retained only as the
    missing-anchor FALLBACK in :func:`synthesize_positions`; the primary path
    now anchors on the prior reconciled snapshot. Do not use directly.

    Uses the actually-filled share count (``filled_shares`` when present, else
    ``shares``). Returns only tickers with a positive net position."""
    rows = conn.execute(
        "SELECT ticker, action, COALESCE(filled_shares, shares) AS sh "
        "FROM trades WHERE date <= ? ORDER BY date, created_at",
        (as_of_date,),
    ).fetchall()
    net: dict[str, int] = {}
    for ticker, action, sh in rows:
        if sh is None:
            continue
        sh = int(sh)
        if action in _BUY_ACTIONS:
            net[ticker] = net.get(ticker, 0) + sh
        elif action in _SELL_ACTIONS:
            net[ticker] = net.get(ticker, 0) - sh
    return {t: q for t, q in net.items() if q > 0}


def day_share_deltas(conn: sqlite3.Connection, as_of_date: str) -> dict[str, int]:
    """Net share delta per ticker from fills **ON** ``as_of_date`` only.

    ENTER adds shares; EXIT/REDUCE remove them. Uses the actually-filled count
    (``filled_shares`` when present, else ``shares``). This is the day's change
    to the book — applied to the prior reconciled snapshot in
    :func:`synthesize_positions` so error never compounds across the replay.
    Returns {} when no trades executed that day (the common halt case)."""
    rows = conn.execute(
        "SELECT ticker, action, COALESCE(filled_shares, shares) AS sh "
        "FROM trades WHERE date = ? ORDER BY created_at",
        (as_of_date,),
    ).fetchall()
    delta: dict[str, int] = {}
    for ticker, action, sh in rows:
        if sh is None:
            continue
        sh = int(sh)
        if action in _BUY_ACTIONS:
            delta[ticker] = delta.get(ticker, 0) + sh
        elif action in _SELL_ACTIONS:
            delta[ticker] = delta.get(ticker, 0) - sh
    return delta


def _prior_snapshot_shares(prior_positions: dict[str, dict]) -> dict[str, int]:
    """Broker-verified held shares per ticker from the prior reconciled
    `positions_snapshot` — the trusted anchor. Drops non-positive holdings."""
    anchor: dict[str, int] = {}
    for ticker, pos in (prior_positions or {}).items():
        shares = (pos or {}).get("shares")
        if shares is None:
            continue
        try:
            shares = int(shares)
        except (TypeError, ValueError):
            continue
        if shares > 0:
            anchor[ticker] = shares
    return anchor


def synthesize_positions(
    prior_positions: dict[str, dict],
    day_deltas: dict[str, int],
    conn: sqlite3.Connection | None = None,
    as_of_date: str | None = None,
) -> tuple[dict[str, int], bool]:
    """ANCHOR the day's held positions on the prior day's RECONCILED snapshot
    (broker-book-aligned) + apply ONLY that day's fills (config#1281).

    This replaces the free-running full-ledger :func:`replay_positions`: each
    day is computed from a TRUSTED anchor, so replay error cannot compound. For
    a no-trade halt day this is exact — the result is the prior snapshot's
    positions verbatim (re-marked at the day's closes by the caller).

    Missing-anchor fallback: when no prior reconciled snapshot exists (empty
    ``prior_positions``) AND a connection/date is supplied, fall back to the
    legacy full-ledger replay through ``as_of_date`` so the tool still produces
    *a* result on a cold-start day; the second return value is ``True`` to flag
    that the result is the un-anchored (drift-prone) fallback. Otherwise the
    second return value is ``False`` (anchored, trusted).

    Returns ``(shares_by_ticker, used_fallback)`` with only positive holdings.
    """
    anchor = _prior_snapshot_shares(prior_positions)
    if not anchor:
        if conn is not None and as_of_date is not None:
            logger.warning(
                "No prior RECONCILED positions_snapshot to anchor on for %s — "
                "falling back to the drift-prone full-ledger replay. Synthesis "
                "is FLAGGED un-anchored; verify against the broker book.",
                as_of_date,
            )
            return replay_positions(conn, as_of_date), True
        # No anchor and no ledger to fall back on: empty book, anchored-trivially.
        return {}, False

    held = dict(anchor)
    for ticker, delta in day_deltas.items():
        held[ticker] = held.get(ticker, 0) + delta
    return {t: q for t, q in held.items() if q > 0}, False


def check_position_divergence(
    synthesized: dict[str, int], prior_positions: dict[str, dict]
) -> tuple[bool, int, int]:
    """Reconciliation guard (config#1281 / config#1276): does the synthesized
    position count diverge MATERIALLY from the prior reconciled snapshot's?

    A no-trade day must reproduce the prior count; a normal trading day moves it
    by a name or two. A large jump (the 7→20 failure) means the synthesis is
    broken — the caller refuses to write and flags the gap instead.

    Returns ``(diverged, n_synth, n_prior)``."""
    n_synth = len(synthesized)
    n_prior = len(_prior_snapshot_shares(prior_positions))
    if n_prior == 0:
        return False, n_synth, n_prior  # nothing to compare against (cold start)
    drift = abs(n_synth - n_prior)
    diverged = drift > _DIVERGENCE_ABS_FLOOR and drift > _DIVERGENCE_FRACTION * n_prior
    return diverged, n_synth, n_prior


def day_cash_flow(conn: sqlite3.Connection, as_of_date: str) -> float:
    """Net cash delta from fills ON ``as_of_date``: SELL fills add cash, BUY
    fills remove it (``filled_shares`` × ``fill_price``).

    Commissions are not modelled (no column; paper-account commissions are
    trivial) — any residual surfaces in the reconcile's `unattributed` bucket.
    Returns 0.0 when no trades executed that day (the common halt case)."""
    rows = conn.execute(
        "SELECT action, COALESCE(filled_shares, shares) AS sh, fill_price "
        "FROM trades WHERE date = ?",
        (as_of_date,),
    ).fetchall()
    cash = 0.0
    for action, sh, fill_price in rows:
        if sh is None or fill_price is None:
            continue
        notional = int(sh) * float(fill_price)
        if action in _SELL_ACTIONS:
            cash += notional
        elif action in _BUY_ACTIONS:
            cash -= notional
    return cash


def _prior_eod_row(conn: sqlite3.Connection, run_date: str) -> dict | None:
    """The most recent eod_pnl row strictly before ``run_date`` — the cash
    baseline to roll forward + the avg_cost source for held tickers."""
    row = conn.execute(
        "SELECT date, total_cash, accrued_interest, positions_snapshot "
        "FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    if not row:
        return None
    snap = {}
    if row[3]:
        try:
            snap = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            snap = {}
    return {
        "date": row[0],
        "total_cash": row[1],
        "accrued_interest": row[2],
        "positions_snapshot": snap,
    }


def synthesize_snapshot(
    run_date: str,
    shares_by_ticker: dict[str, int],
    closes_by_ticker: dict[str, float],
    cash: float,
    accrued_interest: float | None,
    prior_positions: dict[str, dict],
    schema_version: int | str,
) -> dict:
    """Build a snapshot dict byte-compatible with snapshot_capturer's payload.

    NAV is set consistent with the SAME ArcticDB closes the reconcile will
    re-read, so the no-flow case reconciles to ~zero unattributed. avg_cost is
    carried from the prior snapshot when the ticker was held, else seeded to
    the close (held-through positions don't use avg_cost as their baseline —
    only positions opened since the prior trading day do)."""
    positions: dict[str, dict] = {}
    positions_mv = 0.0
    for ticker, shares in shares_by_ticker.items():
        close = closes_by_ticker[ticker]
        mv = shares * close
        positions_mv += mv
        prior = prior_positions.get(ticker) or {}
        positions[ticker] = {
            "shares": int(shares),
            "market_value": mv,  # reconcile overwrites from ArcticDB anyway
            "avg_cost": float(prior.get("avg_cost", close)),
            "sector": prior.get("sector", "Unknown"),
        }
    nav = cash + positions_mv
    return {
        "run_date": run_date,
        "captured_at": datetime.now(UTC).isoformat(),
        "schema_version": schema_version,
        "synthesized": True,  # audit marker — this is a ledger-synthesized row
        "account": {
            "net_liquidation": nav,
            "total_cash": cash,
            "accrued_interest": accrued_interest,
        },
        "positions": positions,
        "accrued_dividends": {},
    }


def _read_closes_for_date(trades_bucket: str, tickers: list[str], run_date: str) -> dict[str, float]:
    """ArcticDB close per ticker on ``run_date`` (universe lib; macro lib for
    macro-routed symbols). Hard-fails if any requested ticker has no row for
    the date — that means the market-data gap for ``run_date`` is not healed,
    so synthesis would be wrong; heal it first (config#1228). Mirrors the same
    authoritative lookup eod_reconcile uses."""
    import pandas as pd

    from executor.price_cache import (
        _MACRO_SYMBOLS,
        _open_macro_library,
        _open_universe_library,
    )

    universe_lib = _open_universe_library(trades_bucket)
    macro_lib = None
    target_ts = pd.Timestamp(run_date).normalize()
    closes: dict[str, float] = {}
    missing: list[str] = []
    for ticker in tickers:
        if ticker in _MACRO_SYMBOLS:
            if macro_lib is None:
                macro_lib = _open_macro_library(trades_bucket)
            lib = macro_lib
        else:
            lib = universe_lib
        try:
            df = lib.read(ticker).data
        except Exception as e:
            missing.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty or "Close" not in df.columns:
            missing.append(f"{ticker} (no Close)")
            continue
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        match = df[idx == target_ts]
        if match.empty:
            missing.append(f"{ticker} (no row for {run_date})")
            continue
        closes[ticker] = float(match["Close"].iloc[-1])
    if missing:
        raise RuntimeError(
            f"ArcticDB close lookup failed for {len(missing)} held ticker(s) on "
            f"{run_date}: {missing}. The market-data gap for {run_date} is not "
            f"healed — run the universe-gap auto-heal first (config#1228); "
            f"synthesizing against an incomplete series would be wrong."
        )
    return closes


def backfill(run_date: str, *, dry_run: bool = False, force: bool = False) -> dict:
    """Reconstruct and write the eod_pnl row for ``run_date`` by synthesis."""
    config = load_config()
    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]
    region = config.get("aws_region", "us-east-1")

    conn = init_db(db_path)

    # Guard: don't clobber an existing row / real snapshot unless forced.
    existing = conn.execute(
        "SELECT 1 FROM eod_pnl WHERE date = ?", (run_date,)
    ).fetchone()
    if existing and not force:
        raise RuntimeError(
            f"eod_pnl row for {run_date} already exists. Re-run with --force to "
            f"overwrite (INSERT OR REPLACE is idempotent), or pick the right date."
        )

    prior = _prior_eod_row(conn, run_date)
    if prior is None or prior.get("total_cash") is None:
        raise RuntimeError(
            f"No prior eod_pnl row with total_cash before {run_date} — cannot roll "
            f"cash forward. Synthesis requires a cash baseline from the prior row."
        )
    cash_prior = float(prior["total_cash"])

    # config#1281: ANCHOR the day's positions on the prior RECONCILED snapshot
    # (broker-book-aligned) + apply ONLY this day's fills, instead of a
    # free-running full-ledger replay that compounds drift away from the broker
    # book. Missing-anchor cold-start days fall back to the legacy replay and
    # are flagged un-anchored.
    prior_positions = prior.get("positions_snapshot", {})
    day_deltas = day_share_deltas(conn, run_date)
    shares_by_ticker, used_fallback = synthesize_positions(
        prior_positions, day_deltas, conn=conn, as_of_date=run_date
    )
    if not shares_by_ticker:
        logger.warning("Position synthesis yields no open positions at %s.", run_date)

    # Reconciliation guard (config#1276 7→20 failure): refuse to write a row
    # whose synthesized position count diverges materially from the prior
    # reconciled snapshot — flag the gap instead of persisting a wrong NAV.
    diverged, n_synth, n_prior = check_position_divergence(shares_by_ticker, prior_positions)
    if diverged and not force:
        raise RuntimeError(
            f"Synthesized position count for {run_date} diverges materially from "
            f"the prior reconciled snapshot ({n_synth} vs {n_prior}). Refusing to "
            f"write a likely-wrong NAV — the gap is FLAGGED for manual review "
            f"(config#1281/#1276). Investigate the ledger/snapshot, or re-run "
            f"with --force only if you have verified the count against the broker "
            f"book."
        )

    closes = _read_closes_for_date(trades_bucket, list(shares_by_ticker), run_date)
    cash_today = cash_prior + day_cash_flow(conn, run_date)

    from executor.snapshot_capturer import SCHEMA_VERSION, _snapshot_key, load_snapshot

    existing_snap = load_snapshot(bucket=trades_bucket, run_date=run_date, region=region)
    if existing_snap is not None and not existing_snap.get("synthesized") and not force:
        raise RuntimeError(
            f"A real (non-synthesized) snapshot already exists for {run_date}. "
            f"Run `eod_reconcile.py --date {run_date}` directly, or use --force."
        )

    snapshot = synthesize_snapshot(
        run_date=run_date,
        shares_by_ticker=shares_by_ticker,
        closes_by_ticker=closes,
        cash=cash_today,
        accrued_interest=prior.get("accrued_interest"),
        prior_positions=prior.get("positions_snapshot", {}),
        schema_version=SCHEMA_VERSION,
    )

    summary = {
        "run_date": run_date,
        "prior_eod_date": prior["date"],
        "n_positions": len(shares_by_ticker),
        "cash_prior": round(cash_prior, 2),
        "cash_today": round(cash_today, 2),
        "synthesized_nav": round(snapshot["account"]["net_liquidation"], 2),
        "anchored_on_prior_snapshot": not used_fallback,
        "n_prior_snapshot_positions": n_prior,
        "dry_run": dry_run,
    }
    if used_fallback:
        summary["warning"] = (
            "no prior reconciled snapshot — used drift-prone full-ledger "
            "fallback; verify against broker book"
        )
    logger.info("Synthesized snapshot for %s: %s", run_date, summary)

    if dry_run:
        summary["snapshot_preview"] = snapshot
        return summary

    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=trades_bucket,
        Key=_snapshot_key(run_date),
        Body=json.dumps(snapshot, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Wrote synthesized snapshot s3://%s/%s", trades_bucket, _snapshot_key(run_date))

    # Run the canonical reconcile against the synthesized snapshot — it
    # re-prices from ArcticDB, recomputes the NAV reconciliation, and writes
    # the eod_pnl row + CSVs exactly as a normal EOD would.
    from executor.eod_reconcile import run as eod_run
    # A backfill is a post-hoc recovery of a PAST day: never resend that day's
    # EOD email, and never trigger the trailing reconcile_audit self-heal from
    # inside a correction (the audit pass is what may have called us — config#1276).
    eod_run(run_date, send_email=False, run_audit=False)
    summary["reconcile"] = "ran"
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Synthesize a missing eod_pnl row (config#1229).")
    parser.add_argument("--date", required=True, help="Trading day to backfill (YYYY-MM-DD).")
    parser.add_argument("--dry-run", action="store_true", help="Compute + print; write nothing.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing row/snapshot.")
    args = parser.parse_args()
    result = backfill(args.date, dry_run=args.dry_run, force=args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
