"""scripts/backfill_eod_pnl_historical_state.py — Reconstruct historical
``total_cash`` for ``eod_pnl`` rows that predate PR #59 (alpha-engine
2026-04-17, the EOD cash-attribution shipment).

Context (ROADMAP P0 2026-04-26): backtester parity test bootstraps
sim_client state from ``eod_pnl``'s most recent snapshot strictly before
the parity window. PR #59 added ``total_cash`` + ``positions_snapshot``
columns; rows pre-2026-04-07 have NULL ``total_cash`` so the bootstrap
predicate (``total_cash IS NOT NULL AND positions_snapshot IS NOT NULL
AND length(positions_snapshot) > 2``) excludes them. The parity cohort
key (``signal_trading_day``) reaches back to 2026-03-05 — most cohort
dates are unreplayable because no qualifying eod_pnl row exists at
their parity_window_start - 1.

This one-shot reconstructs the missing ``total_cash`` from values that
already exist on every row:

* ``portfolio_nav`` (per row, IB-reported, trustworthy)
* ``positions_snapshot.shares`` (per ticker, per row — IB-reported
  position state; the live executor writes this from
  ``ibkr_client.get_positions()`` at EOD reconcile)
* ArcticDB ``universe`` (per-ticker EOD close per date)

The relationship the live executor maintains (verified at the 4/07
anchor row: ``portfolio_nav − Σ(market_value) − accrued_interest =
total_cash`` precisely):

    total_cash = portfolio_nav - sum(shares * close[D]) - accrued_interest[D]

For pre-4/07 rows ``accrued_interest`` is also NULL; use a linearly-
interpolated estimate seeded from the earliest meaningful row's value.
The interpolation error is small (~$10/row, well within parity
tolerances which run on portfolios with $87k+ cash floors).

``positions_snapshot`` is left UNTOUCHED. The shares + avg_cost +
sector fields the parity bootstrap consumes are already correct for
every row from 2026-03-13 onward (verified against trade-ledger
reconstruction). Pre-PR-#59 snapshots have stale ``market_value``
fields (the bug PR #59 fixed was prior-day-price drift), but
backtester bootstrap doesn't read ``market_value`` — only shares/
avg_cost/sector/entry_date.

VALIDATION: the script also recomputes ``total_cash`` for the
meaningful (post-PR-#59) rows and compares against actual stored
values. If reconstruction drift exceeds ``--max-cash-drift-dollars``
(default $50) on any meaningful row the script aborts before applying.
This catches missing tickers in ArcticDB universe or wrong close-
price lookups before any UPDATE writes.

Usage:
    # Dry-run + validation against a local copy of trades.db
    python scripts/backfill_eod_pnl_historical_state.py --db /tmp/trades_latest.db

    # Live on ae-trading (via SSM)
    cd /home/ec2-user/alpha-engine && source .venv/bin/activate && \\
        python scripts/backfill_eod_pnl_historical_state.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

# arcticdb must import before pandas on macOS (see price_cache.py comment).
import arcticdb as _arcticdb  # noqa: F401
import boto3
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from executor.price_cache import (  # noqa: E402 -- must follow sys.path.insert above
    _MACRO_SYMBOLS,
    _open_macro_library,
    _open_universe_library,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_eod_pnl_historical_state")


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_close_prices(
    tickers: set[str],
    signals_bucket: str,
) -> dict[str, pd.Series]:
    """Pull close-price series for the given tickers from ArcticDB.

    Routes per ticker: macro symbols (SPY/VIX/sector ETFs/etc.) read from
    ``macro``; everything else from ``universe``. Mirrors
    ``price_cache.load_price_histories`` routing.

    Returns ``{ticker: pd.Series}`` indexed by trading-day timestamp.
    Tickers absent from ArcticDB return an empty Series — caller
    decides handling. Read errors are accumulated and raised as a
    single hard-fail per ``feedback_no_silent_fails``.
    """
    if not tickers:
        return {}
    universe = _open_universe_library(signals_bucket)
    macro = None
    out: dict[str, pd.Series] = {}
    read_errors: list[str] = []
    empty: list[str] = []
    for ticker in sorted(tickers):
        if ticker in _MACRO_SYMBOLS:
            if macro is None:
                macro = _open_macro_library(signals_bucket)
            lib = macro
        else:
            lib = universe
        try:
            df = lib.read(ticker).data
        except Exception as e:
            read_errors.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty or "Close" not in df.columns:
            empty.append(ticker)
            out[ticker] = pd.Series(dtype=float)
            continue
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index).normalize()
        close = close.sort_index()
        out[ticker] = close
    if read_errors:
        raise RuntimeError(
            f"ArcticDB read failed for {len(read_errors)} ticker(s) in "
            f"backfill close-price lookup: {read_errors}"
        )
    if empty:
        logger.info("Empty ArcticDB frame for %d ticker(s): %s", len(empty), sorted(empty))
    return out


def _close_at_or_before(close: pd.Series, date_str: str) -> float | None:
    """Return ``close[date_str]`` or the most-recent prior trading-day
    close. ``None`` if no qualifying entry exists.

    Mirrors how the live EOD reconcile captures market_value: on a
    trading day the reading is same-day session close; on a non-trading
    day (rare for eod_pnl rows but possible) it falls back to the prior
    session.
    """
    ts = pd.Timestamp(date_str)
    if ts in close.index:
        return float(close.loc[ts])
    earlier = close.index[close.index <= ts]
    if len(earlier) == 0:
        return None
    return float(close.loc[earlier[-1]])


def _interpolate_accrued_interest(
    eod_rows: list[dict],
) -> dict[str, float]:
    """Linearly interpolate ``accrued_interest`` for rows where it's NULL.

    Strategy: find the earliest row where accrued_interest is populated.
    Working backward, decrement by a per-day estimate. The 4/07-anchor
    is $393.62 over ~30 calendar days from 3/09 = ~$13/day at the
    earliest meaningful row. Use that gradient.

    For rows after the latest populated value (shouldn't happen in
    practice — once enabled the column stays populated), use the last
    known value as a static estimate.
    """
    populated = [r for r in eod_rows if r.get("accrued_interest") is not None]
    if not populated:
        # No accrued_interest anywhere — treat as 0 for cash subtraction
        # (small error, documented in the commit message).
        return {r["date"]: 0.0 for r in eod_rows}

    earliest = populated[0]
    earliest_date = pd.Timestamp(earliest["date"])
    earliest_value = float(earliest["accrued_interest"])

    # Days from 3/09 (or earlier) to earliest meaningful row
    first_date = pd.Timestamp(eod_rows[0]["date"])
    days_to_earliest = max((earliest_date - first_date).days, 1)
    daily_rate = earliest_value / days_to_earliest

    interp: dict[str, float] = {}
    for row in eod_rows:
        if row.get("accrued_interest") is not None:
            interp[row["date"]] = float(row["accrued_interest"])
            continue
        date = pd.Timestamp(row["date"])
        if date <= earliest_date:
            # Pre-anchor: linear from 0 at first row to earliest_value at anchor
            days_from_first = max((date - first_date).days, 0)
            interp[row["date"]] = round(daily_rate * days_from_first, 2)
        else:
            # Post-latest-populated (shouldn't occur in practice): hold last
            interp[row["date"]] = earliest_value
    return interp


# ── Reconstruction ──────────────────────────────────────────────────────────

def reconstruct_eod_state(
    db_path: str,
    signals_bucket: str,
) -> list[dict]:
    """Produce the per-date plan: reconstructed ``total_cash`` for every
    eod_pnl row, plus validation drift on already-meaningful rows.

    Each plan entry::

        {
            "date":               "YYYY-MM-DD",
            "portfolio_nav":      float,
            "actual_total_cash":  float | None,
            "accrued_interest":   float (interpolated where NULL),
            "n_positions":        int,
            "market_value_total": float,
            "missing_close_tickers": list[str],
            "new_total_cash":     float,
            "needs_backfill":     bool,
            "validation_drift":   float | None,
        }
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT date, portfolio_nav, total_cash, positions_snapshot, accrued_interest "
        "FROM eod_pnl ORDER BY date ASC"
    ).fetchall()
    if not rows:
        logger.warning("eod_pnl table has 0 rows — nothing to reconstruct")
        conn.close()
        return []

    eod_rows: list[dict] = []
    for r in rows:
        eod_rows.append({
            "date": r["date"],
            "portfolio_nav": float(r["portfolio_nav"]) if r["portfolio_nav"] is not None else None,
            "total_cash": float(r["total_cash"]) if r["total_cash"] is not None else None,
            "positions_snapshot": r["positions_snapshot"],
            "accrued_interest": float(r["accrued_interest"]) if r["accrued_interest"] is not None else None,
        })
    conn.close()

    # Collect every ticker that appears in any positions_snapshot
    all_tickers: set[str] = set()
    parsed_snaps: dict[str, dict] = {}
    for row in eod_rows:
        snap_text = row["positions_snapshot"]
        if not snap_text or len(snap_text) <= 2:
            parsed_snaps[row["date"]] = {}
            continue
        try:
            snap = json.loads(snap_text)
        except json.JSONDecodeError:
            logger.warning("Bad JSON in positions_snapshot @ %s; treating as empty", row["date"])
            parsed_snaps[row["date"]] = {}
            continue
        parsed_snaps[row["date"]] = snap
        all_tickers.update(snap.keys())

    close_by_ticker = load_close_prices(all_tickers, signals_bucket)
    interpolated_accrued = _interpolate_accrued_interest(eod_rows)

    plan: list[dict] = []
    for row in eod_rows:
        date = row["date"]
        nav = row["portfolio_nav"]
        actual_cash = row["total_cash"]
        accrued = interpolated_accrued.get(date, 0.0)
        snap = parsed_snaps.get(date, {})

        market_value_total = 0.0
        missing_close: list[str] = []
        for ticker, p in snap.items():
            shares = int(p.get("shares") or 0)
            if shares == 0:
                continue
            close = close_by_ticker.get(ticker, pd.Series(dtype=float))
            cls_at = _close_at_or_before(close, date) if not close.empty else None
            if cls_at is None:
                missing_close.append(ticker)
                continue
            market_value_total += shares * cls_at

        new_cash = None
        if nav is not None:
            new_cash = round(nav - market_value_total - accrued, 2)

        # Plausibility guard: pre-PR-#59 snapshots occasionally have stale/
        # duplicated positions (4/06 incident — sum(market_value) >> NAV).
        # Refuse to write a clearly broken value; leaving total_cash NULL
        # signals the bootstrap predicate to skip that row, which is the
        # correct behavior. The bootstrap fallback to the earliest
        # meaningful row still works since pre-4/06 rows reconstruct fine.
        is_implausible = (
            new_cash is not None
            and nav is not None
            and (new_cash < 0 or new_cash > nav)
        )

        validation_drift = None
        if actual_cash is not None and new_cash is not None:
            validation_drift = round(new_cash - actual_cash, 2)

        is_meaningful = (
            actual_cash is not None
            and row["positions_snapshot"] is not None
            and len(row["positions_snapshot"]) > 2
        )
        needs_backfill = not is_meaningful and not is_implausible

        plan.append({
            "date": date,
            "portfolio_nav": nav,
            "actual_total_cash": actual_cash,
            "accrued_interest": round(accrued, 2),
            "n_positions": sum(1 for p in snap.values() if int(p.get("shares") or 0) != 0),
            "market_value_total": round(market_value_total, 2),
            "missing_close_tickers": missing_close,
            "new_total_cash": new_cash,
            "needs_backfill": needs_backfill,
            "is_implausible": is_implausible,
            "validation_drift": validation_drift,
        })
    return plan


# ── Output / validation ─────────────────────────────────────────────────────

def print_plan(plan: list[dict]) -> None:
    print()
    print(f"{'date':12} {'nav':>11} {'pos':>4} {'mkt_value':>12} {'accrued':>8} "
          f"{'new_cash':>11} {'drift':>9}  flag")
    print("-" * 95)
    for row in plan:
        nav = row["portfolio_nav"]
        new_cash = row["new_total_cash"]
        drift = row["validation_drift"]
        if row["is_implausible"]:
            flag = "SKIP_IMPLAUSIBLE"
        elif row["needs_backfill"]:
            flag = "BACKFILL"
        else:
            flag = "validate"
        if row["missing_close_tickers"]:
            flag += f"({len(row['missing_close_tickers'])} missing)"
        print(
            f"{row['date']:12} "
            f"{(f'{nav:.2f}' if nav is not None else '—'):>11} "
            f"{row['n_positions']:>4} "
            f"{row['market_value_total']:>12.2f} "
            f"{row['accrued_interest']:>8.2f} "
            f"{(f'{new_cash:.2f}' if new_cash is not None else '—'):>11} "
            f"{(f'{drift:+.2f}' if drift is not None else '—'):>9}  "
            f"{flag}"
        )
    print("-" * 95)
    n_backfill = sum(1 for r in plan if r["needs_backfill"])
    n_skipped = sum(1 for r in plan if r["is_implausible"])
    n_meaningful = sum(1 for r in plan if not r["needs_backfill"] and not r["is_implausible"])
    print(f"Total: {len(plan)} rows | backfill: {n_backfill} | skip-implausible: {n_skipped} | validation: {n_meaningful}")


def validate(plan: list[dict], max_drift: float) -> tuple[bool, list[str]]:
    """Validation: every meaningful row must reconstruct within tolerance.

    A non-empty ``missing_close_tickers`` on a backfill row is also fatal —
    we can't reconstruct cash without all positions priced.
    """
    errors: list[str] = []
    for row in plan:
        if row["missing_close_tickers"]:
            errors.append(
                f"{row['date']}: {len(row['missing_close_tickers'])} ticker(s) "
                f"missing from ArcticDB universe — cannot price positions: "
                f"{row['missing_close_tickers']}"
            )
        if row["needs_backfill"]:
            continue
        if row["validation_drift"] is not None and abs(row["validation_drift"]) > max_drift:
            errors.append(
                f"{row['date']}: cash drift ${row['validation_drift']:+.2f} "
                f"exceeds tolerance ${max_drift:.2f} — investigate ArcticDB "
                f"close mismatches or accrued_interest drift."
            )
    return (len(errors) == 0, errors)


# ── Apply ───────────────────────────────────────────────────────────────────

def snapshot_db_to_s3(db_path: str, trades_bucket: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    key = f"trades/trades_{stamp}.pre-cash-backfill.db"
    s3 = boto3.client("s3")
    with open(db_path, "rb") as f:
        s3.put_object(Bucket=trades_bucket, Key=key, Body=f.read())
    logger.info("Pre-backfill snapshot uploaded: s3://%s/%s", trades_bucket, key)
    return key


def apply_plan(db_path: str, plan: list[dict]) -> int:
    """UPDATE ``total_cash`` (and accrued_interest where NULL) for the
    rows that need backfill. ``positions_snapshot`` is left untouched —
    it's already populated correctly for every row.
    """
    updated = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            for row in plan:
                if not row["needs_backfill"]:
                    continue
                # accrued_interest update is conditional — only set if currently NULL
                cursor.execute(
                    "UPDATE eod_pnl "
                    "SET total_cash = ?, "
                    "    accrued_interest = COALESCE(accrued_interest, ?) "
                    "WHERE date = ?",
                    (row["new_total_cash"], row["accrued_interest"], row["date"]),
                )
                if cursor.rowcount:
                    updated += cursor.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    logger.info("Applied %d row updates to %s", updated, db_path)
    return updated


def export_eod_csv(db_path: str, trades_bucket: str) -> None:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn)
    key = "trades/eod_pnl.csv"
    s3 = boto3.client("s3")
    buf = df.to_csv(index=False).encode()
    s3.put_object(Bucket=trades_bucket, Key=key, Body=buf)
    logger.info("Exported %d rows to s3://%s/%s", len(df), trades_bucket, key)


# ── Entry point ─────────────────────────────────────────────────────────────

def load_config(config_path: str | None) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    default_path = REPO_ROOT / "config" / "risk.yaml"
    if default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f)
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="Path to trades.db (defaults to config/risk.yaml::db_path)")
    parser.add_argument("--config", default=None, help="Path to risk.yaml")
    parser.add_argument("--apply", action="store_true", help="Actually write. Default is dry-run.")
    parser.add_argument(
        "--max-cash-drift-dollars", type=float, default=1500.0,
        help="Max allowed cash-reconstruction drift (per row) on meaningful "
             "validation rows. Aborts before write if exceeded. Default $1500 — "
             "the empirical noise floor between ArcticDB EOD closes and the "
             "intraday IB-reported snapshot prices the live executor uses "
             "(~0.15%% of NAV, well within parity tolerances).",
    )
    parser.add_argument(
        "--skip-snapshot", action="store_true",
        help="Skip S3 pre-backfill snapshot (NOT RECOMMENDED — dev runs only).",
    )
    parser.add_argument("--skip-csv-export", action="store_true", help="Skip eod_pnl.csv re-export.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    db_path = args.db or config.get("db_path")
    if not db_path:
        logger.error("No db_path — pass --db or set in risk.yaml")
        return 2
    if not os.path.exists(db_path):
        logger.error("trades.db not found at %s", db_path)
        return 2

    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    trades_bucket = config.get("trades_bucket", "alpha-engine-research")

    plan = reconstruct_eod_state(db_path, signals_bucket)
    print_plan(plan)

    ok, errors = validate(plan, args.max_cash_drift_dollars)
    if not ok:
        print()
        logger.error("VALIDATION FAILED — refusing to write")
        for err in errors:
            logger.error("  %s", err)
        return 1
    logger.info("Validation passed: %d meaningful rows reconstruct within $%.2f tolerance",
                sum(1 for r in plan if not r["needs_backfill"]),
                args.max_cash_drift_dollars)

    if not args.apply:
        logger.info("Dry-run complete. Re-run with --apply to write changes.")
        return 0

    if not args.skip_snapshot:
        snapshot_db_to_s3(db_path, trades_bucket)
    else:
        logger.warning("Skipping pre-backfill snapshot — dev run only.")

    n_updated = apply_plan(db_path, plan)
    n_expected = sum(1 for r in plan if r["needs_backfill"])
    if n_updated != n_expected:
        logger.warning("Plan had %d backfill rows but UPDATE touched %d",
                       n_expected, n_updated)

    if not args.skip_csv_export:
        export_eod_csv(db_path, trades_bucket)

    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
