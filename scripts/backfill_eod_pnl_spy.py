"""
scripts/backfill_eod_pnl_spy.py — Rewrite eod_pnl.spy_close / spy_return_pct /
daily_alpha_pct from the authoritative ArcticDB macro.SPY series.

Context (ROADMAP P0 #2, 2026-04-20): `trades.db::eod_pnl` is the source of
truth for `eod_reconcile`'s prior-session SPY lookup. Before 2026-04-20,
the executor pulled SPY intraday from IB's 15-min delayed feed (or from the
morning-written daily_closes parquet), meaning every historical row's
`spy_close` is shifted off the true post-close value by ~0.1-0.4%. The S3
`eod_pnl.csv` is re-exported from this table every EOD run — fixing the
CSV in isolation would be overwritten next cycle.

This one-shot:
  1. Dry-run (default): prints a diff table of proposed changes, no writes.
  2. --apply:
     a. Snapshots trades.db to s3://{trades_bucket}/trades/trades_{today}.pre-backfill.db
     b. Rewrites eod_pnl rows in a single transaction
     c. Re-exports eod_pnl.csv to s3://{trades_bucket}/trades/eod_pnl.csv
     d. Keeps the local .db on whatever machine ran it — point at ae-trading's
        /home/ec2-user/alpha-engine/trades.db when running there.

ArcticDB macro.SPY is a Close-only single-symbol frame written by
alpha-engine-data's daily_append + backfill pipelines. Index = trading day.
The prior trading day for run_date is simply the largest index entry < run_date.

Usage:
    # Dry-run against a local copy of trades.db
    python scripts/backfill_eod_pnl_spy.py --db /tmp/trades_latest.db

    # Live on ae-trading (via SSM)
    cd /home/ec2-user/alpha-engine && source .venv/bin/activate && \\
        python scripts/backfill_eod_pnl_spy.py --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

# arcticdb must import before pandas on macOS (see price_cache.py comment).
import arcticdb as _arcticdb  # noqa: F401
import boto3
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from executor.price_cache import _open_macro_library  # noqa: E402 -- must follow sys.path.insert above

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_eod_pnl_spy")


def _load_spy_series(signals_bucket: str) -> pd.Series:
    """Pull ArcticDB macro.SPY's Close column as a date-indexed Series."""
    macro = _open_macro_library(signals_bucket)
    df = macro.read("SPY").data
    if "Close" not in df.columns:
        raise RuntimeError(f"macro.SPY frame missing Close column (has {list(df.columns)})")
    spy = df["Close"].dropna()
    spy.index = pd.to_datetime(spy.index).normalize()
    spy = spy.sort_index()
    logger.info(
        "Loaded ArcticDB macro.SPY: %d rows, %s → %s",
        len(spy), spy.index[0].date(), spy.index[-1].date(),
    )
    return spy


def _prior_trading_day_close(spy: pd.Series, run_date: date) -> tuple[date, float] | None:
    """Return (date, close) of the largest SPY index entry strictly before run_date."""
    ts = pd.Timestamp(run_date)
    earlier = spy.index[spy.index < ts]
    if len(earlier) == 0:
        return None
    prior = earlier[-1]
    return prior.date(), float(spy.loc[prior])


def _same_day_close(spy: pd.Series, run_date: date) -> float | None:
    ts = pd.Timestamp(run_date)
    if ts in spy.index:
        return float(spy.loc[ts])
    return None


def compute_corrections(db_path: str, spy: pd.Series) -> list[dict]:
    """Produce a list of per-row planned updates.

    Each entry:
      {
        "date": "YYYY-MM-DD",
        "old_spy_close": float | None,
        "new_spy_close": float,
        "old_spy_return_pct": float | None,
        "new_spy_return_pct": float,
        "daily_return_pct": float | None,
        "old_daily_alpha_pct": float | None,
        "new_daily_alpha_pct": float | None,
        "prior_date": "YYYY-MM-DD",
      }

    Rows whose new values match old values within 1e-9 are still emitted
    (with changed=False) so the dry-run shows coverage.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT date, spy_close, spy_return_pct, daily_return_pct, daily_alpha_pct "
            "FROM eod_pnl ORDER BY date ASC"
        ).fetchall()

    plan: list[dict] = []
    for row in rows:
        run_date = datetime.strptime(row["date"], "%Y-%m-%d").date()

        new_close = _same_day_close(spy, run_date)
        if new_close is None:
            logger.warning(
                "ArcticDB macro.SPY has no row for %s — skipping (non-trading day?)",
                run_date,
            )
            continue

        prior = _prior_trading_day_close(spy, run_date)
        if prior is None:
            # First seed row in the universe; can't compute return. Leave alpha None.
            plan.append({
                "date": row["date"],
                "old_spy_close": row["spy_close"],
                "new_spy_close": new_close,
                "old_spy_return_pct": row["spy_return_pct"],
                "new_spy_return_pct": None,
                "daily_return_pct": row["daily_return_pct"],
                "old_daily_alpha_pct": row["daily_alpha_pct"],
                "new_daily_alpha_pct": None,
                "prior_date": None,
            })
            continue

        prior_date, prior_close = prior
        new_spy_return = (new_close / prior_close - 1.0) * 100.0
        daily_ret = row["daily_return_pct"]
        new_alpha = (daily_ret - new_spy_return) if daily_ret is not None else None

        plan.append({
            "date": row["date"],
            "old_spy_close": row["spy_close"],
            "new_spy_close": new_close,
            "old_spy_return_pct": row["spy_return_pct"],
            "new_spy_return_pct": new_spy_return,
            "daily_return_pct": daily_ret,
            "old_daily_alpha_pct": row["daily_alpha_pct"],
            "new_daily_alpha_pct": new_alpha,
            "prior_date": prior_date.isoformat(),
        })

    return plan


def print_plan(plan: list[dict]) -> None:
    """Human-readable dry-run preview."""
    print("")
    print(f"{'date':12} {'prior':12} {'old_close':>10} {'new_close':>10} "
          f"{'old_spy%':>9} {'new_spy%':>9} {'old_α%':>9} {'new_α%':>9}  flag")
    print("-" * 105)
    changed = 0
    for row in plan:
        old_close = row["old_spy_close"]
        new_close = row["new_spy_close"]
        old_spy = row["old_spy_return_pct"]
        new_spy = row["new_spy_return_pct"]
        old_a = row["old_daily_alpha_pct"]
        new_a = row["new_daily_alpha_pct"]

        close_diff = (
            abs((old_close or 0) - new_close) > 1e-6
            if old_close is not None else True
        )
        spy_diff = (
            new_spy is not None and old_spy is not None
            and abs(old_spy - new_spy) > 1e-9
        ) or (new_spy is None) != (old_spy is None)
        alpha_diff = (
            new_a is not None and old_a is not None
            and abs(old_a - new_a) > 1e-9
        ) or (new_a is None) != (old_a is None)
        row_changed = close_diff or spy_diff or alpha_diff
        if row_changed:
            changed += 1

        flag = "CHANGE" if row_changed else "ok"
        print(
            f"{row['date']:12} "
            f"{(row['prior_date'] or '—'):12} "
            f"{(f'{old_close:.2f}' if old_close is not None else '—'):>10} "
            f"{new_close:>10.2f} "
            f"{(f'{old_spy:.4f}' if old_spy is not None else '—'):>9} "
            f"{(f'{new_spy:.4f}' if new_spy is not None else '—'):>9} "
            f"{(f'{old_a:.4f}' if old_a is not None else '—'):>9} "
            f"{(f'{new_a:.4f}' if new_a is not None else '—'):>9}  "
            f"{flag}"
        )
    print("-" * 105)
    print(f"Total: {len(plan)} rows, {changed} would change, {len(plan) - changed} unchanged")
    print("")


def snapshot_db_to_s3(db_path: str, trades_bucket: str) -> str:
    """Upload a pre-backfill copy of trades.db to S3. Returns the S3 key."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    key = f"trades/trades_{stamp}.pre-backfill.db"
    s3 = boto3.client("s3")
    with open(db_path, "rb") as f:
        s3.put_object(Bucket=trades_bucket, Key=key, Body=f.read())
    logger.info("Pre-backfill snapshot uploaded: s3://%s/%s", trades_bucket, key)
    return key


def apply_plan(db_path: str, plan: list[dict]) -> int:
    """Apply the plan inside a single transaction. Returns rows updated."""
    updated = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            for row in plan:
                cursor.execute(
                    "UPDATE eod_pnl "
                    "SET spy_close = ?, spy_return_pct = ?, daily_alpha_pct = ? "
                    "WHERE date = ?",
                    (
                        row["new_spy_close"],
                        row["new_spy_return_pct"],
                        row["new_daily_alpha_pct"],
                        row["date"],
                    ),
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
    """Re-export the corrected eod_pnl table to s3://{trades_bucket}/trades/eod_pnl.csv.

    Mirrors `eod_reconcile.py`'s post-reconcile export logic so the dashboard
    sees the fix immediately instead of waiting until the next EOD timer.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn)
    key = "trades/eod_pnl.csv"
    s3 = boto3.client("s3")
    buf = df.to_csv(index=False).encode()
    s3.put_object(Bucket=trades_bucket, Key=key, Body=buf)
    logger.info("Exported %d rows to s3://%s/%s", len(df), trades_bucket, key)


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
    parser.add_argument(
        "--db",
        default=None,
        help="Path to trades.db (defaults to config/risk.yaml::db_path).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to risk.yaml (defaults to <repo>/config/risk.yaml).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write. Default is dry-run.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip S3 pre-backfill snapshot (NOT RECOMMENDED — reserved for dev runs against a local copy).",
    )
    parser.add_argument(
        "--skip-csv-export",
        action="store_true",
        help="Skip the eod_pnl.csv re-export to S3.",
    )
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

    spy = _load_spy_series(signals_bucket)
    plan = compute_corrections(db_path, spy)
    print_plan(plan)

    if not args.apply:
        logger.info("Dry-run complete. Re-run with --apply to write changes.")
        return 0

    if not args.skip_snapshot:
        snapshot_db_to_s3(db_path, trades_bucket)
    else:
        logger.warning("Skipping pre-backfill snapshot — if this is not a dev run, abort now.")

    updated = apply_plan(db_path, plan)
    if updated != len(plan):
        logger.warning(
            "Plan had %d rows but SQL UPDATE touched %d. Check for missing dates.",
            len(plan), updated,
        )

    if not args.skip_csv_export:
        export_eod_csv(db_path, trades_bucket)

    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
