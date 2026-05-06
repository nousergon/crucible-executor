"""
backfill_trades_sector.py — one-shot backfill for trades.sector rows
that landed with sector="Unknown" before the research-side preflight
(alpha-engine-research#126) and the executor-side patch
(alpha-engine#141) shipped.

Surface for the 2026-05-04 EOG/NVT incident: research wrote the first
pass of signals.json with sector="Unknown" because the constituents
sector_map hadn't loaded yet, then re-ran 10 minutes later with
correct values. The morning planner had already consumed v1 and the
daemon's intraday fills wrote "Unknown" into trades.db. The bad rows
survive because no UPDATE path overwrites trades.sector — eod_reconcile
only enriches the in-memory positions snapshot.

Idempotent: running this script multiple times produces the same
result. A trade row is updated only if its current sector is NULL,
empty, or "Unknown" — real GICS values are never overwritten.

Usage:
    # Dry run — print what would be updated, no writes.
    python scripts/backfill_trades_sector.py --db-path /path/to/trades.db --dry-run

    # Live backfill on ae-trading. Re-uploads the corrected
    # trades.db to S3 if --backup-to-s3 is set.
    python scripts/backfill_trades_sector.py --db-path /path/to/trades.db

After running, the next EOD reconcile will export the corrected
trades_full.csv to S3 automatically. Run --backup-to-s3 only if you
need the corrected DB mirrored before the next EOD pass.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys

# Resolve `executor.*` imports when invoked as `python scripts/...` from the
# repo root — same pattern as executor/main.py:34. Lets the script be run
# directly without setting PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _ensure_sector_column_exists(conn: sqlite3.Connection) -> None:
    """The trade_logger init_db migration adds sector. Hard-fail with a
    clear pointer if the migration hasn't run on this DB yet."""
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    if "sector" not in cols:
        sys.stderr.write(
            "ERROR: trades table is missing sector column. "
            "Run trade_logger.init_db() against this DB first to apply "
            "the schema migration, then re-run the backfill.\n"
        )
        sys.exit(2)


def _load_constituents_sector_map(bucket: str) -> dict[str, str]:
    """Reuse the executor's existing loader so behavior matches eod_reconcile's
    chain. Falls back to an empty dict if the constituents file isn't
    available — caller logs the empty case."""
    from executor.eod_reconcile import _load_constituents_sector_map as _load
    return _load(bucket)


def backfill(
    db_path: str,
    bucket: str,
    dry_run: bool,
) -> dict:
    """Run the backfill. Returns a stats dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_sector_column_exists(conn)

        rows = conn.execute(
            "SELECT trade_id, ticker, date, action, sector "
            "FROM trades "
            "WHERE sector IS NULL OR sector = '' OR sector = 'Unknown' "
            "ORDER BY date, ticker"
        ).fetchall()

        if not rows:
            logger.info("No trades.sector rows need backfill — nothing to do.")
            return {"candidates": 0, "patched": 0, "unresolved": 0}

        logger.info("Found %d candidate row(s) with sector NULL/empty/Unknown", len(rows))

        sector_map = _load_constituents_sector_map(bucket)
        if not sector_map:
            logger.error(
                "constituents sector_map unavailable from s3://%s — cannot "
                "resolve any rows. Aborting.", bucket,
            )
            return {"candidates": len(rows), "patched": 0, "unresolved": len(rows)}

        logger.info("Loaded sector_map with %d tickers", len(sector_map))

        patched = 0
        unresolved: list[str] = []
        for r in rows:
            ticker = r["ticker"]
            mapped = sector_map.get(ticker)
            if not mapped:
                unresolved.append(f"{r['date']} {ticker} ({r['action']})")
                continue
            logger.info(
                "[%s] %s %s %s: %r → %r",
                "DRY-RUN" if dry_run else "UPDATE",
                r["date"], ticker, r["action"], r["sector"], mapped,
            )
            if not dry_run:
                conn.execute(
                    "UPDATE trades SET sector = ? WHERE trade_id = ?",
                    (mapped, r["trade_id"]),
                )
            patched += 1

        if not dry_run:
            conn.commit()

        if unresolved:
            logger.warning(
                "%d row(s) unresolved (ticker not in constituents sector_map): %s",
                len(unresolved), unresolved,
            )

        logger.info(
            "Backfill summary: candidates=%d patched=%d unresolved=%d (dry_run=%s)",
            len(rows), patched, len(unresolved), dry_run,
        )
        return {
            "candidates": len(rows),
            "patched": patched,
            "unresolved": len(unresolved),
        }
    finally:
        conn.close()


def _backup_db_to_s3(db_path: str, bucket: str) -> None:
    """Mirror the corrected trades.db to S3. Optional — the next EOD
    reconcile will re-export trades_full.csv from the local DB anyway,
    which is the consumer-facing artifact."""
    import boto3
    from datetime import datetime, timezone
    s3 = boto3.client("s3")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"trades/backups/trades.db.{ts}.post-sector-backfill"
    with open(db_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read())
    logger.info("Backed up trades.db to s3://%s/%s", bucket, key)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db-path", required=True, help="Path to trades.db")
    p.add_argument(
        "--bucket", default="alpha-engine-research",
        help="S3 bucket holding constituents.json (default: alpha-engine-research)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print planned UPDATE statements without executing them.",
    )
    p.add_argument(
        "--backup-to-s3", action="store_true",
        help="After backfill, mirror the corrected trades.db to "
             "s3://{bucket}/trades/backups/. Optional — the next EOD "
             "reconcile re-exports trades_full.csv automatically.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = _parse_args()

    stats = backfill(args.db_path, args.bucket, args.dry_run)

    if args.backup_to_s3 and not args.dry_run and stats["patched"] > 0:
        _backup_db_to_s3(args.db_path, args.bucket)

    return 0 if stats["unresolved"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
