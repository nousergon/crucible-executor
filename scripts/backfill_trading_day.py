"""
backfill_trading_day.py — one-shot backfill for the trading_day +
signal_trading_day columns added to trades.db on 2026-04-24.

Per alpha-engine-docs/private/DATE_CONVENTIONS.md:
  - trading_day = last completed NYSE trading session at the moment of the
    trade's fill_time (or created_at as fallback). Backward-looking.
  - signal_trading_day = the trading_day of the signals.json that
    originated this trade. For ENTER actions, back-resolved by walking
    signals/{date}/signals.json files in S3 looking for the ticker. For
    EXIT/REDUCE/manual-tool actions, NULL is acceptable.

Idempotent: running this script multiple times produces the same result.
A trade row is updated only if the target column is currently NULL — so
forward-write call sites (the new daemon entry path) keep authoritative
values and the backfill never overwrites them.

Usage:
    # Dry run — print what would be updated, no writes.
    python scripts/backfill_trading_day.py --db-path /path/to/trades.db --dry-run

    # Live backfill on ae-trading.
    python scripts/backfill_trading_day.py --db-path /path/to/trades.db

    # Limit signal-date back-resolution window (default 30 calendar days).
    python scripts/backfill_trading_day.py --db-path ... --signal-lookback-days 60

The signal-date back-resolution walks S3 from the trade's date backward.
For each candidate date in [trade_date - lookback, trade_date], if
signals/{date}/signals.json exists and contains the trade's ticker as
ENTER (or any signal action — we match liberally because exits and
reduces also originate from signal events), that date is the
signal_trading_day. The first match wins (most recent signal that
referenced the ticker).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import UTC, date, datetime

logger = logging.getLogger(__name__)


# ── Lib helpers ─────────────────────────────────────────────────────────────


def _import_lib_dates():
    """Import nousergon_lib.dates with a clear error if the pin is stale."""
    try:
        from nousergon_lib.dates import session_for_timestamp
        return session_for_timestamp
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: nousergon_lib.dates not importable. "
            "This script requires alpha-engine-lib v0.2.0+. "
            "Update requirements.txt and reinstall before running.\n"
            f"Underlying ImportError: {exc}\n"
        )
        sys.exit(2)


# ── Column-presence guard ───────────────────────────────────────────────────


def _ensure_columns_exist(conn: sqlite3.Connection) -> None:
    """Verify trading_day and signal_trading_day columns are present.
    The init_db migration in trade_logger.py adds them; this script must
    run AFTER trade_logger has touched the DB at least once on this deploy.
    Hard-fail with a clear pointer if missing."""
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    missing = {"trading_day", "signal_trading_day"} - cols
    if missing:
        sys.stderr.write(
            f"ERROR: trades table is missing columns {missing}. "
            f"Run trade_logger.init_db() against this DB first to apply "
            f"the schema migration, then re-run the backfill.\n"
        )
        sys.exit(2)


# ── trading_day backfill (from fill_time / created_at) ──────────────────────


def _trading_day_for_row(row: dict, session_for_timestamp) -> str | None:
    """Compute trading_day for a row from its existing timestamp columns.

    Prefers fill_time (when the IB fill landed) and falls back to
    created_at (when log_trade inserted the row). Returns None only if
    neither column has a parseable timestamp — indicates malformed
    historical data, logged as a warning.
    """
    for col in ("fill_time", "created_at"):
        val = row.get(col)
        if not val:
            continue
        try:
            ts = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            return session_for_timestamp(ts)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Could not parse %s=%r for trade_id=%s: %s",
                col, val, row.get("trade_id"), exc,
            )
            continue
    return None


# ── signal_trading_day back-resolution (S3 signals.json walk) ───────────────


class _SignalIndex:
    """Lazy ticker→signal_trading_day cache.

    Walks signals/{date}/signals.json in S3 once per backfill run, building
    a map of (ticker, set-of-signal-dates) so per-trade lookups are O(1).
    Avoids repeated S3 GETs when many trades share the same originating
    signal week.
    """

    def __init__(self, bucket: str, lookback_days: int):
        self._bucket = bucket
        self._lookback_days = lookback_days
        self._ticker_to_signal_dates: dict[str, list[str]] = {}
        self._loaded = False

    def _load(self, earliest: date) -> None:
        if self._loaded:
            return
        try:
            import boto3
        except ImportError as exc:
            sys.stderr.write(f"ERROR: boto3 required for signal back-resolve: {exc}\n")
            sys.exit(2)
        s3 = boto3.client("s3")
        # List signals/ keys under the bucket. The prefix is intentionally
        # broad — we filter to ones >= earliest after listing.
        paginator = s3.get_paginator("list_objects_v2")
        n_files = 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix="signals/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/signals.json"):
                    continue
                # signals/{date}/signals.json — extract date.
                parts = key.split("/")
                if len(parts) < 3:
                    continue
                signal_date = parts[1]
                try:
                    dt = date.fromisoformat(signal_date)
                except ValueError:
                    continue
                if dt < earliest:
                    continue
                try:
                    body = s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
                    payload = json.loads(body)
                except Exception as exc:
                    logger.warning("Could not read s3://%s/%s: %s", self._bucket, key, exc)
                    continue
                # Walk universe + buy_candidates + any other ticker-bearing
                # arrays. Liberal match: any ticker that appears in this
                # signals.json is considered a candidate originator.
                tickers = self._extract_tickers(payload)
                for t in tickers:
                    self._ticker_to_signal_dates.setdefault(t, []).append(signal_date)
                n_files += 1
        # Each ticker's date list sorted descending so most-recent is first.
        for t in self._ticker_to_signal_dates:
            self._ticker_to_signal_dates[t].sort(reverse=True)
        self._loaded = True
        logger.info(
            "Signal index built: %d signals.json files, %d unique tickers",
            n_files, len(self._ticker_to_signal_dates),
        )

    @staticmethod
    def _extract_tickers(payload: dict) -> set[str]:
        tickers: set[str] = set()
        # Common locations in the signals.json schema. Match liberally —
        # back-resolution is best-effort and a missing match yields NULL,
        # which is the documented soft-fail outcome.
        for key in ("universe", "buy_candidates", "watchlist", "exits"):
            arr = payload.get(key) or []
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        t = item.get("ticker")
                        if t:
                            tickers.add(t)
                    elif isinstance(item, str):
                        tickers.add(item)
            elif isinstance(arr, dict):
                # universe sometimes keyed by ticker.
                for t in arr.keys():
                    tickers.add(t)
        return tickers

    def lookup(self, ticker: str, trade_date: str) -> str | None:
        """Return the most recent signal_trading_day ≤ trade_date for ticker."""
        if not self._loaded:
            try:
                earliest = date.fromisoformat(trade_date) - _timedelta_days(self._lookback_days)
            except ValueError:
                earliest = date.fromisoformat("2026-01-01")
            self._load(earliest)
        candidates = self._ticker_to_signal_dates.get(ticker, [])
        for sd in candidates:
            if sd <= trade_date:
                return sd
        return None


def _timedelta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


# ── Main loop ───────────────────────────────────────────────────────────────


def backfill(
    db_path: str,
    bucket: str,
    *,
    dry_run: bool = False,
    signal_lookback_days: int = 30,
) -> dict:
    """Run the backfill. Returns a stats dict for operator visibility."""
    session_for_timestamp = _import_lib_dates()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_columns_exist(conn)

    rows = conn.execute(
        """
        SELECT trade_id, ticker, action, date, fill_time, created_at,
               trading_day, signal_trading_day
        FROM trades
        ORDER BY created_at
        """
    ).fetchall()

    signal_index = _SignalIndex(bucket=bucket, lookback_days=signal_lookback_days)

    stats = {
        "total_rows": len(rows),
        "trading_day_filled": 0,
        "trading_day_already_set": 0,
        "trading_day_unresolved": 0,
        "signal_trading_day_filled": 0,
        "signal_trading_day_already_set": 0,
        "signal_trading_day_unresolved_enter": 0,
        "non_enter_skipped": 0,
        "writes_committed": 0,
    }

    for row in rows:
        row_d = dict(row)
        updates: dict[str, str] = {}

        # trading_day backfill
        if row_d.get("trading_day"):
            stats["trading_day_already_set"] += 1
        else:
            td = _trading_day_for_row(row_d, session_for_timestamp)
            if td:
                updates["trading_day"] = td
                stats["trading_day_filled"] += 1
            else:
                stats["trading_day_unresolved"] += 1

        # signal_trading_day backfill (ENTER actions only — others stay NULL)
        if row_d.get("signal_trading_day"):
            stats["signal_trading_day_already_set"] += 1
        elif row_d.get("action") == "ENTER" and row_d.get("ticker") and row_d.get("date"):
            std = signal_index.lookup(row_d["ticker"], row_d["date"])
            if std:
                updates["signal_trading_day"] = std
                stats["signal_trading_day_filled"] += 1
            else:
                stats["signal_trading_day_unresolved_enter"] += 1
        else:
            stats["non_enter_skipped"] += 1

        if updates and not dry_run:
            # S608 false positive: `updates` keys are always one of the two
            # hardcoded literals "trading_day"/"signal_trading_day" set above
            # in this function (never derived from row/user data), and every
            # actual value is passed through the `?` parameter placeholders
            # below — no untrusted data reaches the query text.
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())  # noqa: S608
            params = list(updates.values()) + [row_d["trade_id"]]
            conn.execute(
                f"UPDATE trades SET {set_clause} WHERE trade_id = ?",  # noqa: S608
                params,
            )
            stats["writes_committed"] += 1

    if not dry_run:
        conn.commit()

    return stats


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db-path", required=True, help="Path to trades.db on disk.")
    p.add_argument(
        "--bucket",
        default="alpha-engine-research",
        help="S3 bucket for signals.json walk (default: alpha-engine-research).",
    )
    p.add_argument(
        "--signal-lookback-days",
        type=int,
        default=30,
        help="How many calendar days to walk back when listing signals.json (default 30).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates but don't write to the database.",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    stats = backfill(
        db_path=args.db_path,
        bucket=args.bucket,
        dry_run=args.dry_run,
        signal_lookback_days=args.signal_lookback_days,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
