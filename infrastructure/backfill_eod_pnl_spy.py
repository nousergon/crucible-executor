"""One-shot backfill for eod_pnl.csv — repair SPY return misattribution.

Every row in `s3://alpha-engine-research/trades/eod_pnl.csv` has `spy_return_pct`
computed from T-1 vs T-2 closes because DataPhase1 ran pre-market and polygon
grouped-daily silently returned T-1's aggregate stamped under T in ArcticDB.
The forward-fix (alpha-engine#62, alpha-engine-data#56/#58) corrects all rows
from 2026-04-20 onward. This script corrects the historical rows.

For each row: re-fetch authoritative SPY close for the row's date and prior
trading session via yfinance, recompute `spy_return_pct`, `daily_alpha_pct`,
and rewrite `spy_close`. Original CSV is snapshotted to
`eod_pnl.backup.2026-04-20.csv` on S3 before overwrite.

ArcticDB macro/SPY is NOT consulted — it carries the same one-session shift
and would re-introduce the bug.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import timedelta
from pathlib import Path

import boto3
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_eod_pnl_spy")

BUCKET = "alpha-engine-research"
KEY = "trades/eod_pnl.csv"
BACKUP_KEY = "trades/eod_pnl.backup.2026-04-20.csv"


def fetch_spy_closes(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return a Series indexed by trading-day date → SPY close."""
    lookback_start = (start - timedelta(days=10)).strftime("%Y-%m-%d")
    lookback_end = (end + timedelta(days=2)).strftime("%Y-%m-%d")
    log.info("Fetching SPY closes from yfinance %s → %s", lookback_start, lookback_end)
    hist = yf.Ticker("SPY").history(start=lookback_start, end=lookback_end, auto_adjust=False)
    if hist.empty:
        raise RuntimeError("yfinance returned empty SPY history")
    closes = hist["Close"].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def prior_trading_close(closes: pd.Series, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    prior = closes.loc[:target - timedelta(days=1)]
    if prior.empty:
        raise RuntimeError(f"no prior trading session before {target.date()}")
    prior_date = prior.index[-1]
    return prior_date, float(prior.iloc[-1])


def recompute(df: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for i, row in out.iterrows():
        target = pd.Timestamp(row["date"]).normalize()
        if target not in closes.index:
            log.warning(
                "%s: no SPY close (market closed) — leaving row unchanged",
                target.date(),
            )
            continue
        today_close = float(closes.loc[target])
        prior_date, prior_close = prior_trading_close(closes, target)
        spy_return = (today_close / prior_close - 1.0) * 100.0
        daily_return = row.get("daily_return_pct")
        daily_alpha = (
            float(daily_return) - spy_return if pd.notna(daily_return) else None
        )
        old_close = row.get("spy_close")
        old_return = row.get("spy_return_pct")
        log.info(
            "%s: spy_close %s → %.4f | spy_return %s → %.4f | alpha → %s",
            target.date(),
            f"{float(old_close):.4f}" if pd.notna(old_close) else "None",
            today_close,
            f"{float(old_return):.4f}" if pd.notna(old_return) else "None",
            spy_return,
            f"{daily_alpha:.4f}" if daily_alpha is not None else "None",
        )
        out.at[i, "spy_close"] = today_close
        out.at[i, "spy_return_pct"] = spy_return
        if daily_alpha is not None:
            out.at[i, "daily_alpha_pct"] = daily_alpha
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute corrections + print diff, skip any S3 write",
    )
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=None,
        help="read from local CSV instead of S3 (useful for verification)",
    )
    parser.add_argument(
        "--write-local",
        type=Path,
        default=None,
        help="also write corrected CSV to this local path",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3")

    if args.local_csv:
        log.info("Reading local CSV: %s", args.local_csv)
        raw = args.local_csv.read_bytes()
    else:
        log.info("Reading s3://%s/%s", BUCKET, KEY)
        raw = s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()

    df = pd.read_csv(io.BytesIO(raw))
    log.info("Loaded %d rows, dates %s → %s", len(df), df["date"].iloc[0], df["date"].iloc[-1])

    closes = fetch_spy_closes(
        pd.Timestamp(df["date"].iloc[0]), pd.Timestamp(df["date"].iloc[-1])
    )
    log.info("Fetched %d SPY closes", len(closes))

    corrected = recompute(df, closes)

    buf = io.StringIO()
    corrected.to_csv(buf, index=False)
    corrected_bytes = buf.getvalue().encode("utf-8")

    if args.write_local:
        args.write_local.write_bytes(corrected_bytes)
        log.info("Wrote corrected CSV locally: %s", args.write_local)

    if args.dry_run:
        log.info("--dry-run: skipping S3 writes")
        return 0

    log.info("Uploading backup: s3://%s/%s", BUCKET, BACKUP_KEY)
    s3.put_object(Bucket=BUCKET, Key=BACKUP_KEY, Body=raw, ContentType="text/csv")

    log.info("Uploading corrected: s3://%s/%s", BUCKET, KEY)
    s3.put_object(Bucket=BUCKET, Key=KEY, Body=corrected_bytes, ContentType="text/csv")

    log.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
