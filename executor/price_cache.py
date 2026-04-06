"""
Load OHLCV price histories from the predictor's S3 caches.

Uses the slim cache (2y per-ticker parquets, refreshed weekly Sunday)
as the primary source. No new yfinance fetches required.

S3 layout:
    s3://alpha-engine-research/predictor/price_cache_slim/{TICKER}.parquet
    Columns: Open, High, Low, Close, Volume (capitalized)
    Index: DatetimeIndex (timezone-naive)

    s3://alpha-engine-research/predictor/daily_closes/{date}.parquet
    Columns: date, Open, High, Low, Close, Adj_Close, Volume, VWAP
    Index: ticker (str)
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import boto3
import pandas as pd

from executor.market_hours import is_trading_day

logger = logging.getLogger(__name__)


def load_price_histories(
    tickers: list[str],
    signals_bucket: str,
) -> dict[str, list[dict]]:
    """
    Load OHLCV histories for a list of tickers from predictor slim cache on S3.

    Returns:
        {ticker: [{date, open, high, low, close}, ...]} sorted ascending by date.
        Tickers without cached data are omitted.
    """
    s3 = boto3.client("s3")
    histories: dict[str, list[dict]] = {}

    for ticker in tickers:
        key = f"predictor/price_cache_slim/{ticker}.parquet"
        try:
            obj = s3.get_object(Bucket=signals_bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception as e:
            logger.debug(f"No slim cache for {ticker}: {e}")
            continue

        if df.empty:
            continue

        # Normalize column names to lowercase for exit_manager compatibility
        df.columns = [c.lower() for c in df.columns]

        # Index is DatetimeIndex — convert to date strings
        records = []
        for dt, row in df.iterrows():
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        histories[ticker] = records
        logger.debug(f"Loaded {len(records)} bars for {ticker} from slim cache")

    logger.info(f"Price histories loaded for {len(histories)}/{len(tickers)} tickers from S3 slim cache")
    return histories


def load_daily_vwap(
    signals_bucket: str,
    run_date: str | None = None,
    max_lookback: int = 5,
) -> dict[str, float]:
    """
    Load VWAP values from the most recent daily_closes parquet on S3.

    Scans backward from run_date (skipping weekends/holidays) to find
    the most recent daily_closes file with VWAP data.

    Returns:
        {ticker: vwap} for tickers with a valid VWAP value.
        Empty dict if no file found or no VWAP column.
    """
    s3 = boto3.client("s3")
    start = date.fromisoformat(run_date) if run_date else date.today()

    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        if candidate.weekday() > 4:
            continue
        if not is_trading_day(candidate):
            continue

        key = f"predictor/daily_closes/{candidate.isoformat()}.parquet"
        try:
            obj = s3.get_object(Bucket=signals_bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception:
            continue

        if "VWAP" not in df.columns:
            logger.info("daily_closes/%s has no VWAP column — skipping", candidate)
            continue

        vwap_map: dict[str, float] = {}
        for ticker, row in df.iterrows():
            v = row.get("VWAP")
            if pd.notna(v) and v > 0:
                vwap_map[str(ticker)] = float(v)

        if vwap_map:
            logger.info("Loaded VWAP for %d tickers from daily_closes/%s", len(vwap_map), candidate)
            return vwap_map

    logger.warning("No daily_closes with VWAP found in last %d days", max_lookback)
    return {}
