"""
Load OHLCV price histories from the predictor's S3 caches.

Uses the slim cache (2y per-ticker parquets, refreshed weekly Sunday)
as the primary source. No new yfinance fetches required.

S3 layout:
    s3://alpha-engine-research/predictor/price_cache_slim/{TICKER}.parquet
    Columns: Open, High, Low, Close, Volume (capitalized)
    Index: DatetimeIndex (timezone-naive)
"""

from __future__ import annotations

import io
import logging

import boto3
import pandas as pd

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
