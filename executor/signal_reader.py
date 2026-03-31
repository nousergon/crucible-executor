"""
Read signals.json from S3 and return parsed, validated signal data.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def read_signals(s3_bucket: str, run_date: str | None = None) -> dict:
    """
    Download signals/{date}/signals.json from S3.
    Returns parsed signals dict. Raises ClientError if not found.
    """
    d = run_date or str(date.today())
    key = f"signals/{d}/signals.json"
    s3 = boto3.client("s3")
    logger.info(f"Reading signals from s3://{s3_bucket}/{key}")
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    data = json.loads(obj["Body"].read())
    logger.info(
        f"Signals loaded | market_regime={data.get('market_regime')} "
        f"| universe={len(data.get('universe', []))} "
        f"| candidates={len(data.get('buy_candidates', []))}"
    )
    return data


def read_signals_with_fallback(s3_bucket: str, run_date: str | None = None, max_lookback: int = 5) -> dict:
    """
    Try to read signals for run_date, falling back to previous days if not found.

    Research runs on Saturday and writes signals with the Saturday date,
    so the lookback must include weekends.
    Tries up to max_lookback calendar days back before giving up.

    Returns the signals dict. Raises RuntimeError if nothing found within the lookback window.
    """
    start = date.fromisoformat(run_date) if run_date else date.today()
    tried: list[str] = []

    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        try:
            signals = read_signals(s3_bucket, str(candidate))
            if days_back > 0:
                logger.warning(
                    f"No signals for {start} — using {candidate} "
                    f"({days_back} calendar day(s) old). Dates tried: {tried}"
                )
            return signals
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"No signals file for {candidate}, looking further back...")
                tried.append(str(candidate))
                continue
            raise

    raise RuntimeError(
        f"No signals found within {max_lookback} calendar days of {start}. "
        f"Dates tried: {tried}. Check that the research pipeline ran recently."
    )


def get_actionable_signals(signals: dict) -> dict:
    """
    Filter signals to actionable entries by signal type.

    Returns:
        {
            "enter":  [signal, ...],
            "exit":   [signal, ...],
            "reduce": [signal, ...],
            "hold":   [signal, ...],
            "market_regime": str,
            "sector_ratings": {sector: {"rating": str, "modifier": float, "rationale": str}},
        }
    """
    universe = signals.get("universe", [])
    candidates = signals.get("buy_candidates", [])
    # Candidates take precedence — dedupe by ticker
    seen: set[str] = set()
    all_stocks: list[dict] = []
    for s in candidates + universe:
        ticker = s.get("ticker")
        if ticker and ticker not in seen:
            seen.add(ticker)
            all_stocks.append(s)

    return {
        "enter":  [s for s in all_stocks if s.get("signal") == "ENTER"],
        "exit":   [s for s in all_stocks if s.get("signal") == "EXIT"],
        "reduce": [s for s in all_stocks if s.get("signal") == "REDUCE"],
        "hold":   [s for s in all_stocks if s.get("signal") == "HOLD"],
        "market_regime": signals.get("market_regime", "neutral"),
        "sector_ratings": signals.get("sector_ratings", {}),
    }
