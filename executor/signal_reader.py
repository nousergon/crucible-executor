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


def read_predictions(s3_bucket: str) -> dict[str, dict]:
    """
    Read predictor/predictions/latest.json from S3.

    Returns: {ticker: prediction_dict}. Empty dict if not available.
    """
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key="predictor/predictions/latest.json")
        data = json.loads(obj["Body"].read())
        if data.get("timed_out"):
            logger.warning(
                "Predictions timed out — using partial set (%d tickers)",
                len(data.get("predictions", [])),
            )
        preds = data.get("predictions", [])
        result = {p["ticker"]: p for p in preds if "ticker" in p}
        logger.info("Predictions loaded | n=%d", len(result))
        return result
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning("predictions/latest.json not found — running without GBM input")
            return {}
        raise


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


def read_signals_with_fallback(s3_bucket: str, run_date: str | None = None, max_lookback: int = 14) -> dict:
    """
    Read the latest signals from S3.

    Tries signals/latest.json first (written by Research alongside the dated file).
    Falls back to date-scanning if the pointer doesn't exist.

    The default max_lookback of 14 days covers the Research pipeline's weekly
    cadence (Saturday 00:00 UTC) plus a one-week buffer for a missed Saturday
    run. Shorter windows are fragile: by Friday of any given week, the most
    recent Saturday signals file is already 6 days old, and a 5-day window
    would fail even on a normal week.

    A staleness WARNING is logged when the signals being returned are more
    than 7 calendar days old, so a quietly-missed Saturday run becomes visible
    in the executor's log stream even if the signals still load successfully.

    Returns the signals dict. Raises RuntimeError if nothing found.
    """
    s3 = boto3.client("s3")

    # Try the latest.json pointer first
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key="signals/latest.json")
        data = json.loads(obj["Body"].read())
        logger.info(
            f"Signals loaded from signals/latest.json | date={data.get('date')} "
            f"| universe={len(data.get('universe', []))}"
        )
        _warn_if_stale(data.get("date"), run_date)
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("signals/latest.json not found — falling back to date scan")
        else:
            raise

    # Fallback: scan backward by date
    start = date.fromisoformat(run_date) if run_date else date.today()
    tried: list[str] = []

    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        try:
            signals = read_signals(s3_bucket, str(candidate))
            if days_back > 0:
                log_fn = logger.warning if days_back > 7 else logger.info
                log_fn(
                    f"No signals for {start} — using {candidate} "
                    f"({days_back} calendar day(s) old). Dates tried: {tried}. "
                    f"Research pipeline may have missed a weekly run."
                    if days_back > 7
                    else f"No signals for {start} — using {candidate} "
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


def _warn_if_stale(signals_date: str | None, run_date: str | None) -> None:
    """Log a WARNING if the loaded signals are more than 7 days old relative to run_date.

    Staleness >7 days means the Research pipeline likely missed its Saturday
    run — the signals will still work, but something upstream needs investigation.
    """
    if not signals_date:
        return
    try:
        sig = date.fromisoformat(signals_date)
    except (ValueError, TypeError):
        return
    ref = date.fromisoformat(run_date) if run_date else date.today()
    age = (ref - sig).days
    if age > 7:
        logger.warning(
            f"Loaded signals are {age} calendar days old (signals_date={sig}, "
            f"run_date={ref}). Research pipeline may have missed a weekly run."
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
