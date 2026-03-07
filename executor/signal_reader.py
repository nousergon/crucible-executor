"""
Read signals.json from S3 and return parsed, validated signal data.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import boto3

logger = logging.getLogger(__name__)


def read_signals(s3_bucket: str, run_date: str | None = None) -> dict:
    """
    Download signals/{date}/signals.json from S3.
    Returns parsed signals dict. Raises if not found.
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
