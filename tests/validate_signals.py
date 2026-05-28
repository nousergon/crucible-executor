"""
Validate signals.json against executor's expected schema.

Run against a local file or fetch from S3:
    python tests/validate_signals.py signals.json
    python tests/validate_signals.py --s3 2026-03-24
    python tests/validate_signals.py --s3              # today's date

Exits 0 if valid, 1 if errors found. Designed to run in CI or as a
pre-deploy check after research generates new signals.
"""

from __future__ import annotations

import argparse
import json
import sys

_VALID_SIGNALS = {"ENTER", "EXIT", "REDUCE", "HOLD"}
_VALID_CONVICTIONS = {"rising", "stable", "declining"}
_VALID_SECTOR_RATINGS = {"overweight", "market_weight", "underweight"}
# 3-class Ang-Bekaert taxonomy (v0.42.0 / 2026-05-28 —
# caution-regime-retirement-260528.md). Legacy "caution" grandfathered
# on read for historical signals.json artifacts but no longer emitted
# by the macro agent (whose _validate_regime coerces raw LLM caution
# → neutral upstream).
_VALID_REGIMES = {"bull", "neutral", "bear", "caution"}


def validate(data: dict) -> list[str]:
    """Validate signals.json structure. Returns list of error strings."""
    errors: list[str] = []

    # Top-level keys
    if "universe" not in data:
        errors.append("Missing top-level key: 'universe'")
        return errors  # can't continue without universe

    if not isinstance(data["universe"], list):
        errors.append("'universe' must be a list")
        return errors

    regime = data.get("market_regime")
    if regime and regime not in _VALID_REGIMES:
        errors.append(f"Invalid market_regime: '{regime}' (expected {_VALID_REGIMES})")

    # Per-stock validation
    for i, stock in enumerate(data["universe"]):
        ticker = stock.get("ticker", f"<index {i}>")
        prefix = f"universe[{ticker}]"

        if not stock.get("ticker"):
            errors.append(f"{prefix}: missing 'ticker'")

        signal = stock.get("signal")
        if signal not in _VALID_SIGNALS:
            errors.append(f"{prefix}: invalid signal '{signal}' (expected {_VALID_SIGNALS})")

        conviction = stock.get("conviction")
        if conviction not in _VALID_CONVICTIONS:
            errors.append(f"{prefix}: invalid conviction '{conviction}' (expected {_VALID_CONVICTIONS})")

        score = stock.get("score")
        if score is not None and not isinstance(score, (int, float)):
            errors.append(f"{prefix}: score must be numeric or null, got {type(score).__name__}")

        sector_rating = stock.get("sector_rating")
        if sector_rating and sector_rating not in _VALID_SECTOR_RATINGS:
            errors.append(f"{prefix}: invalid sector_rating '{sector_rating}' (expected {_VALID_SECTOR_RATINGS})")

    # Buy candidates (if present) should be a subset of universe tickers with ENTER signal
    candidates = data.get("buy_candidates", [])
    for c in candidates:
        ticker = c.get("ticker", "?")
        conviction = c.get("conviction")
        if conviction and conviction not in _VALID_CONVICTIONS:
            errors.append(f"buy_candidates[{ticker}]: invalid conviction '{conviction}'")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate signals.json for executor compatibility")
    parser.add_argument("file", nargs="?", help="Path to local signals.json file")
    parser.add_argument("--s3", nargs="?", const="today", metavar="DATE",
                        help="Fetch from S3 (default: today's date)")
    args = parser.parse_args()

    if args.s3:
        import os
        os.environ.setdefault("PATH", "/opt/homebrew/bin:" + os.environ.get("PATH", ""))
        import boto3
        from datetime import date
        d = args.s3 if args.s3 != "today" else str(date.today())
        bucket = "alpha-engine-research"
        key = f"signals/{d}/signals.json"
        print(f"Fetching s3://{bucket}/{key}")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
    elif args.file:
        with open(args.file) as f:
            data = json.load(f)
    else:
        parser.error("Provide a file path or --s3 [DATE]")

    errors = validate(data)

    # Summary
    universe = data.get("universe", [])
    print(f"Date: {data.get('date', '?')}")
    print(f"Regime: {data.get('market_regime', '?')}")
    print(f"Universe: {len(universe)} stocks")
    from collections import Counter
    signals = Counter(s.get("signal") for s in universe)
    convictions = Counter(s.get("conviction") for s in universe)
    print(f"Signals: {dict(signals)}")
    print(f"Convictions: {dict(convictions)}")

    if errors:
        print(f"\nFAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nPASSED — signals.json is executor-compatible")
        sys.exit(0)


if __name__ == "__main__":
    main()
