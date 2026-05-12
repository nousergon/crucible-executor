#!/usr/bin/env python
"""
Replay the morning planner's shadow portfolio-optimizer call against today's
S3 inputs + a fresh IBKR account snapshot. Writes the result to a separate
S3 replay key so it does not clobber today's live shadow log or the
``latest.json`` pointer.

Motivation: 2026-05-12 morning planner's first shadow firing tripped the
universe-shape ``TypeError`` (fixed in alpha-engine #165). Tomorrow's
weekday SF is the next live exercise of the fixed code path; this replay
exercises it today against the same S3 inputs the planner saw this morning.

Run from the trading EC2 (needs IB Gateway on 127.0.0.1:4002):

    ae-trading "cd ~/alpha-engine && source .venv/bin/activate && \\
        python scripts/replay_shadow_optimizer.py"

Output:

    s3://<signals_bucket>/predictor/optimizer_shadow/replays/{run_date}_replay_{ts}.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone

import boto3
import yaml

from executor.ibkr import IBKRClient
from executor.optimizer_shadow import _build_and_solve
from executor.price_cache import load_price_histories
from executor.signal_reader import read_predictions, read_signals_with_fallback
from executor.strategies.exit_manager import SECTOR_ETF_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("replay_shadow_optimizer")


def _extract_universe_tickers(universe) -> list[str]:
    out: list[str] = []
    for item in universe or []:
        if isinstance(item, dict) and "ticker" in item:
            out.append(item["ticker"])
        elif isinstance(item, str):
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-date",
        default=str(date.today()),
        help="Run date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--config",
        default="config/risk.yaml",
        help="Path to risk.yaml.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    signals_bucket = config["signals_bucket"]
    run_date = args.run_date
    logger.info(
        "Replay shadow optimizer | run_date=%s | bucket=%s", run_date, signals_bucket,
    )

    signals_raw = read_signals_with_fallback(signals_bucket, run_date)
    predictions_by_ticker, predictions_date = read_predictions(signals_bucket)
    logger.info(
        "Predictions loaded | date=%s | n=%d",
        predictions_date, len(predictions_by_ticker),
    )

    ibkr = IBKRClient(
        host=config["ibkr_host"],
        port=config["ibkr_port"],
        client_id=config["ibkr_client_id"],
        reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
    )
    try:
        portfolio_nav = ibkr.get_portfolio_nav()
        current_positions = ibkr.get_positions()
    finally:
        ibkr.disconnect()

    logger.info(
        "IBKR snapshot | NAV=$%.0f | n_positions=%d",
        portfolio_nav, len(current_positions),
    )

    tickers_for_prices: set[str] = set()
    tickers_for_prices.update(current_positions.keys())
    tickers_for_prices.update(predictions_by_ticker.keys())
    tickers_for_prices.update(_extract_universe_tickers(signals_raw.get("universe", [])))
    for sig in signals_raw.get("signals", {}).values():
        if isinstance(sig, dict) and sig.get("ticker"):
            tickers_for_prices.add(sig["ticker"])
    tickers_for_prices.add("SPY")
    tickers_for_prices.update(SECTOR_ETF_MAP.values())

    logger.info("Loading price histories for %d tickers", len(tickers_for_prices))
    price_histories = load_price_histories(
        tickers=sorted(tickers_for_prices),
        signals_bucket=signals_bucket,
    )

    log = _build_and_solve(
        signals_raw=signals_raw,
        predictions_by_ticker=predictions_by_ticker,
        current_positions=current_positions,
        portfolio_nav=portfolio_nav,
        price_histories=price_histories,
        config=config,
        run_date=run_date,
        legacy_orders=[],
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    replay_key = f"predictor/optimizer_shadow/replays/{run_date}_replay_{ts}.json"
    body = json.dumps(log, default=str, indent=2).encode("utf-8")
    boto3.client("s3").put_object(
        Bucket=signals_bucket,
        Key=replay_key,
        Body=body,
        ContentType="application/json",
    )
    logger.info("Wrote replay artifact: s3://%s/%s", signals_bucket, replay_key)

    diag = log.get("diagnostics", {})
    tickers = log.get("tickers", [])
    weights = log.get("target_weights", [])
    w_by_ticker = dict(zip(tickers, weights))
    n_trades = len(log.get("would_be_trades", []))
    print("")
    print("── Shadow optimizer replay summary ──")
    print(f"  status:              {diag.get('status')}")
    print(f"  n_active_positions:  {diag.get('n_active_positions')}")
    print(f"  portfolio_vol_ann:   {diag.get('portfolio_vol_ann')}")
    print(f"  active_share_vs_spy: {diag.get('active_share_vs_spy')}")
    print(f"  turnover_one_way:    {diag.get('turnover_one_way')}")
    print(f"  expected_alpha:      {diag.get('expected_alpha')}")
    print(f"  weight_sum:          {diag.get('weight_sum')}")
    print(f"  cash_weight:         {w_by_ticker.get('CASH')}")
    print(f"  spy_weight:          {w_by_ticker.get('SPY')}")
    print(f"  n_would_be_trades:   {n_trades}")
    print(f"  artifact:            s3://{signals_bucket}/{replay_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
