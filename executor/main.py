"""
Alpha Engine Executor — daily morning trading loop.

Reads signals.json from S3, applies risk rules and position sizing,
places market orders via IB Gateway (paper trading).

Strategy layer (added 2026-03-14):
  - Graduated drawdown response scales position sizes by drawdown tier
  - ATR trailing stops exit positions when price falls below trailing stop
  - Time-based decay reduces/exits stale positions after N trading days

Cron (EC2, America/Los_Angeles):
    30 6 * * 1-5  python /home/ec2-user/alpha-engine/executor/main.py >> /var/log/executor.log 2>&1

Usage:
    python main.py              # live paper trading
    python main.py --dry-run    # print orders without placing them
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.ibkr import IBKRClient, SimulatedIBKRClient
from executor.position_sizer import compute_position_size
from executor.risk_guard import check_order, compute_drawdown_multiplier
from executor.signal_reader import get_actionable_signals, read_signals_with_fallback
from executor.strategies.config import load_strategy_config
from executor.strategies.exit_manager import evaluate_exits
from executor.price_cache import load_price_histories
from executor.trade_logger import backup_to_s3, get_entry_dates, init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")

# S3-delivered executor params (loaded once per cold-start)
_executor_params_cache: dict | None = None
_executor_params_loaded: bool = False

# Flat param name → nested config path mapping
_PARAM_MAP = {
    "atr_multiplier": ("strategy", "exit_manager", "atr_multiplier"),
    "time_decay_reduce_days": ("strategy", "exit_manager", "time_decay_reduce_days"),
    "time_decay_exit_days": ("strategy", "exit_manager", "time_decay_exit_days"),
    "min_score": ("min_score_to_enter",),
    "max_position_pct": ("max_position_pct",),
}


_EXECUTOR_PARAMS_CACHE_PATH = Path(__file__).resolve().parent.parent / "config" / ".executor_params_cache.json"


def _load_executor_params_from_s3(bucket: str) -> dict | None:
    """Read config/executor_params.json from S3. Cache per cold-start.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    params survive transient S3 failures.
    """
    global _executor_params_cache, _executor_params_loaded
    if _executor_params_loaded:
        return _executor_params_cache
    _executor_params_loaded = True

    try:
        import json
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/executor_params.json")
        data = json.loads(obj["Body"].read())
        # Only keep safe-to-override params
        safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
        if safe:
            logger.info("Loaded executor params from S3: %s", safe)
            _executor_params_cache = safe
            # Persist to local cache for fault tolerance
            try:
                _EXECUTOR_PARAMS_CACHE_PATH.write_text(json.dumps(safe, indent=2))
            except Exception:
                pass  # best-effort
        return _executor_params_cache
    except Exception as e:
        logger.warning("Could not read executor params from S3: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if _EXECUTOR_PARAMS_CACHE_PATH.exists():
            import json
            data = json.loads(_EXECUTOR_PARAMS_CACHE_PATH.read_text())
            safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
            if safe:
                logger.info("Loaded executor params from local cache (last known optimal): %s", safe)
                _executor_params_cache = safe
                return _executor_params_cache
    except Exception as e2:
        logger.debug("Could not read local executor params cache: %s", e2)

    return None


def _merge_s3_params(config: dict, s3_params: dict) -> dict:
    """Merge flat S3 param names into nested config structure."""
    for param, value in s3_params.items():
        path = _PARAM_MAP.get(param)
        if not path:
            continue
        target = config
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    return config


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run(
    dry_run: bool = False,
    simulate: bool = False,
    ibkr_client=None,           # injected by backtester when simulate=True
    signals_override: dict = None,  # injected signals dict (skips S3 read)
    price_histories: dict = None,   # injected by backtester for exit manager
    config_override: dict = None,   # injected by backtester param sweep
) -> "list[dict] | None":
    """
    Returns list of order dicts when simulate=True, else None.
    All other behaviour (risk guard, position sizer, trade logger) is unchanged.
    """
    orders = []
    run_date = str(date.today())
    logger.info(f"Executor starting | date={run_date} | dry_run={dry_run} | simulate={simulate}")

    config = load_config()
    if config_override:
        for key, val in config_override.items():
            if key == "strategy" and isinstance(val, dict) and "strategy" in config:
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict) and isinstance(config["strategy"].get(sub_key), dict):
                        config["strategy"][sub_key].update(sub_val)
                    else:
                        config["strategy"][sub_key] = sub_val
            else:
                config[key] = val
    # Merge S3-delivered params (backtester recommendations) if not in simulate mode
    if not simulate and not config_override:
        s3_params = _load_executor_params_from_s3(config.get("signals_bucket", "alpha-engine-research"))
        if s3_params:
            config = _merge_s3_params(config, s3_params)

    db_path = config["db_path"]
    signals_bucket = config["signals_bucket"]
    trades_bucket = config["trades_bucket"]

    conn = None if simulate else init_db(db_path)

    # ── 1. Read signals from S3 (or use injected override) ──────────────────
    signal_source = config.get("signal_source", "research")

    if signals_override is not None:
        signals_raw = signals_override
        run_date = signals_raw.get("date", run_date)
    elif signal_source == "population":
        # NEW PATH: generate trading signals from population + technicals + GBM
        try:
            from executor.population_reader import read_population
            from executor.signal_generator import generate_trading_signals, read_predictions

            pop_data = read_population(signals_bucket)
            predictions = read_predictions(signals_bucket)
            pop_tickers = [p["ticker"] for p in pop_data.get("population", [])]
            price_histories_for_scoring = load_price_histories(
                tickers=pop_tickers,
                signals_bucket=signals_bucket,
            )
            signals_raw = generate_trading_signals(
                population=pop_data["population"],
                predictions=predictions,
                price_histories=price_histories_for_scoring,
                market_regime=pop_data.get("market_regime", "neutral"),
                sector_ratings=pop_data.get("sector_ratings", {}),
                config=config,
            )
            logger.info("Signal source: population-based (technical + GBM)")
        except Exception as e:
            logger.error(f"Population-based signal generation failed: {e}")
            if conn:
                conn.close()
            return
    else:
        # LEGACY PATH: read Research signals.json directly
        try:
            signals_raw = read_signals_with_fallback(signals_bucket, run_date)
        except RuntimeError as e:
            logger.error(f"Cannot proceed without signals: {e}")
            if conn:
                conn.close()
            return
    signals = get_actionable_signals(signals_raw)
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]

    # Load GBM predictions for rationale capture (reuse if already loaded in population path)
    if not simulate:
        try:
            from executor.signal_generator import read_predictions
            predictions_by_ticker = read_predictions(signals_bucket)
        except Exception:
            predictions_by_ticker = {}
    else:
        predictions_by_ticker = {}

    logger.info(
        f"Signals | regime={market_regime} "
        f"| ENTER={len(signals['enter'])} EXIT={len(signals['exit'])} "
        f"REDUCE={len(signals['reduce'])} HOLD={len(signals['hold'])}"
    )

    # ── 2. Connect to IBKR (or use injected simulated client) ───────────────
    if simulate:
        ibkr = ibkr_client
    else:
        ibkr = IBKRClient(
            host=config["ibkr_host"],
            port=config["ibkr_port"],
            client_id=config["ibkr_client_id"],
        )
    portfolio_nav = ibkr.get_portfolio_nav()
    current_positions = ibkr.get_positions()
    peak_nav = ibkr.get_peak_nav(conn)

    # Enrich positions with sector data from signals
    universe_sectors = {
        s["ticker"]: s.get("sector", "")
        for s in signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])
        if s.get("ticker")
    }
    for ticker, pos in current_positions.items():
        pos["sector"] = universe_sectors.get(ticker, "")

    # ── 2b. Enrich positions with entry_date from trades.db ──────────────────
    if conn and current_positions:
        entry_dates = get_entry_dates(conn, list(current_positions.keys()))
        for ticker, pos in current_positions.items():
            pos["entry_date"] = entry_dates.get(ticker)
        logger.info(f"Entry dates resolved for {len(entry_dates)}/{len(current_positions)} positions")

    # ── 2c. Compute graduated drawdown multiplier ──────────────────────────
    dd_multiplier, dd_reason = compute_drawdown_multiplier(portfolio_nav, peak_nav, config)
    if dd_multiplier < 1.0:
        logger.info(f"Drawdown tier active: {dd_reason}")

    # ── 2d. Strategy layer: evaluate exit rules on held positions ──────────
    strategy_config = load_strategy_config(config)

    # Build signals lookup for exit manager
    signals_by_ticker = {}
    for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
        t = s.get("ticker")
        if t and t not in signals_by_ticker:
            signals_by_ticker[t] = s

    # Load price histories from predictor S3 cache (unless injected by backtester)
    if price_histories is None and current_positions:
        price_histories = load_price_histories(
            tickers=list(current_positions.keys()),
            signals_bucket=signals_bucket,
        )

    strategy_exits = evaluate_exits(
        current_positions=current_positions,
        signals_by_ticker=signals_by_ticker,
        run_date=run_date,
        price_histories=price_histories or {},
        ibkr_client=ibkr,
        strategy_config=strategy_config,
    )

    if strategy_exits:
        logger.info(
            f"Strategy layer generated {len(strategy_exits)} exit signal(s): "
            + ", ".join(f"{s['ticker']}({s['action']}: {s['reason']})" for s in strategy_exits)
        )

    enter_signals = signals["enter"]

    # ── 3. Process ENTER signals ─────────────────────────────────────────────
    for sig in enter_signals:
        ticker = sig["ticker"]
        sector = sig.get("sector", "Technology")
        sector_info = sector_ratings.get(sector, {})
        sector_rating_str = sector_info.get("rating", "market_weight")

        if ticker in current_positions:
            logger.info(f"SKIP ENTER {ticker} — already in portfolio")
            continue

        current_price = ibkr.get_current_price(ticker)
        if not current_price:
            logger.warning(f"SKIP ENTER {ticker} — no price available")
            continue

        sizing = compute_position_size(
            ticker=ticker,
            portfolio_nav=portfolio_nav,
            enter_signals=enter_signals,
            signal=sig,
            sector_rating=sector_rating_str,
            current_price=current_price,
            config=config,
            drawdown_multiplier=dd_multiplier,
        )

        if sizing["shares"] == 0:
            logger.info(f"SKIP ENTER {ticker} — position too small (${sizing['dollar_size']:.0f})")
            continue

        # Inject sector_rating into signal for risk guard
        sig_with_sector = {**sig, "sector_rating": sector_rating_str}

        approved, reason = check_order(
            ticker=ticker,
            action="ENTER",
            dollar_size=sizing["dollar_size"],
            portfolio_nav=portfolio_nav,
            peak_nav=peak_nav,
            current_positions=current_positions,
            sector=sector,
            market_regime=market_regime,
            signal=sig_with_sector,
            config=config,
        )

        if not approved:
            logger.info(f"BLOCKED {ticker} — {reason}")
            continue

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER ENTER {ticker} "
            f"{sizing['shares']} shares @ ~${current_price:.2f} "
            f"(${sizing['dollar_size']:.0f}, {sizing['position_pct']*100:.1f}% NAV)"
        )

        if simulate:
            orders.append({
                "date": run_date,
                "ticker": ticker,
                "action": "ENTER",
                "shares": sizing["shares"],
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": sizing["position_pct"],
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": sector_rating_str,
                "market_regime": market_regime,
                "price_target_upside": sig.get("price_target_upside"),
                "thesis_summary": sig.get("thesis_summary"),
            })
        elif not dry_run:
            order_result = ibkr.place_market_order(ticker, "BUY", sizing["shares"])
            pred = predictions_by_ticker.get(ticker, {})
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "ENTER",
                "shares": sizing["shares"],
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": sizing["position_pct"],
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": sector_rating_str,
                "market_regime": market_regime,
                "price_target_upside": sig.get("price_target_upside"),
                "thesis_summary": sig.get("thesis_summary"),
                "ib_order_id": order_result.get("ib_order_id"),
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "rationale_json": json.dumps({
                    "action": "ENTER",
                    "research_score": sig.get("score"),
                    "conviction": sig.get("conviction"),
                    "thesis_summary": sig.get("thesis_summary"),
                    "price_target_upside": sig.get("price_target_upside"),
                    "sector_rating": sector_rating_str,
                    "market_regime": market_regime,
                    "predicted_direction": pred.get("predicted_direction"),
                    "prediction_confidence": pred.get("prediction_confidence"),
                    "predicted_alpha": pred.get("predicted_alpha"),
                    "sizing_factors": {
                        "sector_adj": sizing.get("sector_adj"),
                        "conviction_adj": sizing.get("conviction_adj"),
                        "upside_adj": sizing.get("upside_adj"),
                        "dd_multiplier": sizing.get("dd_multiplier"),
                    },
                    "risk_guard_reason": reason,
                }),
            })

    # ── 4. Process EXIT signals (Research + Strategy) ───────────────────────
    # Merge Research exits with strategy-generated exits (deduplicate by ticker)
    all_exit_tickers = set()
    all_exits = []
    for sig in signals["exit"]:
        t = sig["ticker"]
        if t not in all_exit_tickers:
            all_exit_tickers.add(t)
            all_exits.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig["action"] == "EXIT" and strat_sig["ticker"] not in all_exit_tickers:
            all_exit_tickers.add(strat_sig["ticker"])
            all_exits.append(strat_sig)

    for sig in all_exits:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            logger.info(f"SKIP EXIT {ticker} — not in portfolio")
            continue

        shares_held = int(current_positions[ticker]["shares"])
        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}ORDER EXIT {ticker} {shares_held} shares{reason_tag}")

        if simulate:
            current_price = ibkr.get_current_price(ticker)
            orders.append({
                "date": run_date,
                "ticker": ticker,
                "action": "EXIT",
                "shares": shares_held,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": 0.0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "exit_reason": sig.get("reason"),
            })
        elif not dry_run:
            current_price = ibkr.get_current_price(ticker)
            order_result = ibkr.place_market_order(ticker, "SELL", shares_held)
            pred = predictions_by_ticker.get(ticker, {})
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "EXIT",
                "shares": shares_held,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": 0.0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "ib_order_id": order_result.get("ib_order_id"),
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "rationale_json": json.dumps({
                    "action": "EXIT",
                    "exit_reason": sig.get("reason", "research_signal"),
                    "exit_detail": sig.get("detail", ""),
                    "research_score": sig.get("score"),
                    "predicted_direction": pred.get("predicted_direction"),
                    "predicted_alpha": pred.get("predicted_alpha"),
                }),
            })

    # ── 5. Process REDUCE signals (Research + Strategy) ─────────────────────
    all_reduce_tickers = set()
    all_reduces = []
    for sig in signals["reduce"]:
        t = sig["ticker"]
        if t not in all_reduce_tickers:
            all_reduce_tickers.add(t)
            all_reduces.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig["action"] == "REDUCE" and strat_sig["ticker"] not in all_reduce_tickers:
            # Also skip if we already have an EXIT for this ticker
            if strat_sig["ticker"] not in all_exit_tickers:
                all_reduce_tickers.add(strat_sig["ticker"])
                all_reduces.append(strat_sig)

    for sig in all_reduces:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            continue

        shares_held = int(current_positions[ticker]["shares"])
        shares_to_sell = shares_held // 2
        if shares_to_sell == 0:
            logger.info(f"SKIP REDUCE {ticker} — position too small to halve")
            continue

        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER REDUCE {ticker} "
            f"{shares_to_sell} shares (50% reduction){reason_tag}"
        )

        if simulate:
            current_price = ibkr.get_current_price(ticker)
            remaining_value = (shares_held - shares_to_sell) * (current_price or 0)
            orders.append({
                "date": run_date,
                "ticker": ticker,
                "action": "REDUCE",
                "shares": shares_to_sell,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": remaining_value / portfolio_nav if portfolio_nav else 0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "exit_reason": sig.get("reason"),
            })
        elif not dry_run:
            current_price = ibkr.get_current_price(ticker)
            order_result = ibkr.place_market_order(ticker, "SELL", shares_to_sell)
            remaining_value = (shares_held - shares_to_sell) * (current_price or 0)
            pred = predictions_by_ticker.get(ticker, {})
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "REDUCE",
                "shares": shares_to_sell,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": remaining_value / portfolio_nav if portfolio_nav else 0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "ib_order_id": order_result.get("ib_order_id"),
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "rationale_json": json.dumps({
                    "action": "REDUCE",
                    "exit_reason": sig.get("reason", "research_signal"),
                    "exit_detail": sig.get("detail", ""),
                    "research_score": sig.get("score"),
                    "predicted_direction": pred.get("predicted_direction"),
                    "predicted_alpha": pred.get("predicted_alpha"),
                }),
            })

    # ── 6. Backup and disconnect ─────────────────────────────────────────────
    if not dry_run and not simulate:
        backup_to_s3(db_path, run_date, trades_bucket)

    ibkr.disconnect()
    if conn:
        conn.close()
    logger.info(f"Executor complete | dry_run={dry_run} | simulate={simulate}")

    if simulate:
        return orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orders without placing them",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
