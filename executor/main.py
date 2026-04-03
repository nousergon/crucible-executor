"""
Alpha Engine Executor — daily morning order-book planner.

Reads signals.json from S3, applies risk rules and position sizing,
writes approved entries and urgent exits to the intraday order book.
The daemon (daemon.py) is the sole order executor — it uses technical
triggers to time entries and executes exits immediately.

No orders are placed by this module. All trade execution happens in
the daemon via IB Gateway.

Runs on boot via systemd (alpha-engine-morning.service) on the trading
instance, which is started/stopped daily by the micro instance's cron.

Usage:
    python main.py              # write order book (requires IB Gateway for NAV/positions)
    python main.py --dry-run    # print planned orders without writing order book
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.ibkr import IBKRClient, SimulatedIBKRClient
from executor.order_book import OrderBook
from executor.position_sizer import compute_position_size
from executor.risk_guard import check_order, compute_drawdown_multiplier
from executor.signal_reader import get_actionable_signals, read_signals_with_fallback
from executor.strategies.config import load_strategy_config
from executor.strategies.exit_manager import evaluate_exits, SECTOR_ETF_MAP
from executor.price_cache import load_price_histories
from executor.trade_logger import backup_to_s3, get_entry_dates, init_db, log_trade, log_shadow_book_block

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
    "reduce_fraction": ("reduce_fraction",),
    "atr_sizing_target_risk": ("atr_sizing_target_risk",),
    "confidence_sizing_min": ("confidence_sizing_min",),
    "confidence_sizing_range": ("confidence_sizing_range",),
    "staleness_decay_per_day": ("staleness_decay_per_day",),
    "earnings_sizing_reduction": ("earnings_sizing_reduction",),
    "earnings_proximity_days": ("earnings_proximity_days",),
    "momentum_gate_threshold": ("momentum_gate_threshold",),
    "correlation_block_threshold": ("correlation_block_threshold",),
    "profit_take_pct": ("strategy", "exit_manager", "profit_take_pct"),
    "momentum_exit_threshold": ("strategy", "exit_manager", "momentum_exit_threshold"),
}


# (type, min, max) for each S3-delivered param — values outside range are rejected
_PARAM_VALIDATORS = {
    "atr_multiplier":              (float, 0.5, 10.0),
    "time_decay_reduce_days":      (int,   1,   30),
    "time_decay_exit_days":        (int,   1,   60),
    "min_score":                   (float, 0,   100),
    "max_position_pct":            (float, 0.01, 0.25),
    "reduce_fraction":             (float, 0.1,  1.0),
    "atr_sizing_target_risk":      (float, 0.005, 0.10),
    "confidence_sizing_min":       (float, 0.3,  1.0),
    "confidence_sizing_range":     (float, 0.1,  1.0),
    "staleness_decay_per_day":     (float, 0.0,  0.2),
    "earnings_sizing_reduction":   (float, 0.0,  1.0),
    "earnings_proximity_days":     (int,   1,    30),
    "momentum_gate_threshold":     (float, -30,  0),
    "correlation_block_threshold": (float, 0.3,  1.0),
    "profit_take_pct":             (float, 0.05, 1.0),
    "momentum_exit_threshold":     (float, -50,  0),
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
        # Advisory schema validation (log warnings, never block)
        _unknown_keys = [k for k in data if k not in _PARAM_MAP and k not in (
            "disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend",
            "updated_at", "best_sharpe", "best_alpha", "improvement_pct",
            "n_combos_tested", "manual_override",
        )]
        if _unknown_keys:
            logger.warning("executor_params.json contains unknown keys: %s", _unknown_keys)
        # Only keep safe-to-override params (numeric) + special non-numeric params
        safe = {k: v for k, v in data.items() if k in _PARAM_MAP}
        # Phase 4 non-numeric params: disabled_triggers (list), p_up sizing (bool)
        for special_key in ("disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend"):
            if special_key in data:
                safe[special_key] = data[special_key]
        if safe:
            logger.info("Loaded executor params from S3: %s", safe)
            _executor_params_cache = safe
            # Persist to local cache for fault tolerance
            try:
                _EXECUTOR_PARAMS_CACHE_PATH.write_text(json.dumps(safe, indent=2))
            except Exception:
                logger.debug("Failed to write executor params cache", exc_info=True)
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
        logger.warning("Could not read local executor params cache: %s", e2)

    logger.warning("Both S3 and local cache failed for executor params — using hardcoded defaults")
    return None


def _merge_s3_params(config: dict, s3_params: dict) -> dict[str, Any]:
    """Merge flat S3 param names into nested config structure with validation."""
    for param, value in s3_params.items():
        # Phase 4 non-numeric params: merge directly into top-level config
        if param == "disabled_triggers" and isinstance(value, list):
            config.setdefault("intraday", {}).setdefault("entry_triggers", {})["disabled_triggers"] = value
            logger.info("S3 disabled_triggers: %s", value)
            continue
        if param == "use_p_up_sizing" and isinstance(value, bool):
            config["use_p_up_sizing"] = value
            logger.info("S3 use_p_up_sizing: %s", value)
            continue
        if param == "p_up_sizing_blend" and isinstance(value, (int, float)):
            config["p_up_sizing_blend"] = float(value)
            continue

        path = _PARAM_MAP.get(param)
        if not path:
            continue
        validator = _PARAM_VALIDATORS.get(param)
        if validator:
            expected_type, lo, hi = validator
            if not isinstance(value, (int, float)):
                logger.warning("S3 param %s: invalid type %s — skipping", param, type(value).__name__)
                continue
            value = expected_type(value)
            if not (lo <= value <= hi):
                logger.warning("S3 param %s=%s out of range [%s, %s] — skipping", param, value, lo, hi)
                continue
        target = config
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    return config


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _compute_support_level(price_history: list[dict], strategy_config: dict) -> float | None:
    """Compute N-day low from price history for support-bounce entry trigger."""
    lookback = strategy_config.get("intraday_support_lookback_days", 20)
    if not price_history or len(price_history) < lookback:
        return None
    recent = price_history[-lookback:]
    lows = [bar["low"] for bar in recent if bar.get("low")]
    return min(lows) if lows else None


def _read_signals(
    config: dict,
    signals_bucket: str,
    run_date: str,
    simulate: bool,
    signals_override: dict | None,
    conn,
) -> tuple[dict, dict, str, dict]:
    """Read and validate signals from S3 or override.

    Returns (signals_raw, signals, run_date, predictions_by_ticker).
    """
    signal_source = config.get("signal_source", "research")

    if signals_override is not None:
        signals_raw = signals_override
        run_date = signals_raw.get("date", run_date)
    elif signal_source == "population":
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
            raise RuntimeError(f"Population signal generation failed: {e}")
    else:
        try:
            signals_raw = read_signals_with_fallback(signals_bucket, run_date)
        except RuntimeError as e:
            logger.error(f"Cannot proceed without signals: {e}")
            if conn:
                conn.close()
            raise

    signals = get_actionable_signals(signals_raw)

    # Alert if signals are stale (research didn't run recently)
    if not simulate:
        try:
            signals_date_raw = signals_raw.get("date", run_date)
            _sig_age = (date.fromisoformat(run_date) - date.fromisoformat(signals_date_raw)).days
            if _sig_age > 2:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    f"\u26a0\ufe0f *Stale signals*\n"
                    f"Using signals from {signals_date_raw} ({_sig_age} days old)\n"
                    f"Research may not have run this week."
                )
        except Exception:
            logger.debug("Stale signals Telegram notification failed", exc_info=True)

    # Load GBM predictions for rationale capture
    if not simulate:
        try:
            from executor.signal_generator import read_predictions
            predictions_by_ticker = read_predictions(signals_bucket)
        except Exception as e:
            logger.warning("Failed to load GBM predictions: %s", e)
            predictions_by_ticker = {}
    else:
        predictions_by_ticker = {}

    logger.info(
        f"Signals | regime={signals['market_regime']} "
        f"| ENTER={len(signals['enter'])} EXIT={len(signals['exit'])} "
        f"REDUCE={len(signals['reduce'])} HOLD={len(signals['hold'])}"
    )

    return signals_raw, signals, run_date, predictions_by_ticker


def _plan_entries(
    enter_signals: list[dict],
    signals_raw: dict,
    predictions_by_ticker: dict,
    config: dict,
    strategy_config: dict,
    market_regime: str,
    sector_ratings: dict,
    ibkr,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict,
    price_histories: dict | None,
    dd_multiplier: float,
    signal_age_days: int,
    earnings_by_ticker: dict,
    ob: OrderBook,
    run_date: str,
    dry_run: bool,
    simulate: bool,
) -> tuple[int, list[dict], list[dict]]:
    """Process ENTER signals through momentum gate, sizing, risk guard.

    Returns (n_entered, orders_if_simulating, blocked_entries).
    blocked_entries is a list of {"ticker": str, "reason": str} for transparency.
    """
    orders = []
    n_entered = 0
    blocked: list[dict] = []

    for sig in enter_signals:
        ticker = sig["ticker"]
        sector = sig.get("sector", "Technology")
        sector_info = sector_ratings.get(sector, {})
        sector_rating_str = sector_info.get("rating", "market_weight")

        # Common context for shadow book logging
        _shadow_base = {
            "ticker": ticker,
            "date": run_date,
            "sector": sector,
            "sector_rating": sector_rating_str,
            "research_score": sig.get("score"),
            "conviction": sig.get("conviction"),
            "market_regime": market_regime,
            "portfolio_nav": portfolio_nav,
        }

        if ticker in current_positions:
            logger.info(f"SKIP ENTER {ticker} — already in portfolio")
            blocked.append({**_shadow_base, "block_reason": "already in portfolio"})
            continue

        # Momentum confirmation gate
        if config.get("momentum_gate_enabled", True) and price_histories:
            ticker_history = price_histories.get(ticker, [])
            if len(ticker_history) >= 21:
                momentum_20d = (ticker_history[-1]["close"] / ticker_history[-21]["close"] - 1) * 100
                mom_threshold = config.get("momentum_gate_threshold", -5.0)
                if momentum_20d < mom_threshold:
                    reason = f"momentum gate: 20d={momentum_20d:.1f}% < {mom_threshold}%"
                    logger.info(f"SKIP ENTER {ticker} — {reason}")
                    blocked.append({**_shadow_base, "block_reason": reason})
                    continue

        # Earnings proximity warning
        earnings_warning_days = config.get("earnings_proximity_warning_days", 2)
        pred_data = predictions_by_ticker.get(ticker, {})
        next_earnings_days = earnings_by_ticker.get(ticker) or pred_data.get("next_earnings_days") or sig.get("next_earnings_days")
        if next_earnings_days is not None and next_earnings_days <= earnings_warning_days:
            logger.warning(
                f"EARNINGS WARNING: {ticker} reports in {next_earnings_days} day(s) — "
                f"entering before earnings carries elevated event risk"
            )

        current_price = ibkr.get_current_price(ticker)
        if not current_price:
            logger.warning(f"SKIP ENTER {ticker} — no price available")
            blocked.append({**_shadow_base, "block_reason": "no price available"})
            continue

        # Compute ATR % for ATR-based sizing
        atr_pct = None
        if config.get("atr_sizing_enabled", True) and price_histories:
            ticker_history = price_histories.get(ticker, [])
            if ticker_history:
                from executor.strategies.exit_manager import _compute_atr
                atr_val = _compute_atr(ticker_history, period=14)
                if atr_val and current_price > 0:
                    atr_pct = atr_val / current_price

        # Get prediction confidence for confidence-weighted sizing
        pred_confidence = pred_data.get("prediction_confidence")

        sizing = compute_position_size(
            ticker=ticker,
            portfolio_nav=portfolio_nav,
            enter_signals=enter_signals,
            signal=sig,
            sector_rating=sector_rating_str,
            current_price=current_price,
            config=config,
            drawdown_multiplier=dd_multiplier,
            atr_pct=atr_pct,
            prediction_confidence=pred_confidence,
            p_up=pred_data.get("p_up"),
            signal_age_days=signal_age_days,
            days_to_earnings=earnings_by_ticker.get(ticker),
        )

        if sizing["shares"] == 0:
            logger.info(
                f"SKIP ENTER {ticker} — shares round to 0 "
                f"(weight={sizing['position_pct']:.3f}, dollar=${sizing['dollar_size']:.0f}, "
                f"price=${current_price:.2f})"
            )
            blocked.append({
                **_shadow_base, "block_reason": f"shares round to 0 (${sizing['dollar_size']:.0f} / ${current_price:.2f})",
                "current_price": current_price, "intended_position_pct": sizing["position_pct"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
            continue

        # GBM veto check
        pred_data = predictions_by_ticker.get(ticker, {})
        if pred_data.get("gbm_veto"):
            reason = f"GBM veto: α={pred_data.get('predicted_alpha', 0):.2%}, rank={pred_data.get('combined_rank')}"
            logger.info(f"VETO {ticker} — {reason}")
            blocked.append({
                **_shadow_base, "block_reason": reason,
                "current_price": current_price,
                "intended_position_pct": sizing["position_pct"],
                "intended_shares": sizing["shares"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
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
            price_histories=price_histories,
        )

        if not approved:
            logger.info(f"BLOCKED {ticker} — {reason}")
            blocked.append({
                **_shadow_base, "block_reason": reason,
                "current_price": current_price,
                "intended_position_pct": sizing["position_pct"],
                "intended_shares": sizing["shares"],
                "intended_dollars": sizing["dollar_size"],
                "predicted_direction": pred_data.get("predicted_direction"),
                "prediction_confidence": pred_data.get("prediction_confidence"),
            })
            continue

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER ENTER {ticker} "
            f"{sizing['shares']} shares @ ~${current_price:.2f} "
            f"(${sizing['dollar_size']:.0f}, {sizing['position_pct']*100:.1f}% NAV)"
        )

        n_entered += 1
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
            # Write approved entry to order book — daemon executes via technical triggers
            from executor.strategies.exit_manager import _compute_atr
            ticker_hist = (price_histories or {}).get(ticker, [])
            atr_dollar = _compute_atr(ticker_hist, period=14) if ticker_hist else None

            pred = predictions_by_ticker.get(ticker, {})
            ob.add_entry({
                "ticker": ticker,
                "signal": "ENTER",
                "shares": sizing["shares"],
                "current_price": current_price,
                "dollar_size": sizing["dollar_size"],
                "position_pct": sizing["position_pct"],
                "atr_value": atr_dollar or 0,
                "triggers": {
                    "pullback_pct": strategy_config.get("intraday_pullback_pct", 0.02),
                    "vwap_discount": strategy_config.get("intraday_vwap_discount_pct", 0.005),
                    "support_level": _compute_support_level(ticker_hist, strategy_config),
                },
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": sector_rating_str,
                "market_regime": market_regime,
                "price_target_upside": sig.get("price_target_upside"),
                "thesis_summary": sig.get("thesis_summary"),
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "predicted_alpha": pred.get("predicted_alpha"),
                "sizing_factors": {
                    "sector_adj": sizing.get("sector_adj"),
                    "conviction_adj": sizing.get("conviction_adj"),
                    "upside_adj": sizing.get("upside_adj"),
                    "dd_multiplier": sizing.get("dd_multiplier"),
                    "atr_adj": sizing.get("atr_adj"),
                    "confidence_adj": sizing.get("confidence_adj"),
                    "staleness_adj": sizing.get("staleness_adj"),
                    "earnings_adj": sizing.get("earnings_adj"),
                },
            })

    if len(enter_signals) > 0 and n_entered == 0:
        logger.warning("All %d ENTER signals blocked by risk guard", len(enter_signals))

    return n_entered, orders, blocked


def _plan_exits_and_reduces(
    signals: dict,
    strategy_exits: list[dict],
    predictions_by_ticker: dict,
    current_positions: dict,
    ibkr,
    portfolio_nav: float,
    config: dict,
    market_regime: str,
    ob: OrderBook,
    run_date: str,
    dry_run: bool,
    simulate: bool,
) -> list[dict]:
    """Process EXIT and REDUCE signals, write urgent exits to order book.

    Returns orders list (populated only when simulating).
    """
    orders = []

    # ── EXIT signals (Research + Strategy) ───────────────────────
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
            if current_price is None:
                current_price = current_positions[ticker].get("avg_cost", 0)
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
            pred = predictions_by_ticker.get(ticker, {})
            ob.add_urgent_exit({
                "ticker": ticker,
                "signal": "EXIT",
                "shares": shares_held,
                "reason": sig.get("reason", "research_signal"),
                "detail": sig.get("detail", ""),
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "predicted_alpha": pred.get("predicted_alpha"),
            })

    # ── REDUCE signals (Research + Strategy) ─────────────────────
    all_reduce_tickers = set()
    all_reduces = []
    for sig in signals["reduce"]:
        t = sig["ticker"]
        if t not in all_reduce_tickers:
            all_reduce_tickers.add(t)
            all_reduces.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig["action"] == "REDUCE" and strat_sig["ticker"] not in all_reduce_tickers:
            if strat_sig["ticker"] not in all_exit_tickers:
                all_reduce_tickers.add(strat_sig["ticker"])
                all_reduces.append(strat_sig)

    for sig in all_reduces:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            continue

        shares_held = int(current_positions[ticker]["shares"])
        reduce_frac = config.get("reduce_fraction", 0.50)
        shares_to_sell = int(shares_held * reduce_frac)
        if shares_to_sell == 0:
            logger.info(f"SKIP REDUCE {ticker} — position too small to reduce")
            continue

        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER REDUCE {ticker} "
            f"{shares_to_sell} shares ({reduce_frac:.0%} reduction){reason_tag}"
        )

        if simulate:
            current_price = ibkr.get_current_price(ticker)
            if current_price is None:
                current_price = current_positions[ticker].get("avg_cost", 0)
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
            pred = predictions_by_ticker.get(ticker, {})
            ob.add_urgent_exit({
                "ticker": ticker,
                "signal": "REDUCE",
                "shares": shares_to_sell,
                "reason": sig.get("reason", "research_signal"),
                "detail": sig.get("detail", ""),
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "predicted_direction": pred.get("predicted_direction"),
                "prediction_confidence": pred.get("prediction_confidence"),
                "predicted_alpha": pred.get("predicted_alpha"),
            })

    return orders


def _write_order_book_summary(
    ob: OrderBook,
    blocked_entries: list[dict] | None,
    signals_bucket: str,
    run_date: str,
) -> None:
    """Write a public-safe order book summary to S3 for the dashboard."""
    import boto3

    summary = {
        "date": run_date,
        "entries_approved": [
            {"ticker": e["ticker"]} for e in ob.pending_entries()
        ],
        "entries_blocked": [
            {"ticker": b["ticker"], "reason": b.get("block_reason", b.get("reason", "unknown"))}
            for b in (blocked_entries or [])
        ],
        "exits": [
            {"ticker": e["ticker"], "reason": e.get("reason", "research_signal")}
            for e in ob.pending_urgent_exits()
            if e.get("signal") != "COVER"
        ],
        "covers": [
            {"ticker": e["ticker"]}
            for e in ob.pending_urgent_exits()
            if e.get("signal") == "COVER"
        ],
    }

    try:
        s3 = boto3.client("s3")
        key = f"signals/{run_date}/order_book_summary.json"
        s3.put_object(
            Bucket=signals_bucket,
            Key=key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        logger.info("Order book summary written to s3://%s/%s", signals_bucket, key)
    except Exception as e:
        logger.warning("Failed to write order book summary (non-fatal): %s", e)


def _write_stops_and_finalize(
    ibkr,
    ob: OrderBook,
    price_histories: dict | None,
    strategy_config: dict,
    conn,
    run_date: str,
    blocked_entries: list[dict] | None = None,
    signals_bucket: str | None = None,
) -> None:
    """Write stop records for held positions, detect shorts, save order book, notify."""
    from executor.strategies.exit_manager import _compute_atr

    # Add stop records for all current positions
    current_pos = ibkr.get_positions()
    for t, pos in current_pos.items():
        pos_shares = int(pos.get("shares", 0))
        if pos_shares <= 0:
            continue
        # Skip tickers with pending urgent exits
        urgent_exit_tickers = {u["ticker"] for u in ob.pending_urgent_exits()}
        if t in urgent_exit_tickers:
            continue
        ticker_hist = (price_histories or {}).get(t, [])
        atr_val = _compute_atr(ticker_hist, period=14) if ticker_hist else None
        entry_price = pos.get("avg_cost", 0)
        atr_mult = strategy_config.get("intraday_trailing_stop_atr_multiple", 2.0)
        if not atr_val or atr_val <= 0:
            logger.warning("No ATR for %s — skipping stop (no price history)", t)
            continue
        stop_price = round(entry_price - atr_val * atr_mult, 2)
        ob.add_stop({
            "ticker": t,
            "entry_price": entry_price,
            "current_stop": stop_price,
            "trail_atr": atr_val or 0,
            "atr_multiple": atr_mult,
            "high_water": entry_price,
            "entry_date": (conn and get_entry_dates(conn, [t]).get(t)) or run_date,
            "shares": pos_shares,
        })

    # Detect short positions and add urgent cover orders
    for t, pos in current_pos.items():
        pos_shares = int(pos.get("shares", 0))
        if pos_shares < 0:
            cover_shares = abs(pos_shares)
            logger.warning(
                "SHORT DETECTED: %s has %d shares — adding urgent COVER for %d shares",
                t, pos_shares, cover_shares,
            )
            ob.add_urgent_exit({
                "ticker": t,
                "signal": "COVER",
                "shares": cover_shares,
                "reason": "short_position_cover",
                "detail": f"Covering accidental short of {cover_shares} shares",
            })

    ob.save()

    # Backup full order book to S3 for audit trail
    if signals_bucket:
        ob.backup_to_s3(signals_bucket, run_date)

    # Write public-safe summary for dashboard
    if signals_bucket:
        _write_order_book_summary(ob, blocked_entries, signals_bucket, run_date)

    n_entries = len(ob.pending_entries())
    n_urgent = len(ob.pending_urgent_exits())
    n_stops = len(ob.active_stops())
    n_covers = sum(1 for u in ob.pending_urgent_exits() if u.get("signal") == "COVER")
    logger.info(
        "Order book written: %d entries, %d urgent exits (%d covers), %d stops",
        n_entries, n_urgent, n_covers, n_stops,
    )
    # Build notification with blocked entry transparency
    blocked_lines = ""
    if blocked_entries:
        blocked_lines = f"\nBlocked ({len(blocked_entries)}):\n"
        for b in blocked_entries:
            blocked_lines += f"  {b['ticker']}: {b.get('block_reason', b.get('reason', 'unknown'))}\n"

    try:
        from executor.notifier import send_daemon_status
        send_daemon_status(
            f"\u2705 *Order book written*\n"
            f"Date: {run_date}\n"
            f"Entries: {n_entries} | Urgent exits: {n_urgent} | Stops: {n_stops}"
            f"{blocked_lines}"
        )
    except Exception:
        logger.debug("Order book Telegram notification failed", exc_info=True)


def run(
    dry_run: bool = False,
    simulate: bool = False,
    ibkr_client=None,           # injected by backtester when simulate=True
    signals_override: dict = None,  # injected signals dict (skips S3 read)
    price_histories: dict = None,   # injected by backtester for exit manager
    config_override: dict = None,   # injected by backtester param sweep
) -> list[dict] | None:
    """
    Returns list of order dicts when simulate=True, else None.
    All other behaviour (risk guard, position sizer, trade logger) is unchanged.
    """
    orders = []
    run_date = str(date.today())
    _health_start = _time.time()
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
            elif key in _PARAM_MAP:
                # Route flat param names through the same mapping as S3 params
                # so backtester sweep keys (e.g. "min_score") land in the right
                # nested config location (e.g. "min_score_to_enter").
                path = _PARAM_MAP[key]
                target = config
                for p in path[:-1]:
                    target = target.setdefault(p, {})
                target[path[-1]] = val
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

    # ── 0. Check upstream health (warn + Telegram alert, never blocks) ───
    if not simulate:
        _health_warnings = []
        try:
            from executor.health_status import check_upstream_health
            upstream = check_upstream_health(
                signals_bucket,
                ["research", "predictor_inference"],
                max_age_hours=48,
            )
            for mod, info in upstream.items():
                if info["status"] == "unknown":
                    _health_warnings.append(f"{mod}: no health data found")
                    logger.warning("Upstream %s: no health data found", mod)
                elif info["status"] == "failed":
                    _health_warnings.append(f"{mod}: last run FAILED")
                    logger.warning("Upstream %s last run FAILED", mod)
                elif info["stale"]:
                    max_hrs = 192 if mod == "research" else 48  # 8 days vs 2 days
                    if info["age_hours"] > max_hrs:
                        msg = f"{mod}: {info['age_hours']:.0f}h ({info['age_hours']/24:.1f}d) stale"
                        _health_warnings.append(msg)
                        logger.warning("Upstream %s", msg)
        except Exception as _ue:
            logger.debug("Upstream health check failed (non-blocking): %s", _ue)

        if _health_warnings:
            try:
                from executor.notifier import send_daemon_status
                send_daemon_status(
                    "\u26a0\ufe0f *Upstream health warning*\n"
                    f"Date: {run_date}\n"
                    + "\n".join(f"- {w}" for w in _health_warnings)
                    + "\n\nExecutor proceeding — check research/predictor."
                )
            except Exception:
                logger.debug("Upstream health warning Telegram notification failed", exc_info=True)

    conn = None if simulate else init_db(db_path)

    # ── 1. Read signals from S3 (or use injected override) ──────────────────
    try:
        signals_raw, signals, run_date, predictions_by_ticker = _read_signals(
            config, signals_bucket, run_date, simulate, signals_override, conn,
        )
    except (RuntimeError, Exception):
        if conn:
            conn.close()
        return
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]

    # ── 2. Connect to IBKR (or use injected simulated client) ───────────────
    if simulate:
        ibkr = ibkr_client
    else:
        ibkr = IBKRClient(
            host=config["ibkr_host"],
            port=config["ibkr_port"],
            client_id=config["ibkr_client_id"],
            reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
        )

    try:
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
        # Include ENTER tickers for ATR sizing, momentum gate, and correlation check
        if price_histories is None:
            enter_tickers = [s["ticker"] for s in signals.get("enter", [])]
            all_tickers = list(set(list(current_positions.keys()) + enter_tickers))
            # Also load sector ETF histories for sector-relative exit veto
            held_sectors = set(pos.get("sector", "") for pos in current_positions.values())
            etf_tickers = [SECTOR_ETF_MAP.get(s, "SPY") for s in held_sectors if s]
            etf_tickers = list(set(etf_tickers))
            all_tickers_with_etfs = list(set(all_tickers + etf_tickers))
            if all_tickers_with_etfs:
                price_histories = load_price_histories(
                    tickers=all_tickers_with_etfs,
                    signals_bucket=signals_bucket,
                )
            else:
                price_histories = {}
    
        # Separate sector ETF histories for exit manager
        sector_etf_histories = {
            t: price_histories[t] for t in SECTOR_ETF_MAP.values()
            if t in (price_histories or {})
        }
        # Also include SPY as fallback
        if "SPY" in (price_histories or {}):
            sector_etf_histories["SPY"] = price_histories["SPY"]
    
        strategy_exits = evaluate_exits(
            current_positions=current_positions,
            signals_by_ticker=signals_by_ticker,
            run_date=run_date,
            price_histories=price_histories or {},
            ibkr_client=ibkr,
            strategy_config=strategy_config,
            sector_etf_histories=sector_etf_histories or None,
        )
    
        if strategy_exits:
            logger.info(
                f"Strategy layer generated {len(strategy_exits)} exit signal(s): "
                + ", ".join(f"{s['ticker']}({s['action']}: {s['reason']})" for s in strategy_exits)
            )
    
        enter_signals = signals["enter"]

        # Initialize order book for the day (daemon reads this after main.py completes).
        # reset_pending() makes this idempotent — if main.py runs twice, the second
        # run replaces the first rather than appending duplicate orders.
        ob = OrderBook.load()
        ob.set_date(run_date)
        ob.reset_pending()

        # ── 2e. Compute signal age for staleness discount ─────────────────────
        # Signal age is fixed for the trading day (main.py runs once on boot).
        # Signals are never refreshed mid-day, so this doesn't need recomputation.
        signals_date_str = signals_raw.get("date", run_date)
        try:
            signals_date = date.fromisoformat(signals_date_str)
            signal_age_days = (date.fromisoformat(run_date) - signals_date).days
        except (ValueError, TypeError):
            signal_age_days = 0

        # ── 2f. Batch-fetch earnings dates for ENTER candidates ──────────────
        earnings_by_ticker: dict[str, int | None] = {}
        if config.get("earnings_sizing_enabled", True) and not simulate:
            for sig in enter_signals:
                t = sig["ticker"]
                try:
                    import yfinance as yf_mod
                    cal = yf_mod.Ticker(t).calendar
                    if cal is not None and not cal.empty:
                        next_date = cal.iloc[0, 0] if hasattr(cal, 'iloc') else None
                        if next_date is not None:
                            from datetime import datetime as dt_mod
                            if hasattr(next_date, 'date'):
                                next_date = next_date.date()
                            elif isinstance(next_date, str):
                                next_date = date.fromisoformat(next_date)
                            days_until = (next_date - date.fromisoformat(run_date)).days
                            if days_until >= 0:
                                earnings_by_ticker[t] = days_until
                except Exception:
                    logger.debug("Failed to load earnings data", exc_info=True)

        # ── 2g. Drawdown forced exits ─────────────────────────────────────────
        if strategy_config.get("drawdown_forced_exit_enabled", True) and dd_multiplier < 1.0:
            forced_exit_count = 0
            if dd_multiplier <= 0.25:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier3_count", 2)
            elif dd_multiplier <= 0.50:
                forced_exit_count = strategy_config.get("drawdown_forced_exit_tier2_count", 1)

            if forced_exit_count > 0 and current_positions:
                existing_exit_tickers = set(
                    s["ticker"] for s in signals.get("exit", [])
                ) | set(
                    s["ticker"] for s in strategy_exits if s["action"] == "EXIT"
                )

                def _conviction_rank(ticker_pos):
                    t, pos = ticker_pos
                    sig_data = signals_by_ticker.get(t, {})
                    score = sig_data.get("score") or 50
                    mv = pos.get("market_value", 0)
                    return (score, mv)

                ranked = sorted(current_positions.items(), key=_conviction_rank)
                for t, pos in ranked[:forced_exit_count]:
                    if t not in existing_exit_tickers:
                        shares_held = int(pos.get("shares", 0))
                        if shares_held > 0:
                            forced_sig = {
                                "ticker": t,
                                "action": "EXIT",
                                "reason": "drawdown_forced_exit",
                                "detail": f"forced exit due to drawdown (dd_mult={dd_multiplier})",
                            }
                            strategy_exits.append(forced_sig)
                            logger.info(
                                f"DRAWDOWN FORCED EXIT: {t} (score={_conviction_rank((t, pos))[0]}, "
                                f"dd_multiplier={dd_multiplier})"
                            )

        # ── 3. Process ENTER signals ─────────────────────────────────────────────
        n_entered, entry_orders, blocked_entries = _plan_entries(
            enter_signals=enter_signals,
            signals_raw=signals_raw,
            predictions_by_ticker=predictions_by_ticker,
            config=config,
            strategy_config=strategy_config,
            market_regime=market_regime,
            sector_ratings=sector_ratings,
            ibkr=ibkr,
            portfolio_nav=portfolio_nav,
            peak_nav=peak_nav,
            current_positions=current_positions,
            price_histories=price_histories,
            dd_multiplier=dd_multiplier,
            signal_age_days=signal_age_days,
            earnings_by_ticker=earnings_by_ticker,
            ob=ob,
            run_date=run_date,
            dry_run=dry_run,
            simulate=simulate,
        )
        orders.extend(entry_orders)

        # Log blocked entries to shadow book for evaluation
        if conn and blocked_entries:
            for be in blocked_entries:
                try:
                    log_shadow_book_block(conn, be)
                except Exception as e:
                    logger.debug("Shadow book log failed for %s: %s", be.get("ticker"), e)

        # ── 4–5. Process EXIT and REDUCE signals ────────────────────────────────
        exit_orders = _plan_exits_and_reduces(
            signals=signals,
            strategy_exits=strategy_exits,
            predictions_by_ticker=predictions_by_ticker,
            current_positions=current_positions,
            ibkr=ibkr,
            portfolio_nav=portfolio_nav,
            config=config,
            market_regime=market_regime,
            ob=ob,
            run_date=run_date,
            dry_run=dry_run,
            simulate=simulate,
        )
        orders.extend(exit_orders)

        # ── 6. Write stop records and save order book for daemon ────────────────
        if not simulate and not dry_run:
            try:
                _write_stops_and_finalize(ibkr, ob, price_histories, strategy_config, conn, run_date, blocked_entries, signals_bucket)
            except Exception as e:
                logger.warning("Failed to write order book: %s", e)

        # ── 7. Backup and disconnect ─────────────────────────────────────────
        if not dry_run and not simulate:
            backup_to_s3(db_path, run_date, trades_bucket)

        # ── 8. Write health status ────────────────────────────────────────────
        if not simulate:
            try:
                from executor.health_status import write_health
                n_exit = len(exit_orders)
                n_blocked = len(enter_signals) - n_entered
                write_health(
                    bucket=signals_bucket,
                    module_name="executor",
                    status="ok",
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    summary={
                        "n_orders": n_entered + n_exit,
                        "n_enter": n_entered,
                        "n_exit": n_exit,
                        "n_blocked": n_blocked,
                    },
                )
            except Exception as _he:
                logger.warning("Health status write failed: %s", _he)

            # ── Data manifest ──────────────────────────────────────────────────
            try:
                from executor.health_status import write_data_manifest
                write_data_manifest(
                    bucket=signals_bucket,
                    module_name="executor_morning",
                    run_date=run_date,
                    manifest={
                        "signals_date": signals_raw.get("date", run_date),
                        "signals_count": len(signals.get("enter", [])) + len(signals.get("exit", [])),
                        "predictions_available": bool(predictions_by_ticker),
                        "entries_planned": n_entered,
                        "entries_blocked": n_blocked,
                        "blocked_reasons": [
                            {"ticker": b.get("ticker"), "reason": b.get("block_reason", b.get("reason"))}
                            for b in blocked_entries[:20]
                        ] if 'blocked_entries' in dir() else [],
                        "exits_planned": n_exit,
                    },
                )
            except Exception as _me:
                logger.warning("Data manifest write failed: %s", _me)

        logger.info(f"Executor complete | dry_run={dry_run} | simulate={simulate}")

        if simulate:
            return orders
    except Exception:
        logger.exception("Executor error — ensuring IBKR disconnect")
        if not simulate:
            try:
                from executor.health_status import write_health
                write_health(
                    bucket=config.get("signals_bucket", "alpha-engine-research") if 'config' in dir() else "alpha-engine-research",
                    module_name="executor",
                    status="failed",
                    run_date=run_date,
                    duration_seconds=_time.time() - _health_start,
                    error=str(sys.exc_info()[1]),
                )
            except Exception:
                logger.debug("Health status write failed on error path", exc_info=True)
        raise
    finally:
        ibkr.disconnect()
        if conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orders without placing them",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run locally with simulated IB client (no IB Gateway needed). "
             "Uses synthetic positions and real signals from S3.",
    )
    args = parser.parse_args()

    if args.simulate:
        from executor.ibkr import SimulatedIBKRClient
        # Seed prices from S3 slim cache (last close for each ticker)
        sim_prices = {}
        try:
            config = load_config()
            bucket = config.get("signals_bucket", "alpha-engine-research")
            from executor.price_cache import load_price_histories
            import json
            import subprocess
            # Read signals to know which tickers to price
            result = subprocess.run(
                ["aws", "s3", "cp", f"s3://{bucket}/signals/{date.today()}/signals.json", "-"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                sig_data = json.loads(result.stdout)
                tickers = [s["ticker"] for s in sig_data.get("universe", [])]
                histories = load_price_histories(tickers=tickers, signals_bucket=bucket)
                for t, hist in histories.items():
                    if hist and len(hist) > 0:
                        sim_prices[t] = hist[-1]["close"]
                logger.info("Seeded %d simulated prices from S3 slim cache", len(sim_prices))
        except Exception as e:
            logger.warning("Could not seed simulated prices: %s — entries will show no price", e)
        sim_client = SimulatedIBKRClient(prices=sim_prices, nav=1_000_000.0)
        logger.info("SIMULATE MODE: using SimulatedIBKRClient (no IB Gateway)")
        orders = run(simulate=True, ibkr_client=sim_client, dry_run=True)
        if orders:
            logger.info("Simulated orders: %d", len(orders))
            for o in orders:
                logger.info(
                    "  %s %s shares=%s",
                    o.get("action", "?"), o.get("ticker", "?"), o.get("shares", "?"),
                )
        else:
            logger.info("No simulated orders generated")
    else:
        run(dry_run=args.dry_run)
