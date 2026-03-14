"""
Exit manager — generates EXIT and REDUCE signals from quantitative rules.

Two independent exit strategies run in parallel:
  1. ATR trailing stop: exit if price falls below highest_high - ATR * multiplier
  2. Time-based decay: reduce after N days, exit after M days without thesis refresh

These are additive to Research signals — if Research says HOLD but exit_manager
says EXIT, the exit fires. Research EXIT signals always take precedence.

All logic uses data available at trade time (OHLCV from yfinance or IBKR).
No LLM calls required — fully backtestable.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def check_atr_trailing_stop(
    ticker: str,
    current_price: float,
    entry_date: str,
    price_history: list[dict],
    strategy_config: dict,
) -> dict | None:
    """
    Check if a position should be exited based on ATR trailing stop.

    The trailing stop is: highest_high_since_entry - ATR(period) * multiplier.
    If current_price <= stop_level, return an EXIT signal.

    Args:
        ticker: stock symbol
        current_price: current market price
        entry_date: ISO date string (YYYY-MM-DD) when position was entered
        price_history: list of dicts with keys {date, open, high, low, close},
                       sorted ascending by date. Must cover at least ATR period
                       days before entry_date through today.
        strategy_config: from load_strategy_config()

    Returns:
        EXIT signal dict if stop triggered, else None.
    """
    if not strategy_config.get("atr_trailing_enabled", True):
        return None

    if not price_history or current_price is None:
        return None

    period = strategy_config.get("atr_period", 14)
    multiplier = strategy_config.get("atr_multiplier", 3.0)

    # Filter to bars on or after entry_date
    entry_dt = date.fromisoformat(entry_date)
    post_entry = [b for b in price_history if date.fromisoformat(b["date"]) >= entry_dt]

    if len(post_entry) < 2:
        # Not enough data since entry to compute trailing stop
        return None

    # Compute ATR from the full price history (need period+1 bars minimum)
    atr = _compute_atr(price_history, period)
    if atr is None or atr <= 0:
        return None

    # Highest high since entry
    highest_high = max(b["high"] for b in post_entry)

    stop_level = highest_high - (atr * multiplier)

    if current_price <= stop_level:
        logger.info(
            f"ATR TRAILING STOP triggered for {ticker}: "
            f"price=${current_price:.2f} <= stop=${stop_level:.2f} "
            f"(high=${highest_high:.2f} - ATR={atr:.2f} x {multiplier})"
        )
        return {
            "ticker": ticker,
            "action": "EXIT",
            "reason": "atr_trailing_stop",
            "detail": (
                f"price=${current_price:.2f} <= stop=${stop_level:.2f} "
                f"(highest_high=${highest_high:.2f} - ATR({period})={atr:.2f} x {multiplier})"
            ),
            "stop_level": round(stop_level, 2),
            "atr": round(atr, 2),
            "highest_high": round(highest_high, 2),
        }

    return None


def check_time_decay(
    ticker: str,
    entry_date: str,
    run_date: str,
    signal_action: str,
    strategy_config: dict,
) -> dict | None:
    """
    Check if a position should be reduced or exited based on holding period.

    Only fires if the Research signal is HOLD (not actively recommending
    the position). If Research says ENTER (reaffirming), time decay resets.

    Args:
        ticker: stock symbol
        entry_date: ISO date string when position was entered
        run_date: today's date (ISO string)
        signal_action: Research signal for this ticker today ("ENTER"|"HOLD"|"EXIT"|"REDUCE")
        strategy_config: from load_strategy_config()

    Returns:
        REDUCE or EXIT signal dict if time limit hit, else None.
    """
    if not strategy_config.get("time_decay_enabled", True):
        return None

    # If Research is actively reaffirming (ENTER) or already exiting, skip time decay
    if signal_action in ("ENTER", "EXIT", "REDUCE"):
        return None

    reduce_days = strategy_config.get("time_decay_reduce_days", 5)
    exit_days = strategy_config.get("time_decay_exit_days", 10)

    entry_dt = date.fromisoformat(entry_date)
    run_dt = date.fromisoformat(run_date)
    calendar_days = (run_dt - entry_dt).days

    # Approximate trading days (exclude weekends): ~5 trading days per 7 calendar days
    trading_days = _approx_trading_days(entry_dt, run_dt)

    if trading_days >= exit_days:
        logger.info(
            f"TIME DECAY EXIT for {ticker}: held ~{trading_days} trading days "
            f"(>= {exit_days} day exit threshold)"
        )
        return {
            "ticker": ticker,
            "action": "EXIT",
            "reason": "time_decay_exit",
            "detail": f"held ~{trading_days} trading days (exit threshold: {exit_days})",
            "trading_days_held": trading_days,
        }

    if trading_days >= reduce_days:
        logger.info(
            f"TIME DECAY REDUCE for {ticker}: held ~{trading_days} trading days "
            f"(>= {reduce_days} day reduce threshold)"
        )
        return {
            "ticker": ticker,
            "action": "REDUCE",
            "reason": "time_decay_reduce",
            "detail": f"held ~{trading_days} trading days (reduce threshold: {reduce_days})",
            "trading_days_held": trading_days,
        }

    return None


def evaluate_exits(
    current_positions: dict[str, dict],
    signals_by_ticker: dict[str, dict],
    run_date: str,
    price_histories: dict[str, list[dict]],
    ibkr_client,
    strategy_config: dict,
) -> list[dict]:
    """
    Evaluate all held positions against exit rules.

    Returns a list of strategy-generated EXIT/REDUCE signals. These are
    merged with Research signals in main.py — strategy exits supplement
    Research exits (they don't conflict).

    Args:
        current_positions: {ticker: {shares, market_value, avg_cost, sector, entry_date}}
        signals_by_ticker: {ticker: signal_dict} from Research
        run_date: today's date
        price_histories: {ticker: [{date, open, high, low, close}, ...]}
        ibkr_client: for fetching current prices
        strategy_config: from load_strategy_config()

    Returns:
        List of signal dicts with action="EXIT" or "REDUCE" and reason field.
    """
    strategy_signals = []

    for ticker, pos in current_positions.items():
        entry_date = pos.get("entry_date")
        if not entry_date:
            continue

        research_signal = signals_by_ticker.get(ticker, {})
        research_action = research_signal.get("signal", "HOLD")

        # Skip if Research is already exiting this position
        if research_action in ("EXIT", "REDUCE"):
            continue

        current_price = ibkr_client.get_current_price(ticker)
        if current_price is None:
            continue

        # 1. ATR trailing stop
        history = price_histories.get(ticker, [])
        atr_signal = check_atr_trailing_stop(
            ticker=ticker,
            current_price=current_price,
            entry_date=entry_date,
            price_history=history,
            strategy_config=strategy_config,
        )
        if atr_signal:
            strategy_signals.append(atr_signal)
            continue  # ATR exit takes priority over time decay

        # 2. Time-based decay
        time_signal = check_time_decay(
            ticker=ticker,
            entry_date=entry_date,
            run_date=run_date,
            signal_action=research_action,
            strategy_config=strategy_config,
        )
        if time_signal:
            strategy_signals.append(time_signal)

    return strategy_signals


# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_atr(price_history: list[dict], period: int = 14) -> float | None:
    """
    Compute Average True Range over the last `period` bars.

    Uses Wilder's smoothing (EWM with alpha=1/period).
    Returns None if insufficient data.
    """
    if len(price_history) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(price_history)):
        bar = price_history[i]
        prev_close = price_history[i - 1]["close"]
        high = bar["high"]
        low = bar["low"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Wilder's smoothed ATR: start with SMA, then EWM
    atr = sum(true_ranges[:period]) / period
    alpha = 1.0 / period
    for tr in true_ranges[period:]:
        atr = atr * (1 - alpha) + tr * alpha

    return atr


def _approx_trading_days(start: date, end: date) -> int:
    """
    Approximate trading days between two dates (excludes weekends).
    Does not account for market holidays — close enough for decay logic.
    """
    if end <= start:
        return 0
    total = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 through Friday=4
            total += 1
    return total
