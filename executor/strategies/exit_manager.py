"""
Exit manager — generates EXIT and REDUCE signals from quantitative rules.

Five independent exit strategies run in parallel:
  1. ATR trailing stop: exit if price falls below highest_high - ATR * multiplier
  2. Profit-taking: reduce when unrealized gain exceeds threshold
  3. Momentum exit: exit on severe negative momentum + oversold RSI
  4. Time-based decay: reduce after N days, exit after M days without thesis refresh
  5. Sector-relative veto: cancel ATR exit if stock is outperforming its sector

These are additive to Research signals — if Research says HOLD but exit_manager
says EXIT, the exit fires. Research EXIT signals always take precedence.

All logic uses data available at trade time (OHLCV from yfinance or IBKR).
No LLM calls required — fully backtestable.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Communication Services": "XLC",
}


# Stance-conditional exit rule overrides (stance taxonomy arc 2026-05-11).
# Maps stance label → keys to override on strategy_config when evaluating
# exits for a position with that stance. None values in the override dict
# disable the corresponding rule outright. Falls through to baseline
# config when stance is None / unknown (legacy entries, ML drift, etc.).
#
# Defaults below are the cold-start values; backtester adds them to the
# parameter search space in a follow-up PR.
STANCE_EXIT_OVERRIDES: dict[str, dict] = {
    "momentum": {
        # Baseline — no overrides. Pinned explicitly so future readers see
        # the contrast with the other stances; the dict is otherwise
        # empty for performance + clarity.
    },
    "value": {
        # Contrarian thesis needs room to play out. Wider stop +
        # extended hold:
        "atr_multiplier": 4.5,           # vs default 3.0 — give bounce more room
        "time_decay_reduce_days": 20,    # vs default 5
        "time_decay_exit_days": 30,      # vs default 10 — full 30d window for mean reversion
    },
    "quality": {
        # Defensive: long hold, time decay disabled. ATR untouched
        # (rare exit on the trailing stop is still appropriate).
        "time_decay_enabled": False,
    },
    "catalyst": {
        # Event-driven: standard checks PLUS the hard catalyst-date
        # exit (see check_catalyst_hard_exit). Time decay disabled
        # because the catalyst_date deadline is the canonical exit
        # boundary — having time_decay fire earlier would defeat the
        # stance's "ride to the event" thesis.
        "time_decay_enabled": False,
    },
}


def _resolve_strategy_config_for_stance(
    base_config: dict, stance: str | None,
) -> dict:
    """Return a stance-overridden view of strategy_config.

    Returns the base config unmodified when stance is None or unknown
    (legacy / ML drift / break-glass). When stance is recognized,
    returns a shallow-copy with the stance's overrides merged in. The
    base config is never mutated — callers can re-use it for other
    positions with different stances.

    Idempotent + pure — no I/O. Stance-conditional behavior should
    flow through this helper rather than per-check inline branches so
    the override surface stays auditable.
    """
    if stance is None or stance not in STANCE_EXIT_OVERRIDES:
        return base_config
    overrides = STANCE_EXIT_OVERRIDES[stance]
    if not overrides:
        return base_config
    merged = dict(base_config)
    merged.update(overrides)
    return merged


def check_atr_trailing_stop(
    ticker: str,
    current_price: float,
    entry_date: str,
    price_history: pd.DataFrame,
    strategy_config: dict,
    *,
    feature_lookup=None,
    run_date: str | None = None,
) -> dict | None:
    """
    Check if a position should be exited based on ATR trailing stop.

    The trailing stop is: highest_high_since_entry - ATR(period) * multiplier.
    If current_price <= stop_level, return an EXIT signal.

    Args:
        ticker: stock symbol
        current_price: current market price
        entry_date: ISO date string (YYYY-MM-DD) when position was entered
        price_history: pd.DataFrame with columns [open, high, low, close],
                       indexed by DatetimeIndex sorted ascending. Must cover
                       at least ATR period days before entry_date through today.
        strategy_config: from load_strategy_config()
        feature_lookup: optional ``executor.feature_lookup.FeatureLookup`` —
                       when provided AND ``period == DEFAULT_ATR_PERIOD``
                       (14), the ATR is read from the precomputed lookup
                       in O(log N) instead of recomputed per call. Tier 3
                       Part C (2026-04-27).
        run_date: ISO date string for the FeatureLookup query (only used
                  when ``feature_lookup`` is provided). Defaults to the
                  last bar in ``price_history`` if not supplied.

    Returns:
        EXIT signal dict if stop triggered, else None.
    """
    if not strategy_config.get("atr_trailing_enabled", True):
        return None

    if price_history is None or len(price_history) == 0 or current_price is None:
        logger.info("ATR skip %s: no price history or current_price", ticker)
        return None

    period = strategy_config.get("atr_period", 14)
    multiplier = strategy_config.get("atr_multiplier", 3.0)

    # Filter to bars on or after entry_date.
    # ``price_history.index`` is a normalized DatetimeIndex; the bars
    # covered are inclusive of entry_date itself (executor enters at
    # close, so the entry-day bar belongs to the held position).
    entry_ts = pd.Timestamp(entry_date)
    post_entry = price_history.loc[entry_ts:]

    if len(post_entry) < 2:
        logger.info("ATR skip %s: %d bars since entry %s (need 2+)", ticker, len(post_entry), entry_date)
        return None

    # Tier 3 Part C: prefer precomputed ATR from FeatureLookup when
    # available + period matches the lookup's hardcoded 14. Falls back
    # to per-call _compute_atr otherwise (live executor without
    # feature_lookup wired, or any non-default period).
    atr: float | None = None
    if feature_lookup is not None:
        from executor.feature_lookup import DEFAULT_ATR_PERIOD
        if period == DEFAULT_ATR_PERIOD:
            query_date = run_date or price_history.index[-1]
            atr = feature_lookup.atr_dollar_at(ticker, query_date)
    if atr is None:
        atr = _compute_atr(price_history, period)
    if atr is None or atr <= 0:
        logger.info("ATR skip %s: insufficient data for ATR(%d) (have %d bars)",
                     ticker, period, len(price_history))
        return None

    # Highest high since entry
    highest_high = float(post_entry["high"].max())

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


def check_fallback_stop(
    ticker: str,
    current_price: float,
    entry_price: float | None,
    strategy_config: dict,
) -> dict | None:
    """
    Fallback fixed-percentage stop when ATR has insufficient data.
    Simple safety net: exit if price falls below entry * (1 - fallback_stop_pct).
    """
    if not strategy_config.get("fallback_stop_enabled", True):
        return None
    if not entry_price or entry_price <= 0 or not current_price:
        return None

    stop_pct = strategy_config.get("fallback_stop_pct", 0.10)
    stop_level = entry_price * (1 - stop_pct)

    if current_price <= stop_level:
        loss_pct = (current_price / entry_price - 1) * 100
        logger.info(
            "FALLBACK STOP %s: price=$%.2f <= stop=$%.2f (entry=$%.2f - %.0f%%, loss=%.1f%%)",
            ticker, current_price, stop_level, entry_price, stop_pct * 100, loss_pct,
        )
        return {
            "ticker": ticker,
            "action": "EXIT",
            "reason": "fallback_stop",
            "detail": (f"price=${current_price:.2f} <= fallback stop=${stop_level:.2f} "
                       f"(entry=${entry_price:.2f} - {stop_pct:.0%})"),
            "stop_level": round(stop_level, 2),
            "entry_price": round(entry_price, 2),
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


def check_catalyst_hard_exit(
    ticker: str,
    catalyst_date: str | None,
    run_date: str,
    strategy_config: dict,
) -> dict | None:
    """Hard exit catalyst-stance positions at ``catalyst_date + N`` trading days.

    Stance taxonomy arc — catalyst stance's exit boundary. Event-driven
    theses depend on the catalyst date arriving; once the event passes
    (plus a short follow-through window for the post-event move to
    settle), the thesis is mechanically expired regardless of price
    action. Standard ATR / profit-take / momentum checks still run
    before this — those can exit earlier on adverse moves; this is the
    terminal exit-by-deadline.

    Args:
        ticker: stock symbol
        catalyst_date: ISO date (YYYY-MM-DD) of the expected event,
                       set at entry from predictions.json catalyst_date.
                       None → not a catalyst-stance position, no-op.
        run_date: today's date (ISO string)
        strategy_config: must include ``catalyst_followthrough_days``
                         (default 3) — trading days after catalyst_date
                         before the hard exit fires.

    Returns:
        EXIT signal dict if today > catalyst_date + N trading days,
        else None.
    """
    if not catalyst_date:
        return None
    try:
        cat_dt = date.fromisoformat(catalyst_date)
        run_dt = date.fromisoformat(run_date)
    except ValueError:
        logger.warning(
            "check_catalyst_hard_exit %s: invalid date "
            "(catalyst_date=%r, run_date=%r); skipping",
            ticker, catalyst_date, run_date,
        )
        return None

    followthrough_days = strategy_config.get("catalyst_followthrough_days", 3)
    trading_days_past = _approx_trading_days(cat_dt, run_dt)
    if trading_days_past < followthrough_days:
        return None

    logger.info(
        f"CATALYST HARD EXIT for {ticker}: catalyst_date={catalyst_date}, "
        f"~{trading_days_past} trading days past (>= {followthrough_days} "
        f"follow-through threshold)"
    )
    return {
        "ticker": ticker,
        "action": "EXIT",
        "reason": "catalyst_hard_exit",
        "detail": (
            f"catalyst_date={catalyst_date}, ~{trading_days_past} trading "
            f"days past (follow-through threshold: {followthrough_days})"
        ),
        "catalyst_date": catalyst_date,
        "trading_days_past_catalyst": trading_days_past,
    }


def check_profit_take(
    ticker: str,
    current_price: float,
    avg_cost: float | None,
    strategy_config: dict,
) -> dict | None:
    """
    Check if a position should be partially sold to lock in gains.

    If unrealized gain exceeds the configured threshold, return a REDUCE signal.

    Args:
        ticker: stock symbol
        current_price: current market price
        avg_cost: average cost basis per share
        strategy_config: from load_strategy_config()

    Returns:
        REDUCE signal dict if profit threshold exceeded, else None.
    """
    if not strategy_config.get("profit_take_enabled", True):
        return None

    if avg_cost is None or avg_cost <= 0:
        return None

    unrealized_gain = (current_price - avg_cost) / avg_cost
    threshold = strategy_config.get("profit_take_pct", 0.25)

    if unrealized_gain >= threshold:
        logger.info(
            f"PROFIT TAKE triggered for {ticker}: "
            f"gain={unrealized_gain:.2%} >= threshold={threshold:.2%}"
        )
        return {
            "ticker": ticker,
            "action": "REDUCE",
            "reason": "profit_take",
            "detail": (
                f"unrealized gain {unrealized_gain:.2%} >= "
                f"threshold {threshold:.2%}"
            ),
            "unrealized_gain": round(unrealized_gain, 4),
        }

    return None


def check_sector_relative_veto(
    ticker: str,
    sector: str,
    price_history: pd.DataFrame,
    sector_etf_history: pd.DataFrame,
    strategy_config: dict,
) -> bool:
    """
    Veto an exit if the stock is outperforming its sector ETF.

    If the stock's recent return exceeds the sector ETF return by more than
    the configured threshold, the exit should be vetoed (stock still has
    relative momentum).

    Args:
        ticker: stock symbol
        sector: sector name (used for logging only)
        price_history: stock OHLCV DataFrame sorted ascending by date
        sector_etf_history: sector ETF OHLCV DataFrame sorted ascending
        strategy_config: from load_strategy_config()

    Returns:
        True if exit should be vetoed, False otherwise.
    """
    if not strategy_config.get("sector_relative_veto_enabled", True):
        return False

    if price_history is None or len(price_history) < 5:
        return False

    if sector_etf_history is None or len(sector_etf_history) < 5:
        return False

    lookback = min(20, len(price_history), len(sector_etf_history))

    stock_close = price_history["close"]
    sector_close = sector_etf_history["close"]
    stock_return = float(stock_close.iloc[-1] / stock_close.iloc[-lookback]) - 1.0
    sector_return = float(sector_close.iloc[-1] / sector_close.iloc[-lookback]) - 1.0

    outperformance = stock_return - sector_return
    threshold = strategy_config.get("sector_relative_outperform_threshold", 0.05)

    if outperformance > threshold:
        logger.warning(
            f"SECTOR VETO for {ticker}: outperforming {sector} by "
            f"{outperformance:.2%} (threshold={threshold:.2%}) — exit vetoed"
        )
        return True

    return False


def check_momentum_exit(
    ticker: str,
    price_history: pd.DataFrame,
    strategy_config: dict,
    *,
    feature_lookup=None,
    run_date: str | None = None,
) -> dict | None:
    """
    Check if a position should be exited based on severe negative momentum.

    Triggers when both 20-day momentum is deeply negative AND RSI is oversold,
    indicating a sustained downtrend with no reversal signal.

    Args:
        ticker: stock symbol
        price_history: OHLCV DataFrame sorted ascending (needs >= 21 bars)
        strategy_config: from load_strategy_config()
        feature_lookup: optional FeatureLookup for O(log N) momentum + RSI
                       reads (Tier 3 Part C).
        run_date: ISO date for FeatureLookup queries.

    Returns:
        EXIT signal dict if momentum criteria met, else None.
    """
    if not strategy_config.get("momentum_exit_enabled", True):
        return None

    if price_history is None or len(price_history) < 21:
        return None

    # 20-day momentum + RSI(14): prefer precomputed lookups when
    # available, fall back to per-call computation.
    momentum: float | None = None
    rsi: float | None = None
    if feature_lookup is not None:
        query_date = run_date or price_history.index[-1]
        momentum = feature_lookup.momentum_20d_pct_at(ticker, query_date)
        rsi = feature_lookup.rsi_at(ticker, query_date)

    if momentum is None:
        close = price_history["close"]
        momentum = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100
    if rsi is None:
        rsi = _compute_rsi(price_history, period=14)

    mom_threshold = strategy_config.get("momentum_exit_threshold", -15.0)
    rsi_threshold = strategy_config.get("momentum_exit_rsi", 30)

    if momentum < mom_threshold and rsi is not None and rsi < rsi_threshold:
        logger.info(
            f"MOMENTUM EXIT triggered for {ticker}: "
            f"20d momentum={momentum:.1f}% (< {mom_threshold}%), "
            f"RSI={rsi:.1f} (< {rsi_threshold})"
        )
        return {
            "ticker": ticker,
            "action": "EXIT",
            "reason": "momentum_exit",
            "detail": (
                f"20d momentum={momentum:.1f}% (threshold={mom_threshold}%), "
                f"RSI(14)={rsi:.1f} (threshold={rsi_threshold})"
            ),
        }

    return None


def check_position_loss_floor(
    ticker: str,
    current_price: float | None,
    avg_cost: float | None,
    strategy_config: dict,
) -> dict | None:
    """Hard-risk maximum-adverse-excursion (MAE) floor — full EXIT.

    Fires when a held position's loss from average cost breaches
    ``position_loss_floor_pct`` (a negative decimal, e.g. -0.15). This is
    the cumulative-drawdown sibling of the intraday ``catastrophic_gap_stop``
    (single-move): it cuts a falling-knife position whose thesis has been
    wrong by too much, regardless of stance / catalyst / Research signal.

    STANCE-AGNOSTIC AND HARD: ``position_loss_floor_pct`` must NEVER be added
    to ``STANCE_EXIT_OVERRIDES`` — the floor is a risk backstop the value
    stance's loosened ATR (4.5x) / time-decay (30d) must not be able to
    suppress (the gap that let COIN bleed -19% un-cut while ranked #1,
    L4549a). Backtester-tuned within a protective band thereafter; the band's
    loose end is the real safeguard.
    """
    if not strategy_config.get("position_loss_floor_enabled", True):
        return None
    # Inline default mirrors strategies.config.POSITION_LOSS_FLOOR_PCT so the
    # floor is active even on a partial config (active on merge with no
    # risk.yaml edit). An explicit ``None`` still disables it.
    floor = strategy_config.get("position_loss_floor_pct", -0.15)
    if (
        floor is None
        or avg_cost is None
        or current_price is None
        or float(avg_cost) <= 0
    ):
        return None
    loss = float(current_price) / float(avg_cost) - 1.0
    if loss <= float(floor):
        logger.info(
            f"POSITION LOSS FLOOR EXIT {ticker}: loss={loss * 100:.1f}% "
            f"<= floor={float(floor) * 100:.1f}% "
            f"(avg_cost={float(avg_cost):.2f}, px={float(current_price):.2f})"
        )
        return {
            "ticker": ticker,
            "action": "EXIT",
            "reason": "position_loss_floor",
            "detail": (
                f"MAE floor breached: {loss * 100:.1f}% from avg cost "
                f"(floor {float(floor) * 100:.1f}%)"
            ),
        }
    return None


def _evaluate_single_position(
    *,
    ticker: str,
    pos: dict,
    research_action: str,
    current_price: float,
    history,
    sector_etf_histories: dict | None,
    stance_config: dict,
    catalyst_date,
    entry_date,
    run_date: str,
    feature_lookup,
) -> tuple[dict | None, str | None]:
    """Run all exit-rule checks for a single held position.

    Returns ``(signal, fired_rule_key)`` where ``fired_rule_key`` is one of:
    ``catalyst_hard_exit`` / ``atr_trailing_stop`` / ``sector_veto_blocked`` /
    ``fallback_stop`` / ``profit_take`` / ``momentum_exit`` / ``time_decay``,
    or ``(None, None)`` if no rule fired.

    Extracted from the loop body of ``evaluate_exits`` to enable per-
    position capture wiring (L2308 PR 4b) without losing the existing
    mid-iteration short-circuit semantics. Behavior is identical to the
    pre-refactor loop body — caller verifies via the existing
    ``test_exit_manager.py`` suite.

    Note: ``sector_veto_blocked`` is reported when ATR fired but the
    sector-relative veto suppressed it (load-bearing for grading the
    sector-veto decision separately from the ATR-fire decision).
    """
    # 0. Position loss floor (MAE) — hard-risk, stance-agnostic, runs
    # FIRST. A position down past the floor is cut regardless of stance,
    # catalyst, or price-based checks (the falling-knife backstop, L4549a).
    # ``position_loss_floor_pct`` is intentionally NOT in STANCE_EXIT_OVERRIDES,
    # so ``stance_config`` carries its base value here — the value stance's
    # loosened ATR/time-decay cannot subordinate this floor.
    loss_floor_exit = check_position_loss_floor(
        ticker=ticker,
        current_price=current_price,
        avg_cost=pos.get("avg_cost"),
        strategy_config=stance_config,
    )
    if loss_floor_exit:
        return loss_floor_exit, "position_loss_floor"

    # 0b. Catalyst hard exit — runs next so the deadline supersedes
    # price-based checks (the thesis is mechanically expired).
    catalyst_exit = check_catalyst_hard_exit(
        ticker=ticker,
        catalyst_date=catalyst_date,
        run_date=run_date,
        strategy_config=stance_config,
    )
    if catalyst_exit:
        return catalyst_exit, "catalyst_hard_exit"

    # 1. ATR trailing stop (with sector-relative veto)
    atr_signal = check_atr_trailing_stop(
        ticker=ticker,
        current_price=current_price,
        entry_date=entry_date,
        price_history=history,
        strategy_config=stance_config,
        feature_lookup=feature_lookup,
        run_date=run_date,
    )
    if atr_signal:
        sector = pos.get("sector", "")
        etf_ticker = SECTOR_ETF_MAP.get(sector, "SPY")
        etf_history = (
            sector_etf_histories.get(etf_ticker)
            if sector_etf_histories
            else None
        )
        if check_sector_relative_veto(
            ticker, sector, history, etf_history, stance_config
        ):
            logger.info(
                f"ATR exit for {ticker} vetoed — outperforming sector ({sector})"
            )
            # Sector veto blocked the ATR exit; fall through to other
            # checks. Mark this position-iteration so capture can record
            # the suppressed-fire event separately from the ultimate
            # outcome (no-fire / time-decay / etc.).
            atr_signal = None  # consumed by veto
            atr_fired_then_vetoed = True
        else:
            return atr_signal, "atr_trailing_stop"
    else:
        atr_fired_then_vetoed = False

    if atr_signal is None and stance_config.get("fallback_stop_enabled", True):
        # ATR returned None (or was vetoed by sector check) — try fallback
        # fixed-percentage stop. Note: post-sector-veto the fallback stop
        # is still considered (matches pre-refactor behavior where the
        # `elif stance_config.get(...)` only ran when ATR returned None
        # AT THE check_atr_trailing_stop call site, not after sector veto.
        # To preserve identical behavior, guard fallback on "ATR returned
        # None at the call site" — track via the original `atr_signal`
        # before the veto branch consumed it).
        if not atr_fired_then_vetoed:
            fallback_signal = check_fallback_stop(
                ticker=ticker,
                current_price=current_price,
                entry_price=pos.get("avg_cost"),
                strategy_config=stance_config,
            )
            if fallback_signal:
                return fallback_signal, "fallback_stop"

    # 2. Profit-taking
    avg_cost = pos.get("avg_cost")
    profit_signal = check_profit_take(
        ticker=ticker,
        current_price=current_price,
        avg_cost=avg_cost,
        strategy_config=stance_config,
    )
    if profit_signal:
        return profit_signal, "profit_take"

    # 3. Momentum exit
    momentum_signal = check_momentum_exit(
        ticker=ticker,
        price_history=history,
        strategy_config=stance_config,
        feature_lookup=feature_lookup,
        run_date=run_date,
    )
    if momentum_signal:
        return momentum_signal, "momentum_exit"

    # 4. Time-based decay
    time_signal = check_time_decay(
        ticker=ticker,
        entry_date=entry_date,
        run_date=run_date,
        signal_action=research_action,
        strategy_config=stance_config,
    )
    if time_signal:
        return time_signal, "time_decay"

    # Sector-vetoed-ATR positions are recorded for the artifact's
    # fired-then-vetoed signal even though the outcome is no_fire.
    if atr_fired_then_vetoed:
        return None, "sector_veto_blocked"

    return None, None


def evaluate_exits(
    current_positions: dict[str, dict],
    signals_by_ticker: dict[str, dict],
    run_date: str,
    price_histories: dict[str, pd.DataFrame],
    ibkr_client,
    strategy_config: dict,
    sector_etf_histories: dict[str, pd.DataFrame] | None = None,
    *,
    feature_lookup=None,
) -> list[dict]:
    """
    Evaluate all held positions against exit rules.

    Returns a list of strategy-generated EXIT/REDUCE signals. These are
    merged with Research signals in main.py — strategy exits supplement
    Research exits (they don't conflict).

    Check order:
      1. ATR trailing stop (with sector-relative veto)
      2. Profit-taking
      3. Momentum exit
      4. Time-based decay

    Args:
        current_positions: {ticker: {shares, market_value, avg_cost, sector, entry_date}}
        signals_by_ticker: {ticker: signal_dict} from Research
        run_date: today's date
        price_histories: {ticker: pd.DataFrame[open, high, low, close]}
                         indexed by DatetimeIndex sorted ascending
        ibkr_client: for fetching current prices
        strategy_config: from load_strategy_config()
        sector_etf_histories: {etf_ticker: pd.DataFrame} for sector-relative
                              veto. None disables veto.
        feature_lookup: optional ``executor.feature_lookup.FeatureLookup``
                        with precomputed ATR / RSI / momentum series.
                        Threaded into ``check_atr_trailing_stop`` and
                        ``check_momentum_exit`` for O(log N) feature
                        lookups instead of per-call recompute. Tier 3
                        Part C (2026-04-27).

    Returns:
        List of signal dicts with action="EXIT" or "REDUCE" and reason field.

    L2308 PR 4b: per-position decision-capture artifacts are emitted via
    ``executor.decision_capture.capture_planner_exit`` for every position
    that reaches the rule-evaluation phase (regardless of whether a rule
    fired). Skipped positions (missing entry_date, research-already-
    exiting, no price) are NOT captured — they have no exit-decision
    semantics to grade.
    """
    # Local import to avoid circular: decision_capture imports the lib,
    # exit_manager is imported deep in the planner stack — keep the
    # capture dependency at call time only.
    from executor.decision_capture import (
        DecisionCaptureWriteError,
        capture_planner_exit,
        is_decision_capture_enabled,
    )

    capture_enabled = is_decision_capture_enabled()

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

        history = price_histories.get(ticker)

        # Resolve stance-conditional config view.
        stance = pos.get("stance")
        catalyst_date = pos.get("catalyst_date")
        stance_config = _resolve_strategy_config_for_stance(
            strategy_config, stance,
        )

        signal, fired_rule_key = _evaluate_single_position(
            ticker=ticker,
            pos=pos,
            research_action=research_action,
            current_price=current_price,
            history=history,
            sector_etf_histories=sector_etf_histories,
            stance_config=stance_config,
            catalyst_date=catalyst_date,
            entry_date=entry_date,
            run_date=run_date,
            feature_lookup=feature_lookup,
        )

        if signal is not None:
            strategy_signals.append(signal)

        # Emit executor:exit_rules planner-side DecisionArtifact (L2308
        # PR 4b). Captures BOTH fired and no-fire positions to give
        # grading the counterfactual coverage (no-fire decisions that
        # later produced drawdowns are gradable as missed exits).
        # Best-effort: capture failure must never kill planning flow.
        if capture_enabled:
            try:
                capture_planner_exit(
                    run_date=run_date,
                    ticker=ticker,
                    pos=pos,
                    research_signal=research_signal,
                    current_price=current_price,
                    stance=stance,
                    catalyst_date=catalyst_date,
                    stance_config=stance_config,
                    signal=signal,
                    fired_rule_key=fired_rule_key,
                )
            except DecisionCaptureWriteError as _cap_exc:
                logger.warning(
                    "decision_capture S3 write failed for PLANNER_EXIT %s "
                    "— continuing planning (capture is observability, not "
                    "load-bearing): %s",
                    ticker, _cap_exc,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "decision_capture raised unexpected exception for "
                    "PLANNER_EXIT %s — continuing planning",
                    ticker,
                )

    return strategy_signals


# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_atr(price_history: pd.DataFrame, period: int = 14) -> float | None:
    """
    Compute Average True Range over the last `period` bars.

    Uses Wilder's smoothing (EWM with alpha=1/period). Vectorized via
    numpy + pandas — output matches the prior scalar Python loop to
    within float precision (Wilder smoothing is well-conditioned;
    seed-bar contribution decays as (1-1/period)^N → ~0 within 5×period
    bars).

    Returns None if insufficient data.
    """
    if price_history is None or len(price_history) < period + 1:
        return None

    high = price_history["high"].to_numpy(dtype=float)
    low = price_history["low"].to_numpy(dtype=float)
    close = price_history["close"].to_numpy(dtype=float)
    prev_close = close[:-1]

    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])

    if len(tr) < period:
        return None

    # Wilder's smoothed ATR: SMA seed over first `period` bars, then EWM.
    # ``ewm(alpha=1/period, adjust=False)`` matches the recurrence
    # ``atr_i = atr_{i-1} * (1 - alpha) + tr_i * alpha`` exactly. Seeding
    # the first ATR value with the SMA of the first `period` true ranges
    # mirrors the pre-vectorized implementation.
    sma_seed = float(np.mean(tr[:period]))
    if len(tr) == period:
        return sma_seed

    smoothed = pd.Series(tr[period:]).ewm(alpha=1.0 / period, adjust=False).mean()
    # Re-seed the EWM with sma_seed by treating the first smoothed value
    # as ``sma_seed * (1 - alpha) + tr[period] * alpha``. pandas.ewm seeds
    # with the first sample, so we manually walk the first step here.
    alpha = 1.0 / period
    first_step = sma_seed * (1.0 - alpha) + float(tr[period]) * alpha
    if len(tr) == period + 1:
        return first_step
    # Subsequent steps: pandas.ewm on tr[period+1:] seeded with first_step.
    # Equivalent to a manual loop, just C-vectorized.
    rest = pd.Series(np.concatenate(([first_step], tr[period + 1:].astype(float))))
    return float(rest.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def _compute_rsi(price_history: pd.DataFrame, period: int = 14) -> float | None:
    """
    Compute Relative Strength Index over the last `period` bars.

    Uses Wilder's smoothing (same as ATR) for average gain/loss.
    Returns None if insufficient data.

    Vectorized via numpy + pandas — output matches the prior scalar
    Python loop to within float precision.
    """
    if price_history is None or len(price_history) < period + 1:
        return None

    close = price_history["close"].to_numpy(dtype=float)
    changes = np.diff(close)
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)

    if len(gains) < period:
        return None

    # Wilder's smoothing: SMA seed over first `period`, then recurrence
    # ``avg_i = avg_{i-1} * (1 - alpha) + x_i * alpha`` with alpha=1/period.
    alpha = 1.0 / period
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if len(gains) > period:
        # Walk the recurrence over remaining bars via pandas.ewm seeded
        # at the SMA. The seed's contribution decays as (1-alpha)^N — for
        # period=14 and ≥70 bars in price_history, the seed is < 1% of
        # the final value, so this matches the prior loop to ~14 sig figs.
        gain_step = avg_gain * (1.0 - alpha) + float(gains[period]) * alpha
        loss_step = avg_loss * (1.0 - alpha) + float(losses[period]) * alpha
        if len(gains) > period + 1:
            gain_rest = pd.Series(np.concatenate(([gain_step], gains[period + 1:])))
            loss_rest = pd.Series(np.concatenate(([loss_step], losses[period + 1:])))
            avg_gain = float(gain_rest.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
            avg_loss = float(loss_rest.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
        else:
            avg_gain = gain_step
            avg_loss = loss_step

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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
