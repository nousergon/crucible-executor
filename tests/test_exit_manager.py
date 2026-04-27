"""Unit tests for executor.strategies.exit_manager — pure exit logic, no IBKR/S3."""
import pandas as pd
import pytest
from datetime import date, timedelta

from executor.strategies.exit_manager import (
    check_atr_trailing_stop,
    check_time_decay,
    check_profit_take,
    check_fallback_stop,
    check_sector_relative_veto,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_price_history(
    n_bars: int = 30,
    base_price: float = 100.0,
    start_date: str = "2026-01-01",
    trend: float = 0.0,
    high_offset: float = 2.0,
    low_offset: float = 2.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV price history as a DataFrame.

    Args:
        n_bars: number of daily bars
        base_price: starting close price
        start_date: ISO start date
        trend: daily trend increment (positive = uptrend)
        high_offset: how much higher than close
        low_offset: how much lower than close

    Returns:
        DataFrame indexed by DatetimeIndex with [open, high, low, close]
        columns. Skips weekends to mirror trading-day cadence.
    """
    dates = []
    opens, highs, lows, closes = [], [], [], []
    dt = date.fromisoformat(start_date)
    for i in range(n_bars):
        close = base_price + trend * i
        dates.append(pd.Timestamp(dt.isoformat()))
        opens.append(close - 0.5)
        highs.append(close + high_offset)
        lows.append(close - low_offset)
        closes.append(close)
        dt += timedelta(days=1)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.DatetimeIndex(dates),
    )


def _strategy_config(**overrides):
    """Minimal strategy config dict."""
    cfg = {
        "atr_trailing_enabled": True,
        "atr_period": 14,
        "atr_multiplier": 3.0,
        "time_decay_enabled": True,
        "time_decay_reduce_days": 5,
        "time_decay_exit_days": 10,
        "profit_take_enabled": True,
        "profit_take_pct": 0.25,
        "sector_relative_veto_enabled": True,
        "sector_relative_outperform_threshold": 0.05,
        "fallback_stop_enabled": True,
        "fallback_stop_pct": 0.10,
    }
    cfg.update(overrides)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# check_atr_trailing_stop
# ═══════════════════════════════════════════════════════════════════════════════


class TestAtrTrailingStop:

    def test_stop_triggers_when_price_below_trail(self):
        """Price well below highest_high - ATR * multiplier → EXIT."""
        history = _make_price_history(
            n_bars=20,
            base_price=100.0,
            start_date="2026-01-02",
            high_offset=2.0,
            low_offset=2.0,
        )
        # ATR is roughly the high-low range (4.0) for uniform bars.
        # With multiplier=3.0, stop ~ highest_high - 3*ATR = 102 - 12 = 90.
        # Set current price well below the stop level.
        result = check_atr_trailing_stop(
            ticker="AAPL",
            current_price=80.0,
            entry_date="2026-01-02",
            price_history=history,
            strategy_config=_strategy_config(),
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["reason"] == "atr_trailing_stop"

    def test_stop_does_not_trigger_above_trail(self):
        """Price above the trailing stop → no exit."""
        history = _make_price_history(
            n_bars=20,
            base_price=100.0,
            start_date="2026-01-02",
            high_offset=2.0,
            low_offset=2.0,
        )
        result = check_atr_trailing_stop(
            ticker="AAPL",
            current_price=100.0,
            entry_date="2026-01-02",
            price_history=history,
            strategy_config=_strategy_config(),
        )
        assert result is None

    def test_atr_disabled_returns_none(self):
        history = _make_price_history(n_bars=20, start_date="2026-01-02")
        result = check_atr_trailing_stop(
            ticker="AAPL",
            current_price=50.0,
            entry_date="2026-01-02",
            price_history=history,
            strategy_config=_strategy_config(atr_trailing_enabled=False),
        )
        assert result is None

    def test_insufficient_bars_returns_none(self):
        """Fewer than ATR period + 1 bars → skip."""
        history = _make_price_history(n_bars=5, start_date="2026-01-02")
        result = check_atr_trailing_stop(
            ticker="AAPL",
            current_price=50.0,
            entry_date="2026-01-02",
            price_history=history,
            strategy_config=_strategy_config(),
        )
        assert result is None

    def test_empty_history_returns_none(self):
        result = check_atr_trailing_stop(
            ticker="AAPL",
            current_price=50.0,
            entry_date="2026-01-02",
            price_history=pd.DataFrame(
                columns=["open", "high", "low", "close"],
                index=pd.DatetimeIndex([]),
            ),
            strategy_config=_strategy_config(),
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# check_time_decay
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeDecay:

    def test_reduce_triggers_after_n_days_with_hold(self):
        """After 5+ trading days with HOLD signal → REDUCE."""
        # 5 trading days ~ 7 calendar days
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-03-02",  # Monday
            run_date="2026-03-09",    # Next Monday (5 trading days)
            signal_action="HOLD",
            strategy_config=_strategy_config(time_decay_reduce_days=5),
        )
        assert result is not None
        assert result["action"] == "REDUCE"
        assert result["reason"] == "time_decay_reduce"

    def test_exit_triggers_after_m_days_with_hold(self):
        """After 10+ trading days with HOLD signal → EXIT."""
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-03-02",  # Monday
            run_date="2026-03-16",    # Two Mondays later (10 trading days)
            signal_action="HOLD",
            strategy_config=_strategy_config(
                time_decay_reduce_days=5,
                time_decay_exit_days=10,
            ),
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["reason"] == "time_decay_exit"

    def test_no_decay_when_signal_is_enter(self):
        """ENTER signal (reaffirming) → no time decay, even after many days."""
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-01-02",
            run_date="2026-03-16",  # Many weeks later
            signal_action="ENTER",
            strategy_config=_strategy_config(),
        )
        assert result is None

    def test_no_decay_when_signal_is_exit(self):
        """EXIT signal from research → skip time decay (research handles it)."""
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-01-02",
            run_date="2026-03-16",
            signal_action="EXIT",
            strategy_config=_strategy_config(),
        )
        assert result is None

    def test_no_decay_when_signal_is_reduce(self):
        """REDUCE from research → skip time decay."""
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-01-02",
            run_date="2026-03-16",
            signal_action="REDUCE",
            strategy_config=_strategy_config(),
        )
        assert result is None

    def test_no_decay_within_threshold(self):
        """Position held for fewer than reduce_days → no decay."""
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-03-02",  # Monday
            run_date="2026-03-04",    # Wednesday (2 trading days)
            signal_action="HOLD",
            strategy_config=_strategy_config(time_decay_reduce_days=5),
        )
        assert result is None

    def test_time_decay_disabled_returns_none(self):
        result = check_time_decay(
            ticker="AAPL",
            entry_date="2026-01-02",
            run_date="2026-03-16",
            signal_action="HOLD",
            strategy_config=_strategy_config(time_decay_enabled=False),
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# check_profit_take
# ═══════════════════════════════════════════════════════════════════════════════


class TestProfitTake:

    def test_profit_exceeds_threshold_triggers_reduce(self):
        result = check_profit_take(
            ticker="AAPL",
            current_price=130.0,
            avg_cost=100.0,
            strategy_config=_strategy_config(profit_take_pct=0.25),
        )
        assert result is not None
        assert result["action"] == "REDUCE"
        assert result["reason"] == "profit_take"

    def test_profit_below_threshold_returns_none(self):
        result = check_profit_take(
            ticker="AAPL",
            current_price=120.0,
            avg_cost=100.0,
            strategy_config=_strategy_config(profit_take_pct=0.25),
        )
        assert result is None

    def test_no_avg_cost_returns_none(self):
        result = check_profit_take(
            ticker="AAPL",
            current_price=130.0,
            avg_cost=None,
            strategy_config=_strategy_config(),
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# check_fallback_stop
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallbackStop:

    def test_fallback_triggers_below_stop(self):
        """Price < entry * (1 - 10%) → EXIT."""
        result = check_fallback_stop(
            ticker="AAPL",
            current_price=85.0,
            entry_price=100.0,
            strategy_config=_strategy_config(fallback_stop_pct=0.10),
        )
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["reason"] == "fallback_stop"

    def test_fallback_does_not_trigger_above_stop(self):
        result = check_fallback_stop(
            ticker="AAPL",
            current_price=95.0,
            entry_price=100.0,
            strategy_config=_strategy_config(fallback_stop_pct=0.10),
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# check_sector_relative_veto
# ═══════════════════════════════════════════════════════════════════════════════


class TestSectorRelativeVeto:

    def test_outperformance_vetoes_exit(self):
        """Stock outperforming sector ETF by > threshold → veto exit."""
        # Stock goes from 100 → 115 (15% return)
        stock_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.6)
        # ETF goes from 100 → 102 (2% return)
        etf_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.08)

        vetoed = check_sector_relative_veto(
            ticker="AAPL",
            sector="Technology",
            price_history=stock_history,
            sector_etf_history=etf_history,
            strategy_config=_strategy_config(sector_relative_outperform_threshold=0.05),
        )
        assert vetoed is True

    def test_no_outperformance_does_not_veto(self):
        """Stock and sector have similar returns → no veto."""
        stock_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.1)
        etf_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.1)

        vetoed = check_sector_relative_veto(
            ticker="AAPL",
            sector="Technology",
            price_history=stock_history,
            sector_etf_history=etf_history,
            strategy_config=_strategy_config(),
        )
        assert vetoed is False

    def test_veto_disabled_returns_false(self):
        stock_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.6)
        etf_history = _make_price_history(n_bars=25, base_price=100.0, trend=0.08)

        vetoed = check_sector_relative_veto(
            ticker="AAPL",
            sector="Technology",
            price_history=stock_history,
            sector_etf_history=etf_history,
            strategy_config=_strategy_config(sector_relative_veto_enabled=False),
        )
        assert vetoed is False
