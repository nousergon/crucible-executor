"""Parity tests for executor/feature_lookup.py (Tier 3 Part B).

Pins the invariant that ``FeatureLookup`` lookups produce IDENTICAL
output (within float precision) to the scalar implementations they
replace. The test suite is the contract: if a future PR alters either
side and breaks parity, this test fires before the v13 spot dispatch.

Coverage:
  * ATR(14) bulk vs scalar — final-bar value across multiple fixtures
  * RSI(14) bulk vs scalar — final-bar value across multiple fixtures
  * 20-day momentum % bulk vs inline calc
  * 20-day support level (rolling min of low) vs inline calc
  * Daily returns bulk vs inline pct_change
  * returns_window matches the slice ``check_correlation`` consumes

The bulk vs scalar comparison runs at the FINAL bar of the fixture
(matching what scalar callers actually produce). Earlier bars in the
bulk series differ slightly because the scalar reference computes only
at one point; FeatureLookup builds the full series.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from executor.feature_lookup import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_MOMENTUM_LOOKBACK,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SUPPORT_LOOKBACK,
    FeatureLookup,
)
from executor.strategies.exit_manager import _compute_atr, _compute_rsi


def _make_ohlcv(n_bars: int, base: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Deterministic OHLCV with realistic noise."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0.0, 1.0, n_bars))
    highs = closes + np.abs(rng.normal(0.5, 0.3, n_bars))
    lows = closes - np.abs(rng.normal(0.5, 0.3, n_bars))
    opens = closes + rng.normal(0.0, 0.2, n_bars)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


def _last_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1]


class TestATRParity:
    @pytest.mark.parametrize("n_bars,seed", [
        (50, 0), (100, 1), (250, 2), (400, 3),
    ])
    def test_atr_dollar_at_final_bar_matches_scalar(self, n_bars, seed):
        df = _make_ohlcv(n_bars, seed=seed)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})

        bulk_value = lookup.atr_dollar_at("AAPL", _last_date(df))
        scalar_value = _compute_atr(df, period=DEFAULT_ATR_PERIOD)

        assert bulk_value is not None and scalar_value is not None
        assert math.isclose(bulk_value, scalar_value, rel_tol=1e-9, abs_tol=1e-12), (
            f"ATR bulk vs scalar drift at n_bars={n_bars}, seed={seed}: "
            f"bulk={bulk_value!r} scalar={scalar_value!r}"
        )

    def test_atr_returns_none_for_short_history(self):
        df = _make_ohlcv(DEFAULT_ATR_PERIOD)  # need period+1 bars
        lookup = FeatureLookup.from_ohlcv_by_ticker({"SHORT": df})
        assert lookup.atr_dollar_at("SHORT", _last_date(df)) is None

    def test_atr_returns_none_for_unknown_ticker(self):
        df = _make_ohlcv(50)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        assert lookup.atr_dollar_at("MSFT", _last_date(df)) is None


class TestRSIParity:
    @pytest.mark.parametrize("n_bars,seed", [
        (50, 10), (100, 11), (250, 12), (400, 13),
    ])
    def test_rsi_at_final_bar_matches_scalar(self, n_bars, seed):
        df = _make_ohlcv(n_bars, seed=seed)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})

        bulk_value = lookup.rsi_at("AAPL", _last_date(df))
        scalar_value = _compute_rsi(df, period=DEFAULT_RSI_PERIOD)

        assert bulk_value is not None and scalar_value is not None
        assert math.isclose(bulk_value, scalar_value, rel_tol=1e-9, abs_tol=1e-12), (
            f"RSI bulk vs scalar drift at n_bars={n_bars}, seed={seed}: "
            f"bulk={bulk_value!r} scalar={scalar_value!r}"
        )

    def test_rsi_strict_uptrend_returns_100(self):
        n = 30
        closes = np.arange(100.0, 100.0 + n, 1.0)
        df = pd.DataFrame(
            {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        lookup = FeatureLookup.from_ohlcv_by_ticker({"UP": df})
        assert lookup.rsi_at("UP", _last_date(df)) == 100.0


class TestMomentumParity:
    @pytest.mark.parametrize("n_bars,seed", [
        (30, 20), (50, 21), (100, 22),
    ])
    def test_momentum_20d_pct_matches_inline_calc(self, n_bars, seed):
        df = _make_ohlcv(n_bars, seed=seed)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})

        bulk = lookup.momentum_20d_pct_at("AAPL", _last_date(df))
        # Inline reference (matches check_momentum_exit:351 + _plan_entries momentum gate)
        close = df["close"]
        scalar = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100

        assert bulk is not None
        assert math.isclose(bulk, scalar, rel_tol=1e-9, abs_tol=1e-12), (
            f"Momentum drift: bulk={bulk!r} scalar={scalar!r}"
        )

    def test_momentum_short_history_returns_none(self):
        # Need at least 21 bars for 20-day momentum
        df = _make_ohlcv(20)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"SHORT": df})
        result = lookup.momentum_20d_pct_at("SHORT", _last_date(df))
        # 20 bars produces NaN at every point (pct_change(20) needs index 20)
        assert result is None


class TestSupportParity:
    def test_support_20_low_matches_inline_min(self):
        df = _make_ohlcv(50, seed=30)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})

        bulk = lookup.support_20_low_at("AAPL", _last_date(df))
        # Reference: _compute_support_level uses last-N min of valid lows.
        # FeatureLookup uses a strict 20-bar rolling-min (no positivity filter
        # — fixtures don't have zero/negative lows).
        scalar = float(df["low"].iloc[-DEFAULT_SUPPORT_LOOKBACK:].min())

        assert bulk is not None
        assert math.isclose(bulk, scalar, rel_tol=1e-12, abs_tol=1e-12), (
            f"Support drift: bulk={bulk!r} scalar={scalar!r}"
        )


class TestReturnsWindowParity:
    def test_returns_window_matches_pct_change_dropna_tail(self):
        """Matches risk_guard.check_correlation:
            candidate_returns = candidate_history["close"].iloc[-lookback:].pct_change().dropna()
        """
        df = _make_ohlcv(100, seed=40)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})

        lookback = 60  # default correlation_lookback_days
        bulk = lookup.returns_window("AAPL", _last_date(df), lookback - 1)
        # Reference: pct_change over the last `lookback` closes, drop the
        # leading NaN. That gives `lookback - 1` returns.
        scalar = (
            df["close"].iloc[-lookback:].pct_change().dropna().to_numpy(dtype=float)
        )

        assert bulk is not None
        assert bulk.shape == scalar.shape, (
            f"Shape drift: bulk={bulk.shape} scalar={scalar.shape}"
        )
        np.testing.assert_allclose(bulk, scalar, rtol=1e-12, atol=1e-12)

    def test_returns_window_too_short_returns_none(self):
        df = _make_ohlcv(10, seed=41)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        # Asking for 60 returns with only 10 bars → None
        assert lookup.returns_window("AAPL", _last_date(df), 60) is None


class TestConstructorAlias:
    def test_from_price_histories_equivalent_to_from_ohlcv(self):
        """``from_price_histories`` is a semantic alias for the live
        executor — same shape input, same lookups."""
        df = _make_ohlcv(50, seed=50)
        a = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        b = FeatureLookup.from_price_histories({"AAPL": df})

        assert (
            a.atr_dollar_at("AAPL", _last_date(df))
            == b.atr_dollar_at("AAPL", _last_date(df))
        )
        assert (
            a.rsi_at("AAPL", _last_date(df))
            == b.rsi_at("AAPL", _last_date(df))
        )


class TestHasData:
    def test_has_data_true_at_final_bar(self):
        df = _make_ohlcv(50, seed=60)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        assert lookup.has_data("AAPL", _last_date(df)) is True

    def test_has_data_false_for_unknown_ticker(self):
        df = _make_ohlcv(50, seed=61)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        assert lookup.has_data("MSFT", _last_date(df)) is False

    def test_has_data_false_before_history_start(self):
        df = _make_ohlcv(50, seed=62)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        # Date before the fixture's start → asof returns NaN → has_data False
        before = pd.Timestamp("2020-01-01")
        assert lookup.has_data("AAPL", before) is False


class TestFrozenSafety:
    def test_dataclass_is_frozen(self):
        df = _make_ohlcv(30, seed=70)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"AAPL": df})
        # Frozen → assigning a top-level field raises
        with pytest.raises(Exception):  # FrozenInstanceError
            lookup.atr_dollar = {}  # type: ignore[misc]


class TestEmptyAndDegenerateInputs:
    def test_empty_dict_yields_empty_lookups(self):
        lookup = FeatureLookup.from_ohlcv_by_ticker({})
        assert lookup.atr_dollar == {}
        assert lookup.rsi == {}
        assert lookup.momentum_20d_pct == {}
        assert lookup.returns == {}
        assert lookup.support_20_low == {}

    def test_empty_dataframe_skipped(self):
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )
        df = _make_ohlcv(50)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"EMPTY": empty, "AAPL": df})
        assert "EMPTY" not in lookup.atr_dollar
        assert "AAPL" in lookup.atr_dollar

    def test_missing_columns_skipped(self):
        # A frame without 'high'/'low' is dropped
        bad_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        df = _make_ohlcv(50)
        lookup = FeatureLookup.from_ohlcv_by_ticker({"BAD": bad_df, "AAPL": df})
        assert "BAD" not in lookup.atr_dollar
        assert "AAPL" in lookup.atr_dollar
