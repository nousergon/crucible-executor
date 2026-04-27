"""Parity test for vectorized _compute_atr / _compute_rsi (2026-04-27).

The pre-vectorized scalar Python implementations are reproduced inline
here as the reference oracle. Each test asserts the production
DataFrame-input version matches within ``rel=1e-9``.

Wilder's smoothing is well-conditioned: the seed-bar (SMA over first
period) contribution decays as ``(1-1/period)^N``, falling below 1% of
the final value within ~5*period bars. For period=14 the convergence is
~70 bars; for the 30/100/400-bar fixtures here the agreement is to
machine precision (~14 sig figs).

If a future PR alters either implementation and breaks parity, this
test catches it before live executor and backtester silently diverge on
ATR-trailing-stop / momentum-exit decisions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from executor.strategies.exit_manager import _compute_atr, _compute_rsi


def _scalar_compute_atr(price_history: list[dict], period: int = 14) -> float | None:
    """Pre-vectorized reference implementation (preserved for parity).

    Bit-for-bit copy of the scalar Python loop that lived in
    executor/strategies/exit_manager.py prior to 2026-04-27.
    """
    if len(price_history) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(price_history)):
        bar = price_history[i]
        prev_close = price_history[i - 1]["close"]
        high = bar["high"]
        low = bar["low"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges[:period]) / period
    alpha = 1.0 / period
    for tr in true_ranges[period:]:
        atr = atr * (1 - alpha) + tr * alpha
    return atr


def _scalar_compute_rsi(price_history: list[dict], period: int = 14) -> float | None:
    """Pre-vectorized reference implementation (preserved for parity)."""
    if len(price_history) < period + 1:
        return None
    changes = [
        price_history[i]["close"] - price_history[i - 1]["close"]
        for i in range(1, len(price_history))
    ]
    gains = [max(c, 0) for c in changes]
    losses = [max(-c, 0) for c in changes]
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _make_bars(n: int, base: float = 100.0, trend: float = 0.0, vol: float = 1.0,
               seed: int = 0) -> tuple[list[dict], pd.DataFrame]:
    """Build matched list-of-dicts + DataFrame fixtures for a deterministic
    OHLCV series."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(trend, vol, n))
    highs = closes + np.abs(rng.normal(0.5, 0.3, n))
    lows = closes - np.abs(rng.normal(0.5, 0.3, n))
    opens = closes + rng.normal(0.0, 0.2, n)
    bars = [
        {"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
         "open": float(opens[i]),
         "high": float(highs[i]),
         "low": float(lows[i]),
         "close": float(closes[i])}
        for i in range(n)
    ]
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    return bars, df


class TestATRParity:
    @pytest.mark.parametrize("n,trend,vol,seed", [
        (30, 0.0, 1.0, 0),
        (50, 0.1, 0.5, 1),
        (100, -0.05, 2.0, 2),
        (400, 0.02, 1.5, 3),
    ])
    def test_atr_matches_scalar_within_float_precision(self, n, trend, vol, seed):
        bars, df = _make_bars(n, trend=trend, vol=vol, seed=seed)
        scalar = _scalar_compute_atr(bars, period=14)
        vector = _compute_atr(df, period=14)
        assert scalar is not None and vector is not None
        # Wilder convergence is deterministic — agreement to machine precision
        assert math.isclose(scalar, vector, rel_tol=1e-9, abs_tol=1e-12), (
            f"ATR drift: scalar={scalar!r} vs vector={vector!r}"
        )

    def test_atr_returns_none_for_insufficient_history(self):
        bars, df = _make_bars(14)  # need period + 1 = 15
        assert _scalar_compute_atr(bars, period=14) is None
        assert _compute_atr(df, period=14) is None

    def test_atr_returns_seed_value_when_exactly_period_bars(self):
        # period + 1 bars total, period TRs — should return the SMA seed
        bars, df = _make_bars(15)
        scalar = _scalar_compute_atr(bars, period=14)
        vector = _compute_atr(df, period=14)
        assert scalar is not None and vector is not None
        assert math.isclose(scalar, vector, rel_tol=1e-9)


class TestRSIParity:
    @pytest.mark.parametrize("n,trend,vol,seed", [
        (30, 0.0, 1.0, 10),
        (50, 0.1, 0.5, 11),
        (100, -0.05, 2.0, 12),
        (400, 0.02, 1.5, 13),
    ])
    def test_rsi_matches_scalar_within_float_precision(self, n, trend, vol, seed):
        bars, df = _make_bars(n, trend=trend, vol=vol, seed=seed)
        scalar = _scalar_compute_rsi(bars, period=14)
        vector = _compute_rsi(df, period=14)
        assert scalar is not None and vector is not None
        assert math.isclose(scalar, vector, rel_tol=1e-9, abs_tol=1e-12), (
            f"RSI drift: scalar={scalar!r} vs vector={vector!r}"
        )

    def test_rsi_returns_none_for_insufficient_history(self):
        bars, df = _make_bars(14)
        assert _scalar_compute_rsi(bars, period=14) is None
        assert _compute_rsi(df, period=14) is None

    def test_rsi_all_gains_returns_100(self):
        # Strict uptrend → no losses → RSI = 100
        n = 30
        closes = np.arange(100.0, 100.0 + n, 1.0)
        bars = [{"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                 "open": float(closes[i]), "high": float(closes[i] + 1),
                 "low": float(closes[i] - 1), "close": float(closes[i])}
                for i in range(n)]
        df = pd.DataFrame(
            {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        assert _scalar_compute_rsi(bars, period=14) == 100.0
        assert _compute_rsi(df, period=14) == 100.0
