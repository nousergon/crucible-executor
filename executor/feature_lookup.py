"""Vectorized cross-sectional feature precompute (Tier 3 Part B,
2026-04-27).

Provides ``FeatureLookup`` — per-(ticker, date) scalar feature lookups
backed by pandas Series. Replaces per-call recomputation in deciders
(``_compute_atr``, ``_compute_rsi``, ``check_correlation``,
``check_momentum_exit``, ``check_sector_relative_veto``,
``_compute_support_level``) with O(log N) DatetimeIndex lookups.

Two construction modes share the same lookup interface:

  * Backtester (``from_ohlcv_by_ticker``): bulk vectorized precompute
    across all tickers × all dates at simulation start. ONE pandas
    pass per feature per ticker, amortized across 60 combos × 2316
    dates. Total cost: ~5-30 sec for 10y × 911 tickers × 5 features.

  * Live executor (``from_price_histories``): scalar pass per ticker
    for the ~50 active tickers at executor boot. ~1 ms per ticker × 50
    = ~50 ms once per day, then O(1) lookups for the rest of the
    morning planner call.

Both paths produce IDENTICAL lookup outputs (within float precision —
Wilder's exponential smoothing converges within 5*period bars to the
seed-independent steady-state).

Tier 3 Part C (separate PR) wires the deciders to consume
``FeatureLookup`` instead of recomputing per call. This module is the
infrastructure prereq.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level defaults — match the per-decider scalar callers' periods.
# Centralizing them here lets deciders ask FeatureLookup for the
# canonical period without duplicating the constant.
DEFAULT_ATR_PERIOD = 14
DEFAULT_RSI_PERIOD = 14
DEFAULT_MOMENTUM_LOOKBACK = 20
DEFAULT_SUPPORT_LOOKBACK = 20


@dataclass(frozen=True)
class FeatureLookup:
    """Per-ticker × per-date precomputed scalar feature lookups.

    Each attribute is ``dict[ticker, pd.Series]`` indexed by
    DatetimeIndex. Lookup methods (``atr_dollar_at`` etc.) wrap the
    Series .asof(date) accessor so callers don't deal with pandas
    directly.

    Construct via:
      * ``FeatureLookup.from_ohlcv_by_ticker(ohlcv)`` — bulk vectorized.
      * ``FeatureLookup.from_price_histories(histories)`` — sparse / live.

    Frozen: callers must NOT mutate the inner Series across combos in a
    param sweep (would pollute shared state). The dataclass freeze
    catches accidental rebinding; the inner mutability is by convention.
    """

    # Wilder ATR (dollar-units) per ticker, indexed by date.
    # Matches scalar ``_compute_atr(price_history, period=14)`` to
    # within float precision.
    atr_dollar: dict

    # Wilder RSI(14) per ticker.
    # Matches scalar ``_compute_rsi(price_history, period=14)``.
    rsi: dict

    # 20-day percentage momentum: 100 * (close[t] / close[t-20] - 1).
    # Matches the inline calculation in
    # ``check_momentum_exit`` and ``_plan_entries`` momentum gate.
    momentum_20d_pct: dict

    # Daily simple returns (close.pct_change). Used by check_correlation
    # — caller asks for a window of N consecutive returns ending at a
    # date, FeatureLookup returns a numpy slice; pearson computed
    # on those slices directly.
    returns: dict

    # 20-day rolling MIN of `low`. Matches
    # ``_compute_support_level(history, lookback=20)``.
    support_20_low: dict

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_ohlcv_by_ticker(
        cls,
        ohlcv_by_ticker: dict,
        *,
        atr_period: int = DEFAULT_ATR_PERIOD,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
        support_lookback: int = DEFAULT_SUPPORT_LOOKBACK,
    ) -> "FeatureLookup":
        """Bulk vectorized precompute across all (ticker, date) pairs.

        Each ticker's DataFrame must have ``[open, high, low, close]``
        columns and a sorted DatetimeIndex. Tickers with empty / None
        DataFrames are silently skipped (consistent with prior scalar
        callers' early-return behavior).

        Wilder's exponential smoothing implementation matches the
        scalar ``_compute_atr`` / ``_compute_rsi`` byte-for-byte (modulo
        float-precision noise of ~1e-12): SMA seed over the first
        `period` values, then ``ewm(alpha=1/period, adjust=False)`` —
        which IS Wilder's recurrence ``out_i = (1-alpha)*out_{i-1} +
        alpha*x_i``.

        For the seed: pandas.ewm doesn't accept a custom seed, so we
        build the smoothed series by manually computing the first
        ``period`` SMA value then concatenating with the ewm of the
        remaining bars. See the parity tests for the equivalence proof.
        """
        atr_dollar: dict = {}
        rsi: dict = {}
        momentum_20d_pct: dict = {}
        returns: dict = {}
        support_20_low: dict = {}

        for ticker, df in ohlcv_by_ticker.items():
            if df is None or df.empty:
                continue

            # Defensive: ensure required columns exist.
            cols = set(df.columns)
            if not {"open", "high", "low", "close"}.issubset(cols):
                logger.debug(
                    "FeatureLookup: skipping %s — missing OHLCV columns (%s)",
                    ticker, sorted(cols),
                )
                continue

            atr_series = _compute_atr_series(df, period=atr_period)
            if atr_series is not None:
                atr_dollar[ticker] = atr_series

            rsi_series = _compute_rsi_series(df, period=rsi_period)
            if rsi_series is not None:
                rsi[ticker] = rsi_series

            close = df["close"]
            momentum_20d_pct[ticker] = (close.pct_change(periods=momentum_lookback) * 100.0)
            returns[ticker] = close.pct_change()
            support_20_low[ticker] = df["low"].rolling(window=support_lookback).min()

        return cls(
            atr_dollar=atr_dollar,
            rsi=rsi,
            momentum_20d_pct=momentum_20d_pct,
            returns=returns,
            support_20_low=support_20_low,
        )

    @classmethod
    def from_price_histories(
        cls,
        price_histories: dict,
        **kwargs,
    ) -> "FeatureLookup":
        """Sparse precompute for the live-executor path.

        ``price_histories`` is the same shape as ``ohlcv_by_ticker``
        (dict of DataFrames) — they're interchangeable post-PR-#108.
        This alias exists so live shell code can express "I have the
        per-ticker histories already loaded" without conflating with
        the bulk-cross-sectional connotation of ``ohlcv_by_ticker``.

        kwargs are forwarded to ``from_ohlcv_by_ticker``.
        """
        return cls.from_ohlcv_by_ticker(price_histories, **kwargs)

    # ── Lookups ─────────────────────────────────────────────────────

    def atr_dollar_at(
        self, ticker: str, date: "pd.Timestamp | str",
    ) -> float | None:
        """Wilder ATR(period) at ``date`` for ``ticker``, in dollar units.

        Returns None if ticker isn't tracked or the date precedes the
        start of computed history (insufficient bars to seed Wilder's
        smoothing).
        """
        return _series_value_at(self.atr_dollar.get(ticker), date)

    def rsi_at(
        self, ticker: str, date: "pd.Timestamp | str",
    ) -> float | None:
        return _series_value_at(self.rsi.get(ticker), date)

    def momentum_20d_pct_at(
        self, ticker: str, date: "pd.Timestamp | str",
    ) -> float | None:
        return _series_value_at(self.momentum_20d_pct.get(ticker), date)

    def support_20_low_at(
        self, ticker: str, date: "pd.Timestamp | str",
    ) -> float | None:
        return _series_value_at(self.support_20_low.get(ticker), date)

    def returns_window(
        self,
        ticker: str,
        end_date: "pd.Timestamp | str",
        n: int,
    ) -> "np.ndarray | None":
        """N consecutive daily returns ending at ``end_date``.

        Returns None if ticker isn't tracked or fewer than n returns
        precede ``end_date``. Matches the scalar
        ``risk_guard.check_correlation`` window: the last N values of
        ``close.pct_change().dropna()`` up to and including ``end_date``.
        """
        s = self.returns.get(ticker)
        if s is None:
            return None
        ts = pd.Timestamp(end_date)
        # ``s.loc[:ts]`` is binary search on the DatetimeIndex; tail
        # then drops NaN matching ``check_correlation``'s ``.dropna()``.
        slice_ = s.loc[:ts].dropna()
        if len(slice_) < n:
            return None
        return slice_.iloc[-n:].to_numpy(dtype=float, copy=False)

    def has_data(
        self, ticker: str, date: "pd.Timestamp | str",
    ) -> bool:
        """True if any tracked feature has a non-NaN value for
        ``(ticker, date)``. Used by deciders to short-circuit when a
        ticker's history doesn't reach the date (e.g. a recent IPO at
        an early synth signal date)."""
        ts = pd.Timestamp(date)
        for store in (
            self.atr_dollar, self.rsi, self.momentum_20d_pct,
            self.returns, self.support_20_low,
        ):
            s = store.get(ticker)
            if s is None:
                continue
            try:
                v = s.asof(ts)
            except (KeyError, ValueError):
                continue
            if pd.notna(v):
                return True
        return False


# ── Internal helpers ────────────────────────────────────────────────


def _compute_atr_series(df: pd.DataFrame, period: int) -> "pd.Series | None":
    """Wilder ATR(period) as a Series indexed by ``df.index``.

    Returns None if ``df`` has fewer than ``period + 1`` rows.

    Matches scalar ``_compute_atr(price_history, period)`` byte-for-byte
    at the final bar (within float precision). The intermediate values
    along the series differ slightly because the scalar reference only
    computes the final bar; the FeatureLookup builds the full series so
    historical-date queries return the value as-of-then.

    Implementation:
      1. true_range = max(high-low, |high - prev_close|, |low - prev_close|).
      2. Wilder's smoothed ATR is seeded with SMA of first `period` true
         ranges, then ``out_i = (1-alpha)*out_{i-1} + alpha*tr_i``
         with alpha = 1/period — equivalent to ``ewm(alpha=1/period,
         adjust=False)`` PROVIDED the seed is the first sample
         (the default ewm seed). Since we want SMA seed instead of
         first-sample seed, we replace the first `period` values with
         the SMA, then run ewm from that point.
    """
    if len(df) < period + 1:
        return None

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    # tr[0] is high[0] - low[0] (since prev_close=NaN; max with NaN
    # propagates to NaN — np.maximum.reduce treats NaN as NaN).
    # Match the scalar reference which starts true_ranges at index 1
    # (using prev_close = close[0]). Set tr[0] = NaN explicitly.
    tr[0] = np.nan

    # Build Wilder ATR series matching the scalar implementation.
    # Scalar starts the smoothing series at bar `period` (index `period`
    # in tr), with value = SMA of tr[1..period] (period values; tr[0]
    # is NaN). Subsequent values compounded via Wilder's recurrence.
    atr_arr = np.full(len(df), np.nan, dtype=float)
    if len(tr) <= period:
        return None  # not enough non-NaN TRs

    sma_seed = float(np.mean(tr[1 : period + 1]))
    atr_arr[period] = sma_seed
    alpha = 1.0 / period
    for i in range(period + 1, len(tr)):
        atr_arr[i] = atr_arr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    return pd.Series(atr_arr, index=df.index, name="atr")


def _compute_rsi_series(df: pd.DataFrame, period: int) -> "pd.Series | None":
    """Wilder RSI(period) as a Series indexed by ``df.index``.

    Matches scalar ``_compute_rsi`` byte-for-byte at the final bar.
    """
    if len(df) < period + 1:
        return None

    close = df["close"].to_numpy(dtype=float)
    changes = np.diff(close, prepend=np.nan)
    # Drop the leading NaN slot — first valid change is at index 1.
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    gains[0] = np.nan
    losses[0] = np.nan

    if len(gains) <= period:
        return None

    rsi_arr = np.full(len(df), np.nan, dtype=float)
    avg_gain = float(np.mean(gains[1 : period + 1]))
    avg_loss = float(np.mean(losses[1 : period + 1]))
    alpha = 1.0 / period

    # First RSI value at bar `period` (index period in df).
    rsi_arr[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period + 1, len(close)):
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha
        rsi_arr[i] = _rsi_from_avgs(avg_gain, avg_loss)

    return pd.Series(rsi_arr, index=df.index, name="rsi")


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _series_value_at(
    series: "pd.Series | None", date: "pd.Timestamp | str",
) -> float | None:
    """Look up a series value as-of ``date``, returning None for
    missing/NaN.

    Uses ``Series.asof`` which falls back to the last value at-or-before
    ``date``. For features that are NaN at the queried date (e.g. early
    in history before Wilder smoothing converges), returns None.
    """
    if series is None:
        return None
    ts = pd.Timestamp(date)
    try:
        value = series.asof(ts)
    except (KeyError, ValueError):
        return None
    if value is None or pd.isna(value):
        return None
    return float(value)
