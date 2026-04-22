"""Tests for executor.price_cache.load_feature_coverage.

Contract (2026-04-22):
    Coverage for each ticker = non-NaN feature cols / total feature cols
    on the most-recent ArcticDB universe row. Features = everything
    except OHLCV+VWAP raw market data. A full-history ticker returns
    ~1.0; SNDK-class short-history tickers return < 1.0 because
    252-day features stay NaN on every row.

Used by the position sizer's coverage derate AND the admission gate —
so this helper's correctness is load-bearing for the graceful-degrade
chain established 2026-04-21 evening.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if "arcticdb" not in sys.modules:
    sys.modules["arcticdb"] = MagicMock()

from executor import price_cache  # noqa: E402
from executor.price_cache import load_feature_coverage  # noqa: E402


def _frame_full_coverage() -> pd.DataFrame:
    """Full-coverage ticker — every feature column populated on last row."""
    return pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [101.0, 102.0],
        "Low": [99.0, 100.0],
        "Close": [100.5, 101.5],
        "Volume": [1_000_000, 1_100_000],
        "VWAP": [100.2, 101.2],
        "atr_14_pct": [0.02, 0.021],
        "rsi_14": [55.0, 56.0],
        "momentum_60d": [0.05, 0.06],
        "momentum_252d": [0.15, 0.16],
        "dist_from_52w_high": [-0.03, -0.02],
    }, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))


def _frame_partial_coverage() -> pd.DataFrame:
    """SNDK-class short-history ticker — 3 out of 5 feature cols NaN on last row.

    atr_14_pct + rsi_14 populate (short warmup). momentum_60d / momentum_252d /
    dist_from_52w_high stay NaN (warmup exceeds ticker's history).
    """
    return pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000],
        "VWAP": [100.2],
        "atr_14_pct": [0.02],
        "rsi_14": [55.0],
        "momentum_60d": [float("nan")],
        "momentum_252d": [float("nan")],
        "dist_from_52w_high": [float("nan")],
    }, index=pd.DatetimeIndex(["2024-01-02"]))


def _frame_zero_coverage() -> pd.DataFrame:
    """Brand-new listing — every feature NaN (warmup exceeds history for all)."""
    return pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000],
        "VWAP": [100.2],
        "atr_14_pct": [float("nan")],
        "rsi_14": [float("nan")],
        "momentum_60d": [float("nan")],
    }, index=pd.DatetimeIndex(["2024-01-02"]))


def _mock_universe(frames: dict[str, pd.DataFrame | Exception]):
    """Build a mock universe library that returns a given frame per ticker,
    or raises when the value is an Exception instance.
    """
    universe = MagicMock()

    def _read(ticker):
        value = frames[ticker]
        if isinstance(value, Exception):
            raise value
        result = MagicMock()
        result.data = value
        return result

    universe.read.side_effect = _read
    return universe


class TestLoadFeatureCoverage:
    def test_full_coverage_ticker_returns_one(self):
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"AAPL": _frame_full_coverage()}),
        ):
            out = load_feature_coverage(["AAPL"], "test-bucket")

        assert out["AAPL"] == pytest.approx(1.0)

    def test_partial_coverage_ticker_returns_fraction(self):
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"SNDK": _frame_partial_coverage()}),
        ):
            out = load_feature_coverage(["SNDK"], "test-bucket")

        # 5 feature cols (atr, rsi, momentum_60d, momentum_252d, dist_from_52w_high);
        # 2 populated (atr + rsi). 2/5 = 0.4.
        assert out["SNDK"] == pytest.approx(0.4)

    def test_zero_coverage_ticker_returns_zero(self):
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"IPO": _frame_zero_coverage()}),
        ):
            out = load_feature_coverage(["IPO"], "test-bucket")

        assert out["IPO"] == pytest.approx(0.0)

    def test_ohlcv_cols_not_counted_as_features(self):
        """Full-coverage frame should report 1.0 — OHLCV+VWAP in the denominator
        would dilute the ratio toward 1.0 even for short-history tickers and
        break the partial-coverage discrimination.
        """
        frame = _frame_partial_coverage()
        # Confirm OHLCV cols present but excluded from feature count.
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"SNDK": frame}),
        ):
            out = load_feature_coverage(["SNDK"], "test-bucket")
        assert out["SNDK"] == pytest.approx(0.4)
        assert out["SNDK"] < 1.0, (
            "OHLCV+VWAP columns must not be counted as features — otherwise "
            "coverage reports ~1.0 for tickers whose real feature coverage "
            "is much lower and the derate/admission signal disappears."
        )

    def test_ohlcv_only_frame_returns_zero(self):
        """A frame with no engineered features at all → 0.0 coverage +
        WARNING log. Admission gate naturally rejects.
        """
        frame = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1_000_000], "VWAP": [100.2],
        }, index=pd.DatetimeIndex(["2024-01-02"]))

        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"OHLCV_ONLY": frame}),
        ):
            out = load_feature_coverage(["OHLCV_ONLY"], "test-bucket")
        assert out["OHLCV_ONLY"] == 0.0

    def test_empty_frame_returns_zero(self):
        frame = pd.DataFrame(columns=["Close", "atr_14_pct"])
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({"DELISTED": frame}),
        ):
            out = load_feature_coverage(["DELISTED"], "test-bucket")
        assert out["DELISTED"] == 0.0

    def test_arcticdb_read_error_hard_fails(self):
        """Any ticker's read raising must hard-fail the whole call.

        This is an infrastructure problem (library unreachable, IAM miss),
        not a data gap — same posture as load_atr_14_pct.
        """
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({
                "AAPL": _frame_full_coverage(),
                "BROKEN": RuntimeError("NoSuchVersionException"),
            }),
        ):
            with pytest.raises(RuntimeError, match="BROKEN"):
                load_feature_coverage(["AAPL", "BROKEN"], "test-bucket")

    def test_empty_ticker_list_returns_empty_dict(self):
        # Library should NOT be opened if there are no tickers to score.
        with patch.object(price_cache, "_open_universe_library") as open_lib:
            out = load_feature_coverage([], "test-bucket")
        assert out == {}
        open_lib.assert_not_called()

    def test_multi_ticker_mixed_coverage(self):
        """Realistic Saturday run: some full-history tickers, one short-history."""
        with patch.object(
            price_cache, "_open_universe_library",
            return_value=_mock_universe({
                "AAPL": _frame_full_coverage(),
                "MSFT": _frame_full_coverage(),
                "SNDK": _frame_partial_coverage(),
                "IPO": _frame_zero_coverage(),
            }),
        ):
            out = load_feature_coverage(["AAPL", "MSFT", "SNDK", "IPO"], "test-bucket")

        assert out["AAPL"] == pytest.approx(1.0)
        assert out["MSFT"] == pytest.approx(1.0)
        assert out["SNDK"] == pytest.approx(0.4)
        assert out["IPO"] == pytest.approx(0.0)
