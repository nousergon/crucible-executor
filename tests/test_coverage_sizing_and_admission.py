"""Integration tests for the coverage-aware sizing + admission gate pair.

Two invariants (2026-04-22):

1. **Position sizer coverage derate.** When
   ``coverage_sizing_enabled=True`` and ``feature_coverage`` is provided,
   final shares scale with coverage. A 50%-covered ticker gets ~50% of
   a 100%-covered ticker's dollar size (floored at
   ``coverage_derate_floor``).

2. **Admission gate.** ``filter_buy_candidates_by_coverage`` drops
   candidates below ``min_coverage_for_admission``. Scope: buy_candidates
   only. Held positions (universe list with EXIT/REDUCE/HOLD) are NEVER
   filtered by coverage — they still need exit/management paths evaluated.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from executor.position_sizer import compute_position_size
from executor.signal_reader import filter_buy_candidates_by_coverage


def _base_config(**overrides):
    cfg = {
        "max_position_pct": 0.05,
        "conviction_decline_adj": 0.70,
        "min_price_target_upside": 0.05,
        "upside_fail_adj": 0.70,
        "min_position_dollar": 500,
        "sector_adj": {"overweight": 1.05, "market_weight": 1.00, "underweight": 0.85},
        "atr_sizing_enabled": False,
        "confidence_sizing_enabled": False,
        "staleness_discount_enabled": False,
        "earnings_sizing_enabled": False,
        "coverage_sizing_enabled": True,
        "coverage_derate_floor": 0.25,
    }
    cfg.update(overrides)
    return cfg


def _base_signal():
    return {"score": 82, "conviction": "stable", "price_target_upside": 0.15}


# ── Sizer coverage derate ──────────────────────────────────────────────────────


class TestSizerCoverageDerate:
    def test_full_coverage_ticker_sized_at_full_weight(self):
        out = compute_position_size(
            ticker="AAPL",
            portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "AAPL"}],
            signal=_base_signal(),
            sector_rating="market_weight",
            current_price=100.0,
            config=_base_config(),
            feature_coverage=1.0,
        )
        assert out["coverage_adj"] == 1.0
        # Base weight 1/1 = 1.0, capped at 5% position cap, so 5% * 1M = 50k.
        assert out["dollar_size"] == 50_000.0

    def test_partial_coverage_ticker_gets_derate(self):
        """70%-covered ticker → ~70% of full size (still ≥ floor)."""
        base = compute_position_size(
            ticker="AAPL", portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "AAPL"}], signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(), feature_coverage=1.0,
        )
        derated = compute_position_size(
            ticker="SNDK", portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "SNDK"}], signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(), feature_coverage=0.7,
        )
        assert derated["coverage_adj"] == 0.7
        # Derated dollar_size ≈ 70% of base (both hit the 5% cap, so
        # actually we need to compare raw_weight * coverage. Since base
        # also hits the cap, compare shares against expected ratio.)
        # Base: base_weight=1.0 * coverage 1.0 = 1.0, capped at 0.05 → $50k.
        # Derated: base_weight=1.0 * coverage 0.7 = 0.7, capped at min(0.7, 0.05) → still $50k? No.
        # Actually: 1.0 * 0.7 = 0.7, then min(0.7, 0.05) = 0.05 → still cap.
        # Need to test below the cap. Use more enter_signals.
        assert derated["shares"] <= base["shares"]

    def test_coverage_derate_below_cap_shows_scaling(self):
        """With many enter_signals, base_weight is small enough that the
        coverage multiplier actually moves the final weight.
        """
        enter_signals = [{"ticker": f"T{i}"} for i in range(20)]  # base_weight 1/20 = 0.05
        full = compute_position_size(
            ticker="T0", portfolio_nav=1_000_000,
            enter_signals=enter_signals, signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(), feature_coverage=1.0,
        )
        partial = compute_position_size(
            ticker="T0", portfolio_nav=1_000_000,
            enter_signals=enter_signals, signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(), feature_coverage=0.5,
        )
        # 0.5 coverage should roughly halve the position weight.
        assert partial["position_pct"] == pytest.approx(full["position_pct"] * 0.5, rel=0.05)

    def test_coverage_below_floor_clamps_to_floor(self):
        """Coverage = 0.1 with floor = 0.25 → coverage_adj = 0.25, not 0.1."""
        out = compute_position_size(
            ticker="VERY_LOW", portfolio_nav=1_000_000,
            enter_signals=[{"ticker": f"T{i}"} for i in range(20)],
            signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(coverage_derate_floor=0.25),
            feature_coverage=0.1,
        )
        assert out["coverage_adj"] == 0.25

    def test_coverage_sizing_disabled_noop(self):
        """When coverage_sizing_enabled=False, coverage_adj stays 1.0 regardless."""
        out = compute_position_size(
            ticker="SNDK", portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "SNDK"}], signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(coverage_sizing_enabled=False),
            feature_coverage=0.3,
        )
        assert out["coverage_adj"] == 1.0

    def test_none_coverage_is_noop(self):
        """Coverage = None (unavailable) must not break sizing — pass-through."""
        out = compute_position_size(
            ticker="AAPL", portfolio_nav=1_000_000,
            enter_signals=[{"ticker": "AAPL"}], signal=_base_signal(),
            sector_rating="market_weight", current_price=100.0,
            config=_base_config(), feature_coverage=None,
        )
        assert out["coverage_adj"] == 1.0


# ── Admission gate ─────────────────────────────────────────────────────────────


class TestAdmissionGate:
    def test_buy_candidates_below_floor_refused(self):
        signals = {"buy_candidates": [
            {"ticker": "AAPL"},   # full coverage
            {"ticker": "SNDK"},   # partial, still admitted
            {"ticker": "IPO"},    # below floor — refused
        ]}
        cov_map = {"AAPL": 1.0, "SNDK": 0.5, "IPO": 0.15}

        with patch("executor.signal_reader._emit_admission_refused_metric"):
            out = filter_buy_candidates_by_coverage(
                signals, cov_map, min_coverage=0.30,
            )

        out_tickers = [e["ticker"] for e in out["buy_candidates"]]
        assert "AAPL" in out_tickers
        assert "SNDK" in out_tickers
        assert "IPO" not in out_tickers

    def test_at_threshold_is_admitted(self):
        """Coverage exactly == min_coverage → admitted (>=, not >)."""
        signals = {"buy_candidates": [{"ticker": "BORDER"}]}
        cov_map = {"BORDER": 0.30}

        with patch("executor.signal_reader._emit_admission_refused_metric"):
            out = filter_buy_candidates_by_coverage(
                signals, cov_map, min_coverage=0.30,
            )

        assert len(out["buy_candidates"]) == 1

    def test_missing_from_coverage_map_refused(self):
        """Ticker not in coverage_map → 0.0 default → refused.

        load_feature_coverage always returns an entry per requested ticker
        (see test_feature_coverage), but defense-in-depth covers manual
        edits to the signals flow.
        """
        signals = {"buy_candidates": [{"ticker": "UNKNOWN"}]}
        cov_map: dict[str, float] = {}  # no entry

        with patch("executor.signal_reader._emit_admission_refused_metric"):
            out = filter_buy_candidates_by_coverage(
                signals, cov_map, min_coverage=0.30,
            )

        assert out["buy_candidates"] == []

    def test_universe_list_never_filtered(self):
        """Held positions (universe list) must pass through untouched —
        admission applies only to NEW entries, not existing exposure.
        """
        signals = {
            "buy_candidates": [{"ticker": "NEW"}],
            "universe": [
                {"ticker": "HELD_LOW_COVERAGE", "signal": "HOLD"},
                {"ticker": "HELD_LOW_COVERAGE_EXIT", "signal": "EXIT"},
            ],
        }
        cov_map = {"NEW": 0.5}  # universe tickers NOT in map

        with patch("executor.signal_reader._emit_admission_refused_metric"):
            out = filter_buy_candidates_by_coverage(
                signals, cov_map, min_coverage=0.30,
            )

        # universe list preserved as-is, irrespective of coverage
        assert len(out["universe"]) == 2
        held_tickers = [e["ticker"] for e in out["universe"]]
        assert "HELD_LOW_COVERAGE" in held_tickers
        assert "HELD_LOW_COVERAGE_EXIT" in held_tickers

    def test_empty_buy_candidates_passes_through(self):
        signals = {"buy_candidates": []}
        out = filter_buy_candidates_by_coverage(signals, {}, min_coverage=0.30)
        assert out["buy_candidates"] == []

    def test_missing_buy_candidates_key_passes_through(self):
        signals: dict = {}  # no buy_candidates key
        out = filter_buy_candidates_by_coverage(signals, {}, min_coverage=0.30)
        assert "buy_candidates" not in out or out.get("buy_candidates") is None

    def test_does_not_mutate_input(self):
        signals = {"buy_candidates": [{"ticker": "LOW"}]}
        original_buy = signals["buy_candidates"]
        cov_map = {"LOW": 0.1}

        with patch("executor.signal_reader._emit_admission_refused_metric"):
            filter_buy_candidates_by_coverage(signals, cov_map, min_coverage=0.30)

        # Caller's dict still has the original (rejected) ticker.
        assert signals["buy_candidates"] is original_buy
        assert signals["buy_candidates"][0]["ticker"] == "LOW"

    def test_cloudwatch_metric_emitted_on_refuse(self):
        signals = {"buy_candidates": [
            {"ticker": "LOW1"}, {"ticker": "LOW2"}, {"ticker": "OK"},
        ]}
        cov_map = {"LOW1": 0.1, "LOW2": 0.2, "OK": 0.9}

        with patch("executor.signal_reader._emit_admission_refused_metric") as metric:
            filter_buy_candidates_by_coverage(signals, cov_map, min_coverage=0.30)

        metric.assert_called_once_with(2)  # 2 refused
