"""Tests for batch-confidence-aware min_score tightening.

Per the 2026-05-07 predictor audit's Phase 1: when the predictor batch's
mean prediction_confidence is broadly low (degenerate output pattern),
tighten min_score_to_enter as a backstop. Feature-flagged off by default;
these tests validate the helpers and the integration with decide_entries.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.deciders import (
    _apply_batch_confidence_tightening,
    _batch_confidence_mean,
)


# ── _batch_confidence_mean ────────────────────────────────────────────────

class TestBatchConfidenceMean:

    def test_simple_mean(self):
        preds = {
            "A": {"prediction_confidence": 0.5},
            "B": {"prediction_confidence": 0.7},
            "C": {"prediction_confidence": 0.9},
        }
        assert _batch_confidence_mean(preds) == pytest.approx(0.7)

    def test_skips_missing_confidence_field(self):
        preds = {
            "A": {"prediction_confidence": 0.6},
            "B": {},  # no confidence — skipped
            "C": {"prediction_confidence": 0.8},
        }
        assert _batch_confidence_mean(preds) == pytest.approx(0.7)

    def test_skips_none_confidence(self):
        preds = {
            "A": {"prediction_confidence": 0.6},
            "B": {"prediction_confidence": None},
            "C": {"prediction_confidence": 0.8},
        }
        assert _batch_confidence_mean(preds) == pytest.approx(0.7)

    def test_returns_none_on_empty_dict(self):
        assert _batch_confidence_mean({}) is None

    def test_returns_none_on_none(self):
        assert _batch_confidence_mean(None) is None

    def test_returns_none_when_no_valid_confidences(self):
        preds = {
            "A": {},
            "B": {"prediction_confidence": None},
        }
        assert _batch_confidence_mean(preds) is None

    def test_skips_non_dict_values(self):
        preds = {
            "A": {"prediction_confidence": 0.7},
            "B": "garbage",  # not a dict — skipped
        }
        assert _batch_confidence_mean(preds) == pytest.approx(0.7)


# ── _apply_batch_confidence_tightening ────────────────────────────────────

class TestApplyBatchConfidenceTightening:

    def test_returns_unchanged_when_disabled(self):
        config = {
            "batch_confidence_tightening_enabled": False,
            "min_score_to_enter": 30,
        }
        preds = {"A": {"prediction_confidence": 0.20}}  # very low
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result is config  # same object, no copy
        assert result["min_score_to_enter"] == 30

    def test_returns_unchanged_when_flag_missing(self):
        # Default behavior when the key isn't set in risk.yaml at all.
        config = {"min_score_to_enter": 30}
        preds = {"A": {"prediction_confidence": 0.20}}
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result is config

    def test_does_not_trigger_when_confidence_above_threshold(self):
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        preds = {
            "A": {"prediction_confidence": 0.80},
            "B": {"prediction_confidence": 0.90},
        }
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result is config  # threshold not crossed → no copy

    def test_triggers_when_confidence_below_threshold(self):
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        preds = {
            "A": {"prediction_confidence": 0.55},
            "B": {"prediction_confidence": 0.60},
        }
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result is not config  # different object — copy was made
        assert result["min_score_to_enter"] == 40  # 30 + 10 bump
        # Original unchanged.
        assert config["min_score_to_enter"] == 30

    def test_triggers_at_exact_threshold_boundary(self):
        # Threshold is "< threshold" (strict), so equal does NOT trigger.
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.60,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        preds = {"A": {"prediction_confidence": 0.60}}
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result is config  # no trigger at exact boundary

    def test_records_diagnostic_metadata_on_trigger(self):
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        preds = {"A": {"prediction_confidence": 0.50}}
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        meta = result.get("_batch_confidence_tightening_applied")
        assert meta is not None
        assert meta["mean_confidence"] == 0.50
        assert meta["threshold"] == 0.65
        assert meta["base_min_score"] == 30
        assert meta["tightened_min_score"] == 40

    def test_no_trigger_when_predictions_empty(self):
        # Empty predictions dict → can't compute mean → don't trigger.
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        result = _apply_batch_confidence_tightening(config, {}, "2026-05-07")
        assert result is config

    def test_no_trigger_when_predictions_none(self):
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 30,
        }
        result = _apply_batch_confidence_tightening(config, None, "2026-05-07")
        assert result is config

    def test_uses_yaml_default_min_score_when_unset(self):
        # When config doesn't explicitly set min_score_to_enter, the helper
        # uses the same default the risk_guard does (70).
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.65,
            "batch_confidence_min_score_bump": 10,
        }
        preds = {"A": {"prediction_confidence": 0.50}}
        result = _apply_batch_confidence_tightening(config, preds, "2026-05-07")
        assert result["min_score_to_enter"] == 80  # 70 default + 10 bump


# ── Integration with decide_entries ──────────────────────────────────────

import pandas as pd
from executor.deciders import decide_entries


def _df_history(base: float = 100.0):
    closes = [base * (1 + 0.001 * i) for i in range(40)]
    return pd.DataFrame(
        {"close": closes, "high": [c * 1.01 for c in closes],
         "low": [c * 0.99 for c in closes], "volume": [1_000_000] * 40},
        index=pd.date_range("2026-03-01", periods=40, freq="B"),
    )


def _base_config():
    return {
        "min_score_to_enter": 50,
        "max_position_pct": 0.05,
        "max_sector_pct": 0.30,
        "max_total_equity_pct": 0.95,
        "drawdown_halt_pct": 0.20,
        "atr_sizing_enabled": False,
        "momentum_gate_enabled": False,
        "earnings_proximity_warning_days": 0,
    }


def _signal(ticker: str, score: int):
    return {
        "ticker": ticker,
        "signal": "ENTER",
        "score": score,
        "conviction": "rising",
        "sector": "Technology",
        "rating": "BUY",
        "price_target_upside": 0.15,
        "thesis_summary": "test",
    }


class TestDecideEntriesAppliesTightening:

    def test_low_batch_confidence_blocks_marginal_score_when_enabled(self):
        # Setup: signal with score=55 (clears base min_score=50), low batch
        # confidence (0.50), tightening flag on, bump=10 → tightened to 60 →
        # signal now blocked.
        config = _base_config()
        config["batch_confidence_tightening_enabled"] = True
        config["batch_confidence_threshold"] = 0.65
        config["batch_confidence_min_score_bump"] = 10

        sig = _signal("MARG", 55)
        preds = {"MARG": {"prediction_confidence": 0.50, "predicted_alpha": 0.001,
                          "gbm_veto": False}}

        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": [sig],
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": [sig],
            "buy_candidates": [sig],
        }
        all_tickers = ["MARG"]
        plan = decide_entries(
            enter_signals=[sig],
            signals_raw=signals_raw,
            predictions_by_ticker=preds,
            config=config,
            strategy_config={"vwap_threshold_pct": 1.0,
                             "intraday_support_lookback_days": 20,
                             "drawdown_forced_exit_enabled": False},
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={"MARG": 100.0},
            price_histories={"MARG": _df_history()},
            atr_map={"MARG": 0.02},
            vwap_map={"MARG": 100.0},
            coverage_map={"MARG": 1.0},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )

        # Tightening should have raised min_score 50 → 60; score=55 now fails.
        assert plan.n_entered == 0
        # Block reason should reference the new tightened min_score (60), not 50.
        assert any("60" in (b.get("block_reason") or "") for b in plan.blocked)

    def test_low_batch_confidence_does_nothing_when_disabled(self):
        # Same input as above but with the flag off — signal should clear.
        config = _base_config()
        # Tightening explicitly disabled (mirrors current production default).
        config["batch_confidence_tightening_enabled"] = False

        sig = _signal("MARG", 55)
        preds = {"MARG": {"prediction_confidence": 0.50, "predicted_alpha": 0.001,
                          "gbm_veto": False}}

        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": [sig],
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": [sig],
            "buy_candidates": [sig],
        }
        plan = decide_entries(
            enter_signals=[sig],
            signals_raw=signals_raw,
            predictions_by_ticker=preds,
            config=config,
            strategy_config={"vwap_threshold_pct": 1.0,
                             "intraday_support_lookback_days": 20,
                             "drawdown_forced_exit_enabled": False},
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={"MARG": 100.0},
            price_histories={"MARG": _df_history()},
            atr_map={"MARG": 0.02},
            vwap_map={"MARG": 100.0},
            coverage_map={"MARG": 1.0},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )

        # With flag off, score=55 clears base min_score=50 — entry approved.
        assert plan.n_entered == 1

    def test_does_not_mutate_caller_config(self):
        # Verify the helper makes a copy when triggered, never mutating input.
        config = _base_config()
        config["batch_confidence_tightening_enabled"] = True
        config["batch_confidence_threshold"] = 0.65
        config["batch_confidence_min_score_bump"] = 10

        sig = _signal("MARG", 90)  # well above either threshold
        preds = {"MARG": {"prediction_confidence": 0.50, "predicted_alpha": 0.001,
                          "gbm_veto": False}}

        signals_raw = {
            "date": "2026-05-07",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "enter": [sig],
            "exit": [], "reduce": [], "hold": [],
            "universe": [sig], "buy_candidates": [sig],
        }
        decide_entries(
            enter_signals=[sig],
            signals_raw=signals_raw,
            predictions_by_ticker=preds,
            config=config,
            strategy_config={"vwap_threshold_pct": 1.0,
                             "intraday_support_lookback_days": 20,
                             "drawdown_forced_exit_enabled": False},
            market_regime="neutral",
            sector_ratings=signals_raw["sector_ratings"],
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={},
            prices_now={"MARG": 100.0},
            price_histories={"MARG": _df_history()},
            atr_map={"MARG": 0.02},
            vwap_map={"MARG": 100.0},
            coverage_map={"MARG": 1.0},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2026-05-07",
            predictions_date="2026-05-07",
        )

        # Caller's config must remain unchanged regardless of trigger outcome.
        assert config["min_score_to_enter"] == 50
        assert "_batch_confidence_tightening_applied" not in config
