"""Tests for executor/eod_reconcile.py — testable logic without IB Gateway."""

from unittest.mock import patch

import pytest

from executor.eod_reconcile import (
    _apply_dividend_delta,
    _resolve_prior_price,
    _synthesize_rationales,
)


class TestResolvePriorPrice:
    """Phase 3: prior-day price source resolution."""

    def test_prefers_explicit_closing_price(self):
        prior = {"closing_price": 105.0, "market_value": 500.0, "shares": 10}
        pos = {"avg_cost": 100.0}
        # closing_price wins even though MV/shares would give 50
        assert _resolve_prior_price(prior, pos, current_price=110.0) == 105.0

    def test_falls_back_to_mv_over_shares_for_legacy_snapshot(self):
        # Pre-Phase-3 snapshots have no closing_price
        prior = {"market_value": 1050.0, "shares": 10}
        pos = {"avg_cost": 100.0}
        assert _resolve_prior_price(prior, pos, current_price=110.0) == 105.0

    def test_uses_avg_cost_when_no_prior_snapshot(self):
        # Position opened today — no prior snapshot
        pos = {"avg_cost": 99.50}
        assert _resolve_prior_price(None, pos, current_price=101.0) == 99.50

    def test_falls_back_to_current_price_when_no_avg_cost(self):
        # Degenerate case — position has no avg_cost either
        assert _resolve_prior_price(None, {}, current_price=110.0) == 110.0


class TestApplyDividendDelta:
    """Day-over-day dividend accrual delta is attributed to the position."""

    def test_no_accrual_is_noop(self):
        pos = {"daily_return_usd": 1.5, "daily_return_pct": 0.1}
        _apply_dividend_delta(pos, {"accrued_dividend": 0.0}, prior_price=150.0, shares=10)
        assert pos["daily_return_usd"] == 1.5
        assert "dividend_usd" not in pos

    def test_new_accrual_added(self):
        pos = {"accrued_dividend": 5.0, "daily_return_usd": 2.0, "daily_return_pct": 0.1}
        _apply_dividend_delta(pos, {"accrued_dividend": 0.0}, prior_price=100.0, shares=10)
        assert pos["dividend_usd"] == 5.0
        assert pos["daily_return_usd"] == 7.0
        # prior_mv = 1000, daily_usd = 7 → pct = 0.7%
        assert pos["daily_return_pct"] == pytest.approx(0.7)

    def test_dividend_payout_does_not_touch_position_pnl(self):
        """On payout day, accrual drops to 0 and cash rises by the same amount.

        IB's NetLiquidation is invariant to the payout (accrual↓ = cash↑), so
        position P&L must NOT be reduced. The dividend was already earned on
        the ex-dividend day. Record it in dividend_paid_usd for visibility.
        """
        pos = {"accrued_dividend": 0.0, "daily_return_usd": 2.0, "daily_return_pct": 0.2}
        _apply_dividend_delta(pos, {"accrued_dividend": 5.0}, prior_price=100.0, shares=10)
        assert "dividend_usd" not in pos
        # daily_return_usd unchanged — payout is not a loss
        assert pos["daily_return_usd"] == 2.0
        assert pos["dividend_paid_usd"] == 5.0

    def test_no_prior_snapshot_treats_accrual_as_new(self):
        pos = {"accrued_dividend": 3.0, "daily_return_usd": 1.0}
        _apply_dividend_delta(pos, None, prior_price=100.0, shares=5)
        assert pos["dividend_usd"] == 3.0
        assert pos["daily_return_usd"] == 4.0


class TestSynthesizeRationales:
    """Test the template fallback path of _synthesize_rationales."""

    def test_empty_contexts(self):
        assert _synthesize_rationales([]) == {}

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_fallback_basic(self):
        contexts = [{
            "ticker": "AAPL",
            "entry_date": "2026-04-01",
            "entry_price": 150.0,
            "research_score": 82.0,
            "conviction": "rising",
        }]
        result = _synthesize_rationales(contexts)
        assert "AAPL" in result
        assert "150.00" in result["AAPL"]
        assert "82" in result["AAPL"]
        assert "rising" in result["AAPL"]

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_with_predictor(self):
        contexts = [{
            "ticker": "MSFT",
            "predicted_direction": "UP",
            "prediction_confidence": 0.75,
            "predicted_alpha": 0.025,
        }]
        result = _synthesize_rationales(contexts)
        assert "UP" in result["MSFT"]
        assert "75%" in result["MSFT"]

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_with_thesis(self):
        contexts = [{
            "ticker": "GOOG",
            "thesis_summary": "Strong AI momentum driving cloud revenue growth across enterprise segment.",
        }]
        result = _synthesize_rationales(contexts)
        assert "AI momentum" in result["GOOG"]

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_long_thesis_truncated(self):
        contexts = [{
            "ticker": "AMZN",
            "thesis_summary": "x" * 200,
        }]
        result = _synthesize_rationales(contexts)
        assert len(result["AMZN"]) < 200
        assert result["AMZN"].endswith("...")

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_with_today_actions(self):
        contexts = [{
            "ticker": "NVDA",
            "today_actions": [{"action": "BUY", "shares": 10}],
        }]
        result = _synthesize_rationales(contexts)
        assert "BUY" in result["NVDA"]
        assert "10 shares" in result["NVDA"]

    @patch.dict("sys.modules", {"anthropic": None})
    def test_template_no_data(self):
        contexts = [{"ticker": "TSLA"}]
        result = _synthesize_rationales(contexts)
        assert "No rationale" in result["TSLA"]

    @patch.dict("sys.modules", {"anthropic": None})
    def test_multiple_tickers(self):
        contexts = [
            {"ticker": "AAPL", "research_score": 85},
            {"ticker": "MSFT", "research_score": 72},
        ]
        result = _synthesize_rationales(contexts)
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
