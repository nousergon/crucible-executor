"""Tests for executor/eod_reconcile.py — testable logic without IB Gateway."""

from unittest.mock import patch

import pytest

from executor.eod_reconcile import _synthesize_rationales


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
