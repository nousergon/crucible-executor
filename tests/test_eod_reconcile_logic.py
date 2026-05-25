"""Tests for executor/eod_reconcile.py — testable logic without IB Gateway."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from executor.eod_reconcile import (
    _apply_dividend_delta,
    _compute_unattributed_residual_pct,
    _load_constituents_sector_map,
    _Narrative,
    _RationalesResponse,
    _resolve_prior_price,
    _synthesize_rationales,
)


class TestComputeUnattributedResidualPct:
    """Phase 2 transparency-inventory headline metric: residual P&L
    not attributable to position MTM, interest, or dividends, expressed
    as % of NAV. Inventory gate is ≤1%."""

    def test_typical_small_residual(self):
        # $50 unattributed on $100,000 NAV → 0.05%
        assert _compute_unattributed_residual_pct(50.0, 100_000.0) == pytest.approx(0.05)

    def test_breaches_one_percent_gate(self):
        # $1,500 unattributed on $100,000 NAV → 1.5% > 1% gate
        result = _compute_unattributed_residual_pct(1_500.0, 100_000.0)
        assert result == pytest.approx(1.5)
        assert abs(result) > 1.0  # the alarm condition

    def test_zero_residual_returns_zero(self):
        assert _compute_unattributed_residual_pct(0.0, 100_000.0) == 0.0

    def test_negative_residual_preserves_sign(self):
        # Position pnl + interest exceeded actual NAV change (unaccounted fee)
        assert _compute_unattributed_residual_pct(-105.0, 100_000.0) == pytest.approx(-0.105)

    def test_none_unattributed_returns_none(self):
        """First-ever EOD run has no prior_nav → nav_reconciliation is {}
        → unattributed_usd is None. Persist NULL, not 0 — they mean
        different things."""
        assert _compute_unattributed_residual_pct(None, 100_000.0) is None

    def test_zero_nav_returns_none_not_div_by_zero(self):
        assert _compute_unattributed_residual_pct(50.0, 0.0) is None

    def test_none_nav_returns_none(self):
        assert _compute_unattributed_residual_pct(50.0, None) is None


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


class TestRationalesResponsePydantic:
    """L1248/L2669: Pydantic model that validates the tool-use payload
    returned by Haiku. Validation here makes the parse-failure-mode that
    bare json.loads used to hit (markdown fences, preamble, trailing
    text) structurally impossible — the SDK has already shape-checked
    the tool_use.input before we see it; this re-validates field types."""

    def test_valid_payload(self):
        payload = {"narratives": [{"ticker": "AAPL", "narrative": "x" * 50}]}
        parsed = _RationalesResponse.model_validate(payload)
        assert len(parsed.narratives) == 1
        assert parsed.narratives[0].ticker == "AAPL"

    def test_empty_narratives_list_valid(self):
        # The model permits an empty list — Haiku may emit zero narratives if
        # the contexts list was empty (the caller short-circuits this earlier,
        # but the contract should still accept it).
        parsed = _RationalesResponse.model_validate({"narratives": []})
        assert parsed.narratives == []

    def test_missing_narratives_field_raises(self):
        from pydantic import ValidationError as PydValidationError
        with pytest.raises(PydValidationError):
            _RationalesResponse.model_validate({})

    def test_narrative_missing_ticker_raises(self):
        from pydantic import ValidationError as PydValidationError
        with pytest.raises(PydValidationError):
            _RationalesResponse.model_validate({"narratives": [{"narrative": "no ticker"}]})

    def test_narrative_missing_narrative_raises(self):
        from pydantic import ValidationError as PydValidationError
        with pytest.raises(PydValidationError):
            _RationalesResponse.model_validate({"narratives": [{"ticker": "AAPL"}]})

    def test_narrative_wrong_type_raises(self):
        from pydantic import ValidationError as PydValidationError
        with pytest.raises(PydValidationError):
            _RationalesResponse.model_validate({"narratives": [{"ticker": 123, "narrative": "x"}]})


class TestSynthesizeRationalesToolUse:
    """End-to-end coverage of the Anthropic tool-use path. The Anthropic
    client is fully mocked so no real API call happens; the assertions
    pin (a) the tool/tool_choice wiring (b) Pydantic-validated input
    flows through to the returned dict (c) malformed / missing tool_use
    blocks fall back to the template path."""

    def _make_mock_anthropic(self, tool_use_input: dict | None, *, stop_reason: str = "tool_use", include_text_block: bool = False):
        """Build a MagicMock anthropic module + client + response chain.
        ``tool_use_input=None`` simulates a response with no tool_use block
        (degenerate-mode probe). Otherwise the mocked tool_use block carries
        ``input=tool_use_input``."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        blocks = []
        if include_text_block:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Sure, here are the rationales:"
            blocks.append(text_block)
        if tool_use_input is not None:
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.input = tool_use_input
            blocks.append(tool_block)

        mock_response = MagicMock()
        mock_response.content = blocks
        mock_response.stop_reason = stop_reason
        mock_client.messages.create.return_value = mock_response
        return mock_anthropic, mock_client

    def test_tool_use_happy_path(self):
        mock_anthropic, mock_client = self._make_mock_anthropic(
            {"narratives": [
                {"ticker": "AAPL", "narrative": "Held — research score 82, GBM UP."},
                {"ticker": "MSFT", "narrative": "Reduced 5 shares today on profit-take."},
            ]}
        )
        contexts = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            # llm_enabled=True is now opt-in per the standing rule that
            # LLM calls live in the research module. Default path is
            # template-only; this test exercises the LLM branch.
            result = _synthesize_rationales(contexts, llm_enabled=True)
        assert result == {
            "AAPL": "Held — research score 82, GBM UP.",
            "MSFT": "Reduced 5 shares today on profit-take.",
        }
        # Verify the SDK was invoked with the forced tool_choice wiring.
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "emit_rationales"}
        assert call_kwargs["tools"][0]["name"] == "emit_rationales"

    def test_tool_use_with_preceding_text_block(self):
        # Anthropic permits a text block before the tool_use block — the
        # synthesizer must pick the tool_use block, not the first content block.
        mock_anthropic, _ = self._make_mock_anthropic(
            {"narratives": [{"ticker": "GOOG", "narrative": "y" * 40}]},
            include_text_block=True,
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _synthesize_rationales([{"ticker": "GOOG"}], llm_enabled=True)
        assert result == {"GOOG": "y" * 40}

    def test_missing_tool_use_falls_back_to_template(self):
        # Haiku stopped without emitting the forced tool — template fallback.
        mock_anthropic, _ = self._make_mock_anthropic(None, stop_reason="end_turn", include_text_block=True)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _synthesize_rationales(
                [{"ticker": "AAPL", "research_score": 82.0, "conviction": "rising"}],
                llm_enabled=True,
            )
        # Template fallback populates from the context — research_score should
        # be in the rendered text.
        assert "AAPL" in result
        assert "82" in result["AAPL"]
        assert "rising" in result["AAPL"]

    def test_malformed_tool_input_falls_back_to_template(self):
        # tool_use block present but input doesn't match the Pydantic schema.
        mock_anthropic, _ = self._make_mock_anthropic({"narratives": [{"ticker": "AAPL"}]})  # missing 'narrative'
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _synthesize_rationales(
                [{"ticker": "AAPL", "research_score": 90.0}],
                llm_enabled=True,
            )
        # Template fallback fires; AAPL is still rendered from the context.
        assert "AAPL" in result
        assert "90" in result["AAPL"]

    def test_empty_contexts_short_circuits(self):
        # Empty input never calls the SDK — verify by leaving anthropic
        # unmocked; a real import would still resolve but not be invoked.
        assert _synthesize_rationales([]) == {}


class TestLoadConstituentsSectorMap:
    """Sector enrichment fallback reads latest weekly constituents.json."""

    def _mock_s3(self, keys: list[str], sector_map: dict | None):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": k} for k in keys],
        }
        body = {"sector_map": sector_map} if sector_map is not None else {}
        s3.get_object.return_value = {
            "Body": io.BytesIO(json.dumps(body).encode()),
        }
        return s3

    @patch("executor.eod_reconcile.boto3")
    def test_picks_latest_weekly_snapshot(self, mock_boto3):
        # Lexicographic max of ISO dates == chronological latest
        s3 = self._mock_s3(
            keys=[
                "market_data/weekly/2026-04-04/constituents.json",
                "market_data/weekly/2026-04-18/constituents.json",
                "market_data/weekly/2026-04-11/constituents.json",
            ],
            sector_map={"VRTX": "Health Care", "MSFT": "Information Technology"},
        )
        mock_boto3.client.return_value = s3

        result = _load_constituents_sector_map("alpha-engine-research")

        assert result == {"VRTX": "Health Care", "MSFT": "Information Technology"}
        # Confirms the most recent key was the one fetched
        s3.get_object.assert_called_once_with(
            Bucket="alpha-engine-research",
            Key="market_data/weekly/2026-04-18/constituents.json",
        )

    @patch("executor.eod_reconcile.boto3")
    def test_empty_when_no_snapshots_listed(self, mock_boto3):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": []}
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}
        s3.get_object.assert_not_called()

    @patch("executor.eod_reconcile.boto3")
    def test_empty_on_s3_exception(self, mock_boto3):
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = RuntimeError("boom")
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}

    @patch("executor.eod_reconcile.boto3")
    def test_empty_when_sector_map_missing_from_payload(self, mock_boto3):
        s3 = self._mock_s3(
            keys=["market_data/weekly/2026-04-18/constituents.json"],
            sector_map=None,  # body has no sector_map key
        )
        mock_boto3.client.return_value = s3
        assert _load_constituents_sector_map("bucket") == {}
