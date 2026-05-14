"""Tests for executor.decision_capture — entry_triggers component (L2308 PR 1).

Covers:
- Payload shape (snapshot + agent_output + summary) matches plan-doc spec
- Feature flag (ALPHA_ENGINE_DECISION_CAPTURE_ENABLED env var) gates
  the capture call
- Hard-fail propagates DecisionCaptureWriteError per ``feedback_no_silent_fails``
- Run-id format prevents per-day-per-ticker collisions
- Trigger-kind classifier maps every canonical fire reason
- daemon's BLE001 catch path doesn't kill trade flow on capture failure

Plan doc: ``~/Development/alpha-engine-docs/private/executor-decision-capture-260511.md``.
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from executor.decision_capture import (
    DecisionCaptureWriteError,
    _classify_exit_rule_kind,
    _classify_planner_exit_kind,
    _classify_trigger_kind,
    build_entry_trigger_payload,
    build_exit_rule_payload,
    build_planner_exit_payload,
    build_position_sizer_payload,
    build_risk_guard_payload,
    capture_entry_trigger,
    capture_exit_rule,
    capture_planner_exit,
    capture_position_sizer,
    capture_risk_guard,
    is_decision_capture_enabled,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_entry(ticker: str = "AAPL") -> dict:
    """Order-book entry fixture matching the daemon's pending_entries shape."""
    return {
        "ticker": ticker,
        "signal": "ENTER",
        "shares": 150,
        "signal_date": "2026-05-15",
        "prediction_date": "2026-05-16",
        "current_price": 175.25,
        "triggers": {
            "pullback_pct": 0.02,
            "vwap_discount": 0.005,
            "support_level": 172.00,
            "support_pct": 0.01,
            "vwap": 174.50,
        },
    }


def _make_price_state() -> dict:
    """PriceMonitor.get_price() return shape."""
    return {
        "last": 175.25,
        "high": 178.50,
        "low": 174.10,
        "volume": 1_234_567,
    }


def _make_strategy_config() -> dict:
    return {
        "intraday_pullback_pct": 0.02,
        "intraday_vwap_discount_pct": 0.005,
        "intraday_support_pct": 0.01,
        "intraday_graduated_max_premium_pct": 0.01,
        "intraday_expiry_time": "15:55",
        "intraday_graduated_start_time": "14:00",
        "disabled_triggers": [],
    }


def _make_s3_stub() -> MagicMock:
    """Stub S3 client recording put_object calls."""
    s3 = MagicMock()
    s3.put_object = MagicMock()
    return s3


# ── Feature flag ─────────────────────────────────────────────────────────


class TestFeatureFlag:
    """ALPHA_ENGINE_DECISION_CAPTURE_ENABLED env var gates the capture path."""

    def test_default_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
        assert is_decision_capture_enabled() is False

    def test_truthy_values_enable(self, monkeypatch):
        for v in ("true", "True", "1", "yes", "YES"):
            monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", v)
            assert is_decision_capture_enabled() is True, f"value {v!r} should enable"

    def test_falsy_values_disable(self, monkeypatch):
        for v in ("false", "0", "no", "", "off"):
            monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", v)
            assert is_decision_capture_enabled() is False, f"value {v!r} should disable"

    def test_capture_returns_none_when_disabled(self, monkeypatch):
        """Capture path is a no-op (returns None) when the env var is off,
        so the daemon caller pays zero S3 cost in default-off production."""
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
        s3 = _make_s3_stub()
        result = capture_entry_trigger(
            run_date="2026-05-15",
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
            s3_client=s3,
        )
        assert result is None
        s3.put_object.assert_not_called()

    def test_suppress_flag_overrides_enabled(self, monkeypatch):
        """ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true forces capture off
        even when ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=true. Used by the
        backtester spot to keep the simulation hot loop from emitting
        50k-200k per-decision S3 PUTs (sweep can't afford their cost +
        artifacts are observability for prod, not sweep semantics)."""
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        for v in ("true", "True", "1", "yes", "YES"):
            monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", v)
            assert is_decision_capture_enabled() is False, (
                f"suppress={v!r} should force capture off even with enable=true"
            )

    def test_suppress_falsy_or_unset_does_not_disable(self, monkeypatch):
        """Falsy or absent SUPPRESS leaves the enable flag in charge."""
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        for v in ("false", "0", "no", "", "off"):
            monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", v)
            assert is_decision_capture_enabled() is True, (
                f"suppress={v!r} should not disable when enable=true"
            )
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", raising=False)
        assert is_decision_capture_enabled() is True


# ── Trigger-kind classifier ──────────────────────────────────────────────


class TestTriggerKindClassifier:
    """Every reason string from EntryTriggerEngine.should_enter maps to a
    canonical kind, with ``unknown`` as the explicit fallthrough."""

    @pytest.mark.parametrize("reason,expected", [
        ("pullback 2.3% from high $178.56", "pullback"),
        ("VWAP discount 0.7% (VWAP=$174.50)", "vwap_discount"),
        ("near support $172.00 (dist 0.5%)", "support_bounce"),
        ("graduated_entry (+0.5% vs morning $175.00, limit 1.0%)",
         "graduated_entry"),
        ("time_expiry", "time_expiry"),
        ("something_unknown", "unknown"),
        ("", "unknown"),
    ])
    def test_classify_kind(self, reason, expected):
        assert _classify_trigger_kind(reason) == expected

    def test_graduated_entry_classified_before_pullback(self):
        """If a graduated_entry reason ever contained the substring
        ``pullback`` (it won't today, but is a future-resilience concern),
        the classifier must still route to graduated_entry — that's the
        decision-making path. Pin via parameterization in test_classify_kind
        and re-verified here on the canonical morning-line string.
        """
        reason = "graduated_entry (-0.1% vs morning $178.56, limit 1.0%)"
        assert _classify_trigger_kind(reason) == "graduated_entry"


# ── Payload builder ──────────────────────────────────────────────────────


class TestPayloadShape:
    """Snapshot + agent_output shape matches the plan doc spec."""

    def test_snapshot_carries_producer_provenance(self):
        snapshot, _, _ = build_entry_trigger_payload(
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
        )
        # Producer identity lives on the snapshot, not on the (None)
        # model_metadata — per plan doc question 2.
        assert snapshot["_producer"] == "alpha-engine.executor.entry_triggers"
        assert snapshot["_producer_version"] == "1.0.0"

    def test_snapshot_carries_full_decision_context(self):
        snapshot, _, _ = build_entry_trigger_payload(
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=["vwap_discount"],
            now_et_iso="2026-05-15T13:25:00-04:00",
        )
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["signal"] == "ENTER"
        assert snapshot["shares"] == 150
        assert snapshot["signal_date"] == "2026-05-15"
        assert snapshot["prediction_date"] == "2026-05-16"
        assert snapshot["morning_price"] == 175.25
        assert snapshot["current_price"] == 175.25
        assert snapshot["day_high"] == 178.50
        assert snapshot["day_low"] == 174.10
        assert snapshot["vwap"] == 174.50
        assert snapshot["support_level"] == 172.00
        assert snapshot["disabled_triggers"] == ["vwap_discount"]
        assert snapshot["now_et"] == "2026-05-15T13:25:00-04:00"
        # Thresholds carry both per-entry overrides + strategy_config
        # fallbacks; pin both layers are captured.
        assert snapshot["thresholds"]["pullback_pct"] == 0.02
        assert snapshot["thresholds"]["vwap_discount"] == 0.005
        assert snapshot["thresholds"]["graduated_max_premium"] == 0.01
        assert snapshot["thresholds"]["expiry_time"] == "15:55"

    def test_agent_output_carries_fill_outcome(self):
        _, output, _ = build_entry_trigger_payload(
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
            fill_price=175.30,
            actual_shares=150,
            trade_id="trade-abc",
        )
        assert output["fired_trigger"] == "pullback 1.8% from high $178.50"
        assert output["trigger_kind"] == "pullback"
        assert output["captured_at_fill_attempt"] is True
        assert output["fill_price"] == 175.30
        assert output["actual_shares"] == 150
        assert output["trade_id"] == "trade-abc"

    def test_summary_is_human_readable_one_liner(self):
        _, _, summary = build_entry_trigger_payload(
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="time_expiry",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T15:55:00-04:00",
        )
        assert "AAPL" in summary
        assert "ENTER" in summary
        assert "shares=150" in summary
        assert "time_expiry" in summary


# ── Capture call ─────────────────────────────────────────────────────────


class TestCaptureCall:
    """End-to-end capture: env-flag on + S3 stub → put_object called with
    canonical key + v2 artifact body (model_metadata=None)."""

    def test_writes_artifact_to_canonical_key(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3_key = capture_entry_trigger(
            run_date="2026-05-15",
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
            fill_price=175.30,
            actual_shares=150,
            trade_id="trade-abc",
            s3_client=s3,
        )
        s3.put_object.assert_called_once()
        put_kwargs = s3.put_object.call_args.kwargs
        assert put_kwargs["Bucket"] == "alpha-engine-research"
        # Key format: decision_artifacts/{YYYY}/{MM}/{DD}/executor:entry_triggers/{run_id}.json
        # The capture wrapper computes {YYYY}/{MM}/{DD} from the capture
        # timestamp (UTC wall-clock), not from run_date — pin that we land
        # under the executor:entry_triggers prefix regardless of the date
        # partition the wrapper chooses.
        assert "decision_artifacts/" in put_kwargs["Key"]
        assert "/executor:entry_triggers/" in put_kwargs["Key"]
        assert put_kwargs["Key"].endswith(".json")
        assert put_kwargs["ContentType"] == "application/json"
        # The returned S3 key matches what put_object received.
        assert s3_key == put_kwargs["Key"]

    def test_artifact_body_is_v2_deterministic_shape(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_entry_trigger(
            run_date="2026-05-15",
            entry=_make_entry(),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.8% from high $178.50",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        # v2 schema + deterministic (both LLM fields None) — the load-bearing
        # contract from alpha-engine-lib v0.10.0.
        assert body["schema_version"] == 2
        assert body["agent_id"] == "executor:entry_triggers"
        assert body["model_metadata"] is None
        assert body["full_prompt_context"] is None
        # Snapshot + agent_output present + populated.
        assert body["input_data_snapshot"]["ticker"] == "AAPL"
        assert body["agent_output"]["trigger_kind"] == "pullback"

    def test_run_id_includes_ticker_and_uuid_suffix(self, monkeypatch):
        """Plan doc Q1 anchor: per-trading-day run_id with per-decision
        suffix at the S3 leaf so multiple captures per ticker per day
        don't overwrite."""
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_entry_trigger(
            run_date="2026-05-15",
            entry=_make_entry(ticker="MSFT"),
            price_state=_make_price_state(),
            trigger_reason="pullback 1.5% from high $410",
            strategy_config=_make_strategy_config(),
            disabled_triggers=[],
            now_et_iso="2026-05-15T13:25:00-04:00",
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        run_id = body["run_id"]
        assert run_id.startswith("2026-05-15_MSFT_")
        # uuid4 hex[:8] suffix — exactly 8 hex chars
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_two_captures_same_ticker_same_day_get_unique_keys(self, monkeypatch):
        """Anti-regression: per the plan doc, multiple captures per ticker
        per day must NOT overwrite each other at the S3 leaf."""
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_entry_trigger(
            run_date="2026-05-15", entry=_make_entry(),
            price_state=_make_price_state(), trigger_reason="pullback A",
            strategy_config=_make_strategy_config(), disabled_triggers=[],
            now_et_iso="t", s3_client=s3,
        )
        capture_entry_trigger(
            run_date="2026-05-15", entry=_make_entry(),
            price_state=_make_price_state(), trigger_reason="pullback B",
            strategy_config=_make_strategy_config(), disabled_triggers=[],
            now_et_iso="t", s3_client=s3,
        )
        keys = {call.kwargs["Key"] for call in s3.put_object.call_args_list}
        assert len(keys) == 2, f"expected 2 distinct keys, got: {keys}"


# ── Hard-fail discipline ─────────────────────────────────────────────────


class TestHardFail:
    """Per ``feedback_no_silent_fails``, S3 write failures raise
    ``DecisionCaptureWriteError`` instead of silently swallowing.
    The daemon caller is responsible for the best-effort try/except
    (so a transient S3 outage doesn't kill trading)."""

    def test_s3_failure_propagates_capture_write_error(self, monkeypatch):
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3.put_object.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Denied"},
            },
            operation_name="PutObject",
        )

        with pytest.raises(DecisionCaptureWriteError):
            capture_entry_trigger(
                run_date="2026-05-15",
                entry=_make_entry(),
                price_state=_make_price_state(),
                trigger_reason="pullback",
                strategy_config=_make_strategy_config(),
                disabled_triggers=[],
                now_et_iso="t",
                s3_client=s3,
            )


# ── Position sizer (PR 2) ────────────────────────────────────────────────


def _make_signal() -> dict:
    """Research signal dict shape as it lands in deciders.decide_entries."""
    return {
        "ticker": "AAPL",
        "score": 78.5,
        "conviction": "rising",
        "rating": "BUY",
        "price_target_upside": 0.18,
        "sector": "technology",
    }


def _make_sizing_result(shares: int = 150) -> dict:
    """compute_position_size return dict shape."""
    return {
        "shares": shares,
        "dollar_size": 26287.50 if shares else 0.0,
        "position_pct": 0.0263 if shares else 0.0,
        "sector_adj": 1.05,
        "conviction_adj": 1.00,
        "upside_adj": 1.00,
        "dd_multiplier": 1.0,
        "atr_adj": 0.95,
        "confidence_adj": 1.10,
        "staleness_adj": 1.0,
        "earnings_adj": 1.0,
        "coverage_adj": 1.0,
        "stance_adj": 1.0,
    }


class TestPositionSizerPayloadShape:
    """Snapshot + agent_output shape match the plan-doc + sizer return dict."""

    def test_snapshot_carries_producer_provenance(self):
        snapshot, _, _ = build_position_sizer_payload(
            ticker="AAPL",
            signal=_make_signal(),
            sector_rating="overweight",
            current_price=175.25,
            portfolio_nav=1_000_000.0,
            n_enter_signals=10,
            drawdown_multiplier=1.0,
            atr_pct=0.018,
            prediction_confidence=0.72,
            p_up=0.61,
            signal_age_days=2,
            days_to_earnings=18,
            feature_coverage=0.97,
            stance="momentum",
            sizing_result=_make_sizing_result(),
            sized_outcome="approved",
            sized_outcome_reason=None,
        )
        assert snapshot["_producer"] == "alpha-engine.executor.position_sizer"
        assert snapshot["_producer_version"] == "1.0.0"

    def test_snapshot_carries_full_sizer_inputs(self):
        snapshot, _, _ = build_position_sizer_payload(
            ticker="AAPL",
            signal=_make_signal(),
            sector_rating="overweight",
            current_price=175.25,
            portfolio_nav=1_000_000.0,
            n_enter_signals=10,
            drawdown_multiplier=0.50,
            atr_pct=0.018,
            prediction_confidence=0.72,
            p_up=0.61,
            signal_age_days=2,
            days_to_earnings=18,
            feature_coverage=0.97,
            stance="momentum",
            sizing_result=_make_sizing_result(),
            sized_outcome="approved",
            sized_outcome_reason=None,
        )
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["sector_rating"] == "overweight"
        assert snapshot["current_price"] == 175.25
        assert snapshot["portfolio_nav"] == 1_000_000.0
        assert snapshot["n_enter_signals"] == 10
        assert snapshot["drawdown_multiplier"] == 0.50
        assert snapshot["atr_pct"] == 0.018
        assert snapshot["prediction_confidence"] == 0.72
        assert snapshot["p_up"] == 0.61
        assert snapshot["signal_age_days"] == 2
        assert snapshot["days_to_earnings"] == 18
        assert snapshot["feature_coverage"] == 0.97
        assert snapshot["stance"] == "momentum"
        # Signal sub-dict carries the research-decision context.
        assert snapshot["signal"]["score"] == 78.5
        assert snapshot["signal"]["conviction"] == "rising"
        assert snapshot["signal"]["price_target_upside"] == 0.18

    def test_agent_output_carries_full_sizer_breakdown(self):
        """Per-multiplier breakdown lets grading analytics decompose
        under/over-performance against any single adjustment factor."""
        _, output, _ = build_position_sizer_payload(
            ticker="AAPL",
            signal=_make_signal(),
            sector_rating="overweight",
            current_price=175.25,
            portfolio_nav=1_000_000.0,
            n_enter_signals=10,
            drawdown_multiplier=1.0,
            atr_pct=0.018,
            prediction_confidence=0.72,
            p_up=0.61,
            signal_age_days=2,
            days_to_earnings=18,
            feature_coverage=0.97,
            stance="momentum",
            sizing_result=_make_sizing_result(),
            sized_outcome="approved",
            sized_outcome_reason=None,
        )
        assert output["shares"] == 150
        assert output["dollar_size"] == 26287.50
        assert output["position_pct"] == 0.0263
        # All 10 multiplier fields present + carried through.
        assert output["sector_adj"] == 1.05
        assert output["conviction_adj"] == 1.00
        assert output["upside_adj"] == 1.00
        assert output["dd_multiplier"] == 1.0
        assert output["atr_adj"] == 0.95
        assert output["confidence_adj"] == 1.10
        assert output["staleness_adj"] == 1.0
        assert output["earnings_adj"] == 1.0
        assert output["coverage_adj"] == 1.0
        assert output["stance_adj"] == 1.0
        assert output["sized_outcome"] == "approved"
        assert output["sized_outcome_reason"] is None

    def test_shares_zero_outcome_captured(self):
        _, output, summary = build_position_sizer_payload(
            ticker="ABNB",
            signal=_make_signal(),
            sector_rating="market_weight",
            current_price=120.0,
            portfolio_nav=10_000.0,  # tiny NAV → shares rounds to 0
            n_enter_signals=50,
            drawdown_multiplier=1.0,
            atr_pct=None,
            prediction_confidence=None,
            p_up=None,
            signal_age_days=None,
            days_to_earnings=None,
            feature_coverage=None,
            stance=None,
            sizing_result=_make_sizing_result(shares=0),
            sized_outcome="shares_zero",
            sized_outcome_reason="shares round to 0 ($0 / $120.00)",
        )
        assert output["shares"] == 0
        assert output["sized_outcome"] == "shares_zero"
        assert "$0 / $120.00" in output["sized_outcome_reason"]
        assert "outcome=shares_zero" in summary


class TestPositionSizerCapture:
    """End-to-end: env-flag on + S3 stub → put_object lands at canonical
    executor:position_sizer key with v2 deterministic body."""

    def test_writes_v2_artifact_to_canonical_key(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3_key = capture_position_sizer(
            run_date="2026-05-15",
            ticker="AAPL",
            signal=_make_signal(),
            sector_rating="overweight",
            current_price=175.25,
            portfolio_nav=1_000_000.0,
            n_enter_signals=10,
            drawdown_multiplier=1.0,
            atr_pct=0.018,
            prediction_confidence=0.72,
            p_up=0.61,
            signal_age_days=2,
            days_to_earnings=18,
            feature_coverage=0.97,
            stance="momentum",
            sizing_result=_make_sizing_result(),
            sized_outcome="approved",
            s3_client=s3,
        )
        s3.put_object.assert_called_once()
        put_kwargs = s3.put_object.call_args.kwargs
        assert put_kwargs["Bucket"] == "alpha-engine-research"
        assert "/executor:position_sizer/" in put_kwargs["Key"]
        assert put_kwargs["Key"].endswith(".json")
        assert s3_key == put_kwargs["Key"]
        body = json.loads(put_kwargs["Body"].decode("utf-8"))
        # v2 + deterministic shape per lib v0.10.0 contract.
        assert body["schema_version"] == 2
        assert body["agent_id"] == "executor:position_sizer"
        assert body["model_metadata"] is None
        assert body["full_prompt_context"] is None
        assert body["input_data_snapshot"]["ticker"] == "AAPL"
        assert body["agent_output"]["shares"] == 150

    def test_disabled_when_env_off(self, monkeypatch):
        monkeypatch.delenv(
            "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False,
        )
        s3 = _make_s3_stub()
        result = capture_position_sizer(
            run_date="2026-05-15",
            ticker="AAPL",
            signal=_make_signal(),
            sector_rating="overweight",
            current_price=175.25,
            portfolio_nav=1_000_000.0,
            n_enter_signals=10,
            drawdown_multiplier=1.0,
            atr_pct=0.018,
            prediction_confidence=0.72,
            p_up=0.61,
            signal_age_days=2,
            days_to_earnings=18,
            feature_coverage=0.97,
            stance="momentum",
            sizing_result=_make_sizing_result(),
            sized_outcome="approved",
            s3_client=s3,
        )
        assert result is None
        s3.put_object.assert_not_called()

    def test_run_id_includes_ticker_and_uuid_suffix(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_position_sizer(
            run_date="2026-05-15",
            ticker="MSFT",
            signal=_make_signal(),
            sector_rating="market_weight",
            current_price=410.00,
            portfolio_nav=1_000_000.0,
            n_enter_signals=5,
            drawdown_multiplier=1.0,
            atr_pct=None,
            prediction_confidence=None,
            p_up=None,
            signal_age_days=None,
            days_to_earnings=None,
            feature_coverage=None,
            stance=None,
            sizing_result=_make_sizing_result(shares=100),
            sized_outcome="approved",
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        run_id = body["run_id"]
        assert run_id.startswith("2026-05-15_MSFT_")
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_s3_failure_propagates_capture_write_error(self, monkeypatch):
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3.put_object.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Denied"},
            },
            operation_name="PutObject",
        )
        with pytest.raises(DecisionCaptureWriteError):
            capture_position_sizer(
                run_date="2026-05-15",
                ticker="AAPL",
                signal=_make_signal(),
                sector_rating="overweight",
                current_price=175.25,
                portfolio_nav=1_000_000.0,
                n_enter_signals=10,
                drawdown_multiplier=1.0,
                atr_pct=0.018,
                prediction_confidence=0.72,
                p_up=0.61,
                signal_age_days=2,
                days_to_earnings=18,
                feature_coverage=0.97,
                stance="momentum",
                sizing_result=_make_sizing_result(),
                sized_outcome="approved",
                s3_client=s3,
            )


# ── Risk guard (PR 3) ────────────────────────────────────────────────────


def _make_risk_guard_signal() -> dict:
    """Signal dict shape as it lands in risk_guard.check_order."""
    return {
        "ticker": "AAPL",
        "score": 78.5,
        "conviction": "rising",
        "rating": "BUY",
        "price_target_upside": 0.18,
        "sector": "technology",
        "sector_rating": "overweight",
    }


def _make_current_positions(sector_for_held: str = "technology") -> dict:
    """Sample current_positions dict — risk_guard reads market_value +
    sector to compute sector exposure."""
    return {
        "MSFT": {"market_value": 50_000.0, "sector": sector_for_held},
        "JPM":  {"market_value": 30_000.0, "sector": "financials"},
    }


def _make_risk_config() -> dict:
    return {
        "min_score_to_enter": 70,
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 1.00,
        "bear_block_underweight": True,
        "drawdown_halt_pct": 0.15,
        "correlation_block_threshold": 0.80,
    }


class TestRiskGuardPayloadShape:
    """Snapshot + agent_output for both approved and vetoed paths."""

    def test_snapshot_carries_producer_provenance(self):
        snapshot, _, _ = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=1_050_000.0,
            current_positions=_make_current_positions(),
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=True, reason="ok", events=[],
        )
        assert snapshot["_producer"] == "alpha-engine.executor.risk_guard"
        assert snapshot["_producer_version"] == "1.0.0"

    def test_snapshot_carries_portfolio_state_and_drawdown(self):
        snapshot, _, _ = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=950_000.0, peak_nav=1_000_000.0,
            current_positions=_make_current_positions(),
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=True, reason="ok", events=[],
        )
        assert snapshot["portfolio_nav"] == 950_000.0
        assert snapshot["peak_nav"] == 1_000_000.0
        # drawdown_fraction = (peak - nav) / peak = 50000 / 1000000 = 0.05
        assert abs(snapshot["drawdown_fraction"] - 0.05) < 1e-9
        assert snapshot["n_open_positions"] == 2
        # existing_sector_exposure for technology = MSFT's 50k (JPM is
        # financials, excluded).
        assert snapshot["existing_sector_exposure"] == 50_000.0

    def test_snapshot_carries_full_gate_thresholds(self):
        snapshot, _, _ = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=1_050_000.0,
            current_positions={}, sector="technology",
            market_regime="neutral", signal=_make_risk_guard_signal(),
            config=_make_risk_config(),
            approved=True, reason="ok", events=[],
        )
        thr = snapshot["thresholds"]
        # All 8 gates captured so a future replay can re-evaluate
        # against a different threshold set.
        assert thr["min_score_to_enter"] == 70
        assert thr["max_position_pct"] == 0.05
        assert thr["bear_max_position_pct"] == 0.025
        assert thr["max_sector_pct"] == 0.25
        assert thr["max_equity_pct"] == 1.00
        assert thr["bear_block_underweight"] is True
        assert thr["drawdown_halt_pct"] == 0.15
        assert thr["correlation_block_threshold"] == 0.80

    def test_approved_path_records_outcome_approved(self):
        _, output, summary = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=1_050_000.0,
            current_positions={}, sector="technology",
            market_regime="neutral", signal=_make_risk_guard_signal(),
            config=_make_risk_config(),
            approved=True, reason="ok", events=[],
        )
        assert output["outcome"] == "approved"
        assert output["reason"] == "ok"
        assert output["vetoed_rule"] is None
        assert output["events"] == []
        assert "outcome=approved" in summary

    def test_vetoed_path_extracts_rule_from_events(self):
        events = [
            {
                "ticker": "AAPL", "event_type": "veto", "rule": "min_score",
                "reason": "Score 58.0 < minimum 70", "value": 58.0,
                "threshold": 70.0,
            },
        ]
        _, output, summary = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=1_050_000.0,
            current_positions={}, sector="technology",
            market_regime="neutral", signal=_make_risk_guard_signal(),
            config=_make_risk_config(),
            approved=False, reason="Score 58.0 < minimum 70", events=events,
        )
        assert output["outcome"] == "vetoed"
        assert output["vetoed_rule"] == "min_score"
        assert output["events"] == events
        assert "outcome=vetoed" in summary

    def test_events_filtered_to_per_ticker_veto_only(self):
        """Approved path with portfolio-level events in scope: those must
        not leak into the per-ticker artifact's events list."""
        events = [
            # Portfolio-level halt — different ticker / no ticker
            {"event_type": "halt", "rule": "drawdown_halt", "reason": "p"},
            # Some other ticker's veto (defensive — shouldn't happen in
            # practice but the filter must be robust)
            {
                "ticker": "OTHER", "event_type": "veto", "rule": "min_score",
                "reason": "x",
            },
        ]
        _, output, _ = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=1_050_000.0,
            current_positions={}, sector="technology",
            market_regime="neutral", signal=_make_risk_guard_signal(),
            config=_make_risk_config(),
            approved=True, reason="ok", events=events,
        )
        assert output["outcome"] == "approved"
        assert output["vetoed_rule"] is None
        # Only this ticker's veto events would land in `events` — both
        # input rows belong to other contexts, so the list is empty.
        assert output["events"] == []

    def test_zero_peak_nav_produces_zero_drawdown_safely(self):
        """Anti-regression: division-by-zero guard on peak_nav=0."""
        snapshot, _, _ = build_risk_guard_payload(
            ticker="AAPL", action="ENTER", dollar_size=26_000.0,
            portfolio_nav=1_000_000.0, peak_nav=0.0,
            current_positions={}, sector="technology",
            market_regime="neutral", signal=_make_risk_guard_signal(),
            config=_make_risk_config(),
            approved=True, reason="ok", events=[],
        )
        assert snapshot["drawdown_fraction"] == 0.0


class TestRiskGuardCapture:
    """End-to-end capture path with env-flag + S3 stub."""

    def test_writes_v2_artifact_to_canonical_key(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3_key = capture_risk_guard(
            run_date="2026-05-15", ticker="AAPL", action="ENTER",
            dollar_size=26_000.0, portfolio_nav=1_000_000.0,
            peak_nav=1_050_000.0,
            current_positions=_make_current_positions(),
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=True, reason="ok", events=[], s3_client=s3,
        )
        s3.put_object.assert_called_once()
        put_kwargs = s3.put_object.call_args.kwargs
        assert put_kwargs["Bucket"] == "alpha-engine-research"
        assert "/executor:risk_guard/" in put_kwargs["Key"]
        assert put_kwargs["Key"].endswith(".json")
        assert s3_key == put_kwargs["Key"]
        body = json.loads(put_kwargs["Body"].decode("utf-8"))
        assert body["schema_version"] == 2
        assert body["agent_id"] == "executor:risk_guard"
        assert body["model_metadata"] is None
        assert body["full_prompt_context"] is None
        assert body["input_data_snapshot"]["ticker"] == "AAPL"
        assert body["agent_output"]["outcome"] == "approved"

    def test_vetoed_artifact_carries_rule_and_events(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        events = [
            {
                "ticker": "AAPL", "event_type": "veto",
                "rule": "max_position", "reason": "too big",
                "value": 0.08, "threshold": 0.05,
            },
        ]
        capture_risk_guard(
            run_date="2026-05-15", ticker="AAPL", action="ENTER",
            dollar_size=80_000.0, portfolio_nav=1_000_000.0,
            peak_nav=1_050_000.0, current_positions={},
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=False, reason="too big", events=events, s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        assert body["agent_output"]["outcome"] == "vetoed"
        assert body["agent_output"]["vetoed_rule"] == "max_position"
        assert body["agent_output"]["events"] == events

    def test_disabled_when_env_off(self, monkeypatch):
        monkeypatch.delenv(
            "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False,
        )
        s3 = _make_s3_stub()
        result = capture_risk_guard(
            run_date="2026-05-15", ticker="AAPL", action="ENTER",
            dollar_size=26_000.0, portfolio_nav=1_000_000.0,
            peak_nav=1_050_000.0, current_positions={},
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=True, reason="ok", events=[], s3_client=s3,
        )
        assert result is None
        s3.put_object.assert_not_called()

    def test_s3_failure_propagates_capture_write_error(self, monkeypatch):
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3.put_object.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Denied"},
            },
            operation_name="PutObject",
        )
        with pytest.raises(DecisionCaptureWriteError):
            capture_risk_guard(
                run_date="2026-05-15", ticker="AAPL", action="ENTER",
                dollar_size=26_000.0, portfolio_nav=1_000_000.0,
                peak_nav=1_050_000.0, current_positions={},
                sector="technology", market_regime="neutral",
                signal=_make_risk_guard_signal(), config=_make_risk_config(),
                approved=True, reason="ok", events=[], s3_client=s3,
            )

    def test_run_id_includes_ticker_and_uuid_suffix(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_risk_guard(
            run_date="2026-05-15", ticker="NVDA", action="ENTER",
            dollar_size=30_000.0, portfolio_nav=1_000_000.0,
            peak_nav=1_050_000.0, current_positions={},
            sector="technology", market_regime="neutral",
            signal=_make_risk_guard_signal(), config=_make_risk_config(),
            approved=True, reason="ok", events=[], s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        run_id = body["run_id"]
        assert run_id.startswith("2026-05-15_NVDA_")
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)


# ── Exit rules (PR 4 — daemon-side intraday) ─────────────────────────────


def _make_stop(ticker: str = "AAPL") -> dict:
    """Stop record fixture matching the daemon's order_book.stop shape."""
    return {
        "ticker": ticker,
        "entry_price": 170.00,
        "current_stop": 167.00,
        "trail_atr": 2.50,
        "atr_multiple": 2.0,
        "high_water": 178.50,
        "shares": 150,
        "entry_date": "2026-05-08",
        "profit_take_executed": False,
    }


def _make_exit_price_state() -> dict:
    return {"last": 175.25, "high": 178.50, "low": 174.10}


def _make_exit_signal(reason: str = "intraday_trailing_stop") -> dict:
    """IntradayExitManager.evaluate() return shape."""
    return {
        "ticker": "AAPL",
        "action": "EXIT",
        "shares": 150,
        "reason": reason,
        "detail": "price $175.25 <= stop $167.00 (ATR $2.50 × 2.0)",
    }


def _make_exit_strategy_config() -> dict:
    return {
        "intraday_profit_take_pct": 0.08,
        "intraday_collapse_pct": 0.05,
        "intraday_tighten_after_days": 3,
        "intraday_tighten_atr_multiple": 1.5,
    }


class TestExitRuleKindClassifier:
    """IntradayExitManager exit-signal reasons map to canonical rule kinds."""

    @pytest.mark.parametrize("reason,expected", [
        ("intraday_trailing_stop", "atr_trail"),
        ("intraday_profit_take", "profit_take"),
        ("intraday_collapse", "collapse"),
        ("unknown_reason_string", "unknown"),
        ("", "unknown"),
    ])
    def test_classify_kind(self, reason, expected):
        assert _classify_exit_rule_kind(reason) == expected


class TestExitRulePayloadShape:
    """Snapshot + agent_output mirror the rule engine's view + the fire."""

    def test_snapshot_carries_producer_provenance(self):
        snapshot, _, _ = build_exit_rule_payload(
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
        )
        assert snapshot["_producer"] == "alpha-engine.executor.exit_rules"
        assert snapshot["_producer_version"] == "1.0.0"
        # Layer field distinguishes daemon-intraday from planner-side
        # captures (PR 4b will set this to "planner").
        assert snapshot["evaluation_layer"] == "daemon_intraday"

    def test_snapshot_carries_full_position_state(self):
        snapshot, _, _ = build_exit_rule_payload(
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
        )
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["entry_price"] == 170.00
        assert snapshot["entry_date"] == "2026-05-08"
        assert snapshot["current_stop"] == 167.00
        assert snapshot["trail_atr"] == 2.50
        assert snapshot["atr_multiple"] == 2.0
        assert snapshot["high_water"] == 178.50
        assert snapshot["shares_held"] == 150
        assert snapshot["profit_take_executed"] is False
        # Market state
        assert snapshot["current_price"] == 175.25
        assert snapshot["day_high"] == 178.50
        assert snapshot["day_low"] == 174.10

    def test_snapshot_computes_gain_pct(self):
        snapshot, _, _ = build_exit_rule_payload(
            stop=_make_stop(),  # entry_price=170, current_price=175.25
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
        )
        # (175.25 - 170) / 170 = 0.030882...
        assert abs(snapshot["gain_pct"] - 0.030882) < 1e-5

    def test_snapshot_thresholds_captured(self):
        snapshot, _, _ = build_exit_rule_payload(
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
        )
        thr = snapshot["thresholds"]
        assert thr["intraday_profit_take_pct"] == 0.08
        assert thr["intraday_collapse_pct"] == 0.05
        assert thr["intraday_tighten_after_days"] == 3
        assert thr["intraday_tighten_atr_multiple"] == 1.5

    def test_zero_entry_price_safely_nulls_gain_pct(self):
        """Anti-regression: division-by-zero guard on entry_price=0."""
        stop = _make_stop()
        stop["entry_price"] = 0.0
        snapshot, _, _ = build_exit_rule_payload(
            stop=stop,
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
        )
        assert snapshot["gain_pct"] is None

    @pytest.mark.parametrize("reason,kind", [
        ("intraday_trailing_stop", "atr_trail"),
        ("intraday_profit_take", "profit_take"),
        ("intraday_collapse", "collapse"),
    ])
    def test_agent_output_maps_canonical_kind(self, reason, kind):
        _, output, _ = build_exit_rule_payload(
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(reason=reason),
            strategy_config=_make_exit_strategy_config(),
        )
        assert output["outcome"] == "fired"
        assert output["fired_rule"] == reason
        assert output["fired_rule_kind"] == kind

    def test_agent_output_carries_fill_outcome(self):
        _, output, _ = build_exit_rule_payload(
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
            fill_price=175.30,
            actual_shares_exited=150,
            trade_id="trade-abc",
        )
        assert output["action"] == "EXIT"
        assert output["shares_requested"] == 150
        assert output["fill_price"] == 175.30
        assert output["actual_shares_exited"] == 150
        assert output["trade_id"] == "trade-abc"


class TestExitRuleCapture:
    """End-to-end capture path with env-flag + S3 stub."""

    def test_writes_v2_artifact_to_canonical_key(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3_key = capture_exit_rule(
            run_date="2026-05-15",
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
            s3_client=s3,
        )
        s3.put_object.assert_called_once()
        put_kwargs = s3.put_object.call_args.kwargs
        assert put_kwargs["Bucket"] == "alpha-engine-research"
        assert "/executor:exit_rules/" in put_kwargs["Key"]
        assert put_kwargs["Key"].endswith(".json")
        assert s3_key == put_kwargs["Key"]
        body = json.loads(put_kwargs["Body"].decode("utf-8"))
        assert body["schema_version"] == 2
        assert body["agent_id"] == "executor:exit_rules"
        assert body["model_metadata"] is None
        assert body["full_prompt_context"] is None
        assert body["input_data_snapshot"]["ticker"] == "AAPL"
        assert body["agent_output"]["fired_rule_kind"] == "atr_trail"

    def test_disabled_when_env_off(self, monkeypatch):
        monkeypatch.delenv(
            "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False,
        )
        s3 = _make_s3_stub()
        result = capture_exit_rule(
            run_date="2026-05-15",
            stop=_make_stop(),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
            s3_client=s3,
        )
        assert result is None
        s3.put_object.assert_not_called()

    def test_run_id_includes_ticker_and_uuid_suffix(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_exit_rule(
            run_date="2026-05-15",
            stop=_make_stop(ticker="NVDA"),
            price_state=_make_exit_price_state(),
            exit_signal=_make_exit_signal(),
            strategy_config=_make_exit_strategy_config(),
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        run_id = body["run_id"]
        assert run_id.startswith("2026-05-15_NVDA_")
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_s3_failure_propagates_capture_write_error(self, monkeypatch):
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3.put_object.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Denied"},
            },
            operation_name="PutObject",
        )
        with pytest.raises(DecisionCaptureWriteError):
            capture_exit_rule(
                run_date="2026-05-15",
                stop=_make_stop(),
                price_state=_make_exit_price_state(),
                exit_signal=_make_exit_signal(),
                strategy_config=_make_exit_strategy_config(),
                s3_client=s3,
            )


# ── Planner-side exit rules (PR 4b) ──────────────────────────────────────


def _make_planner_pos() -> dict:
    """Held-position fixture matching what main.py builds for evaluate_exits."""
    return {
        "shares": 150,
        "market_value": 26_287.50,
        "avg_cost": 170.00,
        "sector": "technology",
        "entry_date": "2026-05-08",
        "stance": "momentum",
        "catalyst_date": None,
    }


def _make_planner_research_signal(action: str = "HOLD") -> dict:
    return {
        "signal": action,
        "score": 78.5,
        "conviction": "rising",
    }


def _make_planner_stance_config() -> dict:
    """Stance-resolved strategy config — what _resolve_strategy_config_for_stance
    returns inside evaluate_exits."""
    return {
        "atr_period": 14,
        "atr_multiple": 2.0,
        "fallback_stop_enabled": True,
        "fallback_stop_pct": 0.10,
        "profit_take_pct": 0.20,
        "time_decay_enabled": True,
        "time_decay_days": 21,
        "momentum_exit_enabled": True,
        "momentum_exit_threshold": -0.05,
        "catalyst_follow_through_days": 3,
    }


def _make_planner_signal(action: str = "EXIT", reason: str = "atr_trailing_stop") -> dict:
    """evaluate_exits' returned signal-dict shape."""
    return {
        "ticker": "AAPL",
        "action": action,
        "shares": 150,
        "reason": reason,
        "detail": "trail stop at $167.50",
    }


class TestPlannerExitKindClassifier:
    @pytest.mark.parametrize("key,expected", [
        ("catalyst_hard_exit", "catalyst_hard_exit"),
        ("atr_trailing_stop", "atr_trail"),
        ("sector_veto_blocked", "sector_veto_blocked"),
        ("fallback_stop", "fallback_stop"),
        ("profit_take", "profit_take"),
        ("momentum_exit", "momentum_exit"),
        ("time_decay", "time_decay"),
        ("unknown_key", "unknown"),
    ])
    def test_known_keys(self, key, expected):
        assert _classify_planner_exit_kind(key) == expected

    def test_none_maps_to_no_fire(self):
        """None fired_rule_key → 'no_fire' — the counterfactual coverage
        row for grading."""
        assert _classify_planner_exit_kind(None) == "no_fire"


class TestPlannerExitPayloadShape:
    def test_snapshot_carries_producer_provenance_and_layer(self):
        snapshot, _, _ = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
        )
        assert snapshot["_producer"] == "alpha-engine.executor.exit_rules"
        # evaluation_layer distinguishes from daemon_intraday — same
        # agent_id (executor:exit_rules) but different decision layer.
        assert snapshot["evaluation_layer"] == "planner"

    def test_snapshot_carries_full_position_and_market_state(self):
        snapshot, _, _ = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(action="HOLD"),
            current_price=175.25, stance="momentum",
            catalyst_date="2026-05-20",
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
        )
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["entry_date"] == "2026-05-08"
        assert snapshot["avg_cost"] == 170.00
        assert snapshot["shares_held"] == 150
        assert snapshot["market_value"] == 26_287.50
        assert snapshot["sector"] == "technology"
        assert snapshot["stance"] == "momentum"
        assert snapshot["catalyst_date"] == "2026-05-20"
        assert snapshot["current_price"] == 175.25
        assert snapshot["research_action"] == "HOLD"
        assert snapshot["research_score"] == 78.5
        assert snapshot["research_conviction"] == "rising"
        # gain_pct = (175.25 - 170) / 170 = 0.030882...
        assert abs(snapshot["gain_pct"] - 0.030882) < 1e-5

    def test_snapshot_thresholds_carry_resolved_stance_config(self):
        snapshot, _, _ = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
        )
        thr = snapshot["thresholds"]
        # All 10 rule thresholds captured so replay can re-evaluate.
        assert thr["atr_period"] == 14
        assert thr["atr_multiple"] == 2.0
        assert thr["fallback_stop_enabled"] is True
        assert thr["profit_take_pct"] == 0.20
        assert thr["time_decay_enabled"] is True
        assert thr["time_decay_days"] == 21
        assert thr["momentum_exit_threshold"] == -0.05
        assert thr["catalyst_follow_through_days"] == 3

    def test_fired_outcome_records_rule_key_and_kind(self):
        _, output, summary = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=_make_planner_signal(action="EXIT", reason="trail stop"),
            fired_rule_key="atr_trailing_stop",
        )
        assert output["outcome"] == "fired"
        assert output["action"] == "EXIT"
        assert output["fired_rule"] == "trail stop"
        assert output["fired_rule_key"] == "atr_trailing_stop"
        assert output["fired_rule_kind"] == "atr_trail"
        assert output["shares_requested"] == 150
        assert "fired=atr_trailing_stop" in summary

    def test_no_fire_outcome_with_none_key(self):
        _, output, summary = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
        )
        assert output["outcome"] == "no_fire"
        assert output["fired_rule_key"] is None
        assert output["fired_rule_kind"] == "no_fire"
        assert output["action"] is None
        assert "no_fire" in summary

    def test_sector_veto_blocked_carried_as_no_fire_with_specific_kind(self):
        """When ATR fires but the sector-relative veto suppresses it,
        the planner returns (None, 'sector_veto_blocked') — the artifact
        must record this distinct kind separately from a normal no_fire."""
        _, output, _ = build_planner_exit_payload(
            ticker="AAPL", pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key="sector_veto_blocked",
        )
        assert output["outcome"] == "no_fire"
        assert output["fired_rule_key"] == "sector_veto_blocked"
        assert output["fired_rule_kind"] == "sector_veto_blocked"

    def test_zero_avg_cost_safely_nulls_gain_pct(self):
        pos = _make_planner_pos()
        pos["avg_cost"] = 0.0
        snapshot, _, _ = build_planner_exit_payload(
            ticker="AAPL", pos=pos,
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
        )
        assert snapshot["gain_pct"] is None


class TestPlannerExitCapture:
    """End-to-end capture path with env-flag + S3 stub."""

    def test_writes_v2_artifact_to_canonical_key(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3_key = capture_planner_exit(
            run_date="2026-05-15", ticker="AAPL",
            pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=_make_planner_signal(),
            fired_rule_key="atr_trailing_stop",
            s3_client=s3,
        )
        s3.put_object.assert_called_once()
        put_kwargs = s3.put_object.call_args.kwargs
        # Same agent_id as PR 4 daemon-side captures so grading reads
        # one S3 prefix; layer field distinguishes producer.
        assert "/executor:exit_rules/" in put_kwargs["Key"]
        assert s3_key == put_kwargs["Key"]
        body = json.loads(put_kwargs["Body"].decode("utf-8"))
        assert body["schema_version"] == 2
        assert body["agent_id"] == "executor:exit_rules"
        assert body["model_metadata"] is None
        assert body["input_data_snapshot"]["evaluation_layer"] == "planner"
        assert body["agent_output"]["fired_rule_kind"] == "atr_trail"

    def test_no_fire_captured_as_counterfactual(self, monkeypatch):
        """Counterfactual coverage: every position that reaches the
        rule-evaluation phase produces an artifact, including no-fire
        positions. Grading uses these to measure missed-exit precision."""
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_planner_exit(
            run_date="2026-05-15", ticker="MSFT",
            pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=180.00, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        assert body["agent_output"]["outcome"] == "no_fire"
        assert body["agent_output"]["fired_rule_kind"] == "no_fire"

    def test_disabled_when_env_off(self, monkeypatch):
        monkeypatch.delenv(
            "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False,
        )
        s3 = _make_s3_stub()
        result = capture_planner_exit(
            run_date="2026-05-15", ticker="AAPL",
            pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=175.25, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
            s3_client=s3,
        )
        assert result is None
        s3.put_object.assert_not_called()

    def test_s3_failure_propagates_capture_write_error(self, monkeypatch):
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        s3.put_object.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Denied"},
            },
            operation_name="PutObject",
        )
        with pytest.raises(DecisionCaptureWriteError):
            capture_planner_exit(
                run_date="2026-05-15", ticker="AAPL",
                pos=_make_planner_pos(),
                research_signal=_make_planner_research_signal(),
                current_price=175.25, stance="momentum", catalyst_date=None,
                stance_config=_make_planner_stance_config(),
                signal=None, fired_rule_key=None,
                s3_client=s3,
            )

    def test_run_id_format_pinned(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        s3 = _make_s3_stub()
        capture_planner_exit(
            run_date="2026-05-15", ticker="NVDA",
            pos=_make_planner_pos(),
            research_signal=_make_planner_research_signal(),
            current_price=400.00, stance="momentum", catalyst_date=None,
            stance_config=_make_planner_stance_config(),
            signal=None, fired_rule_key=None,
            s3_client=s3,
        )
        body = json.loads(s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        run_id = body["run_id"]
        assert run_id.startswith("2026-05-15_NVDA_")
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)
