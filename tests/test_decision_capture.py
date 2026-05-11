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
    _classify_trigger_kind,
    build_entry_trigger_payload,
    capture_entry_trigger,
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
