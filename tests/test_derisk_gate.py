"""Tests for executor/derisk_gate.py — expectancy-gated de-risk sizing
enforcement (alpha-engine-config#1259 / config-PR2071 / config-I2820).

Covers the four scenarios named in config-I2820's acceptance criteria, plus
the fail-loud config/ledger validation and the risk_aversion-floor helper:

  * flag on + negative expectancy (any metric breaches red-line) → capped
  * flag on + positive expectancy (all metrics clear red-lines) → nominal
  * flag off (default) → nominal, zero S3 I/O
  * malformed block/ledger + flag on → raises DeriskGateConfigError

All hermetic — S3 is a tiny in-memory fake (mirrors test_champion.py's
_FakeS3), no real boto3/network calls.
"""

from __future__ import annotations

import io
import json

import pytest
from botocore.exceptions import ClientError

from executor.derisk_gate import (
    CARRYOVER_LEDGER_KEY,
    DeriskGateConfigError,
    DeriskGateState,
    apply_risk_aversion_floor,
    evaluate_derisk_gate,
)


class _FakeS3:
    """Minimal get_object stand-in over a dict of {key: bytes}."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                operation_name="GetObject",
            )
        return {"Body": io.BytesIO(self.objects[Key])}


def _ledger_bytes(
    alpha_vs_spy: float = -0.02,
    information_ratio_ci_lower: float = -3.0,
    sharpe_ratio: float = 0.8,
    **extra,
) -> bytes:
    """Build a carryover_ledger.json payload with all three metrics clear of
    the default thresholds (alpha_vs_spy >= -0.05, ir_ci_lower >= -7,
    sharpe >= 0.5) unless overridden — the "positive expectancy" default."""
    payload = {
        "stances": [
            {
                "name": "halt-or-derisk-live-deployment",
                "metrics": {
                    "alpha_vs_spy": alpha_vs_spy,
                    "information_ratio_ci_lower": information_ratio_ci_lower,
                    "sharpe_ratio": sharpe_ratio,
                },
            }
        ],
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def _base_config(**overrides) -> dict:
    cfg = {
        "derisk_on_expectancy_enabled": True,
        "derisk_sizing_multiplier": 0.50,
        "derisk_expectancy_thresholds": {
            "alpha_vs_spy": -0.05,
            "information_ratio_ci_lower": -7,
            "sharpe_ratio": 0.5,
        },
        "portfolio_optimizer": {"risk_aversion": 3.0},
    }
    cfg.update(overrides)
    return cfg


BUCKET = "test-bucket"


# ═══════════════════════════════════════════════════════════════════════════
# The four acceptance-criteria scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestFourScenarios:
    def test_flag_on_negative_expectancy_caps_sizing(self):
        """flag on + negative expectancy (a metric breaches its red-line) →
        gate active, sizing capped to derisk_sizing_multiplier, MVO
        risk_aversion floored at configured/multiplier."""
        s3 = _FakeS3({
            CARRYOVER_LEDGER_KEY: _ledger_bytes(alpha_vs_spy=-0.11),  # breached (real PR2071 reading)
        })
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

        assert gate.active is True
        assert gate.sizing_multiplier == 0.50
        assert gate.risk_aversion_floor == pytest.approx(3.0 / 0.50)
        assert "alpha_vs_spy" in gate.triggering_metrics

    def test_flag_on_positive_expectancy_is_nominal(self):
        """flag on + positive expectancy (all metrics clear red-lines) →
        gate inactive, nominal (1.0x) sizing, no risk_aversion floor."""
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

        assert gate.active is False
        assert gate.sizing_multiplier == 1.0
        assert gate.risk_aversion_floor is None
        assert gate.triggering_metrics == ()

    def test_flag_off_is_nominal_and_never_touches_s3(self):
        """flag off (default / explicit False) → nominal sizing, and the
        S3 client is never called — zero I/O when not opted in."""
        s3 = _FakeS3()  # no objects — would raise if get_object were called

        cfg = _base_config(derisk_on_expectancy_enabled=False)
        gate = evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

        assert gate.active is False
        assert gate.enabled is False
        assert gate.sizing_multiplier == 1.0
        assert gate.risk_aversion_floor is None

    def test_flag_absent_defaults_off(self):
        """Absent derisk_on_expectancy_enabled key (pre-PR2071 risk.yaml,
        the state of `main` before config#2071 merges) behaves identically
        to an explicit False — this PR must not require PR2071 to have
        merged first."""
        s3 = _FakeS3()
        cfg = {"portfolio_optimizer": {"risk_aversion": 3.0}}  # no derisk keys at all
        gate = evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

        assert gate.enabled is False
        assert gate.sizing_multiplier == 1.0

    def test_malformed_block_with_flag_on_raises(self):
        """malformed block (derisk_expectancy_thresholds missing a required
        key) + flag on → raises DeriskGateConfigError, never silently
        defaults to full sizing."""
        cfg = _base_config(
            derisk_expectancy_thresholds={"alpha_vs_spy": -0.05},  # missing 2 of 3
        )
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="information_ratio_ci_lower"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)


# ═══════════════════════════════════════════════════════════════════════════
# Threshold-breach semantics — each metric independently, and multi-breach
# ═══════════════════════════════════════════════════════════════════════════


class TestThresholdEvaluation:
    def test_information_ratio_breach_triggers(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(information_ratio_ci_lower=-8.0)})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert gate.active is True
        assert gate.triggering_metrics == ("information_ratio_ci_lower",)

    def test_sharpe_ratio_breach_triggers(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(sharpe_ratio=0.23)})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert gate.active is True
        assert gate.triggering_metrics == ("sharpe_ratio",)

    def test_multiple_breaches_all_named(self):
        """PR2071's own worked example: alpha_vs_spy=-0.11 (breach),
        information_ratio_ci_lower=-3.06 (OK), sharpe_ratio=0.23 (breach)."""
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(
            alpha_vs_spy=-0.11, information_ratio_ci_lower=-3.06, sharpe_ratio=0.23,
        )})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert gate.active is True
        assert set(gate.triggering_metrics) == {"alpha_vs_spy", "sharpe_ratio"}

    def test_exact_boundary_is_not_a_breach(self):
        """Threshold semantics are `>= red-line` clears (per PR2071's body:
        'must be >= this; if below, derisk is active') — a metric sitting
        exactly ON the red-line is NOT a breach."""
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(
            alpha_vs_spy=-0.05, information_ratio_ci_lower=-7.0, sharpe_ratio=0.5,
        )})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert gate.active is False


# ═══════════════════════════════════════════════════════════════════════════
# Fail-loud config validation
# ═══════════════════════════════════════════════════════════════════════════


class TestFailLoudConfig:
    def test_missing_multiplier_key_raises(self):
        cfg = _base_config()
        del cfg["derisk_sizing_multiplier"]
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="derisk_sizing_multiplier"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

    def test_non_numeric_multiplier_raises(self):
        cfg = _base_config(derisk_sizing_multiplier="half")
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="non-numeric"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

    @pytest.mark.parametrize("bad_value", [0.0, -0.5, 1.5])
    def test_multiplier_out_of_range_raises(self, bad_value):
        cfg = _base_config(derisk_sizing_multiplier=bad_value)
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="out of range"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

    def test_thresholds_not_a_dict_raises(self):
        cfg = _base_config(derisk_expectancy_thresholds=[1, 2, 3])
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="mapping"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

    def test_missing_thresholds_key_raises(self):
        cfg = _base_config()
        del cfg["derisk_expectancy_thresholds"]
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        with pytest.raises(DeriskGateConfigError, match="derisk_expectancy_thresholds"):
            evaluate_derisk_gate(cfg, bucket=BUCKET, s3_client=s3)

    def test_no_bucket_supplied_raises(self):
        with pytest.raises(DeriskGateConfigError, match="bucket"):
            evaluate_derisk_gate(_base_config(), bucket=None)


# ═══════════════════════════════════════════════════════════════════════════
# Fail-loud ledger validation
# ═══════════════════════════════════════════════════════════════════════════


class TestFailLoudLedger:
    def test_missing_ledger_raises(self):
        s3 = _FakeS3()  # no objects — NoSuchKey
        with pytest.raises(DeriskGateConfigError, match="unreadable"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_malformed_json_raises(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: b"{not valid json"})
        with pytest.raises(DeriskGateConfigError, match="malformed"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_non_dict_ledger_raises(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: json.dumps([1, 2, 3]).encode()})
        with pytest.raises(DeriskGateConfigError, match="JSON object"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_missing_stance_entry_raises(self):
        payload = json.dumps({"stances": [{"name": "some-other-stance", "metrics": {}}]}).encode()
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: payload})
        with pytest.raises(DeriskGateConfigError, match="halt-or-derisk-live-deployment"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_missing_metric_in_stance_raises(self):
        payload = json.dumps({
            "stances": [{
                "name": "halt-or-derisk-live-deployment",
                "metrics": {"alpha_vs_spy": -0.02, "sharpe_ratio": 0.8},
                # information_ratio_ci_lower missing
            }],
        }).encode()
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: payload})
        with pytest.raises(DeriskGateConfigError, match="information_ratio_ci_lower"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_non_numeric_metric_raises(self):
        payload = json.dumps({
            "stances": [{
                "name": "halt-or-derisk-live-deployment",
                "metrics": {
                    "alpha_vs_spy": "bad",
                    "information_ratio_ci_lower": -3.0,
                    "sharpe_ratio": 0.8,
                },
            }],
        }).encode()
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: payload})
        with pytest.raises(DeriskGateConfigError, match="non-numeric"):
            evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)

    def test_flat_mapping_ledger_shape_also_supported(self):
        """Tolerates a flat {stance_name: {...}} ledger shape as well as the
        {"stances": [...]} list shape — PR2071's body doesn't pin an exact
        ledger schema beyond 'find the halt-or-derisk-live-deployment entry'."""
        payload = json.dumps({
            "halt-or-derisk-live-deployment": {
                "alpha_vs_spy": -0.02,
                "information_ratio_ci_lower": -3.0,
                "sharpe_ratio": 0.8,
            },
        }).encode()
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: payload})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert gate.active is False
        assert gate.metrics["sharpe_ratio"] == 0.8


# ═══════════════════════════════════════════════════════════════════════════
# apply_risk_aversion_floor
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyRiskAversionFloor:
    def test_inactive_gate_is_noop(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes()})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert apply_risk_aversion_floor(3.0, gate) == 3.0

    def test_active_gate_floors_up(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(alpha_vs_spy=-0.11)})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        # configured 3.0 / multiplier 0.50 = 6.0 floor (matches PR2071's worked example)
        assert apply_risk_aversion_floor(3.0, gate) == pytest.approx(6.0)

    def test_active_gate_never_lowers_an_already_higher_value(self):
        """A tuner-selected risk_aversion already above the floor is left
        untouched — the gate can only push MORE conservative, never less."""
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(alpha_vs_spy=-0.11)})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        assert apply_risk_aversion_floor(10.0, gate) == 10.0


# ═══════════════════════════════════════════════════════════════════════════
# to_log_dict — structured-event / decision-artifact serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestToLogDict:
    def test_serializes_active_state(self):
        s3 = _FakeS3({CARRYOVER_LEDGER_KEY: _ledger_bytes(alpha_vs_spy=-0.11)})
        gate = evaluate_derisk_gate(_base_config(), bucket=BUCKET, s3_client=s3)
        d = gate.to_log_dict()
        assert d["active"] is True
        assert d["enabled"] is True
        assert d["sizing_multiplier"] == 0.50
        assert d["triggering_metrics"] == ["alpha_vs_spy"]
        assert isinstance(d["metrics"], dict)
        assert isinstance(d["thresholds"], dict)
        # JSON-serializable end to end (no dataclass/tuple leakage)
        json.dumps(d)

    def test_serializes_disabled_state(self):
        gate = evaluate_derisk_gate({"derisk_on_expectancy_enabled": False})
        d = gate.to_log_dict()
        assert d["enabled"] is False
        assert d["active"] is False
        json.dumps(d)
