"""Executor-side drawdown leg (regime ensemble leg 3) — read +
extract helpers. Mirrors tests/test_regime_fast_signal_executor.py.

regime-drawdown-hysteresis-260518.md PR 4 (executor consumer). The
main.py posture-override block itself is a thin most-protective rank
comparison over these helpers + the forced-bear pattern (both already
covered); this file pins the read/extract contract the override relies
on.
"""
from __future__ import annotations

from unittest.mock import patch

from executor.signal_reader import (
    REGIME_DRAWDOWN_PREFIX,
    extract_drawdown_effective_regime,
    extract_drawdown_protective_severity,
    read_drawdown_substrate,
)


class TestExtractDrawdownEffectiveRegime:
    def test_nested_compose_dict_shape(self):
        # The daily stage stores compose_effective_regime's dict.
        payload = {
            "effective_regime": {
                "effective_regime": "bear",
                "drivers": {"drawdown_spy": "bear"},
            }
        }
        assert extract_drawdown_effective_regime(payload) == "bear"

    def test_bare_string_tolerated(self):
        assert extract_drawdown_effective_regime(
            {"effective_regime": "caution"}
        ) == "caution"

    def test_none_payload(self):
        assert extract_drawdown_effective_regime(None) is None

    def test_non_dict(self):
        assert extract_drawdown_effective_regime("nope") is None
        assert extract_drawdown_effective_regime(7) is None

    def test_missing_key(self):
        assert extract_drawdown_effective_regime({"spy": {}}) is None

    def test_empty_or_non_string(self):
        assert extract_drawdown_effective_regime(
            {"effective_regime": ""}
        ) is None
        assert extract_drawdown_effective_regime(
            {"effective_regime": {"effective_regime": None}}
        ) is None
        assert extract_drawdown_effective_regime(
            {"effective_regime": {"effective_regime": 3}}
        ) is None


class TestExtractDrawdownProtectiveSeverity:
    """v0.42.0 Phase 2B (caution-regime-retirement-260528.md): canonical
    drawdown-axis severity ordinal exposed for executor + future
    half-step protection consumers."""

    def test_canonical_payload_severity_2(self):
        # New (post-Phase-2A) compose_effective_regime emits the ordinal
        # in the nested dict alongside effective_regime/drivers.
        payload = {
            "effective_regime": {
                "effective_regime": "bear",
                "drivers": {"drawdown_spy": "bear"},
                "drawdown_tier": "risk_off",
                "drawdown_protective_severity": 2,
            }
        }
        assert extract_drawdown_protective_severity(payload) == 2

    def test_canonical_payload_severity_1(self):
        payload = {
            "effective_regime": {
                "effective_regime": "neutral",
                "drivers": {"drawdown_spy": None},
                "drawdown_tier": "caution",
                "drawdown_protective_severity": 1,
            }
        }
        assert extract_drawdown_protective_severity(payload) == 1

    def test_canonical_payload_severity_0(self):
        payload = {
            "effective_regime": {
                "effective_regime": "neutral",
                "drivers": {},
                "drawdown_tier": "risk_on",
                "drawdown_protective_severity": 0,
            }
        }
        assert extract_drawdown_protective_severity(payload) == 0

    def test_legacy_grandfather_string_bear_derives_severity_2(self):
        # Pre-Phase-2A payloads carried only the string field. Grandfather:
        # derive severity from the legacy string for backward-compat.
        payload = {"effective_regime": "bear"}
        assert extract_drawdown_protective_severity(payload) == 2

    def test_legacy_grandfather_string_caution_derives_severity_1(self):
        payload = {"effective_regime": "caution"}
        assert extract_drawdown_protective_severity(payload) == 1

    def test_legacy_grandfather_string_bull_derives_severity_0(self):
        payload = {"effective_regime": "bull"}
        assert extract_drawdown_protective_severity(payload) == 0

    def test_none_or_missing_returns_zero(self):
        assert extract_drawdown_protective_severity(None) == 0
        assert extract_drawdown_protective_severity({}) == 0
        assert extract_drawdown_protective_severity("nope") == 0
        assert extract_drawdown_protective_severity({"spy": {}}) == 0

    def test_severity_clamped_to_valid_range(self):
        # Defensive — if a malformed payload had severity > 2 or < 0,
        # the helper clamps to [0, 2] rather than passing through.
        payload = {"effective_regime": {"drawdown_protective_severity": 5}}
        assert extract_drawdown_protective_severity(payload) == 2
        payload_neg = {"effective_regime": {"drawdown_protective_severity": -1}}
        assert extract_drawdown_protective_severity(payload_neg) == 0


class TestReadDrawdownSubstrate:
    @patch("executor.signal_reader.boto3")
    @patch("executor.signal_reader.load_latest_eval_artifact")
    def test_reads_canonical_prefix(self, mock_load, _mock_boto3):
        assert REGIME_DRAWDOWN_PREFIX == "regime/drawdown"
        mock_load.return_value = {"effective_regime": {"effective_regime": "bear"}}
        out = read_drawdown_substrate("alpha-engine-research")
        assert out == {"effective_regime": {"effective_regime": "bear"}}
        _, kwargs = mock_load.call_args
        assert kwargs["bucket"] == "alpha-engine-research"
        assert kwargs["prefix"] == REGIME_DRAWDOWN_PREFIX

    @patch("executor.signal_reader.boto3")
    @patch("executor.signal_reader.load_latest_eval_artifact", return_value=None)
    def test_none_artifact_composes_to_none(self, _mock_load, _mock_boto3):
        assert read_drawdown_substrate("b") is None
        # composes with extract → safe None (⇒ no override)
        assert extract_drawdown_effective_regime(read_drawdown_substrate("b")) is None
