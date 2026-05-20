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
