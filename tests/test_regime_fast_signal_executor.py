"""Executor-side Stage F2 — read_fast_signal + extract_forced_bear.

regime-fast-signal-260515.md. Mirrors the Wire-2 substrate-helper
schema-defense tests. The main.py forced-bear override itself is
exercised via the existing planner integration paths; here we pin the
two pure helpers + the read wrapper.
"""
from __future__ import annotations

from unittest.mock import patch

from executor.signal_reader import (
    REGIME_FAST_SIGNAL_PREFIX,
    extract_forced_bear,
    read_fast_signal,
)


class TestExtractForcedBear:
    def test_true_when_latched_and_not_warmup(self):
        assert extract_forced_bear(
            {"forced_bear": True, "warmup": False}
        ) is True

    def test_false_when_not_latched(self):
        assert extract_forced_bear(
            {"forced_bear": False, "warmup": False}
        ) is False

    def test_warmup_suppresses_even_if_latched(self):
        # Defence-in-depth: a warming detector must never assert a break
        # even if a stale/odd artifact carries forced_bear=True.
        assert extract_forced_bear(
            {"forced_bear": True, "warmup": True}
        ) is False

    def test_none_payload_is_false(self):
        assert extract_forced_bear(None) is False

    def test_non_dict_is_false(self):
        assert extract_forced_bear("nope") is False
        assert extract_forced_bear(42) is False

    def test_missing_forced_bear_key_is_false(self):
        assert extract_forced_bear({"intensity_z": -2.0}) is False

    def test_non_bool_forced_bear_is_false(self):
        # Strict identity check — only literal True latches.
        assert extract_forced_bear({"forced_bear": 1}) is False
        assert extract_forced_bear({"forced_bear": "true"}) is False


class TestReadFastSignal:
    @patch("executor.signal_reader.boto3")
    @patch("executor.signal_reader.load_latest_eval_artifact")
    def test_reads_canonical_prefix(self, mock_load, _mock_boto3):
        mock_load.return_value = {"forced_bear": True, "warmup": False}
        out = read_fast_signal("alpha-engine-research")
        assert out == {"forced_bear": True, "warmup": False}
        _, kwargs = mock_load.call_args
        assert kwargs["bucket"] == "alpha-engine-research"
        assert kwargs["prefix"] == REGIME_FAST_SIGNAL_PREFIX == "regime/fast_signal"

    @patch("executor.signal_reader.boto3")
    @patch("executor.signal_reader.load_latest_eval_artifact", return_value=None)
    def test_none_artifact_propagates(self, _mock_load, _mock_boto3):
        assert read_fast_signal("b") is None
        # composes with extract_forced_bear → safe False
        assert extract_forced_bear(read_fast_signal("b")) is False
