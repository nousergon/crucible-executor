"""Tests for read_distribution_gate — the executor's reader for the predictor
output-distribution gate block that drives the hold-book safeguard (2026-06-01).

The gate block is carried on predictions/latest.json (surfaced 2026-06-01 in the
predictor's write_output stage). The executor reads it and, if passed is False
("strongly biased" batch), holds the current book instead of letting the
optimizer rotate off the flagged predictions.

Fail-open contract: a missing gate (older predictions / no predictions) returns
None, and the caller must NOT hold-book on None — only an explicit passed=False.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from executor.signal_reader import read_distribution_gate


def _fake_s3(payload: dict):
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode())}
    return s3


def test_returns_gate_block_when_failed():
    payload = {
        "date": "2026-06-01",
        "output_distribution_gate": {
            "passed": False,
            "failed_check": "direction_skew",
            "reason": "direction skew 89.66% exceeds 85%",
            "metrics": {"direction_skew": 0.8966, "n_up": 3, "n_down": 26},
        },
        "predictions": [],
    }
    with patch("executor.signal_reader.boto3.client", return_value=_fake_s3(payload)):
        gate = read_distribution_gate("bucket")
    assert gate is not None
    assert gate["passed"] is False
    assert gate["failed_check"] == "direction_skew"


def test_returns_gate_block_when_passed():
    payload = {"output_distribution_gate": {"passed": True}, "predictions": []}
    with patch("executor.signal_reader.boto3.client", return_value=_fake_s3(payload)):
        gate = read_distribution_gate("bucket")
    assert gate is not None and gate["passed"] is True


def test_missing_gate_returns_none_fail_open():
    """Older predictions without the block → None (caller proceeds normally)."""
    payload = {"date": "2026-05-29", "predictions": [{"ticker": "AAPL"}]}
    with patch("executor.signal_reader.boto3.client", return_value=_fake_s3(payload)):
        assert read_distribution_gate("bucket") is None


def test_no_predictions_object_returns_none():
    """NoSuchKey → None, never raises (fail-open)."""
    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        assert read_distribution_gate("bucket") is None
