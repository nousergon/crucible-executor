"""Tests for executor/accepted_gaps.py — durable accepted-gap declarations for
eod_pnl (alpha-engine-config#5570)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3

from executor.accepted_gaps import load_accepted_gap_entries, load_accepted_gaps


# A fake NoSuchKey exception class — must inherit from BaseException so that
# ``except s3.exceptions.NoSuchKey:`` in the production code doesn't raise
# ``TypeError: catching classes that do not inherit from BaseException``
# when the S3 client is a MagicMock.
class _FakeNoSuchKey(Exception):
    """Simulates ``botocore.exceptions.ClientError`` for NoSuchKey responses."""
    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"NoSuchKey: {error_response}")


def _mock_s3(*, no_such_key: bool = False, get_object_rv=None) -> MagicMock:
    """Build a mock boto3 S3 client with a real exception type for NoSuchKey.

    The production code catches ``s3.exceptions.NoSuchKey``, which must be a
    real class inheriting from ``BaseException`` — a bare MagicMock fails
    because Python rejects catching non-exception types at the frame level.
    """
    s3 = MagicMock(name="s3")
    s3.exceptions.NoSuchKey = _FakeNoSuchKey
    if no_such_key:
        s3.get_object.side_effect = _FakeNoSuchKey(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject")
    elif get_object_rv is not None:
        s3.get_object.return_value = get_object_rv
    return s3


def _s3_body(data):
    """Return a mock S3 response body for the given dict or bytes."""
    body_data = json.dumps(data).encode() if isinstance(data, dict) else data
    return {"Body": MagicMock(read=lambda: body_data)}


class TestLoadAcceptedGaps:
    def test_no_file_returns_empty_dict(self):
        s3 = _mock_s3(no_such_key=True)
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert gaps == {}

    def test_unexpected_error_returns_empty_dict_fail_safe(self):
        """Any non-NoSuchKey error is caught and returns {} — fail-safe by design.
        A missing registry at worst means a gap gets flagged when it could have
        been silent, which is safer than silently suppressing a real gap."""
        s3 = _mock_s3()
        s3.get_object.side_effect = Exception("AccessDenied")
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert gaps == {}

    def test_parses_valid_list_registry(self):
        payload = {
            "schema_version": 1,
            "gaps": [
                {
                    "date": "2026-07-27",
                    "reason": "CaptureSnapshot failed",
                    "ruling": "nousergon/alpha-engine-config#5325",
                    "accepted_at": "2026-07-29T00:00:00Z",
                },
            ],
        }
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert "2026-07-27" in gaps
        assert gaps["2026-07-27"]["ruling"] == "nousergon/alpha-engine-config#5325"
        assert gaps["2026-07-27"]["reason"] == "CaptureSnapshot failed"

    def test_malformed_json_returns_empty(self):
        s3 = _mock_s3(
            get_object_rv={"Body": MagicMock(read=lambda: b"not json{{}}")},
        )
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert gaps == {}

    def test_missing_gaps_key_returns_empty(self):
        payload = {"schema_version": 1}
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert gaps == {}

    def test_gaps_not_a_list_returns_empty(self):
        """If 'gaps' is a dict instead of a list, treat as empty (schema mismatch)."""
        payload = {"schema_version": 1, "gaps": {"2026-07-27": {"reason": "test"}}}
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert gaps == {}

    def test_skips_entries_without_date(self):
        """Entries missing the 'date' key are filtered out."""
        payload = {
            "schema_version": 1,
            "gaps": [
                {"reason": "no date here", "ruling": "ref"},
                {"date": "2026-07-27", "reason": "valid", "ruling": "ref"},
            ],
        }
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert len(gaps) == 1
        assert "2026-07-27" in gaps

    def test_skips_non_dict_entries(self):
        """Non-dict entries in the gaps list are filtered out."""
        payload = {
            "schema_version": 1,
            "gaps": [
                "not a dict",
                {"date": "2026-07-27", "reason": "valid", "ruling": "ref"},
            ],
        }
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert len(gaps) == 1
        assert "2026-07-27" in gaps

    def test_multiple_gaps(self):
        payload = {
            "schema_version": 1,
            "gaps": [
                {"date": "2026-07-27", "reason": "first", "ruling": "ref1"},
                {"date": "2026-07-28", "reason": "second", "ruling": "ref2"},
            ],
        }
        s3 = _mock_s3(get_object_rv=_s3_body(payload))
        with patch.object(boto3, "client", return_value=s3):
            gaps = load_accepted_gaps("my-bucket")
        assert len(gaps) == 2
        assert "2026-07-27" in gaps
        assert "2026-07-28" in gaps


class TestLoadAcceptedGapEntries:
    """The gate's loader keeps malformed entries so the gate can reject them
    (alpha-engine-config-I11087), and fails closed to [] (accepts nothing)."""

    def test_returns_the_raw_list_including_dateless_entries(self):
        gaps = [{"date": "2026-07-27", "ruling": "nousergon/alpha-engine-config#5325"},
                {"reason": "no date"}]
        s3 = _mock_s3(get_object_rv=_s3_body({"schema_version": 1, "gaps": gaps}))
        with patch.object(boto3, "client", return_value=s3):
            assert load_accepted_gap_entries("b") == gaps

    def test_missing_registry_accepts_nothing(self):
        s3 = _mock_s3(no_such_key=True)
        with patch.object(boto3, "client", return_value=s3):
            assert load_accepted_gap_entries("b") == []

    def test_unreadable_registry_accepts_nothing(self):
        s3 = _mock_s3()
        s3.get_object.side_effect = RuntimeError("boom")
        with patch.object(boto3, "client", return_value=s3):
            assert load_accepted_gap_entries("b") == []

    def test_non_list_gaps_accepts_nothing(self):
        s3 = _mock_s3(get_object_rv=_s3_body({"gaps": {"2026-07-27": {}}}))
        with patch.object(boto3, "client", return_value=s3):
            assert load_accepted_gap_entries("b") == []
