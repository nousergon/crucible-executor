"""Unit tests for executor.data_manifest — S3 freshness probe + manifest writer."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from executor.data_manifest import verify_s3_object_fresh, write_data_manifest


def _s3_mock(last_modified=None, raise_exc=None):
    s3 = MagicMock()
    if raise_exc is not None:
        s3.head_object.side_effect = raise_exc
    else:
        s3.head_object.return_value = {"LastModified": last_modified}
    return s3


class TestVerifyS3ObjectFresh:

    def test_missing_object_raises_runtime_error(self):
        s3 = _s3_mock(raise_exc=Exception("NoSuchKey"))
        with pytest.raises(RuntimeError, match="not found"):
            verify_s3_object_fresh(s3, "bucket", "key", "2026-04-16")

    def test_fresh_today_passes(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        recent = datetime.now(UTC) - timedelta(minutes=30)
        s3 = _s3_mock(last_modified=recent)
        verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_stale_today_raises_runtime_error(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        yesterday_write = datetime.now(UTC) - timedelta(hours=25)
        s3 = _s3_mock(last_modified=yesterday_write)
        with pytest.raises(RuntimeError, match="stale"):
            verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_backfill_skips_freshness_check(self):
        old_write = datetime.now(UTC) - timedelta(days=30)
        s3 = _s3_mock(last_modified=old_write)
        verify_s3_object_fresh(s3, "bucket", "key", "2026-03-10")

    def test_just_under_threshold_today_passes(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        almost_stale = datetime.now(UTC) - timedelta(hours=11, minutes=59)
        s3 = _s3_mock(last_modified=almost_stale)
        verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_custom_max_age_hours(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        three_hours_old = datetime.now(UTC) - timedelta(hours=3)
        s3 = _s3_mock(last_modified=three_hours_old)
        verify_s3_object_fresh(s3, "bucket", "key", today, max_age_hours=6)
        with pytest.raises(RuntimeError, match="stale"):
            verify_s3_object_fresh(s3, "bucket", "key", today, max_age_hours=2)


class TestWriteDataManifest:

    def test_writes_dated_manifest(self):
        s3 = MagicMock()
        with patch("executor.data_manifest.boto3.client", return_value=s3):
            write_data_manifest(
                bucket="b", module_name="executor", run_date="2026-05-12",
                manifest={"trades": 5, "alpha_pct": 0.42},
            )
        call = s3.put_object.call_args
        assert call.kwargs["Key"] == "data_manifest/executor/2026-05-12.json"
        payload = json.loads(call.kwargs["Body"])
        assert payload["module"] == "executor"
        assert payload["run_date"] == "2026-05-12"
        assert "written_at" in payload
        assert payload["trades"] == 5
        assert payload["alpha_pct"] == 0.42

    def test_put_failure_swallowed(self):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("AccessDenied")
        with patch("executor.data_manifest.boto3.client", return_value=s3):
            write_data_manifest(
                bucket="b", module_name="executor", run_date="2026-05-12",
                manifest={"trades": 0},
            )
