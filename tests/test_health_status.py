"""Unit tests for executor.health_status — freshness utilities."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from executor.health_status import verify_s3_object_fresh


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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        s3 = _s3_mock(last_modified=recent)
        verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_stale_today_raises_runtime_error(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday_write = datetime.now(timezone.utc) - timedelta(hours=25)
        s3 = _s3_mock(last_modified=yesterday_write)
        with pytest.raises(RuntimeError, match="stale"):
            verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_backfill_skips_freshness_check(self):
        """Historical run_date: existence-only, LastModified irrelevant."""
        old_write = datetime.now(timezone.utc) - timedelta(days=30)
        s3 = _s3_mock(last_modified=old_write)
        verify_s3_object_fresh(s3, "bucket", "key", "2026-03-10")

    def test_just_under_threshold_today_passes(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        almost_stale = datetime.now(timezone.utc) - timedelta(hours=11, minutes=59)
        s3 = _s3_mock(last_modified=almost_stale)
        verify_s3_object_fresh(s3, "bucket", "key", today)

    def test_custom_max_age_hours(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        three_hours_old = datetime.now(timezone.utc) - timedelta(hours=3)
        s3 = _s3_mock(last_modified=three_hours_old)
        verify_s3_object_fresh(s3, "bucket", "key", today, max_age_hours=6)
        with pytest.raises(RuntimeError, match="stale"):
            verify_s3_object_fresh(s3, "bucket", "key", today, max_age_hours=2)
