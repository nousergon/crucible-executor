"""config#2891 — consumer-side staleness assertion for config/executor_params.json.

Verifies the WARN-only staleness signal fires when the S3 pointer's
LastModified is older than the 2-weekly-cycle threshold, and stays silent
when fresh or absent — this is a defense-in-depth signal (config#1724), it
must never raise or block param loading.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import MagicMock

# arcticdb has no aarch64 wheel on some CI/dev boxes; executor/__init__.py
# imports it purely for a macOS-only side-effect priming fix (see its
# docstring), irrelevant to this module's own logic under test.
sys.modules.setdefault("arcticdb", MagicMock())

from executor import main


def _make_s3_response(body: dict, last_modified) -> dict:
    return {"Body": BytesIO(json.dumps(body).encode()), "LastModified": last_modified}


def test_check_executor_params_staleness_warns_past_threshold(caplog):
    old = datetime.now(UTC) - timedelta(hours=main._EXECUTOR_PARAMS_STALE_HOURS + 1)
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(old)
    assert any("STALE config/executor_params.json" in r.message for r in caplog.records)


def test_check_executor_params_staleness_silent_when_fresh(caplog):
    fresh = datetime.now(UTC) - timedelta(hours=1)
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(fresh)
    assert not any("STALE" in r.message for r in caplog.records)


def test_check_executor_params_staleness_silent_when_absent(caplog):
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(None)
    assert not any("STALE" in r.message for r in caplog.records)


def test_load_executor_params_from_s3_warns_on_stale_write(monkeypatch, caplog, tmp_path):
    """End-to-end: a simulated stale write on the real S3 read path logs the WARN."""
    main._executor_params_loaded = False
    main._executor_params_cache = None
    # Redirect the fault-tolerance local cache write away from the shared
    # repo-relative default path so this test can't leak a cached param set
    # into other tests / a developer's working tree via the on-disk fallback.
    monkeypatch.setattr(main, "_EXECUTOR_PARAMS_CACHE_PATH", tmp_path / "executor_params_cache.json")
    stale_time = datetime.now(UTC) - timedelta(days=30)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(
        {"min_score": 0.5}, stale_time
    )
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", mock_boto3)
    with caplog.at_level(logging.ERROR):
        main._load_executor_params_from_s3("alpha-engine-research")
    assert any("STALE config/executor_params.json" in r.message for r in caplog.records)
    main._executor_params_loaded = False
    main._executor_params_cache = None
