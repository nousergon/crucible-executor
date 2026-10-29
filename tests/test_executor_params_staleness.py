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

from executor import main  # noqa: E402 -- must follow the arcticdb stub above


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


# ── alpha-engine-config-I12697: ground the signal in the Evaluator's apply audit ──


def _audit(outcome: str, *, blocked_by=("alpha_floor",), weeks: int = 18) -> dict:
    return {
        "schema_version": 1,
        "as_of": "2026-10-23",
        "loops": {
            "executor_params": {
                "outcome": outcome,
                "blocked_by": list(blocked_by) if blocked_by else None,
                "consecutive_blocked_weeks": weeks,
            }
        },
    }


def _s3_with_audit(audit, audit_modified):
    s3 = MagicMock()

    def _get(Bucket, Key):  # noqa: N803 -- boto3 kwarg names
        assert Key == main._APPLY_AUDIT_KEY
        if isinstance(audit, Exception):
            raise audit
        return _make_s3_response(audit, audit_modified)

    s3.get_object.side_effect = _get
    return s3


def _stale():
    return datetime.now(UTC) - timedelta(hours=main._EXECUTOR_PARAMS_STALE_HOURS + 43)


def test_declared_block_in_fresh_audit_is_warning_not_error(caplog):
    s3 = _s3_with_audit(_audit("blocked"), datetime.now(UTC) - timedelta(days=2))
    with caplog.at_level(logging.WARNING):
        main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("EXPECTED" in m and "alpha_floor" in m and "I10064" in m for m in warns)


def test_disabled_and_insufficient_are_declared_no_write(caplog):
    for outcome in ("disabled", "insufficient"):
        caplog.clear()
        s3 = _s3_with_audit(_audit(outcome, blocked_by=None), datetime.now(UTC))
        with caplog.at_level(logging.WARNING):
            main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], outcome


def test_promoted_but_pointer_did_not_move_stays_error(caplog):
    """The audit claims a write the pointer does not show: a silent no-op."""
    s3 = _s3_with_audit(_audit("promoted", blocked_by=None), datetime.now(UTC))
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("STALE config/executor_params.json" in m and "outcome=promoted" in m for m in errs)


def test_error_outcome_stays_error(caplog):
    s3 = _s3_with_audit(_audit("error", blocked_by=None), datetime.now(UTC))
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
    assert any("STALE" in r.getMessage() for r in caplog.records)


def test_stale_audit_does_not_explain_a_stale_pointer(caplog):
    """A blocked verdict from an Evaluator that has itself stopped running proves nothing."""
    old_audit = datetime.now(UTC) - timedelta(hours=main._EXECUTOR_PARAMS_STALE_HOURS + 1)
    s3 = _s3_with_audit(_audit("blocked"), old_audit)
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
    assert any("STALE" in r.getMessage() for r in caplog.records)


def test_unreadable_audit_stays_error(caplog):
    s3 = _s3_with_audit(RuntimeError("NoSuchKey"), None)
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(_stale(), s3=s3, bucket="b")
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("apply_audit unreadable" in m for m in errs)


def test_fresh_pointer_never_reads_the_audit():
    s3 = MagicMock()
    main._check_executor_params_staleness(datetime.now(UTC), s3=s3, bucket="b")
    s3.get_object.assert_not_called()


def test_load_path_consults_the_audit(monkeypatch, caplog, tmp_path):
    """End-to-end on the real read path: a stale pointer + fresh blocked audit does not page."""
    main._executor_params_loaded = False
    main._executor_params_cache = None
    monkeypatch.setattr(main, "_EXECUTOR_PARAMS_CACHE_PATH", tmp_path / "executor_params_cache.json")
    mock_s3 = MagicMock()

    def _get(Bucket, Key):  # noqa: N803
        if Key == "config/executor_params.json":
            return _make_s3_response({"min_score": 75}, _stale())
        return _make_s3_response(_audit("blocked"), datetime.now(UTC) - timedelta(days=2))

    mock_s3.get_object.side_effect = _get
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(sys.modules, "boto3", mock_boto3)
    with caplog.at_level(logging.WARNING):
        params = main._load_executor_params_from_s3("alpha-engine-research")
    assert params is not None and params.get("min_score") == 75
    assert not any("STALE" in r.getMessage() for r in caplog.records)
    assert any("EXPECTED" in r.getMessage() for r in caplog.records)
    main._executor_params_loaded = False
    main._executor_params_cache = None


# ── alpha-engine-config-I12779: the 2026-10-10 values are FIXED BY RULING ──


def test_ruled_object_is_declared_warning_not_error(caplog):
    """The one object the ruling names reads as declared, even with no audit."""
    etag = '"' + main._EXECUTOR_PARAMS_FIXED_BY_RULING["etag"] + '"'
    with caplog.at_level(logging.WARNING):
        main._check_executor_params_staleness(_stale(), etag=etag)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("DECLARED FIXED BY RULING" in m and "I12779" in m for m in warns)


def test_ruled_object_does_not_need_the_audit():
    """The declaration stands on the object identity; the retired audit is not read."""
    s3 = MagicMock()
    main._check_executor_params_staleness(
        _stale(), s3=s3, bucket="b", etag=main._EXECUTOR_PARAMS_FIXED_BY_RULING["etag"],
    )
    s3.get_object.assert_not_called()


def test_any_other_object_is_not_covered_by_the_ruling(caplog):
    """A different write is not the ruled object: the normal ERROR path stands."""
    old_audit = datetime.now(UTC) - timedelta(hours=main._EXECUTOR_PARAMS_STALE_HOURS + 1)
    s3 = _s3_with_audit(_audit("blocked"), old_audit)
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(
            _stale(), s3=s3, bucket="b", etag='"0123456789abcdef0123456789abcdef"',
        )
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("STALE config/executor_params.json" in m for m in errs)
    assert not any("DECLARED FIXED" in r.getMessage() for r in caplog.records)


def test_missing_etag_is_not_covered_by_the_ruling(caplog):
    with caplog.at_level(logging.ERROR):
        main._check_executor_params_staleness(_stale(), etag=None)
    assert any("STALE" in r.getMessage() for r in caplog.records)


def test_load_path_passes_the_etag_and_applies_the_same_values(monkeypatch, caplog, tmp_path):
    """End-to-end: the ruled object loads unchanged and does not page."""
    main._executor_params_loaded = False
    main._executor_params_cache = None
    monkeypatch.setattr(main, "_EXECUTOR_PARAMS_CACHE_PATH", tmp_path / "executor_params_cache.json")
    resp = _make_s3_response({"min_score": 75, "max_position_pct": 0.10}, _stale())
    resp["ETag"] = '"' + main._EXECUTOR_PARAMS_FIXED_BY_RULING["etag"] + '"'
    mock_s3 = MagicMock()

    def _get(Bucket, Key):  # noqa: N803
        assert Key == "config/executor_params.json", "the ruled object must not need the audit"
        return resp

    mock_s3.get_object.side_effect = _get
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(sys.modules, "boto3", mock_boto3)
    with caplog.at_level(logging.WARNING):
        params = main._load_executor_params_from_s3("alpha-engine-research")
    assert params is not None
    assert params.get("min_score") == 75 and params.get("max_position_pct") == 0.10
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("DECLARED FIXED BY RULING" in r.getMessage() for r in caplog.records)
    main._executor_params_loaded = False
    main._executor_params_cache = None
