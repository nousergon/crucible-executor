"""L4515 — turnover tripwire (daily + rolling band on executed turnover).

Pins: no-breach quiet path; daily breach pages at ERROR with the run-date
dedup key; rolling breach sums the prior shadow artifacts and pages at WARN;
sentinel/unreadable prior days are excluded (not fabricated); a partial
window still alerts when its sum already breaches; the disabled flag and the
missing-metric upstream-contract case; publish failure is recorded in the
artifact block, never raised into the planner.
"""
from __future__ import annotations

import io
import json

import pytest

from executor import turnover_tripwire as tw


class _FakeS3:
    """list_objects_v2 + get_object over a dict of {key: dict-payload}."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def list_objects_v2(self, Bucket, Prefix, **kwargs):  # noqa: N803
        return {
            "Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)],
            "IsTruncated": False,
        }

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


def _shadow(date, turnover):
    diag = {} if turnover is None else {"turnover_one_way": turnover}
    return {"run_date": date, "diagnostics": diag}


_CFG = {
    "max_daily_turnover": 0.20,
    "turnover_tripwire_enabled": True,
    "turnover_tripwire_daily_multiple": 1.25,
    "turnover_tripwire_rolling_days": 5,
    "turnover_tripwire_rolling_sum_band": 0.60,
}


@pytest.fixture
def published(monkeypatch):
    calls = []
    from nousergon_lib import alerts

    monkeypatch.setattr(alerts, "publish", lambda **kw: calls.append(kw))
    return calls


def test_quiet_day_no_alert(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-09.json": _shadow("2026-06-09", 0.05),
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05}, _CFG, "bkt", "2026-06-10", s3)
    assert out["status"] == "ok"
    assert out["daily_breach"] is False and out["rolling_breach"] is False
    assert out["rolling_sum"] == pytest.approx(0.10)
    assert out["n_days_used"] == 2
    assert published == []


def test_daily_breach_pages_error(published):
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert out["daily_breach"] is True            # 0.30 > 0.20 × 1.25
    severities = [c["severity"] for c in published]
    assert "ERROR" in severities
    daily = next(c for c in published if c["severity"] == "ERROR")
    assert daily["dedup_key"] == "turnover_tripwire_daily_2026-06-10"
    assert daily["sns"] is True and daily["telegram"] is False


def test_rolling_breach_sums_prior_days_and_pages_warn(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json": _shadow(f"2026-06-{d:02d}", t)
        for d, t in [(4, 0.15), (5, 0.15), (8, 0.15), (9, 0.15)]
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.10}, _CFG, "bkt", "2026-06-10", s3)
    assert out["daily_breach"] is False           # every day under the cap…
    assert out["rolling_breach"] is True          # …but the week churned 70%
    assert out["rolling_sum"] == pytest.approx(0.70)
    assert [c["severity"] for c in published] == ["WARN"]
    assert published[0]["dedup_key"] == "turnover_tripwire_rolling_2026-06-10"


def test_window_takes_newest_n_and_ignores_future_and_latest(published):
    objs = {
        f"{tw._SHADOW_PREFIX}latest.json": _shadow("x", 9.9),       # not dated
        f"{tw._SHADOW_PREFIX}2026-06-11.json": _shadow("f", 9.9),   # future
    }
    for d in range(2, 10):  # 06-02..06-09 all small
        objs[f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json"] = _shadow(f"2026-06-{d:02d}", 0.01)
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.01}, _CFG, "bkt", "2026-06-10", _FakeS3(objs))
    assert out["n_days_used"] == 5                # today + newest 4 dated
    assert out["rolling_breach"] is False
    assert published == []


def test_sentinel_prior_day_excluded_partial_window_still_alerts(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-08.json": _shadow("2026-06-08", None),  # sentinel
        f"{tw._SHADOW_PREFIX}2026-06-09.json": _shadow("2026-06-09", 0.45),
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.20}, _CFG, "bkt", "2026-06-10", s3)
    assert out["n_days_used"] == 2                # sentinel day excluded
    assert out["rolling_breach"] is True          # 0.65 > 0.60 on a partial window
    assert [c["severity"] for c in published] == ["WARN"]


def test_disabled_and_missing_metric(published):
    off = tw.check_turnover_tripwire(
        {"turnover_one_way": 9.9}, {**_CFG, "turnover_tripwire_enabled": False},
        "bkt", "2026-06-10", _FakeS3())
    assert off == {"status": "disabled"}
    missing = tw.check_turnover_tripwire({}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert missing == {"status": "no_turnover_metric"}
    assert published == []


def test_governor_off_uses_absolute_band(published):
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.26}, {**_CFG, "max_daily_turnover": None},
        "bkt", "2026-06-10", _FakeS3())
    assert out["daily_band"] == pytest.approx(tw._DAILY_BAND_GOVERNOR_OFF)
    assert out["daily_breach"] is True


def test_publish_failure_recorded_not_raised(monkeypatch):
    from nousergon_lib import alerts

    def _boom(**kw):
        raise RuntimeError("sns down")

    monkeypatch.setattr(alerts, "publish", _boom)
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert out["daily_breach"] is True            # verdict still recorded
    assert "publish_error" in out                 # failure recorded in artifact


def test_internal_error_returns_sentinel(monkeypatch):
    class _ExplodingS3:
        def list_objects_v2(self, **kw):
            raise RuntimeError("s3 down")

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05}, _CFG, "bkt", "2026-06-10", _ExplodingS3())
    assert out["status"] == "error"               # recorded in the artifact
    assert "s3 down" in out["error"]
