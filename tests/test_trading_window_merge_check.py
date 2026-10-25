"""alpha-engine-config-I12640: the trading-window merge detector."""

import importlib.util
import io
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "trading_window_merge_check.py"
_spec = importlib.util.spec_from_file_location("trading_window_merge_check", _PATH)
twm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = twm  # dataclasses resolves the defining module by name
_spec.loader.exec_module(twm)


def _ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


@pytest.mark.parametrize(
    "ts, expected",
    [
        ("2026-10-23 20:12:00", True),  # crucible-executor#621, Fri — the incident
        ("2026-10-23 13:30:00", True),  # window start is inclusive
        ("2026-10-23 20:29:59", True),
        ("2026-10-23 20:30:00", False),  # window end is exclusive
        ("2026-10-23 13:29:59", False),
        ("2026-10-24 15:00:00", False),  # Saturday
        ("2026-11-26 15:00:00", False),  # Thanksgiving — NYSE closed
        ("2026-11-27 15:00:00", True),  # day after: early close, still a trading day
    ],
)
def test_window_uses_the_real_calendar(ts, expected):
    assert twm.in_trading_window(_ts(ts)) is expected


def test_naive_timestamp_is_refused():
    with pytest.raises(ValueError):
        twm.in_trading_window(datetime(2026, 10, 23, 20, 12))


def test_window_is_utc_not_eastern():
    # 2026-10-23 20:12Z is 16:12 ET — after the close, which is how "after
    # market hours" read as fine. The run rule's window is UTC.
    assert twm.in_trading_window(datetime.fromisoformat("2026-10-23T16:12:00-04:00"))


def _commits_payload(*rows):
    return json.dumps(
        [{"sha": sha, "commit": {"committer": {"date": d}, "message": msg}} for sha, d, msg in rows]
    ).encode()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_list_merges_reads_committer_date_and_filters():
    seen = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        return _Resp(
            _commits_payload(
                ("a" * 40, "2026-10-23T20:12:00Z", "feat(notifier): order-book telegram (#621)\n\nbody"),
                ("b" * 40, "2026-10-23T21:05:00Z", "docs: after hours"),
            )
        )

    merges = twm.list_merges("nousergon/crucible-executor", _ts("2026-10-23 19:00:00"), "tok", opener=opener)
    assert "sha=main" in seen["url"] and "since=2026-10-23T19:00:00Z" in seen["url"]
    assert seen["auth"] == "Bearer tok"
    hits = twm.in_window_merges(merges, is_trading_day=lambda d: d == date(2026, 10, 23))
    assert [m.sha for m in hits] == ["a" * 40]
    assert hits[0].title == "feat(notifier): order-book telegram (#621)"


def test_alert_carries_auto_fix_and_resolution_naming_the_issue():
    m = twm.Merge("nousergon/crucible-executor", "a" * 40, _ts("2026-10-23 20:12:00"), "x")
    message, auto_fix, resolution = twm.alert_text(m)
    assert "13:30-20:30Z" in message and "2026-10-23 20:12Z" in message
    assert auto_fix and resolution
    assert "alpha-engine-config-I12640" in resolution


def test_unreadable_repo_turns_the_run_red(monkeypatch):
    def boom(*a, **k):
        raise OSError("rate limited")

    monkeypatch.setattr(twm, "list_merges", boom)
    assert twm.main(["--repo", "nousergon/crucible-trader"]) == 1


def test_print_only_by_default(monkeypatch):
    m = twm.Merge("nousergon/crucible-trader", "c" * 40, _ts("2026-10-23 15:00:00"), "x")
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    def no_publish(*a, **k):
        raise AssertionError("published without --publish")

    monkeypatch.setattr(krepis.alerts, "publish", no_publish)
    assert twm.main(["--repo", "nousergon/crucible-trader"]) == 0


def test_publish_passes_lines_and_per_sha_dedup(monkeypatch):
    m = twm.Merge("nousergon/crucible-trader", "c" * 40, _ts("2026-10-23 15:00:00"), "x")
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    calls = []
    monkeypatch.setattr(krepis.alerts, "publish", lambda msg, **kw: calls.append(kw))
    assert twm.main(["--repo", "nousergon/crucible-trader", "--publish"]) == 0
    assert calls[0]["source"] == twm.ALERT_SOURCE
    assert calls[0]["dedup_key"] == f"trading-window-merge-nousergon/crucible-trader-{'c' * 40}"
    assert calls[0]["auto_fix"] and calls[0]["resolution"]
