"""alpha-engine-config-I12640: the trading-window merge detector."""

import importlib.util
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


def _rows(*rows):
    return [{"sha": sha, "commit": {"committer": {"date": d}, "message": msg}} for sha, d, msg in rows]


def test_list_merges_reads_committer_date_and_filters():
    seen = {}

    def api(path, params):
        seen["path"], seen["params"] = path, params
        return _rows(
            ("a" * 40, "2026-10-23T20:12:00Z", "feat(notifier): order-book telegram (#621)\n\nbody"),
            ("b" * 40, "2026-10-23T21:05:00Z", "docs: after hours"),
        )

    merges = twm.list_merges("nousergon/crucible-executor", _ts("2026-10-23 19:00:00"), api=api)
    assert seen["path"] == "repos/nousergon/crucible-executor/commits"
    assert seen["params"]["sha"] == "main" and seen["params"]["since"] == "2026-10-23T19:00:00Z"
    hits = twm.in_window_merges(merges, is_trading_day=lambda d: d == date(2026, 10, 23))
    assert [m.sha for m in hits] == ["a" * 40]
    assert hits[0].title == "feat(notifier): order-book telegram (#621)"


def test_gh_api_builds_a_get_with_fields(monkeypatch):
    calls = []

    class _Out:
        stdout = "[]"

    monkeypatch.setattr(twm.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Out())
    assert twm._gh_api("repos/x/y/commits", {"sha": "main"}) == []
    assert calls[0] == ["gh", "api", "-X", "GET", "repos/x/y/commits", "-f", "sha=main"]


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


def _result(*, telegram_ok, sns_ok=True, dedup_skipped=False):
    from krepis.alerts import ChannelResult, PublishResult

    return PublishResult(
        sns=ChannelResult(ok=sns_ok, detail="sns"),
        telegram=ChannelResult(ok=telegram_ok, detail="telegram" if telegram_ok else "telegram not configured"),
        dedup_skipped=dedup_skipped,
    )


def test_publish_passes_lines_and_per_sha_dedup(monkeypatch):
    m = twm.Merge("nousergon/crucible-trader", "c" * 40, _ts("2026-10-23 15:00:00"), "x")
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    calls = []
    monkeypatch.setattr(twm, "missing_telegram_values", lambda **k: [])
    monkeypatch.setattr(krepis.alerts, "publish", lambda msg, **kw: calls.append(kw) or _result(telegram_ok=True))
    assert twm.main(["--repo", "nousergon/crucible-trader", "--publish"]) == 0
    assert calls[0]["source"] == twm.ALERT_SOURCE
    assert calls[0]["dedup_key"] == f"trading-window-merge-nousergon/crucible-trader-{'c' * 40}"
    assert calls[0]["auto_fix"] and calls[0]["resolution"]


# --- The 2026-10-26 miss: crucible-executor#627 merged 15:04Z, the detector
# saw it (run 40345778242), Telegram logged MISSING/MISSING, SNS carried the
# page alone, and the job stayed green. A missing channel must be red.


def _clean_env_secrets(monkeypatch):
    """Resolve secrets from the environment only, with the Telegram values unset.

    ALPHA_ENGINE_SECRETS_SOURCE=env keeps the test off SSM (no AWS call from
    CI), so this is the runner's state on 10-26: nothing in the env, nothing
    readable in SSM.
    """
    import krepis.secrets

    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setattr(krepis.secrets, "_cache", {})
    for name in twm.TELEGRAM_SECRETS:
        monkeypatch.delenv(name, raising=False)


def test_missing_telegram_values_reads_through_get_secret(monkeypatch):
    _clean_env_secrets(monkeypatch)
    assert twm.missing_telegram_values() == ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    assert twm.missing_telegram_values() == ["TELEGRAM_CHAT_ID"]
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    assert twm.missing_telegram_values() == []


def test_missing_channel_is_red_even_on_a_quiet_day(monkeypatch, capsys):
    _clean_env_secrets(monkeypatch)
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [])
    assert twm.main(["--repo", "nousergon/crucible-executor", "--publish"]) == 1
    out = capsys.readouterr().out
    assert "TELEGRAM_BOT_TOKEN=MISSING TELEGRAM_CHAT_ID=MISSING" in out
    assert "Auto-fix:" in out and "Resolution: alpha-engine-config-I12640" in out


def test_missing_channel_does_not_gate_a_print_only_run(monkeypatch):
    _clean_env_secrets(monkeypatch)
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [])
    assert twm.main(["--repo", "nousergon/crucible-executor"]) == 0


def test_in_window_merge_with_missing_channel_still_publishes_and_is_red(monkeypatch):
    # The 10-26 run exactly: SNS took the page, Telegram did not.
    _clean_env_secrets(monkeypatch)
    m = twm.Merge("nousergon/crucible-executor", "7ca2f7971" + "0" * 31, _ts("2026-10-26 15:04:51"), "deps (#627)")
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    calls = []
    monkeypatch.setattr(krepis.alerts, "publish", lambda msg, **kw: calls.append(kw) or _result(telegram_ok=False))
    assert twm.main(["--repo", "nousergon/crucible-executor", "--publish"]) == 1
    assert len(calls) == 1  # SNS still gets the page; the run is red as well


def test_sns_only_delivery_is_red(monkeypatch):
    m = twm.Merge("nousergon/crucible-trader", "c" * 40, _ts("2026-10-23 15:00:00"), "x")
    monkeypatch.setattr(twm, "missing_telegram_values", lambda **k: [])
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    monkeypatch.setattr(krepis.alerts, "publish", lambda msg, **kw: _result(telegram_ok=False))
    assert twm.main(["--repo", "nousergon/crucible-trader", "--publish"]) == 1


def test_dedup_skip_counts_as_already_paged(monkeypatch):
    m = twm.Merge("nousergon/crucible-trader", "c" * 40, _ts("2026-10-23 15:00:00"), "x")
    monkeypatch.setattr(twm, "missing_telegram_values", lambda **k: [])
    monkeypatch.setattr(twm, "list_merges", lambda *a, **k: [m])
    monkeypatch.setattr(twm, "in_window_merges", lambda ms, **k: ms)
    import krepis.alerts

    monkeypatch.setattr(
        krepis.alerts, "publish", lambda msg, **kw: _result(telegram_ok=False, sns_ok=False, dedup_skipped=True)
    )
    assert twm.main(["--repo", "nousergon/crucible-trader", "--publish"]) == 0


def test_workflow_maps_the_telegram_secrets_into_the_publish_step():
    import yaml

    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "trading-window-merge-check.yml"
    steps = yaml.safe_load(wf.read_text())["jobs"]["check"]["steps"]
    publish = [s for s in steps if "--publish" in s.get("run", "")]
    assert len(publish) == 1
    env = publish[0]["env"]
    assert env["TELEGRAM_BOT_TOKEN"] == "${{ secrets.TELEGRAM_BOT_TOKEN }}"
    assert env["TELEGRAM_CHAT_ID"] == "${{ secrets.TELEGRAM_CHAT_ID }}"
