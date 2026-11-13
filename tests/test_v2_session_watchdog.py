"""`executor.v2_session_watchdog`: restart a dead or stalled v2 session the same day (alpha-engine-config-I12024)."""

from __future__ import annotations

import datetime as dt
import io
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from executor import v2_session_watchdog as w
from executor.v2_market_alert import Unit
from executor.v2_trade_alert import Store

NY = ZoneInfo("America/New_York")
TODAY = dt.date(2026, 11, 11)  # Wednesday; the session serves Tuesday 2026-11-10
DAY = "2026-11-10"


def _at(hour: int, minute: int, day: dt.date = TODAY) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(dt.UTC)


def _stamp(moment: dt.datetime) -> str:
    return f"@{int(moment.timestamp())}"


def _iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


STARTED = _at(9, 30)


def _session(
    active: str = "activating",
    result: str = "success",
    *,
    started: dt.datetime | None = STARTED,
    ended: dt.datetime | None = None,
) -> Unit:
    props = {"ActiveState": active, "Result": result}
    if started is not None:
        props["InactiveExitTimestamp"] = _stamp(started)
        props["ExecMainStartTimestamp"] = _stamp(started)
    if ended is not None:
        props["ExecMainExitTimestamp"] = _stamp(ended)
    return Unit(props)


RUNNING = _session()
FAILED = _session("failed", "exit-code", ended=_at(11, 0))
STOPPED = _session("inactive", "success", ended=_at(11, 0))  # what `systemctl stop` leaves


def _watch(state: str = "watching", updated: dt.datetime | None = None) -> dict[str, Any]:
    return {"state": state, "updated_at_utc": _iso(updated or _at(11, 55))}


def _facts(**kw: Any) -> w.Facts:
    base: dict[str, Any] = {"now": _at(12, 0), "session": RUNNING, "watch": _watch()}
    base.update(kw)
    return w.Facts(**base)


# --- decide: the pure decision ---------------------------------------------------


def test_a_healthy_running_session_is_left_alone() -> None:
    d = w.decide(_facts())
    assert d.action == "none" and "running" in d.reason


def test_a_failed_session_is_restarted() -> None:
    d = w.decide(_facts(session=FAILED, watch=_watch("stopped", _at(10, 59))))
    assert d.action == "start"
    assert "exit-code" in d.reason


@pytest.mark.parametrize("result", ["exit-code", "signal", "core-dump", "timeout"])
def test_every_failure_result_is_restarted(result: str) -> None:
    d = w.decide(_facts(session=_session("failed", result, ended=_at(11, 0)), watch=None))
    assert d.action == "start"


def test_a_running_session_whose_watch_went_stale_is_restarted() -> None:
    d = w.decide(_facts(watch=_watch(updated=_at(11, 40))))
    assert d.action == "restart"
    assert "11:40" in d.reason


def test_a_running_session_with_no_watch_after_the_grace_is_restarted() -> None:
    d = w.decide(_facts(watch=None, now=_at(10, 0)))
    assert d.action == "restart"


def test_a_young_session_is_never_judged_stale() -> None:
    """Just (re)started: the watch on the store is the previous run's."""
    young = _session(started=_at(11, 50))
    d = w.decide(_facts(session=young, watch=_watch("stopped", _at(10, 0))))
    assert d.action == "none"


def test_a_watch_older_than_the_current_run_does_not_count_against_it() -> None:
    restarted = _session(started=_at(11, 50))
    d = w.decide(_facts(session=restarted, watch=_watch(updated=_at(10, 0)), now=_at(12, 0)))
    assert d.action == "none"
    d = w.decide(_facts(session=restarted, watch=_watch(updated=_at(10, 0)), now=_at(12, 10)))
    assert d.action == "restart"


def test_a_session_being_stopped_is_judged_on_the_next_tick() -> None:
    d = w.decide(_facts(session=_session("deactivating", "success"), watch=None))
    assert d.action == "none"


def test_a_session_that_never_ran_today_is_started() -> None:
    d = w.decide(_facts(session=_session("inactive", "success", started=None), watch=None))
    assert d.action == "start" and "not run today" in d.reason
    yesterday = _session("inactive", "success", started=_at(9, 30, dt.date(2026, 11, 10)))
    assert w.decide(_facts(session=yesterday, watch=None)).action == "start"


def test_a_start_the_hours_gate_refused_is_not_a_run_and_is_started() -> None:
    """alpha-engine-config-I13544: a start the session's ExecCondition= refused
    (say a timer fired at 08:20 ET) ends inactive with Result=exec-condition.
    Nothing ran, so it is neither an operator stop nor a failure: start it."""
    skipped = _session("inactive", w.SKIPPED_BY_GATE, started=_at(8, 20), ended=None)
    for stop_requested in (False, None, True):
        d = w.decide(_facts(session=skipped, watch=None, stop_requested=stop_requested))
        assert d.action == "start" and "hours gate" in d.reason
    # Last night's 20:38 ET refusal: today has simply not run yet.
    evening = _session("inactive", w.SKIPPED_BY_GATE, started=_at(20, 38, dt.date(2026, 11, 10)))
    assert w.decide(_facts(session=evening, watch=None)).action == "start"


def test_a_gate_refused_start_still_obeys_the_cap_and_the_kill_switch() -> None:
    skipped = _session("inactive", w.SKIPPED_BY_GATE, started=_at(8, 20))
    assert w.decide(_facts(session=skipped, watch=None, restarts=w.MAX_RESTARTS)).action == "give_up"
    d = w.decide(_facts(session=skipped, watch=None, kill_switch=("engaged", "freeze")))
    assert d.action == "hold" and d.report == "kill_switch"


# --- the completed-session no-op ----------------------------------------------------


@pytest.mark.parametrize("session", [STOPPED, FAILED, RUNNING])
def test_a_completed_watch_is_never_restarted(session: Unit) -> None:
    d = w.decide(_facts(session=session, watch=_watch("complete", _at(10, 0))))
    assert d.action == "none" and "done" in d.reason


def test_an_ok_run_record_is_never_restarted() -> None:
    """A held session (no input) is also an ok run: nothing to restart."""
    run = {"status": "ok", "metrics": [{"name": "trader_session_served", "value": 0}]}
    d = w.decide(_facts(session=_session("inactive", "success", ended=_at(9, 40)), watch=None, run=run))
    assert d.action == "none"


def test_a_failed_run_record_is_restarted() -> None:
    d = w.decide(_facts(session=FAILED, run={"status": "failed"}, watch=_watch("stopped")))
    assert d.action == "start"


# --- an operator stop is never restarted --------------------------------------------


def test_a_clean_stop_is_an_operator_stop_and_is_never_restarted() -> None:
    """`systemctl stop` on the session leaves inactive + Result=success."""
    d = w.decide(_facts(session=STOPPED, watch=_watch("stopped", _at(11, 0))))
    assert d.action == "hold" and d.report == "operator_stop"


def test_a_failure_after_a_stop_request_is_an_operator_stop() -> None:
    """The trader turns SIGTERM into an error, so a stop can end exit-code."""
    d = w.decide(_facts(session=FAILED, stop_requested=True, watch=_watch("stopped")))
    assert d.action == "hold" and d.report == "operator_stop"


def test_an_unreadable_journal_fails_closed_toward_the_operator() -> None:
    d = w.decide(_facts(session=FAILED, stop_requested=None, watch=_watch("stopped")))
    assert d.action == "hold" and d.report == "operator_stop"


def test_an_operator_stop_outranks_the_cap_and_every_other_gate() -> None:
    d = w.decide(_facts(session=STOPPED, watch=None, restarts=2, busy=("x",), kill_switch=("engaged", "freeze")))
    assert d.report == "operator_stop"


# --- the kill switch -----------------------------------------------------------------


@pytest.mark.parametrize("state", ["engaged", "unreadable"])
def test_the_kill_switch_blocks_a_restart(state: str) -> None:
    d = w.decide(_facts(session=FAILED, watch=None, kill_switch=(state, "freeze since 10:00")))
    assert d.action == "hold" and d.report == "kill_switch"


def test_the_kill_switch_blocks_a_stale_restart_too() -> None:
    d = w.decide(_facts(watch=_watch(updated=_at(11, 0)), kill_switch=("engaged", "hold")))
    assert d.action == "hold" and d.report == "kill_switch"


# --- the money-path guards -------------------------------------------------------


def test_never_beside_another_money_path_job() -> None:
    d = w.decide(_facts(session=FAILED, watch=None, busy=("alpha-engine-trader-reconcile.service",)))
    assert d.action == "wait"


@pytest.mark.parametrize("state", ["engaged", "unreadable"])
def test_a_chain_successor_blocks_a_restart(state: str) -> None:
    d = w.decide(_facts(session=FAILED, watch=None, chain=(state, "index 13 claimed")))
    assert d.action == "hold" and d.report == "chain"


# --- the restart cap -------------------------------------------------------------


def test_the_third_problem_gives_up() -> None:
    assert w.decide(_facts(session=FAILED, watch=None, restarts=1)).action == "start"
    d = w.decide(_facts(session=FAILED, watch=None, restarts=2))
    assert d.action == "give_up" and d.report == "give_up"


# --- the window ------------------------------------------------------------------


@pytest.mark.parametrize(("hour", "minute", "acts"), [(9, 40, False), (9, 45, True), (15, 50, True), (15, 55, False)])
def test_the_market_hours_window(hour: int, minute: int, acts: bool) -> None:
    d = w.decide(_facts(session=FAILED, watch=None, now=_at(hour, minute)))
    assert (d.action == "start") is acts


def test_an_early_close_ends_the_window_ten_minutes_before_it() -> None:
    # 2026-11-27, the day after Thanksgiving, closes at 13:00.
    friday = dt.date(2026, 11, 27)
    failed = _session("failed", "exit-code", started=_at(9, 30, friday), ended=_at(11, 0, friday))
    assert w.decide(_facts(session=failed, watch=None, now=_at(12, 50, friday))).action == "start"
    assert w.decide(_facts(session=failed, watch=None, now=_at(12, 55, friday))).action == "none"


@pytest.mark.parametrize("day", [dt.date(2026, 11, 26), dt.date(2026, 11, 14)])
def test_a_non_trading_day_does_nothing(day: dt.date) -> None:
    """Thanksgiving, and a Saturday."""
    failed = _session("failed", "exit-code", started=_at(9, 30, day), ended=_at(11, 0, day))
    d = w.decide(_facts(session=failed, watch=None, now=_at(12, 0, day)))
    assert d.action == "none" and "not an NYSE session" in d.reason


def test_the_hard_stop_pass_stops_only_a_session_it_restarted() -> None:
    late = _at(16, 5)
    assert w.decide(_facts(now=late, restarts=1)).action == "hard_stop"
    assert w.decide(_facts(now=late, restarts=0)).action == "none"
    assert w.decide(_facts(now=late, restarts=1, session=STOPPED)).action == "none"


# --- reading the gates ---------------------------------------------------------------


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Paginator:
    def __init__(self, keys: set[str], broken: bool) -> None:
        self.keys, self.broken = keys, broken

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        if self.broken:
            raise _ClientError("AccessDenied")
        found = [{"Key": k} for k in sorted(self.keys) if k.startswith(Prefix)]
        return [{"Contents": found}] if found else [{"KeyCount": 0}]


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, Any], *, broken_list: bool = False, failing: Sequence[str] = ()) -> None:
        self.objects, self.broken_list, self.failing = objects, broken_list, set(failing)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key in self.failing:
            raise _ClientError("InternalError")
        if Key not in self.objects:
            raise _ClientError("AccessDenied")  # no ListBucket: absent reads as denied
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}

    def get_paginator(self, _name: str) -> _Paginator:
        return _Paginator(set(self.objects), self.broken_list)


def _store(objects: dict[str, Any], **kw: Any) -> Store:
    return Store(_S3({f"crucible/{k}": v for k, v in objects.items()}, **kw), "bucket", "crucible")


def test_the_kill_switch_reads_as_the_trader_reads_it() -> None:
    assert w.read_kill_switch(_store({}), DAY)[0] == "clear"
    off = {"engaged": False, "mode": "freeze"}
    assert w.read_kill_switch(_store({w.KILL_SWITCH_KEY: off}), DAY)[0] == "clear"
    freeze = {"engaged": True, "mode": "freeze", "trading_day": "2026-10-01", "since": "x"}
    assert w.read_kill_switch(_store({w.KILL_SWITCH_KEY: freeze}), DAY)[0] == "engaged"
    hold_today = {"engaged": True, "mode": "hold", "trading_day": DAY, "since": "x"}
    assert w.read_kill_switch(_store({w.KILL_SWITCH_KEY: hold_today}), DAY)[0] == "engaged"
    hold_old = {"engaged": True, "mode": "hold", "trading_day": "2026-11-09", "since": "x"}
    assert w.read_kill_switch(_store({w.KILL_SWITCH_KEY: hold_old}), DAY)[0] == "clear"
    broken = _store({}, failing=[f"crucible/{w.KILL_SWITCH_KEY}"])
    assert w.read_kill_switch(broken, DAY)[0] == "unreadable"


def test_the_chain_gate() -> None:
    linked = {"status": "failed", "money_path_link": {"index": 12}}
    assert w.read_chain(_store({}), None)[0] == "clear"
    assert w.read_chain(_store({}), {"status": "failed"})[0] == "clear"
    assert w.read_chain(_store({}), linked)[0] == "clear"
    own = {"runs/_money_path/claims/00000012.json": {}}
    assert w.read_chain(_store(own), linked)[0] == "clear"
    successor = {"runs/_money_path/claims/00000013.json": {}}
    assert w.read_chain(_store(successor), linked)[0] == "engaged"
    assert w.read_chain(_store({}, broken_list=True), linked)[0] == "unreadable"
    assert w.read_chain(_store({}), {"money_path_link": {"index": "12"}})[0] == "unreadable"


class _Box:
    """systemctl / journalctl / sudo answers, and a record of every command."""

    def __init__(
        self,
        session: dict[str, str],
        *,
        others: dict[str, str] | None = None,
        journal: str = "",
        journal_rc: int = 0,
        start_rc: int = 0,
    ) -> None:
        self.session, self.journal, self.journal_rc, self.start_rc = session, journal, journal_rc, start_rc
        self.others = others or {}
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[:3] == ["sudo", "-n", "journalctl"]:
            return subprocess.CompletedProcess(args, self.journal_rc, stdout=self.journal, stderr="")
        if args[:3] == ["sudo", "-n", w.SYSTEMCTL]:
            return subprocess.CompletedProcess(args, self.start_rc, stdout="", stderr="denied" if self.start_rc else "")
        assert args[:2] == [w.SYSTEMCTL, "show"]
        if args[2] == w.SESSION_UNIT:
            out = "\n".join(f"{k}={v}" for k, v in self.session.items())
        else:
            out = f"ActiveState={self.others.get(args[2], 'inactive')}"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    @property
    def actions(self) -> list[list[str]]:
        return [c for c in self.calls if c[:3] == ["sudo", "-n", w.SYSTEMCTL]]


FAILED_PROPS = dict(FAILED.props)
STOPPED_PROPS = dict(STOPPED.props)


def test_stop_requested_reads_the_journal_since_the_run_began() -> None:
    box = _Box(FAILED_PROPS, journal="Starting x...\nStopping v2 trader daily session...\n")
    assert w.stop_requested(FAILED, box) is True
    since = box.calls[0][box.calls[0].index("--since") + 1]
    assert since == f"@{int(STARTED.timestamp()) + 1}"
    assert w.stop_requested(FAILED, _Box(FAILED_PROPS, journal="Starting x...\nMain process exited\n")) is False
    assert w.stop_requested(FAILED, _Box(FAILED_PROPS, journal_rc=1)) is None


def test_an_unreadable_neighbour_reads_as_busy() -> None:
    class _Broken(_Box):
        def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="")

    assert len(w.busy_units(_Broken({}))) == len(w.MONEY_PATH_UNITS)


# --- one tick, end to end ------------------------------------------------------------


class _Sends:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[str, str]] = []

    def __call__(self, text: str, *, auto_fix: str) -> bool:
        self.sent.append((text, auto_fix))
        return self.ok


CMD = w.start_command("<box>", "us-east-1")


def _tick(
    box: _Box, sends: _Sends, tmp_path: Path, *, now: dt.datetime | None = None, objects: dict[str, Any] | None = None
) -> w.Outcome:
    return w.tick(_store(objects or {}), now=now or _at(12, 0), run=box, send=sends, state_dir=tmp_path, command=CMD)


def test_a_failed_session_is_started_once_and_says_so(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS), _Sends()
    outcome = _tick(box, sends, tmp_path)
    assert outcome.decision.action == "start" and outcome.delivered
    assert box.actions == [["sudo", "-n", w.SYSTEMCTL, "start", "--no-block", w.SESSION_UNIT]]
    ((text, auto_fix),) = sends.sent
    assert "restarted the session at 12:00 ET" in auto_fix
    assert "adopts orders already at the broker" in auto_fix
    assert "Impact:" in text and "Auto-fix:" in text and "You:" in text
    state = json.loads((tmp_path / "2026-11-11.json").read_text())
    assert len(state["restarts"]) == 1


def test_a_stalled_session_is_restarted_not_started(tmp_path: Path) -> None:
    running = dict(RUNNING.props)
    watch = {f"trader/entry_watch/{DAY}.json": _watch(updated=_at(11, 30))}
    box, sends = _Box(running), _Sends()
    _tick(box, sends, tmp_path, objects=watch)
    assert box.actions == [["sudo", "-n", w.SYSTEMCTL, "restart", "--no-block", w.SESSION_UNIT]]


def test_the_cap_holds_across_ticks_and_the_give_up_is_sent_once(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS), _Sends()
    for minute in (0, 5, 10, 15, 20):
        _tick(box, sends, tmp_path, now=_at(12, minute))
    assert len(box.actions) == w.MAX_RESTARTS
    assert len(sends.sent) == w.MAX_RESTARTS + 1
    text, auto_fix = sends.sent[-1]
    assert auto_fix == "none left — 2 restarts used"
    assert "aws ssm send-command" in text and w.SESSION_UNIT in text


def test_a_refused_start_still_counts_against_the_cap(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS, start_rc=1), _Sends()
    outcome = _tick(box, sends, tmp_path)
    assert "failed" in outcome.decision.reason
    assert len(json.loads((tmp_path / "2026-11-11.json").read_text())["restarts"]) == 1


def test_an_operator_stop_is_reported_once_and_never_restarted(tmp_path: Path) -> None:
    box, sends = _Box(STOPPED_PROPS), _Sends()
    for minute in (0, 5, 10):
        outcome = _tick(box, sends, tmp_path, now=_at(12, minute))
        assert outcome.decision.report == "operator_stop"
    assert box.actions == []
    ((text, _),) = sends.sent
    assert "not restarted" in text and "stopped on purpose" in text
    assert "systemctl start --no-block" in text


def test_a_stop_request_in_the_journal_is_never_restarted(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS, journal="Stopping alpha-engine-trader-session.service - v2...\n"), _Sends()
    _tick(box, sends, tmp_path)
    assert box.actions == []


def test_the_kill_switch_is_honoured_end_to_end(tmp_path: Path) -> None:
    freeze = {w.KILL_SWITCH_KEY: {"engaged": True, "mode": "freeze", "trading_day": DAY, "since": "x"}}
    box, sends = _Box(FAILED_PROPS), _Sends()
    outcome = _tick(box, sends, tmp_path, objects=freeze)
    assert outcome.decision.report == "kill_switch"
    assert box.actions == []


def test_never_beside_the_reconcile_end_to_end(tmp_path: Path) -> None:
    box = _Box(FAILED_PROPS, others={"alpha-engine-trader-pin-smoke.service": "activating"})
    sends = _Sends()
    assert _tick(box, sends, tmp_path).decision.action == "wait"
    assert box.actions == [] and sends.sent == []


def test_a_completed_session_is_a_silent_no_op(tmp_path: Path) -> None:
    done = {f"trader/entry_watch/{DAY}.json": _watch("complete", _at(11, 0))}
    box, sends = _Box(STOPPED_PROPS), _Sends()
    assert _tick(box, sends, tmp_path, objects=done).decision.action == "none"
    assert box.actions == [] and sends.sent == []


def test_an_undelivered_notice_is_reported_and_retried(tmp_path: Path) -> None:
    box = _Box(STOPPED_PROPS)
    assert not _tick(box, _Sends(ok=False), tmp_path).delivered
    sends = _Sends()
    _tick(box, sends, tmp_path, now=_at(12, 5))
    assert len(sends.sent) == 1


def test_the_hard_stop_pass_stops_a_restarted_session(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS), _Sends()
    _tick(box, sends, tmp_path)
    box.session = dict(_session(started=_at(12, 0)).props)
    outcome = _tick(box, sends, tmp_path, now=_at(16, 5))
    assert outcome.decision.action == "hard_stop"
    assert box.actions[-1] == ["sudo", "-n", w.SYSTEMCTL, "stop", "--no-block", w.SESSION_UNIT]


def test_an_unreadable_state_file_counts_as_the_cap_reached(tmp_path: Path) -> None:
    (tmp_path / "2026-11-11.json").write_text("{not json")
    box, sends = _Box(FAILED_PROPS), _Sends()
    assert _tick(box, sends, tmp_path).decision.action == "give_up"
    assert box.actions == []


def test_main_does_nothing_outside_its_window() -> None:
    assert w.main([], now=_at(9, 0)) == 0
    assert w.main([], now=_at(12, 0, dt.date(2026, 11, 14))) == 0
    assert w.main(["x"]) == 2


def test_the_open_and_close_notices_name_the_watchdog_truthfully() -> None:
    from executor import v2_market_alert as m

    assert "session watchdog restarts" in m.OPEN_AUTO_FIX and "at most twice a day" in m.OPEN_AUTO_FIX
    assert m.CLOSE_AUTO_FIX.startswith("none tonight") and "watchdog" in m.CLOSE_AUTO_FIX
    assert "nothing on the box retries or restarts" not in m.OPEN_AUTO_FIX + m.CLOSE_AUTO_FIX
    assert w.WINDOW_OPEN == dt.time(9, 45) and w.MAX_RESTARTS == 2  # the numbers those texts state


# --- back up after a down notice (Brian 2026-11-12) ----------------------------------


def test_a_running_session_after_a_down_notice_is_announced_once() -> None:
    d = w.decide(_facts(reported=frozenset({"give_up"})))
    assert d.action == "recovered" and "after the give up notice" in d.reason


def test_a_running_session_with_no_down_notice_stays_silent() -> None:
    assert w.decide(_facts(reported=frozenset())).action == "none"


def test_a_young_or_stale_session_is_not_announced_as_back() -> None:
    young = w.decide(_facts(session=_session(started=_at(11, 55)), reported=frozenset({"give_up"})))
    assert young.action == "none"
    stale = w.decide(_facts(watch=_watch(updated=_at(11, 30)), reported=frozenset({"give_up"})))
    assert stale.action != "recovered"


def test_back_up_after_out_of_restarts_end_to_end(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS), _Sends()
    for minute in (0, 5, 10):
        _tick(box, sends, tmp_path, now=_at(11, minute))
    assert sends.sent[-1][1] == "none left — 2 restarts used"
    # A manual start at 11:30; its watch writes.
    box.session = dict(_session(started=_at(11, 30)).props)
    watch = {f"trader/entry_watch/{DAY}.json": _watch(updated=_at(11, 58))}
    outcome = _tick(box, sends, tmp_path, now=_at(12, 0), objects=watch)
    assert outcome.decision.action == "recovered" and outcome.delivered
    text, auto_fix = sends.sent[-1]
    assert "running again" in text and auto_fix == "none needed"
    assert box.actions == [["sudo", "-n", w.SYSTEMCTL, "start", "--no-block", w.SESSION_UNIT]] * 2
    # Once: the next tick is silent.
    count = len(sends.sent)
    assert _tick(box, sends, tmp_path, now=_at(12, 5), objects=watch).decision.action == "none"
    assert len(sends.sent) == count


def test_a_second_death_after_recovery_is_announced_again(tmp_path: Path) -> None:
    box, sends = _Box(FAILED_PROPS), _Sends()
    for minute in (0, 5, 10):
        _tick(box, sends, tmp_path, now=_at(11, minute))
    box.session = dict(_session(started=_at(11, 30)).props)
    watch = {f"trader/entry_watch/{DAY}.json": _watch(updated=_at(11, 58))}
    _tick(box, sends, tmp_path, now=_at(12, 0), objects=watch)
    box.session = dict(_session("failed", "exit-code", started=_at(11, 30), ended=_at(12, 30)).props)
    outcome = _tick(box, sends, tmp_path, now=_at(12, 35))
    assert outcome.decision.action == "give_up" and outcome.delivered
    assert sends.sent[-1][1] == "none left — 2 restarts used"


def test_an_undelivered_back_up_notice_is_retried(tmp_path: Path) -> None:
    state = tmp_path / "2026-11-11.json"
    state.write_text(json.dumps({"restarts": [], "reported": ["operator_stop"]}))
    box = _Box(dict(RUNNING.props))
    watch = {f"trader/entry_watch/{DAY}.json": _watch(updated=_at(11, 58))}
    assert not _tick(box, _Sends(ok=False), tmp_path, objects=watch).delivered
    sends = _Sends()
    _tick(box, sends, tmp_path, now=_at(12, 5), objects=watch)
    assert len(sends.sent) == 1 and json.loads(state.read_text())["reported"] == []
