"""The day-end box stop (alpha-engine-config-I12024): executor.v2_day_end."""

from __future__ import annotations

import ast
import datetime as dt
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from executor import v2_day_end as m

NY = m.NEW_YORK
#: Wednesday 2026-11-11, 17:30 ET (EST).
FIRE = dt.datetime(2026, 11, 11, 17, 30, tzinfo=NY)
MORNING_BOOT = dt.datetime(2026, 11, 11, 8, 15, tzinfo=NY)


def _uptime(boot: dt.datetime, now: dt.datetime) -> float:
    return (now - boot).total_seconds()


class _Box:
    """A fake systemctl: ``busy`` is a list of unit sets, one per poll."""

    def __init__(self, polls: Sequence[Sequence[str]] = ((),), *, jobs: Sequence[str] = (), fail: bool = False):
        self.polls = list(polls)
        self.jobs = list(jobs)
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        out, code = "", 0
        if argv[1] == "list-units":
            busy = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
            out = "".join(f"{u} loaded active running x\n" for u in busy)
            out += "alpha-engine-trader-session.service loaded inactive dead v2 session\n"
            out += "alpha-engine-trader-failed@x.service loaded failed failed notifier\n"
            code = 1 if self.fail else 0
        elif argv[1] == "list-jobs":
            out = "".join(f"12 {u} start waiting\n" for u in self.jobs)
            out += "13 unrelated.service start waiting\n"
        return subprocess.CompletedProcess(argv, code, out, "")

    @property
    def poweroffs(self) -> int:
        return sum(1 for c in self.calls if c[1:] == ["--no-block", "poweroff"])

    @property
    def stops(self) -> list[list[str]]:
        return [c for c in self.calls if c[1] == "stop"]


class _Clock:
    def __init__(self, start: dt.datetime):
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)


def _main(box: _Box, *, now: dt.datetime = FIRE, boot: dt.datetime = MORNING_BOOT, argv: list[str] | None = None):
    clock = _Clock(now)
    status = m.main(
        [] if argv is None else argv,
        run=box,
        clock=clock,
        uptime=lambda: _uptime(boot, now),
        sleep=clock.sleep,
    )
    return status, clock, []


def test_a_quiet_day_end_powers_off_at_once() -> None:
    box = _Box()
    status, clock, pages = _main(box)
    assert status == 0
    assert box.poweroffs == 1 and box.stops == [] and pages == []
    assert clock.now == FIRE


def test_it_waits_while_a_v2_unit_is_running_and_powers_off_once_it_ends() -> None:
    reconcile = "alpha-engine-trader-reconcile.service"
    box = _Box([[reconcile], [reconcile], []])
    status, clock, _ = _main(box)
    assert status == 0 and box.poweroffs == 1 and box.stops == []
    assert clock.now == FIRE + dt.timedelta(seconds=2 * m.POLL_SECONDS)
    # Never a poweroff while the unit was still running.
    first_poweroff = next(i for i, c in enumerate(box.calls) if c[1:] == ["--no-block", "poweroff"])
    assert first_poweroff == len(box.calls) - 1


def test_a_queued_job_counts_as_busy() -> None:
    box = _Box(jobs=["alpha-engine-trader-pin-smoke.service"])
    status, _, pages = _main(box)
    # The job never leaves the queue: the deadline forces the stop.
    assert status == 0 and box.poweroffs == 1
    assert box.stops == [[m.SYSTEMCTL, "stop", "alpha-engine-trader-pin-smoke.service"]]
    assert pages == []


def test_a_failure_page_in_flight_is_waited_for() -> None:
    box = _Box([["alpha-engine-trader-failed@alpha-engine-trader-reconcile.service.service"], []])
    _, clock, _ = _main(box)
    assert clock.now > FIRE and box.poweroffs == 1


def test_units_outside_the_wait_set_do_not_hold_the_box() -> None:
    box = _Box([["ibgateway.service", "alpha-engine-daemon.service"]])
    _, clock, _ = _main(box)
    assert clock.now == FIRE and box.poweroffs == 1


def test_at_the_deadline_it_stops_what_still_runs_then_powers_off_and_pages() -> None:
    stuck = "alpha-engine-trader-reconcile.service"
    box = _Box([[stuck]])
    status, clock, pages = _main(box)
    assert status == 0
    deadline = dt.datetime.combine(FIRE.date(), m.DEADLINE, tzinfo=NY)
    assert clock.now >= deadline
    assert box.stops == [[m.SYSTEMCTL, "stop", stuck]]
    stop_at = box.calls.index([m.SYSTEMCTL, "stop", stuck])
    assert box.calls[-1] == [m.SYSTEMCTL, "--no-block", "poweroff"] and stop_at < len(box.calls) - 1
    assert pages == []  # logged, never sent to chat


def test_an_unreadable_systemd_is_busy_never_idle() -> None:
    box = _Box(fail=True)
    _, clock, pages = _main(box)
    deadline = dt.datetime.combine(FIRE.date(), m.DEADLINE, tzinfo=NY)
    assert clock.now >= deadline and box.poweroffs == 1
    assert box.stops == [] and pages == []


@pytest.mark.parametrize(
    "boot",
    [
        dt.datetime(2026, 11, 11, 12, 0, tzinfo=NY),  # an afternoon boot
        dt.datetime(2026, 11, 11, 17, 20, tzinfo=NY),  # an operator / replay boot just before
        dt.datetime(2026, 11, 10, 20, 37, tzinfo=NY),  # last night's reconcile boot, never stopped
        dt.datetime(2026, 11, 10, 8, 15, tzinfo=NY),  # yesterday's morning boot
    ],
)
def test_a_box_not_booted_this_morning_is_left_running(boot: dt.datetime) -> None:
    box = _Box()
    status, _, pages = _main(box, boot=boot)
    assert status == 0
    assert box.calls == [] and pages == []


@pytest.mark.parametrize(
    "now",
    [
        dt.datetime(2026, 11, 11, 17, 46, tzinfo=NY),  # fired late
        dt.datetime(2026, 11, 11, 20, 36, tzinfo=NY),  # a reload re-armed the timer (I13447)
        dt.datetime(2026, 11, 11, 17, 29, tzinfo=NY),  # a manual start before the day end
    ],
)
def test_a_run_outside_the_day_end_window_does_nothing(now: dt.datetime) -> None:
    box = _Box()
    status, _, _ = _main(box, now=now)
    assert status == 0 and box.calls == []


def test_the_window_holds_in_edt_too() -> None:
    summer = dt.datetime(2026, 7, 15, 21, 31, tzinfo=dt.UTC)  # 17:31 EDT
    boot = dt.datetime(2026, 7, 15, 12, 15, tzinfo=dt.UTC)  # 08:15 EDT
    assert m.guard(summer, (summer - boot).total_seconds()).act


def test_dry_run_never_powers_off_nor_stops() -> None:
    box = _Box([["alpha-engine-trader-reconcile.service"]])
    status, _, pages = _main(box, argv=["--dry-run"])
    assert status == 0 and box.poweroffs == 0 and box.stops == [] and pages == []


def test_a_refused_poweroff_fails_the_unit_so_it_pages() -> None:
    class _Refuses(_Box):
        def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
            done = super().__call__(argv)
            if list(argv)[1:] == ["--no-block", "poweroff"]:
                return subprocess.CompletedProcess(list(argv), 1, "", "Access denied")
            return done

    status, _, _ = _main(_Refuses())
    assert status == 1


def test_bad_arguments_are_refused() -> None:
    box = _Box()
    status, _, _ = _main(box, argv=["--now"])
    assert status == 2 and box.calls == []


def test_the_uptime_reader(tmp_path: Path) -> None:
    proc = tmp_path / "uptime"
    proc.write_text("33300.52 120000.00\n")
    assert m.read_uptime(proc) == pytest.approx(33300.52)


def test_the_decision_path_is_stdlib_only() -> None:
    """A broken venv package must not keep the box up, and it sends no alert."""
    tree = ast.parse(Path(m.__file__).read_text())
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert top_level <= set(sys.stdlib_module_names) | {"__future__"}
    assert "publish_ops_alert" not in Path(m.__file__).read_text()


def test_the_deadline_ends_before_the_evening_reconcile_pipeline_can_start() -> None:
    """The EOD collection that starts the evening reconcile pipeline begins at 18:15 ET."""
    assert m.FIRE_AT < m.LATE_LIMIT < m.DEADLINE < dt.time(18, 15)
    assert m.MORNING_BOOT_CUTOFF < m.FIRE_AT
