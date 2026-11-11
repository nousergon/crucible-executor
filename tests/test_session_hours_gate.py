"""`infrastructure/session-hours-gate.sh`: the v2 session's ExecCondition= (alpha-engine-config-I13544).

Exit 0 runs the unit; 1-254 is a quiet systemd skip (Result=exec-condition,
no OnFailure=); 255 is a failure (nothing runs, the unit fails and pages).
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import stat
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from executor import v2_session_watchdog as watchdog

ROOT = Path(__file__).parent.parent
GATE = ROOT / "infrastructure" / "session-hours-gate.sh"
NY = ZoneInfo("America/New_York")

BASH = shutil.which("bash") or "/bin/bash"

RUN, SKIP, FAIL = 0, 1, 255


def _gate(*args: str, now: dt.datetime | str | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    full = {k: v for k, v in os.environ.items() if not k.startswith("SESSION_HOURS_GATE_")}
    if isinstance(now, dt.datetime):
        full["SESSION_HOURS_GATE_NOW"] = str(int(now.timestamp()))
    elif now is not None:
        full["SESSION_HOURS_GATE_NOW"] = now
    full.update(env)
    return subprocess.run([BASH, str(GATE), *args], capture_output=True, text=True, env=full, check=False)


def _ny(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=NY)


def _utc(y: int, m: int, d: int, hh: int, mm: int) -> dt.datetime:
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.UTC)


def test_the_gate_is_valid_executable_bash() -> None:
    subprocess.run([BASH, "-n", str(GATE)], check=True)
    assert GATE.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize(
    "moment",
    [
        _ny(2026, 11, 11, 9, 30),  # the 09:30 timer
        _ny(2026, 11, 11, 9, 25),  # first admitted minute
        _ny(2026, 11, 11, 12, 0),
        _ny(2026, 11, 11, 15, 59, 59),  # last admitted second
        _ny(2026, 11, 13, 10, 0),  # Friday
    ],
    ids=str,
)
def test_a_weekday_inside_the_hours_runs(moment: dt.datetime) -> None:
    proc = _gate(now=moment)
    assert proc.returncode == RUN, proc.stdout + proc.stderr
    assert "ok:" in proc.stdout


@pytest.mark.parametrize(
    "moment",
    [
        _ny(2026, 11, 11, 9, 24, 59),  # before 09:25
        _ny(2026, 11, 11, 8, 15),  # the box's boot
        _ny(2026, 11, 11, 16, 0),  # the close is exclusive
        _ny(2026, 11, 11, 16, 5),  # the hard stop is a stop, never a start
        _ny(2026, 11, 10, 20, 38),  # alpha-engine-config-I13544
        _ny(2026, 11, 9, 20, 36),  # alpha-engine-config-I13447
        _ny(2026, 11, 12, 0, 30),
    ],
    ids=str,
)
def test_a_weekday_outside_the_hours_is_skipped_with_a_reason(moment: dt.datetime) -> None:
    proc = _gate(now=moment)
    assert proc.returncode == SKIP, proc.stdout + proc.stderr
    assert "refused:" in proc.stdout and "outside" in proc.stdout
    assert moment.strftime("%Y-%m-%d %H:%M") in proc.stdout


@pytest.mark.parametrize("moment", [_ny(2026, 11, 14, 10, 0), _ny(2026, 11, 15, 12, 0)], ids=str)
def test_a_weekend_is_skipped(moment: dt.datetime) -> None:
    proc = _gate(now=moment)
    assert proc.returncode == SKIP
    assert "weekend" in proc.stdout


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        # DST ends Sun 2026-11-01. The same UTC instant is 09:30 EDT the
        # Friday before and 08:30 EST the Monday after.
        (_utc(2026, 10, 30, 13, 30), RUN),
        (_utc(2026, 11, 2, 13, 30), SKIP),
        (_utc(2026, 11, 2, 14, 30), RUN),
        (_utc(2026, 10, 30, 20, 30), SKIP),  # 16:30 EDT
        (_utc(2026, 11, 2, 20, 30), RUN),  # 15:30 EST
        # DST starts Sun 2026-03-08.
        (_utc(2026, 3, 6, 14, 26), RUN),  # 09:26 EST
        (_utc(2026, 3, 9, 14, 26), RUN),  # 10:26 EDT
        (_utc(2026, 3, 9, 13, 26), RUN),  # 09:26 EDT
        (_utc(2026, 3, 6, 13, 26), SKIP),  # 08:26 EST
        (_utc(2026, 3, 9, 20, 0), SKIP),  # 16:00 EDT
        (_utc(2026, 3, 6, 20, 0), RUN),  # 15:00 EST
    ],
    ids=str,
)
def test_new_york_time_is_used_across_dst(moment: dt.datetime, expected: int) -> None:
    assert _gate(now=moment).returncode == expected


def test_the_watchdog_restart_window_is_inside_the_gate() -> None:
    """The watchdog's starts (09:45 to ten minutes before the close) are never
    refused; its 16:05 pass only stops, so it is never gated."""
    day = dt.date(2026, 11, 11)
    first = dt.datetime.combine(day, watchdog.WINDOW_OPEN, tzinfo=NY)
    last = dt.datetime.combine(day, dt.time(16, 0), tzinfo=NY) - watchdog.WINDOW_CLOSE_BEFORE
    assert _gate(now=first).returncode == RUN
    assert _gate(now=last).returncode == RUN


@pytest.mark.parametrize("now", ["", "yesterday", "-5", "1.5"])
def test_an_unreadable_clock_fails_the_unit_never_skips_it(now: str) -> None:
    """255 is a systemd FAILURE: nothing runs and the unit pages. A gate that
    cannot tell the time must not quietly skip a trading day."""
    if now == "":
        # Empty means "use the real clock"; break the clock by hiding date.
        proc = _gate(PATH="/nonexistent")
    else:
        proc = _gate(now=now)
    assert proc.returncode == FAIL, proc.stdout + proc.stderr
    assert "ERROR" in proc.stdout


def test_a_missing_new_york_zoneinfo_fails_rather_than_reading_utc(tmp_path: Path) -> None:
    proc = _gate(now=_ny(2026, 11, 11, 9, 30), TZDIR=str(tmp_path))
    assert proc.returncode == FAIL
    assert "zoneinfo" in proc.stdout


def test_an_unknown_mode_fails() -> None:
    assert _gate("bogus", now=_ny(2026, 11, 11, 9, 30)).returncode == FAIL


# --- session-ran: the trade alert's ExecCondition= ------------------------------


def _systemctl(tmp_path: Path, body: str) -> str:
    fake = tmp_path / "systemctl"
    fake.write_text(f"#!/usr/bin/env bash\n{body}\n")
    fake.chmod(0o755)
    return str(fake)


@pytest.mark.parametrize(
    ("result", "expected"),
    [("exec-condition", SKIP), ("success", RUN), ("exit-code", RUN), ("signal", RUN), ("timeout", RUN)],
)
def test_the_trade_alert_is_skipped_only_when_the_gate_refused_the_session(
    tmp_path: Path, result: str, expected: int
) -> None:
    fake = _systemctl(
        tmp_path,
        '[ "$*" = "show -P Result alpha-engine-trader-session.service" ] || exit 9\n' + f"echo {result}",
    )
    proc = _gate("session-ran", SESSION_HOURS_GATE_SYSTEMCTL=fake)
    assert proc.returncode == expected, proc.stdout + proc.stderr


def test_an_unreadable_session_result_still_sends_the_alert(tmp_path: Path) -> None:
    fake = _systemctl(tmp_path, "exit 1")
    proc = _gate("session-ran", SESSION_HOURS_GATE_SYSTEMCTL=fake)
    assert proc.returncode == RUN
    assert "anyway" in proc.stdout
