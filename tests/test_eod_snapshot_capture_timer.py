"""The EOD broker snapshot is captured at the close (alpha-engine-config-I13663).

`ne-postclose-reconcile-pipeline` failed on 2026-11-13 at the NAV mark-basis
level gate: IB NetLiquidation sat $5,774 (56.7bp) below the settled rebuild.
Every IB mark was inside the day's traded range and off its close, because
the snapshot was captured at 21:15 UTC, 16:15 EST, and IB marks a position at
its last print, extended hours included. Until 2026-10-30 the v1 postclose
pipeline captured at 16:01 New York time.

These tests pin the properties the fix rests on, without touching any gate:

* the capture fires once per weekday at 16:01 New York time, in both EST and
  EDT, and never as a boot catch-up;
* it fires before every other capturer of the same write-once snapshot, so it
  is the one that sets the day's broker marks;
* at that instant the trading day it keys the snapshot by is the session
  that just closed;
* it never runs while the v2 trader session is still running.
"""

from __future__ import annotations

import configparser
import re
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from nousergon_lib.dates import now_dual

ROOT = Path(__file__).parent.parent
SYSTEMD = ROOT / "infrastructure" / "systemd"
STEM = "alpha-engine-eod-snapshot-capture"
NEW_YORK = ZoneInfo("America/New_York")
CAPTURE_ET = time(16, 1)
REGULAR_CLOSE_ET = time(16, 0)

#: The other units that capture the same snapshot when it is absent.
OTHER_CAPTURERS = (
    "alpha-engine-eod-reconcile-standalone",
    "alpha-engine-reference-rate-publish",
)

#: One EST and one EDT session, so a UTC-pinned schedule cannot pass by luck.
SESSIONS = (date(2026, 11, 13), date(2026, 7, 15))


def _unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read(path)
    return parser


def _utc_fire_time(on_calendar: str, session: date) -> datetime:
    """When a ``... HH:MM:SS UTC`` weekday OnCalendar fires on ``session``."""
    match = re.search(r"(\d\d):(\d\d):(\d\d) UTC$", on_calendar)
    assert match, f"expected a UTC wall-clock OnCalendar, got {on_calendar!r}"
    hh, mm, ss = map(int, match.groups())
    return datetime.combine(session, time(hh, mm, ss), tzinfo=UTC)


def test_the_timer_fires_at_1601_new_york_time_and_never_catches_up() -> None:
    timer = _unit(SYSTEMD / f"{STEM}.timer")["Timer"]
    assert timer["OnCalendar"] == "Mon..Fri *-*-* 16:01:00 America/New_York"
    # A catch-up at a later boot would capture an after-hours mark.
    assert timer["Persistent"] == "false"
    assert timer["Unit"] == f"{STEM}.service"


def test_the_service_runs_the_existing_capturer_after_the_gateway() -> None:
    unit = _unit(SYSTEMD / f"{STEM}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot"
    assert service["User"] == "ec2-user"
    assert service["ExecStart"] == (
        "/home/ec2-user/alpha-engine/.venv/bin/python -m executor.snapshot_capturer"
    )
    assert unit["Unit"]["After"] == "ibgateway.service"
    assert unit["Unit"]["Wants"] == "ibgateway.service"
    assert int(service["TimeoutStartSec"]) > 420  # the capturer's gateway wait
    source = (ROOT / "executor" / "snapshot_capturer.py").read_text()
    assert 'if __name__ == "__main__":' in source


def test_it_never_captures_while_the_trader_session_runs() -> None:
    condition = _unit(SYSTEMD / f"{STEM}.service")["Service"]["ExecCondition"]
    assert "! systemctl is-active --quiet alpha-engine-trader-session.service" in condition
    assert (SYSTEMD / "alpha-engine-trader-session.service").is_file()


@pytest.mark.parametrize("session", SESSIONS, ids=lambda d: d.isoformat())
def test_it_is_the_first_capturer_of_the_write_once_snapshot(session: date) -> None:
    ours = datetime.combine(session, CAPTURE_ET, tzinfo=NEW_YORK)
    for stem in OTHER_CAPTURERS:
        other = _unit(SYSTEMD / f"{stem}.timer")["Timer"]["OnCalendar"]
        assert ours < _utc_fire_time(other, session), stem


@pytest.mark.parametrize("session", SESSIONS, ids=lambda d: d.isoformat())
def test_it_fires_after_the_close_and_keys_the_session_that_closed(session: date) -> None:
    fire = datetime.combine(session, CAPTURE_ET, tzinfo=NEW_YORK)
    assert fire > datetime.combine(session, REGULAR_CLOSE_ET, tzinfo=NEW_YORK)
    # snapshot_capturer.run() keys by now_dual().trading_day when no --date
    # is passed; at 16:01 that must already be the session just closed.
    assert now_dual(now=fire.astimezone(UTC)).trading_day == session.isoformat()
