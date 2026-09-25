"""The v2 trader's units on this box (alpha-engine-config-I11545).

Brian's rulings of 2026-09-24: the v2 trader (crucible-trader) runs on the
executor box beside IB Gateway paper, and a shadow session counts as a
served day. These tests pin the properties that make that safe to deploy by
merge onto the box that trades v1:

* the trader never routes orders from anything this repository ships:
  routing is its private env file's decision (``off``), and no unit or script
  here names a value for it;
* each trader unit provisions itself, so nothing about the trader enters
  boot-pull.sh's 120-second, v1-gating budget or its failure count;
* the timers sit inside the box's measured up-window, in New York time,
  and the two day-bound runs cannot double up on a catch-up.
"""

from __future__ import annotations

import configparser
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SYSTEMD = ROOT / "infrastructure" / "systemd"
PROVISION = ROOT / "infrastructure" / "trader-provision.sh"
BOOT_PULL = ROOT / "infrastructure" / "boot-pull.sh"

ENV_FILE = "/home/ec2-user/alpha-engine-config/crucible-trader/trader.env"
PROVISION_ON_BOX = "/home/ec2-user/alpha-engine/infrastructure/trader-provision.sh"
TRADER_CHECKOUT = "/home/ec2-user/crucible-trader"

#: unit stem -> (ExecStart tail, OnCalendar, Persistent)
TRADER_UNITS = {
    "alpha-engine-trader-session": (
        "scripts/trader_pinned.sh daily_session",
        "Mon..Fri *-*-* 10:45:00 America/New_York",
        "false",
    ),
    "alpha-engine-trader-shadow-books": (
        "scripts/trader_pinned.sh shadow_books_daily",
        "Mon..Fri *-*-* 11:00:00 America/New_York",
        "false",
    ),
    "alpha-engine-trader-reconcile": (
        "scripts/trader_reconcile.sh",
        "Mon..Fri *-*-* 16:45:00 America/New_York",
        "true",
    ),
    "alpha-engine-trader-pin-smoke": (
        "scripts/trader_pin_request_smoke.sh",
        "Mon..Fri *-*-* 12:15:00 America/New_York",
        "false",
    ),
}

#: The units that connect to IB Gateway with the trader's ONE client id
#: (trader.env). IB refuses a second connection on an id in use, so no two of
#: these may be scheduled to overlap. Each is a oneshot bounded by
#: TimeoutStartSec; shadow-books never connects and is not listed.
BROKER_UNITS = (
    "alpha-engine-trader-session",
    "alpha-engine-trader-pin-smoke",
    "alpha-engine-trader-reconcile",
)


def _unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read(path)
    return parser


def _trader_files() -> list[Path]:
    return sorted(SYSTEMD.glob("alpha-engine-trader-*")) + [PROVISION]


def test_every_trader_unit_ships_as_a_service_and_timer_pair() -> None:
    shipped = {p.stem for p in SYSTEMD.glob("alpha-engine-trader-*")}
    assert shipped == set(TRADER_UNITS)
    for stem in TRADER_UNITS:
        assert (SYSTEMD / f"{stem}.service").is_file()
        assert (SYSTEMD / f"{stem}.timer").is_file()


@pytest.mark.parametrize("stem", sorted(TRADER_UNITS))
def test_the_service_runs_the_trader_from_its_env_file_after_provisioning(stem: str) -> None:
    service = _unit(SYSTEMD / f"{stem}.service")["Service"]
    exec_tail, _, _ = TRADER_UNITS[stem]

    assert service["Type"] == "oneshot"
    assert service["User"] == "ec2-user"
    # No leading '-': a missing env file fails the unit instead of running
    # the trader with no store, no gateway and no routing decision.
    assert service["EnvironmentFile"] == ENV_FILE
    assert service["ExecStartPre"] == f"/usr/bin/env bash {PROVISION_ON_BOX}"
    assert service["ExecStart"] == f"/usr/bin/env bash {TRADER_CHECKOUT}/{exec_tail}"
    # The checkout may not exist until ExecStartPre clones it, and systemd
    # applies WorkingDirectory to ExecStartPre as well.
    assert service["WorkingDirectory"] == "/home/ec2-user"


@pytest.mark.parametrize("stem", sorted(TRADER_UNITS))
def test_the_timer_fires_in_new_york_time_inside_the_boxs_up_window(stem: str) -> None:
    timer = _unit(SYSTEMD / f"{stem}.timer")["Timer"]
    _, on_calendar, persistent = TRADER_UNITS[stem]

    assert timer["OnCalendar"] == on_calendar
    assert timer["Persistent"] == persistent
    assert timer["Unit"] == f"{stem}.service"
    hour, minute = map(int, re.search(r"(\d\d):(\d\d):00", on_calendar).groups())
    # Box up ~08:15, stopped by the v1 postclose pipeline 17:10-17:50 New
    # York time (CloudTrail 2026-09-11 to -24).
    assert (8, 30) <= (hour, minute) <= (16, 50)


def test_the_two_day_bound_runs_never_catch_up_at_boot() -> None:
    """A boot catch-up binds to the same last closed session as that day's run."""
    for stem in (
        "alpha-engine-trader-session",
        "alpha-engine-trader-shadow-books",
        "alpha-engine-trader-pin-smoke",
    ):
        assert _unit(SYSTEMD / f"{stem}.timer")["Timer"]["Persistent"] == "false"


def _start_minutes(stem: str) -> int:
    hour, minute = map(int, re.search(r"(\d\d):(\d\d):00", TRADER_UNITS[stem][1]).groups())
    return hour * 60 + minute


def _timeout_minutes(stem: str) -> int:
    value = _unit(SYSTEMD / f"{stem}.service")["Service"]["TimeoutStartSec"]
    return int(re.fullmatch(r"(\d+)min", value).group(1))


def test_no_two_broker_units_can_hold_the_client_id_at_once() -> None:
    """Each broker unit's worst case (start + TimeoutStartSec) ends before the
    next one starts: the smoke shares the session's and the reconcile's IB
    client id, and IB refuses a second connection on an id in use."""
    windows = sorted(
        (_start_minutes(stem), _start_minutes(stem) + _timeout_minutes(stem), stem)
        for stem in BROKER_UNITS
    )
    for (_, end, earlier), (start, _, later) in zip(windows, windows[1:], strict=False):
        assert end < start, f"{earlier} can still hold the client id when {later} starts"


def test_the_pin_smoke_runs_after_the_session_and_before_the_close() -> None:
    session_end = _start_minutes("alpha-engine-trader-session") + _timeout_minutes(
        "alpha-engine-trader-session"
    )
    smoke = _start_minutes("alpha-engine-trader-pin-smoke")
    assert session_end < smoke < 16 * 60


def test_the_calendar_specs_parse(tmp_path: Path) -> None:
    analyze = subprocess.run(["which", "systemd-analyze"], capture_output=True, text=True)
    if analyze.returncode != 0:
        pytest.skip("systemd-analyze not installed")
    for stem, (_, on_calendar, _) in TRADER_UNITS.items():
        result = subprocess.run(["systemd-analyze", "calendar", on_calendar], capture_output=True, text=True)
        assert result.returncode == 0, f"{stem}: {result.stderr}"


@pytest.mark.parametrize("path", _trader_files(), ids=lambda p: p.name)
def test_nothing_shipped_here_sets_order_routing(path: Path) -> None:
    """Routing is the env file's decision, and it is off (I11545 ruling 3)."""
    text = path.read_text()
    assert not re.search(r"CRUCIBLE_TRADER_ORDER_ROUTING\s*=", text)
    assert "ib_paper" not in text


@pytest.mark.parametrize("path", _trader_files(), ids=lambda p: p.name)
def test_no_private_literals_in_this_public_repository(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"\b\d{12}\b", text), "an AWS account id"
    assert "arn:aws:" not in text
    assert "s3://" not in text


def test_the_trader_is_not_a_boot_pull_repo() -> None:
    """A trader failure must never count as a v1 boot-pull failure."""
    repos = re.search(r"REPOS=\((.*?)\)", BOOT_PULL.read_text(), re.S).group(1)
    assert "crucible-trader" not in repos


def test_provisioning_is_valid_bash_and_pins_uv() -> None:
    subprocess.run(["bash", "-n", str(PROVISION)], check=True)
    text = PROVISION.read_text()
    assert re.search(r'^UV_VERSION="\d+\.\d+\.\d+"$', text, re.M)
    assert "set -euo pipefail" in text
    # The same git-sync lock every other git writer on this box takes.
    assert 'flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK"' in text
    assert "--frozen --extra ib" in text
