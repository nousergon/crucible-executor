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
        "Mon..Fri *-*-* 09:30:00 America/New_York",
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
        # alpha-engine-config-I12376: no boot catch-up (see the timer).
        "false",
    ),
    "alpha-engine-trader-pin-smoke": (
        "scripts/trader_pin_request_smoke.sh",
        "Mon..Fri *-*-* 16:10:00 America/New_York",
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


#: The failure notifier (alpha-engine-config-I12353): a systemd template the
#: four trader units name in ``OnFailure=``. Not a trader unit and has no timer.
FAILURE_NOTIFIER = "alpha-engine-trader-failed@"
EXECUTOR_PYTHON = "/home/ec2-user/alpha-engine/.venv/bin/python"
#: The v2 session's Telegram alert (alpha-engine-config-I13005).
TRADE_ALERT = "alpha-engine-v2-trade-alert"


def test_every_trader_unit_ships_as_a_service_and_timer_pair() -> None:
    shipped = {p.stem for p in SYSTEMD.glob("alpha-engine-trader-*")} - {FAILURE_NOTIFIER}
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
    """A boot catch-up binds to the same last closed session as that day's run.

    The reconcile is in the list since alpha-engine-config-I12376: the v1
    postclose pipeline runs it at the close, and a catch-up at an evening boot
    after 00:00 UTC would re-file that session with none of its executions."""
    for stem in (
        "alpha-engine-trader-session",
        "alpha-engine-trader-shadow-books",
        "alpha-engine-trader-pin-smoke",
        "alpha-engine-trader-reconcile",
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


def test_the_pin_smoke_runs_after_the_session_and_before_the_reconcile() -> None:
    session_end = _start_minutes("alpha-engine-trader-session") + _timeout_minutes(
        "alpha-engine-trader-session"
    )
    smoke = _start_minutes("alpha-engine-trader-pin-smoke")
    smoke_end = smoke + _timeout_minutes("alpha-engine-trader-pin-smoke")
    assert session_end < smoke and smoke_end < _start_minutes("alpha-engine-trader-reconcile")


def test_the_session_outlives_its_last_entry_trigger() -> None:
    """Brian 2026-11-09: the session holds its buys for v1's entry triggers, the
    last at 15:55 ET (crucible-trader `entry_timing`). It starts at the open
    (Brian 2026-11-10; systemd never fires a calendar timer early) and its hard stop must leave room for that last buy's fill wait."""
    start = _start_minutes("alpha-engine-trader-session")
    session_end = start + _timeout_minutes("alpha-engine-trader-session")
    assert 9 * 60 + 30 <= start
    assert session_end >= 15 * 60 + 55 + 10


@pytest.mark.parametrize("stem", sorted(TRADER_UNITS))
def test_every_trader_unit_pages_through_the_notifier_when_it_fails(stem: str) -> None:
    """alpha-engine-config-I12353: four runs ended `failed` on 2026-10-14/15
    and nothing paged, because each died before writing the record the sweep
    reads. `%n` hands the notifier the failed unit's own full name."""
    unit = _unit(SYSTEMD / f"{stem}.service")
    expected = f"{FAILURE_NOTIFIER}%n.service"
    if stem == "alpha-engine-trader-session":
        expected += f" {TRADE_ALERT}.service"
    assert unit["Unit"]["OnFailure"] == expected


def test_every_session_ending_sends_its_trade_alerts() -> None:
    """alpha-engine-config-I13005: the first v2 orders (2026-11-04) sent no
    Telegram. The session starts the alert on success and on failure; the
    alert reads the store, never the trader's env as its own environment."""
    session = _unit(SYSTEMD / "alpha-engine-trader-session.service")["Unit"]
    assert session["OnSuccess"] == f"{TRADE_ALERT}.service"
    assert f"{TRADE_ALERT}.service" in session["OnFailure"].split()
    path = SYSTEMD / f"{TRADE_ALERT}.service"
    unit = _unit(path)
    service = unit["Service"]
    assert service["Type"] == "oneshot"
    assert service["User"] == "ec2-user"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.v2_trade_alert"
    assert "EnvironmentFile" not in service
    assert "ExecStartPre" not in service
    # It chains nothing on success, and fails only into the notifier (below).
    assert "OnSuccess" not in unit["Unit"]
    assert not (SYSTEMD / f"{TRADE_ALERT}.timer").exists()


def test_an_undelivered_trade_alert_pages_through_the_notifier() -> None:
    """alpha-engine-config-I13284: the alert exits non-zero when its summary
    was not delivered or the run record or fills were unreadable; that must page.
    Only the notifier: never itself, so a failing alert cannot loop."""
    unit = _unit(SYSTEMD / f"{TRADE_ALERT}.service")["Unit"]
    assert unit["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"


def test_the_notifier_only_reports_from_the_executor_venv() -> None:
    """It runs from the executor's venv, never the trader's (the venv build
    that broke the trader must not break the page), and it changes nothing
    about the trader: no ExecStartPre, no provisioning, no trader env file."""
    path = SYSTEMD / f"{FAILURE_NOTIFIER}.service"
    unit = _unit(path)
    service = unit["Service"]
    assert service["Type"] == "oneshot"
    assert service["User"] == "ec2-user"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.trader_unit_failed %i"
    assert "ExecStartPre" not in service
    assert "EnvironmentFile" not in service
    assert "OnFailure" not in unit["Unit"]  # a failed notifier must not recurse
    assert TRADER_CHECKOUT not in path.read_text()
    assert not list(SYSTEMD.glob(f"{FAILURE_NOTIFIER}*.timer"))


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


def test_provisioning_clones_as_the_box_not_as_the_trader() -> None:
    """alpha-engine-config-I11545: trader.env's AWS_PROFILE reaches the
    ExecStartPre too, and the clone's credential helper cannot read the
    GitHub App as the trader role. The profile is dropped before any git."""
    text = PROVISION.read_text()
    unset = text.index("unset AWS_PROFILE AWS_CONFIG_FILE")
    assert unset < text.index("git clone")
    assert unset < text.index("fetch --quiet origin main")


def test_a_changed_timer_is_stopped_before_its_new_file_lands() -> None:
    """alpha-engine-config-I13447. systemd schedules a calendar timer from its
    in-memory last trigger, which survives daemon-reload whatever Persistent=
    says. Reloading a moved OnCalendar on a box that was up since the morning
    fired the session at 20:36 ET. Stopping the changed timer first clears that
    trigger. The later `enable --now` then schedules it from now."""
    text = BOOT_PULL.read_text()
    body = text[text.index("sync_systemd_units_from() {") :]
    body = body[: body.index("\n}\n")]
    changed = body[body.index('elif ! diff -q "$unit" "$target"') :]
    changed = changed[: changed.index("CHANGED=true")]
    stop = changed.index('*.timer) sudo systemctl stop "$name"')
    assert stop < changed.index('sudo cp "$unit" /etc/systemd/system/')
    assert body.index("sudo systemctl daemon-reload") < body.index(
        'sudo systemctl enable --now "$tname"'
    )
    # A NEW timer has no last trigger to clear; only the changed branch stops.
    new = body[body.index('if [ ! -f "$target" ]; then') : body.index('elif ! diff -q')]
    assert "systemctl stop" not in new
