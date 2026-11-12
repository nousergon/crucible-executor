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
    # alpha-engine-config-I12024: the day's order book, before the open.
    "alpha-engine-trader-order-book-plan": (
        "scripts/trader_pinned.sh order_book_plan",
        "Mon..Fri *-*-* 08:40:00 America/New_York",
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
    return sorted(SYSTEMD.glob("alpha-engine-trader-*")) + [
        PROVISION,
        ROOT / "infrastructure" / "session-hours-gate.sh",
    ]


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
        "alpha-engine-trader-order-book-plan",
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


GATE_ON_BOX = "/home/ec2-user/alpha-engine/infrastructure/session-hours-gate.sh"


def _service_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]


def test_the_session_refuses_a_start_outside_its_hours_before_anything_runs() -> None:
    """alpha-engine-config-I13544: a replaced timer fired the session at 20:36
    and 20:38 ET; the trader held but still wrote a held run manifest. The
    gate is an ExecCondition=, which systemd runs before ExecStartPre=, and it
    is written above ExecStartPre= so the file reads in the order it runs."""
    path = SYSTEMD / "alpha-engine-trader-session.service"
    service = _unit(path)["Service"]
    assert service["ExecCondition"] == f"/usr/bin/env bash {GATE_ON_BOX}"
    lines = _service_lines(path)
    condition = lines.index(f"ExecCondition=/usr/bin/env bash {GATE_ON_BOX}")
    assert condition < lines.index(f"ExecStartPre=/usr/bin/env bash {PROVISION_ON_BOX}")
    assert sum(line.startswith("ExecCondition=") for line in lines) == 1
    gate = ROOT / "infrastructure" / "session-hours-gate.sh"
    assert gate.is_file()
    # Only the session is gated: the other broker units run after the close.
    for stem in set(TRADER_UNITS) - {"alpha-engine-trader-session"}:
        assert "ExecCondition" not in _unit(SYSTEMD / f"{stem}.service")["Service"]


def test_a_gate_refused_session_sends_no_trade_alert() -> None:
    """systemd starts OnSuccess= on a skipped ExecCondition= too; the alert
    skips itself when the session's Result says the gate refused it."""
    service = _unit(SYSTEMD / f"{TRADE_ALERT}.service")["Service"]
    assert service["ExecCondition"] == f"/usr/bin/env bash {GATE_ON_BOX} session-ran"


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


# --- The pre-open order book (alpha-engine-config-I12024) ----------------------

ORDER_BOOK_PLAN = "alpha-engine-trader-order-book-plan"
ORDER_BOOK_ALERT = "alpha-engine-v2-order-book-alert"
NYSE_OPEN_MINUTES = 9 * 60 + 30
BOX_BOOT_MINUTES = 8 * 60 + 15


def test_the_order_book_plan_ends_before_the_open_and_overlaps_no_broker_unit() -> None:
    """Brian 2026-11-09/10: the book is saved BEFORE 09:30 ET. The plan starts
    after the ~08:15 boot, and its worst case (start + TimeoutStartSec) ends
    before the open and before the session, the pin smoke and the reconcile
    can start, so it never holds the gateway while any of them runs."""
    start = _start_minutes(ORDER_BOOK_PLAN)
    end = start + _timeout_minutes(ORDER_BOOK_PLAN)
    assert BOX_BOOT_MINUTES < start and end < NYSE_OPEN_MINUTES
    for stem in BROKER_UNITS:
        assert end < _start_minutes(stem), f"the plan can still run when {stem} starts"


def test_the_order_book_plan_connects_on_its_own_client_id() -> None:
    """IB refuses a second connection on an id in use. The plan's id is set in
    its own unit, and is none of v1's (1-3); crucible-trader refuses the
    session's and the master's ids at run time, which live in the private env
    file, not here."""
    service = (SYSTEMD / f"{ORDER_BOOK_PLAN}.service").read_text()
    ids = re.findall(r"^Environment=CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID=(\d+)$", service, re.M)
    assert len(ids) == 1 and int(ids[0]) not in {1, 2, 3}
    for stem in BROKER_UNITS:
        assert "CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID" not in (SYSTEMD / f"{stem}.service").read_text()


def test_the_pre_open_book_is_announced_by_its_own_alert_once() -> None:
    """The plan starts the order-book alert on success only: a failed plan
    pages through the notifier and announces nothing. The alert has no timer,
    so it runs only when the plan does."""
    plan = _unit(SYSTEMD / f"{ORDER_BOOK_PLAN}.service")["Unit"]
    assert plan["OnSuccess"] == f"{ORDER_BOOK_ALERT}.service"
    assert plan["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"
    unit = _unit(SYSTEMD / f"{ORDER_BOOK_ALERT}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot" and service["User"] == "ec2-user"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.v2_order_book_alert"
    assert "EnvironmentFile" not in service and "ExecStartPre" not in service
    assert unit["Unit"]["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"
    assert "OnSuccess" not in unit["Unit"]
    assert not (SYSTEMD / f"{ORDER_BOOK_ALERT}.timer").exists()
    # The session's own alert is unchanged: it never announces a book.
    session = _unit(SYSTEMD / "alpha-engine-trader-session.service")["Unit"]
    assert ORDER_BOOK_ALERT not in session["OnSuccess"] + session["OnFailure"]


# --- The market-open and market-close notices (alpha-engine-config-I12024) -----

MARKET_ALERTS = {
    # stem -> (argument, OnCalendar)
    "alpha-engine-v2-market-open-alert": ("open", "Mon..Fri *-*-* 09:40:00 America/New_York"),
    "alpha-engine-v2-market-close-alert": ("close", "Mon..Fri *-*-* 16:07:00 America/New_York"),
}
MARKET_OPEN_MINUTES = 9 * 60 + 30
SMOKE = "alpha-engine-trader-pin-smoke"
SESSION = "alpha-engine-trader-session"


def _alert_window(stem: str) -> tuple[int, int]:
    hour, minute = map(int, re.search(r"(\d\d):(\d\d):00", MARKET_ALERTS[stem][1]).groups())
    timeout = _unit(SYSTEMD / f"{stem}.service")["Service"]["TimeoutStartSec"]
    start = hour * 60 + minute
    return start, start + int(re.fullmatch(r"(\d+)min", timeout).group(1))


@pytest.mark.parametrize("stem", sorted(MARKET_ALERTS))
def test_each_market_notice_is_its_own_timer_and_never_the_sessions(stem: str) -> None:
    """Brian 2026-11-10: a Telegram at the open and at the close. Each runs on
    its own timer from the executor venv; the session's unit is not touched, so
    the trading path is exactly what it was."""
    arg, on_calendar = MARKET_ALERTS[stem]
    unit = _unit(SYSTEMD / f"{stem}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot" and service["User"] == "ec2-user"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.v2_market_alert {arg}"
    assert "EnvironmentFile" not in service and "ExecStartPre" not in service
    assert unit["Unit"]["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"
    assert "OnSuccess" not in unit["Unit"]
    timer = _unit(SYSTEMD / f"{stem}.timer")["Timer"]
    assert timer["OnCalendar"] == on_calendar
    assert timer["Persistent"] == "false"
    assert timer["Unit"] == f"{stem}.service"
    session = _unit(SYSTEMD / f"{SESSION}.service")["Unit"]
    assert stem not in session["OnSuccess"] + session["OnFailure"]


def test_the_open_notice_runs_while_the_session_trades() -> None:
    """After the 09:30 open and session start, and done long before the
    session's hard stop."""
    start, end = _alert_window("alpha-engine-v2-market-open-alert")
    session_start = _start_minutes(SESSION)
    session_end = session_start + _timeout_minutes(SESSION)
    assert MARKET_OPEN_MINUTES <= session_start < start
    assert end < session_end


def test_the_close_notice_fits_between_the_session_stop_and_the_pin_smoke() -> None:
    """It reports a session that has ended: it starts after the session's hard
    stop (start + TimeoutStartSec) and ends before the 16:10 pin smoke starts."""
    start, end = _alert_window("alpha-engine-v2-market-close-alert")
    session_end = _start_minutes(SESSION) + _timeout_minutes(SESSION)
    assert 16 * 60 <= session_end < start
    assert end < _start_minutes(SMOKE)


def test_the_market_notices_hold_their_own_read_only_client_id() -> None:
    """IB refuses a second connection on an id in use. The notices share one id
    (they run six hours apart), set in their own units and in no other unit
    here, and never v1's 1-3. The module also refuses the trader's ids from its
    private env file at run time."""
    ids: set[str] = set()
    for stem in MARKET_ALERTS:
        text = (SYSTEMD / f"{stem}.service").read_text()
        ids |= set(re.findall(r"^Environment=CRUCIBLE_V2_ALERT_IB_CLIENT_ID=(\d+)$", text, re.M))
    assert len(ids) == 1
    (alert_id,) = ids
    assert int(alert_id) not in {1, 2, 3}
    others = [p for p in SYSTEMD.glob("*.service") if p.stem not in MARKET_ALERTS]
    for path in others:
        for found in re.findall(r"CLIENT_ID=(\d+)\s*$", path.read_text(), re.M):
            assert found != alert_id, f"{path.name} uses the notices' client id"


def test_every_unit_that_pages_through_the_notifier_is_one_it_accepts() -> None:
    """The notifier refuses a unit name outside its allowlist and exits 2 without
    paging, so a unit naming it in OnFailure= must be on that list."""
    from executor.trader_unit_failed import UNIT_RE

    for path in SYSTEMD.glob("*.service"):
        on_failure = _unit(path)["Unit"].get("OnFailure", "") if _unit(path).has_section("Unit") else ""
        if f"{FAILURE_NOTIFIER}%n.service" in on_failure.split():
            assert UNIT_RE.fullmatch(path.name), f"{path.name} would not page"


@pytest.mark.parametrize("stem", sorted(MARKET_ALERTS))
def test_each_market_notice_attributes_its_sends_to_its_registry_row(stem: str) -> None:
    """The alert-tier registry row (nousergon-data playbooks.yaml,
    crucible_v2_market_open_close) is keyed on this source."""
    text = (SYSTEMD / f"{stem}.service").read_text()
    assert re.search(r"^Environment=KREPIS_EVENT_SOURCE=crucible-executor/v2-market-alert$", text, re.M)


# --- The day-end box stop (alpha-engine-config-I12024) --------------------------

DAY_END = "alpha-engine-v2-day-end"


def test_the_day_end_stop_is_a_root_oneshot_on_its_own_timer() -> None:
    """Brian 2026-11-11: the box idled ~10 hours a weekday after the v1 post-close
    stop stopped running. The box stops itself once the v2 day is done."""
    unit = _unit(SYSTEMD / f"{DAY_END}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot"
    # poweroff and `systemctl stop` need root.
    assert "User" not in service
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.v2_day_end"
    assert "EnvironmentFile" not in service and "ExecStartPre" not in service
    assert unit["Unit"]["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"
    timer = _unit(SYSTEMD / f"{DAY_END}.timer")["Timer"]
    assert timer["OnCalendar"] == "Mon..Fri *-*-* 17:30:00 America/New_York"
    assert timer["Unit"] == f"{DAY_END}.service"


def test_the_day_end_never_catches_up_at_a_later_boot() -> None:
    """A catch-up would power off a box the evening reconcile pipeline just booted."""
    assert _unit(SYSTEMD / f"{DAY_END}.timer")["Timer"]["Persistent"] == "false"


def test_the_day_end_fires_after_every_v2_unit_has_hit_its_hard_stop() -> None:
    """Nothing of the v2 day can still be running at 17:30 on a normal day: each
    unit's start + TimeoutStartSec ends first. The module waits regardless."""
    from executor import v2_day_end as day_end

    fire = _unit(SYSTEMD / f"{DAY_END}.timer")["Timer"]["OnCalendar"]
    hour, minute = map(int, re.search(r"(\d\d):(\d\d):00", fire).groups())
    assert (hour, minute) == (day_end.FIRE_AT.hour, day_end.FIRE_AT.minute)
    for stem in BROKER_UNITS:
        assert _start_minutes(stem) + _timeout_minutes(stem) < hour * 60 + minute, stem
    for stem in MARKET_ALERTS:
        assert _alert_window(stem)[1] < hour * 60 + minute, stem


def test_the_day_end_service_timeout_covers_its_own_budget() -> None:
    from executor import v2_day_end as day_end

    deadline, fire = day_end.DEADLINE, day_end.FIRE_AT
    budget = (
        (deadline.hour - fire.hour) * 60
        + deadline.minute
        - fire.minute
        + (day_end.STOP_TIMEOUT_SECONDS + day_end.SHIP_TIMEOUT_SECONDS) / 60
    )
    assert budget < _timeout_minutes(DAY_END)


def test_the_day_end_waits_for_every_v2_unit_after_the_close() -> None:
    from executor import v2_day_end as day_end

    for stem in (*BROKER_UNITS, *MARKET_ALERTS, TRADE_ALERT):
        if stem == "alpha-engine-v2-market-open-alert":
            continue
        assert f"{stem}.service" in day_end.WAIT_UNITS, stem
        assert (SYSTEMD / f"{stem}.service").is_file()
    assert f"{FAILURE_NOTIFIER}*.service" in day_end.WAIT_UNITS


def test_the_day_end_calendar_parses() -> None:
    analyze = subprocess.run(["which", "systemd-analyze"], capture_output=True, text=True)
    if analyze.returncode != 0:
        pytest.skip("systemd-analyze not installed")
    on_calendar = _unit(SYSTEMD / f"{DAY_END}.timer")["Timer"]["OnCalendar"]
    result = subprocess.run(["systemd-analyze", "calendar", on_calendar], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suffix", [".service", ".timer"])
def test_the_day_end_unit_carries_no_private_literals(suffix: str) -> None:
    text = (SYSTEMD / f"{DAY_END}{suffix}").read_text()
    assert not re.search(r"\b\d{12}\b", text)
    assert "arn:aws:" not in text and "s3://" not in text
    assert not re.search(r"\bi-[0-9a-f]{8,17}\b", text)


# --- The session watchdog (alpha-engine-config-I12024) ---------------------------

WATCHDOG = "alpha-engine-v2-session-watchdog"


def _calendars(stem: str) -> list[str]:
    text = (SYSTEMD / f"{stem}.timer").read_text()
    return re.findall(r"^OnCalendar=(.+)$", text, re.M)


def test_the_watchdog_is_an_ec2_user_oneshot_on_its_own_timer() -> None:
    """Brian 2026-11-11: a dead or stalled session is restarted the same day.
    It runs as ec2-user (state beside the notices'), starts the session through
    sudo, and is never started by the session itself."""
    unit = _unit(SYSTEMD / f"{WATCHDOG}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot" and service["User"] == "ec2-user"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.v2_session_watchdog"
    assert "EnvironmentFile" not in service and "ExecStartPre" not in service
    assert unit["Unit"]["OnFailure"] == f"{FAILURE_NOTIFIER}%n.service"
    timer = _unit(SYSTEMD / f"{WATCHDOG}.timer")["Timer"]
    assert timer["Persistent"] == "false"
    assert timer["Unit"] == f"{WATCHDOG}.service"
    session = _unit(SYSTEMD / f"{SESSION}.service")["Unit"]
    assert WATCHDOG not in session["OnSuccess"] + session["OnFailure"]


def test_the_watchdog_fires_every_five_minutes_in_market_hours_and_at_the_hard_stop() -> None:
    from executor import v2_session_watchdog as w

    calendars = _calendars(WATCHDOG)
    assert calendars == [
        "Mon..Fri *-*-* 09..15:00/5:00 America/New_York",
        "Mon..Fri *-*-* 16:05:00 America/New_York",
    ]
    assert (w.HARD_STOP_AT.hour, w.HARD_STOP_AT.minute) == (16, 5)
    # The 16:05 pass ends before the 16:10 pin smoke needs the client id.
    assert w.HARD_STOP_UNTIL.hour * 60 + w.HARD_STOP_UNTIL.minute < _start_minutes(SMOKE)
    # Its own run ends well inside the next five-minute firing.
    assert _timeout_minutes(WATCHDOG) < 5


def test_the_watchdog_only_restarts_after_the_open_notice() -> None:
    from executor import v2_session_watchdog as w

    start, _ = _alert_window("alpha-engine-v2-market-open-alert")
    assert start < w.WINDOW_OPEN.hour * 60 + w.WINDOW_OPEN.minute


def test_the_watchdog_waits_for_the_other_money_path_units() -> None:
    from executor import v2_session_watchdog as w

    for stem in BROKER_UNITS:
        if stem != SESSION:
            assert f"{stem}.service" in w.MONEY_PATH_UNITS


def test_the_watchdog_attributes_its_sends_to_its_registry_row() -> None:
    text = (SYSTEMD / f"{WATCHDOG}.service").read_text()
    assert re.search(r"^Environment=KREPIS_EVENT_SOURCE=crucible-executor/v2-session-watchdog$", text, re.M)


def test_the_watchdog_calendars_parse() -> None:
    analyze = subprocess.run(["which", "systemd-analyze"], capture_output=True, text=True)
    if analyze.returncode != 0:
        pytest.skip("systemd-analyze not installed")
    for on_calendar in _calendars(WATCHDOG):
        result = subprocess.run(["systemd-analyze", "calendar", on_calendar], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suffix", [".service", ".timer"])
def test_the_watchdog_unit_carries_no_private_literals(suffix: str) -> None:
    text = (SYSTEMD / f"{WATCHDOG}{suffix}").read_text()
    assert not re.search(r"\b\d{12}\b", text)
    assert "arn:aws:" not in text and "s3://" not in text
    assert not re.search(r"\bi-[0-9a-f]{8,17}\b", text)


# --- The box log copy (alpha-engine-config-I12024) ---------------------------------

LOG_SHIP = "alpha-engine-v2-box-log-ship"


def test_the_log_copy_is_an_ec2_user_oneshot_that_never_pages() -> None:
    """It reads every unit's journal through the journal group, writes with the
    instance role (no trader env file), and has no failure notifier."""
    unit = _unit(SYSTEMD / f"{LOG_SHIP}.service")
    service = unit["Service"]
    assert service["Type"] == "oneshot" and service["User"] == "ec2-user"
    assert service["SupplementaryGroups"] == "systemd-journal"
    assert service["ExecStart"] == f"{EXECUTOR_PYTHON} -m executor.box_log_ship"
    assert "EnvironmentFile" not in service
    assert "AWS_PROFILE" not in (SYSTEMD / f"{LOG_SHIP}.service").read_text()
    assert "OnFailure" not in unit["Unit"]
    timer = _unit(SYSTEMD / f"{LOG_SHIP}.timer")["Timer"]
    assert timer["OnCalendar"] == "*-*-* *:00/10:00"
    assert timer["Persistent"] == "false"
    assert timer["Unit"] == f"{LOG_SHIP}.service"


def test_the_log_copy_covers_every_v2_unit_on_this_box() -> None:
    import fnmatch

    from executor import box_log_ship as ship

    stems = {p.stem for p in SYSTEMD.glob("alpha-engine-trader-*.service")}
    stems |= {p.stem for p in SYSTEMD.glob("alpha-engine-v2-*.service")}
    stems |= {"boot-pull", "ibgateway"}
    for stem in stems:
        name = f"{stem}.service".replace("@.service", "@x.service")
        assert any(fnmatch.fnmatchcase(name, p) for p in ship.UNIT_PATTERNS), name


def test_the_log_copy_calendar_parses() -> None:
    analyze = subprocess.run(["which", "systemd-analyze"], capture_output=True, text=True)
    if analyze.returncode != 0:
        pytest.skip("systemd-analyze not installed")
    on_calendar = _unit(SYSTEMD / f"{LOG_SHIP}.timer")["Timer"]["OnCalendar"]
    result = subprocess.run(["systemd-analyze", "calendar", on_calendar], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "path",
    [SYSTEMD / f"{LOG_SHIP}.service", SYSTEMD / f"{LOG_SHIP}.timer", ROOT / "executor" / "box_log_ship.py"],
    ids=lambda p: p.name,
)
def test_the_log_copy_carries_no_private_literals(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"\b\d{12}\b", text)
    assert "arn:aws:" not in text and "s3://" not in text
    assert "alpha-engine-research" not in text
    assert not re.search(r"\bi-[0-9a-f]{8,17}\b", text)
