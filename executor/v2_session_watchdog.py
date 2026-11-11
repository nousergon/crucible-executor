"""Restart a dead or stalled v2 trading session, the same day (alpha-engine-config-I12024).

Run by ``alpha-engine-v2-session-watchdog.timer`` every five minutes from 09:00
to 15:55 New York time on weekdays, and once more at 16:05. It acts only
between :data:`WINDOW_OPEN` (09:45) and ten minutes before the session's close
(:data:`WINDOW_CLOSE_BEFORE`), on an NYSE trading day, plus the 16:05 pass.

**Why this exists.** Brian 2026-11-11: the open notice said "Auto-fix: none;
nothing on the box retries or restarts the trader". His standing rule is no
holds: when a session has a problem, fix it and keep trading the same day; he
cannot receive alerts overnight, so the box must heal itself.

**What it restarts.** ``alpha-engine-trader-session.service`` when, and only
when, today's session has not finished (its entry/exit watch is not
``complete`` and its run record is not ``ok``) and either:

* the unit ended in FAILURE (``Result`` other than ``success``) with no stop
  request in its journal, or it never started today; or
* the unit is still running but its watch has not been written for
  :data:`STALE_AFTER` (the open/close notices use the same 15 minutes): it is
  stuck, so it is restarted.

**Why a rerun of the same day is safe** (crucible-trader ``order_router`` and
``order_intent``). Every order is keyed one per name per trading day, and an
immutable intent is created (create-if-absent) BEFORE it is submitted. A rerun
adopts every order the broker already knows by that key and never resends it;
a name whose intent exists but the broker does not know is UNKNOWN, refused,
and left to reconciliation. So a rerun only sends names never attempted. The
book is rebuilt against what the account actually holds, so what already
filled is not bought or sold twice.

**What it never does.**

* **Restart a session a human stopped.** ``systemctl stop`` on the session
  ends it ``inactive`` with ``Result=success`` (SIGTERM is a clean exit), and
  puts a ``Stopping ...`` job line in the unit's journal. Either one reads as
  an operator stop: reported once, never restarted. A journal it cannot read
  is treated the same way (fail closed toward the operator).
* **Restart under the kill switch.** ``trader/kill_switch.json`` engaged
  (flatten or freeze, or a hold for this day: crucible-trader
  ``assert_trading_permitted``) means no restart. An unreadable switch is read
  as engaged; an absent one reads as AccessDenied for the trader role (it has
  no ListBucket there), which is read as absent, as the notices do. The
  restarted trader itself re-reads the switch before every order.
* **Break the money-path chain.** A session that wrote its run record joined
  the money-path chain at an index N. A rerun rewrites that same record and is
  honoured at N only while no later run has claimed N+1 (crucible
  ``manifest._claim_index``); otherwise the rewrite would move it past its
  successor and break the chain the trader verifies before every session. So
  when the record carries a chain link and ``runs/_money_path/claims/`` lists
  N+1 (or cannot be listed), it does not restart.
* **Run beside another money-path job.** Not while the trader reconcile or the
  pin smoke (they share the trader's IB client id) is running or starting.
* **Restart more than :data:`MAX_RESTARTS` times a day.** Then it sends one
  message that no restarts are left, with the exact command to start it.

**The 16:05 pass.** The session's own hard stop is 16:05 ET from a 09:30
start (``TimeoutStartSec``). A session this watchdog restarted later carries a
later stop, and it shares the trader's IB client id with the 16:10 pin smoke.
So at 16:05 a session it restarted today that is still running is stopped.

Each restart sends one Telegram (Impact / Auto-fix / You). The process exits
non-zero only when a message was not delivered, which pages through
``alpha-engine-trader-failed@``. It runs as ``ec2-user`` and starts or stops
the session through ``sudo -n systemctl --no-block``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

from krepis.telegram import brief_alert
from krepis.trading_calendar import is_trading_day, previous_trading_day, session_close_et

from executor.v2_market_alert import CONSOLE_URL, ENTRY_WATCH_PREFIX, SESSION_UNIT, Unit
from executor.v2_trade_alert import RUNS_PREFIX, TRADER_ENV, Doc, Store, open_store, read_env_file

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")

#: The first minute a dead session is restarted: 15 minutes after the 09:30
#: start, five after the 09:40 open notice.
WINDOW_OPEN = dt.time(9, 45)
#: The last restart is this long before the session's close (15:50 on a full
#: day): later, a rerun could not get anything out before the 15:55 expiry.
WINDOW_CLOSE_BEFORE = dt.timedelta(minutes=10)
#: The 16:05 pass: stop a session this watchdog restarted, at the hard stop a
#: 09:30 start has, before the 16:10 pin smoke needs the client id.
HARD_STOP_AT = dt.time(16, 5)
HARD_STOP_UNTIL = dt.time(16, 9)
#: The watch rewrites its document at least every 600 s (crucible-trader
#: ``entry_timing.HEARTBEAT_S``); the open/close notices call it stopped after
#: 15 minutes (``v2_market_alert.WATCH_STALE_AFTER``). A session younger than
#: this is not judged stale: it has not had the time to write.
STALE_AFTER = dt.timedelta(minutes=15)
MAX_RESTARTS = 2

#: Other money-path jobs on this box. Never restart beside one.
MONEY_PATH_UNITS = (
    "alpha-engine-trader-reconcile.service",
    "alpha-engine-trader-pin-smoke.service",
)
KILL_SWITCH_KEY = "trader/kill_switch.json"
_PERSISTENT_MODES = frozenset({"flatten", "freeze"})
#: crucible ``keys.MONEY_PATH_CLAIMS_PREFIX`` / ``money_path_claim_key``.
CLAIMS_PREFIX = "runs/_money_path/claims/"

STATE_DIR = Path("/home/ec2-user/.local/state/crucible-v2/session-watchdog")
INSTANCE_ID_FILE = Path("/var/lib/cloud/data/instance-id")
SYSTEMCTL = "/usr/bin/systemctl"

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=60, check=False)


# --- the facts ------------------------------------------------------------------

#: ``clear``; ``engaged`` (with the reason); ``unreadable`` (read as engaged).
Gate = tuple[Literal["clear", "engaged", "unreadable"], str]


@dataclass(frozen=True)
class Facts:
    """Everything one tick decides on. Gathered by :func:`gather`; pure data."""

    now: dt.datetime
    session: Unit
    watch: Doc | None = None
    run: Doc | None = None
    #: Whether the session's journal shows a stop request since its last start:
    #: ``True``/``False``, ``None`` when the journal could not be read.
    stop_requested: bool | None = False
    busy: tuple[str, ...] = ()
    kill_switch: Gate = ("clear", "")
    chain: Gate = ("clear", "")
    restarts: int = 0
    reported: frozenset[str] = field(default_factory=lambda: frozenset[str]())


Action = Literal["none", "wait", "restart", "start", "give_up", "hold", "hard_stop"]


@dataclass(frozen=True)
class Decision:
    """What to do, why, and (for ``hold``) which once-a-day report it is."""

    action: Action
    reason: str
    report: str = ""


def _when(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _hm(moment: dt.datetime) -> str:
    return moment.astimezone(NEW_YORK).strftime("%H:%M")


def _finished(f: Facts) -> str | None:
    """Why today's session counts as done, or ``None``."""
    if f.watch is not None and f.watch.get("state") == "complete":
        return "its entry/exit watch is complete"
    if f.run is not None and f.run.get("status") == "ok":
        return "its run record is ok"
    return None


def _started(session: Unit) -> dt.datetime | None:
    """When the current (or last) run of the unit began, ExecStartPre included."""
    return session.stamp("InactiveExitTimestamp") or session.stamp("ExecMainStartTimestamp")


def _problem(f: Facts, local: dt.datetime) -> tuple[Action, str] | Decision | None:
    """``(restart|start, why)`` for a dead or stalled session; a Decision to stop there; None if healthy."""
    done = _finished(f)
    if done is not None:
        return Decision("none", f"today's session is done: {done}")
    started = _started(f.session)
    started_today = started is not None and started.astimezone(NEW_YORK).date() == local.date()
    if f.session.running:
        if started is None or f.now - started < STALE_AFTER:
            return Decision("none", "the session is running and too young to judge")
        updated = _when(None if f.watch is None else f.watch.get("updated_at_utc"))
        # A watch written before this run began is the previous run's.
        last = max(updated, started) if updated is not None else started
        if f.now - last <= STALE_AFTER:
            return Decision("none", f"the session is running; its watch wrote at {_hm(last)} ET")
        return "restart", f"the session is running but its watch has not written since {_hm(last)} ET"
    if f.session.active not in ("inactive", "failed"):
        return Decision("none", f"the session is {f.session.active}; judged on the next tick")
    if not started_today:
        return "start", "the session has not run today"
    if f.session.result == "success":
        return Decision(
            "hold",
            f"the session was stopped at {_hm(f.session.stamp('ExecMainExitTimestamp') or f.now)} ET "
            "without finishing (a clean stop is an operator stop)",
            report="operator_stop",
        )
    if f.stop_requested is None:
        return Decision(
            "hold",
            f"the session ended {f.session.result}, and its journal could not be read to rule out an operator stop",
            report="operator_stop",
        )
    if f.stop_requested:
        return Decision(
            "hold",
            f"the session ended {f.session.result} after a stop request (an operator stop)",
            report="operator_stop",
        )
    return (
        "start",
        f"the session ended {f.session.result} at {_hm(f.session.stamp('ExecMainExitTimestamp') or f.now)} ET",
    )


def decide(f: Facts) -> Decision:
    """Facts to action. Pure: every rule of the module docstring, in order."""
    local = f.now.astimezone(NEW_YORK)
    if not is_trading_day(local.date()):
        return Decision("none", f"{local.date()} is not an NYSE session")
    if HARD_STOP_AT <= local.time() <= HARD_STOP_UNTIL:
        if f.session.running and f.restarts > 0:
            return Decision("hard_stop", "a session this watchdog restarted is still running at 16:05 ET")
        return Decision("none", "16:05 pass: nothing this watchdog restarted is running")
    close = dt.datetime.combine(local.date(), session_close_et(local.date()), tzinfo=NEW_YORK)
    last = (close - WINDOW_CLOSE_BEFORE).time()
    if not WINDOW_OPEN <= local.time() <= last:
        return Decision("none", f"{local:%H:%M} ET is outside the {WINDOW_OPEN:%H:%M}-{last:%H:%M} ET window")
    found = _problem(f, local)
    if found is None:  # pragma: no cover -- _problem always answers
        return Decision("none", "healthy")
    if isinstance(found, Decision):
        return found
    action, why = found
    state, detail = f.kill_switch
    if state != "clear":
        return Decision("hold", f"{why}; not restarting: the kill switch is {state} ({detail})", report="kill_switch")
    state, detail = f.chain
    if state != "clear":
        return Decision("hold", f"{why}; not restarting: {detail}", report="chain")
    if f.busy:
        return Decision("wait", f"{why}; waiting: {', '.join(f.busy)} is running")
    if f.restarts >= MAX_RESTARTS:
        return Decision("give_up", f"{why}; {MAX_RESTARTS} restarts already used today", report="give_up")
    return Decision(action, why)


# --- gathering ------------------------------------------------------------------

_PROPERTIES = (
    "ActiveState",
    "SubState",
    "Result",
    "InactiveExitTimestamp",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
)


def session_state(run: Runner = _run) -> Unit:
    """``systemctl show`` of the session, with when its last start began."""
    proc = run([SYSTEMCTL, "show", SESSION_UNIT, "--timestamp=unix", f"--property={','.join(_PROPERTIES)}"])
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in _PROPERTIES:
            props[key] = value
    return Unit(props)


def busy_units(run: Runner = _run) -> tuple[str, ...]:
    """The other money-path units that are running or starting. Unreadable reads as busy."""
    busy: list[str] = []
    for unit in MONEY_PATH_UNITS:
        proc = run([SYSTEMCTL, "show", unit, "--property=ActiveState"])
        state = proc.stdout.strip().partition("=")[2]
        if proc.returncode != 0 or not state:
            busy.append(f"{unit} (state unreadable)")
        elif state in ("active", "activating", "deactivating", "reloading"):
            busy.append(unit)
    return tuple(busy)


def stop_requested(session: Unit, run: Runner = _run) -> bool | None:
    """Whether systemd logged a stop job for the session since its last start.

    systemd writes ``Stopping <unit or description>...`` only when a stop was
    requested (an operator's ``systemctl stop``, or a restart); a run that
    fails on its own does not. Read through ``sudo -n``, as the failure
    notifier does. ``None`` when it cannot be read.
    """
    started = _started(session)
    if started is None:
        return None
    # One second past the start: a watchdog restart's own ``Stopping`` line
    # lands in the second its new run begins, and belongs to the run before.
    since = f"@{int(started.timestamp()) + 1}"
    argv = ["sudo", "-n", "journalctl", "--unit", SESSION_UNIT, "--since", since]
    proc = run([*argv, "--output", "cat", "--no-pager"])
    if proc.returncode != 0:
        return None
    return any(line.startswith("Stopping ") for line in proc.stdout.splitlines())


def read_kill_switch(store: Store, day: str) -> Gate:
    """The kill switch as crucible-trader ``assert_trading_permitted`` reads it."""
    doc, reason = store.read(KILL_SWITCH_KEY)
    if reason in ("AccessDenied", "403"):
        return ("clear", "absent")
    if reason is not None:
        return ("unreadable", reason[:80])
    if doc is None or not doc.get("engaged"):
        return ("clear", "not engaged")
    mode = str(doc.get("mode"))
    if mode in _PERSISTENT_MODES or doc.get("trading_day") == day:
        return ("engaged", f"{mode} since {doc.get('since')}")
    return ("clear", f"a {mode} for {doc.get('trading_day')} has lapsed")


def read_chain(store: Store, run: Doc | None) -> Gate:
    """Whether a rerun can rewrite the session's run record at its own chain index."""
    link = None if run is None else run.get("money_path_link")
    if not isinstance(link, dict):
        return ("clear", "the run record is not in the money-path chain")
    index = cast(Mapping[str, object], link).get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        return ("unreadable", "the run record's money-path index is not a number")
    successor = f"{CLAIMS_PREFIX}{index + 1:08d}.json"
    try:
        paginator = store.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=store.bucket, Prefix=store.key(successor)):
            if page.get("Contents"):
                return (
                    "engaged",
                    f"a later money-path run has claimed chain index {index + 1}, so a rerun "
                    "would move the session's record past it and break the chain",
                )
    except Exception as exc:  # noqa: BLE001 -- fail closed: no restart on a chain it could not read
        return ("unreadable", f"the money-path claims could not be listed ({type(exc).__name__})")
    return ("clear", f"chain index {index + 1} is unclaimed")


@dataclass
class State:
    """Today's restarts and the once-a-day reports already sent."""

    restarts: list[dict[str, str]] = field(default_factory=lambda: list[dict[str, str]]())
    reported: list[str] = field(default_factory=lambda: list[str]())


def load_state(path: Path) -> State:
    try:
        raw: object = json.loads(path.read_text())
    except FileNotFoundError:
        return State()
    except (OSError, ValueError):
        # Unreadable counts as the cap reached: never more restarts than allowed.
        return State(restarts=[{"at": "?", "why": "state unreadable"}] * MAX_RESTARTS)
    doc = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
    restarts = doc.get("restarts")
    reported = doc.get("reported")
    return State(
        restarts=[cast(dict[str, str], r) for r in cast(list[object], restarts) if isinstance(r, dict)]
        if isinstance(restarts, list)
        else [],
        reported=[str(r) for r in cast(list[object], reported)] if isinstance(reported, list) else [],
    )


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"restarts": state.restarts, "reported": state.reported}, indent=2) + "\n")
    tmp.replace(path)


def gather(store: Store, *, now: dt.datetime, state: State, run: Runner = _run) -> Facts:
    """Read everything :func:`decide` needs. The store reads are the notices' own."""
    day = previous_trading_day(now.astimezone(NEW_YORK).date()).isoformat()
    session = session_state(run)
    watch, _ = store.read(f"{ENTRY_WATCH_PREFIX}{day}.json")
    record, _ = store.read(f"{RUNS_PREFIX}{day}/run.json")
    return Facts(
        now=now,
        session=session,
        watch=watch,
        run=record,
        stop_requested=False if session.running else stop_requested(session, run),
        busy=busy_units(run),
        kill_switch=read_kill_switch(store, day),
        chain=read_chain(store, record),
        restarts=len(state.restarts),
        reported=frozenset(state.reported),
    )


# --- messages -------------------------------------------------------------------


def start_command(instance_id: str, region: str) -> str:
    """The exact command that starts the session on this box from a laptop."""
    return (
        f"aws ssm send-command --region {region} --instance-ids {instance_id} "
        "--document-name AWS-RunShellScript "
        f"--parameters 'commands=[\"sudo systemctl start --no-block {SESSION_UNIT}\"]'"
    )


def _instance_id(path: Path = INSTANCE_ID_FILE) -> str:
    try:
        return path.read_text().strip() or "<trading box instance id>"
    except OSError:
        return "<trading box instance id>"


def message(d: Decision, *, now: dt.datetime, restarts: int, command: str) -> tuple[str, str]:
    """``(text, auto_fix)`` for a decision that is reported."""
    at = _hm(now)
    if d.action in ("restart", "start"):
        auto_fix = (
            f"restarted the session at {at} ET (restart {restarts} of {MAX_RESTARTS} today); "
            "it adopts orders already at the broker and never resends one"
        )
        body = f"*v2 trading session restarted by the watchdog*\nWhy: {d.reason}."
        return brief_alert(
            body,
            CONSOLE_URL,
            impact="orders were not going out while it was down; the rest of today's book now can",
            auto_fix=auto_fix,
            action="nothing unless another notice follows; the watchdog retries up to its daily cap.",
        ), auto_fix
    if d.action == "give_up":
        auto_fix = f"none left — {MAX_RESTARTS} restarts used"
        body = f"*v2 trading session DOWN — watchdog out of restarts*\nWhy: {d.reason}."
        return brief_alert(
            body,
            CONSOLE_URL,
            impact="orders still waiting on their triggers will not go out today",
            auto_fix=auto_fix,
            action=f"after the cause in the session's failure page is fixed, start it: {command}",
        ), auto_fix
    if d.action == "hard_stop":
        auto_fix = f"stopped the restarted session at {at} ET, the hard stop a 09:30 start has"
        body = f"*v2 trading session stopped at its hard stop*\nWhy: {d.reason}."
        return brief_alert(
            body,
            CONSOLE_URL,
            impact="orders it had not sent did not go out; those it sent are on the record",
            auto_fix=auto_fix,
            action="nothing; the 16:10 pin smoke and the 16:45 reconcile now have the client id.",
        ), auto_fix
    # hold
    auto_fix = "none — the watchdog does not restart this session"
    why = {
        "operator_stop": "the session was stopped on purpose; it is left stopped",
        "kill_switch": "the kill switch holds trading; the session is left down",
        "chain": "a rerun now would break the money-path chain the trader verifies",
    }.get(d.report, "the watchdog will not restart it")
    body = f"*v2 trading session not restarted*\nWhy: {d.reason}."
    return brief_alert(
        body,
        CONSOLE_URL,
        impact=f"no orders go out for the rest of today; {why}",
        auto_fix=auto_fix,
        action=f"to resume trading today, start it: {command}",
    ), auto_fix


# --- one tick -------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    decision: Decision
    delivered: bool = True


def act(
    d: Decision,
    *,
    now: dt.datetime,
    state: State,
    state_path: Path,
    run: Runner = _run,
    send: Callable[..., bool],
    command: str,
) -> Outcome:
    """Carry out one decision; sends at most one message."""
    if d.action in ("none", "wait"):
        return Outcome(d)
    if d.action in ("hold", "give_up"):
        if d.report in state.reported:
            return Outcome(d)
        text, auto_fix = message(d, now=now, restarts=len(state.restarts), command=command)
        delivered = send(text, auto_fix=auto_fix)
        if delivered:
            state.reported.append(d.report)
            save_state(state_path, state)
        return Outcome(d, delivered)
    if d.action == "hard_stop":
        result = run(["sudo", "-n", SYSTEMCTL, "stop", "--no-block", SESSION_UNIT])
        if result.returncode != 0:
            d = Decision("hard_stop", f"{d.reason}; the stop was refused ({result.stderr.strip()[:120]})")
        text, auto_fix = message(d, now=now, restarts=len(state.restarts), command=command)
        return Outcome(d, send(text, auto_fix=auto_fix))
    # restart / start: the attempt counts against the cap whether or not it took.
    verb = "restart" if d.action == "restart" else "start"
    state.restarts.append({"at": now.astimezone(dt.UTC).isoformat(), "why": d.reason})
    save_state(state_path, state)
    result = run(["sudo", "-n", SYSTEMCTL, verb, "--no-block", SESSION_UNIT])
    if result.returncode != 0:
        d = Decision(d.action, f"{d.reason}; but `systemctl {verb}` failed ({result.stderr.strip()[:120]})")
    text, auto_fix = message(d, now=now, restarts=len(state.restarts), command=command)
    return Outcome(d, send(text, auto_fix=auto_fix))


def tick(
    store: Store,
    *,
    now: dt.datetime,
    run: Runner = _run,
    send: Callable[..., bool],
    state_dir: Path = STATE_DIR,
    command: str = "",
) -> Outcome:
    local = now.astimezone(NEW_YORK)
    state_path = state_dir / f"{local.date().isoformat()}.json"
    state = load_state(state_path)
    if not is_trading_day(local.date()):
        return Outcome(Decision("none", f"{local.date()} is not an NYSE session"))
    d = decide(gather(store, now=now, state=state, run=run))
    return act(d, now=now, state=state, state_path=state_path, run=run, send=send, command=command)


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_session_watchdog``."""
    from executor.notifier import send_trading_notice

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print(f"usage: v2_session_watchdog; got {args!r}", file=sys.stderr)
        return 2
    moment = now or dt.datetime.now(dt.UTC)
    local = moment.astimezone(NEW_YORK)
    in_window = WINDOW_OPEN <= local.time() <= HARD_STOP_UNTIL
    if not in_window or not is_trading_day(local.date()):
        logger.info("v2 session watchdog: %s ET is not a trading-day window; nothing to do", f"{local:%Y-%m-%d %H:%M}")
        return 0
    env = read_env_file(TRADER_ENV)
    command = start_command(_instance_id(), os.environ.get("AWS_REGION", "us-east-1"))
    outcome = tick(open_store(env), now=moment, send=send_trading_notice, command=command)
    d = outcome.decision
    log = logger.info if d.action in ("none", "wait") else logger.warning
    log("v2 session watchdog: %s: %s", d.action, d.reason)
    if not outcome.delivered:
        logger.error("v2 session watchdog: the %s notice was not delivered", d.action)
        return 1
    return 0


__all__ = ["Decision", "Facts", "decide", "gather", "main", "message", "tick"]


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-session-watchdog", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
