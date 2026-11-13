"""Page that a v2 trader unit failed on this box (alpha-engine-config-I12353).

Run by ``alpha-engine-trader-failed@.service``, which every
``alpha-engine-trader-*.service`` names in ``OnFailure=``. systemd starts it
when the unit ends ``failed``, wherever it died: in ``ExecStartPre``
(``trader-provision.sh``), in the venv build inside ``trader_pinned.sh`` /
``trader_reconcile.sh``, or in the trader itself.

**Why this exists.** From 2026-10-14 to -15 four trader unit runs ended
``ActiveState=failed Result=exit-code`` and nothing paged. The crucible
sweep reads what a unit WRITES, so a unit that dies before writing anything
is invisible to it until a deadline passes, and the shadow-books and pin-smoke
units have no such deadline at all. The units log only to this box's journal,
which nobody can read once the box is stopped for the night, so the page
carries the failed run's last journal lines itself.

**What it does not do.** It never touches the trader: no retry, no restart,
no order, no store write. It reads ``systemctl show`` and ``journalctl`` for
the failed unit and publishes one ops alert through the executor's existing
path (:func:`executor.notifier.publish_ops_alert`: SNS through
``nousergon_lib.alerts``, then flow-doctor's Telegram topic). It runs from the
executor's own venv, never the trader's, so the venv build that broke the
trader cannot also break the page about it.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from zoneinfo import ZoneInfo

from executor.notifier import publish_ops_alert

logger = logging.getLogger(__name__)

#: How many of the failed unit's journal lines the page carries.
TAIL_LINES = 50
#: Telegram refuses a message over 4096 characters; the tail is cut from the
#: front so the lines nearest the failure survive.
MAX_TAIL_CHARS = 3000

#: Only the trader's own units, plus the v2 session's Telegram alert
#: (alpha-engine-config-I13284), which pages here when it could not deliver
#: its summary or read the run record, and the market-open and market-close
#: notices (alpha-engine-config-I12024), which page here when their message was
#: not delivered, the day-end box stop (alpha-engine-config-I12024), which
#: pages here when the poweroff was refused, and the session watchdog, which
#: pages here when its restart notice was not delivered. The instance name comes from
#: systemd's ``%i``, but it is still checked: this process runs
#: ``journalctl`` and ``systemctl`` on it.
UNIT_RE = re.compile(
    r"alpha-engine-(?:trader-[a-z0-9-]+|v2-(?:trade|market-open|market-close|order-book)-alert|v2-day-end|v2-session-watchdog)\.service"
)
SESSION_UNIT = "alpha-engine-trader-session.service"

_NEW_YORK = ZoneInfo("America/New_York")
_STATE_PROPERTIES = ("ActiveState", "Result", "ExecMainStatus", "ExecMainExitTimestamp")

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=60, check=False)


def unit_state(unit: str, run: Runner = _run) -> dict[str, str]:
    """The failed unit's state, as ``systemctl show`` reports it."""
    proc = run(["systemctl", "show", unit, f"--property={','.join(_STATE_PROPERTIES)}"])
    state: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in _STATE_PROPERTIES:
            state[key] = value
    return state


def journal_tail(unit: str, run: Runner = _run) -> str:
    """The unit's last :data:`TAIL_LINES` journal lines.

    Read as ``ec2-user`` first, then through ``sudo -n``: a user outside the
    ``systemd-journal`` group sees no system-unit lines. A tail that cannot
    be read is said so on the page, never left blank.
    """
    argv = ["journalctl", "--unit", unit, "--lines", str(TAIL_LINES), "--no-pager"]
    argv += ["--output", "short-iso"]
    problem = "no entries"
    for prefix in ((), ("sudo", "-n")):
        proc = run([*prefix, *argv])
        text = proc.stdout.strip()
        if proc.returncode == 0 and text and "-- No entries --" not in text:
            return text[-MAX_TAIL_CHARS:]
        problem = proc.stderr.strip()[:200] or "no entries"
    return f"(journal tail unreadable: {problem})"


def _auto_fix(unit: str) -> str:
    if unit == SESSION_UNIT:
        return (
            "the session watchdog restarts the session within 5 minutes, from 09:45 ET to "
            "10 minutes before the close, at most twice a day (never one an operator stopped, "
            "never under the kill switch); each restart sends its own notice."
        )
    return "none. The trader timers do not catch up (Persistent=false), so this run is not retried today."


def build_message(unit: str, state: dict[str, str], tail: str, now: dt.datetime) -> str:
    """One ops alert: the unit, how it ended, the tail, Auto-fix and Resolution."""
    local = now.astimezone(_NEW_YORK)
    ended = state.get("ExecMainExitTimestamp") or "unknown"
    return (
        f"*v2 trader unit FAILED: {unit}*\n"
        f"ActiveState={state.get('ActiveState', '?')} Result={state.get('Result', '?')} "
        f"ExecMainStatus={state.get('ExecMainStatus', '?')} ended {ended} "
        f"(noticed {local.strftime('%Y-%m-%d %H:%M %Z')})\n"
        f"Auto-fix: {_auto_fix(unit)}\n"
        "Resolution: read the tail below, fix the cause (provisioning, the "
        "pinned venv build or the trader); the unit's next "
        "scheduled run shows whether the fix held.\n"
        f"Last {TAIL_LINES} journal lines:\n"
        f"```\n{tail}\n```"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Runner = _run,
    now: dt.datetime | None = None,
) -> int:
    """``python -m executor.trader_unit_failed <unit>``; the unit is ``%i``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or not UNIT_RE.fullmatch(args[0]):
        print(
            "usage: trader_unit_failed "
            f"<alpha-engine-trader-*.service | alpha-engine-v2-*-alert.service | alpha-engine-v2-day-end.service>; got {args!r}",
            file=sys.stderr,
        )
        return 2
    unit = args[0]
    moment = now or dt.datetime.now(dt.UTC)
    message = build_message(unit, unit_state(unit, run), journal_tail(unit, run), moment)
    day = moment.astimezone(_NEW_YORK).date().isoformat()
    publish_ops_alert(
        message,
        severity="error",
        source="alpha-engine/executor/trader_unit_failed.py",
        dedup_key=f"trader-unit-failed:{unit}:{day}",
    )
    logger.info("trader unit failure alert published for %s", unit)
    return 0


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("trader-unit-failed", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        status = main()
    sys.exit(status)
