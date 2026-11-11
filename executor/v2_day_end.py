"""Stop the trading box once the v2 trading day is done (alpha-engine-config-I12024).

Run by ``alpha-engine-v2-day-end.timer`` at 17:30 New York time, weekdays.

**Why this exists.** Brian 2026-11-11: "Why was the instance left running? We
really need to clean crucible v2 up!" The box starts ~08:15 ET each weekday.
Its after-close stop lived in the v1 post-close pipeline
(nousergon-data ``step_function_eod.json``, ``StopTradingInstance``), which
only the v1 daemon's exit starts. That daemon is skipped since v2, so the
post-close pipeline stopped running (last execution 2026-10-30) and nothing
stopped the box after the close. The evening reconcile pipeline then found the
box already running, read it as another actor's (its ``ResolveBoxOwnership``)
and left it up, and only the 01:00 ET backstop schedule stopped it: about ten
idle hours every weekday. The v2 day is box-local (its units are timers on this
box), so the box stops itself when that day is done.

**What it waits for.** Every unit of the v2 day after the close: the session
(hard stop 16:05), the close notice (16:07), the pin smoke (16:10) and the
trader reconcile (16:45, hard stop 17:15), plus what those can start (the
session's trade alert and the failure notifier's pages) and the two v1
component-3 post-close units still on this box, whose UTC timers fire at 17:05
and 17:15 ET while New York is on EDT. It powers off only when none of them is
running or queued. Each v2 unit's own ``TimeoutStartSec`` has ended by 17:30,
so on a normal day nothing is running and the box goes down at once.

**Bounded.** It waits until :data:`DEADLINE` (18:10 ET) at the latest, which is
before the earliest the evening reconcile pipeline can start (the EOD
collection starts at 18:15 ET). Past it, an idle box is the worse outcome: it
stops what is still running (``systemctl stop``, so nothing is running when the
power goes), logs why in this unit's journal, and powers off anyway. It sends no
alert of its own: a box stop is not a chat event, and a unit it had to stop
reports through that unit's own failure path.

**Whose box it stops: the morning's, and only that.** The evening reconcile
pipeline, an operator replay or a heal replay may boot this box for its own
work, and they decide box ownership from the state they found it in. This must
never take down a box one of them started. Two guards, both fail-closed toward
leaving the box up (the 01:00 ET backstop schedule still stops it):

* **The boot was this morning's.** The box booted on the same New York day,
  before :data:`MORNING_BOOT_CUTOFF` (12:00 ET). The pre-open pipeline boots it
  ~08:15; anything booted from noon on was booted by someone else for their own
  reason. A boot after 17:30 cannot fire the timer at all (``Persistent=false``).
* **It fired on time.** It acts only between :data:`FIRE_AT` and
  :data:`LATE_LIMIT` (17:45 ET). A run outside that window (a timer re-armed by
  a ``daemon-reload``, alpha-engine-config-I13447; a stray manual start) is not
  the day end, and by then an evening actor may be using the box.

**Its logs leave first.** Just before the poweroff it runs
:mod:`executor.box_log_ship` once, bounded by :data:`SHIP_TIMEOUT_SECONDS`, in a
child process so this module stays stdlib-only. A failed or slow copy is a
warning; it never holds the box up.

Power-off is an OS ``poweroff``; the instance's shutdown behaviour is ``stop``,
so the box stops, it does not terminate. Stdlib only: a broken executor venv
package cannot keep the box up.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")
SYSTEMCTL = "/usr/bin/systemctl"
PROC_UPTIME = Path("/proc/uptime")

#: The timer's OnCalendar time (``alpha-engine-v2-day-end.timer``).
FIRE_AT = dt.time(17, 30)
#: A run that starts later than this is not the day end; it does nothing.
LATE_LIMIT = dt.time(17, 45)
#: The wait ends here at the latest; the box then powers off anyway.
DEADLINE = dt.time(18, 10)
#: A boot at or after this New York time was not the morning's boot.
MORNING_BOOT_CUTOFF = dt.time(12, 0)
POLL_SECONDS = 15
STOP_TIMEOUT_SECONDS = 120
#: The final log copy (:func:`ship_logs`) is abandoned after this long.
SHIP_TIMEOUT_SECONDS = 90

#: The units that may still be working at the day end (``systemctl`` glob
#: patterns). The box never powers off while one is running or queued.
WAIT_UNITS = (
    # The v2 day after the close.
    "alpha-engine-trader-session.service",
    "alpha-engine-v2-market-close-alert.service",
    "alpha-engine-trader-pin-smoke.service",
    "alpha-engine-trader-reconcile.service",
    # What those start: the session's trade alert, and a failure page in flight.
    "alpha-engine-v2-trade-alert.service",
    "alpha-engine-trader-failed@*.service",
    # v1 component-3 post-close units (21:05 / 21:15 UTC: 17:05 / 17:15 ET in EDT).
    "alpha-engine-eod-reconcile-standalone.service",
    "alpha-engine-reference-rate-publish.service",
)
_BUSY_STATES = frozenset({"active", "activating", "deactivating", "reloading", "refreshing"})

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=STOP_TIMEOUT_SECONDS + 30, check=False)


def ship_logs() -> None:
    """Copy the day's logs off the box once more (``executor.box_log_ship``), best-effort.

    Never raises and never waits past :data:`SHIP_TIMEOUT_SECONDS`."""
    try:
        subprocess.run(
            [sys.executable, "-m", "executor.box_log_ship"],
            capture_output=True,
            timeout=SHIP_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - a log copy never holds the box up
        logger.warning("v2 day end: final log copy skipped: %s: %s", type(exc).__name__, exc)


def read_uptime(path: Path = PROC_UPTIME) -> float:
    """Seconds since boot, from ``/proc/uptime``."""
    return float(path.read_text().split()[0])


def _matches(unit: str) -> bool:
    return any(fnmatch.fnmatchcase(unit, pattern) for pattern in WAIT_UNITS)


def busy_units(run: Runner = _run) -> list[str]:
    """The wait units that are running or have a queued job, sorted.

    Fails closed: if systemd cannot be asked, that is reported as busy, so the
    box is never powered off on a reading it did not get."""
    busy: set[str] = set()
    units = run([SYSTEMCTL, "list-units", "--all", "--plain", "--no-legend", "--full", "--type=service", *WAIT_UNITS])
    jobs = run([SYSTEMCTL, "list-jobs", "--no-legend", "--full"])
    if units.returncode != 0 or jobs.returncode != 0:
        return ["<systemctl unreadable>"]
    for line in units.stdout.splitlines():
        cols = line.split()
        # UNIT LOAD ACTIVE SUB DESCRIPTION...
        if len(cols) >= 4 and _matches(cols[0]) and cols[2] in _BUSY_STATES:
            busy.add(cols[0])
    for line in jobs.stdout.splitlines():
        cols = line.split()
        # JOB UNIT TYPE STATE
        if len(cols) >= 2 and _matches(cols[1]):
            busy.add(cols[1])
    return sorted(busy)


@dataclass(frozen=True)
class Verdict:
    """Whether this run may stop the box, and why."""

    act: bool
    reason: str


def guard(now: dt.datetime, uptime_seconds: float) -> Verdict:
    """The two ownership guards (module docstring): on time, and the morning's boot."""
    local = now.astimezone(NEW_YORK)
    if not FIRE_AT <= local.time() <= LATE_LIMIT:
        return Verdict(
            False,
            f"started at {local:%H:%M} ET, outside the {FIRE_AT:%H:%M}-{LATE_LIMIT:%H:%M} day-end window; "
            "not the day end, so the box is left to the backstop stop",
        )
    booted = (now - dt.timedelta(seconds=uptime_seconds)).astimezone(NEW_YORK)
    if booted.date() != local.date() or booted.time() >= MORNING_BOOT_CUTOFF:
        return Verdict(
            False,
            f"box booted {booted:%Y-%m-%d %H:%M} ET, not this morning (before {MORNING_BOOT_CUTOFF:%H:%M} ET "
            f"on {local:%Y-%m-%d}); another actor started it, so the box is left running",
        )
    return Verdict(True, f"box booted {booted:%H:%M} ET this morning")


def _deadline(now: dt.datetime) -> dt.datetime:
    local = now.astimezone(NEW_YORK)
    return dt.datetime.combine(local.date(), DEADLINE, tzinfo=NEW_YORK)


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Runner = _run,
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    uptime: Callable[[], float] = read_uptime,
    sleep: Callable[[float], None] = time.sleep,
    ship: Callable[[], None] = ship_logs,
) -> int:
    """``python -m executor.v2_day_end [--dry-run]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["--dry-run"]):
        print(f"usage: v2_day_end [--dry-run]; got {args!r}", file=sys.stderr)
        return 2
    dry_run = args == ["--dry-run"]

    start = clock()
    verdict = guard(start, uptime())
    if not verdict.act:
        logger.warning("v2 day end: not stopping the box: %s", verdict.reason)
        return 0
    logger.info("v2 day end: %s; waiting for the evening's units", verdict.reason)

    deadline = _deadline(start)
    busy = busy_units(run)
    while busy and clock() < deadline:
        logger.info("v2 day end: still running or queued: %s", ", ".join(busy))
        sleep(POLL_SECONDS)
        busy = busy_units(run)

    if busy:
        reason = (
            f"v2 day end: {', '.join(busy)} still running or queued at the "
            f"{DEADLINE:%H:%M} ET deadline; stopping them and powering the box off anyway rather than idle"
        )
        logger.error(reason)
        if not dry_run:
            stoppable = [u for u in busy if not u.startswith("<")]
            if stoppable:
                run([SYSTEMCTL, "stop", *stoppable])
    else:
        logger.info("v2 day end: the v2 day is done; powering the box off")

    if dry_run:
        logger.info("v2 day end: --dry-run, not powering off")
        return 0
    try:
        ship()
    except Exception as exc:  # noqa: BLE001 - a log copy never holds the box up
        logger.warning("v2 day end: final log copy failed: %s", exc)
    result = run([SYSTEMCTL, "--no-block", "poweroff"])
    if result.returncode != 0:
        logger.error("v2 day end: poweroff failed (%s): %s", result.returncode, result.stderr.strip())
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    sys.exit(main())
