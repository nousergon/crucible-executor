"""One Telegram per v2 order the moment its intraday trigger releases it (alpha-engine-config-I12024).

Run every five minutes in the session by ``alpha-engine-v2-session-watchdog.service``
(a second, ``-``-prefixed ``ExecStart=`` on that oneshot, so this module failing
never stops the watchdog), not by a timer of its own.

**Why this exists.** Brian 2026-11-18: the day's four v2 orders were released by
their timing triggers at 09:45, 09:49, 10:11 and 10:30 ET, and all four Telegram
alerts arrived together at 10:31, because the per-fill alerts
(:mod:`executor.v2_trade_alert`) run only when the session unit ends. It read as
if the timing mechanism was not being used. He asked for each trade to be
alerted when its trigger fires, saying why the trigger was pulled.

**What it reads.** The session's own entry/exit watch,
``trader/entry_watch/{book day}.json``, which crucible-trader rewrites on every
release and on a heartbeat, read as the trader's role exactly as the watchdog
reads it (book day = the trading day before today, New York). An entry whose
``status`` is ``triggered`` or ``sent`` has been released to the broker; for
each one not yet alerted it sends ONE message to the ``#trades`` topic through
:func:`executor.notifier.send_trading_notice` (never flow-doctor, so no daily
cap holds it back): symbol, side and shares, the release time in ET, the
release price against the price the order was planned at, and the trigger in
plain words (:func:`plain_trigger`). It says "order sent", never "filled": the
fill is confirmed by the end-of-session per-fill alert.

**Never twice.** The order keys already alerted are kept in a per-day state
file under :data:`STATE_DIR`, written atomically after every delivered
message. A message that was not delivered is not recorded, so the next tick
retries it (and every one after it, which this tick then does not try), and
the process exits non-zero. A state file that cannot be read
sends nothing (a repeat is the failure this file exists to prevent) and exits
non-zero.

**Quiet otherwise.** Nothing on a day that is not an NYSE session, outside
:data:`WINDOW_OPEN` to :data:`WINDOW_CLOSE`, or when the watch is absent, is
for another session date, or released nothing.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from krepis.telegram import INFORMATIONAL, brief_alert
from krepis.trading_calendar import is_trading_day, previous_trading_day

from executor.v2_market_alert import CONSOLE_URL, ENTRY_WATCH_PREFIX
from executor.v2_trade_alert import TRADER_ENV, Doc, Store, open_store, read_env_file

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")

#: The session starts at 09:30; its first release can be the same minute.
WINDOW_OPEN = dt.time(9, 30)
#: The watchdog's last pass is 16:05 (its ``HARD_STOP_UNTIL``): it catches the
#: 15:55 expiry releases.
WINDOW_CLOSE = dt.time(16, 9)
#: Statuses of an entry the trader has released to the broker
#: (crucible-trader ``entry_timing.EntryStatus``): ``triggered`` is written the
#: moment the order goes out, ``sent`` once it has.
RELEASED = frozenset({"triggered", "sent"})
STATE_DIR = Path("/home/ec2-user/.local/state/crucible-v2/release-alert")

#: The default clock times of the graduated window and the expiry
#: (crucible-trader ``EntryTimingParams``); the watch's own schedule wins.
_GRADUATED_DEFAULT = "15:30"
_EXPIRY_DEFAULT = "15:55"


# --- the trigger, in plain words ------------------------------------------------

_NUM = r"([+-]?\d+(?:\.\d+)?)"
_RALLY = re.compile(rf"^rally {_NUM}% off low {_NUM}$")
_PULLBACK = re.compile(rf"^pullback {_NUM}% from high {_NUM}$")
_VWAP = re.compile(rf"^vwap_discount {_NUM}% below VWAP {_NUM}$")
_SUPPORT = re.compile(rf"^support_bounce {_NUM}% above support {_NUM}$")
_GRADUATED = re.compile(rf"^graduated_(entry|exit) \({_NUM}% vs sized price {_NUM}, limit {_NUM}%\)$")


def _pct(text: str) -> str:
    return f"{abs(float(text)):.1f}%"


def _usd(value: float | str) -> str:
    return f"${float(value):,.2f}"


def _signed_pct(text: str) -> str:
    return f"{float(text):+.1f}%"


def plain_trigger(trigger: object, *, side: str = "buy", graduated_at: str = "", expires_at: str = "") -> str:
    """crucible-trader's trigger text (``entry_timing``) as a sentence; the raw text if unrecognised.

    ``rally 2.08% off low 691.25`` reads "sold into a 2.1% rally off the day's
    low of $691.25". ``graduated_at`` / ``expires_at`` are the day's ET clock
    times (shifted on a half day); 15:30 and 15:55 when not given.
    """
    raw = str(trigger or "").strip()
    if not raw:
        return "no trigger recorded"
    verb = "sold" if side == "sell" else "bought"
    if m := _RALLY.match(raw):
        return f"sold into a {_pct(m[1])} rally off the day's low of {_usd(m[2])}"
    if m := _PULLBACK.match(raw):
        return f"bought on a {_pct(m[1])} pullback from the day's high of {_usd(m[2])}"
    if m := _VWAP.match(raw):
        return f"bought {_pct(m[1])} below today's VWAP of {_usd(m[2])}"
    if m := _SUPPORT.match(raw):
        return f"bought on a bounce {_pct(m[1])} above its support level (its recent low) of {_usd(m[2])}"
    if m := _GRADUATED.match(raw):
        at = graduated_at or _GRADUATED_DEFAULT
        side_verb = "sold" if m[1] == "exit" else "bought"
        return (
            f"{side_verb} in the {at} ET graduated window: the price was {_signed_pct(m[2])} against "
            f"the planned {_usd(m[3])}, inside its {_signed_pct(m[4])} limit"
        )
    if raw == "time_expiry":
        at = expires_at or _EXPIRY_DEFAULT
        return f"{verb} at the {at} ET expiry: no other trigger fired, and nothing waits past it"
    if raw == "catastrophic_gap_stop":
        return "sold at the start on the catastrophic gap stop: hard risk never waits for a trigger"
    if raw.startswith("adopted"):
        return "already at the broker from an earlier run of today's session"
    # Unrecognised: the trader's own words. No ``*``, which would open a bold entity.
    return raw.replace("*", "")


# --- reading the watch -----------------------------------------------------------


def _hm_et(value: object) -> str:
    """``HH:MM`` ET of an ISO instant, or ``""``."""
    if not isinstance(value, str) or not value:
        return ""
    try:
        moment = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.UTC)
    return moment.astimezone(NEW_YORK).strftime("%H:%M")


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def entries(watch: Mapping[str, object] | None) -> list[Doc]:
    """The watch's entries (one per order); anything else reads as none."""
    raw = None if watch is None else watch.get("entries")
    if not isinstance(raw, list):
        return []
    return [cast(Doc, e) for e in cast(list[object], raw) if isinstance(e, dict)]


def released(watch: Mapping[str, object] | None) -> list[Doc]:
    """The entries the trader has released to the broker, in release order."""
    found = [e for e in entries(watch) if e.get("status") in RELEASED]
    return sorted(found, key=lambda e: str(e.get("released_at_utc") or ""))


def order_key(entry: Mapping[str, object]) -> str:
    return str(entry.get("order_key") or f"?|{entry.get('symbol')}")


def schedule_times(watch: Mapping[str, object] | None) -> tuple[str, str]:
    """``(graduated window start, expiry)`` as ``HH:MM`` ET, from the watch's own schedule."""
    raw = None if watch is None else watch.get("schedule")
    schedule: Mapping[str, object] = cast(Mapping[str, object], raw) if isinstance(raw, dict) else {}
    return _hm_et(schedule.get("graduated_at")), _hm_et(schedule.get("expires_at"))


def why(entry: Mapping[str, object], watch: Mapping[str, object] | None) -> str:
    """The entry's trigger in plain words, with the day's own window times."""
    graduated_at, expires_at = schedule_times(watch)
    return plain_trigger(
        entry.get("trigger"),
        side=str(entry.get("side") or "buy"),
        graduated_at=graduated_at,
        expires_at=expires_at,
    )


def shares(entry: Mapping[str, object]) -> int:
    """The shares released (a buy can be clipped to the cash), else the planned quantity."""
    for name in ("released_quantity", "quantity"):
        value = entry.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return 0


def price_line(entry: Mapping[str, object]) -> str:
    """``$705.62 vs planned $721.90 (-2.3%)``; what is missing is said so."""
    release, planned = _float(entry.get("release_price")), _float(entry.get("reference_price"))
    if release is None and planned is None:
        return "price not recorded"
    if release is None:
        return f"release price not recorded (planned {_usd(cast(float, planned))})"
    if planned is None:
        return f"{_usd(release)} (no planned price recorded)"
    return f"{_usd(release)} vs planned {_usd(planned)} ({(release / planned - 1.0) * 100.0:+.1f}%)"


def release_time_et(entry: Mapping[str, object]) -> str:
    """``10:11 ET``, or ``time not recorded``."""
    at = _hm_et(entry.get("released_at_utc"))
    return f"{at} ET" if at else "time not recorded"


def release_message(entry: Mapping[str, object], watch: Mapping[str, object] | None) -> str:
    """The one Telegram for a released order: what, when, at what price, and why."""
    side = "SELL" if entry.get("side") == "sell" else "BUY"
    emoji = "\U0001f534" if side == "SELL" else "\U0001f7e2"
    when = release_time_et(entry)
    body = "\n".join(
        [
            f"{emoji} *{side} {str(entry.get('symbol')).replace('*', '')}: order sent* (not yet a fill)",
            f"Shares: {shares(entry):,} · Released: {when}",
            f"Price: {price_line(entry)}",
            f"Why: {why(entry, watch)}",
        ]
    )
    return brief_alert(body, CONSOLE_URL, auto_fix=INFORMATIONAL)


# --- state ------------------------------------------------------------------------


class StateUnreadable(Exception):
    """The day's state file exists and cannot be read: nothing is sent."""


def load_alerted(path: Path) -> set[str]:
    """The order keys already alerted today; raises :class:`StateUnreadable`."""
    try:
        raw: object = json.loads(path.read_text())
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        raise StateUnreadable(f"{path}: {type(exc).__name__}") from exc
    alerted = cast(dict[str, object], raw).get("alerted") if isinstance(raw, dict) else None
    if not isinstance(alerted, list):
        raise StateUnreadable(f"{path}: no 'alerted' list")
    return {str(k) for k in cast(list[object], alerted)}


def save_alerted(path: Path, alerted: Iterable[str]) -> None:
    """Write atomically: a crash mid-write leaves the previous file whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"alerted": sorted(alerted)}, indent=2) + "\n")
    tmp.replace(path)


# --- one tick ---------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What one tick did."""

    sent: int = 0
    undelivered: int = 0
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.undelivered > 0 or self.reason.startswith("state unreadable")


def in_window(now: dt.datetime) -> bool:
    local = now.astimezone(NEW_YORK)
    return is_trading_day(local.date()) and WINDOW_OPEN <= local.time() <= WINDOW_CLOSE


def tick(
    store: Store,
    *,
    now: dt.datetime,
    send: Callable[..., bool],
    state_dir: Path = STATE_DIR,
) -> Outcome:
    """Alert every released order not alerted yet; one message each."""
    local = now.astimezone(NEW_YORK)
    if not in_window(now):
        return Outcome(reason=f"{local:%Y-%m-%d %H:%M} ET is not in a trading-day window")
    day = previous_trading_day(local.date()).isoformat()
    watch, error = store.read(f"{ENTRY_WATCH_PREFIX}{day}.json")
    if watch is None:
        return Outcome(reason=f"no entry watch for book day {day} ({error or 'absent'})")
    session_date = watch.get("session_date")
    if isinstance(session_date, str) and session_date != local.date().isoformat():
        return Outcome(reason=f"the watch for {day} is for session {session_date}, not today")
    state_path = state_dir / f"{local.date().isoformat()}.json"
    try:
        alerted = load_alerted(state_path)
    except StateUnreadable as exc:
        return Outcome(reason=f"state unreadable ({exc}); nothing sent so nothing is sent twice")
    sent = undelivered = 0
    for entry in released(watch):
        key = order_key(entry)
        if key in alerted:
            continue
        if send(release_message(entry, watch), auto_fix=INFORMATIONAL):
            alerted.add(key)
            save_alerted(state_path, alerted)
            sent += 1
            continue
        # Telegram refused it on every route: the rest would wait out the same
        # timeouts inside the watchdog's start budget. All retry next tick.
        undelivered = sum(1 for e in released(watch) if order_key(e) not in alerted)
        break
    return Outcome(sent, undelivered, f"book day {day}: {sent} sent, {undelivered} undelivered")


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_release_alert``; takes no arguments."""
    from executor.notifier import send_trading_notice

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print(f"usage: v2_release_alert (no arguments); got {args!r}", file=sys.stderr)
        return 2
    moment = now or dt.datetime.now(dt.UTC)
    if not in_window(moment):
        logger.info(
            "v2 release alert: %s ET is not a trading-day window", f"{moment.astimezone(NEW_YORK):%Y-%m-%d %H:%M}"
        )
        return 0
    outcome = tick(open_store(read_env_file(TRADER_ENV)), now=moment, send=send_trading_notice)
    logger.info("v2 release alert: %s", outcome.reason)
    if outcome.undelivered:
        logger.error("v2 release alert: %d release alert(s) not delivered; retried next tick", outcome.undelivered)
    if outcome.failed:
        return 1
    return 0


__all__ = ["Outcome", "plain_trigger", "release_message", "tick"]


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-release-alert", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
