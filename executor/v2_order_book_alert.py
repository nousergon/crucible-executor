"""Telegram announcement of the v2 trader's pre-open order book (alpha-engine-config-I12024).

Run by ``alpha-engine-v2-order-book-alert.service``, which
``alpha-engine-trader-order-book-plan.service`` starts when it succeeds
(``OnSuccess=``). Brian 2026-11-09/10: the daily order book is built and saved
before the US market opens, and it is the book announced, once a day.

**What it does.** Reads the day's book the plan wrote,
``trader/order_books/{day}.json`` (crucible-trader
``crucible_trader.order_book_plan``), and sends one essential message through
:func:`executor.notifier.send_daemon_status`: the champions, the positions, cash
and turnover, every name the book enters, exits, increases or reduces, and the
largest targets. ``{day}`` is the last closed session, the day the plan and the
session that trades the book both bind to.

**Once a day, never two.** It announces only a book whose ``stage`` is
``pre_open`` and that was written today (New York date), and only once: a
delivered announcement leaves a marker for its day, and a second start (a
rerun of the plan, or of this unit) sends nothing. The session's own book is
never announced: when it matches the pre-open book the session reuses it, and
when it differs it is kept under ``trader/order_books/as_traded/``, which the
session's summary (:mod:`executor.v2_trade_alert`) reports in its one
``Order book:`` line.

**No book is not a failure here.** The plan exits 0 without a book on a day it
holds (no feed yet, a degenerate feed, the kill switch) and pages on its own
failures. This unit says nothing then. It exits non-zero only when the book
could not be read for a reason other than absence, or the announcement was not
delivered, which pages through ``alpha-engine-trader-failed@``.

**Credentials.** As :mod:`executor.v2_trade_alert`: the store is read as the
trader's role through the trader's env file, and the Telegram secrets under
this box's instance role.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from krepis.trading_calendar import is_trading_day, last_closed_trading_day

from executor.notifier import send_daemon_status
from executor.v2_trade_alert import TRADER_ENV, Store, open_store, read_env_file

logger = logging.getLogger(__name__)

_NEW_YORK = ZoneInfo("America/New_York")

#: Where a delivered announcement leaves its day's marker. On the box's own
#: disk, so it survives the box's nightly stop and a rerun the same day.
ANNOUNCED_DIR = Path("/home/ec2-user/.local/state/crucible-v2/order-book-announced")

PRE_OPEN = "pre_open"

#: How many of the largest targets the message lists.
TOP_TARGETS = 10

_ACTIONS = ("enter", "exit", "increase", "reduce")


Row = Mapping[str, object]


def _rows(book: Mapping[str, object]) -> list[Row]:
    value = book.get("rows")
    items = cast(list[object], value) if isinstance(value, list) else []
    return [cast(Row, item) for item in items if isinstance(item, dict)]


def _num(doc: Mapping[str, object], key: str) -> float:
    value = doc.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def order_book_key(day: str) -> str:
    return f"trader/order_books/{day}.json"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(day: str, book: Mapping[str, object]) -> str:
    """The one message: what the book holds and what it changes."""
    rows = _rows(book)
    raw = book.get("summary")
    summary: Row = cast(Row, raw) if isinstance(raw, dict) else {}
    lines = [
        f"*v2 order book {day} (pre-open)*",
        f"M {book.get('m_champion') or 'none'} · S {book.get('s_champion') or 'none'}",
        f"{int(_num(summary, 'positions'))} positions · cash {_pct(_num(summary, 'cash_weight'))}"
        f" · turnover {_pct(_num(book, 'turnover_one_way_ratio'))} one way",
    ]
    for action in _ACTIONS:
        names = sorted(str(r.get("ticker")) for r in rows if r.get("action") == action)
        if names:
            lines.append(f"{action.capitalize()}: {', '.join(names)}")
    held = [r for r in rows if r.get("action") not in ("cash", "benchmark") and _num(r, "target_weight") > 0]
    held.sort(key=lambda r: -_num(r, "target_weight"))
    if held:
        top = ", ".join(f"{r.get('ticker')} {_pct(_num(r, 'target_weight'))}" for r in held[:TOP_TARGETS])
        lines.append(f"Top: {top}")
    lines.append(f"Written {book.get('written_at_utc')}; the session trades it after the open.")
    return "\n".join(lines)


def _written_today(book: Mapping[str, object], today: dt.date) -> bool:
    stamp = book.get("written_at_utc")
    if not isinstance(stamp, str):
        return False
    try:
        written = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    return written.astimezone(_NEW_YORK).date() == today


@dataclass(frozen=True)
class Outcome:
    """What one run did: ``announced``, ``already_announced``, ``no_book`` or ``failed``."""

    outcome: str
    reason: str

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"


def announce(
    store: Store,
    *,
    now: dt.datetime,
    status: Callable[..., bool] = send_daemon_status,
    announced_dir: Path = ANNOUNCED_DIR,
) -> Outcome:
    """Announce today's pre-open book once. Sends at most one message."""
    today = now.astimezone(_NEW_YORK).date()
    if not is_trading_day(today):
        return Outcome("no_book", f"{today} is not an NYSE session")
    day = last_closed_trading_day(now).isoformat()
    marker = announced_dir / day
    if marker.exists():
        return Outcome("already_announced", f"the {day} book was announced already")
    book, reason = store.read(order_book_key(day))
    if reason is not None and reason not in ("AccessDenied", "403"):
        return Outcome("failed", f"{order_book_key(day)} unreadable: {reason[:160]}")
    if book is None:
        # Without ListBucket an absent key reads as AccessDenied: the plan held.
        return Outcome("no_book", f"no book at {order_book_key(day)}")
    if book.get("stage") != PRE_OPEN or not _written_today(book, today):
        return Outcome("no_book", f"{order_book_key(day)} is not today's pre-open book")
    if not status(render(day, book), essential=True):
        return Outcome("failed", f"the {day} order book announcement was not delivered")
    announced_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{now.astimezone(dt.UTC).isoformat()}\n")
    return Outcome("announced", f"the {day} pre-open book was announced")


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_order_book_alert``; takes no arguments."""
    if argv:
        print(f"usage: v2_order_book_alert (no arguments); got {argv!r}", file=sys.stderr)
        return 2
    store = open_store(read_env_file(TRADER_ENV))
    outcome = announce(store, now=now or dt.datetime.now(dt.UTC))
    (logger.error if outcome.failed else logger.info)("v2 order book alert: %s: %s", outcome.outcome, outcome.reason)
    return 1 if outcome.failed else 0


__all__ = ["ANNOUNCED_DIR", "Outcome", "announce", "main", "order_book_key", "render"]


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-order-book-alert", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
