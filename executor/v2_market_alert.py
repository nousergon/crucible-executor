"""Telegram market-open and market-close notices for the v2 trading day (alpha-engine-config-I12024).

Run by two timers of their own, never by the session:

* ``alpha-engine-v2-market-open-alert.timer`` at 09:40 New York time, ten
  minutes after ``alpha-engine-trader-session`` starts at the 09:30 open;
* ``alpha-engine-v2-market-close-alert.timer`` at 16:07, after the session's
  16:05 hard stop and before the 16:10 pin smoke.

**Why this exists.** Brian 2026-11-10: "I want to receive Telegram alerts when
the market opens and when the market closes, along with the status of the order
book and all of the related infrastructure that goes online during the market
open and comes down at market close." Until this module the v2 day sent one
message, the session summary (:mod:`executor.v2_trade_alert`), and only when the
session unit ended, whenever that was. Nothing said the session had started,
and nothing ran at the close.

**What each one says.** One message each, a pointer and not a copy: two or
three sentences and the console's trading page. The open names the order book
(its day, when it was written, what it changes, how many orders are waiting on
their triggers) and what came up (the box, IB Gateway and the account type, the
session, its entry/exit watch, the trader release). The close names what the
orders did (filled, partly filled, not filled, and the notional each way),
whether the session stopped cleanly with no timed order left waiting, what the
broker still holds open (only GTC stops are expected), and what still runs
before the box stops. Anything that is not as expected becomes a problem line
with ``Impact:``, ``Auto-fix:`` and ``You:``; an all-clear message carries no
footer (Brian 2026-11-04).

**Read-only, and off the trading path.** It reads the session's own documents
in the v2 store (as the trader's role, through the trader's env file), asks
systemd for unit state, and opens a READ-ONLY IB API session on its OWN client
id (:data:`CLIENT_ID_VAR`, set in its units) to see the account type and the
open orders. It never places, modifies or cancels an order and never touches a
trader unit. It refuses to connect on any id the trader's env file uses.

**Delivery.** Straight to the ``#trades`` topic through
:func:`executor.notifier.send_trading_notice`, never through flow-doctor, so
its daily cap cannot hold it back. A marker per day and kind makes a second run
the same day send nothing. The process exits non-zero only when the message was
not delivered, which pages through ``alpha-engine-trader-failed@``.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

from krepis.telegram import INFORMATIONAL, brief_alert
from krepis.trading_calendar import is_trading_day, previous_trading_day

from executor.notifier import send_trading_notice
from executor.v2_trade_alert import (
    RUNS_PREFIX,
    TRADER_ENV,
    Doc,
    Store,
    open_store,
    read_env_file,
    shortfall_key,
)

logger = logging.getLogger(__name__)

Kind = Literal["open", "close"]

#: The console page for a Crucible v2 trading day (nous-ergon-ops
#: nousergon-console/config.d/_pages_crucible_trading.yaml).
CONSOLE_URL = "https://console.nousergon.ai/page/crucible-trading"

SESSION_UNIT = "alpha-engine-trader-session.service"
GATEWAY_UNIT = "ibgateway.service"
#: The units still to run after the close, and when (their timers' OnCalendar).
AFTER_CLOSE = (
    ("alpha-engine-trader-pin-smoke.timer", "pin smoke 16:10"),
    ("alpha-engine-trader-reconcile.timer", "broker reconcile 16:45"),
)
#: When the box stops: the 01:00 ET backstop stop (the session timer's own note).
BOX_STOP = "the box stops at the 01:00 ET backstop"

#: This alert's own IB API client id, set in its units. Never one the trader
#: uses (:func:`_refused_client_ids`) and never v1's 1-3.
CLIENT_ID_VAR = "CRUCIBLE_V2_ALERT_IB_CLIENT_ID"
_V1_CLIENT_IDS = frozenset({1, 2, 3})
_TRADER_CLIENT_ID_VARS = ("CRUCIBLE_TRADER_IB_CLIENT_ID", "CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID")
#: IB Gateway's PAPER API port; the trader refuses any other.
PAPER_PORT = 4002

ORDER_BOOK_PREFIX = "trader/order_books/"
ENTRY_WATCH_PREFIX = "trader/entry_watch/"
#: The watch rewrites its document at least every 600 s while it waits
#: (crucible-trader ``entry_timing.HEARTBEAT_S``); older than this, it has stopped.
WATCH_STALE_AFTER = dt.timedelta(minutes=15)
#: Order types the close expects to find open overnight: protective stops.
STOP_ORDER_TYPES = frozenset({"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"})

#: Where a delivered notice leaves its marker; the box's own disk, so it
#: survives a rerun the same day.
SENT_DIR = Path("/home/ec2-user/.local/state/crucible-v2/market-alert")

_NEW_YORK = ZoneInfo("America/New_York")
_ACTIONS = ("enter", "exit", "increase", "reduce")

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=60, check=False)


# --- what is observed ---------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One thing that is not as expected, and what it means."""

    line: str
    impact: str
    action: str


@dataclass(frozen=True)
class OpenOrder:
    """One order the broker still has working: no account, no order id."""

    symbol: str
    action: str
    order_type: str
    tif: str

    @property
    def is_stop(self) -> bool:
        return self.order_type.upper() in STOP_ORDER_TYPES and self.tif.upper() == "GTC"


@dataclass(frozen=True)
class Broker:
    """What a read-only API session saw. ``error`` set means it could not connect."""

    error: str | None = None
    account_kind: str = "unknown"
    open_orders: tuple[OpenOrder, ...] | None = None


@dataclass(frozen=True)
class Unit:
    """``systemctl show`` for one unit; empty when systemd could not be asked."""

    props: Mapping[str, str] = field(default_factory=lambda: cast(dict[str, str], {}))

    @property
    def active(self) -> str:
        return self.props.get("ActiveState", "unknown")

    @property
    def result(self) -> str:
        return self.props.get("Result", "unknown")

    def stamp(self, name: str) -> dt.datetime | None:
        """A timestamp property as ``systemctl show --timestamp=unix`` gives it (``@1762875001``)."""
        value = self.props.get(name, "")
        if not value.startswith("@") or not value[1:].isdigit():
            return None
        return dt.datetime.fromtimestamp(int(value[1:]), dt.UTC)


_UNIT_PROPERTIES = (
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ActiveEnterTimestamp",
)


def unit_state(unit: str, run: Runner = _run) -> Unit:
    """The unit's state as ``systemctl show`` reports it."""
    # Epoch timestamps: the box's own timezone never enters the reading.
    proc = run(["systemctl", "show", unit, "--timestamp=unix", f"--property={','.join(_UNIT_PROPERTIES)}"])
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in _UNIT_PROPERTIES:
            props[key] = value
    return Unit(props)


def _refused_client_ids(env: Mapping[str, str]) -> set[int]:
    refused = set(_V1_CLIENT_IDS)
    for var in _TRADER_CLIENT_ID_VARS:
        value = env.get(var, "").strip()
        if value.isdigit():
            refused.add(int(value))
    return refused


def read_broker(env: Mapping[str, str], client_id: int, *, want_orders: bool) -> Broker:
    """A READ-ONLY IB API session on ``client_id``: the account type and, if asked, the open orders.

    Never raises: anything that stops the read is returned as ``error``, which
    the notice reports as a problem line.
    """
    if client_id in _refused_client_ids(env):
        return Broker(error=f"client id {client_id} is one the trader or v1 uses; not connecting")
    host = env.get("CRUCIBLE_TRADER_IB_HOST", "127.0.0.1")
    try:
        port = int(env.get("CRUCIBLE_TRADER_IB_PORT", str(PAPER_PORT)))
    except ValueError:
        return Broker(error="the trader's gateway port is not a number")
    try:
        ib_insync = importlib.import_module("ib_insync")
        ib = ib_insync.IB()
        # readonly=True: the session can read but never place or cancel an order.
        ib.connect(host, port, clientId=client_id, timeout=15, readonly=True)
    except Exception as exc:  # noqa: BLE001 -- reported as a problem line, never raised
        return Broker(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    try:
        accounts = [str(a) for a in cast(list[object], ib.managedAccounts() or [])]
        if not accounts:
            kind = "unknown"
        elif all(a.startswith("DU") for a in accounts):
            kind = "paper"
        else:
            kind = "live"
        orders: tuple[OpenOrder, ...] | None = None
        if want_orders:
            ib.reqAllOpenOrders()
            found: list[OpenOrder] = []
            for trade in cast(list[object], ib.openTrades()):
                contract: object = getattr(trade, "contract", None)
                order: object = getattr(trade, "order", None)
                found.append(
                    OpenOrder(
                        symbol=str(getattr(contract, "symbol", "?")),
                        action=str(getattr(order, "action", "?")),
                        order_type=str(getattr(order, "orderType", "?")),
                        tif=str(getattr(order, "tif", "?")),
                    )
                )
            orders = tuple(found)
        return Broker(account_kind=kind, open_orders=orders)
    except Exception as exc:  # noqa: BLE001 -- reported as a problem line, never raised
        return Broker(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    finally:
        ib.disconnect()


# --- reading the session's documents -------------------------------------------


def _docs(doc: Mapping[str, object] | None, key: str) -> list[Doc]:
    value = None if doc is None else doc.get(key)
    if not isinstance(value, list):
        return []
    return [cast(Doc, item) for item in cast(list[object], value) if isinstance(item, dict)]


def _sub(doc: Mapping[str, object] | None, key: str) -> Doc:
    value = None if doc is None else doc.get(key)
    return cast(Doc, value) if isinstance(value, dict) else {}


def _num(doc: Mapping[str, object], key: str) -> float:
    value = doc.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _when(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _et(moment: dt.datetime | None) -> str:
    return "?" if moment is None else moment.astimezone(_NEW_YORK).strftime("%H:%M ET")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _read(store: Store, relative: str) -> tuple[Doc | None, str | None]:
    """``(document, None)``; ``(None, None)`` when absent; ``(None, reason)`` when unreadable.

    Without ListBucket S3 answers an absent key with AccessDenied, so that
    reads as absent here: every document this notice reads is one the session
    may legitimately not have written yet.
    """
    doc, reason = store.read(relative)
    if reason in ("AccessDenied", "403"):
        return None, None
    return doc, reason


def _held_reason(run: Mapping[str, object] | None) -> str | None:
    for metric in _docs(run, "metrics"):
        if metric.get("name") == "trader_session_served" and metric.get("value") == 0:
            return str(metric.get("status_reason") or "none recorded")
    return None


# --- the open -----------------------------------------------------------------


def book_sentence(day: str, book: Mapping[str, object] | None, watch: Mapping[str, object] | None) -> str:
    """What the order book is and what the session will do with it."""
    if book is None:
        return f"No order book for {day} yet."
    actions = _sub(_sub(book, "summary"), "actions")
    counts = [(a, int(_num(actions, a))) for a in _ACTIONS]
    changes = sum(n for _, n in counts)
    detail = ", ".join(f"{n} {a}" for a, n in counts if n)
    stage = " pre-open" if book.get("stage") == "pre_open" else ""
    text = f"The{stage} order book for {day} was written at {_et(_when(book.get('written_at_utc')))}: "
    text += f"{changes} target change{'s' if changes != 1 else ''}" + (f" ({detail})" if detail else "")
    entries = _docs(watch, "entries")
    if entries:
        sells = sum(1 for e in entries if e.get("side") == "sell")
        waiting = sum(1 for e in entries if e.get("status") == "waiting")
        text += (
            f", {len(entries)} order{'s' if len(entries) != 1 else ''} planned "
            f"({sells} sell, {len(entries) - sells} buy), {waiting} waiting on their triggers"
        )
    return text + "."


@dataclass(frozen=True)
class OpenFacts:
    today: str
    day: str
    session: Unit
    gateway: Unit
    broker: Broker
    book: Doc | None
    watch: Doc | None
    run: Doc | None
    now: dt.datetime
    unreadable: tuple[str, ...] = ()


def open_problems(f: OpenFacts) -> list[Problem]:
    """Everything that should be up at the open and is not."""
    out: list[Problem] = []
    trades_lost = "orders the book needs may not go out today"
    for reason in f.unreadable:
        out.append(Problem(f"{reason}.", "this notice could not check it", "check the trader role's read grants."))
    if f.gateway.active != "active":
        out.append(
            Problem(
                f"IB Gateway service is {f.gateway.active}.",
                "the session cannot quote or send orders",
                "check ibgateway.service on the trading box.",
            )
        )
    if f.broker.error is not None:
        out.append(
            Problem(
                f"IB Gateway did not answer an API session ({f.broker.error}).",
                trades_lost,
                "check the gateway login on the trading box.",
            )
        )
    elif f.broker.account_kind != "paper":
        out.append(
            Problem(
                f"the gateway's account type is {f.broker.account_kind}, not paper.",
                "the trader refuses any account that is not paper, so it will place no order",
                "check which account the gateway logged in to.",
            )
        )
    held = _held_reason(f.run)
    if f.session.active not in ("active", "activating"):
        if held is not None:
            out.append(
                Problem(
                    f"the session HELD and placed no order: {held}.",
                    "no trades today",
                    "once the missing input is published, rerun the session today.",
                )
            )
        else:
            ended = _et(f.session.stamp("ExecMainExitTimestamp"))
            out.append(
                Problem(
                    f"the session is not running ({f.session.active}, result {f.session.result}, ended {ended}).",
                    "orders still waiting on their triggers will not go out",
                    "read the session's journal; its failure page carries the tail if it failed.",
                )
            )
        return out
    if f.book is None:
        out.append(
            Problem(
                f"no order book for {f.day} yet.",
                trades_lost,
                "read the session's journal on the trading box.",
            )
        )
    if f.watch is None:
        out.append(
            Problem(
                "the entry/exit watch has written nothing yet.",
                trades_lost,
                "read the session's journal on the trading box.",
            )
        )
    else:
        updated = _when(f.watch.get("updated_at_utc"))
        state = f.watch.get("state")
        if state not in ("watching", "complete"):
            out.append(
                Problem(
                    f"the entry/exit watch is {state}.",
                    trades_lost,
                    "read the session's journal on the trading box.",
                )
            )
        elif state == "watching" and (updated is None or f.now - updated > WATCH_STALE_AFTER):
            out.append(
                Problem(
                    f"the entry/exit watch has not written since {_et(updated)}.",
                    trades_lost,
                    "read the session's journal on the trading box.",
                )
            )
    return out


def render_open(f: OpenFacts) -> tuple[str, str]:
    """``(text, auto_fix)`` of the market-open notice."""
    problems = open_problems(f)
    head = f"*Market open, v2 trading day {f.today}*"
    if problems:
        head = f"*Market open, v2 trading day {f.today}: {len(problems)} problem{'s' if len(problems) > 1 else ''}*"
    up: list[str] = ["the box"]
    if f.gateway.active == "active" and f.broker.error is None:
        up.append(f"IB Gateway ({f.broker.account_kind})")
    if f.session.active in ("active", "activating"):
        up.append(f"the session (since {_et(f.session.stamp('ExecMainStartTimestamp'))})")
        watch_state = None if f.watch is None else f.watch.get("state")
        if watch_state == "watching":
            up.append("its entry/exit watch")
    sha = None if f.watch is None else f.watch.get("code_sha")
    release = f"trader release {str(sha)[:7]}" if isinstance(sha, str) and sha else "trader release not recorded"
    body = [
        head,
        book_sentence(f.day, f.book, f.watch),
        f"Up: {_join(up)}; {release}.",
    ]
    return _finish(body, problems)


# --- the close ----------------------------------------------------------------


@dataclass(frozen=True)
class CloseFacts:
    today: str
    day: str
    session: Unit
    broker: Broker
    run: Doc | None
    shortfall: Doc | None
    watch: Doc | None
    timers: Mapping[str, Unit]
    unreadable: tuple[str, ...] = ()


def orders_sentence(shortfall: Mapping[str, object] | None, held: str | None) -> str:
    if held is not None:
        return f"The session held and placed no order ({held})."
    orders = _docs(shortfall, "orders")
    if not orders:
        reason = None if shortfall is None else shortfall.get("reason")
        return f"No orders today{f' ({reason})' if reason else ''}."
    full = [o for o in orders if 0 < _num(o, "quantity") <= _num(o, "filled_quantity")]
    partial = [o for o in orders if 0 < _num(o, "filled_quantity") < _num(o, "quantity")]
    unfilled = [o for o in orders if _num(o, "filled_quantity") <= 0]
    sold = sum(_num(o, "notional_usd") for o in orders if o.get("side") == "sell")
    bought = sum(_num(o, "notional_usd") for o in orders if o.get("side") == "buy")
    text = (
        f"{len(orders)} order{'s' if len(orders) != 1 else ''}: {len(full)} filled, "
        f"{len(partial)} partly filled, {len(unfilled)} not filled; sold {_money(sold)}, bought {_money(bought)}"
    )
    named = [str(o.get("symbol")) for o in (*partial, *unfilled)]
    return text + (f" (partial or unfilled: {', '.join(named)})." if named else ".")


def close_problems(f: CloseFacts) -> list[Problem]:
    out: list[Problem] = []
    for reason in f.unreadable:
        out.append(Problem(f"{reason}.", "this notice could not check it", "check the trader role's read grants."))
    if f.session.active in ("active", "activating"):
        out.append(
            Problem(
                "the session is still running after its 16:05 hard stop.",
                "it holds the client id the 16:10 pin smoke and the 16:45 reconcile need",
                "read the session's journal on the trading box.",
            )
        )
    elif f.session.result != "success":
        out.append(
            Problem(
                f"the session ended {f.session.result}"
                f"{' (it hit the 16:05 hard stop)' if f.session.result == 'timeout' else ''}.",
                "orders it had not sent did not go out; those it sent are on the record",
                "its failure page carries the journal tail; fix the cause before the next 09:30 session.",
            )
        )
    if f.run is None:
        out.append(
            Problem(
                f"the session wrote no run record for {f.day}.",
                "today's orders are not on the record here; check the broker",
                "read the session's journal on the trading box.",
            )
        )
    elif f.run.get("status") != "ok":
        out.append(
            Problem(
                f"the session's run ended {f.run.get('status')}: {str(f.run.get('reason') or 'no reason recorded')[:160]}.",
                "the remaining orders were not sent",
                "fix the cause; a rerun adopts the orders already sent and never resends them.",
            )
        )
    waiting = [str(e.get("symbol")) for e in _docs(f.watch, "entries") if e.get("status") == "waiting"]
    if waiting:
        out.append(
            Problem(
                f"timed orders still waiting at the stop: {', '.join(waiting)}.",
                "the book did not reach its target today",
                "nothing tonight; tomorrow's session sizes against what is actually held.",
            )
        )
    if f.broker.error is not None:
        out.append(
            Problem(
                f"the broker could not be read ({f.broker.error}).",
                "open orders at the broker are unchecked",
                "check the gateway on the trading box before 16:45.",
            )
        )
    else:
        stray = [o for o in f.broker.open_orders or () if not o.is_stop]
        if stray:
            listed = ", ".join(f"{o.action} {o.symbol} {o.order_type} {o.tif}" for o in stray[:6])
            out.append(
                Problem(
                    f"{len(stray)} non-stop order{'s' if len(stray) != 1 else ''} still open at the broker: {listed}.",
                    "an order left working can fill after the session, outside the record",
                    "cancel it in the paper account if it is not meant to be there.",
                )
            )
    for timer, label in AFTER_CLOSE:
        state = f.timers.get(timer, Unit()).active
        if state != "active":
            out.append(
                Problem(
                    f"the {label} timer is {state}.",
                    f"the {label} will not run tonight",
                    f"check {timer} on the trading box.",
                )
            )
    return out


def render_close(f: CloseFacts) -> tuple[str, str]:
    """``(text, auto_fix)`` of the market-close notice."""
    problems = close_problems(f)
    head = f"*Market close, v2 trading day {f.today}*"
    if problems:
        head = f"*Market close, v2 trading day {f.today}: {len(problems)} problem{'s' if len(problems) > 1 else ''}*"
    held = _held_reason(f.run)
    stopped = ""
    if f.session.active not in ("active", "activating") and f.session.result == "success":
        not_sent = sum(1 for e in _docs(f.watch, "entries") if e.get("status") == "not_sent")
        stopped = f"The session stopped cleanly at {_et(f.session.stamp('ExecMainExitTimestamp'))}"
        if not_sent:
            stopped += f" ({not_sent} timed order{'s' if not_sent != 1 else ''} not sent)"
        orders = f.broker.open_orders
        if f.broker.error is None and orders is not None and all(o.is_stop for o in orders):
            stopped += f"; the broker holds only GTC stops ({len(orders)})"
        stopped += ". "
    body = [
        head,
        orders_sentence(f.shortfall, held),
        f"{stopped}Still to run: {_join([label for _, label in AFTER_CLOSE])} ET; {BOX_STOP}.",
    ]
    return _finish(body, problems)


# --- shared ---------------------------------------------------------------------


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _finish(body: list[str], problems: Sequence[Problem]) -> tuple[str, str]:
    """The notice: body, any problem lines, then (only with problems) the footer, then the link."""
    if not problems:
        return brief_alert("\n".join(body), CONSOLE_URL, auto_fix=INFORMATIONAL), INFORMATIONAL
    lines = [*body, *(f"Problem: {p.line}" for p in problems)]
    impact = "; ".join(dict.fromkeys(p.impact for p in problems))
    action = " ".join(dict.fromkeys(p.action[0].upper() + p.action[1:] for p in problems))
    auto_fix = "none; nothing on the box retries or restarts the trader."
    text = brief_alert("\n".join(lines), CONSOLE_URL, impact=impact, auto_fix=auto_fix, action=action)
    return text, auto_fix


@dataclass(frozen=True)
class Outcome:
    """``sent``, ``already_sent``, ``not_a_trading_day`` or ``failed``."""

    outcome: str
    reason: str

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"


def gather_open(
    store: Store,
    env: Mapping[str, str],
    *,
    now: dt.datetime,
    client_id: int,
    run: Runner = _run,
    broker: Callable[..., Broker] = read_broker,
) -> OpenFacts:
    today = now.astimezone(_NEW_YORK).date()
    day = previous_trading_day(today).isoformat()
    unreadable: list[str] = []
    docs: dict[str, Doc | None] = {}
    for name, key in (
        ("book", f"{ORDER_BOOK_PREFIX}{day}.json"),
        ("watch", f"{ENTRY_WATCH_PREFIX}{day}.json"),
        ("run", f"{RUNS_PREFIX}{day}/run.json"),
    ):
        docs[name], reason = _read(store, key)
        if reason is not None:
            unreadable.append(f"{key} is unreadable ({reason[:80]})")
    return OpenFacts(
        today=today.isoformat(),
        day=day,
        session=unit_state(SESSION_UNIT, run),
        gateway=unit_state(GATEWAY_UNIT, run),
        broker=broker(env, client_id, want_orders=False),
        book=docs["book"],
        watch=docs["watch"],
        run=docs["run"],
        now=now,
        unreadable=tuple(unreadable),
    )


def gather_close(
    store: Store,
    env: Mapping[str, str],
    *,
    now: dt.datetime,
    client_id: int,
    run: Runner = _run,
    broker: Callable[..., Broker] = read_broker,
) -> CloseFacts:
    today = now.astimezone(_NEW_YORK).date()
    day = previous_trading_day(today).isoformat()
    unreadable: list[str] = []
    docs: dict[str, Doc | None] = {}
    for name, key in (
        ("run", f"{RUNS_PREFIX}{day}/run.json"),
        ("shortfall", shortfall_key(day)),
        ("watch", f"{ENTRY_WATCH_PREFIX}{day}.json"),
    ):
        docs[name], reason = _read(store, key)
        if reason is not None:
            unreadable.append(f"{key} is unreadable ({reason[:80]})")
    return CloseFacts(
        today=today.isoformat(),
        day=day,
        session=unit_state(SESSION_UNIT, run),
        broker=broker(env, client_id, want_orders=True),
        run=docs["run"],
        shortfall=docs["shortfall"],
        watch=docs["watch"],
        timers={timer: unit_state(timer, run) for timer, _ in AFTER_CLOSE},
        unreadable=tuple(unreadable),
    )


def notify(
    kind: Kind,
    store: Store,
    env: Mapping[str, str],
    *,
    now: dt.datetime,
    client_id: int,
    run: Runner = _run,
    broker: Callable[..., Broker] = read_broker,
    send: Callable[..., bool] = send_trading_notice,
    sent_dir: Path = SENT_DIR,
) -> Outcome:
    """Send today's ``kind`` notice once. Sends at most one message."""
    today = now.astimezone(_NEW_YORK).date()
    if not is_trading_day(today):
        return Outcome("not_a_trading_day", f"{today} is not an NYSE session")
    marker = sent_dir / f"{kind}-{today.isoformat()}"
    if marker.exists():
        return Outcome("already_sent", f"the {kind} notice for {today} was sent already")
    if kind == "open":
        text, auto_fix = render_open(gather_open(store, env, now=now, client_id=client_id, run=run, broker=broker))
    else:
        text, auto_fix = render_close(gather_close(store, env, now=now, client_id=client_id, run=run, broker=broker))
    if not send(text, auto_fix=auto_fix):
        return Outcome("failed", f"the {kind} notice for {today} was not delivered")
    sent_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{now.astimezone(dt.UTC).isoformat()}\n")
    return Outcome("sent", f"the {kind} notice for {today} was sent")


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_market_alert open|close``."""
    import os

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in ("open", "close"):
        print(f"usage: v2_market_alert open|close; got {args!r}", file=sys.stderr)
        return 2
    raw = os.environ.get(CLIENT_ID_VAR, "")
    if not raw.isdigit():
        print(f"{CLIENT_ID_VAR} is unset or not a number; its unit sets it", file=sys.stderr)
        return 2
    env = read_env_file(TRADER_ENV)
    kind: Kind = "open" if args[0] == "open" else "close"
    outcome = notify(kind, open_store(env), env, now=now or dt.datetime.now(dt.UTC), client_id=int(raw))
    (logger.error if outcome.failed else logger.info)("v2 market alert: %s: %s", outcome.outcome, outcome.reason)
    return 1 if outcome.failed else 0


__all__ = [
    "CONSOLE_URL",
    "CloseFacts",
    "OpenFacts",
    "Outcome",
    "Problem",
    "main",
    "notify",
    "render_close",
    "render_open",
]


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-market-alert", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
