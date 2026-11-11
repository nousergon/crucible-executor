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

**What each one says: a proof-of-life checklist** (Brian 2026-11-10: "I'm going
to need confirmation that the system is actually doing what it should be during
market hours"). A headline (``Market OPEN — v2 trading day 2026-11-11 (Wed)``),
then one line per process, each ✅ done, ❌ failed or ⏳ still to come, with its
time and its key number, then the console's trading page. Every line is read
from evidence, never assumed: the session's own documents in the v2 store,
systemd, the executor checkout, and the read-only broker session.

* The open: the box up (boot time), the code pulled (boot-pull, the executor
  and trader shas), the day's predictions, the order book (when, what it
  changes, how many orders it plans), IB Gateway (paper or live, never an
  account number), the session (running since), its entry/exit watch (last
  write), and the orders so far: sent, filled, waiting on their triggers.
* The close: the session's run (from, to, stopped cleanly), the trades executed
  (filled, partly filled, not filled, the notional each way, which names fell
  short), the timed orders left unsent, what the broker still holds (only GTC
  stops are expected), and what still runs tonight (⏳).

An order counts as filled only when every share it asked for filled; fewer is
partly filled, never "filled". Any ❌ line adds the ``Impact:``, ``Auto-fix:``
and ``You:`` footer; an all-clear message carries none (Brian 2026-11-04).

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
    run_lists_output,
    shortfall_key,
    split_fills,
)

logger = logging.getLogger(__name__)

Kind = Literal["open", "close"]

#: The console page for a Crucible v2 trading day (nous-ergon-ops
#: nousergon-console/config.d/_pages_crucible_trading.yaml).
CONSOLE_URL = "https://console.nousergon.ai/page/crucible-trading"

SESSION_UNIT = "alpha-engine-trader-session.service"
GATEWAY_UNIT = "ibgateway.service"
BOOT_PULL_UNIT = "boot-pull.service"
#: The executor checkout this module runs from; boot-pull resets it at boot.
EXECUTOR_DIR = Path(__file__).resolve().parents[1]
#: The units still to run after the close, and when (their timers' OnCalendar).
AFTER_CLOSE = (
    ("alpha-engine-trader-pin-smoke.timer", "pin smoke 16:10"),
    ("alpha-engine-trader-reconcile.timer", "reconcile 16:45"),
    # executor.v2_day_end: stops the box once the units above are done.
    ("alpha-engine-v2-day-end.timer", "box stop 17:30"),
)

#: This alert's own IB API client id, set in its units. Never one the trader
#: uses (:func:`_refused_client_ids`) and never v1's 1-3.
CLIENT_ID_VAR = "CRUCIBLE_V2_ALERT_IB_CLIENT_ID"
_V1_CLIENT_IDS = frozenset({1, 2, 3})
_TRADER_CLIENT_ID_VARS = ("CRUCIBLE_TRADER_IB_CLIENT_ID", "CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID")
#: IB Gateway's PAPER API port; the trader refuses any other.
PAPER_PORT = 4002

ORDER_BOOK_PREFIX = "trader/order_books/"
ENTRY_WATCH_PREFIX = "trader/entry_watch/"
#: The champion's serving feed the session trades from (``crucible.keys.predictions_key``).
PREDICTIONS_PREFIX = "predictions/"
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

Mark = Literal["ok", "fail", "wait"]
#: How each checklist line starts: done, failed, still to come.
ICONS: Mapping[Mark, str] = {"ok": "✅", "fail": "❌", "wait": "⏳"}


@dataclass(frozen=True)
class Check:
    """One line of the checklist. A ``fail`` line says what it means and what to do."""

    mark: Mark
    text: str
    impact: str = ""
    action: str = ""

    def render(self) -> str:
        return f"{ICONS[self.mark]} {self.text}"


def _ok(text: str) -> Check:
    return Check("ok", text)


def _wait(text: str) -> Check:
    return Check("wait", text)


def _fail(text: str, impact: str, action: str) -> Check:
    return Check("fail", text, impact, action)


@dataclass(frozen=True)
class OpenOrder:
    """One order the broker reports for today: no account, no order id."""

    symbol: str
    action: str
    order_type: str
    tif: str
    quantity: float = 0.0
    filled: float = 0.0

    @property
    def is_stop(self) -> bool:
        return self.order_type.upper() in STOP_ORDER_TYPES and self.tif.upper() == "GTC"


@dataclass(frozen=True)
class Broker:
    """What a read-only API session saw. ``error`` set means it could not connect.

    ``open_orders`` are the orders still working; ``done`` the orders the API
    sent today that are finished (filled or cancelled). Both ``None`` when the
    orders were not asked for.
    """

    error: str | None = None
    account_kind: str = "unknown"
    open_orders: tuple[OpenOrder, ...] | None = None
    done: tuple[OpenOrder, ...] | None = None


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

    @property
    def running(self) -> bool:
        return self.active in ("active", "activating")

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


def booted_at(run: Runner = _run) -> dt.datetime | None:
    """When this box booted: systemd's own ``KernelTimestamp``."""
    proc = run(["systemctl", "show", "--timestamp=unix", "--property=KernelTimestamp"])
    for line in proc.stdout.splitlines():
        key, _, value = line.partition("=")
        if key == "KernelTimestamp":
            return Unit({key: value}).stamp(key)
    return None


def executor_sha(run: Runner = _run) -> str | None:
    """The short sha the executor checkout (this module's own) is at."""
    proc = run(["git", "-C", str(EXECUTOR_DIR), "rev-parse", "--short=7", "HEAD"])
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def _refused_client_ids(env: Mapping[str, str]) -> set[int]:
    refused = set(_V1_CLIENT_IDS)
    for var in _TRADER_CLIENT_ID_VARS:
        value = env.get(var, "").strip()
        if value.isdigit():
            refused.add(int(value))
    return refused


def _quantity(value: object) -> float:
    """A share count from the SDK; IB's "unset" (a float near max) and garbage read as 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if 0 <= number < 1e12 else 0.0


def _broker_order(trade: object, *, filled: float | None = None) -> OpenOrder:
    contract: object = getattr(trade, "contract", None)
    order: object = getattr(trade, "order", None)
    status: object = getattr(trade, "orderStatus", None)
    return OpenOrder(
        symbol=str(getattr(contract, "symbol", "?")),
        action=str(getattr(order, "action", "?")),
        order_type=str(getattr(order, "orderType", "?")),
        tif=str(getattr(order, "tif", "?")),
        quantity=_quantity(getattr(order, "totalQuantity", 0)),
        filled=_quantity(getattr(status, "filled", 0)) if filled is None else filled,
    )


def read_broker(env: Mapping[str, str], client_id: int, *, want_orders: bool) -> Broker:
    """A READ-ONLY IB API session on ``client_id``: the account type and, if asked, today's orders.

    Never raises: anything that stops the read is returned as ``error``, which
    the notice reports as a failed line.
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
    except Exception as exc:  # noqa: BLE001 -- reported as a failed line, never raised
        return Broker(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    try:
        accounts = [str(a) for a in cast(list[object], ib.managedAccounts() or [])]
        if not accounts:
            kind = "unknown"
        elif all(a.startswith("DU") for a in accounts):
            kind = "paper"
        else:
            kind = "live"
        if not want_orders:
            return Broker(account_kind=kind)
        ib.reqAllOpenOrders()
        working = tuple(_broker_order(t) for t in cast(list[object], ib.openTrades()))
        # Finished orders the API sent this gateway session (the gateway restarts
        # every trading day with the box). A completed order's fill is on the
        # order itself (``filledQuantity``), not on its status.
        done = tuple(
            _broker_order(t, filled=_quantity(getattr(getattr(t, "order", None), "filledQuantity", 0)))
            for t in cast(list[object], ib.reqCompletedOrders(True))
        )
        return Broker(account_kind=kind, open_orders=working, done=done)
    except Exception as exc:  # noqa: BLE001 -- reported as a failed line, never raised
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


def _hm(moment: dt.datetime | None) -> str:
    return "?" if moment is None else moment.astimezone(_NEW_YORK).strftime("%H:%M")


def _et(moment: dt.datetime | None) -> str:
    return f"{_hm(moment)} ET"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _shares(value: float) -> str:
    return f"{value:,.0f}"


def _plain(text: object) -> str:
    """Free text with no ``*``: an odd one would open a bold entity in the message.

    Only ``*``. ``krepis.telegram.send_message`` escapes ``_``, backticks and
    brackets itself, so an identifier (a key, a unit) arrives as it is.
    """
    return str(text).replace("*", "")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _read(store: Store, relative: str) -> tuple[Doc | None, str | None]:
    """``(document, None)``; ``(None, None)`` when absent; ``(None, reason)`` when unreadable.

    Without ListBucket S3 answers an absent key with AccessDenied, so that
    reads as absent here: every document this notice reads is one the session
    may legitimately not have written yet.
    """
    doc, reason, _ = _read_stamped(store, relative)
    return doc, reason


def _read_stamped(store: Store, relative: str) -> tuple[Doc | None, str | None, dt.datetime | None]:
    """:func:`_read`, plus when the document was written (S3 ``LastModified``)."""
    doc, reason, modified = store.read_stamped(relative)
    if reason in ("AccessDenied", "403"):
        return None, None, None
    return doc, reason, modified


def _held_reason(run: Mapping[str, object] | None) -> str | None:
    for metric in _docs(run, "metrics"):
        if metric.get("name") == "trader_session_served" and metric.get("value") == 0:
            return str(metric.get("status_reason") or "none recorded")
    return None


def _unreadable_checks(unreadable: Sequence[str]) -> list[Check]:
    return [
        _fail(f"Could not read {_plain(r)}", "this notice could not check it", "check the trader role's read grants.")
        for r in unreadable
    ]


def _headline(state: str, today: str) -> str:
    weekday = dt.date.fromisoformat(today).strftime("%a")
    return f"*Market {state} — v2 trading day {today} ({weekday})*"


# --- the open -----------------------------------------------------------------


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
    boot: dt.datetime | None = None
    boot_pull: Unit = field(default_factory=Unit)
    executor_sha: str | None = None
    predictions: Doc | None = None
    predictions_at: dt.datetime | None = None
    unreadable: tuple[str, ...] = ()


_TRADES_LOST = "orders the book needs may not go out today"
_READ_JOURNAL = "read the session's journal on the trading box."


def _box_check(f: OpenFacts) -> Check:
    return _ok(f"Box up — booted {_et(f.boot)}" if f.boot is not None else "Box up — boot time not read")


def _code_check(f: OpenFacts) -> Check:
    sha = None if f.watch is None else f.watch.get("code_sha")
    trader = str(sha)[:7] if isinstance(sha, str) and sha else "not recorded yet"
    shas = f"executor {f.executor_sha or 'unknown'}, trader {_plain(trader)}"
    pull = f.boot_pull
    if pull.active == "activating":
        return _wait(f"Code pull still running — {shas}")
    if pull.result != "success":
        return _fail(
            f"Code pull did not succeed (boot-pull {pull.active}, result {pull.result}) — {shas}",
            "today's session may be running stale code",
            "read /var/log/boot-pull.log on the trading box.",
        )
    return _ok(f"Code pulled {_et(pull.stamp('ExecMainExitTimestamp'))} — {shas}")


def _predictions_check(f: OpenFacts) -> Check:
    if f.predictions is None:
        return _fail(
            f"No predictions for {f.day}",
            "the session holds and places no order",
            "check that the predictor published the day's feed.",
        )
    alpha = _sub(f.predictions, "predicted_alpha")
    when = f" published {_et(f.predictions_at)}" if f.predictions_at is not None else ""
    return _ok(f"Predictions for {f.day} present{when} — {_plural(len(alpha), 'name')}")


def _book_check(f: OpenFacts) -> Check:
    if f.book is None:
        return _fail(f"No order book for {f.day}", _TRADES_LOST, _READ_JOURNAL)
    actions = _sub(_sub(f.book, "summary"), "actions")
    counts = [(a, int(_num(actions, a))) for a in _ACTIONS]
    changes = sum(n for _, n in counts)
    detail = ", ".join(f"{n} {a}" for a, n in counts if n)
    text = f"Order book built {_et(_when(f.book.get('written_at_utc')))} — {_plural(changes, 'target change')}"
    text += f" ({detail})" if detail else ""
    entries = _docs(f.watch, "entries")
    if entries:
        sells = sum(1 for e in entries if e.get("side") == "sell")
        text += f", {_plural(len(entries), 'order')} planned ({sells} sell, {len(entries) - sells} buy)"
    return _ok(text)


def _gateway_check(f: OpenFacts) -> Check:
    if f.gateway.active != "active":
        return _fail(
            f"IB Gateway service is {f.gateway.active}",
            "the session cannot quote or send orders",
            "check ibgateway.service on the trading box.",
        )
    if f.broker.error is not None:
        return _fail(
            f"IB Gateway did not answer an API session ({_plain(f.broker.error)})",
            _TRADES_LOST,
            "check the gateway login on the trading box.",
        )
    if f.broker.account_kind != "paper":
        return _fail(
            f"IB Gateway connected to a {f.broker.account_kind} account, not paper",
            "the trader refuses any account that is not paper, so it will place no order",
            "check which account the gateway logged in to.",
        )
    return _ok("IB Gateway connected — paper account")


def _session_check(f: OpenFacts) -> Check:
    if f.session.running:
        return _ok(f"Trading session running since {_et(f.session.stamp('ExecMainStartTimestamp'))}")
    held = _held_reason(f.run)
    if held is not None:
        return _fail(
            f"Trading session HELD and placed no order: {_plain(held)}",
            "no trades today",
            "once the missing input is published, rerun the session today.",
        )
    return _fail(
        f"Trading session not running ({f.session.active}, result {f.session.result}, "
        f"ended {_et(f.session.stamp('ExecMainExitTimestamp'))})",
        "orders still waiting on their triggers will not go out",
        "read the session's journal; its failure page carries the tail if it failed.",
    )


def _watch_check(f: OpenFacts) -> Check:
    if f.watch is None:
        return _fail("Entry/exit watch has written nothing yet", _TRADES_LOST, _READ_JOURNAL)
    updated = _when(f.watch.get("updated_at_utc"))
    state = f.watch.get("state")
    if state == "complete":
        return _ok(f"Entry/exit watch complete — last write {_et(updated)}")
    if state != "watching":
        return _fail(f"Entry/exit watch is {_plain(state)} — last write {_et(updated)}", _TRADES_LOST, _READ_JOURNAL)
    if updated is None or f.now - updated > WATCH_STALE_AFTER:
        return _fail(f"Entry/exit watch has not written since {_et(updated)}", _TRADES_LOST, _READ_JOURNAL)
    return _ok(f"Entry/exit watch live — last write {_et(updated)}")


def broker_fills(broker: Broker) -> tuple[int, int] | None:
    """``(filled, partly filled)`` among today's non-stop orders at the broker; ``None`` if not read.

    Filled means every share ordered; an order with some shares and not all is
    partly filled, whether it is still working or was cancelled with the rest.
    """
    if broker.error is not None or broker.done is None:
        return None
    done = [o for o in broker.done if not o.is_stop]
    working = [o for o in broker.open_orders or () if not o.is_stop]
    filled = sum(1 for o in done if 0 < o.quantity <= o.filled)
    partly = sum(1 for o in (*done, *working) if 0 < o.filled < o.quantity)
    return filled, partly


def _orders_check(f: OpenFacts) -> Check:
    entries = _docs(f.watch, "entries")
    if not entries:
        return _wait("Orders so far today: none recorded yet")
    sent = sum(1 for e in entries if e.get("status") in ("sent", "triggered"))
    waiting = sum(1 for e in entries if e.get("status") == "waiting")
    fills = broker_fills(f.broker)
    if fills is None:
        filled = "fills not read from the broker"
    else:
        filled = f"{fills[0]} filled" + (f", {fills[1]} partly filled" if fills[1] else "")
    return _ok(f"Orders so far today: {sent} sent, {filled}, {waiting} waiting on their triggers")


def open_checks(f: OpenFacts) -> list[Check]:
    """One line per process that should be up and done by the open."""
    return [
        _box_check(f),
        _code_check(f),
        _predictions_check(f),
        _book_check(f),
        _gateway_check(f),
        _session_check(f),
        _watch_check(f),
        _orders_check(f),
        *_unreadable_checks(f.unreadable),
    ]


def render_open(f: OpenFacts) -> tuple[str, str]:
    """``(text, auto_fix)`` of the market-open checklist."""
    return _finish(_headline("OPEN", f.today), open_checks(f))


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


def _close_session_check(f: CloseFacts) -> Check:
    if f.session.running:
        return _fail(
            f"Trading session still running after its 16:05 hard stop (since {_et(f.session.stamp('ExecMainStartTimestamp'))})",
            "it holds the client id the 16:10 pin smoke and the 16:45 reconcile need",
            _READ_JOURNAL,
        )
    ran = f"{_hm(f.session.stamp('ExecMainStartTimestamp'))}–{_et(f.session.stamp('ExecMainExitTimestamp'))}"
    if f.session.result != "success":
        hard_stop = " (it hit the 16:05 hard stop)" if f.session.result == "timeout" else ""
        return _fail(
            f"Trading session ran {ran} and ended {f.session.result}{hard_stop}",
            "orders it had not sent did not go out; those it sent are on the record",
            "its failure page carries the journal tail; fix the cause before the next session.",
        )
    if f.run is None:
        return _fail(
            f"Trading session ran {ran} but wrote no run record for {f.day}",
            "today's orders are not on the record here; check the broker",
            _READ_JOURNAL,
        )
    if f.run.get("status") != "ok":
        reason = _plain(str(f.run.get("reason") or "no reason recorded")[:160])
        return _fail(
            f"Trading session ran {ran}; its run ended {_plain(f.run.get('status'))}: {reason}",
            "the remaining orders were not sent",
            "fix the cause; a rerun adopts the orders already sent and never resends them.",
        )
    return _ok(f"Trading session ran {ran} and stopped cleanly")


def _filled_of(o: Mapping[str, object]) -> str:
    return f"{_plain(o.get('symbol'))} {_shares(_num(o, 'filled_quantity'))}/{_shares(_num(o, 'quantity'))}"


def _trades_check(f: CloseFacts) -> Check:
    held = _held_reason(f.run)
    if held is not None:
        return _fail(
            f"Trades executed today: none — the session held ({_plain(held)})",
            "no trades today",
            "find why the input was missing before the next session.",
        )
    if f.shortfall is None and f.run is not None and run_lists_output(f.run, shortfall_key(f.day)):
        return _fail(
            "Trades executed today: the session filed its fills but they could not be read",
            "today's fills are unconfirmed here; check the broker",
            "check the trader role's read grant on the execution shortfall file.",
        )
    orders = _docs(f.shortfall, "orders")
    if not orders:
        reason = None if f.shortfall is None else f.shortfall.get("reason")
        return _ok(f"Trades executed today: none{f' ({_plain(reason)})' if reason else ''}")
    full, partial, unfilled = split_fills(orders)
    sold = sum(_num(o, "notional_usd") for o in orders if o.get("side") == "sell")
    bought = sum(_num(o, "notional_usd") for o in orders if o.get("side") == "buy")
    text = (
        f"Trades executed today: {_plural(len(orders), 'order')} — {len(full)} filled, "
        f"{len(partial)} partly filled, {len(unfilled)} not filled; sold {_money(sold)}, bought {_money(bought)}"
    )
    if partial:
        text += f"; partly filled: {', '.join(_filled_of(o) for o in partial)}"
    if unfilled:
        text += f"; not filled: {', '.join(_filled_of(o) for o in unfilled)}"
    return _ok(text)


def _timed_check(f: CloseFacts) -> Check:
    entries = _docs(f.watch, "entries")
    waiting = [_plain(e.get("symbol")) for e in entries if e.get("status") == "waiting"]
    if waiting:
        return _fail(
            f"Timed orders still waiting at the stop: {', '.join(waiting)}",
            "the book did not reach its target today",
            "nothing tonight; tomorrow's session sizes against what is actually held.",
        )
    not_sent = [_plain(e.get("symbol")) for e in entries if e.get("status") == "not_sent"]
    if not not_sent:
        return _ok("Timed orders left unsent: none")
    return _ok(f"Timed orders left unsent and cancelled at the close: {len(not_sent)} ({', '.join(not_sent)})")


def _broker_check(f: CloseFacts) -> Check:
    if f.broker.error is not None:
        return _fail(
            f"Broker not read ({_plain(f.broker.error)})",
            "open orders at the broker are unchecked",
            "check the gateway on the trading box before 16:45.",
        )
    orders = f.broker.open_orders or ()
    stray = [o for o in orders if not o.is_stop]
    if stray:
        listed = ", ".join(f"{o.action} {o.symbol} {o.order_type} {o.tif}" for o in stray[:6])
        return _fail(
            f"Broker still has {_plural(len(stray), 'non-stop order')} open: {_plain(listed)}",
            "an order left working can fill after the session, outside the record",
            "cancel it in the paper account if it is not meant to be there.",
        )
    return _ok(f"Broker holds only GTC stops ({len(orders)})")


def _after_close_checks(f: CloseFacts) -> list[Check]:
    out: list[Check] = []
    for timer, label in AFTER_CLOSE:
        state = f.timers.get(timer, Unit()).active
        if state == "active":
            out.append(_wait(f"Still to run: {label} ET"))
        else:
            out.append(
                _fail(
                    f"Still to run: {label} ET — but its timer is {state}",
                    f"the {label.rsplit(' ', 1)[0]} will not run tonight",
                    f"check {timer} on the trading box.",
                )
            )
    return out


def close_checks(f: CloseFacts) -> list[Check]:
    """One line per process the trading day ran, then what still runs tonight."""
    return [
        _close_session_check(f),
        _trades_check(f),
        _timed_check(f),
        _broker_check(f),
        *_unreadable_checks(f.unreadable),
        *_after_close_checks(f),
    ]


def render_close(f: CloseFacts) -> tuple[str, str]:
    """``(text, auto_fix)`` of the market-close checklist."""
    return _finish(_headline("CLOSED", f.today), close_checks(f))


# --- shared ---------------------------------------------------------------------


def _finish(headline: str, checks: Sequence[Check]) -> tuple[str, str]:
    """The headline, one line per check, the footer (only when a check failed), the link."""
    body = "\n".join([headline, *(c.render() for c in checks)])
    failed = [c for c in checks if c.mark == "fail"]
    if not failed:
        return brief_alert(body, CONSOLE_URL, auto_fix=INFORMATIONAL), INFORMATIONAL
    impact = "; ".join(dict.fromkeys(c.impact for c in failed))
    action = " ".join(dict.fromkeys(c.action[0].upper() + c.action[1:] for c in failed))
    auto_fix = "none; nothing on the box retries or restarts the trader."
    return brief_alert(body, CONSOLE_URL, impact=impact, auto_fix=auto_fix, action=action), auto_fix


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
    stamps: dict[str, dt.datetime | None] = {}
    for name, key in (
        ("book", f"{ORDER_BOOK_PREFIX}{day}.json"),
        ("watch", f"{ENTRY_WATCH_PREFIX}{day}.json"),
        ("run", f"{RUNS_PREFIX}{day}/run.json"),
        ("predictions", f"{PREDICTIONS_PREFIX}{day}.json"),
    ):
        docs[name], reason, stamps[name] = _read_stamped(store, key)
        if reason is not None:
            unreadable.append(f"{key} ({reason[:80]})")
    return OpenFacts(
        today=today.isoformat(),
        day=day,
        session=unit_state(SESSION_UNIT, run),
        gateway=unit_state(GATEWAY_UNIT, run),
        broker=broker(env, client_id, want_orders=True),
        book=docs["book"],
        watch=docs["watch"],
        run=docs["run"],
        now=now,
        boot=booted_at(run),
        boot_pull=unit_state(BOOT_PULL_UNIT, run),
        executor_sha=executor_sha(run),
        predictions=docs["predictions"],
        predictions_at=stamps["predictions"],
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
            unreadable.append(f"{key} ({reason[:80]})")
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
    "Check",
    "CloseFacts",
    "OpenFacts",
    "Outcome",
    "close_checks",
    "open_checks",
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
