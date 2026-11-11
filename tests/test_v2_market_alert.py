"""`executor.v2_market_alert`: the v2 market-open and market-close notices (alpha-engine-config-I12024)."""

from __future__ import annotations

import datetime as dt
import io
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from krepis.telegram import INFORMATIONAL

from executor import v2_market_alert as m
from executor.v2_trade_alert import Store

# Wednesday 2026-11-11: the session trades the book of Tuesday 2026-11-10.
OPEN_AT = dt.datetime(2026, 11, 11, 14, 40, tzinfo=dt.UTC)  # 09:40 ET
CLOSE_AT = dt.datetime(2026, 11, 11, 21, 7, tzinfo=dt.UTC)  # 16:07 ET
DAY = "2026-11-10"
ENV = {"CRUCIBLE_TRADER_IB_CLIENT_ID": "41", "CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID": "42"}


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(
        self, objects: dict[str, Any], *, denied: Sequence[str] = (), modified: dict[str, dt.datetime] | None = None
    ) -> None:
        self.objects = objects
        self.denied = set(denied)
        self.modified = modified or {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key in self.denied:
            raise _ClientError("InternalError")
        if Key not in self.objects:
            raise _ClientError("AccessDenied")  # the trader role has no ListBucket
        response: dict[str, Any] = {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}
        if Key in self.modified:
            response["LastModified"] = self.modified[Key]
        return response

    def get_paginator(self, _name: str) -> Any:  # pragma: no cover -- not used here
        raise AssertionError


BOOK = {
    "written_at_utc": "2026-11-11T13:41:00Z",
    "stage": "pre_open",
    "summary": {"actions": {"enter": 2, "exit": 1, "increase": 1, "hold": 3, "ineligible": 800}},
}


def _entry(symbol: str, side: str, status: str = "waiting") -> dict[str, Any]:
    return {"symbol": symbol, "side": side, "status": status}


WATCH = {
    "state": "watching",
    "updated_at_utc": "2026-11-11T14:35:00Z",
    "code_sha": "811eb9b0123456789",
    "entries": [_entry("ANF", "sell"), _entry("CVLT", "sell", "sent"), _entry("FSLR", "buy"), _entry("MU", "buy")],
}


def _order(symbol: str, side: str, qty: float, filled: float, notional: float) -> dict[str, Any]:
    return {"symbol": symbol, "side": side, "quantity": qty, "filled_quantity": filled, "notional_usd": notional}


SHORTFALL = {
    "orders": [
        _order("ANF", "sell", 6, 6, 922.56),
        _order("CVLT", "sell", 657, 220, 36775.2),
        _order("FSLR", "buy", 178, 178, 31974.14),
        _order("NTNX", "buy", 1334, 817, 48337.62),
        _order("MU", "buy", 13, 0, 0.0),
    ]
}
RUN_OK = {"status": "ok", "metrics": [{"name": "trader_session_served", "value": 1.0}]}
PUBLISHED = dt.datetime(2026, 11, 11, 11, 10, tzinfo=dt.UTC)  # 06:10 ET


def _store(objects: dict[str, Any], **kw: Any) -> Store:
    return Store(_S3({f"crucible/{k}": v for k, v in objects.items()}, **kw), "bucket", "crucible")


class _Systemd:
    """``systemctl show`` answers per unit."""

    def __init__(self, units: dict[str, dict[str, str]]) -> None:
        self.units = units

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "git":
            assert list(argv[-3:]) == ["rev-parse", "--short=7", "HEAD"]
            return subprocess.CompletedProcess(list(argv), 0, stdout="93564d6\n", stderr="")
        assert "--timestamp=unix" in argv
        if argv[2] == "--timestamp=unix":  # the manager itself: when the box booted
            return subprocess.CompletedProcess(list(argv), 0, stdout=f"KernelTimestamp={BOOTED}", stderr="")
        props = self.units.get(argv[2], {})
        out = "\n".join(f"{k}={v}" for k, v in props.items())
        return subprocess.CompletedProcess(list(argv), 0, stdout=out, stderr="")


BOOTED = "@1794402000"  # 08:00 ET
PULLED = {"ActiveState": "inactive", "Result": "success", "ExecMainExitTimestamp": "@1794402960"}  # 08:16 ET
RUNNING = {"ActiveState": "activating", "Result": "success", "ExecMainStartTimestamp": "@1794407401"}
STOPPED = {
    "ActiveState": "inactive",
    "Result": "success",
    "ExecMainStartTimestamp": "@1794407701",  # 09:35 ET
    "ExecMainExitTimestamp": "@1794430600",  # 15:56 ET
}
ACTIVE = {"ActiveState": "active"}


def _systemd(session: dict[str, str], pull: dict[str, str] = PULLED) -> _Systemd:
    return _Systemd(
        {
            m.SESSION_UNIT: session,
            m.GATEWAY_UNIT: ACTIVE,
            m.BOOT_PULL_UNIT: pull,
            "alpha-engine-trader-pin-smoke.timer": ACTIVE,
            "alpha-engine-trader-reconcile.timer": ACTIVE,
            "alpha-engine-v2-day-end.timer": ACTIVE,
        }
    )


STOP = m.OpenOrder("SPY", "SELL", "TRAIL", "GTC", 10, 0)


def _broker(**kw: Any) -> Any:
    def read(_env: Any, _client_id: int, *, want_orders: bool) -> m.Broker:
        orders = kw.get("orders", (STOP,)) if want_orders else None
        done = kw.get("done", ()) if want_orders else None
        return m.Broker(error=kw.get("error"), account_kind=kw.get("account", "paper"), open_orders=orders, done=done)

    return read


class _Sent:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, *, auto_fix: str) -> bool:
        self.calls.append((text, auto_fix))
        return self.ok


def _notify(kind: m.Kind, store: Store, tmp_path: Path, *, now: dt.datetime, session: dict[str, str], **kw: Any):
    sent = _Sent(kw.pop("ok", True))
    pull = kw.pop("pull", PULLED)
    outcome = m.notify(
        kind,
        store,
        ENV,
        now=now,
        client_id=44,
        run=_systemd(session, pull),
        broker=_broker(**kw),
        send=sent,
        sent_dir=tmp_path,
    )
    return outcome, sent


# --- the open -----------------------------------------------------------------

PREDICTIONS = {"predicted_alpha": {f"S{i}": 0.01 for i in range(812)}}
OPEN_DOCS: dict[str, Any] = {
    f"trader/order_books/{DAY}.json": BOOK,
    f"trader/entry_watch/{DAY}.json": WATCH,
    f"predictions/{DAY}.json": PREDICTIONS,
}
#: CVLT's sell went out at the start and filled in full.
DONE = (m.OpenOrder("CVLT", "SELL", "MKT", "DAY", 657, 657),)

OPEN_ALL_OK = """\
*Market OPEN — v2 trading day 2026-11-11 (Wed)*
✅ Box up — booted 08:00 ET
✅ Code pulled 08:16 ET — executor 93564d6, trader 811eb9b
✅ Predictions for 2026-11-10 present published 06:10 ET — 812 names
✅ Order book built 08:41 ET — 4 target changes (2 enter, 1 exit, 1 increase), 4 orders planned (2 sell, 2 buy)
✅ IB Gateway connected — paper account
✅ Trading session running since 09:30 ET
✅ Entry/exit watch live — last write 09:35 ET
✅ Orders so far today: 1 sent, 1 filled, 3 waiting on their triggers
Details on console: https://console.nousergon.ai/page/crucible-trading"""


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_an_all_clear_open_is_a_checklist_with_no_footer(tmp_path: Path) -> None:
    store = _store(OPEN_DOCS, modified={f"crucible/predictions/{DAY}.json": PUBLISHED})
    outcome, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING, done=DONE)
    assert outcome.outcome == "sent"
    ((text, auto_fix),) = sent.calls
    assert auto_fix == INFORMATIONAL
    assert text == OPEN_ALL_OK
    for footer in ("Impact:", "Auto-fix:", "You:", "❌"):
        assert footer not in text


def test_a_failure_at_the_open_is_a_cross_with_the_footer(tmp_path: Path) -> None:
    stale = {**WATCH, "updated_at_utc": "2026-11-11T14:00:00Z"}
    store = _store({**OPEN_DOCS, f"trader/entry_watch/{DAY}.json": stale})
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING, error="ConnectionRefusedError: x")
    ((text, auto_fix),) = sent.calls
    assert auto_fix != INFORMATIONAL
    lines = _lines(text)
    assert lines[0] == "*Market OPEN — v2 trading day 2026-11-11 (Wed)*"
    assert "❌ IB Gateway did not answer an API session (ConnectionRefusedError: x)" in lines
    assert "❌ Entry/exit watch has not written since 09:00 ET" in lines
    assert "✅ Orders so far today: 1 sent, fills not read from the broker, 3 waiting on their triggers" in lines
    assert "✅ Trading session running since 09:30 ET" in lines  # what is fine still says so
    for footer in ("*Impact:*", "*Auto-fix:*", "*You:*"):
        assert footer in text
    assert text.rstrip().endswith(m.CONSOLE_URL)


def test_a_failed_code_pull_is_a_cross(tmp_path: Path) -> None:
    failed = {"ActiveState": "failed", "Result": "exit-code", "ExecMainExitTimestamp": "@1794402960"}
    _, sent = _notify("open", _store(OPEN_DOCS), tmp_path, now=OPEN_AT, session=RUNNING, pull=failed)
    ((text, _),) = sent.calls
    assert (
        "❌ Code pull did not succeed (boot-pull failed, result exit-code) — executor 93564d6, trader 811eb9b"
        in _lines(text)
    )
    assert "*You:* Read /var/log/boot-pull.log on the trading box." in text


def test_partial_fills_at_the_open_are_never_counted_as_filled(tmp_path: Path) -> None:
    done = (m.OpenOrder("CVLT", "SELL", "MKT", "DAY", 657, 220), m.OpenOrder("ANF", "SELL", "MKT", "DAY", 6, 6))
    working = (STOP, m.OpenOrder("NTNX", "BUY", "LMT", "DAY", 1334, 817))
    _, sent = _notify("open", _store(OPEN_DOCS), tmp_path, now=OPEN_AT, session=RUNNING, done=done, orders=working)
    ((text, _),) = sent.calls
    assert "✅ Orders so far today: 1 sent, 1 filled, 2 partly filled, 3 waiting on their triggers" in _lines(text)


def test_a_session_that_already_ended_at_the_open_is_a_cross(tmp_path: Path) -> None:
    """A release without the entry timing sends everything at the start and
    ends in minutes: that is exactly what the open must flag."""
    store = _store({f"trader/order_books/{DAY}.json": BOOK, f"runs/trader.session/{DAY}/run.json": RUN_OK})
    ended = {"ActiveState": "inactive", "Result": "success", "ExecMainExitTimestamp": "@1794407770"}
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=ended)
    ((text, _),) = sent.calls
    lines = _lines(text)
    assert "❌ Trading session not running (inactive, result success, ended 09:36 ET)" in lines
    assert "✅ Code pulled 08:16 ET — executor 93564d6, trader not recorded yet" in lines
    assert "❌ Entry/exit watch has written nothing yet" in lines
    assert "⏳ Orders so far today: none recorded yet" in lines


def test_a_held_session_is_named_at_the_open(tmp_path: Path) -> None:
    held = {"status": "ok", "metrics": [{"name": "trader_session_served", "value": 0, "status_reason": "no feed"}]}
    store = _store({f"runs/trader.session/{DAY}/run.json": held})
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session={"ActiveState": "inactive"})
    ((text, _),) = sent.calls
    lines = _lines(text)
    assert "❌ No predictions for 2026-11-10" in lines
    assert "❌ No order book for 2026-11-10" in lines
    assert "❌ Trading session HELD and placed no order: no feed" in lines


def test_a_live_account_on_the_gateway_is_a_cross_and_no_account_id_is_printed(tmp_path: Path) -> None:
    _, sent = _notify("open", _store(OPEN_DOCS), tmp_path, now=OPEN_AT, session=RUNNING, account="live")
    ((text, _),) = sent.calls
    assert "❌ IB Gateway connected to a live account, not paper" in _lines(text)
    assert "DU" not in text


def test_an_unreadable_document_is_a_cross_not_an_absence(tmp_path: Path) -> None:
    store = _store({}, denied=[f"crucible/trader/order_books/{DAY}.json"])
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING)
    ((text, _),) = sent.calls
    assert f"❌ Could not read trader/order_books/{DAY}.json (InternalError)" in _lines(text)


# --- the close ----------------------------------------------------------------


def _close_store(watch: dict[str, Any] | None = None, *, shortfall: dict[str, Any] = SHORTFALL) -> Store:
    objects: dict[str, Any] = {
        f"runs/trader.session/{DAY}/run.json": RUN_OK,
        f"trader/execution_shortfall/{DAY}.json": shortfall,
    }
    if watch is not None:
        objects[f"trader/entry_watch/{DAY}.json"] = watch
    return _store(objects)


DONE_WATCH = {
    **WATCH,
    "state": "complete",
    "entries": [_entry("ANF", "sell", "sent"), _entry("PLTR", "buy", "not_sent")],
}
ALL_FILLED = {
    "orders": [
        _order("ANF", "sell", 6, 6, 922.56),
        _order("CVLT", "sell", 657, 657, 109823.55),
        _order("FSLR", "buy", 178, 178, 31974.14),
    ]
}

CLOSE_ALL_OK = """\
*Market CLOSED — v2 trading day 2026-11-11 (Wed)*
✅ Trading session ran 09:35–15:56 ET and stopped cleanly
✅ Trades executed today: 3 orders — 3 filled, 0 partly filled, 0 not filled; sold $110,746, bought $31,974
✅ Timed orders left unsent and cancelled at the close: 1 (PLTR)
✅ Broker holds only GTC stops (1)
⏳ Still to run: pin smoke 16:10 ET
⏳ Still to run: reconcile 16:45 ET
⏳ Still to run: box stop 17:30 ET
Details on console: https://console.nousergon.ai/page/crucible-trading"""


def test_an_all_clear_close_is_a_checklist_with_no_footer(tmp_path: Path) -> None:
    store = _close_store(DONE_WATCH, shortfall=ALL_FILLED)
    outcome, sent = _notify("close", store, tmp_path, now=CLOSE_AT, session=STOPPED)
    assert outcome.outcome == "sent"
    ((text, auto_fix),) = sent.calls
    assert auto_fix == INFORMATIONAL
    assert text == CLOSE_ALL_OK


def test_partial_fills_at_the_close_are_counted_and_named(tmp_path: Path) -> None:
    _, sent = _notify("close", _close_store(DONE_WATCH), tmp_path, now=CLOSE_AT, session=STOPPED)
    ((text, auto_fix),) = sent.calls
    assert auto_fix == INFORMATIONAL
    assert (
        "✅ Trades executed today: 5 orders — 2 filled, 2 partly filled, 1 not filled; sold $37,698, "
        "bought $80,312; partly filled: CVLT 220/657, NTNX 817/1,334; not filled: MU 0/13"
    ) in _lines(text)


def test_a_working_order_left_at_the_broker_is_a_cross(tmp_path: Path) -> None:
    orders = (STOP, m.OpenOrder("CVLT", "SELL", "MKT", "DAY", 657, 220))
    _, sent = _notify("close", _close_store(), tmp_path, now=CLOSE_AT, session=STOPPED, orders=orders)
    ((text, auto_fix),) = sent.calls
    assert auto_fix != INFORMATIONAL
    assert "❌ Broker still has 1 non-stop order open: SELL CVLT MKT DAY" in _lines(text)
    assert "holds only GTC stops" not in text
    assert "*Impact:*" in text and "*You:*" in text


def test_a_session_that_hit_its_hard_stop_is_a_cross(tmp_path: Path) -> None:
    timeout = {
        "ActiveState": "failed",
        "Result": "timeout",
        "ExecMainStartTimestamp": "@1794407701",
        "ExecMainExitTimestamp": "@1794431100",
    }
    waiting = {**WATCH, "state": "stopped", "entries": [_entry("MU", "buy")]}
    _, sent = _notify("close", _close_store(waiting), tmp_path, now=CLOSE_AT, session=timeout)
    ((text, _),) = sent.calls
    lines = _lines(text)
    assert "❌ Trading session ran 09:35–16:05 ET and ended timeout (it hit the 16:05 hard stop)" in lines
    assert "❌ Timed orders still waiting at the stop: MU" in lines
    assert "stopped cleanly" not in text


def test_a_stopped_after_close_timer_is_a_cross(tmp_path: Path) -> None:
    run = _systemd(STOPPED)
    run.units["alpha-engine-trader-reconcile.timer"] = {"ActiveState": "inactive"}
    sent = _Sent()
    m.notify(
        "close",
        _close_store(),
        ENV,
        now=CLOSE_AT,
        client_id=44,
        run=run,
        broker=_broker(),
        send=sent,
        sent_dir=tmp_path,
    )
    ((text, _),) = sent.calls
    assert "❌ Still to run: reconcile 16:45 ET — but its timer is inactive" in _lines(text)
    assert "⏳ Still to run: pin smoke 16:10 ET" in _lines(text)


def test_a_close_with_no_run_record_says_so(tmp_path: Path) -> None:
    _, sent = _notify("close", _store({}), tmp_path, now=CLOSE_AT, session=STOPPED)
    ((text, _),) = sent.calls
    lines = _lines(text)
    assert "✅ Trades executed today: none" in lines
    assert f"❌ Trading session ran 09:35–15:56 ET but wrote no run record for {DAY}" in lines


def test_fills_the_run_filed_but_this_notice_cannot_read_are_a_cross(tmp_path: Path) -> None:
    run = {**RUN_OK, "outputs": [{"key": f"trader/execution_shortfall/{DAY}.json"}]}
    store = _store({f"runs/trader.session/{DAY}/run.json": run})
    _, sent = _notify("close", store, tmp_path, now=CLOSE_AT, session=STOPPED)
    ((text, _),) = sent.calls
    assert "❌ Trades executed today: the session filed its fills but they could not be read" in _lines(text)


# --- once a day, trading days only, delivery ---------------------------------


def test_a_second_run_the_same_day_sends_nothing(tmp_path: Path) -> None:
    store = _close_store()
    first, _ = _notify("close", store, tmp_path, now=CLOSE_AT, session=STOPPED)
    second, sent = _notify("close", store, tmp_path, now=CLOSE_AT, session=STOPPED)
    assert (first.outcome, second.outcome) == ("sent", "already_sent")
    assert sent.calls == []
    # The open's marker is its own.
    opened, _ = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING)
    assert opened.outcome == "sent"


def test_an_nyse_holiday_sends_nothing(tmp_path: Path) -> None:
    thanksgiving = dt.datetime(2026, 11, 26, 14, 40, tzinfo=dt.UTC)
    outcome, sent = _notify("open", _store({}), tmp_path, now=thanksgiving, session=RUNNING)
    assert outcome.outcome == "not_a_trading_day" and sent.calls == []


def test_an_undelivered_notice_fails_and_leaves_no_marker(tmp_path: Path) -> None:
    outcome, _ = _notify("open", _store({}), tmp_path, now=OPEN_AT, session=RUNNING, ok=False)
    assert outcome.failed
    assert not list(tmp_path.iterdir())


def test_the_alert_never_connects_on_a_trader_or_v1_client_id() -> None:
    for refused in (41, 42, 2):
        broker = m.read_broker(ENV, refused, want_orders=True)
        assert broker.error is not None and "not connecting" in broker.error


@pytest.mark.parametrize("argv", [[], ["opening"], ["open", "close"]])
def test_main_refuses_anything_but_open_or_close(argv: list[str]) -> None:
    assert m.main(argv) == 2


def test_main_requires_its_own_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(m.CLIENT_ID_VAR, raising=False)
    assert m.main(["open"]) == 2
