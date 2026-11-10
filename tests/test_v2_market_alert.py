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

    def __init__(self, objects: dict[str, Any], *, denied: Sequence[str] = ()) -> None:
        self.objects = objects
        self.denied = set(denied)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key in self.denied:
            raise _ClientError("InternalError")
        if Key not in self.objects:
            raise _ClientError("AccessDenied")  # the trader role has no ListBucket
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}

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


def _store(objects: dict[str, Any], **kw: Any) -> Store:
    return Store(_S3({f"crucible/{k}": v for k, v in objects.items()}, **kw), "bucket", "crucible")


class _Systemd:
    """``systemctl show`` answers per unit."""

    def __init__(self, units: dict[str, dict[str, str]]) -> None:
        self.units = units

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert "--timestamp=unix" in argv
        props = self.units.get(argv[2], {})
        out = "\n".join(f"{k}={v}" for k, v in props.items())
        return subprocess.CompletedProcess(list(argv), 0, stdout=out, stderr="")


RUNNING = {"ActiveState": "activating", "Result": "success", "ExecMainStartTimestamp": "@1794407401"}
STOPPED = {"ActiveState": "inactive", "Result": "success", "ExecMainExitTimestamp": "@1794430600"}
ACTIVE = {"ActiveState": "active"}


def _systemd(session: dict[str, str]) -> _Systemd:
    return _Systemd(
        {
            m.SESSION_UNIT: session,
            m.GATEWAY_UNIT: ACTIVE,
            "alpha-engine-trader-pin-smoke.timer": ACTIVE,
            "alpha-engine-trader-reconcile.timer": ACTIVE,
        }
    )


def _broker(**kw: Any) -> Any:
    def read(_env: Any, _client_id: int, *, want_orders: bool) -> m.Broker:
        orders = kw.get("orders", (m.OpenOrder("SPY", "SELL", "TRAIL", "GTC"),)) if want_orders else None
        return m.Broker(error=kw.get("error"), account_kind=kw.get("account", "paper"), open_orders=orders)

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
    outcome = m.notify(
        kind,
        store,
        ENV,
        now=now,
        client_id=44,
        run=_systemd(session),
        broker=_broker(**kw),
        send=sent,
        sent_dir=tmp_path,
    )
    return outcome, sent


# --- the open -----------------------------------------------------------------


def test_an_all_clear_open_is_a_short_pointer_with_no_footer(tmp_path: Path) -> None:
    store = _store({f"trader/order_books/{DAY}.json": BOOK, f"trader/entry_watch/{DAY}.json": WATCH})
    outcome, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING)
    assert outcome.outcome == "sent"
    ((text, auto_fix),) = sent.calls
    assert auto_fix == INFORMATIONAL
    assert text.splitlines()[0] == "*Market open, v2 trading day 2026-11-11*"
    assert "pre-open order book for 2026-11-10 was written at 08:41 ET: 4 target changes" in text
    assert "(2 enter, 1 exit, 1 increase), 4 orders planned (2 sell, 2 buy), 3 waiting on their triggers" in text
    assert "IB Gateway (paper)" in text and "the session (since 09:30 ET)" in text
    assert "its entry/exit watch" in text and "trader release 811eb9b" in text
    assert text.rstrip().endswith(m.CONSOLE_URL)
    for footer in ("Impact:", "Auto-fix:", "You:", "Problem:"):
        assert footer not in text
    assert len(text.splitlines()) == 4


def test_what_is_not_up_at_the_open_is_a_problem_with_a_footer(tmp_path: Path) -> None:
    stale = {**WATCH, "updated_at_utc": "2026-11-11T14:00:00Z"}
    store = _store({f"trader/order_books/{DAY}.json": BOOK, f"trader/entry_watch/{DAY}.json": stale})
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING, error="ConnectionRefusedError: x")
    ((text, auto_fix),) = sent.calls
    assert auto_fix != INFORMATIONAL
    assert ": 2 problems*" in text.splitlines()[0]
    assert "Problem: IB Gateway did not answer an API session (ConnectionRefusedError: x)." in text
    assert "Problem: the entry/exit watch has not written since 09:00 ET." in text
    assert "IB Gateway (" not in text  # not listed as up
    for footer in ("*Impact:*", "*Auto-fix:*", "*You:*"):
        assert footer in text
    assert text.rstrip().endswith(m.CONSOLE_URL)


def test_a_session_that_already_ended_at_the_open_is_a_problem(tmp_path: Path) -> None:
    """A release without the entry timing sends everything at the start and
    ends in minutes: that is exactly what the open must flag."""
    store = _store({f"trader/order_books/{DAY}.json": BOOK, f"runs/trader.session/{DAY}/run.json": RUN_OK})
    ended = {"ActiveState": "inactive", "Result": "success", "ExecMainExitTimestamp": "@1794407770"}
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=ended)
    ((text, _),) = sent.calls
    assert "Problem: the session is not running (inactive, result success, ended 09:36 ET)." in text
    assert "trader release not recorded" in text


def test_a_held_session_is_named_at_the_open(tmp_path: Path) -> None:
    held = {"status": "ok", "metrics": [{"name": "trader_session_served", "value": 0, "status_reason": "no feed"}]}
    store = _store({f"runs/trader.session/{DAY}/run.json": held})
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session={"ActiveState": "inactive"})
    ((text, _),) = sent.calls
    assert "No order book for 2026-11-10 yet." in text
    assert "Problem: the session HELD and placed no order: no feed." in text


def test_a_live_account_on_the_gateway_is_a_problem_and_no_account_id_is_printed(tmp_path: Path) -> None:
    store = _store({f"trader/order_books/{DAY}.json": BOOK, f"trader/entry_watch/{DAY}.json": WATCH})
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING, account="live")
    ((text, _),) = sent.calls
    assert "Problem: the gateway's account type is live, not paper." in text
    assert "DU" not in text


def test_an_unreadable_document_is_a_problem_not_an_absence(tmp_path: Path) -> None:
    store = _store({}, denied=[f"crucible/trader/order_books/{DAY}.json"])
    _, sent = _notify("open", store, tmp_path, now=OPEN_AT, session=RUNNING)
    ((text, _),) = sent.calls
    assert f"Problem: trader/order_books/{DAY}.json is unreadable (InternalError)." in text


# --- the close ----------------------------------------------------------------


def _close_store(watch: dict[str, Any] | None = None) -> Store:
    objects: dict[str, Any] = {
        f"runs/trader.session/{DAY}/run.json": RUN_OK,
        f"trader/execution_shortfall/{DAY}.json": SHORTFALL,
    }
    if watch is not None:
        objects[f"trader/entry_watch/{DAY}.json"] = watch
    return _store(objects)


def test_an_all_clear_close_counts_fills_and_says_what_still_runs(tmp_path: Path) -> None:
    done = {**WATCH, "state": "complete", "entries": [_entry("ANF", "sell", "sent"), _entry("PLTR", "buy", "not_sent")]}
    outcome, sent = _notify("close", _close_store(done), tmp_path, now=CLOSE_AT, session=STOPPED)
    assert outcome.outcome == "sent"
    ((text, auto_fix),) = sent.calls
    assert auto_fix == INFORMATIONAL
    assert text.splitlines()[0] == "*Market close, v2 trading day 2026-11-11*"
    assert (
        "5 orders: 2 filled, 2 partly filled, 1 not filled; sold $37,698, bought $80,312 "
        "(partial or unfilled: CVLT, NTNX, MU)." in text
    )
    assert "The session stopped cleanly at 15:56 ET (1 timed order not sent)" in text
    assert "the broker holds only GTC stops (1)" in text
    assert "pin smoke 16:10 and broker reconcile 16:45 ET" in text and "01:00 ET" in text
    assert "Auto-fix:" not in text and "Problem:" not in text
    assert text.rstrip().endswith(m.CONSOLE_URL)


def test_a_working_order_left_at_the_broker_is_a_problem(tmp_path: Path) -> None:
    orders = (m.OpenOrder("SPY", "SELL", "TRAIL", "GTC"), m.OpenOrder("CVLT", "SELL", "MKT", "DAY"))
    _, sent = _notify("close", _close_store(), tmp_path, now=CLOSE_AT, session=STOPPED, orders=orders)
    ((text, auto_fix),) = sent.calls
    assert auto_fix != INFORMATIONAL
    assert "Problem: 1 non-stop order still open at the broker: SELL CVLT MKT DAY." in text
    assert "holds only GTC stops" not in text
    assert "*Impact:*" in text and "*You:*" in text


def test_a_session_that_hit_its_hard_stop_is_a_problem(tmp_path: Path) -> None:
    timeout = {"ActiveState": "failed", "Result": "timeout"}
    waiting = {**WATCH, "state": "stopped", "entries": [_entry("MU", "buy")]}
    _, sent = _notify("close", _close_store(waiting), tmp_path, now=CLOSE_AT, session=timeout)
    ((text, _),) = sent.calls
    assert "Problem: the session ended timeout (it hit the 16:05 hard stop)." in text
    assert "Problem: timed orders still waiting at the stop: MU." in text
    assert "stopped cleanly" not in text


def test_a_close_with_no_run_record_says_so(tmp_path: Path) -> None:
    _, sent = _notify("close", _store({}), tmp_path, now=CLOSE_AT, session=STOPPED)
    ((text, _),) = sent.calls
    assert "No orders today." in text
    assert f"Problem: the session wrote no run record for {DAY}." in text


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
