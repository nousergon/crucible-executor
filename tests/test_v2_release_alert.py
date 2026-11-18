"""`executor.v2_release_alert`: one Telegram per v2 order when its trigger releases it (alpha-engine-config-I12024)."""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from executor import v2_release_alert as r
from executor import v2_trade_alert as ta
from executor.v2_trade_alert import Store

NY = ZoneInfo("America/New_York")
TODAY = dt.date(2026, 11, 18)  # Wednesday; the session serves Tuesday 2026-11-17
DAY = "2026-11-17"
WATCH_KEY = f"trader/entry_watch/{DAY}.json"


def _at(hour: int, minute: int, day: dt.date = TODAY) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(dt.UTC)


def _iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entry(
    symbol: str,
    *,
    status: str = "sent",
    side: str = "sell",
    released: dt.datetime | None = None,
    trigger: str | None = "rally 2.08% off low 691.25",
    **kw: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "quantity": 33,
        "status": status,
        "released_at_utc": None if released is None else _iso(released),
        "release_price": 705.62,
        "reference_price": 721.9,
        "trigger": trigger,
        "order_key": f"crucible-trader|{DAY}|{symbol}",
    }
    entry.update(kw)
    return entry


def _watch(*entries: dict[str, Any], session_date: str = TODAY.isoformat()) -> dict[str, Any]:
    return {
        "trading_day": DAY,
        "session_date": session_date,
        "state": "watching",
        "schedule": {
            "opens_at": "2026-11-18T09:30:00-05:00",
            "graduated_at": "2026-11-18T15:30:00-05:00",
            "expires_at": "2026-11-18T15:55:00-05:00",
            "closes_at": "2026-11-18T16:00:00-05:00",
        },
        "entries": list(entries),
    }


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, Any]) -> None:
        self.objects = objects
        self.reads: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.reads.append(Key)
        if Key not in self.objects:
            raise _ClientError("AccessDenied")  # no ListBucket: absent reads as denied
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}

    def get_paginator(self, _name: str) -> _Paginator:
        return _Paginator(self.objects)


class _Paginator:
    def __init__(self, objects: dict[str, Any]) -> None:
        self.objects = objects

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        found = [{"Key": k, "LastModified": _at(10, 31)} for k in sorted(self.objects) if k.startswith(Prefix)]
        return [{"Contents": found}]


def _store(objects: dict[str, Any]) -> Store:
    return Store(_S3({f"crucible/{k}": v for k, v in objects.items()}), "bucket", "crucible")


class _Send:
    def __init__(self, *results: bool) -> None:
        self.results = list(results)
        self.texts: list[str] = []

    def __call__(self, text: str, *, auto_fix: str) -> bool:
        self.texts.append(text)
        return self.results.pop(0) if self.results else True


# --- the trigger, in plain words ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "side", "expected"),
    [
        ("rally 2.08% off low 691.25", "sell", "sold into a 2.1% rally off the day's low of $691.25"),
        ("pullback 2.31% from high 118.40", "buy", "bought on a 2.3% pullback from the day's high of $118.40"),
        ("vwap_discount 0.62% below VWAP 54.10", "buy", "bought 0.6% below today's VWAP of $54.10"),
        (
            "support_bounce 0.40% above support 1012.00",
            "buy",
            "bought on a bounce 0.4% above its support level (its recent low) of $1,012.00",
        ),
        (
            "graduated_entry (+0.30% vs sized price 100.00, limit +1.00%)",
            "buy",
            "bought in the 15:30 ET graduated window: the price was +0.3% against the planned $100.00, "
            "inside its +1.0% limit",
        ),
        (
            "graduated_exit (-0.50% vs sized price 42.00, limit -1.00%)",
            "sell",
            "sold in the 15:30 ET graduated window: the price was -0.5% against the planned $42.00, "
            "inside its -1.0% limit",
        ),
        ("time_expiry", "buy", "bought at the 15:55 ET expiry: no other trigger fired, and nothing waits past it"),
        ("time_expiry", "sell", "sold at the 15:55 ET expiry: no other trigger fired, and nothing waits past it"),
        (
            "catastrophic_gap_stop",
            "sell",
            "sold at the start on the catastrophic gap stop: hard risk never waits for a trigger",
        ),
        ("adopted: already at the broker", "buy", "already at the broker from an earlier run of today's session"),
        ("something *new* 1.5%", "buy", "something new 1.5%"),
        (None, "buy", "no trigger recorded"),
    ],
)
def test_every_trigger_kind_reads_in_plain_words(raw: str | None, side: str, expected: str) -> None:
    assert r.plain_trigger(raw, side=side) == expected


def test_a_half_day_names_its_own_window_times() -> None:
    watch = _watch()
    watch["schedule"]["graduated_at"] = "2026-11-27T12:30:00-05:00"
    watch["schedule"]["expires_at"] = "2026-11-27T12:55:00-05:00"
    entry = _entry("X", side="buy", trigger="time_expiry")
    assert "12:55 ET expiry" in r.why(entry, watch)
    entry = _entry("X", side="buy", trigger="graduated_entry (+0.30% vs sized price 100.00, limit +1.00%)")
    assert "12:30 ET graduated window" in r.why(entry, watch)


def test_the_message_says_what_when_at_what_price_and_why() -> None:
    text = r.release_message(_entry("MU", released=dt.datetime(2026, 11, 18, 15, 11, 59, tzinfo=dt.UTC)), _watch())
    assert "*SELL MU: order sent*" in text and "not yet a fill" in text
    assert "Shares: 33" in text
    assert "Released: 10:11 ET" in text
    assert "$705.62 vs planned $721.90 (-2.3%)" in text
    assert "sold into a 2.1% rally off the day's low of $691.25" in text
    assert "filled" not in text.lower().replace("not yet a fill", "")


def test_a_clipped_buy_reports_the_shares_released() -> None:
    entry = _entry("NVDA", side="buy", trigger="time_expiry", released=_at(15, 55), quantity=50, released_quantity=20)
    assert "Shares: 20" in r.release_message(entry, _watch())


# --- one tick ----------------------------------------------------------------------


def test_a_new_release_sends_one_message(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)), _entry("AAPL", status="waiting"))})
    send = _Send()
    out = r.tick(store, now=_at(10, 15), send=send, state_dir=tmp_path)
    assert out.sent == 1 and not out.failed
    assert len(send.texts) == 1 and "MU" in send.texts[0]
    state = json.loads((tmp_path / f"{TODAY}.json").read_text())
    assert state == {"alerted": [f"crucible-trader|{DAY}|MU"]}
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_a_triggered_entry_is_already_released(tmp_path: Path) -> None:
    """``triggered`` is what the watch carries the moment the order goes out."""
    store = _store({WATCH_KEY: _watch(_entry("MU", status="triggered", released=_at(10, 11)))})
    assert r.tick(store, now=_at(10, 15), send=_Send(), state_dir=tmp_path).sent == 1


def test_the_second_tick_sends_nothing(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)))})
    send = _Send()
    r.tick(store, now=_at(10, 15), send=send, state_dir=tmp_path)
    out = r.tick(store, now=_at(10, 20), send=send, state_dir=tmp_path)
    assert out.sent == 0 and not out.failed
    assert len(send.texts) == 1


def test_a_later_release_is_alerted_on_its_own(tmp_path: Path) -> None:
    first = _entry("MU", released=_at(9, 45))
    r.tick(_store({WATCH_KEY: _watch(first)}), now=_at(9, 50), send=_Send(), state_dir=tmp_path)
    send = _Send()
    second = _entry("AAPL", side="buy", trigger="pullback 2.31% from high 118.40", released=_at(10, 30))
    out = r.tick(_store({WATCH_KEY: _watch(first, second)}), now=_at(10, 35), send=send, state_dir=tmp_path)
    assert out.sent == 1 and len(send.texts) == 1 and "AAPL" in send.texts[0]


def test_an_undelivered_alert_is_retried_and_fails_the_tick(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)), _entry("LRCX", released=_at(10, 12)))})
    out = r.tick(store, now=_at(10, 15), send=_Send(False), state_dir=tmp_path)
    assert out.sent == 0 and out.undelivered == 2 and out.failed
    assert not (tmp_path / f"{TODAY}.json").exists()
    send = _Send()
    out = r.tick(store, now=_at(10, 20), send=send, state_dir=tmp_path)
    assert out.sent == 2 and not out.failed and len(send.texts) == 2


def test_a_partly_delivered_tick_records_only_what_went(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)), _entry("LRCX", released=_at(10, 12)))})
    out = r.tick(store, now=_at(10, 15), send=_Send(True, False), state_dir=tmp_path)
    assert out.sent == 1 and out.undelivered == 1 and out.failed
    send = _Send()
    r.tick(store, now=_at(10, 20), send=send, state_dir=tmp_path)
    assert len(send.texts) == 1 and "LRCX" in send.texts[0]


def test_an_unreadable_state_file_sends_nothing_and_fails(tmp_path: Path) -> None:
    (tmp_path / f"{TODAY}.json").write_text("{not json")
    send = _Send()
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)))})
    out = r.tick(store, now=_at(10, 15), send=send, state_dir=tmp_path)
    assert out.failed and not send.texts


def test_a_non_trading_day_does_nothing(tmp_path: Path) -> None:
    saturday = dt.date(2026, 11, 21)
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)))})
    send = _Send()
    out = r.tick(store, now=_at(10, 15, saturday), send=send, state_dir=tmp_path)
    assert not send.texts and not out.failed
    assert store.client.reads == []  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("hour", "minute"), [(9, 0), (9, 25), (16, 30)])
def test_outside_the_session_window_does_nothing(tmp_path: Path, hour: int, minute: int) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(9, 30)))})
    send = _Send()
    out = r.tick(store, now=_at(hour, minute), send=send, state_dir=tmp_path)
    assert not send.texts and not out.failed


def test_the_16_05_pass_alerts_the_expiry_releases(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", trigger="time_expiry", released=_at(15, 55)))})
    assert r.tick(store, now=_at(16, 5), send=_Send(), state_dir=tmp_path).sent == 1


def test_no_watch_does_nothing(tmp_path: Path) -> None:
    send = _Send()
    out = r.tick(_store({}), now=_at(10, 15), send=send, state_dir=tmp_path)
    assert not send.texts and not out.failed and "no entry watch" in out.reason


def test_a_watch_for_another_session_date_does_nothing(tmp_path: Path) -> None:
    store = _store({WATCH_KEY: _watch(_entry("MU", released=_at(10, 11)), session_date="2026-11-17")})
    send = _Send()
    assert not r.tick(store, now=_at(10, 15), send=send, state_dir=tmp_path).failed
    assert not send.texts


def test_unreleased_entries_are_never_alerted(tmp_path: Path) -> None:
    store = _store(
        {
            WATCH_KEY: _watch(
                _entry("A", status="waiting"),
                _entry("B", status="not_sent", trigger="market closed before a trigger"),
            )
        }
    )
    send = _Send()
    assert r.tick(store, now=_at(15, 0), send=send, state_dir=tmp_path).sent == 0
    assert not send.texts


def test_main_outside_the_window_reads_nothing() -> None:
    assert r.main([], now=_at(8, 0)) == 0
    assert r.main(["x"], now=_at(10, 0)) == 2


# --- the per-fill alert at the session's end ---------------------------------------


def _run() -> dict[str, Any]:
    return {"status": "ok", "outputs": [{"key": ta.shortfall_key(DAY)}]}


def test_the_fill_alert_names_when_the_order_was_sent_and_why() -> None:
    order = {
        "symbol": "MU",
        "side": "sell",
        "quantity": 33,
        "filled_quantity": 33,
        "fill_price": 705.1,
        "decision_price": 721.9,
        "submitted_at_utc": "2026-11-18T15:12:01Z",
    }
    store = _store(
        {
            f"{ta.RUNS_PREFIX}{DAY}/run.json": _run(),
            ta.shortfall_key(DAY): {"orders": [order]},
            WATCH_KEY: _watch(_entry("MU", released=dt.datetime(2026, 11, 18, 15, 11, 59, tzinfo=dt.UTC))),
        }
    )
    trades: list[dict[str, Any]] = []

    def trade(*args: Any, **kw: Any) -> bool:
        trades.append({"args": args, **kw})
        return True

    ta.run_alerts(store, now=_at(10, 32), trade=trade, status=lambda *_a, **_k: True)
    assert len(trades) == 1
    trigger = trades[0]["trigger"]
    assert trigger.startswith("fill of the order sent 10:11 ET")
    assert "sold into a 2.1% rally off the day's low of $691.25" in trigger


def test_the_fill_alert_reads_as_before_without_a_watch() -> None:
    order = {
        "symbol": "MU",
        "side": "buy",
        "quantity": 5,
        "filled_quantity": 5,
        "fill_price": 10.0,
        "decision_price": 9.5,
    }
    store = _store({f"{ta.RUNS_PREFIX}{DAY}/run.json": _run(), ta.shortfall_key(DAY): {"orders": [order]}})
    trades: list[dict[str, Any]] = []

    def trade(*args: Any, **kw: Any) -> bool:
        trades.append(kw)
        return True

    ta.run_alerts(store, now=_at(10, 32), trade=trade, status=lambda *_a, **_k: True)
    assert trades[0]["trigger"] == f"v2 book {DAY} (decision $9.50)"
