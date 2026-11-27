"""`executor.v2_order_book_alert`: the pre-open order book, announced once (alpha-engine-config-I12024)."""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
from typing import Any

import pytest

from executor import v2_order_book_alert as book_alert
from executor import v2_trade_alert as alert

#: Tuesday 2026-11-10 08:45 ET: the plan ran at 08:40; the last closed session is 11-09.
NOW = dt.datetime(2026, 11, 10, 13, 45, tzinfo=dt.UTC)
DAY = "2026-11-09"
KEY = f"crucible/trader/order_books/{DAY}.json"


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, Any], *, error: str | None = None) -> None:
        self.objects = objects
        self.error = error

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if self.error is not None:
            raise _ClientError(self.error)
        if Key not in self.objects:
            raise _ClientError("AccessDenied")  # the trader role has no ListBucket
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}

    def get_paginator(self, _name: str) -> _S3:
        return self


def _row(ticker: str, action: str, target: float) -> dict[str, Any]:
    return {"ticker": ticker, "action": action, "target_weight": target}


BOOK = {
    "stage": "pre_open",
    "written_at_utc": "2026-11-10T13:40:12Z",
    "m_champion": "m:arm:abc",
    "s_champion": "s:arm:def",
    "turnover_one_way_ratio": 0.12,
    "summary": {"positions": 3, "cash_weight": 0.05},
    "rows": [
        _row("AAA", "hold", 0.4),
        _row("BBB", "enter", 0.35),
        _row("CCC", "exit", 0.0),
        _row("DDD", "reduce", 0.2),
        _row("EEE", "not_selected", 0.0),
        _row("SPY", "benchmark", 0.0),
        _row("$CASH", "cash", 0.05),
    ],
}


class _Status:
    def __init__(self, delivered: bool = True) -> None:
        self.delivered = delivered
        self.sent: list[tuple[str, bool]] = []

    def __call__(self, message: str, *, essential: bool = False) -> bool:
        self.sent.append((message, essential))
        return self.delivered


def _announce(
    objects: dict[str, Any],
    tmp_path: Path,
    *,
    now: dt.datetime = NOW,
    status: _Status | None = None,
    error: str | None = None,
) -> tuple[book_alert.Outcome, _Status]:
    status = status or _Status()
    store = alert.Store(_S3(objects, error=error), "b", "crucible")
    return book_alert.announce(store, now=now, status=status, announced_dir=tmp_path), status


def test_the_pre_open_book_is_announced_once_as_an_essential_message(tmp_path: Path) -> None:
    outcome, status = _announce({KEY: BOOK}, tmp_path)
    assert outcome.outcome == "announced" and not outcome.failed
    ((message, essential),) = status.sent
    assert essential
    lines = message.splitlines()
    assert lines[0] == f"*v2 order book {DAY} (pre-open)*"
    assert "3 positions · cash 5.0% · turnover 12.0% one way" in lines
    assert "Enter: BBB" in lines and "Exit: CCC" in lines and "Reduce: DDD" in lines
    assert "Top: AAA 40.0%, BBB 35.0%, DDD 20.0%" in lines
    assert (tmp_path / DAY).exists()
    # A second start the same day (a rerun of the plan) sends nothing.
    again, status = _announce({KEY: BOOK}, tmp_path)
    assert again.outcome == "already_announced" and status.sent == []


@pytest.mark.parametrize(
    "book",
    [
        {**BOOK, "stage": "session"},
        {**BOOK, "written_at_utc": "2026-11-09T13:40:00Z"},
        {**BOOK, "written_at_utc": "yesterday"},
        {**BOOK, "written_at_utc": None},
    ],
)
def test_only_todays_pre_open_book_is_announced(tmp_path: Path, book: dict[str, Any]) -> None:
    outcome, status = _announce({KEY: book}, tmp_path)
    assert outcome.outcome == "no_book" and status.sent == [] and not outcome.failed


def test_a_held_day_with_no_book_says_nothing(tmp_path: Path) -> None:
    outcome, status = _announce({}, tmp_path)
    assert outcome.outcome == "no_book" and status.sent == [] and not outcome.failed


def test_a_day_off_says_nothing(tmp_path: Path) -> None:
    saturday = dt.datetime(2026, 11, 14, 13, 45, tzinfo=dt.UTC)
    outcome, status = _announce({KEY: BOOK}, tmp_path, now=saturday)
    assert outcome.outcome == "no_book" and status.sent == []


def test_an_unreadable_book_fails_and_pages(tmp_path: Path) -> None:
    outcome, status = _announce({KEY: BOOK}, tmp_path, error="SlowDown")
    assert outcome.failed and "SlowDown" in outcome.reason and status.sent == []


def test_an_undelivered_announcement_fails_and_can_be_retried(tmp_path: Path) -> None:
    outcome, _ = _announce({KEY: BOOK}, tmp_path, status=_Status(delivered=False))
    assert outcome.failed and not (tmp_path / DAY).exists()


def test_main_takes_no_arguments() -> None:
    assert book_alert.main(["x"]) == 2


def test_main_exits_non_zero_only_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(book_alert, "read_env_file", lambda path: {})
    monkeypatch.setattr(book_alert, "open_store", lambda env: alert.Store(_S3({}), "b", "crucible"))
    assert book_alert.main([], now=NOW) == 0
    monkeypatch.setattr(book_alert, "open_store", lambda env: alert.Store(_S3({}, error="SlowDown"), "b", "crucible"))
    assert book_alert.main([], now=NOW) == 1
