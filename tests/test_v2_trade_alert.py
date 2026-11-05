"""`executor.v2_trade_alert`: the v2 session's Telegram alerts (alpha-engine-config-I13005)."""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
from typing import Any

import pytest

from executor import v2_trade_alert as alert

NOW = dt.datetime(2026, 11, 4, 18, 8, tzinfo=dt.UTC)
DAY = "2026-11-03"


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    """botocore's ClientError shape: the code lives in ``response``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    """A fake S3 client over ``{key: (LastModified, body)}``."""

    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, tuple[dt.datetime, Any]], *, can_list: bool = True) -> None:
        self.objects = objects
        self.can_list = can_list

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            if not self.can_list:
                raise _ClientError("AccessDenied")
            raise _NoSuchKey(Key)
        return {"Body": io.BytesIO(json.dumps(self.objects[Key][1]).encode())}

    def get_paginator(self, _name: str) -> _S3:
        return self

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        return [
            {"Contents": [{"Key": k, "LastModified": m} for k, (m, _) in self.objects.items() if k.startswith(Prefix)]}
        ]


def _order(symbol: str, side: str, qty: float, filled: float | None = None) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "filled_quantity": qty if filled is None else filled,
        "decision_price": 100.0,
        "fill_price": 100.5,
        "notional_usd": qty * 100.5,
        "submitted_at_utc": "2026-11-04T18:06:41Z",
    }


SHORTFALL = {
    "champion": "m:arm:abc",
    "orders": [_order("AAA", "sell", 10), _order("BBB", "buy", 20)],
    "metrics": [{"name": "execution_shortfall_bps", "value": -6.27, "status": "OK"}],
}


def _store(run: dict[str, Any] | None, shortfall: Any = SHORTFALL, *, at: dt.datetime = NOW) -> alert.Store:
    objects: dict[str, tuple[dt.datetime, Any]] = {
        "crucible/runs/trader.session/2026-10-30/run.json": (at - dt.timedelta(days=3), {"status": "ok"}),
    }
    if run is not None:
        objects[f"crucible/runs/trader.session/{DAY}/run.json"] = (at, run)
        objects[f"crucible/runs/trader.session/{DAY}/other.json"] = (at, {})
    if shortfall is not None:
        objects[f"crucible/trader/execution_shortfall/{DAY}.json"] = (at, shortfall)
    return alert.Store(_S3(objects), "bucket", "crucible")


class _Recorder:
    def __init__(self, result: bool = True) -> None:
        self.trades: list[tuple[Any, ...]] = []
        self.statuses: list[str] = []
        self.result = result

    def trade(self, *args: Any, **kwargs: Any) -> bool:
        self.trades.append((*args, kwargs))
        return self.result

    def status(self, message: str, *, essential: bool) -> bool:
        assert essential
        self.statuses.append(message)
        return self.result


def test_every_fill_is_alerted_and_the_session_summarised() -> None:
    rec = _Recorder()
    sent = alert.run_alerts(_store({"status": "ok"}), now=NOW, trade=rec.trade, status=rec.status)
    assert sent == 3
    assert [t[:4] for t in rec.trades] == [("SELL", "AAA", 10, 100.5), ("BUY", "BBB", 20, 100.5)]
    assert rec.trades[0][4]["trigger"] == f"v2 book {DAY} (decision $100.00)"
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: 2/2 orders filled*")
    assert "Sells $1,005 · Buys $2,010" in summary and "Shortfall -6.3 bps (OK)" in summary
    assert "Champion: m:arm:abc" in summary


def test_an_unfilled_order_is_named_and_not_alerted_as_a_trade() -> None:
    shortfall = {**SHORTFALL, "orders": [_order("AAA", "sell", 10), _order("BBB", "buy", 20, 0)]}
    rec = _Recorder(result=False)
    sent = alert.run_alerts(_store({"status": "ok"}, shortfall), now=NOW, trade=rec.trade, status=rec.status)
    assert sent == 0 and len(rec.trades) == 1
    assert "1/2 orders filled" in rec.statuses[0] and "UNFILLED: BBB" in rec.statuses[0]


def test_a_refused_book_says_no_orders_and_why() -> None:
    rec = _Recorder()
    run = {"status": "failed", "reason": "RiskCapExceededError: PLTR over the per-order cap"}
    alert.run_alerts(_store(run), now=NOW, trade=rec.trade, status=rec.status)
    assert rec.trades == []
    assert "NO ORDERS PLACED" in rec.statuses[0] and "PLTR over the per-order cap" in rec.statuses[0]
    alert.run_alerts(_store({"status": "failed"}), now=NOW, trade=rec.trade, status=rec.status)
    assert "Reason: none recorded" in rec.statuses[1]


def test_a_session_with_no_orders_says_so() -> None:
    rec = _Recorder()
    alert.run_alerts(
        _store({"status": "ok"}, {"orders": [], "reason": "shadow"}), now=NOW, trade=rec.trade, status=rec.status
    )
    assert rec.statuses == [f"*v2 session {DAY}: no orders*\nshadow"]
    alert.run_alerts(_store({"status": "ok"}, None), now=NOW, trade=rec.trade, status=rec.status)
    assert rec.statuses[1].endswith("the book needed no orders")


def test_a_stale_or_absent_run_record_is_reported() -> None:
    rec = _Recorder()
    stale = _store({"status": "ok"}, at=NOW - dt.timedelta(hours=4))
    assert alert.run_alerts(stale, now=NOW, trade=rec.trade, status=rec.status) == 1
    assert "2026-11-04: NO RUN RECORD" in rec.statuses[0]
    empty = alert.Store(_S3({}), "bucket", "")
    alert.run_alerts(empty, now=NOW, trade=rec.trade, status=rec.status)
    assert len(rec.statuses) == 2 and rec.trades == []


def test_the_env_file_is_read_without_comments(tmp_path: Path) -> None:
    path = tmp_path / "trader.env"
    path.write_text('# comment\n\nAWS_PROFILE=crucible-v2-trader\nX="quoted"\nnot a line\n')
    assert alert.read_env_file(path) == {"AWS_PROFILE": "crucible-v2-trader", "X": "quoted"}


def test_the_store_is_opened_as_the_traders_profile(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("[profile p]\naws_access_key_id = a\naws_secret_access_key = b\n")
    store = alert.open_store(
        {"CRUCIBLE_TRADER_STORE_URI": "s3://bkt/crucible/", "AWS_PROFILE": "p", "AWS_CONFIG_FILE": str(config)}
    )
    assert (store.bucket, store.prefix) == ("bkt", "crucible")
    assert store.key("x") == "crucible/x"
    assert alert.open_store({"CRUCIBLE_TRADER_STORE_URI": "s3://bkt"}).key("x") == "x"
    with pytest.raises(ValueError, match="not an s3 URI"):
        alert.open_store({"CRUCIBLE_TRADER_STORE_URI": "file:///tmp"})


def test_main_takes_no_arguments_and_reads_the_traders_env(monkeypatch, tmp_path: Path) -> None:
    assert alert.main(["x"]) == 2
    env = tmp_path / "trader.env"
    env.write_text("CRUCIBLE_TRADER_STORE_URI=s3://bkt/crucible\n")
    monkeypatch.setattr(alert, "TRADER_ENV", env)
    seen: list[Any] = []
    monkeypatch.setattr(alert, "open_store", lambda values: values)
    monkeypatch.setattr(alert, "run_alerts", lambda store, now: seen.append((store, now)) or 0)
    assert alert.main([], now=NOW) == 0
    assert seen == [({"CRUCIBLE_TRADER_STORE_URI": "s3://bkt/crucible"}, NOW)]


HELD_RUN = {
    "status": "ok",
    "metrics": [
        {"name": "trader_session_served", "value": 0.0, "status_reason": "held: no predictions feed for trading day"}
    ],
}


def test_a_held_session_without_list_access_is_reported_as_held() -> None:
    """2026-11-05: no ListBucket turned the absent shortfall file into a 403 crash."""
    objects = {f"crucible/runs/trader.session/{DAY}/run.json": (NOW, HELD_RUN)}
    store = alert.Store(_S3(objects, can_list=False), "bucket", "crucible")
    rec = _Recorder()
    assert alert.run_alerts(store, now=NOW, trade=rec.trade, status=rec.status) == 1
    assert rec.trades == []
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: HELD, NO ORDERS PLACED*")
    assert "no predictions feed" in summary


def test_other_s3_errors_still_raise() -> None:
    class _Broken(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise _ClientError("SlowDown")

    with pytest.raises(_ClientError):
        alert.Store(_Broken({}), "bucket", "crucible").read_json("x.json")


def test_an_error_without_a_botocore_response_has_no_code() -> None:
    assert alert._error_code(RuntimeError()) == ""
    err = RuntimeError()
    err.response = {"Error": "x"}  # type: ignore[attr-defined]
    assert alert._error_code(err) == ""
