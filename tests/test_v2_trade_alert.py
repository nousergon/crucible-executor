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


def _store(
    run: dict[str, Any] | None, shortfall: Any = SHORTFALL, *, at: dt.datetime = NOW, book: Any = None
) -> alert.Store:
    objects: dict[str, tuple[dt.datetime, Any]] = {
        "crucible/runs/trader.session/2026-10-30/run.json": (at - dt.timedelta(days=3), {"status": "ok"}),
    }
    if run is not None:
        objects[f"crucible/runs/trader.session/{DAY}/run.json"] = (at, run)
        objects[f"crucible/runs/trader.session/{DAY}/other.json"] = (at, {})
    if shortfall is not None:
        objects[f"crucible/trader/execution_shortfall/{DAY}.json"] = (at, shortfall)
    if book is not None:
        objects[f"crucible/trader/order_books/{DAY}.json"] = (at, book)
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
    outcome = alert.run_alerts(_store({"status": "ok"}), now=NOW, trade=rec.trade, status=rec.status)
    assert outcome.delivered == 3 and not outcome.failed
    assert [t[:4] for t in rec.trades] == [("SELL", "AAA", 10, 100.5), ("BUY", "BBB", 20, 100.5)]
    assert rec.trades[0][4]["trigger"] == f"v2 book {DAY} (decision $100.00)"
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: 2/2 orders filled*")
    assert "Sells $1,005 · Buys $2,010" in summary and "Shortfall -6.3 bps (OK)" in summary
    assert "Champion: m:arm:abc" in summary


def test_an_unfilled_order_is_named_and_not_alerted_as_a_trade() -> None:
    shortfall = {**SHORTFALL, "orders": [_order("AAA", "sell", 10), _order("BBB", "buy", 20, 0)]}
    rec = _Recorder(result=False)
    outcome = alert.run_alerts(_store({"status": "ok"}, shortfall), now=NOW, trade=rec.trade, status=rec.status)
    assert outcome.delivered == 0 and outcome.failed and len(rec.trades) == 1
    assert "1/2 orders filled" in rec.statuses[0] and "UNFILLED: BBB" in rec.statuses[0]


def test_a_partial_fill_is_never_counted_as_filled() -> None:
    """2026-11-10: the summary said 7/7 filled while CVLT filled 220 of 657 and NTNX 817 of 1334."""
    orders = [
        _order("AAA", "sell", 10),
        _order("CVLT", "sell", 657, 220),
        _order("NTNX", "buy", 1334, 817),
        _order("BBB", "buy", 20, 0),
    ]
    rec = _Recorder()
    alert.run_alerts(
        _store({"status": "ok"}, {**SHORTFALL, "orders": orders}), now=NOW, trade=rec.trade, status=rec.status
    )
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: 1/4 orders filled, 2 partly filled*")
    assert "PARTLY FILLED: CVLT 220/657, NTNX 817/1,334" in summary and "UNFILLED: BBB" in summary
    # Each partial fill is still alerted, for the shares that filled.
    assert [t[:3] for t in rec.trades] == [("SELL", "AAA", 10), ("SELL", "CVLT", 220), ("BUY", "NTNX", 817)]


def test_a_failed_run_counts_its_partial_fills_apart() -> None:
    orders = [_order("AAA", "sell", 10, 4), _order("BBB", "buy", 20, 20)]
    run = {"status": "failed", "reason": "RoutingRefusedError: the gateway dropped", "outputs": FILED}
    rec = _Recorder()
    alert.run_alerts(_store(run, {**SHORTFALL, "orders": orders}), now=NOW, trade=rec.trade, status=rec.status)
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: RUN FAILED after 2 order(s) placed, 1 filled, 1 partly filled*")
    assert "PARTLY FILLED: AAA 4/10" in summary


@pytest.mark.parametrize(
    ("quantity", "filled", "bucket"),
    [(10, 10, 0), (10, 11, 0), (10, 9, 1), (10, 0.5, 1), (10, 0, 2), (10, -1, 2)],
)
def test_split_fills_buckets_each_order_once(quantity: float, filled: float, bucket: int) -> None:
    buckets = alert.split_fills([{"symbol": "X", "quantity": quantity, "filled_quantity": filled}])
    assert [len(b) for b in buckets] == [int(i == bucket) for i in range(3)]


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
    assert alert.run_alerts(stale, now=NOW, trade=rec.trade, status=rec.status) == alert.Outcome(1, True)
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
    outcomes = [
        alert.Outcome(2, True),
        alert.Outcome(1, False),
        alert.Outcome(1, True, run_unreadable=True),
        alert.Outcome(1, True, fills_unreadable=True),
    ]
    monkeypatch.setattr(alert, "run_alerts", lambda store, now: seen.append((store, now)) or outcomes.pop(0))
    assert alert.main([], now=NOW) == 0
    assert seen == [({"CRUCIBLE_TRADER_STORE_URI": "s3://bkt/crucible"}, NOW)]
    # Undelivered summary, unreadable run record, unreadable fills: non-zero, so OnFailure= pages.
    assert alert.main([], now=NOW) == 1
    assert alert.main([], now=NOW) == 1
    assert alert.main([], now=NOW) == 1


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
    assert alert.run_alerts(store, now=NOW, trade=rec.trade, status=rec.status).delivered == 1
    assert rec.trades == []
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: HELD, NO ORDERS PLACED*")
    assert "no predictions feed" in summary


def test_read_json_still_raises_other_s3_errors() -> None:
    class _Broken(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise _ClientError("SlowDown")

    with pytest.raises(_ClientError):
        alert.Store(_Broken({}), "bucket", "crucible").read_json("x.json")
    assert alert.Store(_S3({}, can_list=False), "b", "").read_json("x.json") is None


def test_other_s3_errors_are_returned_as_a_reason_not_raised() -> None:
    class _Broken(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise _ClientError("SlowDown")

    assert alert.Store(_Broken({}), "bucket", "crucible").read("x.json") == (None, "SlowDown")


def test_the_store_read_classifies_absent_denied_and_non_object() -> None:
    store = alert.Store(_S3({"crucible/list.json": (NOW, [1])}), "bucket", "crucible")
    assert store.read("absent.json") == (None, None)
    assert store.read("list.json") == (None, "not a JSON object")
    assert alert.Store(_S3({}, can_list=False), "b", "").read("x.json") == (None, "AccessDenied")

    class _NotFound(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise _ClientError("404")

    assert alert.Store(_NotFound({}), "b", "").read("x.json") == (None, None)


def test_an_error_without_a_botocore_response_has_no_code() -> None:
    assert alert._error_code(RuntimeError()) == ""
    err = RuntimeError()
    err.response = {"Error": "x"}  # type: ignore[attr-defined]
    assert alert._error_code(err) == ""


# alpha-engine-config-I13284: on 2026-11-04 the session placed 13 orders, the
# trader role could not read the shortfall file it wrote, and the alert said
# "no orders -- the book needed no orders" with zero per-fill alerts.

BOOK_FAIL = {
    "name": "trader_order_book_written",
    "value": 0.0,
    "status": "FAIL",
    "status_reason": (
        "order book not written: AccessDenied: An error occurred (AccessDenied) when calling the "
        "PutObject operation: User: arn:aws:sts::ACCOUNT:assumed-role/r/s is not authorized"
    ),
}
BOOK_OK = {"name": "trader_order_book_written", "value": 1.0, "status": "OK", "status_reason": "written"}
SERVED = {"name": "trader_session_served", "value": 1.0, "status": "OK"}


def _denied_store(run: dict[str, Any]) -> alert.Store:
    """The trader role as measured: run.json readable, the shortfall denied."""

    class _Denied(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            if "execution_shortfall" in Key:
                raise _ClientError("AccessDenied")
            return super().get_object(Bucket=Bucket, Key=Key)

    objects: dict[str, tuple[dt.datetime, Any]] = {
        f"crucible/runs/trader.session/{DAY}/run.json": (NOW, run),
        f"crucible/trader/execution_shortfall/{DAY}.json": (NOW, SHORTFALL),
    }
    return alert.Store(_Denied(objects), "bucket", "crucible")


def test_fills_the_run_wrote_but_cannot_be_read_are_never_no_orders() -> None:
    run = {
        "status": "ok",
        "metrics": [SERVED, BOOK_FAIL],
        "outputs": [{"key": "trader/evidence.json"}, {"key": f"trader/execution_shortfall/{DAY}.json"}],
    }
    rec = _Recorder()
    outcome = alert.run_alerts(_denied_store(run), now=NOW, trade=rec.trade, status=rec.status)
    # Loud in Telegram AND non-zero, so OnFailure= pages too.
    assert rec.trades == [] and outcome == alert.Outcome(1, True, fills_unreadable=True) and outcome.failed
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: ORDERS PLACED — fill details unreadable (AccessDenied)*")
    assert "no orders" not in summary
    assert "no per-fill alert was sent" in summary


def test_the_money_path_link_also_counts_as_the_run_writing_the_shortfall() -> None:
    run = {"status": "ok", "money_path_link": {"money_path_writes": [f"trader/execution_shortfall/{DAY}.json"]}}
    rec = _Recorder()
    alert.run_alerts(_denied_store(run), now=NOW, trade=rec.trade, status=rec.status)
    assert "fill details unreadable" in rec.statuses[0]


def test_a_listed_shortfall_that_is_missing_is_unreadable_not_no_orders() -> None:
    run = {"status": "ok", "outputs": [{"key": f"trader/execution_shortfall/{DAY}.json"}]}
    rec = _Recorder()
    alert.run_alerts(_store(run, None), now=NOW, trade=rec.trade, status=rec.status)
    assert "fill details unreadable (listed in the run's outputs but not found)" in rec.statuses[0]


def test_another_store_error_on_the_shortfall_is_unreadable_not_a_crash() -> None:
    class _Slow(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            if "execution_shortfall" in Key:
                raise _ClientError("SlowDown")
            return super().get_object(Bucket=Bucket, Key=Key)

    objects = {f"crucible/runs/trader.session/{DAY}/run.json": (NOW, {"status": "ok"})}
    rec = _Recorder()
    alert.run_alerts(alert.Store(_Slow(objects), "b", "crucible"), now=NOW, trade=rec.trade, status=rec.status)
    assert "fill details unreadable (SlowDown)" in rec.statuses[0]


def test_zero_orders_only_when_the_run_lists_no_shortfall() -> None:
    run = {"status": "ok", "metrics": [SERVED, BOOK_OK], "outputs": [{"key": "trader/evidence.json"}]}
    rec = _Recorder()
    assert not alert.run_alerts(_denied_store(run), now=NOW, trade=rec.trade, status=rec.status).failed
    assert rec.statuses == [f"*v2 session {DAY}: no orders*\nthe book needed no orders\nOrder book: written"]


def test_the_order_book_is_reported_and_a_failure_is_loud_without_the_arn() -> None:
    rec = _Recorder()
    alert.run_alerts(_store({"status": "ok", "metrics": [BOOK_OK]}), now=NOW, trade=rec.trade, status=rec.status)
    assert rec.statuses[0].endswith("\nOrder book: written")
    alert.run_alerts(_store({"status": "ok", "metrics": [BOOK_FAIL]}), now=NOW, trade=rec.trade, status=rec.status)
    last = rec.statuses[1].splitlines()[-1]
    assert last.startswith("*ORDER BOOK FAILED*: order book not written: AccessDenied")
    assert "arn:" not in last and "ACCOUNT" not in last
    refused = {"status": "failed", "reason": "x", "metrics": [BOOK_FAIL]}
    alert.run_alerts(_store(refused), now=NOW, trade=rec.trade, status=rec.status)
    assert "*ORDER BOOK FAILED*" in rec.statuses[2]
    held = {**HELD_RUN, "metrics": [*HELD_RUN["metrics"], BOOK_FAIL]}
    alert.run_alerts(_store(held, None), now=NOW, trade=rec.trade, status=rec.status)
    assert "*ORDER BOOK FAILED*" in rec.statuses[3]


def test_a_long_reason_is_cut() -> None:
    assert alert._short("x" * 500).endswith("…") and len(alert._short("x" * 500)) == 160
    assert alert._short(None) == "none recorded"


def test_an_unreadable_run_record_is_reported_and_fails() -> None:
    class _Denied(_S3):
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            raise _ClientError("AccessDenied")

    objects = {f"crucible/runs/trader.session/{DAY}/run.json": (NOW, {"status": "ok"})}
    rec = _Recorder()
    outcome = alert.run_alerts(
        alert.Store(_Denied(objects), "b", "crucible"), now=NOW, trade=rec.trade, status=rec.status
    )
    assert outcome == alert.Outcome(1, True, run_unreadable=True) and outcome.failed
    assert rec.statuses[0].startswith(f"*v2 session {DAY}: RUN RECORD UNREADABLE (AccessDenied)*")
    assert rec.trades == []


FILED = [{"key": f"trader/execution_shortfall/{DAY}.json"}]


def test_a_run_that_failed_mid_day_after_orders_is_never_no_orders() -> None:
    """The all-day session: sells at 09:30, then a dropped gateway at 13:00."""
    shortfall = {**SHORTFALL, "orders": [_order("AAA", "sell", 10), _order("BBB", "buy", 20, 0)]}
    run = {"status": "failed", "reason": "RoutingRefusedError: the gateway dropped", "outputs": FILED}
    rec = _Recorder()
    alert.run_alerts(_store(run, shortfall), now=NOW, trade=rec.trade, status=rec.status)
    assert [t[:3] for t in rec.trades] == [("SELL", "AAA", 10)]  # the fill is still alerted
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: RUN FAILED after 2 order(s) placed, 1 filled*")
    assert "NO ORDERS PLACED" not in summary
    assert "the gateway dropped" in summary and "UNFILLED: BBB" in summary
    assert "Sells $1,005 · Buys $0" in summary and "Orders already at the broker stand" in summary


def test_a_failed_run_whose_filed_orders_cannot_be_read_says_orders_were_placed() -> None:
    run = {"status": "failed", "reason": "ProtectiveStopError: AAA", "outputs": FILED}
    rec = _Recorder()
    outcome = alert.run_alerts(_denied_store(run), now=NOW, trade=rec.trade, status=rec.status)
    assert outcome.fills_unreadable and outcome.failed
    (summary,) = rec.statuses
    assert summary.startswith(f"*v2 session {DAY}: RUN FAILED after ORDERS WERE PLACED")
    assert "NO ORDERS PLACED" not in summary and "ProtectiveStopError" in summary


def test_a_failed_run_that_filed_an_empty_order_list_is_no_orders() -> None:
    run = {"status": "failed", "reason": "x", "outputs": FILED}
    rec = _Recorder()
    alert.run_alerts(_store(run, {"orders": []}), now=NOW, trade=rec.trade, status=rec.status)
    assert rec.statuses[0].startswith(f"*v2 session {DAY}: NO ORDERS PLACED*")


def test_a_missing_run_record_never_claims_no_order_was_placed() -> None:
    rec = _Recorder()
    alert.run_alerts(alert.Store(_S3({}), "bucket", ""), now=NOW, trade=rec.trade, status=rec.status)
    assert "no order was placed" not in rec.statuses[0] and "check the broker" in rec.statuses[0]


# -- the realized gain or loss on a sale (Brian 2026-11-13) ----------------------------


def test_a_sale_carries_its_realized_gain_against_the_broker_cost() -> None:
    book = {
        "rows": [
            {"ticker": "AAA", "avg_cost_usd": 80.0},
            {"ticker": "BBB", "avg_cost_usd": None},
            {"ticker": "$CASH", "avg_cost_usd": True},
        ]
    }
    rec = _Recorder()
    alert.run_alerts(_store({"status": "ok"}, book=book), now=NOW, trade=rec.trade, status=rec.status)
    sell, buy = rec.trades
    # 10 sold at 100.50 against an 80.00 cost: +$205.00, +25.6%.
    assert sell[4]["realized_pnl"] == 205.0
    assert sell[4]["realized_return_pct"] == pytest.approx(25.625)
    # A buy realizes nothing.
    assert buy[4]["realized_pnl"] is None and buy[4]["realized_return_pct"] is None


def test_without_a_book_cost_the_sale_is_still_alerted_without_a_realized_line() -> None:
    rec = _Recorder()
    alert.run_alerts(_store({"status": "ok"}), now=NOW, trade=rec.trade, status=rec.status)
    assert rec.trades[0][4]["realized_pnl"] is None


@pytest.mark.parametrize(
    ("price", "shares", "cost", "expected"),
    [
        (90.0, 10, 100.0, (-100.0, -10.0)),
        (100.0, 10, None, None),
        (100.0, 10, 0.0, None),
        (0.0, 10, 100.0, None),
        (100.0, 0, 100.0, None),
    ],
)
def test_realized_on_sale(price: float, shares: int, cost: float | None, expected: Any) -> None:
    got = alert.realized_on_sale(price, shares, cost)
    if expected is None:
        assert got is None
    else:
        assert got == (expected[0], pytest.approx(expected[1]))
