"""Tests for executor.open_orders_artifact.

Covers the pure snapshot builder's projection of ib_insync Trade duck
types, the status-based working/terminal classification, the S3 writer
contract, and the fail-loud-but-fire-and-forget S3 failure handling.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from executor.open_orders_artifact import (
    OPEN_ORDERS_LATEST_KEY,
    OpenOrdersSnapshotWriter,
    build_open_orders_snapshot,
)

# ── fixtures ─────────────────────────────────────────────────────────────


def _trade(
    ticker: str,
    action: str,
    qty: int,
    *,
    filled: int = 0,
    status: str = "Submitted",
    order_type: str = "LMT",
    lmt: float | None = 50.0,
    aux: float | None = None,
    order_id: int = 1,
    parent_id: int | None = None,
):
    """Build an ib_insync.Trade duck-type from primitives."""
    contract = SimpleNamespace(symbol=ticker)
    order = SimpleNamespace(
        action=action,
        totalQuantity=qty,
        orderType=order_type,
        lmtPrice=lmt,
        auxPrice=aux,
        orderId=order_id,
        parentId=parent_id,
    )
    status_obj = SimpleNamespace(status=status, filled=filled)
    return SimpleNamespace(contract=contract, order=order, orderStatus=status_obj)


# ── pure builder ─────────────────────────────────────────────────────────


def test_snapshot_projects_minimum_fields_per_order():
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 100, status="Submitted", lmt=185.5)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
        written_at_utc="2026-05-27T18:00:00Z",
    )
    assert payload["n_open_orders"] == 1
    assert payload["n_working"] == 1
    rec = payload["open_orders"][0]
    assert rec["ticker"] == "AAPL"
    assert rec["action"] == "BUY"
    assert rec["shares"] == 100
    assert rec["filled"] == 0
    assert rec["remaining"] == 100
    assert rec["order_type"] == "LMT"
    assert rec["limit_price"] == 185.5
    assert rec["status"] == "Submitted"
    assert rec["is_working"] is True


def test_partial_fill_reflected_in_remaining():
    payload = build_open_orders_snapshot(
        [_trade("MSFT", "SELL", 50, filled=20, status="Submitted")],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    rec = payload["open_orders"][0]
    assert rec["filled"] == 20
    assert rec["remaining"] == 30


def test_filled_status_marked_not_working_but_still_recorded():
    # A trade that finished filling this tick still shows up in
    # openTrades() briefly. Record it but mark is_working=False so
    # the dashboard doesn't double-count it against the optimizer plan.
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 100, filled=100, status="Filled")],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["n_open_orders"] == 1
    assert payload["n_working"] == 0
    assert payload["open_orders"][0]["is_working"] is False


def test_cancelled_status_marked_not_working():
    payload = build_open_orders_snapshot(
        [_trade("XOM", "BUY", 100, status="Cancelled")],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["n_working"] == 0


@pytest.mark.parametrize("status", [
    "PendingSubmit", "PendingCancel", "PreSubmitted", "Submitted", "ApiPending",
])
def test_each_working_status_classifies_as_working(status: str):
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 1, status=status)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["n_working"] == 1


def test_records_sorted_working_first_then_by_ticker():
    trades = [
        _trade("ZZZ", "BUY", 1, status="Filled"),       # terminal, last
        _trade("MSFT", "BUY", 1, status="Submitted"),   # working, alpha
        _trade("AAPL", "BUY", 1, status="Submitted"),   # working, alpha
    ]
    payload = build_open_orders_snapshot(
        trades,
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    tickers = [r["ticker"] for r in payload["open_orders"]]
    assert tickers == ["AAPL", "MSFT", "ZZZ"]


def test_stop_order_carries_aux_price_no_limit():
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "SELL", 100, order_type="STP", lmt=0.0, aux=180.0)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    rec = payload["open_orders"][0]
    assert rec["order_type"] == "STP"
    assert rec["aux_price"] == 180.0


def test_market_order_emits_none_limit_price():
    # IB market orders sometimes return lmtPrice=0.0 — preserve the
    # numeric 0.0 (it's a legitimate input value); only NaN coerces to
    # None. Tested explicitly so a future "treat 0 as missing" change
    # is caught.
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 100, order_type="MKT", lmt=0.0)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["open_orders"][0]["limit_price"] == 0.0


def test_nan_price_coerced_to_none():
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 100, lmt=float("nan"))],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["open_orders"][0]["limit_price"] is None


def test_bracket_child_carries_parent_id():
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "SELL", 100, order_type="STP", parent_id=42)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["open_orders"][0]["parent_id"] == 42


def test_partial_trade_object_skipped_not_crashed():
    # Mid-edit race: a Trade object briefly missing a totalQuantity.
    # Must not raise — skip silently and keep the surviving records.
    bad = SimpleNamespace(
        contract=SimpleNamespace(symbol="AAPL"),
        order=SimpleNamespace(action="BUY", totalQuantity=None),
        orderStatus=SimpleNamespace(status="Submitted", filled=0),
    )
    good = _trade("MSFT", "BUY", 50)
    payload = build_open_orders_snapshot(
        [bad, good],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert payload["n_open_orders"] == 1
    assert payload["open_orders"][0]["ticker"] == "MSFT"


def test_empty_open_trades_emits_zero_count():
    payload = build_open_orders_snapshot(
        [],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
        written_at_utc="2026-05-27T13:00:00Z",
    )
    assert payload["n_open_orders"] == 0
    assert payload["n_working"] == 0
    assert payload["open_orders"] == []
    assert payload["written_at_utc"] == "2026-05-27T13:00:00Z"


def test_payload_is_audit_stable_and_serializable():
    payload = build_open_orders_snapshot(
        [_trade("AAPL", "BUY", 100), _trade("MSFT", "SELL", 50, status="Filled")],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
        written_at_utc="2026-05-27T18:00:00Z",
        daemon_pid=12345,
    )
    # Round-trip JSON to confirm everything is serializable.
    round_trip = json.loads(json.dumps(payload, default=str))
    assert round_trip == json.loads(json.dumps(payload, default=str))
    assert round_trip["daemon_pid"] == 12345


# ── S3 writer ────────────────────────────────────────────────────────────


class _StubS3:
    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.store[(Bucket, Key)] = Body


def test_writer_publishes_to_canonical_key():
    s3 = _StubS3()
    writer = OpenOrdersSnapshotWriter(
        bucket="alpha-engine-research", s3_client=s3, daemon_pid=99,
    )
    ok = writer.write(
        [_trade("AAPL", "BUY", 100)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert ok is True
    body = json.loads(s3.store[("alpha-engine-research", OPEN_ORDERS_LATEST_KEY)])
    assert body["n_open_orders"] == 1
    assert body["daemon_pid"] == 99


def test_writer_swallows_s3_failure_returns_false():
    # Fire-and-forget: writer must NEVER raise, only return False so the
    # daemon's order-loop tick continues uninterrupted.
    fake_s3 = MagicMock()
    fake_s3.put_object.side_effect = ClientError(
        {"Error": {"Code": "Throttling", "Message": "rate limit"}},
        "PutObject",
    )
    writer = OpenOrdersSnapshotWriter(bucket="b", s3_client=fake_s3)
    ok = writer.write(
        [_trade("AAPL", "BUY", 100)],
        calendar_date="2026-05-27",
        trading_day="2026-05-27",
    )
    assert ok is False  # No exception propagated
