"""Tests for executor.bracket_orders.place_bracket_with_stop — market BUY + trailing stop."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from executor.bracket_orders import place_bracket_with_stop

# ── Helpers ─────────────────────────────────────────────────────────────────


class _FakeOrderStatus:
    def __init__(self, status):
        self.status = status


class _FakeExecution:
    def __init__(self, shares, price, time=None):
        self.shares = shares
        self.price = price
        self.time = time


class _FakeFill:
    def __init__(self, shares, price, time=None):
        self.execution = _FakeExecution(shares, price, time)


def _make_ib(buy_status="Filled", fills=None, stop_status="Submitted"):
    """Build a MagicMock IB that returns deterministic trade results.

    - placeOrder returns a fake trade with .orderStatus.status + .fills
    - ib.sleep is a no-op
    - ib.client.getReqId increments
    """
    ib = MagicMock()
    counter = {"id": 1000}

    def get_req_id():
        counter["id"] += 1
        return counter["id"]

    ib.client.getReqId.side_effect = get_req_id

    trades_returned = []

    def place_order(contract, order):
        trade = MagicMock()
        if "TRAIL" in getattr(order, "orderType", ""):
            trade.orderStatus = _FakeOrderStatus(stop_status)
            trade.fills = []
        else:
            trade.orderStatus = _FakeOrderStatus(buy_status)
            trade.fills = fills or []
        trades_returned.append(trade)
        return trade

    ib.placeOrder.side_effect = place_order
    ib.placeOrder.trades = trades_returned  # for test introspection
    ib.sleep = MagicMock()
    return ib


def _ib_client(ib, qualify_raises=False):
    client = MagicMock()
    client.ib = ib
    client.ensure_connected = MagicMock()
    if qualify_raises:
        ib.qualifyContracts.side_effect = RuntimeError("contract not found")
    return client


# ── Happy path ──────────────────────────────────────────────────────────────


def test_bracket_with_stop_filled_places_trailing_stop():
    fill_time = datetime(2026, 4, 15, 14, 30, 0)
    ib = _make_ib(
        buy_status="Filled",
        fills=[_FakeFill(50, 100.50, fill_time), _FakeFill(50, 100.60, fill_time)],
    )
    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=2.50,
        atr_multiple=2.0,
        timeout_seconds=10,
    )

    assert result["status"] == "Filled"
    assert result["filled_shares"] == 100
    assert result["fill_price"] == pytest.approx(100.55, abs=0.001)
    assert result["trail_amount"] == 5.00
    assert result["stop_order_id"] is not None
    assert result["ib_order_id"] is not None
    # Both BUY and trailing stop placed (2 placeOrder calls)
    assert ib.placeOrder.call_count == 2


def test_bracket_zero_trail_falls_back_to_market_order():
    ib = _make_ib()
    client = _ib_client(ib)
    client.place_market_order = MagicMock(
        return_value={
            "ib_order_id": 99,
            "status": "Filled",
            "fill_price": 100.0,
            "filled_shares": 50,
            "fill_time": "2026-04-15T14:30:00",
        }
    )

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=50,
        atr_value=0.0,
        atr_multiple=2.0,
    )

    assert result["stop_order_id"] is None
    assert result["trail_amount"] is None
    assert result["status"] == "Filled"
    client.place_market_order.assert_called_once_with("AAPL", "BUY", 50, 30.0)


def test_bracket_qualify_failure_returns_rejected():
    ib = _make_ib()
    client = _ib_client(ib, qualify_raises=True)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="XYZ",
        quantity=100,
        atr_value=1.0,
    )

    assert result["status"] == "Rejected"
    assert result["ib_order_id"] is None
    assert result["stop_order_id"] is None
    assert result["fill_price"] is None


def test_bracket_cancelled_status_normalized_to_rejected():
    ib = _make_ib(buy_status="Cancelled", fills=[])
    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=2.0,
    )

    assert result["status"] == "Rejected"
    # No stop placed on rejected BUY
    assert result["stop_order_id"] is None
    assert ib.placeOrder.call_count == 1


def test_bracket_partial_fill_returned_as_partial():
    """Filled-shares < quantity AND non-terminal status → PartialFill normalization."""
    ib = MagicMock()
    counter = {"id": 1000}
    ib.client.getReqId.side_effect = lambda: counter.update(id=counter["id"] + 1) or counter["id"]
    fill_time = datetime(2026, 4, 15, 14, 30, 0)

    # Build a trade that says PartiallyFilled but with only 50/100 shares
    partial_trade = MagicMock()
    partial_trade.orderStatus = _FakeOrderStatus("PartiallyFilled")
    partial_trade.fills = [_FakeFill(50, 100.0, fill_time)]
    ib.placeOrder.return_value = partial_trade
    ib.sleep = MagicMock()

    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=2.0,
        timeout_seconds=1.5,
    )

    assert result["status"] == "PartialFill"
    assert result["filled_shares"] == 50
    # No stop placed on partial — only "Filled" branch places stop
    assert result["stop_order_id"] is None


def test_bracket_timeout_returns_timeout_status():
    """Status remains non-terminal (e.g. 'Submitted') after timeout → 'Timeout'."""
    ib = MagicMock()
    counter = {"id": 1000}
    ib.client.getReqId.side_effect = lambda: counter.update(id=counter["id"] + 1) or counter["id"]

    pending_trade = MagicMock()
    pending_trade.orderStatus = _FakeOrderStatus("Submitted")
    pending_trade.fills = []
    ib.placeOrder.return_value = pending_trade
    ib.sleep = MagicMock()

    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=2.0,
        timeout_seconds=2.0,
    )

    assert result["status"] == "Timeout"
    assert result["filled_shares"] is None
    assert result["stop_order_id"] is None


def test_bracket_fill_price_rounded_to_4dp():
    fill_time = datetime(2026, 4, 15, 14, 30, 0)
    ib = _make_ib(
        buy_status="Filled",
        fills=[_FakeFill(1, 100.123456789, fill_time)],
    )
    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=1,
        atr_value=1.0,
        atr_multiple=2.0,
    )

    assert result["fill_price"] == 100.1235  # rounded to 4dp
    assert result["filled_shares"] == 1
    assert result["trail_amount"] == 2.00


def test_bracket_no_fills_when_status_filled_has_zero_total_qty():
    """If fills list has zero total_qty, fill_price stays None."""
    fill_time = datetime(2026, 4, 15, 14, 30, 0)
    ib = _make_ib(
        buy_status="Filled",
        fills=[_FakeFill(0, 100.0, fill_time)],  # 0 shares
    )
    client = _ib_client(ib)

    result = place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=1.0,
    )

    assert result["fill_price"] is None
    # filled_shares = int(0) = 0 → falsy → no stop placed
    assert result["stop_order_id"] is None


def test_trailing_stop_carries_explicit_routing_fields():
    """The TRAIL stop must set tif/outsideRth/transmit like every other
    order this repo places.

    Regression for 2026-07-27 16:17:21Z: the AMD trailing stop went
    Cancelled -> PreSubmitted in the same millisecond with Error 10349
    ("Order TIF was set to DAY based on order preset"). The paper-account
    preset reinterprets an order that leaves these fields unset. The BUY leg
    and ibkr.py:place_market_order already set them; the stop was the one
    placement site that did not, leaving the protective stop momentarily
    cancelled and surfacing a Cancelled the daemon never requested.
    """
    fill_time = datetime(2026, 4, 15, 14, 30, 0)
    ib = _make_ib(buy_status="Filled", fills=[_FakeFill(100, 100.50, fill_time)])
    client = _ib_client(ib)

    place_bracket_with_stop(
        ib_client=client,
        ticker="AAPL",
        quantity=100,
        atr_value=2.50,
        atr_multiple=2.0,
        timeout_seconds=10,
    )

    stop = ib.placeOrder.call_args_list[-1][0][1]
    assert stop.orderType == "TRAIL"
    assert stop.tif == "DAY"
    assert stop.outsideRth is False
    assert stop.transmit is True


# ── Optimizer book: hard-risk backstop instead of an alpha trail ───────────


def test_stop_pct_places_plain_stop_below_fill_not_a_trail():
    ib = _make_ib(buy_status="Filled", fills=[_FakeFill(100, 40.00)])
    result = place_bracket_with_stop(
        ib_client=_ib_client(ib),
        ticker="SMCI",
        quantity=100,
        atr_value=2.37,
        timeout_seconds=10,
        stop_pct=0.20,
    )
    stop_order = ib.placeOrder.call_args_list[1].args[1]
    assert stop_order.orderType == "STP"
    assert stop_order.action == "SELL"
    assert stop_order.totalQuantity == 100
    assert stop_order.auxPrice == 32.00
    assert result["stop_price"] == 32.00
    assert result["trail_amount"] is None
    assert result["stop_order_id"] == stop_order.orderId


def test_stop_pct_works_without_atr():
    ib = _make_ib(buy_status="Filled", fills=[_FakeFill(10, 50.00)])
    result = place_bracket_with_stop(
        ib_client=_ib_client(ib), ticker="NEW", quantity=10, atr_value=0.0,
        timeout_seconds=10, stop_pct=0.20,
    )
    assert ib.placeOrder.call_count == 2
    assert result["stop_price"] == 40.00
