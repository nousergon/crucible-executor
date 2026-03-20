"""
Bracket order placement — BUY + trailing stop as parent/child pair.

Every ENTER order gets a broker-side trailing stop so positions are protected
even if the intraday daemon isn't running. IB Gateway enforces the stop
server-side after the parent fill.

Uses ib_insync bracket order pattern: parent.transmit = False, child.transmit = True.
"""

from __future__ import annotations

import logging

from ib_insync import MarketOrder, Order, Stock

logger = logging.getLogger(__name__)


def place_bracket_with_stop(
    ib_client,
    ticker: str,
    quantity: int,
    atr_value: float,
    atr_multiple: float = 2.0,
    timeout_seconds: float = 30.0,
) -> dict:
    """
    Place a market BUY order with a trailing stop child order.

    The trailing stop amount = atr_value * atr_multiple, in dollars.
    IB Gateway activates the child stop after the parent fills.

    Args:
        ib_client: IBKRClient instance (needs .ib attribute)
        ticker: stock symbol
        quantity: number of shares
        atr_value: ATR in dollar terms (e.g., $5.75)
        atr_multiple: multiplier for trail distance (default 2.0)
        timeout_seconds: max seconds to wait for parent fill

    Returns:
        dict with keys: ib_order_id, stop_order_id, status, fill_price,
        filled_shares, fill_time, trail_amount
    """
    ib = ib_client.ib
    ib_client.ensure_connected()

    contract = Stock(ticker, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        logger.error(f"Could not qualify contract for {ticker}: {e}")
        return {
            "ib_order_id": None,
            "stop_order_id": None,
            "status": "Rejected",
            "fill_price": None,
            "filled_shares": None,
            "fill_time": None,
            "trail_amount": None,
        }

    trail_amount = round(atr_value * atr_multiple, 2)
    if trail_amount <= 0:
        logger.warning(f"Trail amount <= 0 for {ticker} (ATR={atr_value}, mult={atr_multiple}) — placing plain market order")
        result = ib_client.place_market_order(ticker, "BUY", quantity, timeout_seconds)
        return {**result, "stop_order_id": None, "trail_amount": None}

    # Parent: market BUY — don't transmit until child is attached
    parent = MarketOrder("BUY", quantity)
    parent.orderId = ib.client.getReqId()
    parent.transmit = False

    # Child: trailing stop SELL — transmit both when this is placed
    stop = Order()
    stop.orderId = ib.client.getReqId()
    stop.action = "SELL"
    stop.orderType = "TRAIL"
    stop.totalQuantity = quantity
    stop.auxPrice = trail_amount  # trailing amount in dollars
    stop.parentId = parent.orderId
    stop.transmit = True

    parent_trade = ib.placeOrder(contract, parent)
    stop_trade = ib.placeOrder(contract, stop)

    logger.info(
        f"Bracket order placed: BUY {quantity} {ticker} "
        f"+ trailing stop (trail=${trail_amount:.2f}) "
        f"| parent={parent.orderId} stop={stop.orderId}"
    )

    # Poll for parent fill
    terminal_states = {"Filled", "Cancelled", "Inactive", "ApiCancelled"}
    elapsed = 0.0
    poll_interval = 0.5
    while elapsed < timeout_seconds:
        ib.sleep(poll_interval)
        elapsed += poll_interval
        if parent_trade.orderStatus.status in terminal_states:
            break

    status = parent_trade.orderStatus.status
    fill_price = None
    filled_shares = None
    fill_time = None

    if parent_trade.fills:
        total_qty = sum(f.execution.shares for f in parent_trade.fills)
        total_cost = sum(f.execution.shares * f.execution.price for f in parent_trade.fills)
        fill_price = round(total_cost / total_qty, 4) if total_qty > 0 else None
        filled_shares = int(total_qty)
        fill_time = (
            parent_trade.fills[-1].execution.time.isoformat()
            if parent_trade.fills[-1].execution.time else None
        )

    # Normalize status
    if status == "Filled":
        result_status = "Filled"
    elif status in ("Cancelled", "Inactive", "ApiCancelled"):
        result_status = "Rejected"
    elif filled_shares and filled_shares < quantity:
        result_status = "PartialFill"
    elif status not in terminal_states:
        result_status = "Timeout"
    else:
        result_status = status

    logger.info(
        f"Bracket parent {result_status}: BUY {quantity} {ticker} "
        f"| fill_price={fill_price} trail=${trail_amount:.2f} "
        f"| stop_status={stop_trade.orderStatus.status}"
    )

    return {
        "ib_order_id": parent.orderId,
        "stop_order_id": stop.orderId,
        "status": result_status,
        "fill_price": fill_price,
        "filled_shares": filled_shares,
        "fill_time": fill_time,
        "trail_amount": trail_amount,
    }
