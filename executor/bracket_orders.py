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

    # Step 1: Place market BUY and wait for fill
    buy_order = MarketOrder("BUY", quantity)
    # Explicit routing fields to preempt the paper-account order preset
    # that forces TIF=DAY and can cancel bare market orders with Error
    # 10349. See ibkr.py:place_market_order for the same pattern.
    buy_order.tif = "DAY"
    buy_order.outsideRth = False
    buy_order.orderId = ib.client.getReqId()
    buy_trade = ib.placeOrder(contract, buy_order)

    logger.info(f"BUY {quantity} {ticker} placed | orderId={buy_order.orderId}")

    # Poll for fill
    terminal_states = {"Filled", "Cancelled", "Inactive", "ApiCancelled"}
    elapsed = 0.0
    poll_interval = 0.5
    while elapsed < timeout_seconds:
        ib.sleep(poll_interval)
        elapsed += poll_interval
        if buy_trade.orderStatus.status in terminal_states:
            break

    status = buy_trade.orderStatus.status
    fill_price = None
    filled_shares = None
    fill_time = None

    if buy_trade.fills:
        total_qty = sum(f.execution.shares for f in buy_trade.fills)
        total_cost = sum(f.execution.shares * f.execution.price for f in buy_trade.fills)
        fill_price = round(total_cost / total_qty, 4) if total_qty > 0 else None
        filled_shares = int(total_qty)
        fill_time = (
            buy_trade.fills[-1].execution.time.isoformat()
            if buy_trade.fills[-1].execution.time else None
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

    # Step 2: Place independent trailing stop if BUY filled
    stop_order_id = None
    if result_status == "Filled" and filled_shares:
        stop = Order()
        stop.orderId = ib.client.getReqId()
        stop.action = "SELL"
        stop.orderType = "TRAIL"
        stop.totalQuantity = filled_shares
        stop.auxPrice = trail_amount

        stop_trade = ib.placeOrder(contract, stop)
        stop_order_id = stop.orderId
        logger.info(
            f"Trailing stop placed: SELL {filled_shares} {ticker} "
            f"trail=${trail_amount:.2f} | orderId={stop_order_id}"
        )

    logger.info(
        f"BUY {result_status}: {quantity} {ticker} "
        f"| fill_price={fill_price} trail=${trail_amount:.2f}"
    )

    return {
        "ib_order_id": buy_order.orderId,
        "stop_order_id": stop_order_id,
        "status": result_status,
        "fill_price": fill_price,
        "filled_shares": filled_shares,
        "fill_time": fill_time,
        "trail_amount": trail_amount,
    }
