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

from executor.ibkr import fill_commission_usd

logger = logging.getLogger(__name__)


def place_bracket_with_stop(
    ib_client,
    ticker: str,
    quantity: int,
    atr_value: float,
    atr_multiple: float = 2.0,
    timeout_seconds: float = 30.0,
    stop_pct: float | None = None,
) -> dict:
    """
    Place a market BUY order with a protective stop child order.

    Two child shapes:
      * ``stop_pct`` None (legacy alpha book): a TRAIL of ``atr_value *
        atr_multiple`` dollars.
      * ``stop_pct`` set (optimizer owns the book): a plain STP at
        ``fill_price * (1 - stop_pct)``. Under the optimizer, alpha stops are
        retired; the broker child is only a hard-risk backstop for when the
        daemon is not running, set wider than the daemon's catastrophic gap
        stop so the daemon (which logs the exit and feeds the re-entry
        lockout) acts first.

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
            "commission_usd": None,
        }

    trail_amount = round(atr_value * atr_multiple, 2)
    if stop_pct is None and trail_amount <= 0:
        logger.warning(
            f"Trail amount <= 0 for {ticker} (ATR={atr_value}, mult={atr_multiple}) — placing plain market order"
        )
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
    commission_usd = None

    if buy_trade.fills:
        total_qty = sum(f.execution.shares for f in buy_trade.fills)
        total_cost = sum(f.execution.shares * f.execution.price for f in buy_trade.fills)
        fill_price = round(total_cost / total_qty, 4) if total_qty > 0 else None
        filled_shares = int(total_qty)
        fill_time = buy_trade.fills[-1].execution.time.isoformat() if buy_trade.fills[-1].execution.time else None
        commission_usd = fill_commission_usd(buy_trade.fills)

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

    # Step 2: Place independent protective stop if BUY filled
    stop_order_id = None
    stop_price = None
    if stop_pct is not None:
        trail_amount = None
        if fill_price:
            stop_price = round(fill_price * (1.0 - float(stop_pct)), 2)
    if result_status == "Filled" and filled_shares and (stop_pct is None or stop_price):
        stop = Order()
        stop.orderId = ib.client.getReqId()
        stop.action = "SELL"
        stop.totalQuantity = filled_shares
        if stop_price is not None:
            stop.orderType = "STP"
            stop.auxPrice = stop_price
        else:
            stop.orderType = "TRAIL"
            stop.auxPrice = trail_amount
        # Same explicit routing fields as the BUY leg above and
        # ibkr.py:place_market_order — without them the paper-account order
        # preset reinterprets the order and cancels it with Error 10349
        # ("Order TIF was set to DAY based on order preset") before
        # re-submitting. Observed live 2026-07-27 16:17:21Z on the AMD
        # trailing stop: Cancelled → PreSubmitted in the same millisecond.
        # The stop survived that round trip, but a cancel is a cancel — the
        # protective stop is briefly absent, and the daemon's own order-state
        # tracking sees a Cancelled it did not ask for.
        #
        # NOT a behavior change: the preset was already forcing TIF=DAY, so
        # the stop's effective lifetime is unchanged. Whether a protective
        # trailing stop SHOULD be GTC rather than DAY is a separate strategy
        # question, deliberately out of scope here.
        stop.tif = "DAY"
        stop.outsideRth = False
        stop.transmit = True

        ib.placeOrder(contract, stop)
        stop_order_id = stop.orderId
        if stop_price is not None:
            logger.info(
                f"Backstop stop placed: SELL {filled_shares} {ticker} stop=${stop_price:.2f} "
                f"({float(stop_pct):.0%} below fill) | orderId={stop_order_id}"
            )
        else:
            logger.info(
                f"Trailing stop placed: SELL {filled_shares} {ticker} trail=${trail_amount:.2f} | orderId={stop_order_id}"
            )

    logger.info(
        f"BUY {result_status}: {quantity} {ticker} | fill_price={fill_price} "
        f"trail={trail_amount} stop={stop_price}"
    )

    return {
        "ib_order_id": buy_order.orderId,
        "stop_order_id": stop_order_id,
        "status": result_status,
        "fill_price": fill_price,
        "filled_shares": filled_shares,
        "fill_time": fill_time,
        "trail_amount": trail_amount,
        "stop_price": stop_price,
        "commission_usd": commission_usd,
    }
