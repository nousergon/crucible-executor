"""
Intraday price monitor — subscribes to 15-minute delayed streaming data via ib_insync.

Maintains a price state dict for all subscribed tickers, updated on each tick event.
The daemon reads this state to evaluate entry triggers and exit rules.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from ib_insync import IB, Stock, Ticker

logger = logging.getLogger(__name__)


class PriceMonitor:
    """Subscribe to delayed market data and maintain live price state."""

    def __init__(self, ib: IB):
        self._ib = ib
        self.prices: dict[str, dict] = {}
        self._contracts: dict[str, Stock] = {}
        self._subscriptions: list[Ticker] = []
        # Per-symbol handle on the reqMktData result so a single symbol can be
        # cancelled precisely on a mid-session resubscribe (config#897), rather
        # than matching by contract identity on the flat list.
        self._sub_by_symbol: dict[str, Ticker] = {}

    def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to delayed (free) market data for all tickers."""
        self._ib.reqMarketDataType(3)  # 3 = delayed (free)

        for symbol in tickers:
            self._subscribe_one(symbol)

        # Register tick handler
        self._ib.pendingTickersEvent += self._on_pending_tickers
        logger.info("Subscribed to %d/%d tickers for delayed streaming", len(self._contracts), len(tickers))

    def _subscribe_one(self, symbol: str) -> bool:
        """Request delayed market data for a single symbol.

        Returns ``True`` if the subscription was established, ``False`` if the
        contract could not be qualified (already-subscribed symbols are a no-op
        and return ``True``). Does NOT register the tick handler — callers do
        that once via :meth:`subscribe` / :meth:`resubscribe`.
        """
        if symbol in self._contracts:
            return True
        contract = Stock(symbol, "SMART", "USD")
        try:
            self._ib.qualifyContracts(contract)
        except Exception as e:
            logger.warning("Could not qualify %s — skipping: %s", symbol, e)
            return False

        ticker_data = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
        self._contracts[symbol] = contract
        self._subscriptions.append(ticker_data)
        self._sub_by_symbol[symbol] = ticker_data
        logger.debug("Subscribed to delayed data for %s", symbol)
        return True

    def subscribed_tickers(self) -> set[str]:
        """Return the set of symbols currently subscribed to market data."""
        return set(self._contracts)

    def resubscribe(self, tickers: list[str]) -> tuple[set[str], set[str]]:
        """Reconcile live subscriptions to ``tickers`` via a minimal diff.

        Subscribes only symbols newly present and cancels only symbols newly
        absent — the shared set is left untouched so an unchanged universe
        produces zero IB churn. Returns ``(added, removed)`` (the symbols for
        which subscribe/cancel was actually attempted) for logging/telemetry.
        """
        desired = set(tickers)
        current = set(self._contracts)
        added = desired - current
        removed = current - desired

        if not added and not removed:
            return set(), set()

        # Ensure the delayed data type is set even if the initial subscribe()
        # happened on a prior (dropped) connection.
        if added:
            self._ib.reqMarketDataType(3)
        added_ok: set[str] = set()
        for symbol in sorted(added):
            if self._subscribe_one(symbol):
                added_ok.add(symbol)

        for symbol in sorted(removed):
            self._cancel_one(symbol)

        logger.info(
            "resubscribe: +%d -%d tickers (now %d subscribed)",
            len(added_ok), len(removed), len(self._contracts),
        )
        return added_ok, set(removed)

    def _cancel_one(self, symbol: str) -> None:
        """Cancel market data for a single symbol and drop its bookkeeping."""
        contract = self._contracts.pop(symbol, None)
        ticker_data = self._sub_by_symbol.pop(symbol, None)
        if contract is None:
            return
        try:
            self._ib.cancelMktData(contract)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not cancel market data for %s: %s", symbol, e)
        if ticker_data is not None:
            self._subscriptions = [t for t in self._subscriptions if t is not ticker_data]
        logger.debug("Cancelled delayed data for %s", symbol)

    def _on_pending_tickers(self, tickers: set[Ticker]) -> None:
        """Callback fired by ib_insync when ticker data updates."""
        for ticker in tickers:
            symbol = ticker.contract.symbol if ticker.contract else None
            if not symbol:
                continue

            # Extract prices — prefer delayed fields, fall back to live
            last = _finite(ticker.last) or _finite(getattr(ticker, "delayedLast", None))
            high = _finite(ticker.high) or _finite(getattr(ticker, "delayedHigh", None))
            low = _finite(ticker.low) or _finite(getattr(ticker, "delayedLow", None))
            close = _finite(ticker.close) or _finite(getattr(ticker, "delayedClose", None))
            volume = ticker.volume if ticker.volume and ticker.volume > 0 else None

            if not last and not close:
                continue  # no usable price

            price = last or close
            prev = self.prices.get(symbol, {})

            self.prices[symbol] = {
                "last": price,
                "high": max(high or price, prev.get("high", price)),  # track intraday high
                "low": min(low or price, prev.get("low", price)),     # track intraday low
                "close": close,
                "volume": volume,
                "updated_at": datetime.now().isoformat(),
            }

    def unsubscribe_all(self) -> None:
        """Cancel all market data subscriptions."""
        self._ib.pendingTickersEvent -= self._on_pending_tickers
        for ticker_data in self._subscriptions:
            try:
                self._ib.cancelMktData(ticker_data.contract)
            except Exception as e:
                logger.debug("cancelMktData failed for %s (non-fatal): %s", ticker_data.contract, e)
        self._subscriptions.clear()
        self._contracts.clear()
        self._sub_by_symbol.clear()
        logger.info("Unsubscribed from all market data")

    def get_price(self, ticker: str) -> dict | None:
        """Return current price state for a ticker, or None."""
        return self.prices.get(ticker)


def _finite(val) -> float | None:
    """Return val if it's a finite positive number, else None."""
    if val is not None and isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
        return float(val)
    return None
