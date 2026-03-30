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

    def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to delayed (free) market data for all tickers."""
        self._ib.reqMarketDataType(3)  # 3 = delayed (free)

        for symbol in tickers:
            contract = Stock(symbol, "SMART", "USD")
            try:
                self._ib.qualifyContracts(contract)
            except Exception as e:
                logger.warning("Could not qualify %s — skipping: %s", symbol, e)
                continue

            ticker_data = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
            self._contracts[symbol] = contract
            self._subscriptions.append(ticker_data)
            logger.debug("Subscribed to delayed data for %s", symbol)

        # Register tick handler
        self._ib.pendingTickersEvent += self._on_pending_tickers
        logger.info("Subscribed to %d/%d tickers for delayed streaming", len(self._contracts), len(tickers))

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
            except Exception:
                pass
        self._subscriptions.clear()
        self._contracts.clear()
        logger.info("Unsubscribed from all market data")

    def get_price(self, ticker: str) -> dict | None:
        """Return current price state for a ticker, or None."""
        return self.prices.get(ticker)


def _finite(val) -> float | None:
    """Return val if it's a finite positive number, else None."""
    if val is not None and isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
        return float(val)
    return None
