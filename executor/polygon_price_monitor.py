"""Real-time polygon.io WebSocket price monitor (config#913).

Drop-in alternative to ``executor.price_monitor.PriceMonitor`` (which streams
IB's 15-minute *delayed* market data via ib_insync) for entry-trigger precision.
Exposes the exact interface the daemon consumes — ``subscribe(tickers)``,
``get_price(ticker)``, ``.prices`` dict, ``unsubscribe_all()`` — so it can be
swapped in behind a config flag without touching the daemon loop.

Why this exists: the daemon evaluates entry triggers (pullback / VWAP / support /
expiry) and intraday exits against ``monitor.prices``. On 15-min delayed data
those decisions act on stale prices; ``eod_reconcile`` already measures the
resulting ``slippage_by_trigger``. When that monitoring shows delayed pricing is
costing slippage (the issue's self-gate), this real-time feed removes the lag.

The WebSocket transport (connect, auth, reconnect) is isolated from the message
parsing: ``_apply_message`` is a pure function over the price-state dict, so the
tick-handling logic — the part that feeds trigger evaluation — is unit-testable
without a live socket. Requires the ``Stocks`` real-time WebSocket entitlement
on the polygon plan; the free/REST tier does not include it (see
``polygon_client.py``). The API key is read from the same secret as the REST
client (``POLYGON_API_KEY``); never hard-code it.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# polygon real-time stocks WebSocket cluster.
POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

# Aggregate-second ("A") and trade ("T") event fields we consume. Using
# second-aggregates ("A") gives OHLCV per ticker per second, which maps cleanly
# onto the daemon's last/high/low/close/volume price-state shape.
_AGG_EVENT = "A"
_TRADE_EVENT = "T"
_STATUS_EVENT = "status"


def _finite_pos(val) -> float | None:
    """Return val as float if it's a finite positive number, else None."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _apply_message(prices: dict[str, dict], msg: dict) -> str | None:
    """Apply one parsed polygon WS event to the price-state dict (pure).

    Mirrors ``PriceMonitor._on_pending_tickers`` semantics: tracks intraday
    high/low as the running max/min, prefers the aggregate close/last as the
    current price, and stamps ``updated_at``. Returns the updated ticker symbol
    (for tests/telemetry), or ``None`` when the event carried no usable price.

    Supported events:
      * ``A`` (second aggregate): ``sym, o, h, l, c, v``.
      * ``T`` (trade): ``sym, p`` (price), ``s`` (size).
    """
    ev = msg.get("ev")
    symbol = msg.get("sym")
    if not symbol:
        return None

    if ev == _AGG_EVENT:
        last = _finite_pos(msg.get("c")) or _finite_pos(msg.get("o"))
        high = _finite_pos(msg.get("h"))
        low = _finite_pos(msg.get("l"))
        close = _finite_pos(msg.get("c"))
        volume = msg.get("v") if msg.get("v") and msg.get("v") > 0 else None
    elif ev == _TRADE_EVENT:
        last = _finite_pos(msg.get("p"))
        high = low = None
        close = last
        size = msg.get("s")
        volume = size if size and size > 0 else None
    else:
        return None

    if not last and not close:
        return None

    price = last or close
    prev = prices.get(symbol, {})
    prices[symbol] = {
        "last": price,
        "high": max(high or price, prev.get("high", price)),
        "low": min(low or price, prev.get("low", price)),
        "close": close if close is not None else prev.get("close"),
        "volume": volume if volume is not None else prev.get("volume"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return symbol


# Env flag selecting the intraday price source. Default keeps the legacy IB
# 15-min delayed monitor; set EXECUTOR_PRICE_SOURCE=polygon_ws to use the
# real-time WebSocket feed (config#913). An env var (not risk.yaml) so the
# source can flip per-process without a config-repo deploy during the soak.
_PRICE_SOURCE_ENV = "EXECUTOR_PRICE_SOURCE"
_POLYGON_WS = "polygon_ws"


def price_source() -> str:
    """Return the configured intraday price source ('ib_delayed' | 'polygon_ws')."""
    import os

    return os.environ.get(_PRICE_SOURCE_ENV, "ib_delayed").strip().lower()


def make_price_monitor(ib):
    """Construct the configured intraday price monitor (config#913).

    Returns a ``PolygonPriceMonitor`` when ``EXECUTOR_PRICE_SOURCE=polygon_ws``,
    else the legacy IB-delayed ``PriceMonitor``. Both expose the same interface
    (``subscribe`` / ``get_price`` / ``.prices`` / ``unsubscribe_all``) so the
    daemon loop is source-agnostic. If the polygon monitor cannot be built
    (missing key / package) it falls back to the IB monitor rather than failing
    the daemon — the soak must never take the executor down.
    """
    if price_source() == _POLYGON_WS:
        try:
            logger.info("Intraday price source: polygon_ws (config#913)")
            return PolygonPriceMonitor()
        except Exception as e:
            logger.error(
                "PolygonPriceMonitor unavailable (%s) — falling back to IB "
                "delayed monitor", e,
            )
    from executor.price_monitor import PriceMonitor

    return PriceMonitor(ib)


class PolygonPriceMonitor:
    """Real-time price state from polygon.io WebSocket (config#913).

    Interface-compatible with ``executor.price_monitor.PriceMonitor`` so the
    daemon can use either behind a flag.
    """

    def __init__(self, api_key: str | None = None):
        from nousergon_lib.secrets import get_secret

        self._api_key = api_key or get_secret(
            "POLYGON_API_KEY", required=False, default=""
        )
        if not self._api_key:
            raise ValueError("POLYGON_API_KEY not set")
        self.prices: dict[str, dict] = {}
        self._tickers: list[str] = []
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._authed = threading.Event()

    # ── Public interface (mirrors PriceMonitor) ──────────────────────────

    def subscribe(self, tickers: list[str]) -> None:
        """Open the WebSocket and subscribe to second-aggregates for tickers."""
        self._tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="polygon-ws", daemon=True
        )
        self._thread.start()
        logger.info(
            "PolygonPriceMonitor: subscribing to %d tickers via %s",
            len(self._tickers), POLYGON_WS_URL,
        )

    def get_price(self, ticker: str) -> dict | None:
        """Return current price state for a ticker, or None."""
        return self.prices.get(ticker)

    def subscribed_tickers(self) -> set[str]:
        """Return the set of symbols currently subscribed to the feed."""
        return set(self._tickers)

    def resubscribe(self, tickers: list[str]) -> tuple[set[str], set[str]]:
        """Reconcile the subscribed universe to ``tickers`` (config#897).

        Interface-compatible with :meth:`PriceMonitor.resubscribe`. The polygon
        WebSocket subscription is (re)issued from the reader thread on its next
        (re)connect using ``self._tickers``; sending a fresh subscribe frame on
        the live socket is out of scope here, so this updates the target list
        and returns the ``(added, removed)`` diff. Returns empty sets (no-op)
        when the universe is unchanged so callers can skip needless work.
        """
        desired = list(dict.fromkeys(tickers))  # de-dupe, keep order
        current = set(self._tickers)
        added = set(desired) - current
        removed = current - set(desired)
        if not added and not removed:
            return set(), set()
        self._tickers = desired
        logger.info(
            "PolygonPriceMonitor.resubscribe: +%d -%d tickers (now %d)",
            len(added), len(removed), len(self._tickers),
        )
        return added, removed

    def unsubscribe_all(self) -> None:
        """Close the socket and stop the reader thread."""
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._tickers = []
        logger.info("PolygonPriceMonitor: unsubscribed and closed")

    # ── Transport (isolated from parsing) ────────────────────────────────

    def _subscribe_params(self) -> str:
        """Build the polygon subscribe params string (A.<ticker> per name)."""
        return ",".join(f"{_AGG_EVENT}.{t}" for t in self._tickers)

    def _on_message(self, raw: str) -> None:
        """Decode a raw WS frame (a JSON list of events) and apply each."""
        try:
            events = json.loads(raw)
        except (ValueError, TypeError):
            logger.debug("PolygonPriceMonitor: undecodable frame")
            return
        if isinstance(events, dict):
            events = [events]
        for msg in events:
            if not isinstance(msg, dict):
                continue
            if msg.get("ev") == _STATUS_EVENT:
                if msg.get("status") == "auth_success":
                    self._authed.set()
                logger.debug("polygon ws status: %s", msg.get("message"))
                continue
            _apply_message(self.prices, msg)

    def _run(self) -> None:  # pragma: no cover — requires a live socket
        """Reader loop: connect, auth, subscribe, dispatch frames to _on_message."""
        try:
            from websocket import create_connection
        except ImportError:
            logger.error(
                "PolygonPriceMonitor requires the 'websocket-client' package — "
                "falling back is the caller's responsibility"
            )
            return

        while not self._stop.is_set():
            try:
                self._ws = create_connection(POLYGON_WS_URL, timeout=30)
                self._ws.send(json.dumps(
                    {"action": "auth", "params": self._api_key}
                ))
                self._ws.send(json.dumps(
                    {"action": "subscribe", "params": self._subscribe_params()}
                ))
                while not self._stop.is_set():
                    raw = self._ws.recv()
                    if not raw:
                        break
                    self._on_message(raw)
            except Exception as e:
                if self._stop.is_set():
                    break
                logger.warning("polygon ws error (%s) — reconnecting in 5s", e)
                self._stop.wait(5)
            finally:
                try:
                    if self._ws is not None:
                        self._ws.close()
                except Exception:
                    pass
