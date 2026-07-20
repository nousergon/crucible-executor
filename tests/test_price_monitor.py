"""Tests for executor.price_monitor — delayed market-data subscriber + tick aggregator."""

from unittest.mock import MagicMock

import pytest

from executor.price_monitor import PriceMonitor, _finite

# ── _finite helper ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("val,expected", [
    (1.5, 1.5),
    (100, 100.0),
    (None, None),
    (0, None),
    (-1, None),
    (float("nan"), None),
    (float("inf"), None),
    (-float("inf"), None),
    ("string", None),
    (True, None),  # bool not allowed; but bool IS isinstance of int — let's pin behavior
])
def test_finite_helper(val, expected):
    # Note: bool IS an int subclass — True returns 1.0 since 1>0
    if isinstance(val, bool):
        expected = 1.0 if val else None
    assert _finite(val) == expected


# ── PriceMonitor.subscribe + tick handler ──────────────────────────────────


def _make_ib_mock():
    """A mock IB connection: reqMarketDataType, qualifyContracts, reqMktData, etc.

    reqMktData returns a DISTINCT MagicMock per call (mirroring ib_insync,
    which hands back a distinct Ticker per subscription) so per-symbol
    subscription bookkeeping can be exercised.
    """
    ib = MagicMock()
    ib.pendingTickersEvent = MagicMock()
    ib.reqMktData.side_effect = lambda *a, **k: MagicMock(name="ticker_data")
    return ib


def _make_ticker(symbol, last=None, high=None, low=None, close=None, volume=None,
                 delayedLast=None, delayedHigh=None, delayedLow=None, delayedClose=None):
    """A mock ib_insync Ticker. .contract.symbol drives dispatch."""
    t = MagicMock()
    t.contract.symbol = symbol
    t.last = last
    t.high = high
    t.low = low
    t.close = close
    t.volume = volume
    t.delayedLast = delayedLast
    t.delayedHigh = delayedHigh
    t.delayedLow = delayedLow
    t.delayedClose = delayedClose
    return t


def test_subscribe_calls_market_data_type_and_registers_handler():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    pm.subscribe(["AAPL", "MSFT"])

    ib.reqMarketDataType.assert_called_once_with(3)
    assert ib.qualifyContracts.call_count == 2
    assert ib.reqMktData.call_count == 2
    assert set(pm._contracts.keys()) == {"AAPL", "MSFT"}
    assert len(pm._subscriptions) == 2


def test_subscribe_skips_tickers_that_fail_to_qualify():
    ib = _make_ib_mock()

    def qualify(contract):
        if contract.symbol == "BAD":
            raise RuntimeError("contract not found")

    ib.qualifyContracts.side_effect = qualify
    pm = PriceMonitor(ib)

    pm.subscribe(["AAPL", "BAD", "MSFT"])

    assert "BAD" not in pm._contracts
    assert set(pm._contracts.keys()) == {"AAPL", "MSFT"}
    assert len(pm._subscriptions) == 2


def test_on_pending_tickers_records_live_prices():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    ticker = _make_ticker("AAPL", last=150.0, high=152.0, low=148.0, close=149.0, volume=1_000_000)
    pm._on_pending_tickers({ticker})

    assert pm.prices["AAPL"]["last"] == 150.0
    assert pm.prices["AAPL"]["high"] == 152.0
    assert pm.prices["AAPL"]["low"] == 148.0
    assert pm.prices["AAPL"]["close"] == 149.0
    assert pm.prices["AAPL"]["volume"] == 1_000_000
    assert "updated_at" in pm.prices["AAPL"]


def test_on_pending_tickers_falls_back_to_delayed_fields():
    """When live fields are empty, delayed fields back-fill."""
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    ticker = _make_ticker(
        "AAPL", last=None, high=None, low=None, close=None,
        delayedLast=150.5, delayedHigh=151.0, delayedLow=149.0, delayedClose=149.5,
    )
    pm._on_pending_tickers({ticker})

    assert pm.prices["AAPL"]["last"] == 150.5
    assert pm.prices["AAPL"]["high"] == 151.0
    assert pm.prices["AAPL"]["low"] == 149.0
    assert pm.prices["AAPL"]["close"] == 149.5


def test_on_pending_tickers_skips_when_no_usable_price():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    ticker = _make_ticker("AAPL")  # all None
    pm._on_pending_tickers({ticker})
    assert "AAPL" not in pm.prices


def test_on_pending_tickers_skips_when_no_contract_symbol():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    bad = MagicMock()
    bad.contract = None  # falsy → symbol None
    pm._on_pending_tickers({bad})

    assert pm.prices == {}


def test_on_pending_tickers_tracks_intraday_high_low_across_updates():
    """Subsequent ticks should aggregate high/low rather than overwrite."""
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    # First tick: high 150, low 148
    t1 = _make_ticker("AAPL", last=149.0, high=150.0, low=148.0)
    pm._on_pending_tickers({t1})
    # Second tick: high 149 (lower), low 147 (lower) — should keep MAX(150, 149) and MIN(148, 147)
    t2 = _make_ticker("AAPL", last=147.5, high=149.0, low=147.0)
    pm._on_pending_tickers({t2})

    assert pm.prices["AAPL"]["high"] == 150.0  # kept from first tick
    assert pm.prices["AAPL"]["low"] == 147.0   # updated to second tick (lower)


def test_on_pending_tickers_volume_only_when_positive():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    t = _make_ticker("AAPL", last=100.0, volume=0)
    pm._on_pending_tickers({t})
    assert pm.prices["AAPL"]["volume"] is None


def test_unsubscribe_all_cancels_subscriptions():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)

    pm.subscribe(["AAPL", "MSFT"])
    pm.unsubscribe_all()

    assert ib.cancelMktData.call_count == 2
    assert pm._subscriptions == []
    assert pm._contracts == {}


def test_unsubscribe_all_swallows_cancel_errors():
    ib = _make_ib_mock()
    ib.cancelMktData.side_effect = RuntimeError("already cancelled")
    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL"])

    # Should not raise
    pm.unsubscribe_all()
    assert pm._subscriptions == []


def test_get_price_returns_state_or_none():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    assert pm.get_price("AAPL") is None

    ticker = _make_ticker("AAPL", last=150.0)
    pm._on_pending_tickers({ticker})

    assert pm.get_price("AAPL")["last"] == 150.0


# ── subscribed_tickers accessor + diff-based resubscribe (config#897) ────────


def test_subscribed_tickers_reflects_current_subscriptions():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    assert pm.subscribed_tickers() == set()

    pm.subscribe(["AAPL", "MSFT"])
    assert pm.subscribed_tickers() == {"AAPL", "MSFT"}


def test_resubscribe_adds_only_new_tickers():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL", "MSFT"])
    ib.reqMktData.reset_mock()
    ib.cancelMktData.reset_mock()

    added, removed = pm.resubscribe(["AAPL", "MSFT", "NVDA", "TSLA"])

    assert added == {"NVDA", "TSLA"}
    assert removed == set()
    # Only the two NEW tickers hit reqMktData — no churn on the shared set.
    assert ib.reqMktData.call_count == 2
    assert ib.cancelMktData.call_count == 0
    assert pm.subscribed_tickers() == {"AAPL", "MSFT", "NVDA", "TSLA"}


def test_resubscribe_cancels_only_removed_tickers():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL", "MSFT", "NVDA"])
    ib.reqMktData.reset_mock()
    ib.cancelMktData.reset_mock()

    added, removed = pm.resubscribe(["AAPL"])

    assert added == set()
    assert removed == {"MSFT", "NVDA"}
    assert ib.cancelMktData.call_count == 2
    assert ib.reqMktData.call_count == 0
    assert pm.subscribed_tickers() == {"AAPL"}
    assert len(pm._subscriptions) == 1


def test_resubscribe_unchanged_universe_is_no_op():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL", "MSFT"])
    ib.reqMktData.reset_mock()
    ib.cancelMktData.reset_mock()
    ib.reqMarketDataType.reset_mock()

    added, removed = pm.resubscribe(["MSFT", "AAPL"])  # same set, different order

    assert added == set()
    assert removed == set()
    assert ib.reqMktData.call_count == 0
    assert ib.cancelMktData.call_count == 0
    # No IB calls at all on an unchanged universe — zero churn.
    assert ib.reqMarketDataType.call_count == 0
    assert pm.subscribed_tickers() == {"AAPL", "MSFT"}


def test_resubscribe_handles_simultaneous_add_and_remove():
    ib = _make_ib_mock()
    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL", "MSFT"])
    ib.reqMktData.reset_mock()
    ib.cancelMktData.reset_mock()

    added, removed = pm.resubscribe(["AAPL", "NVDA"])

    assert added == {"NVDA"}
    assert removed == {"MSFT"}
    assert ib.reqMktData.call_count == 1
    assert ib.cancelMktData.call_count == 1
    assert pm.subscribed_tickers() == {"AAPL", "NVDA"}


def test_resubscribe_skips_unqualifiable_new_ticker():
    ib = _make_ib_mock()

    def qualify(contract):
        if contract.symbol == "BAD":
            raise RuntimeError("contract not found")

    pm = PriceMonitor(ib)
    pm.subscribe(["AAPL"])
    ib.qualifyContracts.side_effect = qualify

    added, removed = pm.resubscribe(["AAPL", "BAD"])

    # BAD failed to qualify → not reported as added, not tracked.
    assert added == set()
    assert "BAD" not in pm.subscribed_tickers()
    assert pm.subscribed_tickers() == {"AAPL"}
