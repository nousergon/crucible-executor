"""Tests for the polygon.io WebSocket price monitor (config#913).

The transport (socket connect/auth/reconnect) is excluded from coverage — these
tests pin the pure parsing logic that feeds entry-trigger evaluation, the
status-frame auth handling, and the source-selection factory. No live socket.
"""


import pytest

from executor import polygon_price_monitor as ppm
from executor.polygon_price_monitor import (
    PolygonPriceMonitor,
    _apply_message,
    make_price_monitor,
    price_source,
)


class TestApplyMessage:
    def test_aggregate_event_sets_state(self):
        prices = {}
        sym = _apply_message(prices, {
            "ev": "A", "sym": "AAPL", "o": 250.0, "h": 252.0,
            "l": 249.0, "c": 251.0, "v": 1000,
        })
        assert sym == "AAPL"
        st = prices["AAPL"]
        assert st["last"] == 251.0      # prefers close
        assert st["high"] == 252.0
        assert st["low"] == 249.0
        assert st["close"] == 251.0
        assert st["volume"] == 1000
        assert "updated_at" in st

    def test_trade_event_sets_last(self):
        prices = {}
        sym = _apply_message(prices, {"ev": "T", "sym": "MSFT", "p": 410.5, "s": 50})
        assert sym == "MSFT"
        assert prices["MSFT"]["last"] == 410.5
        assert prices["MSFT"]["volume"] == 50

    def test_running_high_low_tracked(self):
        prices = {}
        _apply_message(prices, {"ev": "A", "sym": "X", "c": 100.0, "h": 100.0, "l": 100.0})
        _apply_message(prices, {"ev": "A", "sym": "X", "c": 105.0, "h": 106.0, "l": 104.0})
        _apply_message(prices, {"ev": "A", "sym": "X", "c": 95.0, "h": 96.0, "l": 94.0})
        assert prices["X"]["high"] == 106.0  # max across ticks
        assert prices["X"]["low"] == 94.0    # min across ticks
        assert prices["X"]["last"] == 95.0   # latest close

    def test_no_symbol_ignored(self):
        prices = {}
        assert _apply_message(prices, {"ev": "A", "c": 100.0}) is None
        assert prices == {}

    def test_unusable_price_ignored(self):
        prices = {}
        # negative / zero / nan prices are not usable
        assert _apply_message(prices, {"ev": "A", "sym": "X", "c": 0, "o": -1}) is None
        assert prices == {}

    def test_unknown_event_ignored(self):
        prices = {}
        assert _apply_message(prices, {"ev": "Q", "sym": "X", "bp": 1.0}) is None
        assert prices == {}

    def test_close_falls_back_to_prev_when_missing(self):
        prices = {"X": {"last": 10.0, "high": 10.0, "low": 10.0,
                        "close": 10.0, "volume": 5}}
        # trade event has no close field of its own; keeps prev close? No — trade
        # sets close=last. But volume falls back to prev when trade lacks size.
        _apply_message(prices, {"ev": "T", "sym": "X", "p": 11.0})
        assert prices["X"]["last"] == 11.0
        assert prices["X"]["volume"] == 5  # prev volume preserved


class TestOnMessage:
    def _monitor(self, monkeypatch):
        monkeypatch.setattr(
            ppm, "get_secret", lambda *a, **k: "fake-key", raising=False
        )
        # get_secret is imported lazily inside __init__; patch the lib symbol.
        import nousergon_lib.secrets as secrets
        monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: "fake-key")
        return PolygonPriceMonitor()

    def test_list_frame_applies_all(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._on_message('[{"ev":"A","sym":"AAPL","c":251.0},'
                      '{"ev":"A","sym":"MSFT","c":410.0}]')
        assert m.get_price("AAPL")["last"] == 251.0
        assert m.get_price("MSFT")["last"] == 410.0

    def test_auth_success_status_sets_flag(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._on_message('[{"ev":"status","status":"auth_success","message":"ok"}]')
        assert m._authed.is_set()

    def test_undecodable_frame_is_safe(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._on_message("not json")  # must not raise
        assert m.prices == {}

    def test_subscribe_params(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._tickers = ["AAPL", "MSFT"]
        assert m._subscribe_params() == "A.AAPL,A.MSFT"

    def test_requires_api_key(self, monkeypatch):
        import nousergon_lib.secrets as secrets
        monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: "")
        with pytest.raises(ValueError):
            PolygonPriceMonitor()

    # ── mid-session resubscribe parity with PriceMonitor (config#897) ────────

    def test_subscribed_tickers_reflects_target(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._tickers = ["AAPL", "MSFT"]
        assert m.subscribed_tickers() == {"AAPL", "MSFT"}

    def test_resubscribe_updates_target_and_reports_delta(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._tickers = ["AAPL", "MSFT"]
        added, removed = m.resubscribe(["AAPL", "NVDA"])
        assert added == {"NVDA"}
        assert removed == {"MSFT"}
        assert m.subscribed_tickers() == {"AAPL", "NVDA"}

    def test_resubscribe_unchanged_is_no_op(self, monkeypatch):
        m = self._monitor(monkeypatch)
        m._tickers = ["AAPL", "MSFT"]
        added, removed = m.resubscribe(["MSFT", "AAPL"])
        assert added == set() and removed == set()
        assert m._tickers == ["AAPL", "MSFT"]  # target untouched


class TestSourceSelection:
    def test_price_source_default(self, monkeypatch):
        monkeypatch.delenv("EXECUTOR_PRICE_SOURCE", raising=False)
        assert price_source() == "ib_delayed"

    def test_price_source_polygon(self, monkeypatch):
        monkeypatch.setenv("EXECUTOR_PRICE_SOURCE", "polygon_ws")
        assert price_source() == "polygon_ws"

    def test_make_price_monitor_default_is_ib(self, monkeypatch):
        monkeypatch.delenv("EXECUTOR_PRICE_SOURCE", raising=False)
        sentinel = object()

        class _FakeIBMonitor:
            def __init__(self, ib):
                self.ib = ib

        import executor.price_monitor as pm
        monkeypatch.setattr(pm, "PriceMonitor", _FakeIBMonitor)
        mon = make_price_monitor(sentinel)
        assert isinstance(mon, _FakeIBMonitor)
        assert mon.ib is sentinel

    def test_make_price_monitor_polygon(self, monkeypatch):
        monkeypatch.setenv("EXECUTOR_PRICE_SOURCE", "polygon_ws")
        import nousergon_lib.secrets as secrets
        monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: "fake-key")
        mon = make_price_monitor(object())
        assert isinstance(mon, PolygonPriceMonitor)

    def test_make_price_monitor_polygon_falls_back_on_error(self, monkeypatch):
        monkeypatch.setenv("EXECUTOR_PRICE_SOURCE", "polygon_ws")
        import nousergon_lib.secrets as secrets
        monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: "")  # no key → raises

        class _FakeIBMonitor:
            def __init__(self, ib):
                self.ib = ib

        import executor.price_monitor as pm
        monkeypatch.setattr(pm, "PriceMonitor", _FakeIBMonitor)
        mon = make_price_monitor(object())
        assert isinstance(mon, _FakeIBMonitor)  # fell back to IB
