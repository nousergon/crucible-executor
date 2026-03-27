"""Unit tests for daemon helpers extracted in Phase 1-2: validate, retry, cleanup."""
import pytest
from unittest.mock import MagicMock, patch

from executor.daemon import (
    _validate_sell_shares,
    _cleanup_connections,
    _place_order_with_retry,
    MAX_ORDER_RETRIES,
    ORDER_RETRY_DELAYS,
)


# ── _validate_sell_shares ────────────────────────────────────────────────────


class TestValidateSellShares:
    def test_normal_returns_requested_shares(self):
        positions = {"AAPL": {"shares": 50}}
        result = _validate_sell_shares(positions, "AAPL", 30, "SELL", "exit")
        assert result == 30

    def test_caps_when_requested_exceeds_held(self):
        positions = {"AAPL": {"shares": 20}}
        result = _validate_sell_shares(positions, "AAPL", 50, "SELL", "exit")
        assert result == 20

    def test_returns_none_when_no_position(self):
        positions = {"AAPL": {"shares": 0}}
        result = _validate_sell_shares(positions, "AAPL", 10, "SELL", "exit")
        assert result is None

    def test_returns_none_when_negative_position(self):
        positions = {"AAPL": {"shares": -5}}
        result = _validate_sell_shares(positions, "AAPL", 10, "SELL", "exit")
        assert result is None


# ── _place_order_with_retry ──────────────────────────────────────────────────


class TestPlaceOrderWithRetry:
    def test_succeeds_first_try(self):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {"status": "Filled"}
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        assert result["status"] == "Filled"
        assert ibkr.place_market_order.call_count == 1

    @patch("executor.daemon._time.sleep")
    def test_succeeds_after_first_rejection(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.side_effect = [
            {"status": "Rejected"},
            {"status": "Filled"},
        ]
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        assert result["status"] == "Filled"
        assert ibkr.place_market_order.call_count == 2
        mock_sleep.assert_called_once_with(ORDER_RETRY_DELAYS[1])

    @patch("executor.daemon._time.sleep")
    def test_all_retries_fail_returns_last_result(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {"status": "Rejected"}
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        assert result["status"] == "Rejected"
        assert ibkr.place_market_order.call_count == MAX_ORDER_RETRIES

    @patch("executor.daemon._time.sleep")
    @patch("executor.bracket_orders.place_bracket_with_stop")
    def test_uses_bracket_path(self, mock_bracket, mock_sleep):
        ibkr = MagicMock()
        mock_bracket.return_value = {"status": "Filled"}
        kwargs = {"stop_pct": 0.05}
        result = _place_order_with_retry(
            ibkr, "AAPL", "BUY", 10, "entry",
            use_bracket=True, bracket_kwargs=kwargs,
        )
        assert result["status"] == "Filled"
        mock_bracket.assert_called_once_with(ibkr, "AAPL", 10, stop_pct=0.05)
        ibkr.place_market_order.assert_not_called()

    def test_no_sleep_on_first_attempt(self):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {"status": "Filled"}
        with patch("executor.daemon._time.sleep") as mock_sleep:
            _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
            mock_sleep.assert_not_called()

    @patch("executor.daemon._time.sleep")
    def test_correct_sleep_delays_between_retries(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {"status": "Timeout"}
        _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        assert mock_sleep.call_count == MAX_ORDER_RETRIES - 1
        for i in range(1, MAX_ORDER_RETRIES):
            mock_sleep.assert_any_call(ORDER_RETRY_DELAYS[i])


# ── _cleanup_connections ─────────────────────────────────────────────────────


class TestCleanupConnections:
    def test_normal_no_exceptions(self):
        monitor = MagicMock()
        ibkr = MagicMock()
        _cleanup_connections(monitor, ibkr)
        monitor.unsubscribe_all.assert_called_once()
        ibkr.disconnect.assert_called_once()

    def test_handles_both_raising_exceptions(self):
        monitor = MagicMock()
        monitor.unsubscribe_all.side_effect = RuntimeError("sub fail")
        ibkr = MagicMock()
        ibkr.disconnect.side_effect = RuntimeError("disc fail")
        # Should not raise
        _cleanup_connections(monitor, ibkr)
        monitor.unsubscribe_all.assert_called_once()
        ibkr.disconnect.assert_called_once()

    def test_none_monitor(self):
        ibkr = MagicMock()
        _cleanup_connections(None, ibkr)
        ibkr.disconnect.assert_called_once()

    def test_none_ibkr(self):
        monitor = MagicMock()
        _cleanup_connections(monitor, None)
        monitor.unsubscribe_all.assert_called_once()
