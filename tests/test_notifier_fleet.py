"""Tests for executor notifier flow-doctor routing (config#1741)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from executor.notifier import send_trade_alert


@patch("executor.notifier.get_flow_doctor", return_value=None)
@patch("executor.notifier.send_message", return_value=True)
def test_send_trade_alert_legacy_fallback(mock_send, _mock_fd):
    assert send_trade_alert("BUY", "AAPL", 10, 150.0) is True
    mock_send.assert_called_once()


def test_send_trade_alert_routes_via_notify_event():
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "report-id"
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        assert send_trade_alert("SELL", "MSFT", 5, 400.0, "vwap", "daemon") is True
    kwargs = mock_fd.notify_event.call_args.kwargs
    assert kwargs["severity"] == "info"
    assert kwargs["dedup_key"] == "executor:trade:SELL:MSFT:5:400.0000"
