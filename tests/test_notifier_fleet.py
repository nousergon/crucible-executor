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


@patch("executor.notifier.get_flow_doctor", return_value=None)
def test_publish_ops_alert_sns_only_when_flow_doctor_inactive(_mock_fd):
    from executor.notifier import publish_ops_alert
    from nousergon_lib import alerts

    with patch.object(alerts, "publish") as publish_mock:
        publish_ops_alert(
            "turnover breach",
            severity="WARN",
            source="test",
            dedup_key="k1",
        )
    publish_mock.assert_called_once()
    assert publish_mock.call_args.kwargs["telegram"] is False
    assert publish_mock.call_args.kwargs["dedup_key"] == "k1"


@patch("executor.notifier.get_flow_doctor")
def test_publish_ops_alert_routes_telegram_via_flow_doctor(mock_get_fd):
    from executor.notifier import publish_ops_alert
    from nousergon_lib import alerts

    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "rid-1"
    mock_get_fd.return_value = mock_fd
    with patch.object(alerts, "publish") as publish_mock:
        publish_ops_alert(
            "turnover breach",
            severity="ERROR",
            source="alpha-engine/executor/turnover_tripwire.py",
            dedup_key="k2",
        )
    publish_mock.assert_called_once()
    assert publish_mock.call_args.kwargs["telegram"] is False
    mock_fd.notify_event.assert_called_once()
    assert mock_fd.notify_event.call_args.kwargs["severity"] == "error"
    assert mock_fd.notify_event.call_args.kwargs["dedup_key"] == "k2"
