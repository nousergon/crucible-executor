"""Tests for executor notifier flow-doctor routing (config#1741, config#1813)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from executor.notifier import send_daemon_status, send_trade_alert


@patch("executor.notifier.get_flow_doctor", return_value=None)
@patch("executor.notifier.send_message", return_value=True)
def test_send_trade_alert_legacy_fallback(mock_send, _mock_fd):
    assert send_trade_alert("BUY", "AAPL", 10, 150.0) is True
    mock_send.assert_called_once()


def test_send_trade_alert_routes_via_notify_event():
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "report-id"
    mock_fd.last_dispatched.return_value = True
    mock_fd.last_dispatch_reason.return_value = "fired"
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        assert send_trade_alert("SELL", "MSFT", 5, 400.0, "vwap", "daemon") is True
    kwargs = mock_fd.notify_event.call_args.kwargs
    assert kwargs["severity"] == "info"
    assert kwargs["dedup_key"] == "executor:trade:SELL:MSFT:5:400.0000"


def test_send_trade_alert_false_success_regression_config_1813(caplog):
    """config#1813: notify_event() returns a non-None report id even when
    the event was severity_filtered (evaluated, dispatched to ZERO
    notifiers) — e.g. a stale/shadowed flow-doctor override yaml missing
    the trades Telegram topic. send_trade_alert() must NOT log "alert
    sent" in that case, and must return False so callers can tell the
    difference. This is the exact false-success bug from the 2026-07-06
    incident (report id returned, but nothing was actually delivered)."""
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "report-id-but-nothing-sent"
    mock_fd.last_dispatched.return_value = False
    mock_fd.last_dispatch_reason.return_value = "severity_filtered"

    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        with caplog.at_level(logging.INFO, logger="executor.notifier"):
            result = send_trade_alert("REDUCE", "GE", 100, 377.51, "signal", "daemon")

    assert result is False
    assert not any(
        "Telegram alert sent via flow-doctor" in r.getMessage()
        for r in caplog.records
    )
    assert any(
        "NOT delivered" in r.getMessage() and "severity_filtered" in r.getMessage()
        for r in caplog.records
    )


def test_send_daemon_status_false_when_not_dispatched():
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "rid"
    mock_fd.last_dispatched.return_value = False
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        assert send_daemon_status("degraded mode") is False


def test_send_daemon_status_true_when_dispatched():
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "rid"
    mock_fd.last_dispatched.return_value = True
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        assert send_daemon_status("all good") is True


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
