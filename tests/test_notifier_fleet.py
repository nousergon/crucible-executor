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


def test_send_trade_alert_body_includes_timestamp_and_realized_pnl():
    """Trade alerts for closing/reducing trades must carry the execution
    timestamp and realized P&L already computed at the daemon's call
    sites — previously discarded, leaving the delivered alert as a bare
    "REDUCE COIN" with no fill/price/P&L detail."""
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "report-id"
    mock_fd.last_dispatched.return_value = True
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        assert (
            send_trade_alert(
                "REDUCE",
                "COIN",
                12,
                151.23,
                "atr_trail",
                "daemon",
                fill_time="2026-07-21T14:32:07+00:00",
                realized_pnl=340.11,
                realized_return_pct=8.4,
                realized_alpha_pct=6.1,
                days_held=9,
            )
            is True
        )
    body = mock_fd.notify_event.call_args.kwargs["body"]
    assert "2026-07-21 10:32:07 ET" in body  # UTC->ET conversion
    assert "Realized P&L: $+340.11 (+8.4%)" in body
    assert "Alpha vs SPY: +6.1% | Held: 9d" in body
    ctx = mock_fd.notify_event.call_args.kwargs["context"]
    assert ctx["realized_pnl"] == 340.11
    assert ctx["days_held"] == 9


def test_send_trade_alert_buy_omits_realized_pnl_block():
    """An ENTER/BUY has no prior fill to compare against — the realized
    P&L / alpha lines must not appear at all, not render as ``None``."""
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "report-id"
    mock_fd.last_dispatched.return_value = True
    with patch("executor.notifier.get_flow_doctor", return_value=mock_fd):
        send_trade_alert("BUY", "AAPL", 10, 150.0, "pullback", "daemon")
    body = mock_fd.notify_event.call_args.kwargs["body"]
    assert "Realized P&L" not in body
    assert "Alpha vs SPY" not in body
    assert "Time:" in body


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
    assert not any("Telegram alert sent via flow-doctor" in r.getMessage() for r in caplog.records)
    assert any("NOT delivered" in r.getMessage() and "severity_filtered" in r.getMessage() for r in caplog.records)


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
    from nousergon_lib import alerts

    from executor.notifier import publish_ops_alert

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
    from nousergon_lib import alerts

    from executor.notifier import publish_ops_alert

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
