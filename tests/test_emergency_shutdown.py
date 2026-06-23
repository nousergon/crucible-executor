"""Tests for executor.emergency_shutdown — paper-account-only halt + liquidate + notify."""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from executor import emergency_shutdown as mod


def _mock_client(account="DUH123456", nav=100_000.0, positions=None, orders=None):
    client = MagicMock()
    client.ib.managedAccounts.return_value = [account]
    client.get_portfolio_nav.return_value = nav
    client.get_positions.return_value = positions or {}
    client.get_open_orders.return_value = orders or []
    client.cancel_all_orders = MagicMock()
    client.get_current_price.return_value = 100.0
    client.place_market_order.return_value = {
        "ib_order_id": 1, "status": "Filled", "fill_price": 99.5,
        "filled_shares": 10, "fill_time": "2026-04-15T14:30:00",
    }
    client.disconnect = MagicMock()
    return client


@pytest.fixture
def stub_config(monkeypatch):
    cfg = {
        "ib_host": "127.0.0.1",
        "ib_port": 4002,
        "db_path": "/tmp/test-trades.db",
        "signals_bucket": "test-bucket",
    }
    monkeypatch.setattr(mod, "_load_config", lambda: cfg)
    return cfg


@pytest.fixture
def stub_db(monkeypatch):
    fake_conn = MagicMock()
    monkeypatch.setattr(mod, "init_db", MagicMock(return_value=fake_conn))
    monkeypatch.setattr(mod, "log_trade", MagicMock())
    monkeypatch.setattr(mod, "backup_to_s3", MagicMock())
    return fake_conn


# ── Dry-run path (execute=False) ────────────────────────────────────────────


def test_dry_run_reports_state_and_takes_no_action(stub_config, stub_db, monkeypatch, caplog):
    client = _mock_client(positions={"AAPL": {"shares": 10, "market_value": 1500.0}})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    with caplog.at_level("INFO"):
        mod.emergency_shutdown(execute=False, stop_instance=False)

    assert any("DRY RUN" in r.message for r in caplog.records)
    client.cancel_all_orders.assert_not_called()
    client.place_market_order.assert_not_called()
    client.disconnect.assert_called_once()


# ── Live account safety hard-exit ───────────────────────────────────────────


def test_live_account_triggers_sys_exit_99(stub_config, monkeypatch):
    """Account ID not starting with 'D' must hard-exit BEFORE any orders fire."""
    client = _mock_client(account="U999999")  # 'U' = live IB account prefix
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    with pytest.raises(SystemExit) as exc:
        mod.emergency_shutdown(execute=True, stop_instance=False)

    assert exc.value.code == 99
    client.disconnect.assert_called_once()
    # No orders placed even though execute=True
    client.place_market_order.assert_not_called()


def test_account_verification_failure_continues_with_warning(stub_config, stub_db, monkeypatch, caplog):
    """If managedAccounts() raises, log warning but continue (don't exit 99)."""
    client = _mock_client()
    client.ib.managedAccounts.side_effect = RuntimeError("connection lost")
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    with caplog.at_level("WARNING"):
        mod.emergency_shutdown(execute=False, stop_instance=False)

    assert any("Could not verify account type" in r.message for r in caplog.records)


# ── Execute path ────────────────────────────────────────────────────────────


def test_execute_cancels_orders_and_sells_positions(stub_config, stub_db, monkeypatch):
    positions = {
        "AAPL": {"shares": 10, "market_value": 1500.0},
        "MSFT": {"shares": 20, "market_value": 4000.0},
        "ZERO": {"shares": 0, "market_value": 0.0},  # skipped: shares<=0
    }
    client = _mock_client(positions=positions)
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    mod.emergency_shutdown(execute=True, stop_instance=False)

    client.cancel_all_orders.assert_called_once()
    # ZERO position skipped; AAPL and MSFT placed
    assert client.place_market_order.call_count == 2
    tickers_sold = {c.args[0] for c in client.place_market_order.call_args_list}
    assert tickers_sold == {"AAPL", "MSFT"}
    mod.log_trade.assert_called()  # type: ignore[attr-defined]
    mod.backup_to_s3.assert_called_once()  # type: ignore[attr-defined]


def test_execute_continues_when_cancel_orders_fails(stub_config, stub_db, monkeypatch, caplog):
    """Order-cancel failure must NOT abort liquidation — that's the whole point."""
    client = _mock_client(positions={"AAPL": {"shares": 10}})
    client.cancel_all_orders.side_effect = RuntimeError("IB connection lost")
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    with caplog.at_level("ERROR"):
        mod.emergency_shutdown(execute=True, stop_instance=False)

    # Liquidation still ran
    client.place_market_order.assert_called_once()


def test_execute_per_position_sell_failure_logged_but_others_continue(stub_config, stub_db, monkeypatch, caplog):
    positions = {
        "AAPL": {"shares": 10, "market_value": 1500.0},
        "BAD": {"shares": 5, "market_value": 500.0},
        "MSFT": {"shares": 20, "market_value": 4000.0},
    }
    client = _mock_client(positions=positions)

    def sell(ticker, side, shares, *_):
        if ticker == "BAD":
            raise RuntimeError("simulated sell failure")
        return {"ib_order_id": 1, "status": "Filled", "fill_price": 100.0,
                "filled_shares": shares, "fill_time": "2026-04-15"}

    client.place_market_order.side_effect = sell
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    with caplog.at_level("ERROR"):
        mod.emergency_shutdown(execute=True, stop_instance=False)

    # All 3 attempted; BAD failed; AAPL + MSFT succeeded
    assert client.place_market_order.call_count == 3
    assert any("SELL FAILED" in r.message for r in caplog.records)


def test_execute_daemon_stop_failure_swallowed(stub_config, stub_db, monkeypatch, caplog):
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    sub = MagicMock()
    sub.run.side_effect = RuntimeError("daemon already stopped")
    monkeypatch.setattr(mod, "subprocess", sub)
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    with caplog.at_level("WARNING"):
        mod.emergency_shutdown(execute=True, stop_instance=False)
    assert any("Daemon stop failed" in r.message for r in caplog.records)


def test_main_dry_run_invokes_emergency_shutdown(monkeypatch):
    """`main()` parses args and dispatches with execute=False by default."""
    called_with = {}

    def fake_es(execute, stop_instance):
        called_with["execute"] = execute
        called_with["stop_instance"] = stop_instance

    monkeypatch.setattr(mod, "emergency_shutdown", fake_es)
    monkeypatch.setattr(sys, "argv", ["emergency_shutdown.py"])

    mod.main()

    assert called_with == {"execute": False, "stop_instance": False}


def test_main_execute_flag_logs_warning_and_dispatches(monkeypatch, caplog):
    captured = {}

    def fake_es(execute, stop_instance):
        captured["execute"] = execute
        captured["stop_instance"] = stop_instance

    monkeypatch.setattr(mod, "emergency_shutdown", fake_es)
    monkeypatch.setattr(sys, "argv", ["emergency_shutdown.py", "--execute", "--stop-instance"])

    with caplog.at_level("WARNING"):
        mod.main()

    assert captured == {"execute": True, "stop_instance": True}
    assert any("EMERGENCY SHUTDOWN — EXECUTING" in r.message for r in caplog.records)


def test_execute_notification_failure_swallowed(stub_config, stub_db, monkeypatch, caplog):
    """SNS publish failure must NOT crash the shutdown — log warning, continue."""
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())

    fake_sns = MagicMock()
    fake_sns.publish.side_effect = RuntimeError("SNS down")
    fake_boto3 = MagicMock()
    fake_boto3.client = MagicMock(side_effect=lambda svc, **kw:
                                  fake_sns if svc == "sns" else MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    with caplog.at_level("WARNING"):
        mod.emergency_shutdown(execute=True, stop_instance=False)

    assert any("Notification failed" in r.message for r in caplog.records)


def test_execute_backup_failure_swallowed(stub_config, monkeypatch, caplog):
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    fake_conn = MagicMock()
    monkeypatch.setattr(mod, "init_db", MagicMock(return_value=fake_conn))
    monkeypatch.setattr(mod, "log_trade", MagicMock())
    monkeypatch.setattr(mod, "backup_to_s3", MagicMock(side_effect=RuntimeError("S3 down")))

    with caplog.at_level("ERROR"):
        mod.emergency_shutdown(execute=True, stop_instance=False)
    assert any("Backup failed" in r.message for r in caplog.records)


def test_execute_stop_instance_failure_swallowed(stub_config, stub_db, monkeypatch, caplog):
    """EC2 stop_instances failure must NOT crash — logged as error, shutdown completes."""
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())

    fake_ec2 = MagicMock()
    fake_ec2.stop_instances.side_effect = RuntimeError("EC2 API down")
    fake_boto3 = MagicMock()
    fake_boto3.client = MagicMock(side_effect=lambda svc, **kw:
                                  fake_ec2 if svc == "ec2" else MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    fake_urlopen = MagicMock()
    fake_urlopen.return_value.read = MagicMock(side_effect=[b"token-abc", b"i-12345"])

    with caplog.at_level("ERROR"):
        with patch("urllib.request.urlopen", fake_urlopen):
            mod.emergency_shutdown(execute=True, stop_instance=True)

    assert any("EC2 stop failed" in r.message for r in caplog.records)


# ── Trading-day axis (issue config#1016) ────────────────────────────────────


def test_execute_on_weekend_keys_trades_on_prior_trading_day(stub_config, stub_db, monkeypatch):
    """An emergency shutdown run on a Saturday must key its trade artifacts on
    the prior NYSE session, not the weekend calendar date:

      * log_trade receives trading_day = the prior session (2026-04-24),
      * the trades.date column keeps its calendar-audit semantic (2026-04-25),
      * the S3 backup is keyed on the trading day (the artifact key).

    Mirrors the now_dual() weekend contract in nousergon_lib.dates."""
    import nousergon_lib.dates as dates_mod
    from nousergon_lib.dates import now_dual

    saturday = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)  # 8 AM ET Sat
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=saturday))

    client = _mock_client(positions={"AAPL": {"shares": 10, "market_value": 1500.0}})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", MagicMock())

    mod.emergency_shutdown(execute=True, stop_instance=False)

    # log_trade keyed on the dual axis.
    logged = mod.log_trade.call_args_list  # type: ignore[attr-defined]
    assert logged, "expected at least one logged trade"
    trade = logged[0].args[1]
    assert trade["trading_day"] == "2026-04-24"  # prior session, NOT Saturday
    assert trade["date"] == "2026-04-25"  # calendar-audit column unchanged

    # S3 backup keyed on the trading day (artifact key).
    backup_args = mod.backup_to_s3.call_args  # type: ignore[attr-defined]
    assert backup_args.args[1] == "2026-04-24"

    # Behavior unchanged: the position was still sold.
    client.place_market_order.assert_called_once()


def test_execute_stop_instance_calls_ec2_stop(stub_config, stub_db, monkeypatch):
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))
    monkeypatch.setattr(mod, "subprocess", MagicMock())

    fake_ec2 = MagicMock()
    fake_boto3 = MagicMock()
    fake_boto3.client = MagicMock(side_effect=lambda svc, **kw:
                                  fake_ec2 if svc == "ec2" else MagicMock())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    fake_urlopen = MagicMock()
    fake_urlopen.return_value.read = MagicMock(side_effect=[b"token-abc", b"i-12345"])
    with patch("urllib.request.urlopen", fake_urlopen):
        mod.emergency_shutdown(execute=True, stop_instance=True)

    fake_ec2.stop_instances.assert_called_once_with(InstanceIds=["i-12345"])
