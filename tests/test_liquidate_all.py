"""Tests for executor.liquidate_all — paper-account reset tool.

Focused on the trading-day-axis fix (issue config#1016): a liquidation run on a
weekend/holiday must key its trade artifacts on the last *closed* NYSE session,
not the raw calendar date, while preserving the calendar-audit semantic of the
trades.date column. Behavior (which positions get sold) must be unaffected.

``liquidate_all`` uses bare module-level imports (``from ibkr import ...``,
``from trade_logger import ...``) because it is normally run as a script from
inside the ``executor/`` directory. We stub those two modules in ``sys.modules``
before importing so the module loads cleanly under pytest, then monkeypatch the
names bound in its namespace.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Stub the bare-import dependencies BEFORE importing the module under test.
sys.modules.setdefault("ibkr", MagicMock())
sys.modules.setdefault("trade_logger", MagicMock())

from executor import liquidate_all as mod  # noqa: E402


def _mock_client(nav=100_000.0, positions=None):
    client = MagicMock()
    client.get_portfolio_nav.return_value = nav
    client.get_positions.return_value = positions or {}
    client.get_current_price.return_value = 100.0
    client.place_market_order.return_value = {"ib_order_id": 1, "status": "Filled"}
    client.disconnect = MagicMock()
    return client


@pytest.fixture
def stub_env(monkeypatch):
    cfg = {
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 4002,
        "ibkr_client_id": 1,
        "db_path": "/tmp/test-trades.db",
        "trades_bucket": "test-bucket",
    }
    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "init_db", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(mod, "log_trade", MagicMock())
    monkeypatch.setattr(mod, "backup_to_s3", MagicMock())
    return cfg


def test_dry_run_places_no_orders(stub_env, monkeypatch):
    client = _mock_client(positions={"AAPL": {"shares": 10, "market_value": 1500.0}})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    mod.liquidate(execute=False, skip_confirm=True)

    client.place_market_order.assert_not_called()
    mod.log_trade.assert_not_called()  # type: ignore[attr-defined]
    client.disconnect.assert_called_once()


def test_no_positions_is_noop(stub_env, monkeypatch):
    client = _mock_client(positions={})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    mod.liquidate(execute=True, skip_confirm=True)

    client.place_market_order.assert_not_called()
    client.disconnect.assert_called_once()


def test_execute_sells_positions(stub_env, monkeypatch):
    positions = {
        "AAPL": {"shares": 10, "market_value": 1500.0},
        "MSFT": {"shares": 20, "market_value": 4000.0},
        "ZERO": {"shares": 0, "market_value": 0.0},  # skipped: shares<=0
    }
    client = _mock_client(positions=positions)
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    mod.liquidate(execute=True, skip_confirm=True)

    assert client.place_market_order.call_count == 2
    tickers = {c.args[0] for c in client.place_market_order.call_args_list}
    assert tickers == {"AAPL", "MSFT"}


# ── Trading-day axis (issue config#1016) ────────────────────────────────────


def test_execute_on_weekend_keys_trades_on_prior_trading_day(stub_env, monkeypatch):
    """A liquidation run on a Saturday keys its artifacts on the prior session:

      * log_trade receives trading_day = the prior session (2026-04-24),
      * the trades.date column keeps the calendar date (2026-04-25),
      * the S3 backup is keyed on the trading day (the artifact key).

    Behavior (positions sold) is unchanged."""
    import nousergon_lib.dates as dates_mod
    from nousergon_lib.dates import now_dual

    saturday = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)  # 8 AM ET Sat
    monkeypatch.setattr(dates_mod, "now_dual", lambda: now_dual(now=saturday))

    client = _mock_client(positions={"AAPL": {"shares": 10, "market_value": 1500.0}})
    monkeypatch.setattr(mod, "IBKRClient", MagicMock(return_value=client))

    mod.liquidate(execute=True, skip_confirm=True)

    logged = mod.log_trade.call_args_list  # type: ignore[attr-defined]
    assert logged, "expected a logged liquidation trade"
    trade = logged[0].args[1]
    assert trade["trading_day"] == "2026-04-24"  # prior session, NOT Saturday
    assert trade["date"] == "2026-04-25"  # calendar-audit column

    backup_args = mod.backup_to_s3.call_args  # type: ignore[attr-defined]
    assert backup_args.args[1] == "2026-04-24"  # backup key on trading day

    client.place_market_order.assert_called_once()  # behavior unchanged
