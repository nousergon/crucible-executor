"""Unit tests for executor.snapshot_capturer — Phase 2 of EOD-SF cutover.

Snapshot is the date-locked source of truth for end-of-day state. Capture
runs as the CaptureSnapshot SF step (between PostMarketData and
EODReconcile); eod_reconcile reads the snapshot instead of querying live
IB. The architectural invariant: a row keyed by `run_date=X` sources its
inputs from observations made at time X.
"""
from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from executor.snapshot_capturer import _snapshot_key, load_snapshot, run

# ── _snapshot_key ────────────────────────────────────────────────────────────


def test_snapshot_key_format():
    """S3 key path is `trades/snapshots/{run_date}.json` — locked-in
    contract since eod_reconcile's load_snapshot reader uses the same."""
    assert _snapshot_key("2026-04-29") == "trades/snapshots/2026-04-29.json"


# ── load_snapshot ────────────────────────────────────────────────────────────


class TestLoadSnapshot:
    def test_returns_parsed_dict_on_success(self):
        body = MagicMock()
        body.read.return_value = json.dumps({
            "run_date": "2026-04-29",
            "account": {"net_liquidation": 1000000.0},
            "positions": {},
            "accrued_dividends": {},
        }).encode("utf-8")
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": body}
        # NoSuchKey isn't raised in the success path so the exceptions
        # attribute doesn't matter, but mock it for completeness.
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        with patch("boto3.client", return_value=s3):
            result = load_snapshot("alpha-engine-research", "2026-04-29")

        assert result["run_date"] == "2026-04-29"
        assert result["account"]["net_liquidation"] == 1000000.0
        s3.get_object.assert_called_once_with(
            Bucket="alpha-engine-research",
            Key="trades/snapshots/2026-04-29.json",
        )

    def test_returns_none_on_no_such_key(self):
        """Missing snapshot returns None — the caller (eod_reconcile)
        decides what to do (currently: raise with a helpful message)."""
        no_such_key = type("NoSuchKey", (Exception,), {})
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = no_such_key
        s3.get_object.side_effect = no_such_key()

        with patch("boto3.client", return_value=s3):
            result = load_snapshot("alpha-engine-research", "2026-04-29")
        assert result is None

    def test_returns_none_on_404_string_in_error(self):
        """Some S3-compatible backends raise raw HTTPClientError with '404'
        in the string instead of a typed NoSuchKey — handle both."""
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.get_object.side_effect = RuntimeError("HTTP 404 Not Found")

        with patch("boto3.client", return_value=s3):
            result = load_snapshot("alpha-engine-research", "2026-04-29")
        assert result is None

    def test_raises_loud_on_unrelated_errors(self):
        """A genuine S3 error (auth, network, region, etc.) must NOT be
        silenced — the snapshot path is load-bearing."""
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.get_object.side_effect = RuntimeError("AccessDenied")

        with patch("boto3.client", return_value=s3):
            with pytest.raises(RuntimeError, match="AccessDenied"):
                load_snapshot("alpha-engine-research", "2026-04-29")


# ── run ──────────────────────────────────────────────────────────────────────


def _mock_config(db_path="/nonexistent/trades.db"):
    return {
        "trades_bucket": "alpha-engine-research",
        "aws_region": "us-east-1",
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 4002,
        "ibkr_client_id": 99,
        # Read by the fill-reconciliation backstop (alpha-engine-config-I10800).
        # Tests that do not exercise the backstop patch `_reconcile_fills` out;
        # the default path is deliberately unopenable so an unpatched test
        # fails loudly rather than touching a real database.
        "db_path": db_path,
    }


@contextlib.contextmanager
def _no_fill_reconciliation():
    """Stub the fill-reconciliation backstop for tests about snapshot capture."""
    with patch("executor.snapshot_capturer._reconcile_fills") as stub:
        yield stub


class TestRun:
    def test_default_resolves_to_now_dual_trading_day(self):
        with patch("executor.snapshot_capturer.now_dual") as mock_now_dual, \
             patch("executor.snapshot_capturer.load_config") as mock_cfg, \
             patch("executor.snapshot_capturer.IBKRClient") as mock_ib_cls, \
             patch("boto3.client") as mock_boto:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-29", calendar_date="2026-04-29"
            )
            mock_cfg.return_value = _mock_config()
            ib = MagicMock()
            ib.get_account_snapshot.return_value = {"net_liquidation": 1.0}
            ib.get_positions.return_value = {}
            ib.get_accrued_dividends_by_symbol.return_value = {}
            mock_ib_cls.return_value = ib
            s3 = MagicMock()
            mock_boto.return_value = s3

            with _no_fill_reconciliation():
                run(run_date=None)

            # Verify the put_object key was constructed from the resolved
            # run_date, not from None or some other default.
            kwargs = s3.put_object.call_args.kwargs
            assert kwargs["Key"] == "trades/snapshots/2026-04-29.json"

    def test_explicit_past_date_raises(self):
        """Capture is intrinsically live — get_account_snapshot returns
        now-as-of state. A historical run_date would persist today's
        state under yesterday's key, corrupting the contract."""
        with patch("executor.snapshot_capturer.now_dual") as mock_now_dual:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-29", calendar_date="2026-04-29"
            )
            with pytest.raises(RuntimeError, match="refusing run_date"):
                run(run_date="2026-04-27")

    def test_payload_shape(self):
        """Snapshot must contain run_date + captured_at + schema_version
        + the three IB-derived fields. eod_reconcile depends on this shape."""
        with patch("executor.snapshot_capturer.now_dual") as mock_now_dual, \
             patch("executor.snapshot_capturer.load_config") as mock_cfg, \
             patch("executor.snapshot_capturer.IBKRClient") as mock_ib_cls, \
             patch("boto3.client") as mock_boto:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-29", calendar_date="2026-04-29"
            )
            mock_cfg.return_value = _mock_config()
            ib = MagicMock()
            ib.get_account_snapshot.return_value = {
                "net_liquidation": 1_000_000.0,
                "total_cash": 250_000.0,
                "settled_cash": 250_000.0,
                "accrued_interest": 12.34,
                "gross_position_value": 750_000.0,
                "buying_power": 2_000_000.0,
                "unrealized_pnl": 5_000.0,
                "realized_pnl": -123.45,
            }
            ib.get_positions.return_value = {
                "AAPL": {
                    "shares": 100,
                    "market_value": 20000.0,
                    "avg_cost": 195.0,
                    "unrealized_pnl": 500.0,
                    "sector": "",
                }
            }
            ib.get_accrued_dividends_by_symbol.return_value = {"PFE": 12.34}
            mock_ib_cls.return_value = ib
            s3 = MagicMock()
            mock_boto.return_value = s3

            with _no_fill_reconciliation():
                run(run_date="2026-04-29")

            kwargs = s3.put_object.call_args.kwargs
            payload = json.loads(kwargs["Body"].decode("utf-8"))
            assert payload["run_date"] == "2026-04-29"
            assert "captured_at" in payload
            assert payload["schema_version"] == 1
            assert payload["account"]["net_liquidation"] == 1_000_000.0
            assert payload["account"]["accrued_interest"] == 12.34
            assert payload["positions"]["AAPL"]["shares"] == 100
            assert payload["accrued_dividends"]["PFE"] == 12.34
            assert kwargs["ContentType"] == "application/json"

    def test_disconnects_ib_after_read(self):
        """IBKR disconnect must run even on success path. Reuses the
        existing eod_reconcile pattern; capturer is short-lived."""
        with patch("executor.snapshot_capturer.now_dual") as mock_now_dual, \
             patch("executor.snapshot_capturer.load_config") as mock_cfg, \
             patch("executor.snapshot_capturer.IBKRClient") as mock_ib_cls, \
             patch("boto3.client") as mock_boto:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-29", calendar_date="2026-04-29"
            )
            mock_cfg.return_value = _mock_config()
            ib = MagicMock()
            ib.get_account_snapshot.return_value = {"net_liquidation": 1.0}
            ib.get_positions.return_value = {}
            ib.get_accrued_dividends_by_symbol.return_value = {}
            mock_ib_cls.return_value = ib
            mock_boto.return_value = MagicMock()

            with _no_fill_reconciliation():
                run(run_date="2026-04-29")

            ib.disconnect.assert_called_once()

    def test_disconnects_ib_even_on_ib_failure(self):
        """If get_account_snapshot raises, disconnect still runs (finally
        block). Don't leak IB connections across capture-failure retries."""
        with patch("executor.snapshot_capturer.now_dual") as mock_now_dual, \
             patch("executor.snapshot_capturer.load_config") as mock_cfg, \
             patch("executor.snapshot_capturer.IBKRClient") as mock_ib_cls:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-29", calendar_date="2026-04-29"
            )
            mock_cfg.return_value = _mock_config()
            ib = MagicMock()
            ib.get_account_snapshot.side_effect = RuntimeError("IB outage")
            mock_ib_cls.return_value = ib

            with pytest.raises(RuntimeError, match="IB outage"):
                run(run_date="2026-04-29")
            ib.disconnect.assert_called_once()


# ── Fill-reconciliation backstop (alpha-engine-config-I10800) ────────────────


class TestFillReconciliationBackstop:
    """CaptureSnapshot is the EOD backstop for a trade row whose fill landed
    after the daemon's poll window and after the daemon's own last tick. This
    session is freshly connected, so ib_insync has synced the day's executions
    onto it."""

    @staticmethod
    def _patched(cfg):
        return (
            patch("executor.snapshot_capturer.now_dual",
                  return_value=SimpleNamespace(trading_day="2026-04-29", calendar_date="2026-04-29")),
            patch("executor.snapshot_capturer.load_config", return_value=cfg),
            patch("executor.snapshot_capturer.IBKRClient"),
            patch("boto3.client"),
        )

    def _run(self, cfg):
        a, b, c, d = self._patched(cfg)
        with a, b, c as mock_ib_cls, d as mock_boto:
            ib = MagicMock()
            ib.get_account_snapshot.return_value = {"net_liquidation": 1.0}
            ib.get_positions.return_value = {}
            ib.get_accrued_dividends_by_symbol.return_value = {}
            ib.ib.fills.return_value = []
            mock_ib_cls.return_value = ib
            mock_boto.return_value = MagicMock()
            run(run_date="2026-04-29")
            return ib

    def test_reconciles_the_open_row_against_the_session_fills(self, tmp_path):
        from executor.fill_reconciliation import unresolved_trades
        from executor.trade_logger import init_db, log_trade

        db = str(tmp_path / "trades.db")
        conn = init_db(db)
        log_trade(conn, {
            "date": "2026-04-29", "ticker": "PBF", "action": "REDUCE", "shares": 775,
            "price_at_order": 77.33, "fill_price": None, "ib_order_id": 690, "status": "Working",
        })
        conn.close()

        a, b, c, d = self._patched(_mock_config(db))
        with a, b, c as mock_ib_cls, d as mock_boto:
            ib = MagicMock()
            ib.get_account_snapshot.return_value = {"net_liquidation": 1.0}
            ib.get_positions.return_value = {}
            ib.get_accrued_dividends_by_symbol.return_value = {}
            ib.ib.fills.return_value = [SimpleNamespace(
                contract=SimpleNamespace(symbol="PBF"),
                execution=SimpleNamespace(
                    orderId=690, side="SLD", shares=775.0, price=74.0, execId="x",
                    time=datetime(2026, 4, 29, 13, 31, 48, tzinfo=UTC),
                ),
                commissionReport=SimpleNamespace(execId="x", commission=3.5),
            )]
            mock_ib_cls.return_value = ib
            mock_boto.return_value = MagicMock()
            run(run_date="2026-04-29")

        conn = init_db(db)
        assert conn.execute("SELECT status, fill_price FROM trades").fetchone() == ("Filled", 74.0)
        assert unresolved_trades(conn, "2026-04-29") == []

    def test_a_reconciliation_failure_never_costs_the_snapshot(self, caplog):
        """The snapshot is the day's ONE non-re-runnable artifact; a broken
        backstop must not be able to take it down."""
        with patch("executor.snapshot_capturer._reconcile_fills", side_effect=AssertionError("unreachable")):
            a, b, c, d = self._patched(_mock_config())
            with a, b, c as mock_ib_cls, d as mock_boto:
                ib = MagicMock()
                ib.get_account_snapshot.return_value = {"net_liquidation": 1.0}
                ib.get_positions.return_value = {}
                ib.get_accrued_dividends_by_symbol.return_value = {}
                mock_ib_cls.return_value = ib
                s3 = MagicMock()
                mock_boto.return_value = s3
                with pytest.raises(AssertionError):
                    run(run_date="2026-04-29")

        # ...and when the failure is inside _reconcile_fills itself, it is
        # caught there, logged, and the snapshot still lands.
        with caplog.at_level("ERROR"):
            ib = self._run(_mock_config("/nonexistent/dir/trades.db"))
        assert ib.disconnect.called
        assert any("fill reconciliation backstop failed" in r.getMessage() for r in caplog.records)

    def test_runs_before_the_ib_session_is_closed(self):
        """Ordering is the point: ib.fills() needs a live session."""
        ib = self._run(_mock_config("/nonexistent/dir/trades.db"))
        ib.disconnect.assert_called_once()
