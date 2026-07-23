"""Tests for executor/intraday_snapshot.py — surveillance universe
computation + S3 snapshot writer (latest_prices + heartbeat).

Pure-logic + mocked-S3 tests; no real boto3 or IB calls.
"""

from __future__ import annotations

import json
from datetime import UTC
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from executor.intraday_snapshot import (
    HEARTBEAT_KEY,
    LATEST_PRICES_KEY,
    NAV_KEY,
    NAV_SERIES_PREFIX,
    IntradayNavSeriesWriter,
    IntradayNavWriter,
    IntradaySnapshotWriter,
    compute_surveillance_universe,
)

# ── compute_surveillance_universe ──────────────────────────────────────────


class TestComputeSurveillanceUniverse:
    def test_empty_inputs_returns_spy_only(self):
        assert compute_surveillance_universe(None) == ["SPY"]

    def test_universe_only(self):
        sig = {
            "universe": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"ticker": "MSFT", "signal": "EXIT"},
            ],
            "buy_candidates": ["NVDA"],
        }
        assert compute_surveillance_universe(sig) == ["AAPL", "MSFT", "NVDA", "SPY"]

    def test_hold_signals_excluded(self):
        # Regression: config-I3200-adjacent incident (2026-07-20/21) — the
        # weekly-scan population (~900 tickers, virtually all HOLD) was being
        # unioned wholesale into the surveillance universe, blowing through
        # IBKR's concurrent market-data-line cap (error 101 cascade). Only
        # ENTER/EXIT/REDUCE are actionable surveillance signals.
        sig = {
            "universe": [
                {"ticker": "AAPL", "signal": "HOLD"},
                {"ticker": "MSFT", "signal": "ENTER"},
                {"ticker": "GOOG", "signal": "HOLD"},
            ],
        }
        assert compute_surveillance_universe(sig) == ["MSFT", "SPY"]

    def test_signals_entries_missing_signal_field_excluded(self):
        # Defensive: a record with no 'signal' field is treated as non-action,
        # not accidentally included.
        sig = {"universe": [{"ticker": "AAPL"}]}
        assert compute_surveillance_universe(sig) == ["SPY"]

    def test_order_book_only(self):
        result = compute_surveillance_universe(None, order_book_tickers=["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_positions_only(self):
        result = compute_surveillance_universe(None, current_positions=["GOOG"])
        assert result == ["GOOG", "SPY"]

    def test_full_union_dedups_and_sorts(self):
        sig = {
            "universe": [
                {"ticker": "AAPL", "signal": "ENTER"},
                {"ticker": "MSFT", "signal": "EXIT"},
            ],
            "buy_candidates": ["MSFT", "NVDA"],
        }
        result = compute_surveillance_universe(
            sig,
            order_book_tickers=["AAPL", "TSLA"],
            current_positions=["NVDA", "GOOG"],
        )
        # Union: AAPL, MSFT, NVDA, TSLA, GOOG + SPY. Sorted.
        assert result == ["AAPL", "GOOG", "MSFT", "NVDA", "SPY", "TSLA"]

    def test_include_spy_false_omits_spy(self):
        sig = {"universe": [{"ticker": "AAPL", "signal": "ENTER"}]}
        result = compute_surveillance_universe(sig, include_spy=False)
        assert result == ["AAPL"]

    def test_handles_missing_signals_keys(self):
        # signals.json with neither 'signals' nor 'buy_candidates' fields.
        assert compute_surveillance_universe({}) == ["SPY"]

    def test_handles_non_dict_universe_field(self):
        # Defensive: a malformed signals.universe field shouldn't crash.
        sig = {"universe": "not-a-list", "buy_candidates": ["AAPL"]}
        assert compute_surveillance_universe(sig) == ["AAPL", "SPY"]

    def test_handles_non_list_buy_candidates(self):
        sig = {"universe": [{"ticker": "AAPL", "signal": "ENTER"}], "buy_candidates": "not-a-list"}
        assert compute_surveillance_universe(sig) == ["AAPL", "SPY"]

    def test_filters_non_string_buy_candidates(self):
        sig = {"universe": [], "buy_candidates": ["AAPL", None, 123, "MSFT"]}
        result = compute_surveillance_universe(sig)
        assert "AAPL" in result and "MSFT" in result
        assert None not in result and 123 not in result

    def test_filters_empty_string_tickers(self):
        result = compute_surveillance_universe(
            None,
            order_book_tickers=["AAPL", ""],
            current_positions=[""],
        )
        assert "" not in result
        assert "AAPL" in result

    def test_spy_in_signals_no_double(self):
        sig = {"universe": [{"ticker": "SPY", "signal": "ENTER"}, {"ticker": "AAPL", "signal": "HOLD"}]}
        result = compute_surveillance_universe(sig)
        assert result.count("SPY") == 1


# ── IntradaySnapshotWriter ──────────────────────────────────────────────────


@pytest.fixture
def mock_s3():
    return MagicMock()


@pytest.fixture
def writer(mock_s3):
    return IntradaySnapshotWriter(
        bucket="test-bucket",
        daemon_pid=12345,
        s3_client=mock_s3,
    )


class TestIntradaySnapshotWriterHappyPath:
    def test_returns_true_on_success(self, writer, mock_s3):
        result = writer.write(
            prices={"AAPL": {"last": 150.0}},
            ib_connected=True,
            subscribed_tickers=["AAPL"],
        )
        assert result is True

    def test_writes_two_objects(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        assert mock_s3.put_object.call_count == 2

    def test_correct_keys(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        keys = {call.kwargs["Key"] for call in mock_s3.put_object.call_args_list}
        assert keys == {LATEST_PRICES_KEY, HEARTBEAT_KEY}

    def test_correct_bucket(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        for call in mock_s3.put_object.call_args_list:
            assert call.kwargs["Bucket"] == "test-bucket"

    def test_content_type_json(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        for call in mock_s3.put_object.call_args_list:
            assert call.kwargs["ContentType"] == "application/json"

    def test_latest_prices_payload_shape(self, writer, mock_s3):
        writer.write(
            prices={"AAPL": {"last": 150.0, "high": 152.0}},
            ib_connected=True,
            subscribed_tickers=["AAPL", "SPY"],
        )
        prices_call = next(c for c in mock_s3.put_object.call_args_list if c.kwargs["Key"] == LATEST_PRICES_KEY)
        body = json.loads(prices_call.kwargs["Body"].decode("utf-8"))
        assert "timestamp" in body
        assert body["prices"] == {"AAPL": {"last": 150.0, "high": 152.0}}

    def test_heartbeat_payload_shape(self, writer, mock_s3):
        writer.write(
            prices={},
            ib_connected=True,
            subscribed_tickers=["AAPL", "MSFT", "SPY"],
        )
        hb_call = next(c for c in mock_s3.put_object.call_args_list if c.kwargs["Key"] == HEARTBEAT_KEY)
        body = json.loads(hb_call.kwargs["Body"].decode("utf-8"))
        assert body["ib_connected"] is True
        assert body["daemon_pid"] == 12345
        assert body["subscribed_count"] == 3
        assert body["subscribed_tickers"] == ["AAPL", "MSFT", "SPY"]
        assert "timestamp" in body

    def test_ib_disconnected_stamped_on_heartbeat(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=False, subscribed_tickers=[])
        hb_call = next(c for c in mock_s3.put_object.call_args_list if c.kwargs["Key"] == HEARTBEAT_KEY)
        body = json.loads(hb_call.kwargs["Body"].decode("utf-8"))
        assert body["ib_connected"] is False


# ── IntradaySnapshotWriter — failure swallowing ─────────────────────────────


class TestIntradaySnapshotWriterFailureSwallowing:
    def test_s3_client_error_returns_false_no_raise(self, mock_s3):
        mock_s3.put_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
            operation_name="PutObject",
        )
        writer = IntradaySnapshotWriter(
            bucket="test-bucket",
            daemon_pid=1,
            s3_client=mock_s3,
        )
        assert writer.write(prices={}, ib_connected=True, subscribed_tickers=[]) is False

    def test_partial_failure_returns_false(self, mock_s3):
        # First put succeeds, second fails — write returns False overall.
        mock_s3.put_object.side_effect = [
            None,  # first put succeeds
            ClientError(
                error_response={"Error": {"Code": "ServiceUnavailable"}},
                operation_name="PutObject",
            ),
        ]
        writer = IntradaySnapshotWriter(
            bucket="test-bucket",
            daemon_pid=1,
            s3_client=mock_s3,
        )
        assert writer.write(prices={}, ib_connected=True, subscribed_tickers=[]) is False


# ── IntradaySnapshotWriter — daemon_pid default ─────────────────────────────


class TestDaemonPidDefault:
    def test_defaults_to_os_getpid(self, mock_s3):
        writer = IntradaySnapshotWriter(bucket="b", s3_client=mock_s3)
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        hb_call = next(c for c in mock_s3.put_object.call_args_list if c.kwargs["Key"] == HEARTBEAT_KEY)
        body = json.loads(hb_call.kwargs["Body"].decode("utf-8"))
        # Just confirm it's an int — actual value is the test runner's pid.
        assert isinstance(body["daemon_pid"], int)
        assert body["daemon_pid"] > 0


# ── IntradayNavWriter ───────────────────────────────────────────────────────


@pytest.fixture
def nav_writer(mock_s3):
    return IntradayNavWriter(bucket="test-bucket", s3_client=mock_s3)


_ACCT = {
    "net_liquidation": 1_000_564.85,
    "total_cash": 28_634.48,
    "gross_position_value": 971_930.37,
    "unrealized_pnl": -10_465.16,
    "settled_cash": 28_634.48,  # extra field the writer should ignore
}


class TestIntradayNavWriterHappyPath:
    def test_returns_true_and_writes_one_object(self, nav_writer, mock_s3):
        ok = nav_writer.write(_ACCT, spy_last=740.96, ib_connected=True)
        assert ok is True
        assert mock_s3.put_object.call_count == 1

    def test_correct_key_and_bucket(self, nav_writer, mock_s3):
        nav_writer.write(_ACCT, spy_last=740.96, ib_connected=True)
        call = mock_s3.put_object.call_args
        assert call.kwargs["Key"] == NAV_KEY
        assert call.kwargs["Bucket"] == "test-bucket"
        assert call.kwargs["ContentType"] == "application/json"

    def test_payload_shape(self, nav_writer, mock_s3):
        nav_writer.write(_ACCT, spy_last=740.96, ib_connected=True)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        assert body["net_liquidation"] == 1_000_564.85
        assert body["total_cash"] == 28_634.48
        assert body["gross_position_value"] == 971_930.37
        assert body["unrealized_pnl"] == -10_465.16
        assert body["spy_last"] == 740.96
        assert body["ib_connected"] is True
        assert "timestamp" in body
        # Raw marks only — no derived return/alpha leaks into the producer.
        assert "settled_cash" not in body
        assert "daily_return" not in body and "alpha" not in body

    def test_missing_account_fields_become_null(self, nav_writer, mock_s3):
        nav_writer.write({}, spy_last=None, ib_connected=True)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        assert body["net_liquidation"] is None
        assert body["spy_last"] is None

    def test_disconnected_flag_stamped(self, nav_writer, mock_s3):
        nav_writer.write(_ACCT, spy_last=740.96, ib_connected=False)
        body = json.loads(mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8"))
        assert body["ib_connected"] is False


class TestIntradayNavWriterFailureSwallowing:
    def test_s3_error_returns_false_no_raise(self, mock_s3):
        mock_s3.put_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
            operation_name="PutObject",
        )
        writer = IntradayNavWriter(bucket="test-bucket", s3_client=mock_s3)
        assert writer.write(_ACCT, spy_last=740.96, ib_connected=True) is False


# ── IntradayNavSeriesWriter ─────────────────────────────────────────────────


def _get_body(payload: dict) -> dict:
    """Build a mock get_object response wrapping ``payload`` as JSON bytes."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return {"Body": body}


def _no_such_key():
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": "absent"}},
        operation_name="GetObject",
    )


def _written_payload(mock_s3) -> dict:
    return json.loads(mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8"))


@pytest.fixture
def series_writer(mock_s3):
    return IntradayNavSeriesWriter(bucket="test-bucket", s3_client=mock_s3)


@pytest.fixture
def live_session() -> str:
    # The writer's content-vs-key guard (config#1610) refuses a label that
    # doesn't contain the point's real wall-clock timestamp, so write tests
    # must label with the CURRENT session rather than a fixed date.
    from nousergon_lib.dates import session_date

    return session_date().isoformat()


class TestIntradayNavSeriesWriter:
    def test_key_format(self):
        assert IntradayNavSeriesWriter.key_for("2026-06-18") == (NAV_SERIES_PREFIX + "2026-06-18.json")

    def test_first_tick_starts_new_series(self, series_writer, mock_s3, live_session):
        mock_s3.get_object.side_effect = _no_such_key()
        ok = series_writer.write(live_session, _ACCT, spy_last=740.96)
        assert ok is True
        call = mock_s3.put_object.call_args
        assert call.kwargs["Key"] == NAV_SERIES_PREFIX + f"{live_session}.json"
        body = _written_payload(mock_s3)
        assert body["trading_day"] == live_session
        assert body["session_date"] == live_session
        assert len(body["points"]) == 1
        p = body["points"][0]
        assert p["nav"] == _ACCT["net_liquidation"]
        assert p["spy"] == 740.96
        assert "t" in p

    def test_appends_to_existing_series(self, series_writer, mock_s3, live_session):
        prior = {
            "trading_day": live_session,
            "points": [{"t": f"{live_session}T13:45:00Z", "nav": 999_000.0, "spy": 739.0}],
        }
        mock_s3.get_object.return_value = _get_body(prior)
        series_writer.write(live_session, _ACCT, spy_last=740.96)
        body = _written_payload(mock_s3)
        assert len(body["points"]) == 2
        assert body["points"][0]["nav"] == 999_000.0  # prior preserved
        assert body["points"][1]["nav"] == _ACCT["net_liquidation"]  # new appended

    def test_no_nav_skips_write(self, series_writer, mock_s3):
        ok = series_writer.write("2026-06-18", {}, spy_last=740.96)
        assert ok is False
        mock_s3.get_object.assert_not_called()
        mock_s3.put_object.assert_not_called()

    def test_transient_read_error_does_not_clobber(self, series_writer, mock_s3):
        mock_s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "ServiceUnavailable"}},
            operation_name="GetObject",
        )
        ok = series_writer.write("2026-06-18", _ACCT, spy_last=740.96)
        assert ok is False
        # Critically: no PUT — we must not overwrite the day's history with a
        # fresh single-point list on a transient read failure.
        mock_s3.put_object.assert_not_called()

    def test_max_points_trims_oldest(self, mock_s3, live_session):
        writer = IntradayNavSeriesWriter(bucket="test-bucket", s3_client=mock_s3, max_points=3)
        prior = {
            "trading_day": live_session,
            "points": [
                {"t": "t1", "nav": 1.0, "spy": 1.0},
                {"t": "t2", "nav": 2.0, "spy": 2.0},
                {"t": "t3", "nav": 3.0, "spy": 3.0},
            ],
        }
        mock_s3.get_object.return_value = _get_body(prior)
        writer.write(live_session, _ACCT, spy_last=740.96)
        body = _written_payload(mock_s3)
        assert len(body["points"]) == 3
        assert body["points"][0]["nav"] == 2.0  # oldest (t1) dropped
        assert body["points"][-1]["nav"] == _ACCT["net_liquidation"]

    def test_spy_none_stored(self, series_writer, mock_s3, live_session):
        mock_s3.get_object.side_effect = _no_such_key()
        series_writer.write(live_session, _ACCT, spy_last=None)
        body = _written_payload(mock_s3)
        assert body["points"][0]["spy"] is None

    def test_put_failure_returns_false(self, series_writer, mock_s3):
        mock_s3.get_object.side_effect = _no_such_key()
        mock_s3.put_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied"}},
            operation_name="PutObject",
        )
        assert series_writer.write("2026-06-18", _ACCT, spy_last=740.96) is False


class TestNavSeriesSessionRefusalLogging:
    def test_stale_label_logs_error(self, caplog):
        """D-1 label during a live session is a true mis-key — ERROR."""
        import logging
        from datetime import datetime
        from unittest.mock import MagicMock, patch
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        intraday = datetime(2026, 7, 7, 10, 0, tzinfo=ET).astimezone(UTC)
        stale_label = "2026-07-06"  # D-1 relative to the live Tue session

        mock_s3 = MagicMock()
        writer = IntradayNavSeriesWriter(bucket="b", s3_client=mock_s3)
        caplog.set_level(logging.INFO)

        with patch("executor.intraday_snapshot.datetime") as mock_dt:
            real_dt = datetime
            mock_dt.now.return_value = intraday
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            assert writer.write(stale_label, _ACCT, spy_last=740.0) is False

        assert any("nav_series point refused" in r.message for r in caplog.records)
        assert any(r.levelno == logging.ERROR and "nav_series" in r.message for r in caplog.records)
        mock_s3.get_object.assert_not_called()

    def test_post_close_wind_down_logs_info(self, caplog):
        """Frozen run_date after 16:00 ET on session day — INFO, not ERROR."""
        import logging
        from datetime import datetime
        from unittest.mock import MagicMock, patch
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        post_close = datetime(2026, 7, 6, 16, 6, tzinfo=ET).astimezone(UTC)
        labeled = "2026-07-06"

        mock_s3 = MagicMock()
        writer = IntradayNavSeriesWriter(bucket="b", s3_client=mock_s3)
        caplog.set_level(logging.INFO)

        with patch("executor.intraday_snapshot.datetime") as mock_dt:
            real_dt = datetime
            mock_dt.now.return_value = post_close
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            assert writer.write(labeled, _ACCT, spy_last=740.0) is False

        assert any("nav_series point skipped (post-close)" in r.message for r in caplog.records)
        assert not any(r.levelno == logging.ERROR and "nav_series" in r.message for r in caplog.records)
        mock_s3.get_object.assert_not_called()
