"""Tests for executor/intraday_snapshot.py — surveillance universe
computation + S3 snapshot writer (latest_prices + heartbeat).

Pure-logic + mocked-S3 tests; no real boto3 or IB calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from executor.intraday_snapshot import (
    HEARTBEAT_KEY,
    LATEST_PRICES_KEY,
    NAV_KEY,
    IntradayNavWriter,
    IntradaySnapshotWriter,
    compute_surveillance_universe,
)


# ── compute_surveillance_universe ──────────────────────────────────────────


class TestComputeSurveillanceUniverse:
    def test_empty_inputs_returns_spy_only(self):
        assert compute_surveillance_universe(None) == ["SPY"]

    def test_signals_only(self):
        sig = {"signals": {"AAPL": {}, "MSFT": {}}, "buy_candidates": ["NVDA"]}
        assert compute_surveillance_universe(sig) == ["AAPL", "MSFT", "NVDA", "SPY"]

    def test_order_book_only(self):
        result = compute_surveillance_universe(None, order_book_tickers=["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT", "SPY"]

    def test_positions_only(self):
        result = compute_surveillance_universe(None, current_positions=["GOOG"])
        assert result == ["GOOG", "SPY"]

    def test_full_union_dedups_and_sorts(self):
        sig = {"signals": {"AAPL": {}, "MSFT": {}}, "buy_candidates": ["MSFT", "NVDA"]}
        result = compute_surveillance_universe(
            sig,
            order_book_tickers=["AAPL", "TSLA"],
            current_positions=["NVDA", "GOOG"],
        )
        # Union: AAPL, MSFT, NVDA, TSLA, GOOG + SPY. Sorted.
        assert result == ["AAPL", "GOOG", "MSFT", "NVDA", "SPY", "TSLA"]

    def test_include_spy_false_omits_spy(self):
        sig = {"signals": {"AAPL": {}}}
        result = compute_surveillance_universe(sig, include_spy=False)
        assert result == ["AAPL"]

    def test_handles_missing_signals_keys(self):
        # signals.json with neither 'signals' nor 'buy_candidates' fields.
        assert compute_surveillance_universe({}) == ["SPY"]

    def test_handles_non_dict_signals_field(self):
        # Defensive: a malformed signals.signals field shouldn't crash.
        sig = {"signals": "not-a-dict", "buy_candidates": ["AAPL"]}
        assert compute_surveillance_universe(sig) == ["AAPL", "SPY"]

    def test_handles_non_list_buy_candidates(self):
        sig = {"signals": {"AAPL": {}}, "buy_candidates": "not-a-list"}
        assert compute_surveillance_universe(sig) == ["AAPL", "SPY"]

    def test_filters_non_string_buy_candidates(self):
        sig = {"signals": {}, "buy_candidates": ["AAPL", None, 123, "MSFT"]}
        result = compute_surveillance_universe(sig)
        assert "AAPL" in result and "MSFT" in result
        assert None not in result and 123 not in result

    def test_filters_empty_string_tickers(self):
        result = compute_surveillance_universe(
            None, order_book_tickers=["AAPL", ""], current_positions=[""],
        )
        assert "" not in result
        assert "AAPL" in result

    def test_spy_in_signals_no_double(self):
        sig = {"signals": {"SPY": {}, "AAPL": {}}}
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
        prices_call = next(
            c for c in mock_s3.put_object.call_args_list
            if c.kwargs["Key"] == LATEST_PRICES_KEY
        )
        body = json.loads(prices_call.kwargs["Body"].decode("utf-8"))
        assert "timestamp" in body
        assert body["prices"] == {"AAPL": {"last": 150.0, "high": 152.0}}

    def test_heartbeat_payload_shape(self, writer, mock_s3):
        writer.write(
            prices={},
            ib_connected=True,
            subscribed_tickers=["AAPL", "MSFT", "SPY"],
        )
        hb_call = next(
            c for c in mock_s3.put_object.call_args_list
            if c.kwargs["Key"] == HEARTBEAT_KEY
        )
        body = json.loads(hb_call.kwargs["Body"].decode("utf-8"))
        assert body["ib_connected"] is True
        assert body["daemon_pid"] == 12345
        assert body["subscribed_count"] == 3
        assert body["subscribed_tickers"] == ["AAPL", "MSFT", "SPY"]
        assert "timestamp" in body

    def test_ib_disconnected_stamped_on_heartbeat(self, writer, mock_s3):
        writer.write(prices={}, ib_connected=False, subscribed_tickers=[])
        hb_call = next(
            c for c in mock_s3.put_object.call_args_list
            if c.kwargs["Key"] == HEARTBEAT_KEY
        )
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
            bucket="test-bucket", daemon_pid=1, s3_client=mock_s3,
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
            bucket="test-bucket", daemon_pid=1, s3_client=mock_s3,
        )
        assert writer.write(prices={}, ib_connected=True, subscribed_tickers=[]) is False


# ── IntradaySnapshotWriter — daemon_pid default ─────────────────────────────


class TestDaemonPidDefault:
    def test_defaults_to_os_getpid(self, mock_s3):
        writer = IntradaySnapshotWriter(bucket="b", s3_client=mock_s3)
        writer.write(prices={}, ib_connected=True, subscribed_tickers=[])
        hb_call = next(
            c for c in mock_s3.put_object.call_args_list
            if c.kwargs["Key"] == HEARTBEAT_KEY
        )
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
