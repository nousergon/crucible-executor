"""Unit tests for daemon helpers extracted in Phase 1-2: validate, retry, cleanup."""
import pytest
from unittest.mock import MagicMock, patch

from executor.daemon import (
    _validate_sell_shares,
    _cleanup_connections,
    _enqueue_cover_for_unintended_shorts,
    _place_order_with_retry,
    MAX_ORDER_RETRIES,
    ORDER_RETRY_DELAYS,
)
from executor.order_book import OrderBook, _default_book


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

    def test_caps_against_in_flight_pending_sells(self):
        # PFE incident 2026-04-22: retry loop issued three duplicate SELL 77s
        # that each individually passed held=155, summing to 231 → short 76.
        # With in-flight tracking, the second + third retries see the first
        # order's 77 remaining and refuse.
        positions = {"PFE": {"shares": 155}}
        result = _validate_sell_shares(
            positions, "PFE", 77, "REDUCE", "URGENT",
            pending_sell_shares=77,
        )
        assert result == 77  # 155 - 77 = 78 available; requested 77 fits
        result = _validate_sell_shares(
            positions, "PFE", 77, "REDUCE", "URGENT",
            pending_sell_shares=154,
        )
        assert result == 1  # 155 - 154 = 1 available; cap to 1
        result = _validate_sell_shares(
            positions, "PFE", 77, "REDUCE", "URGENT",
            pending_sell_shares=155,
        )
        assert result is None  # no capacity — refuse


# ── _enqueue_cover_for_unintended_shorts ─────────────────────────────────────


class TestEnqueueCoverForUnintendedShorts:
    def _fresh_book(self, tmp_path):
        return OrderBook(_default_book("2026-04-22"), path=tmp_path / "ob.json")

    def test_enqueues_cover_for_negative_position(self, tmp_path):
        book = self._fresh_book(tmp_path)
        positions = {"PFE": {"shares": -76}, "AAPL": {"shares": 100}}
        covered = _enqueue_cover_for_unintended_shorts(positions, book, "2026-04-22")
        assert covered == ["PFE"]
        pending = book.pending_urgent_exits()
        assert len(pending) == 1
        assert pending[0]["ticker"] == "PFE"
        assert pending[0]["signal"] == "COVER"
        assert pending[0]["shares"] == 76
        assert pending[0]["reason"] == "auto_cover_unintended_short"

    def test_skips_long_and_flat_positions(self, tmp_path):
        book = self._fresh_book(tmp_path)
        positions = {"AAPL": {"shares": 100}, "MSFT": {"shares": 0}}
        covered = _enqueue_cover_for_unintended_shorts(positions, book, "2026-04-22")
        assert covered == []
        assert book.pending_urgent_exits() == []

    def test_bypasses_when_allow_shorts_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr("executor.daemon._allow_shorts", True)
        book = self._fresh_book(tmp_path)
        positions = {"PFE": {"shares": -76}}
        covered = _enqueue_cover_for_unintended_shorts(positions, book, "2026-04-22")
        assert covered == []
        assert book.pending_urgent_exits() == []

    def test_dedup_safe_on_repeat_call(self, tmp_path):
        book = self._fresh_book(tmp_path)
        positions = {"PFE": {"shares": -76}}
        _enqueue_cover_for_unintended_shorts(positions, book, "2026-04-22")
        _enqueue_cover_for_unintended_shorts(positions, book, "2026-04-22")
        # OrderBook.add_urgent_exit dedupes by ticker+signal
        assert len(book.pending_urgent_exits()) == 1


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

    @patch("executor.daemon._time.sleep")
    def test_working_status_does_not_retry(self, mock_sleep):
        # PreSubmitted/Submitted → "Working" from place_market_order. The
        # order is live at the broker; retrying would duplicate it (PFE
        # incident 2026-04-22).
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {"status": "Working", "ib_order_id": 287}
        result = _place_order_with_retry(ibkr, "PFE", "SELL", 77, "urgent")
        assert result["status"] == "Working"
        assert ibkr.place_market_order.call_count == 1
        ibkr.cancel_order.assert_not_called()
        mock_sleep.assert_not_called()

    @patch("executor.daemon._time.sleep")
    def test_timeout_cancels_prior_order_before_retry(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.side_effect = [
            {"status": "Timeout", "ib_order_id": 287},
            {"status": "Filled", "ib_order_id": 289},
        ]
        result = _place_order_with_retry(ibkr, "PFE", "SELL", 77, "urgent")
        assert result["status"] == "Filled"
        assert ibkr.place_market_order.call_count == 2
        ibkr.cancel_order.assert_called_once_with(287)

    @patch("executor.daemon._time.sleep")
    def test_rejected_does_not_cancel(self, mock_sleep):
        # Rejected means IB already terminated the order (Cancelled / Inactive
        # / ApiCancelled); cancelling again is a no-op. Skip the call to keep
        # retry fast and avoid log noise.
        ibkr = MagicMock()
        ibkr.place_market_order.side_effect = [
            {"status": "Rejected", "ib_order_id": 287},
            {"status": "Filled", "ib_order_id": 289},
        ]
        _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "urgent")
        ibkr.cancel_order.assert_not_called()

    @patch("executor.daemon._time.sleep")
    def test_cancel_order_failure_does_not_abort_retry(self, mock_sleep):
        # If cancel_order raises, we still want to proceed with the retry —
        # the prior order may have already terminated at IB side.
        ibkr = MagicMock()
        ibkr.cancel_order.side_effect = RuntimeError("ib unreachable")
        ibkr.place_market_order.side_effect = [
            {"status": "Timeout", "ib_order_id": 287},
            {"status": "Filled", "ib_order_id": 289},
        ]
        result = _place_order_with_retry(ibkr, "PFE", "SELL", 77, "urgent")
        assert result["status"] == "Filled"
        assert ibkr.place_market_order.call_count == 2


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


# ── _trigger_eod_pipeline ────────────────────────────────────────────────────
# Daemon shutdown is the canonical trigger for the alpha-engine-eod-pipeline
# Step Function. PR #94 (2026-04-22) removed this trigger and kept the
# systemd timer; PR #117 (2026-04-28) retired the systemd timer; this test
# locks the trigger code in place so a future "let's clean this up" doesn't
# re-introduce the SF-orphaned regression that lasted from 4/22 to 4/28.


class TestTriggerEodPipeline:
    def test_calls_step_functions_start_execution(self):
        from executor.daemon import _trigger_eod_pipeline

        with patch("boto3.client") as mock_boto:
            sfn = MagicMock()
            mock_boto.return_value = sfn
            _trigger_eod_pipeline(
                {"sns_topic_arn": "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"},
                "2026-04-29",
            )
            sfn.start_execution.assert_called_once()
            kwargs = sfn.start_execution.call_args.kwargs
            assert "alpha-engine-eod-pipeline" in kwargs["stateMachineArn"]
            assert kwargs["name"].startswith("eod-2026-04-29-")

    def test_input_payload_shape(self):
        """Input must include trading_instance_id (array — PostMarketData,
        CaptureSnapshot, EODReconcile, StopTradingInstance target it),
        ec2_instance_id (array — DailySubstrateHealthCheck targets the
        dashboard EC2), and sns_topic_arn (HandleFailure publish target)."""
        import json

        from executor.daemon import _trigger_eod_pipeline

        with patch("boto3.client") as mock_boto:
            sfn = MagicMock()
            mock_boto.return_value = sfn
            _trigger_eod_pipeline(
                {"sns_topic_arn": "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"},
                "2026-04-29",
            )
            payload = json.loads(sfn.start_execution.call_args.kwargs["input"])
            assert isinstance(payload["trading_instance_id"], list)
            assert payload["trading_instance_id"][0].startswith("i-")
            assert isinstance(payload["ec2_instance_id"], list)
            assert payload["ec2_instance_id"][0].startswith("i-")
            assert payload["ec2_instance_id"][0] != payload["trading_instance_id"][0], (
                "trading EC2 and dashboard EC2 must be distinct instances"
            )
            assert payload["sns_topic_arn"].startswith("arn:aws:sns:")
            assert payload["run_date"] == "2026-04-29"
            assert payload["triggered_by"] == "daemon_shutdown"

    def test_failure_is_non_fatal(self):
        """Daemon's finally block must not crash if SF start fails. The
        trigger logs a WARN and returns; SF-side failures surface via SNS
        HandleFailure and flow-doctor when the SF eventually fires."""
        from executor.daemon import _trigger_eod_pipeline

        with patch("boto3.client") as mock_boto:
            sfn = MagicMock()
            sfn.start_execution.side_effect = RuntimeError("transient IAM hiccup")
            mock_boto.return_value = sfn
            # Must NOT raise — the daemon shutdown path can't tolerate an
            # exception out of this trigger.
            _trigger_eod_pipeline({}, "2026-04-29")

    def test_uses_default_sns_topic_when_config_missing(self):
        """`_trigger_eod_pipeline({}, ...)` must still produce a valid
        input — the default SNS topic ARN is hardcoded as fallback."""
        import json

        from executor.daemon import _trigger_eod_pipeline

        with patch("boto3.client") as mock_boto:
            sfn = MagicMock()
            mock_boto.return_value = sfn
            _trigger_eod_pipeline({}, "2026-04-29")
            payload = json.loads(sfn.start_execution.call_args.kwargs["input"])
            assert "alpha-engine-alerts" in payload["sns_topic_arn"]


# ── L165: urgent-exits action label semantic correctness ─────────────────


class TestPhase0UrgentExitsActionLabel:
    """Pin the L165 fix (2026-05-22): the urgent-exits loop must pass
    the SEMANTIC action ("EXIT" / "REDUCE" / "COVER") to
    send_trade_alert, NOT the IB side ("SELL" / "BUY").

    Failure mode this catches: morning urgent REDUCEs surface in
    Telegram as "SELL" (because the daemon previously passed `side`
    instead of `action`), while page 16 / OBR shows the real action.
    Asymmetric with the intraday `_execute_exit` path that already
    passes `action=action` correctly. Source-inspection pin so a
    future refactor that drops the fix breaks at CI time, not after
    the next divergence incident.
    """

    def test_phase0_urgent_exit_passes_semantic_action_not_side(self):
        import inspect
        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # The Phase-0 urgent-exits loop must build `send_trade_alert`
        # with `action=action`, not `action=side`. The exact spelling
        # is asserted because that's the single-line bug class the L165
        # fix closes — any future contributor copy-pasting from a
        # pre-fix snippet falls into this pin.
        assert "action=side" not in src, (
            "daemon.py contains `action=side` — the L165 bug class is "
            "back. Pass the semantic action label ('EXIT'/'REDUCE'/"
            "'COVER') to send_trade_alert, not the IB side."
        )
        # And the urgent-exits loop's call must still use action= keyword.
        # Pin the specific surrounding context so we'd notice a future
        # refactor that moves the call to a helper without preserving
        # the label.
        assert 'trigger=f"urgent_{reason}"' in src, (
            "the Phase-0 urgent-exits send_trade_alert call appears "
            "to have been refactored — re-audit the new call site for "
            "the L165 action-label correctness before merging."
        )


# ── L133: retry-tracking audit trail on `_place_order_with_retry` ──────────


class TestPlaceOrderWithRetryAttemptsAudit:
    """Pin the L133 (2026-05-22) addition — `_place_order_with_retry` now
    returns an `attempts` list + `retry_count` so callers can embed the
    audit chain in trades.db `rationale_json`. Closes the home-endnote
    gap PR #100 had to drop the "retry" qualifier from.
    """

    def test_first_attempt_success_yields_single_entry_zero_retries(self):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {
            "status": "Filled", "ib_order_id": 100,
        }
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        assert result["retry_count"] == 0
        assert len(result["attempts"]) == 1
        first = result["attempts"][0]
        assert first["attempt"] == 1
        assert first["status"] == "Filled"
        assert first["ib_order_id"] == 100
        assert first["retry_reason"] is None  # first attempt — no prior to retry

    @patch("executor.daemon._time.sleep")
    def test_retry_count_and_reasons_captured_for_each_failed_attempt(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.side_effect = [
            {"status": "Timeout", "ib_order_id": 287},
            {"status": "Rejected", "ib_order_id": 288},
            {"status": "Filled", "ib_order_id": 289},
        ]
        result = _place_order_with_retry(ibkr, "PFE", "SELL", 77, "urgent")
        assert result["status"] == "Filled"
        assert result["retry_count"] == 2
        assert len(result["attempts"]) == 3
        assert result["attempts"][0]["status"] == "Timeout"
        assert result["attempts"][0]["retry_reason"] is None
        assert result["attempts"][1]["status"] == "Rejected"
        assert result["attempts"][1]["retry_reason"] == "Timeout"  # why we retried
        assert result["attempts"][2]["status"] == "Filled"
        assert result["attempts"][2]["retry_reason"] == "Rejected"

    @patch("executor.daemon._time.sleep")
    def test_all_retries_failed_still_records_full_audit(self, mock_sleep):
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {
            "status": "Rejected", "ib_order_id": 300,
        }
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "exit")
        # MAX_ORDER_RETRIES attempts; all failed; result still carries
        # the full audit trail so trades.db rationale_json captures
        # "we tried 3 times and IB rejected every one."
        assert result["status"] == "Rejected"
        assert result["retry_count"] == MAX_ORDER_RETRIES - 1
        assert len(result["attempts"]) == MAX_ORDER_RETRIES
        for entry in result["attempts"]:
            assert entry["status"] == "Rejected"

    @patch("executor.daemon._time.sleep")
    def test_working_status_records_single_attempt_no_retry(self, mock_sleep):
        """Working status holds — we DON'T retry (PFE incident protection).
        Audit trail captures the single Working attempt.
        """
        ibkr = MagicMock()
        ibkr.place_market_order.return_value = {
            "status": "Working", "ib_order_id": 400,
        }
        result = _place_order_with_retry(ibkr, "AAPL", "SELL", 10, "urgent")
        assert result["status"] == "Working"
        assert result["retry_count"] == 0
        assert len(result["attempts"]) == 1
        assert result["attempts"][0]["status"] == "Working"


class TestRationaleJsonEnrichmentSourceShape:
    """Source-inspection pin on the L133 enrichment of the 3 daemon
    log_trade call sites' rationale_json. Each site MUST include the
    signal_context block + the retry audit. Full-loop tests require IB
    Gateway harness; this pin guards against a future refactor that
    drops the enrichment.
    """

    def test_urgent_exit_rationale_carries_signal_context_and_retry_audit(self):
        import inspect
        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # The Phase 0 urgent-exits log_trade call.
        assert '"phase": "urgent"' in src, "Phase-0 urgent-exit rationale shape changed"
        # signal_context block on the urgent path.
        assert '"signal_context"' in src, (
            "L133 enrichment dropped — log_trade rationale_json must "
            "carry a signal_context block on every trade pathway."
        )
        # retry audit on every log_trade rationale.
        assert '"retry_count":' in src and '"attempts":' in src, (
            "L133 retry audit dropped — log_trade rationale_json must "
            "carry retry_count + attempts on every trade pathway."
        )

    def test_enter_rationale_carries_signal_context_and_retry_audit(self):
        import inspect
        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # The intraday ENTER log_trade call (PFE incident-class fields:
        # trigger_reason + sizing_factors + predicted_alpha must remain).
        assert '"trigger_reason":' in src
        assert '"sizing_factors":' in src
        # Pin: the rationale must reference position_pct inside the
        # signal_context block (not just at the trade dict's top level).
        assert '"position_pct":' in src, (
            "L133 ENTER rationale must include position_pct in "
            "signal_context so the audit log captures the sizing math."
        )
