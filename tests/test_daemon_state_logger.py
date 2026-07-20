"""Tests for ``executor/daemon_state_logger.py`` (ROADMAP L139a substrate)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from executor.daemon_state_logger import (
    DaemonDecisionLogger,
    get_logger,
    reset_singleton_for_tests,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """Each test gets a fresh module-level singleton."""
    reset_singleton_for_tests()
    yield
    reset_singleton_for_tests()


# ── record() ──────────────────────────────────────────────────────────────


class TestRecord:
    def test_empty_buffer_initially(self):
        log = DaemonDecisionLogger()
        assert len(log) == 0
        assert log.snapshot() == []

    def test_record_appends_entry_with_required_fields(self):
        log = DaemonDecisionLogger()
        log.record(
            decision_type="intraday_exit",
            ticker="AAPL",
            action="EXIT",
            trading_day="2026-05-22",
            shares=100,
            fill_price=150.25,
        )
        assert len(log) == 1
        entry = log.snapshot()[0]
        assert entry["decision_type"] == "intraday_exit"
        assert entry["ticker"] == "AAPL"
        assert entry["action"] == "EXIT"
        assert entry["trading_day"] == "2026-05-22"
        assert entry["shares"] == 100
        assert entry["fill_price"] == 150.25
        # Auto-stamped timestamp
        assert "timestamp_utc" in entry
        assert entry["timestamp_utc"].endswith("Z")

    def test_record_admits_arbitrary_context_kwargs(self):
        """**context fan-out — decision-type-specific extras land in
        the entry without schema-level pre-declaration."""
        log = DaemonDecisionLogger()
        log.record(
            decision_type="urgent_exit",
            ticker="PFE",
            action="REDUCE",
            trading_day="2026-05-22",
            trigger_reason="research_signal",
            retry_count=2,
            attempts=[{"attempt": 1, "status": "Timeout"}],
            context={"research_score": 72.0, "sector": "Healthcare"},
        )
        entry = log.snapshot()[0]
        assert entry["trigger_reason"] == "research_signal"
        assert entry["retry_count"] == 2
        assert entry["attempts"] == [{"attempt": 1, "status": "Timeout"}]
        assert entry["context"]["research_score"] == 72.0

    def test_record_never_raises_on_bad_input(self):
        """Secondary observability — record() must never block the
        daemon's primary order-execution path on schema mishaps.
        """
        log = DaemonDecisionLogger()
        # decision_type passed as a non-serializable object — record
        # swallows the exception and the buffer remains empty (or
        # whatever fallback the implementation chose).
        class _Unserializable:
            def __repr__(self):
                raise RuntimeError("kaboom")

        # Passing should not raise even with a weird value.
        try:
            log.record(
                decision_type="intraday_exit",
                ticker="AAPL",
                action="EXIT",
                trading_day="2026-05-22",
                weird_field=_Unserializable(),
            )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"record() should swallow exceptions, raised: {exc}")


# ── thread-safety ─────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_records_do_not_lose_entries(self):
        """Daemon has multiple call paths (Phase-0 sync, intraday
        monitor, entry-trigger callbacks). Lock-protected buffer
        guarantees no entries are dropped under concurrent record().
        """
        import threading

        log = DaemonDecisionLogger()
        n_threads = 8
        n_per = 50

        def _runner():
            for i in range(n_per):
                log.record(
                    decision_type="intraday_exit",
                    ticker=f"T{i}",
                    action="EXIT",
                    trading_day="2026-05-22",
                    shares=i,
                )

        threads = [threading.Thread(target=_runner) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(log) == n_threads * n_per


# ── flush_to_s3() ─────────────────────────────────────────────────────────


class TestFlushToS3:
    def _stub_s3_get_404(self, client):
        """Default: simulate first-write (NoSuchKey) by raising."""
        client.get_object.side_effect = Exception("NoSuchKey")

    def test_empty_buffer_is_no_op_returns_true(self):
        log = DaemonDecisionLogger()
        client = MagicMock()
        assert log.flush_to_s3("bucket", "2026-05-22", s3_client=client) is True
        client.put_object.assert_not_called()
        client.get_object.assert_not_called()

    def test_first_write_uploads_jsonl_to_canonical_key(self):
        log = DaemonDecisionLogger()
        log.record(
            decision_type="intraday_exit",
            ticker="AAPL",
            action="EXIT",
            trading_day="2026-05-22",
            shares=100,
        )
        client = MagicMock()
        self._stub_s3_get_404(client)
        ok = log.flush_to_s3("alpha-engine-research", "2026-05-22", s3_client=client)
        assert ok is True

        client.put_object.assert_called_once()
        call_kwargs = client.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "alpha-engine-research"
        assert call_kwargs["Key"] == "daemon_state/2026-05-22/intraday_decisions.jsonl"
        assert call_kwargs["ContentType"] == "application/x-ndjson"

        # Body is JSONL — each non-empty line parses to the same dict
        body = call_kwargs["Body"]
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        lines = [json.loads(ln) for ln in body.strip().split("\n") if ln]
        assert len(lines) == 1
        assert lines[0]["ticker"] == "AAPL"
        assert lines[0]["action"] == "EXIT"

    def test_append_preserves_prior_jsonl(self):
        """fix-and-rerun on the same trading_day must append, not
        overwrite. Tests this by stubbing get_object to return existing
        JSONL and verifying the put_object body carries both.
        """
        log = DaemonDecisionLogger()
        log.record(
            decision_type="entry_trigger",
            ticker="NVDA",
            action="ENTER",
            trading_day="2026-05-22",
            shares=50,
        )
        client = MagicMock()
        prior_jsonl = (
            json.dumps({
                "timestamp_utc": "2026-05-22T13:00:00Z",
                "trading_day": "2026-05-22",
                "decision_type": "intraday_exit",
                "ticker": "AAPL",
                "action": "EXIT",
            }) + "\n"
        ).encode("utf-8")
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: prior_jsonl)
        }
        ok = log.flush_to_s3("bucket", "2026-05-22", s3_client=client)
        assert ok is True

        call_kwargs = client.put_object.call_args.kwargs
        body = call_kwargs["Body"]
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        lines = [json.loads(ln) for ln in body.strip().split("\n") if ln]
        assert len(lines) == 2
        # Prior entry first, new entry second
        assert lines[0]["ticker"] == "AAPL"
        assert lines[1]["ticker"] == "NVDA"

    def test_flush_failure_returns_false_does_not_raise(self):
        """S3 outage must not crash the daemon's finally block."""
        log = DaemonDecisionLogger()
        log.record(
            decision_type="intraday_exit",
            ticker="AAPL",
            action="EXIT",
            trading_day="2026-05-22",
            shares=100,
        )
        client = MagicMock()
        client.put_object.side_effect = RuntimeError("S3 down")
        ok = log.flush_to_s3("bucket", "2026-05-22", s3_client=client)
        assert ok is False  # signaled the failure, didn't raise


# ── module-level singleton ────────────────────────────────────────────────


class TestSingleton:
    def test_get_logger_returns_same_instance(self):
        a = get_logger()
        b = get_logger()
        assert a is b

    def test_reset_clears_singleton(self):
        a = get_logger()
        reset_singleton_for_tests()
        b = get_logger()
        assert a is not b


# ── source-inspection pins on daemon.py call sites ────────────────────────


class TestDaemonCallSites:
    """Pin that each of the 3 daemon decision call sites records to the
    decision logger. Mirrors the L165 / L171 / L133 source-inspection
    pattern — full-loop tests require IB Gateway mock harness; source
    inspection guards against a future refactor dropping the record()
    call.
    """

    def test_daemon_records_urgent_exits(self):
        import inspect

        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # Phase 0 urgent-exits loop calls record(decision_type="urgent_exit"
        # or "phase0_auto_cover" for COVER). Both spellings must be live.
        assert '"urgent_exit"' in src and '"phase0_auto_cover"' in src, (
            "L139a Phase-0 urgent-exits capture dropped — both regular "
            "urgent_exit + auto-cover variants must record."
        )

    def test_daemon_records_intraday_exits(self):
        import inspect

        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # _execute_exit must record decision_type="intraday_exit" so
        # ATR-trail / time-decay / profit-take exits land in the
        # replay artifact.
        assert '"intraday_exit"' in src, (
            "L139a intraday-exit capture dropped — backtester replay "
            "loses visibility into ATR/time-decay/profit-take exits."
        )

    def test_daemon_records_entry_triggers(self):
        import inspect

        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        # _execute_entry must record decision_type="entry_trigger" so
        # VWAP/pullback/support/time-expiry ENTERs land in replay.
        assert '"entry_trigger"' in src, (
            "L139a entry-trigger capture dropped — backtester replay "
            "loses visibility into intraday ENTER decisions."
        )

    def test_daemon_finally_flushes_to_s3(self):
        import inspect

        import executor.daemon as daemon

        src = inspect.getsource(daemon)
        assert "flush_to_s3" in src, (
            "L139a daemon shutdown flush dropped — captured decisions "
            "would stay in-memory and be lost on daemon exit."
        )
