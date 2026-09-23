"""Tests for executor.eod_reconcile_standalone — component 3's own schedule
for `eod_reconcile.run()` itself (alpha-engine-config-I11066).

The whole point of this entrypoint is that it does not depend on the v1
postclose pipeline having fired: it reuses today's snapshot if one already
exists (the common case during the v1/v2 coexistence window — the v1 SF's
own CaptureSnapshot step still writes it), and captures its own if not (the
case once v1 is disabled). `eod_reconcile.run()` itself is never modified —
the v1 SF's `EODReconcile` step keeps calling it exactly as it does today.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executor import eod_reconcile_standalone as standalone

_REAL_WAIT = standalone._wait_for_settled_closes


@pytest.fixture(autouse=True)
def _closes_already_landed():
    """Every run() test starts after the post-close append; the wait itself
    is covered by TestWaitForSettledCloses through _REAL_WAIT."""
    with patch.object(standalone, "_wait_for_settled_closes"):
        yield


def _config():
    return {"trades_bucket": "alpha-engine-research", "aws_region": "us-east-1"}


def _snapshot():
    return {
        "run_date": "2026-06-18",
        "account": {"net_liquidation": 1_001_593.11},
        "positions": {},
        "accrued_dividends": {},
    }


class TestRunDate:
    def test_default_resolves_to_now_dual_trading_day(self):
        with patch("executor.eod_reconcile_standalone.now_dual") as mock_now_dual, \
             patch("executor.eod_reconcile_standalone.load_config") as mock_cfg:
            mock_now_dual.return_value = SimpleNamespace(trading_day="2026-06-18")
            mock_cfg.side_effect = RuntimeError("expected_test_sentinel")
            with pytest.raises(RuntimeError, match="expected_test_sentinel"):
                standalone.run(run_date=None)
            mock_now_dual.assert_called_once()


class TestSnapshotReuseVsSelfCapture:
    def test_reuses_existing_snapshot_without_self_capturing(self):
        """The common case during v1/v2 coexistence: a snapshot already
        exists (written by the v1 SF's CaptureSnapshot step), so this must
        not open a second IB session before delegating to eod_reconcile.run()."""
        with patch("executor.eod_reconcile_standalone.load_config", return_value=_config()), \
             patch.object(standalone.snapshot_capturer, "load_snapshot", return_value=_snapshot()) as mock_load, \
             patch.object(standalone.snapshot_capturer, "run") as mock_capture, \
             patch.object(standalone.eod_reconcile, "run") as mock_reconcile_run:
            standalone.run(run_date="2026-06-18")

        mock_capture.assert_not_called()
        mock_load.assert_called_once_with(bucket="alpha-engine-research", run_date="2026-06-18", region="us-east-1")
        mock_reconcile_run.assert_called_once_with(run_date="2026-06-18", send_email=True, run_audit=True)

    def test_self_captures_when_no_snapshot_exists(self):
        """No snapshot yet (the case once v1 is disabled, or this timer
        firing ahead of the v1 pipeline on a given day) — self-sufficient:
        captures one directly rather than raising or waiting on v1."""
        with patch("executor.eod_reconcile_standalone.load_config", return_value=_config()), \
             patch.object(standalone.snapshot_capturer, "load_snapshot", return_value=None) as mock_load, \
             patch.object(standalone.snapshot_capturer, "run") as mock_capture, \
             patch.object(standalone.eod_reconcile, "run") as mock_reconcile_run:
            standalone.run(run_date="2026-06-18")

        mock_capture.assert_called_once_with("2026-06-18")
        mock_load.assert_called_once_with(bucket="alpha-engine-research", run_date="2026-06-18", region="us-east-1")
        mock_reconcile_run.assert_called_once_with(run_date="2026-06-18", send_email=True, run_audit=True)

    def test_eod_reconcile_run_failure_propagates(self):
        """Fail loud (fleet default: RAISE) — this is the PRIMARY producer
        of trades/eod_pnl.csv etc., not secondary observability; a failure
        must never be swallowed."""
        with patch("executor.eod_reconcile_standalone.load_config", return_value=_config()), \
             patch.object(standalone.snapshot_capturer, "load_snapshot", return_value=_snapshot()), \
             patch.object(standalone.snapshot_capturer, "run"), \
             patch.object(standalone.eod_reconcile, "run", side_effect=RuntimeError("no snapshot is readable back")):
            with pytest.raises(RuntimeError, match="no snapshot is readable back"):
                standalone.run(run_date="2026-06-18")

    def test_does_not_modify_eod_reconcile_module(self):
        """This entrypoint must never monkeypatch or otherwise mutate
        executor.eod_reconcile — the v1 SF's EODReconcile SSM step keeps
        calling `python -m executor.eod_reconcile` unmodified."""
        import inspect

        from executor import eod_reconcile

        assert "def run(" in inspect.getsource(eod_reconcile.run)


class _FakeS3:
    """get_object over a list of (macro, universe) sentinel pairs, one pair
    per poll; None means the key does not exist yet."""

    class exceptions:  # noqa: N801 — mirrors boto3's client.exceptions
        class NoSuchKey(Exception):
            pass

    def __init__(self, polls):
        self.polls = list(polls)
        self.reads = 0

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        pair = self.polls[min(self.reads // 2, len(self.polls) - 1)]
        doc = pair[0] if Key == standalone._MACRO_SENTINEL_KEY else pair[1]
        self.reads += 1
        if doc is None:
            raise self.exceptions.NoSuchKey()
        import io
        import json as _json
        return {"Body": io.BytesIO(_json.dumps(doc).encode())}


def _macro(day):
    return {"run_date": day, "verified_keys": ["SPY", "VIX"]}


def _universe(day, n=909):
    return {"run_date": day, "verified_ticker_count": n}


class TestWaitForSettledCloses:
    def _wait(self, s3, *, timeout_s=300):
        t = {"now": 0.0}
        sleeps = []

        def sleep(sec):
            sleeps.append(sec)
            t["now"] += sec

        _REAL_WAIT(
            "b", "2026-09-22", "us-east-1", timeout_s=timeout_s, poll_s=60,
            s3=s3, sleep=sleep, clock=lambda: t["now"],
        )
        return sleeps

    def test_returns_at_once_when_both_sentinels_are_current(self):
        assert self._wait(_FakeS3([(_macro("2026-09-22"), _universe("2026-09-22"))])) == []

    def test_waits_through_the_append_then_returns(self):
        """2026-09-22 as measured: at 21:05 both sentinels still carried the
        prior day; the universe sentinel landed a few minutes later."""
        s3 = _FakeS3([
            (_macro("2026-09-21"), _universe("2026-09-21")),
            (_macro("2026-09-22"), _universe("2026-09-21")),
            (_macro("2026-09-22"), _universe("2026-09-22")),
        ])
        assert self._wait(s3) == [60, 60]

    def test_prior_day_sentinel_never_counts(self):
        with pytest.raises(RuntimeError, match="has not landed closes for 2026-09-22"):
            self._wait(_FakeS3([(_macro("2026-09-21"), _universe("2026-09-21"))]))

    def test_missing_sentinel_or_zero_count_keeps_waiting(self):
        with pytest.raises(RuntimeError):
            self._wait(_FakeS3([(None, _universe("2026-09-22"))]))
        with pytest.raises(RuntimeError):
            self._wait(_FakeS3([(_macro("2026-09-22"), _universe("2026-09-22", n=0))]))

    def test_run_waits_before_reconciling(self):
        calls = []
        with patch("executor.eod_reconcile_standalone.load_config", return_value=_config()), \
             patch.object(standalone, "_wait_for_settled_closes",
                          side_effect=lambda *a, **k: calls.append("wait")), \
             patch.object(standalone.snapshot_capturer, "load_snapshot", return_value=_snapshot()), \
             patch.object(standalone.eod_reconcile, "run",
                          side_effect=lambda **k: calls.append("reconcile")):
            standalone.run(run_date="2026-06-18")
        assert calls == ["wait", "reconcile"]
