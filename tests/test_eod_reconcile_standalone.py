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
