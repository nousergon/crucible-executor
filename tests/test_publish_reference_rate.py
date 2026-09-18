"""Tests for executor.publish_reference_rate — component 3's own schedule
for the reference-rate showcase artifact (alpha-engine-config-I11007).

The whole point of this entrypoint is that it does not depend on the v1
postclose pipeline having fired: it reads today's snapshot if one already
exists (the common case during the v1/v2 coexistence window — the v1 SF's
own CaptureSnapshot step still writes it), and captures its own if not (the
case once v1 is disabled).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from executor import publish_reference_rate as pub


def _config():
    return {"trades_bucket": "alpha-engine-research", "aws_region": "us-east-1"}


def _snapshot():
    return {
        "run_date": "2026-06-18",
        "account": {"net_liquidation": 1_001_593.11},
        "positions": {
            "AMD": {
                "shares": 192,
                "market_value": 103175.04,
                "avg_cost": 130.5,
                "sector": "Information Technology",
            },
        },
        "accrued_dividends": {},
    }


class TestRunDate:
    def test_default_resolves_to_now_dual_trading_day(self):
        with patch("executor.publish_reference_rate.now_dual") as mock_now_dual, \
             patch("executor.publish_reference_rate.load_config") as mock_cfg:
            mock_now_dual.return_value = SimpleNamespace(trading_day="2026-06-18")
            mock_cfg.side_effect = RuntimeError("expected_test_sentinel")
            with pytest.raises(RuntimeError, match="expected_test_sentinel"):
                pub.run(run_date=None)
            mock_now_dual.assert_called_once()


class TestSnapshotReuseVsSelfCapture:
    def test_reuses_existing_snapshot_without_capturing(self):
        """The common case during v1/v2 coexistence: a snapshot already
        exists (written by the v1 SF's CaptureSnapshot step), so this must
        not open a second IB session — it just reads and publishes."""
        with patch("executor.publish_reference_rate.load_config", return_value=_config()), \
             patch("executor.publish_reference_rate.boto3"), \
             patch.object(pub.snapshot_capturer, "load_snapshot", return_value=_snapshot()) as mock_load, \
             patch.object(pub.snapshot_capturer, "run") as mock_capture, \
             patch.object(pub, "_load_nav_history", return_value=[]), \
             patch.object(pub.reference_rate, "publish") as mock_publish:
            pub.run(run_date="2026-06-18")

        mock_capture.assert_not_called()
        mock_load.assert_called_with(bucket="alpha-engine-research", run_date="2026-06-18", region="us-east-1")
        assert mock_publish.call_count == 1
        published_bucket = mock_publish.call_args.args[1]
        payload = mock_publish.call_args.args[2]
        assert published_bucket == "alpha-engine-research"
        assert payload["account"] == {"net_liquidation": 1_001_593.11}
        assert {p["ticker"] for p in payload["positions"]} == {"AMD"}

    def test_self_captures_when_no_snapshot_exists(self):
        """No snapshot yet (the case once v1 is disabled, or this timer
        firing ahead of the v1 pipeline on a given day) — self-sufficient:
        captures one directly rather than raising or waiting."""
        calls = {"n": 0}

        def _load_snapshot(**kw):
            calls["n"] += 1
            return None if calls["n"] == 1 else _snapshot()

        with patch("executor.publish_reference_rate.load_config", return_value=_config()), \
             patch("executor.publish_reference_rate.boto3"), \
             patch.object(pub.snapshot_capturer, "load_snapshot", side_effect=_load_snapshot) as mock_load, \
             patch.object(pub.snapshot_capturer, "run") as mock_capture, \
             patch.object(pub, "_load_nav_history", return_value=[]), \
             patch.object(pub.reference_rate, "publish") as mock_publish:
            pub.run(run_date="2026-06-18")

        mock_capture.assert_called_once_with("2026-06-18")
        assert mock_load.call_count == 2
        assert mock_publish.call_count == 1

    def test_raises_when_self_capture_still_leaves_no_snapshot(self):
        """Fail loud (fleet default: RAISE) — refuses to publish an
        artifact built from no data rather than swallowing the gap."""
        with patch("executor.publish_reference_rate.load_config", return_value=_config()), \
             patch("executor.publish_reference_rate.boto3"), \
             patch.object(pub.snapshot_capturer, "load_snapshot", return_value=None), \
             patch.object(pub.snapshot_capturer, "run"), \
             patch.object(pub.reference_rate, "publish") as mock_publish:
            with pytest.raises(RuntimeError, match="no snapshot is readable back"):
                pub.run(run_date="2026-06-18")
        mock_publish.assert_not_called()


class TestNavHistoryBestEffort:
    def test_nav_history_read_failure_is_non_fatal(self):
        """A missing/unreadable eod_pnl.csv must never block the publish —
        current positions + NAV (read above) is the primary content."""
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("NoSuchKey")
        result = pub._load_nav_history(s3, "alpha-engine-research")
        assert result == []

    def test_nav_history_parses_eod_pnl_csv(self):
        import io

        csv_bytes = (
            b"date,portfolio_nav,spy_close\n"
            b"2026-06-17,999000.0,744.0\n"
            b"2026-06-18,1001593.11,746.74\n"
        )
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": io.BytesIO(csv_bytes)}
        result = pub._load_nav_history(s3, "alpha-engine-research")
        assert [r["date"] for r in result] == ["2026-06-17", "2026-06-18"]
        assert result[-1]["nav"] == 1001593.11
