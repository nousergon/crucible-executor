"""Fail-loud contract for the EOD trade-history S3 CSV exports (config#1234).

``_export_trade_csvs_to_s3`` (executor/eod_reconcile.py) exports
``trades_full.csv``, ``eod_pnl.csv``, and ``shadow_book.csv`` to S3 for
dashboard consumption. ``trades_full_log`` and ``eod_reconcile_pnl`` are both
registered in ARTIFACT_REGISTRY.yaml with ``severity: critical`` ("NAV /
alpha-vs-SPY ground truth" / "complete trade audit log") — a swallowed PUT
failure for either is a ghost success: the EOD SF reports OK while the
dashboard's trade log / P&L history silently goes stale.

``shadow_book.csv`` is NOT registered (no SF/Lambda/dashboard-critical
consumer depends on it — purely diagnostic), so it stays fail-soft.

These tests pin the fixed contract: a trades_full.csv or eod_pnl.csv PUT
failure now RE-RAISES (mirroring crucible-research#312 /
crucible-predictor#304), while a shadow_book.csv PUT failure stays
non-fatal and does not block the two ground-truth exports.
"""
from __future__ import annotations

import pytest

from executor.eod_reconcile import _export_trade_csvs_to_s3
from executor.trade_logger import init_db


@pytest.fixture
def db(tmp_path):
    return init_db(str(tmp_path / "trades.db"))


class _FakeS3:
    """Records every put_object call; keyed failures raise on that key only."""

    def __init__(self, fail_keys: set[str] | None = None):
        self.fail_keys = fail_keys or set()
        self.puts: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if Key in self.fail_keys:
            raise RuntimeError(f"S3 PUT failed for {Key}")
        self.puts[Key] = Body


def test_trades_full_csv_failure_reraises(db):
    """A trades_full.csv PUT failure must propagate, not be swallowed."""
    s3 = _FakeS3(fail_keys={"trades/trades_full.csv"})

    with pytest.raises(RuntimeError, match="trades/trades_full.csv"):
        _export_trade_csvs_to_s3(db, s3, "test-bucket")


def test_eod_pnl_csv_failure_reraises(db):
    """An eod_pnl.csv PUT failure must propagate, not be swallowed."""
    s3 = _FakeS3(fail_keys={"trades/eod_pnl.csv"})

    with pytest.raises(RuntimeError, match="trades/eod_pnl.csv"):
        _export_trade_csvs_to_s3(db, s3, "test-bucket")


def test_shadow_book_csv_failure_is_non_fatal(db, caplog):
    """shadow_book.csv is diagnostic-only (not in ARTIFACT_REGISTRY.yaml) —
    its PUT failure must stay fail-soft, and must not block the two
    ground-truth exports that precede it in the loop."""
    s3 = _FakeS3(fail_keys={"trades/shadow_book.csv"})

    with caplog.at_level("WARNING"):
        _export_trade_csvs_to_s3(db, s3, "test-bucket")  # must NOT raise

    assert "trades/trades_full.csv" in s3.puts
    assert "trades/eod_pnl.csv" in s3.puts
    assert "trades/shadow_book.csv" not in s3.puts
    assert any(
        "S3 CSV export failed for trades/shadow_book.csv" in r.message
        for r in caplog.records
    )


def test_all_three_csvs_exported_on_success(db):
    """Happy path: all three CSVs land in S3 under their registered keys."""
    s3 = _FakeS3()

    _export_trade_csvs_to_s3(db, s3, "test-bucket")

    assert set(s3.puts) == {
        "trades/trades_full.csv",
        "trades/eod_pnl.csv",
        "trades/shadow_book.csv",
    }
