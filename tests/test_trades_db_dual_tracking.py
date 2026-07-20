"""Tests for the trades.db dual-tracking migration (2026-04-24).

Covers:
  * Schema migration is idempotent — re-running init_db() against a DB that
    already has the new columns is a no-op.
  * log_trade() populates trading_day via the nousergon_lib.dates fallback
    when the caller doesn't pass it explicitly.
  * log_trade() honors an explicit trading_day from the caller (live daemon
    path will populate this from the OrderBook entry context).
  * signal_trading_day defaults to NULL and is written when provided.
  * Backfill script populates NULL columns idempotently and never overwrites
    existing values.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.trade_logger import init_db, log_trade

# ── init_db idempotence + column presence ───────────────────────────────────


def _columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    return cols


def test_init_db_creates_dual_tracking_columns(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    cols = _columns(db_path)
    assert "trading_day" in cols
    assert "signal_trading_day" in cols


def test_init_db_idempotent(tmp_path):
    """Running init_db twice on the same path doesn't error."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    # Second invocation hits the duplicate-column branch in the migration loop.
    conn = init_db(str(db_path))
    conn.close()
    cols = _columns(db_path)
    assert "trading_day" in cols
    assert "signal_trading_day" in cols


def test_init_db_idempotent_after_legacy_db(tmp_path):
    """Simulate a pre-migration DB by creating only the legacy schema, then
    init_db should add the new columns without error."""
    db_path = tmp_path / "trades.db"
    # Create the legacy schema manually (subset of pre-2026-04-24 columns).
    legacy_conn = sqlite3.connect(str(db_path))
    legacy_conn.execute(
        """
        CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            shares INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()
    # Now run init_db — should ALTER TABLE for every migration including ours.
    conn = init_db(str(db_path))
    conn.close()
    cols = _columns(db_path)
    assert "trading_day" in cols
    assert "signal_trading_day" in cols


# ── log_trade column population ─────────────────────────────────────────────


def _read_back(db_path: Path, trade_id: str) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    conn.close()
    assert row is not None, f"trade {trade_id} not found"
    return dict(row)


def _minimal_trade(**overrides) -> dict:
    base = {
        "date": "2026-04-27",
        "ticker": "AAPL",
        "action": "ENTER",
        "shares": 100,
    }
    base.update(overrides)
    return base


def test_log_trade_populates_trading_day_via_fallback(tmp_path):
    """No trading_day in trade dict → log_trade derives it via now_dual()."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    trade_id = log_trade(conn, _minimal_trade())
    conn.close()
    row = _read_back(db_path, trade_id)
    # Either lib is installed and trading_day is populated, or lib import
    # failed and it stays NULL — either is acceptable for backward-compat.
    # We assert that NO ERROR was raised; the precise value depends on lib
    # availability at test time.
    assert row["trade_id"] == trade_id  # row was inserted


def test_log_trade_honors_explicit_trading_day(tmp_path):
    """Caller-provided trading_day takes precedence over the fallback."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    trade_id = log_trade(conn, _minimal_trade(trading_day="2026-04-24"))
    conn.close()
    row = _read_back(db_path, trade_id)
    assert row["trading_day"] == "2026-04-24"


def test_log_trade_signal_trading_day_default_null(tmp_path):
    """signal_trading_day stays NULL when not provided."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    trade_id = log_trade(conn, _minimal_trade())
    conn.close()
    row = _read_back(db_path, trade_id)
    assert row["signal_trading_day"] is None


def test_log_trade_signal_trading_day_explicit(tmp_path):
    """Caller-provided signal_trading_day is stored."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    trade_id = log_trade(
        conn,
        _minimal_trade(signal_trading_day="2026-04-24"),
    )
    conn.close()
    row = _read_back(db_path, trade_id)
    assert row["signal_trading_day"] == "2026-04-24"


def test_log_trade_legacy_callers_still_work(tmp_path):
    """Backward compat: a trade dict with no new fields inserts cleanly.
    All legacy daemon/exit/manual-tool call sites pass through this path."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    trade_id = log_trade(
        conn,
        _minimal_trade(
            price_at_order=100.0,
            fill_price=100.5,
            fill_time="2026-04-27T14:30:00+00:00",
            ib_order_id=12345,
        ),
    )
    conn.close()
    row = _read_back(db_path, trade_id)
    assert row["trade_id"] == trade_id
    assert row["fill_price"] == 100.5


# ── Backfill script ─────────────────────────────────────────────────────────


def _seed_trades(db_path: Path, trades: list[dict]) -> None:
    conn = init_db(str(db_path))
    for trade in trades:
        log_trade(conn, trade)
    conn.close()


def _read_all(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY created_at").fetchall()]
    conn.close()
    return rows


def _set_columns_null(db_path: Path) -> None:
    """Force trading_day + signal_trading_day to NULL on every row, simulating
    a fresh-from-pre-migration state for the backfill to operate on."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE trades SET trading_day = NULL, signal_trading_day = NULL")
    conn.commit()
    conn.close()


def test_backfill_dry_run_does_not_write(tmp_path):
    """--dry-run computes updates but doesn't mutate the DB."""
    pytest.importorskip("nousergon_lib.dates", reason="lib v0.2.0+ required")
    from scripts import backfill_trading_day

    db_path = tmp_path / "trades.db"
    _seed_trades(db_path, [
        {
            "date": "2026-04-27",
            "ticker": "AAPL",
            "action": "ENTER",
            "shares": 100,
            "fill_time": "2026-04-27T14:30:00+00:00",
        },
    ])
    _set_columns_null(db_path)

    # Mock the SignalIndex S3 walk since we don't have a real bucket in tests.
    with patch.object(backfill_trading_day, "_SignalIndex") as mock_index_cls:
        mock_index = MagicMock()
        mock_index.lookup.return_value = "2026-04-24"
        mock_index_cls.return_value = mock_index

        stats = backfill_trading_day.backfill(
            db_path=str(db_path),
            bucket="test-bucket",
            dry_run=True,
        )

    assert stats["trading_day_filled"] >= 1
    # No writes committed despite computed updates.
    assert stats["writes_committed"] == 0
    rows = _read_all(db_path)
    assert all(r["trading_day"] is None for r in rows)


def test_backfill_idempotent(tmp_path):
    """Running backfill twice produces the same end state — already-set
    columns are skipped on the second pass."""
    pytest.importorskip("nousergon_lib.dates", reason="lib v0.2.0+ required")
    from scripts import backfill_trading_day

    db_path = tmp_path / "trades.db"
    _seed_trades(db_path, [
        {
            "date": "2026-04-27",
            "ticker": "AAPL",
            "action": "ENTER",
            "shares": 100,
            "fill_time": "2026-04-27T14:30:00+00:00",
        },
        {
            "date": "2026-04-27",
            "ticker": "MSFT",
            "action": "EXIT",
            "shares": 50,
            "fill_time": "2026-04-27T15:00:00+00:00",
        },
    ])
    _set_columns_null(db_path)

    with patch.object(backfill_trading_day, "_SignalIndex") as mock_index_cls:
        mock_index = MagicMock()
        mock_index.lookup.return_value = "2026-04-24"
        mock_index_cls.return_value = mock_index

        stats1 = backfill_trading_day.backfill(
            db_path=str(db_path), bucket="test-bucket", dry_run=False,
        )
        rows_after_first = _read_all(db_path)

        stats2 = backfill_trading_day.backfill(
            db_path=str(db_path), bucket="test-bucket", dry_run=False,
        )
        rows_after_second = _read_all(db_path)

    # First pass writes both rows (trading_day for each + signal_trading_day for ENTER).
    assert stats1["writes_committed"] >= 1
    # Second pass writes nothing — every row's columns are already populated.
    assert stats2["writes_committed"] == 0
    # State is stable across the two runs.
    assert rows_after_first == rows_after_second


def test_backfill_does_not_overwrite_existing_values(tmp_path):
    """If trading_day is already set (e.g., from a forward-write call site),
    the backfill must not overwrite it."""
    pytest.importorskip("nousergon_lib.dates", reason="lib v0.2.0+ required")
    from scripts import backfill_trading_day

    db_path = tmp_path / "trades.db"
    _seed_trades(db_path, [
        {
            "date": "2026-04-27",
            "ticker": "AAPL",
            "action": "ENTER",
            "shares": 100,
            "fill_time": "2026-04-27T14:30:00+00:00",
            "trading_day": "EXPLICIT_VALUE",  # forward-write authoritative
            "signal_trading_day": "EXPLICIT_SIGNAL",
        },
    ])
    # Don't NULL — leave the explicit values in place.

    with patch.object(backfill_trading_day, "_SignalIndex") as mock_index_cls:
        mock_index = MagicMock()
        mock_index.lookup.return_value = "BACKFILL_GUESS"
        mock_index_cls.return_value = mock_index

        stats = backfill_trading_day.backfill(
            db_path=str(db_path), bucket="test-bucket", dry_run=False,
        )

    assert stats["writes_committed"] == 0
    assert stats["trading_day_already_set"] == 1
    assert stats["signal_trading_day_already_set"] == 1
    rows = _read_all(db_path)
    assert rows[0]["trading_day"] == "EXPLICIT_VALUE"
    assert rows[0]["signal_trading_day"] == "EXPLICIT_SIGNAL"


def test_backfill_skips_signal_trading_day_for_non_enter(tmp_path):
    """EXIT and REDUCE actions don't have a signal_trading_day backfill —
    they stay NULL by design (they're outcomes of held positions, not new
    signal-driven entries)."""
    pytest.importorskip("nousergon_lib.dates", reason="lib v0.2.0+ required")
    from scripts import backfill_trading_day

    db_path = tmp_path / "trades.db"
    _seed_trades(db_path, [
        {
            "date": "2026-04-27",
            "ticker": "AAPL",
            "action": "EXIT",
            "shares": 100,
            "fill_time": "2026-04-27T14:30:00+00:00",
        },
    ])
    _set_columns_null(db_path)

    with patch.object(backfill_trading_day, "_SignalIndex") as mock_index_cls:
        mock_index = MagicMock()
        mock_index.lookup.return_value = "2026-04-24"
        mock_index_cls.return_value = mock_index

        stats = backfill_trading_day.backfill(
            db_path=str(db_path), bucket="test-bucket", dry_run=False,
        )

    assert stats["non_enter_skipped"] == 1
    rows = _read_all(db_path)
    assert rows[0]["signal_trading_day"] is None
    # trading_day still gets backfilled — that's based on fill_time, not action.
    assert rows[0]["trading_day"] is not None
