"""Tests for scripts/backfill_trades_sector.py.

One-shot backfill script that resolves trades.sector="Unknown" rows
from constituents.json sector_map. Surface for the 2026-05-04 EOG/NVT
incident — the bad rows survive in trades.db because no UPDATE path
overwrites trades.sector.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_trades_sector import backfill  # noqa: E402


def _make_db(tmp_path: Path) -> str:
    db = tmp_path / "trades.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            sector TEXT
        )
    """)
    conn.commit()
    conn.close()
    return str(db)


def _insert(db_path: str, trade_id: str, date: str, ticker: str, action: str, sector):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (trade_id, date, ticker, action, sector) VALUES (?, ?, ?, ?, ?)",
        (trade_id, date, ticker, action, sector),
    )
    conn.commit()
    conn.close()


def _read_sector(db_path: str, trade_id: str) -> str | None:
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT sector FROM trades WHERE trade_id = ?", (trade_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def test_backfill_patches_unknown_rows(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "EOG", "ENTER", "Unknown")
    _insert(db, "t2", "2026-05-04", "NVT", "ENTER", "Unknown")
    _insert(db, "t3", "2026-05-05", "CTAS", "ENTER", "Industrials")

    constituents = {"EOG": "Energy", "NVT": "Industrials", "CTAS": "Industrials"}
    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value=constituents,
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats == {"candidates": 2, "patched": 2, "unresolved": 0}
    assert _read_sector(db, "t1") == "Energy"
    assert _read_sector(db, "t2") == "Industrials"
    assert _read_sector(db, "t3") == "Industrials"


def test_backfill_handles_null_and_empty_sector(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "EOG", "ENTER", None)
    _insert(db, "t2", "2026-05-04", "NVT", "ENTER", "")
    _insert(db, "t3", "2026-05-04", "CTAS", "ENTER", "Unknown")

    constituents = {"EOG": "Energy", "NVT": "Industrials", "CTAS": "Industrials"}
    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value=constituents,
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats["patched"] == 3
    assert _read_sector(db, "t1") == "Energy"
    assert _read_sector(db, "t2") == "Industrials"
    assert _read_sector(db, "t3") == "Industrials"


def test_backfill_dry_run_does_not_write(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "EOG", "ENTER", "Unknown")

    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value={"EOG": "Energy"},
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=True)

    assert stats == {"candidates": 1, "patched": 1, "unresolved": 0}
    assert _read_sector(db, "t1") == "Unknown"


def test_backfill_unresolved_when_ticker_not_in_map(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "RARE", "ENTER", "Unknown")

    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value={"OTHER": "Healthcare"},
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats == {"candidates": 1, "patched": 0, "unresolved": 1}
    assert _read_sector(db, "t1") == "Unknown"


def test_backfill_no_candidates_is_noop(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-05", "CTAS", "ENTER", "Industrials")

    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        side_effect=AssertionError("should not load constituents when no candidates"),
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats == {"candidates": 0, "patched": 0, "unresolved": 0}


def test_backfill_aborts_when_constituents_unavailable(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "EOG", "ENTER", "Unknown")
    _insert(db, "t2", "2026-05-04", "NVT", "ENTER", "Unknown")

    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value={},
    ):
        stats = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats == {"candidates": 2, "patched": 0, "unresolved": 2}
    assert _read_sector(db, "t1") == "Unknown"
    assert _read_sector(db, "t2") == "Unknown"


def test_backfill_idempotent(tmp_path):
    """Running twice on the same data produces the same final state."""
    db = _make_db(tmp_path)
    _insert(db, "t1", "2026-05-04", "EOG", "ENTER", "Unknown")
    _insert(db, "t2", "2026-05-04", "NVT", "ENTER", "Unknown")

    constituents = {"EOG": "Energy", "NVT": "Industrials"}
    with patch(
        "scripts.backfill_trades_sector._load_constituents_sector_map",
        return_value=constituents,
    ):
        stats1 = backfill(db, "alpha-engine-research", dry_run=False)
        stats2 = backfill(db, "alpha-engine-research", dry_run=False)

    assert stats1["patched"] == 2
    assert stats2 == {"candidates": 0, "patched": 0, "unresolved": 0}
    assert _read_sector(db, "t1") == "Energy"
    assert _read_sector(db, "t2") == "Industrials"


def test_backfill_aborts_on_missing_sector_column(tmp_path):
    """Hard-fail with sys.exit(2) when sector column hasn't been migrated."""
    db = tmp_path / "no_sector.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT PRIMARY KEY, ticker TEXT, action TEXT)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit) as exc_info:
        backfill(str(db), "alpha-engine-research", dry_run=True)
    assert exc_info.value.code == 2
