"""
Tests for the `sector` column on the trades table.

Closes the dead-fallback in eod_reconcile's sector lookup chain:
  1. signals.json today (works for tickers in current top-50)
  2. trades.db entry_trade.sector  ← was always None pre-this-change because
                                     the schema lacked the column. Now populated
                                     by log_trade() so EOD reconcile can resolve
                                     sector from history when a held ticker has
                                     dropped out of today's signals.json (e.g.
                                     JHG on 2026-04-30).
  3. constituents.json sector_map (ultimate fallback)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.trade_logger import init_db, log_trade, get_entry_trade


def _columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    return cols


def test_init_db_creates_sector_column(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    assert "sector" in _columns(db_path)


def test_init_db_idempotent_for_sector(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    conn = init_db(str(db_path))
    conn.close()
    assert "sector" in _columns(db_path)


def test_log_trade_persists_sector(tmp_path):
    """log_trade writes sector when caller supplies it."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-04-30",
        "ticker": "JHG",
        "action": "ENTER",
        "shares": 1007,
        "sector": "Financials",
        "sector_rating": "market_weight",
    })
    row = conn.execute("SELECT sector, sector_rating FROM trades WHERE ticker='JHG'").fetchone()
    conn.close()
    assert row == ("Financials", "market_weight")


def test_log_trade_sector_defaults_to_null(tmp_path):
    """Legacy callers that don't pass sector get NULL, not a hard fail."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-04-30",
        "ticker": "AAPL",
        "action": "ENTER",
        "shares": 100,
    })
    row = conn.execute("SELECT sector FROM trades WHERE ticker='AAPL'").fetchone()
    conn.close()
    assert row == (None,)


def test_get_entry_trade_returns_sector(tmp_path):
    """The eod_reconcile fallback path: get_entry_trade() must surface sector."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-04-30",
        "ticker": "JHG",
        "action": "ENTER",
        "shares": 1007,
        "sector": "Financials",
    })
    entry = get_entry_trade(conn, "JHG")
    conn.close()
    assert entry is not None
    assert entry["sector"] == "Financials"
