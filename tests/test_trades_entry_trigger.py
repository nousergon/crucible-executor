"""
Tests for the `entry_trigger` column on the trades table.

Phase 2 transparency-inventory item — closes the *trade execution lineage*
row in alpha_engine_lib/transparency_inventory.yaml. The substrate health
check asserts the column is present in trades_full.csv; the daemon's
ENTER fill site is the only writer that populates it (mirrors the
trigger_reason returned by EntryTriggerEngine.should_enter).

Distinct from `trigger_type`: trigger_type is also populated on exits
with the exit reason, so the entry-only contract requires a separate
column rather than a rename.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.trade_logger import init_db, log_trade


def _columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    return cols


def test_init_db_creates_entry_trigger_column(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    assert "entry_trigger" in _columns(db_path)


def test_init_db_idempotent_for_entry_trigger(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    conn = init_db(str(db_path))
    conn.close()
    assert "entry_trigger" in _columns(db_path)


def test_log_trade_persists_entry_trigger(tmp_path):
    """The daemon ENTER fill site populates entry_trigger from the
    EntryTriggerEngine return string (e.g. 'pullback 2.3% from high $182.10').
    """
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-09",
        "ticker": "NVDA",
        "action": "ENTER",
        "shares": 50,
        "entry_trigger": "pullback 2.3% from high $182.10",
    })
    row = conn.execute(
        "SELECT entry_trigger FROM trades WHERE ticker='NVDA'"
    ).fetchone()
    conn.close()
    assert row == ("pullback 2.3% from high $182.10",)


def test_log_trade_entry_trigger_defaults_to_null(tmp_path):
    """Legacy callers that don't pass entry_trigger get NULL — back-compat
    with rows logged before this column shipped, and intended NULL state
    on exit rows (entry-trigger-only contract)."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-09",
        "ticker": "AAPL",
        "action": "ENTER",
        "shares": 100,
    })
    row = conn.execute(
        "SELECT entry_trigger FROM trades WHERE ticker='AAPL'"
    ).fetchone()
    conn.close()
    assert row == (None,)


def test_entry_trigger_independent_of_trigger_type_on_exits(tmp_path):
    """Exit rows write trigger_type with the exit reason but leave
    entry_trigger NULL. Keeps the entry-only contract clean."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-09",
        "ticker": "MSFT",
        "action": "EXIT",
        "shares": 50,
        "exit_reason": "intraday_atr_stop",
        "trigger_type": "intraday_atr_stop",
        # entry_trigger intentionally omitted
    })
    row = conn.execute(
        "SELECT entry_trigger, trigger_type FROM trades WHERE ticker='MSFT'"
    ).fetchone()
    conn.close()
    assert row == (None, "intraday_atr_stop")


def test_entry_trigger_canonical_engine_strings(tmp_path):
    """Smoke-test all 5 trigger_reason strings produced by
    EntryTriggerEngine.should_enter (executor/entry_triggers.py) round-trip
    cleanly through the column. Catches any future encoding regression."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    canonical = [
        ("AAA", "pullback 1.5% from high $50.10"),
        ("BBB", "VWAP discount 0.8% (VWAP=$22.15)"),
        ("CCC", "near support $98.20 (dist 0.4%)"),
        ("DDD", "time_expiry"),
        ("EEE", "graduated_entry (+1.2% vs morning $44.80, 3.5h)"),
    ]
    for ticker, trigger in canonical:
        log_trade(conn, {
            "date": "2026-05-09",
            "ticker": ticker,
            "action": "ENTER",
            "shares": 10,
            "entry_trigger": trigger,
        })
    rows = conn.execute(
        "SELECT ticker, entry_trigger FROM trades ORDER BY ticker"
    ).fetchall()
    conn.close()
    assert rows == canonical
