"""
Tests for the `signal_date` + `prediction_date` columns on the trades table.

Phase 2 transparency-inventory item — closes the *trade execution decisions*
row in the gate checklist. Every fill is now traceable to the specific
signals.json filename date that drove it (signal_date) and the
predictor/predictions/{date}.json filename the GBM veto consulted
(prediction_date, NULL when the order wasn't predictor-gated).

Distinct from `signal_trading_day` (NYSE attribution day inside the payload):
holiday or backfilled signals files can have filename ≠ trading_day.
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


def test_init_db_creates_signal_and_prediction_date_columns(tmp_path):
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    cols = _columns(db_path)
    assert "signal_date" in cols
    assert "prediction_date" in cols


def test_init_db_idempotent_for_new_columns(tmp_path):
    """Re-running init_db on an existing schema must not raise — the
    sqlite ADD COLUMN guard rails treat duplicates as expected."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    conn.close()
    # Second init_db should be a no-op
    conn = init_db(str(db_path))
    conn.close()
    cols = _columns(db_path)
    assert "signal_date" in cols
    assert "prediction_date" in cols


def test_log_trade_persists_signal_and_prediction_date(tmp_path):
    """log_trade writes both columns when caller supplies them — the
    intended live path through daemon.py:_execute_entry."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-06",
        "ticker": "NVDA",
        "action": "ENTER",
        "shares": 50,
        "signal_date": "2026-05-02",        # last Saturday's signals.json
        "prediction_date": "2026-05-06",    # today's predictions/latest
    })
    row = conn.execute(
        "SELECT signal_date, prediction_date FROM trades WHERE ticker='NVDA'"
    ).fetchone()
    conn.close()
    assert row == ("2026-05-02", "2026-05-06")


def test_log_trade_dates_default_to_null(tmp_path):
    """Legacy callers that don't pass either date get NULL, not a hard
    fail — matches the back-compat policy on every other additive
    column (sector, signal_trading_day, etc.)."""
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-06",
        "ticker": "AAPL",
        "action": "ENTER",
        "shares": 100,
    })
    row = conn.execute(
        "SELECT signal_date, prediction_date FROM trades WHERE ticker='AAPL'"
    ).fetchone()
    conn.close()
    assert row == (None, None)


def test_log_trade_prediction_date_null_for_strategy_exit(tmp_path):
    """Strategy-driven intraday exits (ATR stop, profit-take, time-decay)
    are NOT predictor-gated — the daemon's exit path leaves
    prediction_date unset, which must persist as NULL. Distinct from
    research-driven EXIT/REDUCE which carry both dates from the
    urgent_exits_with_meta payload.
    """
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-06",
        "ticker": "MSFT",
        "action": "EXIT",
        "shares": 50,
        "exit_reason": "intraday_atr_stop",
        # signal_date / prediction_date intentionally omitted
    })
    row = conn.execute(
        "SELECT signal_date, prediction_date, exit_reason "
        "FROM trades WHERE ticker='MSFT'"
    ).fetchone()
    conn.close()
    assert row == (None, None, "intraday_atr_stop")


def test_signal_date_distinct_from_signal_trading_day(tmp_path):
    """The two columns capture different facts: signal_date is the
    signals.json filename date (artifact lineage); signal_trading_day
    is the NYSE attribution day declared inside the payload. They CAN
    differ when Research backfills a holiday or runs off-cadence.
    """
    db_path = tmp_path / "trades.db"
    conn = init_db(str(db_path))
    log_trade(conn, {
        "date": "2026-05-06",
        "ticker": "TSLA",
        "action": "ENTER",
        "shares": 30,
        "signal_date": "2026-05-04",          # signals.json filename (off-cadence backfill)
        "signal_trading_day": "2026-05-02",   # NYSE attribution day
        "prediction_date": "2026-05-06",
    })
    row = conn.execute(
        "SELECT signal_date, signal_trading_day, prediction_date "
        "FROM trades WHERE ticker='TSLA'"
    ).fetchone()
    conn.close()
    assert row == ("2026-05-04", "2026-05-02", "2026-05-06")
