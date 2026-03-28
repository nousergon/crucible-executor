"""
Write trade records to SQLite and back up trades.db to S3.

Schema per design doc B.5.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)


CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id                 TEXT PRIMARY KEY,
    date                     TEXT NOT NULL,
    ticker                   TEXT NOT NULL,
    action                   TEXT NOT NULL,
    shares                   INTEGER NOT NULL,
    price_at_order           REAL,
    portfolio_nav_at_order   REAL,
    position_pct             REAL,
    research_score           REAL,
    research_conviction      TEXT,
    research_rating          TEXT,
    sector_rating            TEXT,
    market_regime            TEXT,
    price_target_upside      REAL,
    thesis_summary           TEXT,
    fill_price               REAL,
    fill_time                TEXT,
    ib_order_id              INTEGER,
    predicted_direction      TEXT,
    prediction_confidence    REAL,
    rationale_json           TEXT,
    created_at               TEXT NOT NULL
);
"""

_TRADES_MIGRATIONS = [
    "ALTER TABLE trades ADD COLUMN predicted_direction TEXT",
    "ALTER TABLE trades ADD COLUMN prediction_confidence REAL",
    "ALTER TABLE trades ADD COLUMN rationale_json TEXT",
    "ALTER TABLE trades ADD COLUMN status TEXT",
    "ALTER TABLE trades ADD COLUMN exit_reason TEXT",
    "ALTER TABLE trades ADD COLUMN filled_shares INTEGER",
    "ALTER TABLE trades ADD COLUMN execution_latency_ms INTEGER",
    "ALTER TABLE trades ADD COLUMN source TEXT",
    # ── Roundtrip linkage + execution quality (2026-03-27) ──
    "ALTER TABLE trades ADD COLUMN entry_trade_id TEXT",
    "ALTER TABLE trades ADD COLUMN signal_price REAL",
    "ALTER TABLE trades ADD COLUMN trigger_price REAL",
    "ALTER TABLE trades ADD COLUMN trigger_type TEXT",
    "ALTER TABLE trades ADD COLUMN spy_price_at_order REAL",
    "ALTER TABLE trades ADD COLUMN realized_pnl REAL",
    "ALTER TABLE trades ADD COLUMN realized_return_pct REAL",
    "ALTER TABLE trades ADD COLUMN spy_return_during_hold REAL",
    "ALTER TABLE trades ADD COLUMN realized_alpha_pct REAL",
    "ALTER TABLE trades ADD COLUMN days_held INTEGER",
    "ALTER TABLE trades ADD COLUMN slippage_vs_signal REAL",
]

_EOD_MIGRATIONS = [
    "ALTER TABLE eod_pnl ADD COLUMN spy_close REAL",
]

CREATE_EOD_TABLE = """
CREATE TABLE IF NOT EXISTS eod_pnl (
    date                TEXT PRIMARY KEY,
    portfolio_nav       REAL,
    daily_return_pct    REAL,
    spy_return_pct      REAL,
    daily_alpha_pct     REAL,
    positions_snapshot  TEXT,
    created_at          TEXT NOT NULL
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables if they don't exist and run any pending migrations. Returns open connection."""
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TRADES_TABLE + CREATE_EOD_TABLE)
    for migration in _TRADES_MIGRATIONS:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                pass  # Column already exists — expected on re-run
            else:
                logging.getLogger(__name__).error("Migration failed: %s — %s", migration.strip()[:80], e)
                raise
    for migration in _EOD_MIGRATIONS:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                pass  # Column already exists — expected on re-run
            else:
                logging.getLogger(__name__).error("Migration failed: %s — %s", migration.strip()[:80], e)
                raise
    conn.commit()
    logger.info(f"trades.db initialized at {db_path}")
    return conn


def log_trade(conn: sqlite3.Connection, trade: dict) -> str:
    """
    Insert a trade record. Returns the trade_id.

    Required keys in trade: date, ticker, action, shares.
    All other keys are optional.
    """
    trade_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO trades (
            trade_id, date, ticker, action, shares,
            price_at_order, portfolio_nav_at_order, position_pct,
            research_score, research_conviction, research_rating,
            sector_rating, market_regime, price_target_upside,
            thesis_summary, fill_price, fill_time, ib_order_id,
            predicted_direction, prediction_confidence, rationale_json,
            status, exit_reason, filled_shares, execution_latency_ms, source,
            entry_trade_id, signal_price, trigger_price, trigger_type,
            spy_price_at_order, realized_pnl, realized_return_pct,
            spy_return_during_hold, realized_alpha_pct, days_held,
            slippage_vs_signal, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trade_id,
            trade["date"],
            trade["ticker"],
            trade["action"],
            trade["shares"],
            trade.get("price_at_order"),
            trade.get("portfolio_nav_at_order"),
            trade.get("position_pct"),
            trade.get("research_score"),
            trade.get("research_conviction"),
            trade.get("research_rating"),
            trade.get("sector_rating"),
            trade.get("market_regime"),
            trade.get("price_target_upside"),
            trade.get("thesis_summary"),
            trade.get("fill_price"),
            trade.get("fill_time"),
            trade.get("ib_order_id"),
            trade.get("predicted_direction"),
            trade.get("prediction_confidence"),
            trade.get("rationale_json"),
            trade.get("status"),
            trade.get("exit_reason"),
            trade.get("filled_shares"),
            trade.get("execution_latency_ms"),
            trade.get("source"),
            trade.get("entry_trade_id"),
            trade.get("signal_price"),
            trade.get("trigger_price"),
            trade.get("trigger_type"),
            trade.get("spy_price_at_order"),
            trade.get("realized_pnl"),
            trade.get("realized_return_pct"),
            trade.get("spy_return_during_hold"),
            trade.get("realized_alpha_pct"),
            trade.get("days_held"),
            trade.get("slippage_vs_signal"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    logger.info(f"Trade logged: {trade['action']} {trade['shares']} {trade['ticker']} | id={trade_id}")
    return trade_id


def log_eod(conn: sqlite3.Connection, eod: dict) -> None:
    """Insert or replace an EOD P&L record."""
    import json
    conn.execute(
        """
        INSERT OR REPLACE INTO eod_pnl
            (date, portfolio_nav, daily_return_pct, spy_return_pct,
             daily_alpha_pct, positions_snapshot, spy_close, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            eod["date"],
            eod.get("portfolio_nav"),
            eod.get("daily_return_pct"),
            eod.get("spy_return_pct"),
            eod.get("daily_alpha_pct"),
            json.dumps(eod.get("positions_snapshot", {})),
            eod.get("spy_close"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_entry_dates(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, str]:
    """
    Look up the most recent ENTER date for each ticker from trades.db.

    Returns:
        {ticker: "YYYY-MM-DD"} for tickers that have an ENTER record.
        Tickers with no ENTER record are omitted.
    """
    entry_dates = {}
    for ticker in tickers:
        row = conn.execute(
            "SELECT date FROM trades WHERE ticker=? AND action='ENTER' ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if row:
            entry_dates[ticker] = row[0]
    return entry_dates


def get_todays_trades(conn: sqlite3.Connection, run_date: str) -> list[dict]:
    """Return all trades for a given date as dicts (including rationale_json)."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE date=? ORDER BY created_at", (run_date,)
    ).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]


def get_entry_trade(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Return the most recent ENTER trade for a ticker, or None."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM trades WHERE ticker=? AND action='ENTER' ORDER BY date DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    conn.row_factory = None
    return dict(row) if row else None


def get_unmatched_entry(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Return the most recent ENTER trade for *ticker* that has no paired exit.

    An entry is "unmatched" if no other trade row references its trade_id
    via the entry_trade_id column.  Returns None if every entry has been
    paired (or if there are no entries at all).
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM trades
           WHERE ticker = ? AND action = 'ENTER'
             AND trade_id NOT IN (
                 SELECT entry_trade_id FROM trades
                 WHERE entry_trade_id IS NOT NULL
             )
           ORDER BY date DESC, created_at DESC
           LIMIT 1""",
        (ticker,),
    ).fetchone()
    conn.row_factory = None
    return dict(row) if row else None


def backup_to_s3(db_path: str, run_date: str, s3_bucket: str) -> None:
    """Upload trades.db to S3 under trades/trades_{date}.db and trades/trades_latest.db."""
    try:
        s3 = boto3.client("s3")
        key = f"trades/trades_{run_date}.db"
        s3.upload_file(db_path, s3_bucket, key)
        logger.info(f"trades.db backed up to s3://{s3_bucket}/{key}")
        s3.upload_file(db_path, s3_bucket, "trades/trades_latest.db")
        logger.info(f"trades.db backed up to s3://{s3_bucket}/trades/trades_latest.db")
    except Exception as e:
        logger.error("S3 backup failed (non-fatal): %s", e)
