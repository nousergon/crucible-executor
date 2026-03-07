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
    created_at               TEXT NOT NULL
);
"""

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
    """Create tables if they don't exist. Returns open connection."""
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TRADES_TABLE + CREATE_EOD_TABLE)
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
            thesis_summary, fill_price, fill_time, ib_order_id, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
             daily_alpha_pct, positions_snapshot, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            eod["date"],
            eod.get("portfolio_nav"),
            eod.get("daily_return_pct"),
            eod.get("spy_return_pct"),
            eod.get("daily_alpha_pct"),
            json.dumps(eod.get("positions_snapshot", {})),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def backup_to_s3(db_path: str, run_date: str, s3_bucket: str) -> None:
    """Upload trades.db to S3 under trades/trades_{date}.db."""
    s3 = boto3.client("s3")
    key = f"trades/trades_{run_date}.db"
    s3.upload_file(db_path, s3_bucket, key)
    logger.info(f"trades.db backed up to s3://{s3_bucket}/{key}")
