"""
Write trade records to SQLite and back up trades.db to S3.

Schema per design doc B.5.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime

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
    sector                   TEXT,
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
    # ── Date-convention dual-tracking (2026-04-24; axes named config#1610) ──
    # See alpha-engine-docs/private/DATE_CONVENTIONS.md. The `date` column is
    # the SESSION axis: the NYSE session the trade physically executed in
    # (the daemon's run_date = session_date; the reconcile/snapshot join
    # key). `trading_day` is the KNOWLEDGE axis: the last CLOSED session at
    # fill time — "what closed data was this trade acting on" (D-1 for an
    # intraday fill). `signal_trading_day` is the knowledge day of the
    # originating signals.json where applicable; `created_at` is wall-clock
    # audit. Both dual-tracking columns are nullable so backfill on existing
    # rows is a separate one-shot script (scripts/backfill_trading_day.py)
    # and old log_trade() callers without the new context keep working as
    # NULLs.
    "ALTER TABLE trades ADD COLUMN trading_day TEXT",
    "ALTER TABLE trades ADD COLUMN signal_trading_day TEXT",
    # GICS sector name (e.g. "Financials"). Populated from signals.json at
    # ENTER time. Closes the dead-fallback in eod_reconcile's sector lookup
    # chain — get_entry_trade(...).sector now resolves instead of always
    # returning None and pushing the lookup through to constituents.json.
    "ALTER TABLE trades ADD COLUMN sector TEXT",
    # ── Phase 2 transparency-inventory: artifact-filename lineage (2026-05-06) ──
    # signal_date = signals/{date}/signals.json filename date the order was
    # sourced from (distinct from signal_trading_day, which is the NYSE
    # attribution day declared inside the payload — a holiday or backfilled
    # file can have filename ≠ trading_day).
    # prediction_date = predictor/predictions/{date}.json filename date the
    # GBM veto gate consulted; NULL for non-predictor-gated orders (strategy-
    # driven intraday exits, urgent COVERs).
    # Both nullable for back-compat with rows logged before this PR.
    "ALTER TABLE trades ADD COLUMN signal_date TEXT",
    "ALTER TABLE trades ADD COLUMN prediction_date TEXT",
    # ── Phase 2 transparency-inventory: entry-trigger lineage (2026-05-07) ──
    # entry_trigger is the canonical name in the substrate inventory
    # (nousergon_lib/transparency_inventory.yaml row trade_execution_lineage).
    # The existing trigger_type column overlaps but is also populated on exits
    # (with the exit reason); separating entry_trigger keeps the
    # entry-trigger-only contract clean. Populated only on ENTER rows; NULL
    # elsewhere.
    "ALTER TABLE trades ADD COLUMN entry_trigger TEXT",
    # ── Stance taxonomy arc (2026-05-11) ──────────────────────────────────
    # Denormalize predictor's stance label + catalyst_date onto the trade
    # row at ENTER time. Stance routes the executor's stance-conditional
    # exit rules in strategies/exit_manager.py:
    #
    #   stance="value"    → ATR multiplier widened (looser stop on
    #                       contrarian bounce play); time decay extended
    #                       to ~30 trading days
    #   stance="quality"  → time decay DISABLED (defensive, hold-through-
    #                       cycle); standard ATR
    #   stance="catalyst" → hard exit at catalyst_date + 3 trading days
    #                       (event-driven exit boundary)
    #   stance="momentum" → unchanged (baseline)
    #
    # Both nullable — rows from pre-stance-arc entries stay NULL and the
    # exit logic falls through to legacy behavior.
    "ALTER TABLE trades ADD COLUMN stance TEXT",
    "ALTER TABLE trades ADD COLUMN catalyst_date TEXT",
    # ── Per-decision idempotency key (config#2436) ────────────────────────
    # entry_id = ticker + session date + sizing_source (order_book._entry_id_for),
    # stamped on ENTER rows only. Durable sibling of the order-book WAL's
    # entry_id dedup: get_executed_entry_ids() lets a caller ask "did THIS
    # exact decision already execute today" without conflating it with
    # "does this ticker have ANY executed entry today" (get_executed_entry_tickers,
    # too coarse — would also block a legitimate same-day top-up decision).
    "ALTER TABLE trades ADD COLUMN entry_id TEXT",
]

_EOD_MIGRATIONS = [
    "ALTER TABLE eod_pnl ADD COLUMN spy_close REAL",
    "ALTER TABLE eod_pnl ADD COLUMN total_cash REAL",
    "ALTER TABLE eod_pnl ADD COLUMN accrued_interest REAL",
    "ALTER TABLE eod_pnl ADD COLUMN unrealized_pnl REAL",
    "ALTER TABLE eod_pnl ADD COLUMN realized_pnl REAL",
    # Phase 2 transparency-inventory: per-day P&L attribution lineage.
    # Closes the *P&L attribution* row in the gate checklist by
    # publishing the previously log-only NAV-reconciliation breakdown
    # as named columns. The headline metric is unattributed_residual_pct
    # = unattributed_usd / portfolio_nav × 100; the inventory gate is
    # ≤1%. The other columns ride along so a downstream reader can
    # reconstruct the attribution waterfall without re-running reconcile.
    "ALTER TABLE eod_pnl ADD COLUMN nav_change_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN position_pnl_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN interest_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN dividend_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN unattributed_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN unattributed_residual_pct REAL",
]

CREATE_SHADOW_BOOK_TABLE = """
CREATE TABLE IF NOT EXISTS executor_shadow_book (
    shadow_id               TEXT PRIMARY KEY,
    date                    TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    block_reason            TEXT NOT NULL,
    research_score          REAL,
    conviction              TEXT,
    sector                  TEXT,
    sector_rating           TEXT,
    predicted_direction     TEXT,
    prediction_confidence   REAL,
    intended_position_pct   REAL,
    intended_shares         INTEGER,
    intended_dollars        REAL,
    current_price           REAL,
    portfolio_nav           REAL,
    market_regime           TEXT,
    created_at              TEXT NOT NULL
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


# Phase 2 transparency-inventory: structured veto/override/halt event log.
# Closes the *risk decisions* row in the gate checklist (ROADMAP 2026-05-05).
# `executor_shadow_book` is the ENTER-block sibling — same family, different
# axis. Shadow book is keyed per-ticker per-day with free-text `block_reason`
# for downstream evaluator backtesting. `risk_events` is the structured-rule
# log that answers *"how often is rule X firing, and how close was the
# measured value to the threshold?"* — the answer the inventory checklist
# requires per gate.
CREATE_RISK_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS risk_events (
    event_id          TEXT PRIMARY KEY,
    date              TEXT NOT NULL,
    trading_day       TEXT,
    event_type        TEXT NOT NULL,
    rule              TEXT NOT NULL,
    ticker            TEXT,
    sector            TEXT,
    reason            TEXT,
    value             REAL,
    threshold         REAL,
    market_regime     TEXT,
    signal_date       TEXT,
    prediction_date   TEXT,
    context_json      TEXT,
    created_at        TEXT NOT NULL
);
"""

_RISK_EVENTS_MIGRATIONS: list[str] = [
    # Placeholder — future column adds follow the same idempotent pattern as
    # `_TRADES_MIGRATIONS` (catch "duplicate column" on re-run).
]


def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables if they don't exist and run any pending migrations. Returns open connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        CREATE_TRADES_TABLE
        + CREATE_EOD_TABLE
        + CREATE_SHADOW_BOOK_TABLE
        + CREATE_RISK_EVENTS_TABLE
    )
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
    for migration in _RISK_EVENTS_MIGRATIONS:
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
    # If trading_day not provided by the caller, derive it from the
    # date-convention helper so legacy call sites that haven't been migrated
    # yet still get a populated trading_day rather than NULL. See
    # alpha-engine-docs/private/DATE_CONVENTIONS.md for the rule
    # (trading_day = last_closed_trading_day(now), strictly backward-looking).
    # signal_trading_day stays NULL by default — only entry trades originating
    # from a known signals.json populate it.
    trading_day = trade.get("trading_day")
    if trading_day is None:
        try:
            from nousergon_lib.dates import now_dual
            trading_day = now_dual().trading_day
        except Exception:
            # Lib not yet bumped on this deploy — leave NULL. Backfill script
            # closes the gap. Don't hard-fail on a missing optional dep.
            trading_day = None
    # Content-vs-key guard (config#1610): `date` is the SESSION axis — the
    # session the trade executed in — so a fill_time in a different session
    # than the label is a mis-key (the daemon's frozen run_date drifting, or
    # a caller passing the knowledge axis by mistake). Deliberate swallow:
    # the ERROR log is the recording surface, and the row is still inserted
    # — refusing to record an ALREADY-EXECUTED order would trade an audit
    # gap for a label bug, strictly worse. The startup strict-guard and the
    # cross-component invariant test are the hard enforcement.
    _fill_time = trade.get("fill_time")
    if _fill_time and trade.get("date"):
        try:
            from datetime import datetime as _dt

            from nousergon_lib.dates import assert_within_session
            assert_within_session(
                _dt.fromisoformat(str(_fill_time).replace("Z", "+00:00")),
                str(trade["date"]),
            )
        except ValueError as _axis_err:
            logger.error(
                "trades.date session mis-key (row inserted anyway): %s",
                _axis_err,
            )
        except Exception as _e:
            # lib not yet bumped / unparseable fill_time — best-effort
            logger.debug("session axis check skipped (best-effort): %s", _e)
    conn.execute(
        """
        INSERT INTO trades (
            trade_id, date, ticker, action, shares,
            price_at_order, portfolio_nav_at_order, position_pct,
            research_score, research_conviction, research_rating,
            sector, sector_rating, market_regime, price_target_upside,
            thesis_summary, fill_price, fill_time, ib_order_id,
            predicted_direction, prediction_confidence, rationale_json,
            status, exit_reason, filled_shares, execution_latency_ms, source,
            entry_trade_id, signal_price, trigger_price, trigger_type,
            spy_price_at_order, realized_pnl, realized_return_pct,
            spy_return_during_hold, realized_alpha_pct, days_held,
            slippage_vs_signal, trading_day, signal_trading_day,
            signal_date, prediction_date, entry_trigger,
            stance, catalyst_date, entry_id, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            trade.get("sector"),
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
            trading_day,
            trade.get("signal_trading_day"),
            trade.get("signal_date"),
            trade.get("prediction_date"),
            trade.get("entry_trigger"),
            trade.get("stance"),
            trade.get("catalyst_date"),
            trade.get("entry_id"),
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    logger.info(f"Trade logged: {trade['action']} {trade['shares']} {trade['ticker']} | id={trade_id}")
    return trade_id


def log_shadow_book_block(conn: sqlite3.Connection, entry: dict) -> str:
    """
    Log a risk guard block to the shadow book for evaluation.
    Returns the shadow_id.
    """
    shadow_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO executor_shadow_book (
            shadow_id, date, ticker, block_reason,
            research_score, conviction, sector, sector_rating,
            predicted_direction, prediction_confidence,
            intended_position_pct, intended_shares, intended_dollars,
            current_price, portfolio_nav, market_regime, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            shadow_id,
            entry["date"],
            entry["ticker"],
            entry["block_reason"],
            entry.get("research_score"),
            entry.get("conviction"),
            entry.get("sector"),
            entry.get("sector_rating"),
            entry.get("predicted_direction"),
            entry.get("prediction_confidence"),
            entry.get("intended_position_pct"),
            entry.get("intended_shares"),
            entry.get("intended_dollars"),
            entry.get("current_price"),
            entry.get("portfolio_nav"),
            entry.get("market_regime"),
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    logger.info("Shadow book: BLOCKED %s — %s | id=%s", entry["ticker"], entry["block_reason"], shadow_id)
    return shadow_id


def log_risk_event(conn: sqlite3.Connection, event: dict) -> str:
    """
    Insert a structured veto/override/halt/throttle event. Returns event_id.

    Required keys: date, event_type, rule.
    Optional keys: trading_day, ticker, sector, reason, value, threshold,
                   market_regime, signal_date, prediction_date, context.

    `context` (dict) is serialized to context_json. Use it for rule-specific
    extra context that doesn't justify a top-level column (e.g., per-ticker
    correlation map for the correlation rule, breached tier description for
    drawdown_tier_throttle). Keep it small — this is a structured log, not
    a debug dump.
    """
    import json
    event_id = str(uuid.uuid4())
    trading_day = event.get("trading_day")
    if trading_day is None:
        try:
            from nousergon_lib.dates import now_dual
            trading_day = now_dual().trading_day
        except Exception:
            trading_day = None
    context = event.get("context")
    context_json = json.dumps(context) if context else None
    conn.execute(
        """
        INSERT INTO risk_events (
            event_id, date, trading_day, event_type, rule, ticker, sector,
            reason, value, threshold, market_regime, signal_date,
            prediction_date, context_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            event_id,
            event["date"],
            trading_day,
            event["event_type"],
            event["rule"],
            event.get("ticker"),
            event.get("sector"),
            event.get("reason"),
            event.get("value"),
            event.get("threshold"),
            event.get("market_regime"),
            event.get("signal_date"),
            event.get("prediction_date"),
            context_json,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    logger.info(
        "Risk event logged: %s/%s ticker=%s | id=%s",
        event["event_type"],
        event["rule"],
        event.get("ticker") or "-",
        event_id,
    )
    return event_id


def log_eod(conn: sqlite3.Connection, eod: dict) -> None:
    """Insert or replace an EOD P&L record.

    Phase 2 transparency-inventory adds 6 attribution fields:
      - nav_change_usd, position_pnl_usd, interest_usd, dividend_usd
      - unattributed_usd  (the residual after attribution)
      - unattributed_residual_pct  (residual / NAV × 100, the inventory's
                                    headline metric — gate is ≤1%)
    All optional for back-compat with legacy callers.
    """
    import json
    conn.execute(
        """
        INSERT OR REPLACE INTO eod_pnl
            (date, portfolio_nav, daily_return_pct, spy_return_pct,
             daily_alpha_pct, positions_snapshot, spy_close,
             total_cash, accrued_interest, unrealized_pnl, realized_pnl,
             nav_change_usd, position_pnl_usd, interest_usd, dividend_usd,
             unattributed_usd, unattributed_residual_pct,
             created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            eod["date"],
            eod.get("portfolio_nav"),
            eod.get("daily_return_pct"),
            eod.get("spy_return_pct"),
            eod.get("daily_alpha_pct"),
            json.dumps(eod.get("positions_snapshot", {})),
            eod.get("spy_close"),
            eod.get("total_cash"),
            eod.get("accrued_interest"),
            eod.get("unrealized_pnl"),
            eod.get("realized_pnl"),
            eod.get("nav_change_usd"),
            eod.get("position_pnl_usd"),
            eod.get("interest_usd"),
            eod.get("dividend_usd"),
            eod.get("unattributed_usd"),
            eod.get("unattributed_residual_pct"),
            datetime.now(UTC).isoformat(),
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


def get_executed_entry_tickers(conn: sqlite3.Connection, run_date: str) -> set[str]:
    """Tickers with an ENTER fill logged for ``run_date`` (session axis).

    Crash-restart seeding surface (config#2328): the daemon rehydrates its
    in-memory ``executed_tickers`` set from this at startup so a restart does
    not re-place a BUY whose fill already reached trades.db even if the order
    book save that follows never landed. ``date`` is the session axis =
    daemon ``run_date`` (see trades schema / _TRADES_MIGRATIONS notes).
    """
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM trades WHERE date=? AND action='ENTER'",
        (run_date,),
    ).fetchall()
    return {r[0] for r in rows}


def get_executed_entry_ids(conn: sqlite3.Connection, run_date: str) -> set[str]:
    """entry_ids with an ENTER fill logged for ``run_date`` (config#2436).

    Precise sibling of ``get_executed_entry_tickers``: keyed on the
    per-decision ``entry_id`` (ticker + session date + sizing_source)
    rather than bare ticker, so a caller can ask "did THIS exact decision
    already execute today" — distinguishing a genuine duplicate replay of
    the same decision from a legitimate second, distinct decision for the
    same ticker (e.g. a portfolio-optimizer top-up of an already-held
    name), which bare-ticker matching cannot tell apart.
    """
    rows = conn.execute(
        "SELECT DISTINCT entry_id FROM trades "
        "WHERE date=? AND action='ENTER' AND entry_id IS NOT NULL",
        (run_date,),
    ).fetchall()
    return {r[0] for r in rows}


def get_entry_stance_and_catalyst(
    conn: sqlite3.Connection, tickers: list[str],
) -> dict[str, dict]:
    """Look up the most recent ENTER stance + catalyst_date per ticker.

    Returns ``{ticker: {"stance": str | None, "catalyst_date": str | None}}``
    for tickers that have an ENTER record. Tickers with no ENTER are
    omitted (caller falls through to legacy non-stance exit logic).

    Used by ``strategies.exit_manager.evaluate_exits`` to resolve
    stance-conditional exit rules — ATR multiplier override for
    value-stance, time-decay disable for quality-stance, hard exit
    at catalyst_date+3 trading days for catalyst-stance.

    Both stance and catalyst_date are nullable in the trades table
    (rows logged before the 2026-05-11 stance arc don't have them);
    callers must tolerate either being None.
    """
    out: dict[str, dict] = {}
    for ticker in tickers:
        row = conn.execute(
            "SELECT stance, catalyst_date FROM trades "
            "WHERE ticker=? AND action='ENTER' ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if row:
            out[ticker] = {"stance": row[0], "catalyst_date": row[1]}
    return out


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
    """Return the most recent ENTER trade for *ticker* that has remaining shares.

    An entry has remaining shares if the total shares of all exits
    referencing its trade_id (via entry_trade_id) is less than the
    entry's shares.  This correctly handles partial fills — a REDUCE
    of 50 shares against a 100-share entry leaves 50 shares for a
    subsequent EXIT to match against.

    The returned dict includes a ``shares_remaining`` key.
    Returns None if every entry is fully matched.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT t.*,
                  t.shares - COALESCE(
                      (SELECT SUM(t2.shares) FROM trades t2
                       WHERE t2.entry_trade_id = t.trade_id), 0
                  ) AS shares_remaining
           FROM trades t
           WHERE t.ticker = ? AND t.action = 'ENTER'
           ORDER BY t.date DESC, t.created_at DESC""",
        (ticker,),
    ).fetchall()
    conn.row_factory = None
    # Return first entry with remaining shares
    for r in row:
        d = dict(r)
        if d.get("shares_remaining", 0) > 0:
            return d
    return None


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
