"""
Write trade records to SQLite and back up trades.db to S3.

Schema per design doc B.5.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
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
    # ── Explicit transaction costs (alpha-engine-config-I8188, defect 2) ──
    # Per-execution commission as IB reports it
    # (``Fill.commissionReport.commission``, summed by
    # ``ibkr.fill_commission_usd``). NULL means IB attached no commission
    # report to any execution — an UNKNOWN figure, which is a different fact
    # from a reported $0.00 and must not render identically. Before this
    # column there was no transaction-cost line anywhere in the schema, so
    # gross and net performance were the same number and neither was
    # labelled; ``pnl_integrity.session_costs`` aggregates this into the
    # session's ``commission_usd`` and reports ``commission_available`` from
    # whether any filled row carried a value.
    "ALTER TABLE trades ADD COLUMN commission_usd REAL",
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
    # ── Performance-measurement integrity (alpha-engine-config-I8188) ──────
    # The attribution sleeves eod_reconcile has computed since config#2457 but
    # never persisted. Without them eod_pnl.csv carried ONE undifferentiated
    # plug, which is why −$20,293 of "unexplained" P&L looked unexplained:
    # reconstructing rotation_realized_usd over the 74 sessions with
    # attribution columns accounts for −$20,815 of it, leaving a TRUE residual
    # of +$522. The bounded quantity is unattributed_true_usd, not the plug.
    "ALTER TABLE eod_pnl ADD COLUMN pricing_timing_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN rotation_realized_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN unattributed_true_usd REAL",
    # Explicit transaction costs + the gross/net split they make possible.
    # commission_usd is a real cash debit already inside nav_change_usd;
    # slippage_usd is implementation shortfall vs the arrival price, which
    # never touched NAV, so the gross figure is an explicit counterfactual.
    "ALTER TABLE eod_pnl ADD COLUMN commission_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN slippage_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN traded_notional_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN commission_available INTEGER",
    "ALTER TABLE eod_pnl ADD COLUMN daily_return_net_pct REAL",
    "ALTER TABLE eod_pnl ADD COLUMN daily_return_gross_pct REAL",
    # Dividend accrual honesty: an absent accrual and a genuine $0 day are
    # different facts. dividend_usd summed to exactly $0.00 across all 115
    # live sessions while the book held seven payers, because the IB paper
    # feed populates no per-symbol accrual and the code read that as zero.
    "ALTER TABLE eod_pnl ADD COLUMN dividend_accrual_available INTEGER",
    "ALTER TABLE eod_pnl ADD COLUMN spy_dividend_per_share REAL",
    # Integrity-gate outcomes, persisted BEFORE the run raises so a red
    # pipeline never costs us the evidence that made it red.
    "ALTER TABLE eod_pnl ADD COLUMN integrity_breach_json TEXT",
    # The broker NAV as received and the repair applied to it. A published NAV
    # that silently differs from what IB sent is untraceable; these three make
    # the correction inspectable from the ledger alone
    # (alpha-engine-config-I9627).
    "ALTER TABLE eod_pnl ADD COLUMN nav_ib_raw_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN nav_mark_correction_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN nav_mark_correction_json TEXT",
    # accrued_today - released_today; the term that keeps the reconciliation
    # identity closed on BOTH the ex-date and the pay date once dividends are
    # accrued on the ex-date against a cash-basis NAV. See
    # CREATE_DIVIDEND_ACCRUALS_TABLE above.
    "ALTER TABLE eod_pnl ADD COLUMN dividend_timing_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN dividend_receivable_usd REAL",
    # ── NAV basis shadow (alpha-engine-config-I9638) ──────────────────────
    # The ruled cut-over from broker NetLiquidation to a settled-close NAV is
    # staged: both figures are computed EVERY run under either basis and the
    # difference is published for two weeks before the default flips.
    # `nav_basis` names which one `portfolio_nav` above actually carries, so a
    # row is never ambiguous about what its headline NAV means — including
    # across the cut-over, where the NAV series changes definition mid-history.
    # nav_ib_usd is the POST-correction broker figure (nav_ib_raw_usd is the
    # pre-correction one), retained as the broker cross-check under both bases.
    "ALTER TABLE eod_pnl ADD COLUMN nav_basis TEXT",
    "ALTER TABLE eod_pnl ADD COLUMN nav_ib_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN nav_settled_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN nav_basis_diff_usd REAL",
    "ALTER TABLE eod_pnl ADD COLUMN nav_basis_diff_bps REAL",
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
# ── Dividend accrual ledger (alpha-engine-config-I8188, defect 3) ──────────
# Dividends are accrued on the EX-DATE (the GIPS-correct treatment, and the
# date on which the settled close drops by the distribution), but the headline
# NAV is IB NetLiquidation, which is CASH-BASIS on this account: no per-symbol
# AccruedDividend is reported and AccruedCash carries interest only. So the
# cash reaches NAV weeks later, on the PAY date.
#
# Without a ledger, ex-date accrual would trade one attribution error for a
# timing one: the residual would read -dividend on the ex-date and +dividend on
# the pay date instead of simply +dividend once. This table is the receivable
# that closes that gap. Each ex-date accrual is recorded with its pay date and
# its dollar amount computed from the shares actually entitled; the reconcile
# releases it on the pay date and reports the day's
# ``dividend_timing_usd = accrued_today - released_today``, which is exactly
# the term that makes the reconciliation identity hold on BOTH dates.
#
# ``accrual_id`` is ``ticker|ex_date`` so a re-reconcile of the same session is
# idempotent — INSERT OR IGNORE, never a second accrual for the same event.
CREATE_DIVIDEND_ACCRUALS_TABLE = """
CREATE TABLE IF NOT EXISTS dividend_accruals (
    accrual_id      TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    ex_date         TEXT NOT NULL,
    pay_date        TEXT,
    per_share       REAL,
    shares          REAL,
    amount_usd      REAL,
    settled_date    TEXT,
    created_at      TEXT NOT NULL
);
"""


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
        + CREATE_DIVIDEND_ACCRUALS_TABLE
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
            # (a) the session-axis check itself could not run — lib not
            # yet bumped / unparseable fill_time (existing comment,
            # narrower than the ValueError branch above, which reports an
            # actual mis-key). The trade row is still inserted either way
            # (see the "Deliberate swallow" comment above this block) —
            # this branch means the check was skipped, not that it passed.
            # (c) recorded at WARNING (visible at the INFO root level,
            # unlike the DEBUG this replaces) — the app log stream is the
            # recording surface. Found by the class guard
            # (tests/test_no_debug_only_swallows.py) beyond the issue's
            # original 28-site count, since a comment line between
            # `except` and the log call hid it from a literal
            # `grep -A1 "except Exception"` (alpha-engine-config-I10031).
            logger.warning("session axis check skipped (best-effort): %s", _e)
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
            stance, catalyst_date, entry_id, commission_usd, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            trade.get("commission_usd"),
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


def _as_int_flag(value) -> int | None:
    """SQLite has no BOOLEAN: persist a tri-state flag as 1/0/NULL.

    NULL means "not evaluated this run" and is deliberately distinct from 0
    ("evaluated, and the answer is no") — the whole class of defect
    alpha-engine-config-I8188 records is an absent measurement rendering
    identically to a measured zero.
    """
    if value is None:
        return None
    return 1 if value else 0


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
             pricing_timing_usd, rotation_realized_usd, unattributed_true_usd,
             commission_usd, slippage_usd, traded_notional_usd,
             commission_available, daily_return_net_pct, daily_return_gross_pct,
             dividend_accrual_available, spy_dividend_per_share,
             integrity_breach_json, dividend_timing_usd,
             dividend_receivable_usd,
             nav_ib_raw_usd, nav_mark_correction_usd, nav_mark_correction_json,
             nav_basis, nav_ib_usd, nav_settled_usd, nav_basis_diff_usd,
             nav_basis_diff_bps,
             created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            eod.get("pricing_timing_usd"),
            eod.get("rotation_realized_usd"),
            eod.get("unattributed_true_usd"),
            eod.get("commission_usd"),
            eod.get("slippage_usd"),
            eod.get("traded_notional_usd"),
            _as_int_flag(eod.get("commission_available")),
            eod.get("daily_return_net_pct"),
            eod.get("daily_return_gross_pct"),
            _as_int_flag(eod.get("dividend_accrual_available")),
            eod.get("spy_dividend_per_share"),
            eod.get("integrity_breach_json"),
            eod.get("dividend_timing_usd"),
            eod.get("dividend_receivable_usd"),
            eod.get("nav_ib_raw_usd"),
            eod.get("nav_mark_correction_usd"),
            eod.get("nav_mark_correction_json"),
            eod.get("nav_basis"),
            eod.get("nav_ib_usd"),
            eod.get("nav_settled_usd"),
            eod.get("nav_basis_diff_usd"),
            eod.get("nav_basis_diff_bps"),
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


class StaleAuditBackupError(RuntimeError):
    """The snapshot about to become the durable audit copy is missing the
    run date's ``eod_pnl`` row. Raised rather than warned: ``backup_to_s3``
    is a PRODUCER of the audit record, and a backup that silently ships the
    previous day's state is indistinguishable from a healthy one.
    """


def _snapshot_db(db_path: str, dest_path: str) -> None:
    """Write a transactionally consistent copy of ``db_path`` to ``dest_path``.

    WHY THIS EXISTS (alpha-engine-config-I8735). ``init_db`` runs
    ``PRAGMA journal_mode=WAL`` (added 2026-04-03, 77ad9d2). In WAL mode a
    COMMIT appends to the ``-wal`` sidecar; the main database file only
    absorbs those frames at a checkpoint. ``s3.upload_file(db_path, ...)``
    copies the main file ALONE — never the ``-wal`` — so the uploaded audit
    copy contains whatever the last checkpoint left behind, which on a day
    where no auto-checkpoint happened to fire is the PREVIOUS session's
    state. Measured: 59 of 100 WAL-era daily backups were missing their own
    day's ``eod_pnl`` row; 0 of 22 pre-WAL ones were.

    ``Connection.backup`` is SQLite's online-backup API: it reads through
    the WAL and produces a single self-contained file with no sidecars, so
    the artifact is correct regardless of journal mode, checkpoint timing or
    a concurrent writer. It replaces the raw file copy rather than papering
    over it with a checkpoint call, because a checkpoint can be blocked by
    any concurrent reader and would leave the same silent staleness behind.
    """
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(dest_path)
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()


def _assert_snapshot_has_eod_row(snapshot_path: str, run_date: str) -> None:
    """Raise unless the snapshot carries an ``eod_pnl`` row for ``run_date``."""
    conn = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
    try:
        present = conn.execute(
            "SELECT 1 FROM eod_pnl WHERE date = ? LIMIT 1", (run_date,)
        ).fetchone()
        newest = conn.execute("SELECT MAX(date) FROM eod_pnl").fetchone()[0]
    finally:
        conn.close()
    if present:
        return
    raise StaleAuditBackupError(
        f"Refusing to publish the trades audit backup for run_date={run_date}: "
        f"the snapshot has no eod_pnl row for that date (newest row is "
        f"{newest!r}). The durable audit copy would have been stale and "
        "nothing downstream could have told."
    )


def backup_to_s3(
    db_path: str,
    run_date: str,
    s3_bucket: str,
    *,
    require_eod_row: bool = False,
    fail_loud: bool = False,
) -> None:
    """Publish a consistent snapshot of trades.db to
    ``trades/trades_{run_date}.db`` and ``trades/trades_latest.db``.

    ``require_eod_row`` — verify the snapshot contains the run date's
    ``eod_pnl`` row before uploading anything, and raise
    :class:`StaleAuditBackupError` if it does not. Set by the EOD reconcile,
    the one caller for which this file IS the day's audit record. The
    intraday/midday/emergency callers legitimately run before any
    ``eod_pnl`` row exists for the day and leave it False.

    ``fail_loud`` — re-raise upload failures instead of logging them. Set by
    the EOD reconcile for the same reason. Left False for the intraday
    callers, where (a) the failure mode swallowed is a transient S3 upload
    error on a SECONDARY snapshot, (b) the primary deliverable — trading and
    the EOD reconcile's own backup — is unaffected, and (c) the recording
    surface is the ERROR log line below plus the next EOD backup, which is
    strict. A verification failure is NEVER swallowed on either path.
    """
    fd, snapshot_path = tempfile.mkstemp(prefix="trades_snapshot_", suffix=".db")
    os.close(fd)
    try:
        _snapshot_db(db_path, snapshot_path)
        if require_eod_row:
            # Raised before the first upload, so a stale snapshot never
            # reaches either key — including trades_latest.db, which is what
            # every downstream reader resolves.
            _assert_snapshot_has_eod_row(snapshot_path, run_date)
        try:
            s3 = boto3.client("s3")
            key = f"trades/trades_{run_date}.db"
            s3.upload_file(snapshot_path, s3_bucket, key)
            logger.info(f"trades.db backed up to s3://{s3_bucket}/{key}")
            s3.upload_file(snapshot_path, s3_bucket, "trades/trades_latest.db")
            logger.info(f"trades.db backed up to s3://{s3_bucket}/trades/trades_latest.db")
        except Exception as e:
            logger.error("S3 backup failed: %s", e)
            if fail_loud:
                raise
    finally:
        try:
            os.unlink(snapshot_path)
        except OSError:  # pragma: no cover — temp file already gone
            pass


# ── Dividend accrual ledger helpers (alpha-engine-config-I8188) ────────────

def record_dividend_accrual(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    ex_date: str,
    pay_date: str | None,
    per_share: float,
    shares: float,
    amount_usd: float,
) -> bool:
    """Record one ex-date dividend accrual. Returns True when newly inserted.

    Idempotent on ``ticker|ex_date`` so re-reconciling a session (the canonical
    correction path) cannot accrue the same distribution twice.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO dividend_accruals "
        "(accrual_id, ticker, ex_date, pay_date, per_share, shares, "
        " amount_usd, settled_date, created_at) "
        "VALUES (?,?,?,?,?,?,?,NULL,?)",
        (f"{ticker}|{ex_date}", ticker, ex_date, pay_date, per_share, shares,
         amount_usd, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return cur.rowcount > 0


# A distribution whose feed record carries no pay date still settles into cash;
# releasing it never would leave the receivable growing forever and the
# residual permanently short by that amount. US equity pay dates run roughly
# 2-4 weeks after the ex-date, so 30 calendar days releases late rather than
# early -- and the day it lands is named in the log either way.
DIVIDEND_PAY_DATE_FALLBACK_DAYS = 30


def settle_due_dividend_accruals(
    conn: sqlite3.Connection, run_date: str
) -> tuple[float, list[dict]]:
    """Release every accrual whose pay date has arrived. Returns (usd, rows).

    Called once per reconcile. An accrual with a NULL pay date is released
    ``DIVIDEND_PAY_DATE_FALLBACK_DAYS`` after its ex-date rather than being
    stranded in the receivable forever.
    """
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    rows = conn.execute(
        "SELECT accrual_id, ticker, ex_date, pay_date, amount_usd "
        "FROM dividend_accruals WHERE settled_date IS NULL"
    ).fetchall()
    due: list[dict] = []
    for accrual_id, ticker, ex_date, pay_date, amount_usd in rows:
        effective = pay_date
        if not effective:
            try:
                effective = str(
                    _date.fromisoformat(ex_date)
                    + _timedelta(days=DIVIDEND_PAY_DATE_FALLBACK_DAYS)
                )
            except (TypeError, ValueError):
                continue
        if effective > run_date:
            continue
        due.append({
            "accrual_id": accrual_id, "ticker": ticker, "ex_date": ex_date,
            "pay_date": pay_date, "effective_pay_date": effective,
            "amount_usd": float(amount_usd or 0.0),
        })
    for row in due:
        conn.execute(
            "UPDATE dividend_accruals SET settled_date = ? WHERE accrual_id = ?",
            (run_date, row["accrual_id"]),
        )
    if due:
        conn.commit()
    return sum(r["amount_usd"] for r in due), due


def dividend_receivable_usd(conn: sqlite3.Connection) -> float:
    """Total accrued-but-not-yet-paid dividend currently outstanding."""
    value = conn.execute(
        "SELECT COALESCE(SUM(amount_usd), 0.0) FROM dividend_accruals "
        "WHERE settled_date IS NULL"
    ).fetchone()[0]
    return float(value or 0.0)
