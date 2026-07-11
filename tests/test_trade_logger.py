"""Tests for executor.trade_logger — DB I/O + S3 backup + entry-lookup helpers."""

import json
import sqlite3
import sys
from unittest.mock import MagicMock, patch

import pytest

from executor.trade_logger import (
    backup_to_s3,
    get_entry_dates,
    get_entry_stance_and_catalyst,
    get_entry_trade,
    get_todays_trades,
    get_unmatched_entry,
    init_db,
    log_eod,
    log_risk_event,
    log_shadow_book_block,
    log_trade,
)


@pytest.fixture
def db(tmp_path):
    return init_db(str(tmp_path / "trades.db"))


def _trade(date="2026-04-15", ticker="AAPL", action="ENTER", shares=100, **kw):
    base = {"date": date, "ticker": ticker, "action": action, "shares": shares}
    base.update(kw)
    return base


# ── init_db ─────────────────────────────────────────────────────────────────


def test_init_db_creates_expected_tables(tmp_path):
    conn = init_db(str(tmp_path / "trades.db"))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "trades" in tables
    assert "eod_pnl" in tables
    assert "executor_shadow_book" in tables
    assert "risk_events" in tables


def test_init_db_idempotent_re_run(tmp_path):
    """Re-initializing must not raise — duplicate-column ALTER errors are absorbed."""
    path = str(tmp_path / "trades.db")
    init_db(path).close()
    init_db(path).close()  # second run must succeed without raising


def test_init_db_propagates_unexpected_migration_error(tmp_path, monkeypatch):
    """A non-duplicate-column OperationalError must surface."""
    import executor.trade_logger as mod

    # Use an out-of-band migration list that triggers an unexpected error
    bad_migrations = ["ALTER TABLE trades NOT_A_REAL_OPERATION"]
    monkeypatch.setattr(mod, "_TRADES_MIGRATIONS", bad_migrations)

    with pytest.raises(sqlite3.OperationalError):
        init_db(str(tmp_path / "trades.db"))


def test_init_db_propagates_unexpected_eod_migration_error(tmp_path, monkeypatch):
    import executor.trade_logger as mod
    monkeypatch.setattr(mod, "_EOD_MIGRATIONS", ["ALTER TABLE eod_pnl NOT_A_REAL_OP"])
    with pytest.raises(sqlite3.OperationalError):
        init_db(str(tmp_path / "trades.db"))


def test_init_db_propagates_unexpected_risk_events_migration_error(tmp_path, monkeypatch):
    import executor.trade_logger as mod
    monkeypatch.setattr(mod, "_RISK_EVENTS_MIGRATIONS", ["ALTER TABLE risk_events NOT_A_REAL_OP"])
    with pytest.raises(sqlite3.OperationalError):
        init_db(str(tmp_path / "trades.db"))


def test_init_db_risk_events_migration_idempotent(tmp_path, monkeypatch):
    """A real migration added to the placeholder list should run once + be
    idempotent on re-init (duplicate-column branch)."""
    import executor.trade_logger as mod
    monkeypatch.setattr(
        mod, "_RISK_EVENTS_MIGRATIONS",
        ["ALTER TABLE risk_events ADD COLUMN test_col TEXT"],
    )
    path = str(tmp_path / "trades.db")
    init_db(path).close()
    # Second init must succeed (duplicate-column path)
    conn = init_db(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(risk_events)").fetchall()}
    assert "test_col" in cols


def test_log_risk_event_falls_through_when_lib_unavailable(tmp_path):
    """The optional nousergon_lib.dates import must NOT hard-fail."""
    conn = init_db(str(tmp_path / "trades.db"))
    fake_lib = MagicMock()
    fake_lib.now_dual.side_effect = RuntimeError("simulated import failure")
    sys.modules.pop("nousergon_lib.dates", None)
    sys.modules["nousergon_lib.dates"] = fake_lib

    try:
        event_id = log_risk_event(conn, {
            "date": "2026-04-15", "event_type": "veto", "rule": "min_score",
        })
        td = conn.execute(
            "SELECT trading_day FROM risk_events WHERE event_id=?", (event_id,),
        ).fetchone()[0]
        assert td is None
    finally:
        sys.modules.pop("nousergon_lib.dates", None)


# ── log_trade ───────────────────────────────────────────────────────────────


def test_log_trade_returns_uuid_and_inserts_row(db):
    trade_id = log_trade(db, _trade())
    assert isinstance(trade_id, str) and len(trade_id) == 36
    rows = db.execute("SELECT trade_id, ticker, action, shares FROM trades").fetchall()
    assert len(rows) == 1
    assert rows[0] == (trade_id, "AAPL", "ENTER", 100)


def test_log_trade_uses_caller_trading_day_when_provided(db):
    log_trade(db, _trade(trading_day="2026-04-14"))
    td = db.execute("SELECT trading_day FROM trades").fetchone()[0]
    assert td == "2026-04-14"


def test_log_trade_falls_through_when_lib_unavailable(db, monkeypatch):
    """If the optional lib raises on import, trading_day stays NULL — no hard fail."""
    # Force the inline `from nousergon_lib.dates import now_dual` to raise
    fake_lib = MagicMock()
    fake_lib.now_dual.side_effect = RuntimeError("simulated import failure")
    # Inject into sys.modules so the inline import picks it up — we need to
    # invalidate the cached import first
    sys.modules.pop("nousergon_lib.dates", None)
    sys.modules["nousergon_lib.dates"] = fake_lib

    try:
        log_trade(db, _trade())  # no trading_day provided
        td = db.execute("SELECT trading_day FROM trades").fetchone()[0]
        assert td is None
    finally:
        sys.modules.pop("nousergon_lib.dates", None)


# ── log_shadow_book_block ───────────────────────────────────────────────────


def test_log_shadow_book_block_inserts_row(db):
    shadow_id = log_shadow_book_block(db, {
        "date": "2026-04-15", "ticker": "AAPL", "block_reason": "low_score",
        "research_score": 45.0, "intended_position_pct": 0.05,
    })
    rows = db.execute(
        "SELECT shadow_id, ticker, block_reason FROM executor_shadow_book"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == (shadow_id, "AAPL", "low_score")


# ── log_risk_event ──────────────────────────────────────────────────────────


def test_log_risk_event_serializes_context_to_json(db):
    event_id = log_risk_event(db, {
        "date": "2026-04-15",
        "event_type": "throttle",
        "rule": "drawdown_tier_throttle",
        "ticker": "AAPL",
        "value": 0.85,
        "context": {"breached_tier": "tier_2", "tier_size_multiplier": 0.5},
    })
    row = db.execute(
        "SELECT event_id, event_type, rule, ticker, context_json FROM risk_events"
    ).fetchone()
    assert row[0] == event_id
    assert row[1] == "throttle"
    assert row[2] == "drawdown_tier_throttle"
    assert row[3] == "AAPL"
    ctx = json.loads(row[4])
    assert ctx == {"breached_tier": "tier_2", "tier_size_multiplier": 0.5}


def test_log_risk_event_null_context_persists_as_null(db):
    log_risk_event(db, {"date": "2026-04-15", "event_type": "veto", "rule": "min_score"})
    ctx = db.execute("SELECT context_json FROM risk_events").fetchone()[0]
    assert ctx is None


def test_catastrophic_gap_watch_cohort_is_queryable(db):
    """config#846: the daemon flushes one row per gap-only position — worst
    drop vs reference, whether the stop fired, threshold in effect — so the
    offline catastrophic_gap_stop_pct tuner has a realized-outcome cohort to
    join. Validate the persisted shape round-trips queryably."""
    log_risk_event(db, {
        "date": "2026-04-15",
        "event_type": "catastrophic_gap_watch",
        "rule": "catastrophic_gap_stop",
        "ticker": "AAPL",
        "value": 0.083,          # worst drop
        "threshold": 0.15,       # pct in effect
        "reason": "watched",     # did not fire
        "context": {"max_drop": 0.083, "reference_price": 200.0,
                    "price_at_max_drop": 183.4, "fired": False},
    })
    log_risk_event(db, {
        "date": "2026-04-15",
        "event_type": "catastrophic_gap_watch",
        "rule": "catastrophic_gap_stop",
        "ticker": "NVDA",
        "value": 0.19,
        "threshold": 0.15,
        "reason": "fired",
        "context": {"max_drop": 0.19, "reference_price": 100.0,
                    "price_at_max_drop": 81.0, "fired": True},
    })
    rows = db.execute(
        "SELECT ticker, value, threshold, reason, context_json FROM risk_events "
        "WHERE event_type='catastrophic_gap_watch' ORDER BY ticker"
    ).fetchall()
    assert len(rows) == 2
    aapl, nvda = rows
    assert aapl[0] == "AAPL" and aapl[3] == "watched"
    assert json.loads(aapl[4])["fired"] is False
    assert nvda[0] == "NVDA" and nvda[3] == "fired"
    assert json.loads(nvda[4])["fired"] is True
    # The tuner's core query: which watched positions crossed a candidate
    # threshold — answerable entirely from the persisted cohort.
    crossed = db.execute(
        "SELECT ticker FROM risk_events "
        "WHERE event_type='catastrophic_gap_watch' AND value >= 0.15"
    ).fetchall()
    assert [r[0] for r in crossed] == ["NVDA"]


def test_intraday_resolve_event_is_queryable(db):
    """config#846: each intraday cash-resolution event records freed cash,
    solver status, redeploy count, and the window/cap params — the cohort the
    intraday_resolve_* tuner needs."""
    event_id = log_risk_event(db, {
        "date": "2026-04-15",
        "event_type": "intraday_resolve",
        "rule": "intraday_resolve",
        "value": 4200.0,           # freed cash
        "threshold": 1000.0,       # min_freed_cash_pct * nav
        "reason": "optimal",
        "context": {"resolve_count": 1, "n_redeployed": 2, "solve_status": "optimal",
                    "vol_ann": 0.14, "nav": 100_000.0, "min_freed_cash_pct": 0.01,
                    "cutoff_et": "15:30", "max_per_day": 5,
                    "redeploy_tickers": ["MSFT", "GOOGL"]},
    })
    row = db.execute(
        "SELECT event_id, value, reason, context_json FROM risk_events "
        "WHERE event_type='intraday_resolve'"
    ).fetchone()
    assert row[0] == event_id
    assert row[1] == 4200.0
    assert row[2] == "optimal"
    ctx = json.loads(row[3])
    assert ctx["redeploy_tickers"] == ["MSFT", "GOOGL"]
    assert ctx["max_per_day"] == 5


# ── log_eod ─────────────────────────────────────────────────────────────────


def test_log_eod_insert_or_replace(db):
    log_eod(db, {
        "date": "2026-04-15",
        "portfolio_nav": 100_000.0,
        "daily_return_pct": 0.5,
        "spy_return_pct": 0.3,
        "daily_alpha_pct": 0.2,
        "positions_snapshot": {"AAPL": 100},
        "unattributed_residual_pct": 0.45,
    })

    # Replace via INSERT OR REPLACE
    log_eod(db, {
        "date": "2026-04-15",
        "portfolio_nav": 100_500.0,
        "daily_return_pct": 0.6,
        "spy_return_pct": 0.3,
        "daily_alpha_pct": 0.3,
    })

    rows = db.execute("SELECT portfolio_nav FROM eod_pnl WHERE date=?", ("2026-04-15",)).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 100_500.0


# ── entry-lookup helpers ────────────────────────────────────────────────────


def test_get_entry_dates_returns_only_existing_enters(db):
    log_trade(db, _trade(ticker="AAPL", date="2026-04-01"))
    log_trade(db, _trade(ticker="AAPL", date="2026-04-10"))  # newer ENTER
    log_trade(db, _trade(ticker="MSFT", date="2026-04-05"))
    log_trade(db, _trade(ticker="AAPL", date="2026-04-15", action="EXIT"))

    result = get_entry_dates(db, ["AAPL", "MSFT", "NOPE"])

    assert result == {"AAPL": "2026-04-10", "MSFT": "2026-04-05"}
    assert "NOPE" not in result


def test_get_entry_stance_and_catalyst_returns_most_recent(db):
    log_trade(db, _trade(ticker="AAPL", date="2026-04-01",
                         stance="momentum", catalyst_date=None))
    log_trade(db, _trade(ticker="AAPL", date="2026-04-10",
                         stance="catalyst", catalyst_date="2026-04-22"))
    log_trade(db, _trade(ticker="MSFT", date="2026-04-05",
                         stance="quality"))

    result = get_entry_stance_and_catalyst(db, ["AAPL", "MSFT", "NOPE"])
    assert result["AAPL"] == {"stance": "catalyst", "catalyst_date": "2026-04-22"}
    assert result["MSFT"] == {"stance": "quality", "catalyst_date": None}
    assert "NOPE" not in result


def test_get_todays_trades_returns_only_run_date(db):
    log_trade(db, _trade(ticker="AAPL", date="2026-04-15"))
    log_trade(db, _trade(ticker="MSFT", date="2026-04-15"))
    log_trade(db, _trade(ticker="GOOGL", date="2026-04-16"))

    trades = get_todays_trades(db, "2026-04-15")
    assert len(trades) == 2
    assert {t["ticker"] for t in trades} == {"AAPL", "MSFT"}


def test_get_entry_trade_picks_most_recent_enter(db):
    log_trade(db, _trade(ticker="AAPL", date="2026-04-01"))
    log_trade(db, _trade(ticker="AAPL", date="2026-04-10"))

    entry = get_entry_trade(db, "AAPL")
    assert entry is not None
    assert entry["date"] == "2026-04-10"


def test_get_entry_trade_returns_none_when_no_enter(db):
    assert get_entry_trade(db, "AAPL") is None


# ── get_unmatched_entry ─────────────────────────────────────────────────────


def test_get_unmatched_entry_finds_entry_with_remaining_shares(db):
    entry_id = log_trade(db, _trade(ticker="AAPL", date="2026-04-01", shares=100))
    # Partial reduce of 30
    log_trade(db, _trade(ticker="AAPL", date="2026-04-05", action="REDUCE",
                         shares=30, entry_trade_id=entry_id))

    result = get_unmatched_entry(db, "AAPL")
    assert result is not None
    assert result["trade_id"] == entry_id
    assert result["shares_remaining"] == 70


def test_get_unmatched_entry_returns_none_when_fully_matched(db):
    entry_id = log_trade(db, _trade(ticker="AAPL", date="2026-04-01", shares=100))
    log_trade(db, _trade(ticker="AAPL", date="2026-04-05", action="EXIT",
                         shares=100, entry_trade_id=entry_id))

    assert get_unmatched_entry(db, "AAPL") is None


def test_get_unmatched_entry_returns_none_with_no_entries(db):
    assert get_unmatched_entry(db, "AAPL") is None


# ── backup_to_s3 ────────────────────────────────────────────────────────────


def test_backup_to_s3_uploads_dated_and_latest(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    init_db(str(db_path)).close()

    fake_s3 = MagicMock()
    monkeypatch.setattr("executor.trade_logger.boto3.client", MagicMock(return_value=fake_s3))

    backup_to_s3(str(db_path), "2026-04-15", "bucket-x")

    assert fake_s3.upload_file.call_count == 2
    keys = [c.args[2] for c in fake_s3.upload_file.call_args_list]
    assert "trades/trades_2026-04-15.db" in keys
    assert "trades/trades_latest.db" in keys


def test_backup_to_s3_swallows_failures(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "trades.db"
    init_db(str(db_path)).close()

    fake_s3 = MagicMock()
    fake_s3.upload_file.side_effect = RuntimeError("S3 down")
    monkeypatch.setattr("executor.trade_logger.boto3.client", MagicMock(return_value=fake_s3))

    with caplog.at_level("ERROR"):
        backup_to_s3(str(db_path), "2026-04-15", "bucket-x")
    assert any("S3 backup failed" in r.message for r in caplog.records)
