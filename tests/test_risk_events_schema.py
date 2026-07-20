"""Schema + log_risk_event tests for the structured risk-event log.

Phase 2 transparency-inventory: closes the *risk decisions* row.
Sibling of `executor_shadow_book` — same family, different axis.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from executor.trade_logger import init_db, log_risk_event


@pytest.fixture
def conn():
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    c = init_db(db_path)
    yield c
    c.close()
    os.unlink(db_path)


class TestRiskEventsSchema:
    def test_table_created_on_init_db(self, conn):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_events'"
        ).fetchall()
        assert rows, "risk_events table should exist after init_db"

    def test_init_db_idempotent(self):
        """Re-running init_db on the same path must not raise."""
        db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        try:
            c1 = init_db(db_path)
            c1.close()
            # Second init on the same file — must be a no-op for the schema.
            c2 = init_db(db_path)
            c2.close()
        finally:
            os.unlink(db_path)

    def test_columns_present(self, conn):
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(risk_events)").fetchall()
        }
        expected = {
            "event_id", "date", "trading_day", "event_type", "rule",
            "ticker", "sector", "reason", "value", "threshold",
            "market_regime", "signal_date", "prediction_date",
            "context_json", "created_at",
        }
        assert expected.issubset(cols), f"missing columns: {expected - cols}"


class TestLogRiskEvent:
    def test_minimal_event_inserts(self, conn):
        eid = log_risk_event(conn, {
            "date": "2026-05-06",
            "event_type": "veto",
            "rule": "min_score",
        })
        row = conn.execute(
            "SELECT event_type, rule, ticker, value, threshold "
            "FROM risk_events WHERE event_id=?",
            (eid,),
        ).fetchone()
        assert row == ("veto", "min_score", None, None, None)

    def test_full_event_roundtrip(self, conn):
        eid = log_risk_event(conn, {
            "date": "2026-05-06",
            "event_type": "veto",
            "rule": "max_position",
            "ticker": "NVDA",
            "sector": "Technology",
            "reason": "Position size 6.2% exceeds max 5.0%",
            "value": 0.062,
            "threshold": 0.05,
            "market_regime": "neutral",
            "signal_date": "2026-05-04",
            "prediction_date": "2026-05-05",
            "context": {"dollar_size": 6200.0, "portfolio_nav": 100000.0},
        })
        row = conn.execute(
            "SELECT date, event_type, rule, ticker, sector, reason, "
            "value, threshold, market_regime, signal_date, "
            "prediction_date, context_json "
            "FROM risk_events WHERE event_id=?",
            (eid,),
        ).fetchone()
        assert row[0] == "2026-05-06"
        assert row[1] == "veto"
        assert row[2] == "max_position"
        assert row[3] == "NVDA"
        assert row[4] == "Technology"
        assert "6.2%" in row[5]
        assert row[6] == pytest.approx(0.062)
        assert row[7] == pytest.approx(0.05)
        assert row[8] == "neutral"
        assert row[9] == "2026-05-04"
        assert row[10] == "2026-05-05"
        ctx = json.loads(row[11])
        assert ctx["dollar_size"] == 6200.0
        assert ctx["portfolio_nav"] == 100000.0

    def test_event_id_is_uuid(self, conn):
        eid = log_risk_event(conn, {
            "date": "2026-05-06",
            "event_type": "halt",
            "rule": "drawdown_halt",
        })
        # uuid4 is 36 chars with 4 dashes
        assert len(eid) == 36 and eid.count("-") == 4

    def test_no_context_yields_null_json(self, conn):
        eid = log_risk_event(conn, {
            "date": "2026-05-06",
            "event_type": "throttle",
            "rule": "drawdown_tier_throttle",
        })
        row = conn.execute(
            "SELECT context_json FROM risk_events WHERE event_id=?", (eid,)
        ).fetchone()
        assert row[0] is None

    def test_query_by_date_and_rule(self, conn):
        log_risk_event(conn, {"date": "2026-05-06", "event_type": "veto", "rule": "min_score", "ticker": "KO"})
        log_risk_event(conn, {"date": "2026-05-06", "event_type": "veto", "rule": "min_score", "ticker": "HSY"})
        log_risk_event(conn, {"date": "2026-05-06", "event_type": "veto", "rule": "max_sector", "ticker": "AAPL"})
        log_risk_event(conn, {"date": "2026-05-07", "event_type": "veto", "rule": "min_score", "ticker": "NVDA"})

        rows = conn.execute(
            "SELECT ticker FROM risk_events WHERE date=? AND rule=? ORDER BY ticker",
            ("2026-05-06", "min_score"),
        ).fetchall()
        assert [r[0] for r in rows] == ["HSY", "KO"]

    def test_required_keys_missing_raises(self, conn):
        # event_type missing → KeyError on the dict access
        with pytest.raises(KeyError):
            log_risk_event(conn, {"date": "2026-05-06", "rule": "min_score"})
        # rule missing
        with pytest.raises(KeyError):
            log_risk_event(conn, {"date": "2026-05-06", "event_type": "veto"})
        # date missing
        with pytest.raises(KeyError):
            log_risk_event(conn, {"event_type": "veto", "rule": "min_score"})
