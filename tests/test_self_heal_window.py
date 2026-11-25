"""Post-close self-heal window start (alpha-engine-config-I14373).

The window starts at the OLDEST day inside the 63-session gate window that
still carries an integrity breach, never more than 21 trading days in total,
dates still run oldest to newest. No tolerance, bound or gate is touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from executor import reconcile_audit
from executor.reconcile_audit import DEFAULT_TRAILING_DAYS, _window_dates, audit_window
from executor.self_heal_window import (
    SELF_HEAL_MAX_WINDOW_DAYS,
    TRAILING_DAYS,
    WINDOW_REASON_OLDEST_OPEN_BREACH,
    WINDOW_REASON_TRAILING,
    SelfHealWindow,
    resolve_self_heal_window,
)
from executor.trade_logger import init_db

END = "2026-11-27"
BREACH = json.dumps([{"kind": "residual_cumulative", "message": "x"}])


def _sessions(n: int = 70) -> list[str]:
    """Ascending trading days ending at END (END is the last element)."""
    return _window_dates(start=None, end=END, trailing_days=n)


def _seed(db: str, breaches: dict[str, str | None] | None = None, n: int = 70) -> list[str]:
    days = _sessions(n)[:-1]  # END itself has no row yet at post-close start
    conn = init_db(db)
    for d in days:
        conn.execute(
            "INSERT OR REPLACE INTO eod_pnl (date, portfolio_nav, spy_close, spy_return_pct, "
            "daily_return_pct, daily_alpha_pct, created_at, total_cash, nav_basis, "
            "integrity_breach_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d, 1_000_000.0, 700.0, 0.0, 0.0, 0.0, f"{d}T20:00:00", 1_000_000.0,
             "settled_close", (breaches or {}).get(d)),
        )
    conn.commit()
    conn.close()
    return days


def _back(k: int) -> str:
    """The trading day k sessions before END."""
    return _sessions(k + 1)[0]


class TestResolveWindow:
    def test_trailing_constant_mirrors_reconcile_audit(self) -> None:
        assert TRAILING_DAYS == DEFAULT_TRAILING_DAYS

    def test_no_breach_rows_is_exactly_trailing_5(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        _seed(db)
        w = resolve_self_heal_window(db_path=db, end=END)
        assert w == SelfHealWindow(start=_sessions(5)[0], reason=WINDOW_REASON_TRAILING,
                                   audit_start=None, left_out=())
        # audit_start None means audit_window runs exactly as before.

    def test_empty_breach_values_do_not_extend(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        _seed(db, {_back(8): "[]", _back(9): "", _back(10): None})
        assert resolve_self_heal_window(db_path=db, end=END).reason == WINDOW_REASON_TRAILING

    def test_breach_8_sessions_back_extends_to_it(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        _seed(db, {_back(8): BREACH, _back(6): BREACH})
        w = resolve_self_heal_window(db_path=db, end=END)
        assert w.start == _back(8) and w.audit_start == _back(8)
        assert w.reason == WINDOW_REASON_OLDEST_OPEN_BREACH
        assert w.left_out == ()

    def test_breach_30_sessions_back_is_capped_and_warns_with_dates(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = str(tmp_path / "t.db")
        old1, old2 = _back(30), _back(25)
        _seed(db, {old1: BREACH, old2: BREACH})
        with caplog.at_level(logging.WARNING, logger="executor.self_heal_window"):
            w = resolve_self_heal_window(db_path=db, end=END)
        # Nothing in reach: the run stays trailing, and says what it left out.
        assert w.reason == WINDOW_REASON_TRAILING and w.audit_start is None
        assert w.left_out == (old1, old2)
        warned = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert old1 in warned and old2 in warned

    def test_cap_never_exceeds_21_trading_days(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        in_reach = _sessions(SELF_HEAL_MAX_WINDOW_DAYS)[0]
        out_of_reach = _back(SELF_HEAL_MAX_WINDOW_DAYS)
        _seed(db, {out_of_reach: BREACH, in_reach: BREACH})
        w = resolve_self_heal_window(db_path=db, end=END)
        assert w.start == in_reach
        assert w.left_out == (out_of_reach,)
        assert len(_window_dates(start=w.audit_start, end=END, trailing_days=5)) == 21

    def test_exclude_dates_are_not_counted(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        _seed(db, {_back(8): BREACH})
        w = resolve_self_heal_window(db_path=db, end=END, exclude_dates={_back(8)})
        assert w.reason == WINDOW_REASON_TRAILING

    def test_unreadable_store_falls_back_to_trailing(self, tmp_path: Path) -> None:
        w = resolve_self_heal_window(db_path=str(tmp_path / "missing.db"), end=END)
        assert w.reason == WINDOW_REASON_TRAILING and w.audit_start is None

    def test_breach_older_than_63_sessions_is_not_read(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        _seed(db, {_back(64): BREACH}, n=70)
        w = resolve_self_heal_window(db_path=db, end=END)
        assert w.reason == WINDOW_REASON_TRAILING and w.left_out == ()


class TestRunsOldestFirstAndClears:
    def _run(self, db: str, settled: dict[str, float], w: SelfHealWindow) -> tuple[dict, list[str]]:
        order: list[str] = []

        def fake_run(d: str, *, send_email: bool, run_audit: bool, post_journal: bool = True) -> None:
            order.append(d)
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=?, integrity_breach_json=NULL WHERE date=?",
                         (settled[d], d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(end=END, start=w.audit_start, exclude_dates={END},
                               config={"db_path": db, "trades_bucket": "", "aws_region": "us-east-1"})
        return res, order

    def test_extended_window_repairs_oldest_first_then_stops_extending(self, tmp_path: Path) -> None:
        db = str(tmp_path / "t.db")
        bad = [_back(8), _back(7), _back(6)]
        _seed(db, dict.fromkeys(bad, BREACH))
        conn = init_db(db)
        for d in bad:
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (690.0, d))
        conn.commit()
        conn.close()
        settled = dict.fromkeys(_sessions(70), 700.0)

        w = resolve_self_heal_window(db_path=db, end=END, exclude_dates={END})
        assert w.start == bad[0]
        res, order = self._run(db, settled, w)
        assert res["window"][0] == bad[0]
        assert order[:3] == bad                      # oldest to newest
        assert order == sorted(order)
        # the day after the last repaired one is re-reconciled through prior_corrected
        assert _back(5) in order

        # Breaches cleared by the repair no longer extend the next run.
        again = resolve_self_heal_window(db_path=db, end=END, exclude_dates={END})
        assert again.reason == WINDOW_REASON_TRAILING and again.audit_start is None

    def test_breach_outside_cap_is_still_reported_not_repaired(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = str(tmp_path / "t.db")
        old = _back(30)
        _seed(db, {old: BREACH})
        with caplog.at_level(logging.WARNING, logger="executor.self_heal_window"):
            w = resolve_self_heal_window(db_path=db, end=END, exclude_dates={END})
        settled = dict.fromkeys(_sessions(70), 700.0)
        res, order = self._run(db, settled, w)
        assert old not in res["window"] and order == []
        conn = init_db(db)
        stored = conn.execute("SELECT integrity_breach_json FROM eod_pnl WHERE date=?", (old,)).fetchone()[0]
        conn.close()
        assert stored == BREACH                       # left exactly as stored, still standing
        assert old in " ".join(r.getMessage() for r in caplog.records)
