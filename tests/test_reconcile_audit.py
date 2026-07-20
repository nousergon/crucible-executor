"""Tests for executor/reconcile_audit.py — T+1 self-heal of EOD market values
that were frozen pre-settlement (config#1276)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from executor import reconcile_audit
from executor.reconcile_audit import _window_dates, audit_window
from executor.trade_logger import init_db


def _seed_eod(db_path, rows):
    """rows: list of (date, spy_close, spy_return_pct, daily_alpha_pct)."""
    conn = init_db(db_path)
    for d, sc, sr, da in rows:
        conn.execute(
            "INSERT OR REPLACE INTO eod_pnl (date, portfolio_nav, spy_close, "
            "spy_return_pct, daily_return_pct, daily_alpha_pct, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (d, 1_000_000.0, sc, sr, 0.0, da, f"{d}T20:00:00"),
        )
    conn.commit()
    conn.close()


def _cfg(db_path):
    return {"db_path": db_path, "trades_bucket": "", "aws_region": "us-east-1"}


# ── _window_dates ─────────────────────────────────────────────────────────────


class TestWindowDates:
    def test_trailing_walks_trading_calendar(self):
        # 06-24/25/26 are consecutive trading days (Wed/Thu/Fri).
        days = _window_dates(start=None, end="2026-06-26", trailing_days=3)
        assert days == ["2026-06-24", "2026-06-25", "2026-06-26"]

    def test_explicit_range_inclusive_ascending(self):
        days = _window_dates(start="2026-06-24", end="2026-06-26", trailing_days=99)
        assert days == ["2026-06-24", "2026-06-25", "2026-06-26"]


# ── audit_window ──────────────────────────────────────────────────────────────


class TestAuditWindow:
    def test_clean_window_no_corrections(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 734.30, 0.10, 1.40), ("2026-06-26", 728.99, -0.72, 0.33)])
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-26", config=_cfg(db))
        assert res["checked"] == 2
        assert res["corrected"] == []
        run_mock.assert_not_called()

    def test_stale_close_triggers_reconcile(self, tmp_path):
        db = str(tmp_path / "t.db")
        # 06-25 stored 733.50 but settled 734.30 → 10.9 bps divergence > tolerance.
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            assert send_email is False and run_audit is False  # never resend / never recurse
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=?, spy_return_pct=?, daily_alpha_pct=? WHERE date=?",
                         (734.30, 0.098, 1.41, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run) as run_mock, \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        run_mock.assert_called_once()
        assert len(res["corrected"]) == 1
        c = res["corrected"][0]
        assert c["date"] == "2026-06-25" and c["reason"] == "stale_close"
        assert c["before"]["spy_close"] == 733.50
        assert c["after"]["spy_close"] == 734.30
        assert c["divergence_bps"] > 10

    def test_within_tolerance_skipped(self, tmp_path):
        db = str(tmp_path / "t.db")
        # 0.3 bp divergence < 1 bp tolerance → no correction.
        _seed_eod(db, [("2026-06-25", 734.30, 0.10, 1.40)])
        settled = {"2026-06-25": 734.32}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["corrected"] == []
        run_mock.assert_not_called()

    def test_missing_row_flagged_as_gap_not_backfilled(self, tmp_path):
        # A missing row must be FLAGGED, never auto-synthesized — ledger-replay
        # backfill can fabricate a wrong NAV (the 06-24 / 20-position incident).
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-23", 733.58, -1.44, 0.0)])  # 06-24 absent → gap
        settled = {"2026-06-24": 733.24}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch("executor.backfill_eod_pnl.backfill") as bf_mock, \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-24", end="2026-06-24", config=_cfg(db))
        bf_mock.assert_not_called()
        run_mock.assert_not_called()
        assert res["corrected"] == []
        assert res["gaps"] and res["gaps"][0]["date"] == "2026-06-24"

    def test_cascade_stale_return_triggers_reconcile(self, tmp_path):
        # 06-26's OWN close is correct, but its stored spy_return was computed
        # against the OLD 06-25 prior before 06-25's close was corrected. The
        # own-close check alone would miss it; the recomputed-return check catches
        # the cascade and re-reconciles.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 734.30, 0.1446, 1.41),
                       ("2026-06-26", 728.99, -0.6149, 0.22)])  # -0.6149 = STALE return
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}
        calls = []

        def fake_run(d, *, send_email, run_audit):
            assert send_email is False and run_audit is False
            calls.append(d)
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_return_pct=? WHERE date=?", (-0.7231, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-26", end="2026-06-26", config=_cfg(db))
        assert calls == ["2026-06-26"]  # only the cascaded day re-reconciled
        assert len(res["corrected"]) == 1
        assert res["corrected"][0]["reason"] == "stale_return"

    def test_dry_run_changes_nothing(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch("executor.backfill_eod_pnl.backfill") as bf_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", dry_run=True, config=_cfg(db))
        run_mock.assert_not_called()
        bf_mock.assert_not_called()
        assert len(res["corrected"]) == 1
        assert res["corrected"][0]["applied"] is False

    def test_exclude_dates_skips_today(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52), ("2026-06-26", 728.99, -0.61, 0.22)])
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}
        seen = []
        with patch.object(reconcile_audit, "_spy_close",
                          side_effect=lambda d, c: seen.append(d) or settled[d]), \
             patch.object(reconcile_audit, "eod_run"), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            audit_window(start="2026-06-25", end="2026-06-26",
                         exclude_dates={"2026-06-26"}, config=_cfg(db))
        assert "2026-06-26" not in seen and "2026-06-25" in seen

    # ── severity banding (config#2145) ──────────────────────────────────────

    def test_in_band_correction_not_paged(self, tmp_path):
        # 1.46bp divergence (the 2026-07-09 case: 751.60 -> 751.71) is below
        # PAGE_THRESHOLD_BPS (5.0) — audit trail is still written, but
        # flow-doctor is never invoked. Every flow-doctor severity maps to
        # SOME Telegram notifier in flow-doctor.yaml, so "not paged" means
        # skipping the fd.report() call entirely, not picking a lower severity.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-07-09", 751.60, -0.01, 1.52)])
        settled = {"2026-07-09": 751.71}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (751.71, d))
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-07-09", end="2026-07-09", config=_cfg(db))
        assert len(res["corrected"]) == 1
        c = res["corrected"][0]
        assert c["paged"] is False
        assert 1.0 < c["divergence_bps"] < 5.0
        fd_mock.report.assert_not_called()
        write_mock.assert_called_once()  # audit trail unaffected by paging decision
        assert write_mock.call_args.kwargs["record"]["paged"] is False

    def test_outlier_correction_pages(self, tmp_path):
        # 10.9bp divergence (>= PAGE_THRESHOLD_BPS) still pages flow-doctor
        # at severity=warning — the config#1276 incident class this guard
        # exists to catch must keep paging.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (734.30, d))
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["corrected"][0]["paged"] is True
        fd_mock.report.assert_called_once()
        assert fd_mock.report.call_args.kwargs["severity"] == "warning"

    def test_second_in_band_correction_in_same_window_pages_as_recurrence(self, tmp_path):
        # Two dates each individually in-band (1-2bp) but BOTH drifting in the
        # same audit pass is systemic, not routine settlement lag — the 2nd+
        # correction pages regardless of its own magnitude.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 734.20, 0.10, 1.40), ("2026-06-26", 728.90, -0.72, 0.33)])
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (settled[d], d))
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-06-25", end="2026-06-26", config=_cfg(db))
        assert len(res["corrected"]) == 2
        assert res["corrected"][0]["paged"] is False  # first: in-band, routine
        assert res["corrected"][1]["paged"] is True   # second: recurrence in same pass
        assert fd_mock.report.call_count == 1

    def test_no_settled_close_skips_gracefully(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])

        def raise_missing(d, c):
            raise RuntimeError(f"ArcticDB has no SPY close for {d}")

        with patch.object(reconcile_audit, "_spy_close", side_effect=raise_missing), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        run_mock.assert_not_called()
        assert res["checked"] == 0
        assert res["skipped"] and res["skipped"][0]["reason"] == "no_settled_close"
