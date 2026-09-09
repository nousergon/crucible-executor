"""Tests for executor/reconcile_audit.py — T+1 self-heal of EOD market values
that were frozen pre-settlement (config#1276)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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


def _seed_eod_with_positions(db_path, date_, spy_close, spy_return_pct, daily_alpha_pct, positions):
    """Like ``_seed_eod`` for a single row, plus a JSON ``positions_snapshot``
    (``{ticker: {"closing_price": ..., "shares": ...}}``) for held-position
    staleness tests (config#6349)."""
    import json as _json

    conn = init_db(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO eod_pnl (date, portfolio_nav, spy_close, "
        "spy_return_pct, daily_return_pct, daily_alpha_pct, positions_snapshot, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (date_, 1_000_000.0, spy_close, spy_return_pct, 0.0, daily_alpha_pct,
         _json.dumps(positions), f"{date_}T20:00:00"),
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

        def fake_run(d, *, send_email, run_audit):
            # Must actually converge — an eod_run mock that changes nothing now
            # trips the post-correction verification (I10288).
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (settled[d], d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close",
                          side_effect=lambda d, c: seen.append(d) or settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
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

    # ── Accepted gaps (alpha-engine-config#5570) ───────────────────────────

    def test_accepted_gap_not_paged(self, tmp_path):
        # A gap date that is registered in the accepted-gaps registry must
        # still appear in the gaps list, but must NOT page flow-doctor or
        # emit a WARNING about manual backfill.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-23", 733.58, -1.44, 0.0)])  # 06-24 absent → gap
        settled = {"2026-06-24": 733.24}
        cfg = {**_cfg(db), "trades_bucket": "test-bucket"}
        fd_mock = MagicMock()
        accepted = {
            "2026-06-24": {
                "date": "2026-06-24",
                "reason": "Test ruling — snapshot never existed",
                "ruling": "test-org/test-repo#1",
            },
        }
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "load_accepted_gaps", return_value=accepted), \
             patch("executor.backfill_eod_pnl.backfill") as bf_mock, \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-06-24", end="2026-06-24", config=cfg)
        bf_mock.assert_not_called()
        run_mock.assert_not_called()
        fd_mock.report.assert_not_called()  # must not page
        assert len(res["gaps"]) == 1
        g = res["gaps"][0]
        assert g["date"] == "2026-06-24"
        assert g.get("accepted") is True
        assert g.get("ruling") == "test-org/test-repo#1"

    def test_accepted_gap_still_listed_in_gaps_without_page(self, tmp_path):
        # Multiple gaps where one is accepted and one is not: the accepted
        # gap must still appear in the gaps list (not suppressed from the
        # record), while the non-accepted gap pages normally.
        db = str(tmp_path / "t.db")
        # 06-24 absent → accepted gap; 06-25 present; 06-26 absent → real gap
        _seed_eod(db, [("2026-06-23", 733.58, -1.44, 0.0), ("2026-06-25", 734.30, 0.10, 1.40)])
        settled = {"2026-06-24": 733.24, "2026-06-26": 728.99}
        cfg = {**_cfg(db), "trades_bucket": "test-bucket"}
        fd_mock = MagicMock()
        accepted = {"2026-06-24": {"date": "2026-06-24", "reason": "test", "ruling": "test#1"}}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "load_accepted_gaps", return_value=accepted), \
             patch("executor.backfill_eod_pnl.backfill") as bf_mock, \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-06-24", end="2026-06-26", config=cfg)
        bf_mock.assert_not_called()
        run_mock.assert_not_called()
        # Two gaps expected: 06-24 (accepted) and 06-26 (real)
        assert len(res["gaps"]) == 2
        accepted_gap = next(g for g in res["gaps"] if g["date"] == "2026-06-24")
        real_gap = next(g for g in res["gaps"] if g["date"] == "2026-06-26")
        assert accepted_gap.get("accepted") is True
        assert "accepted" not in real_gap or real_gap.get("accepted") is not True
        # Only the non-accepted gap should have triggered a page
        assert fd_mock.report.call_count == 1

    def test_unaccepted_gap_still_pages(self, tmp_path):
        # A gap date NOT in the accepted-gaps registry must still be
        # handled exactly as before — WARNING + page for manual backfill.
        # This is a regression guard: the accepted-gap code path must not
        # accidentally absorb all gaps.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-23", 733.58, -1.44, 0.0)])  # 06-24 absent → gap
        settled = {"2026-06-24": 733.24}
        cfg = {**_cfg(db), "trades_bucket": "test-bucket"}
        fd_mock = MagicMock()
        # accepted gaps registry exists but does NOT include 06-24
        accepted = {"2026-06-22": {"date": "2026-06-22", "reason": "other", "ruling": "other#1"}}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "load_accepted_gaps", return_value=accepted), \
             patch("executor.backfill_eod_pnl.backfill") as bf_mock, \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-06-24", end="2026-06-24", config=cfg)
        bf_mock.assert_not_called()
        run_mock.assert_not_called()
        assert len(res["gaps"]) == 1
        assert res["gaps"][0]["date"] == "2026-06-24"
        assert "accepted" not in res["gaps"][0]
        # Must still page
        fd_mock.report.assert_called_once()
        # Reported as a plain string, not RuntimeError(...) — a flagged gap is
        # a finding, not a crashed run (alpha-engine-config-I10288).
        reported = fd_mock.report.call_args[0][0]
        assert isinstance(reported, str)
        assert "Manually run" in reported

    def test_empty_trades_bucket_skips_accepted_gaps_load(self, tmp_path):
        # When trades_bucket is empty (the _cfg default), accepted_gaps must
        # be empty — load_accepted_gaps is never called because the bucket
        # guard in audit_window skips it. This regression guard ensures the
        # existing test_missing_row_flagged_as_gap_not_backfilled path (which
        # uses trades_bucket="") still works identically.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-23", 733.58, -1.44, 0.0)])
        settled = {"2026-06-24": 733.24}
        # load_accepted_gaps MUST NOT be called when bucket is empty
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "load_accepted_gaps") as load_mock, \
             patch("executor.backfill_eod_pnl.backfill"), \
             patch.object(reconcile_audit, "eod_run"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-24", end="2026-06-24", config=_cfg(db))
        load_mock.assert_not_called()
        assert len(res["gaps"]) == 1


# ── Held-position staleness (config#6349) ───────────────────────────────────
#
# The SPY-only checks above catch nothing for a non-SPY held ticker whose
# ArcticDB price was provisional-then-corrected — its stale closing_price
# then silently feeds eod_reconcile's NAV three-way pricing&timing diff via
# prior_positions on every later day, indefinitely. These tests cover the
# generalized per-ticker check (_settled_close_for_ticker).


class TestHeldPositionStaleness:
    def test_stale_held_ticker_close_triggers_reconcile(self, tmp_path):
        # SPY itself is clean, but AMD's stored closing_price (100.00) has
        # since been corrected in ArcticDB to 101.50 (~148bp) — must still
        # trigger a re-reconcile even though the SPY-only checks see nothing.
        db = str(tmp_path / "t.db")
        _seed_eod_with_positions(
            db, "2026-06-25", 734.30, 0.10, 1.40,
            {"AMD": {"closing_price": 100.00, "shares": 10}},
        )
        settled_spy = {"2026-06-25": 734.30}
        settled_tickers = {("AMD", "2026-06-25"): 101.50}

        def fake_run(d, *, send_email, run_audit):
            assert send_email is False and run_audit is False
            conn = init_db(db)
            conn.execute(
                "UPDATE eod_pnl SET positions_snapshot=? WHERE date=?",
                ('{"AMD": {"closing_price": 101.50, "shares": 10}}', d),
            )
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker",
                           lambda t, d, c: settled_tickers.get((t, d))), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run) as run_mock, \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        run_mock.assert_called_once()
        assert len(res["corrected"]) == 1
        c = res["corrected"][0]
        assert c["date"] == "2026-06-25" and c["reason"] == "stale_position_close"
        assert c["before"]["stale_tickers"] == {"AMD": pytest.approx(150.0, abs=0.5)}
        assert c["divergence_bps"] > 100

    def test_held_ticker_within_tolerance_skipped(self, tmp_path):
        db = str(tmp_path / "t.db")
        # 0.5bp divergence on AMD < 1bp tolerance → no correction.
        _seed_eod_with_positions(
            db, "2026-06-25", 734.30, 0.10, 1.40,
            {"AMD": {"closing_price": 100.00, "shares": 10}},
        )
        settled_spy = {"2026-06-25": 734.30}
        settled_tickers = {("AMD", "2026-06-25"): 100.005}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker",
                           lambda t, d, c: settled_tickers.get((t, d))), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["corrected"] == []
        run_mock.assert_not_called()

    def test_ticker_settled_close_unavailable_does_not_trigger(self, tmp_path):
        # _settled_close_for_ticker returning None (no ArcticDB row yet) must
        # be treated as "can't check yet", not as a divergence.
        db = str(tmp_path / "t.db")
        _seed_eod_with_positions(
            db, "2026-06-25", 734.30, 0.10, 1.40,
            {"NEWCO": {"closing_price": 50.00, "shares": 5}},
        )
        settled_spy = {"2026-06-25": 734.30}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker", return_value=None), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["corrected"] == []
        assert res["gaps"] == []
        run_mock.assert_not_called()


# ── Revision vs corruption, honest alert text, convergence verification ──────
#
# alpha-engine-config-I10288. The 2026-09-04 page read
#   "RuntimeError: EOD value for 2026-09-04 corrected post-settlement
#    (stale_position_close): SPY close 770.24 → 770.19"
# and was taken for an aborted run. It was none of those things: the pass had
# already SUCCEEDED, nothing raised, and SPY was not the divergent value —
# 770.24 → 770.19 is 0.65bp, BELOW the pass's own 1bp tolerance, an incidental
# re-price. The held ticker that actually diverged appeared nowhere in the text.


class TestClassification:
    def test_in_band_correction_classified_revision(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-07-09", 751.60, -0.01, 1.52)])  # 1.46bp
        settled = {"2026-07-09": 751.71}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (751.71, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-07-09", end="2026-07-09", config=_cfg(db))
        assert res["corrected"][0]["classification"] == reconcile_audit.CLASSIFICATION_REVISION
        assert res["revisions"] == ["2026-07-09"] and res["corruptions"] == []
        assert write_mock.call_args.kwargs["record"]["classification"] == "revision"

    def test_outlier_correction_classified_corruption(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])  # 10.9bp
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (734.30, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["corrected"][0]["classification"] == reconcile_audit.CLASSIFICATION_CORRUPTION
        assert res["corruptions"] == ["2026-06-25"]

    def test_absent_stored_close_is_corruption_not_revision(self, tmp_path):
        # A None stored close is UNBOUNDED divergence — it must never be
        # ranked as the mildest case by falling through a None comparison.
        assert reconcile_audit._classify(None, page_threshold_bps=5.0) == "corruption"
        assert reconcile_audit._classify(float("inf"), page_threshold_bps=5.0) == "corruption"
        assert reconcile_audit._classify(4.9, page_threshold_bps=5.0) == "revision"
        assert reconcile_audit._classify(5.0, page_threshold_bps=5.0) == "corruption"


class TestAlertTextNamesTheDivergentLeg:
    def test_position_close_page_does_not_claim_spy(self, tmp_path):
        # The reported 2026-09-04 shape, reproduced: SPY within tolerance,
        # a held ticker 60bp stale. The page must name the TICKER, must carry
        # its bps, and must not present SPY as the finding.
        db = str(tmp_path / "t.db")
        _seed_eod_with_positions(
            db, "2026-09-04", 770.24, 0.10, 1.40,
            {"AMD": {"closing_price": 100.00, "shares": 10}},
        )
        settled_spy = {"2026-09-04": 770.19}          # 0.65bp — BELOW tolerance
        settled_tickers = {("AMD", "2026-09-04"): 100.60}  # 60bp — the real finding

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute(
                "UPDATE eod_pnl SET spy_close=?, positions_snapshot=? WHERE date=?",
                (770.19, '{"AMD": {"closing_price": 100.60, "shares": 10}}', d),
            )
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker",
                          lambda t, d, c: settled_tickers.get((t, d))), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-09-04", end="2026-09-04", config=_cfg(db))

        assert res["corrected"][0]["reason"] == "stale_position_close"
        fd_mock.report.assert_called_once()
        msg = fd_mock.report.call_args[0][0]
        # Reported as a completed correction, not an exception.
        assert isinstance(msg, str)
        assert "position_close:AMD" in msg and "100.0" in msg and "100.6" in msg
        assert "corruption" in msg
        # The sub-tolerance SPY re-price is NOT presented as the finding.
        assert "spy_close" not in msg
        assert "770.24" not in msg
        # And the page says WHY it paged.
        assert "PAGED" in msg

    def test_max_divergence_across_legs_drives_the_page(self, tmp_path):
        # SPY 1.2bp (in-band) AND a held ticker 60bp (corruption). The old
        # first-match reason/divergence picked SPY's 1.2bp, so the 60bp
        # corruption did NOT page. Both legs must be recorded and the page
        # decision must read the MAX.
        db = str(tmp_path / "t.db")
        _seed_eod_with_positions(
            db, "2026-09-04", 770.10, 0.10, 1.40,
            {"AMD": {"closing_price": 100.00, "shares": 10}},
        )
        settled_spy = {"2026-09-04": 770.19}               # ~1.17bp
        settled_tickers = {("AMD", "2026-09-04"): 100.60}  # ~60bp

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute(
                "UPDATE eod_pnl SET spy_close=?, positions_snapshot=? WHERE date=?",
                (770.19, '{"AMD": {"closing_price": 100.60, "shares": 10}}', d),
            )
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker",
                          lambda t, d, c: settled_tickers.get((t, d))), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock):
            res = audit_window(start="2026-09-04", end="2026-09-04", config=_cfg(db))

        c = res["corrected"][0]
        assert {leg["leg"] for leg in c["legs"]} == {"spy_close", "position_close:AMD"}
        assert c["divergence_bps"] == pytest.approx(60.0, abs=1.0)   # the MAX, not SPY's 1.2
        assert c["reason"] == "stale_position_close"                  # the worst leg names it
        assert set(c["reasons"]) == {"stale_close", "stale_position_close"}
        assert c["classification"] == "corruption"
        assert c["paged"] is True
        fd_mock.report.assert_called_once()


class TestConvergenceVerification:
    def test_unconverged_correction_raises_and_pages_critical(self, tmp_path):
        # eod_run "corrects" the row to a value that STILL diverges from
        # settled. Recorded, paged at critical, and the window RAISES — this
        # is the one outcome here that no later pass heals.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (733.90, d))  # still 5.4bp off
            conn.commit()
            conn.close()

        fd_mock = MagicMock()
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=fd_mock), \
             pytest.raises(reconcile_audit.ReconciliationUnconvergedError) as exc:
            audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))

        assert "2026-06-25" in str(exc.value)
        assert exc.value.summary["unconverged"][0]["date"] == "2026-06-25"
        # The audit record still landed, and states non-convergence.
        rec = write_mock.call_args.kwargs["record"]
        assert rec["converged"] is False and rec["residual_legs"]
        # Paged at critical, not swallowed.
        severities = [c.kwargs["severity"] for c in fd_mock.report.call_args_list]
        assert "critical" in severities

    def test_converged_correction_records_converged_true(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (734.30, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        assert res["unconverged"] == []
        assert res["corrected"][0]["converged"] is True
        assert write_mock.call_args.kwargs["record"]["converged"] is True

    def test_every_date_checked_before_the_raise(self, tmp_path):
        # Per-date isolation survives: an unconverged 06-25 must not stop
        # 06-26 from being checked and recorded.
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52), ("2026-06-26", 728.00, -0.72, 0.33)])
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            # 06-25 does not converge; 06-26 does.
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?",
                         (733.90 if d == "2026-06-25" else 728.99, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None), \
             pytest.raises(reconcile_audit.ReconciliationUnconvergedError) as exc:
            audit_window(start="2026-06-25", end="2026-06-26", config=_cfg(db))
        summary = exc.value.summary
        assert summary["checked"] == 2
        assert [c["date"] for c in summary["corrected"]] == ["2026-06-25", "2026-06-26"]
        assert [u["date"] for u in summary["unconverged"]] == ["2026-06-25"]
        assert write_mock.call_count == 2  # both audit records written before the raise


class TestAuditRecordStatesBlastRadius:
    def test_record_carries_downstream_map(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 733.50, -0.01, 1.52)])
        settled = {"2026-06-25": 734.30}

        def fake_run(d, *, send_email, run_audit):
            conn = init_db(db)
            conn.execute("UPDATE eod_pnl SET spy_close=? WHERE date=?", (734.30, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k") as write_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            audit_window(start="2026-06-25", end="2026-06-25", config=_cfg(db))
        rec = write_mock.call_args.kwargs["record"]
        assert rec["downstream"] == reconcile_audit.DOWNSTREAM_ON_CORRECTION
        assert any("input_closure" in k for k in rec["downstream"])


class TestPositionLevelCascade:
    def test_next_session_rereconciled_after_prior_correction(self, tmp_path):
        # 06-25's held AMD close was stale and gets corrected. 06-26's OWN
        # stored values are all clean — but its pricing&timing / mark_basis
        # were computed from 06-25's PRE-correction prior_positions, so its
        # frozen consolidated/2026-06-26/eod_report.json is derived from the
        # old basis. Before I10288 no leg fired for 06-26 and it never healed.
        db = str(tmp_path / "t.db")
        _seed_eod_with_positions(db, "2026-06-25", 734.30, 0.10, 1.40,
                                 {"AMD": {"closing_price": 100.00, "shares": 10}})
        _seed_eod_with_positions(db, "2026-06-26", 728.99, -0.7231, 0.22,
                                 {"AMD": {"closing_price": 102.00, "shares": 10}})
        settled_spy = {"2026-06-25": 734.30, "2026-06-26": 728.99}
        settled_tickers = {("AMD", "2026-06-25"): 101.50, ("AMD", "2026-06-26"): 102.00}
        calls = []

        def fake_run(d, *, send_email, run_audit):
            calls.append(d)
            conn = init_db(db)
            snap = f'{{"AMD": {{"closing_price": {settled_tickers[("AMD", d)]}, "shares": 10}}}}'
            conn.execute("UPDATE eod_pnl SET positions_snapshot=? WHERE date=?", (snap, d))
            conn.commit()
            conn.close()

        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled_spy[d]), \
             patch.object(reconcile_audit, "_settled_close_for_ticker",
                          lambda t, d, c: settled_tickers.get((t, d))), \
             patch.object(reconcile_audit, "eod_run", side_effect=fake_run), \
             patch.object(reconcile_audit, "_write_audit_record", return_value="k"), \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-26", config=_cfg(db))

        assert calls == ["2026-06-25", "2026-06-26"]  # the cascade re-ran 06-26
        cascade = res["corrected"][1]
        assert cascade["date"] == "2026-06-26"
        assert cascade["reason"] == "stale_prior_basis"
        assert cascade["legs"][-1]["leg"] == "prior_corrected:2026-06-25"
        # A derived-input change, not a divergence — must not read as corruption.
        assert cascade["legs"][-1]["divergence_bps"] == 0.0
        assert cascade["converged"] is True  # and must not trip the residual check

    def test_no_cascade_when_prior_was_clean(self, tmp_path):
        db = str(tmp_path / "t.db")
        _seed_eod(db, [("2026-06-25", 734.30, 0.10, 1.40), ("2026-06-26", 728.99, -0.7231, 0.22)])
        settled = {"2026-06-25": 734.30, "2026-06-26": 728.99}
        with patch.object(reconcile_audit, "_spy_close", lambda d, c: settled[d]), \
             patch.object(reconcile_audit, "eod_run") as run_mock, \
             patch.object(reconcile_audit, "get_flow_doctor", return_value=None):
            res = audit_window(start="2026-06-25", end="2026-06-26", config=_cfg(db))
        assert res["corrected"] == []
        run_mock.assert_not_called()
