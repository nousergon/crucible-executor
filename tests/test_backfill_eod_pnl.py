"""Tests for executor/backfill_eod_pnl.py — ledger-synthesis recovery of a
missing eod_pnl row (config#1229)."""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from executor.backfill_eod_pnl import (
    _prior_eod_row,
    backfill,
    check_position_divergence,
    day_cash_flow,
    day_share_deltas,
    replay_positions,
    synthesize_positions,
    synthesize_snapshot,
)
from executor.trade_logger import init_db


def _seed_conn():
    conn = init_db(":memory:")
    return conn


def _trade(conn, *, tid, date, ticker, action, shares, fill_price=None, filled=None):
    conn.execute(
        "INSERT INTO trades (trade_id, date, ticker, action, shares, fill_price, "
        "filled_shares, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (tid, date, ticker, action, shares, fill_price, filled, f"{date}T12:00:00"),
    )
    conn.commit()


def _eod_row(conn, *, date, cash, accrued=0.0, positions=None):
    conn.execute(
        "INSERT OR REPLACE INTO eod_pnl (date, portfolio_nav, total_cash, "
        "accrued_interest, positions_snapshot, created_at) VALUES (?,?,?,?,?,?)",
        (date, None, cash, accrued, json.dumps(positions or {}), f"{date}T20:00:00"),
    )
    conn.commit()


# ── replay_positions ──────────────────────────────────────────────────────────


class TestReplayPositions:
    def test_nets_enter_exit_reduce(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        _trade(conn, tid="2", date="2026-06-22", ticker="AAA", action="REDUCE", shares=30)
        _trade(conn, tid="3", date="2026-06-20", ticker="BBB", action="ENTER", shares=50)
        _trade(conn, tid="4", date="2026-06-23", ticker="BBB", action="EXIT", shares=50)
        held = replay_positions(conn, "2026-06-24")
        assert held == {"AAA": 70}  # BBB fully exited → dropped

    def test_respects_as_of_date_cutoff(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        # A reduce AFTER the as-of date must not count.
        _trade(conn, tid="2", date="2026-06-25", ticker="AAA", action="REDUCE", shares=40)
        held = replay_positions(conn, "2026-06-24")
        assert held == {"AAA": 100}

    def test_uses_filled_shares_when_present(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100, filled=80)
        assert replay_positions(conn, "2026-06-24") == {"AAA": 80}


# ── day_cash_flow ─────────────────────────────────────────────────────────────


class TestDayCashFlow:
    def test_no_trades_is_zero(self):
        conn = _seed_conn()
        assert day_cash_flow(conn, "2026-06-24") == 0.0

    def test_buy_out_sell_in(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-24", ticker="AAA", action="ENTER", shares=10, fill_price=100.0)
        _trade(conn, tid="2", date="2026-06-24", ticker="BBB", action="EXIT", shares=5, fill_price=200.0)
        # -10*100 (buy) + 5*200 (sell) = -1000 + 1000 = 0
        assert day_cash_flow(conn, "2026-06-24") == pytest.approx(0.0)

    def test_only_counts_the_target_date(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-23", ticker="AAA", action="ENTER", shares=10, fill_price=100.0)
        assert day_cash_flow(conn, "2026-06-24") == 0.0


# ── synthesize_snapshot ───────────────────────────────────────────────────────


class TestSynthesizeSnapshot:
    def test_nav_is_cash_plus_marked_positions(self):
        snap = synthesize_snapshot(
            run_date="2026-06-24",
            shares_by_ticker={"AAA": 100, "BBB": 50},
            closes_by_ticker={"AAA": 10.0, "BBB": 20.0},
            cash=5000.0,
            accrued_interest=12.0,
            prior_positions={"AAA": {"avg_cost": 9.0, "sector": "Tech"}},
            schema_version=1,
        )
        # NAV = 5000 + 100*10 + 50*20 = 7000
        assert snap["account"]["net_liquidation"] == pytest.approx(7000.0)
        assert snap["account"]["total_cash"] == 5000.0
        assert snap["account"]["accrued_interest"] == 12.0
        assert snap["synthesized"] is True
        assert snap["positions"]["AAA"]["avg_cost"] == 9.0       # carried from prior
        assert snap["positions"]["BBB"]["avg_cost"] == 20.0      # seeded to close (new)
        assert snap["positions"]["AAA"]["shares"] == 100

    def test_no_trade_day_is_exact_reprice_of_prior_book(self):
        # The common halt case: no trades, cash unchanged, prior book re-marked.
        snap = synthesize_snapshot(
            run_date="2026-06-24",
            shares_by_ticker={"AAA": 100},
            closes_by_ticker={"AAA": 11.0},
            cash=5000.0,
            accrued_interest=0.0,
            prior_positions={"AAA": {"avg_cost": 9.0}},
            schema_version=1,
        )
        assert snap["account"]["net_liquidation"] == pytest.approx(5000.0 + 100 * 11.0)


# ── _prior_eod_row ────────────────────────────────────────────────────────────


class TestPriorEodRow:
    def test_picks_latest_before_date_and_parses_snapshot(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-22", cash=100.0, positions={"AAA": {"avg_cost": 9.0}})
        _eod_row(conn, date="2026-06-23", cash=200.0, positions={"AAA": {"avg_cost": 9.5}})
        prior = _prior_eod_row(conn, "2026-06-24")
        assert prior["date"] == "2026-06-23" and prior["total_cash"] == 200.0
        assert prior["positions_snapshot"]["AAA"]["avg_cost"] == 9.5

    def test_none_when_no_prior(self):
        conn = _seed_conn()
        assert _prior_eod_row(conn, "2026-06-24") is None


# ── backfill orchestration (guards + dry-run) ─────────────────────────────────


class TestBackfillOrchestration:
    def _patch(self, stack, conn, closes):
        stack.enter_context(patch(
            "executor.backfill_eod_pnl.load_config",
            return_value={"db_path": ":memory:", "trades_bucket": "b", "aws_region": "us-east-1"},
        ))
        stack.enter_context(patch("executor.backfill_eod_pnl.init_db", return_value=conn))
        stack.enter_context(patch("executor.backfill_eod_pnl._read_closes_for_date", return_value=closes))
        stack.enter_context(patch("executor.snapshot_capturer.load_snapshot", return_value=None))

    def test_dry_run_no_trade_day_rolls_forward_exactly(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-23", cash=5000.0, positions={"AAA": {"avg_cost": 9.0}})
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        with ExitStack() as stack:
            self._patch(stack, conn, {"AAA": 11.0})
            result = backfill("2026-06-24", dry_run=True)
        assert result["dry_run"] is True
        assert result["cash_today"] == 5000.0                    # no trades on 06-24
        assert result["synthesized_nav"] == pytest.approx(5000.0 + 100 * 11.0)
        assert result["n_positions"] == 1

    def test_raises_when_no_prior_cash_baseline(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        with ExitStack() as stack:
            self._patch(stack, conn, {"AAA": 11.0})
            with pytest.raises(RuntimeError, match="No prior eod_pnl row"):
                backfill("2026-06-24", dry_run=True)

    def test_raises_when_row_exists_without_force(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-23", cash=5000.0)
        _eod_row(conn, date="2026-06-24", cash=5100.0)  # the row already exists
        with ExitStack() as stack:
            self._patch(stack, conn, {})
            with pytest.raises(RuntimeError, match="already exists"):
                backfill("2026-06-24", dry_run=True)


# ── day_share_deltas (config#1281) ────────────────────────────────────────────


class TestDayShareDeltas:
    def test_only_the_target_day_and_signed(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-23", ticker="AAA", action="ENTER", shares=100)  # prior day
        _trade(conn, tid="2", date="2026-06-24", ticker="AAA", action="REDUCE", shares=30)
        _trade(conn, tid="3", date="2026-06-24", ticker="BBB", action="ENTER", shares=50)
        assert day_share_deltas(conn, "2026-06-24") == {"AAA": -30, "BBB": 50}

    def test_no_trades_is_empty(self):
        conn = _seed_conn()
        assert day_share_deltas(conn, "2026-06-24") == {}

    def test_uses_filled_shares(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-24", ticker="AAA", action="ENTER", shares=100, filled=80)
        assert day_share_deltas(conn, "2026-06-24") == {"AAA": 80}


# ── synthesize_positions: ANCHOR on prior snapshot + day's fills ──────────────


class TestSynthesizePositionsAnchoring:
    def test_no_trade_day_reproduces_prior_position_set_exactly(self):
        # The Closes-when assertion: a no-trade day must reproduce the prior
        # reconciled position set verbatim (not a free-running ledger net).
        prior = {"AAA": {"shares": 100}, "BBB": {"shares": 50}, "CCC": {"shares": 7}}
        held, used_fallback = synthesize_positions(prior, day_deltas={})
        assert held == {"AAA": 100, "BBB": 50, "CCC": 7}
        assert used_fallback is False

    def test_applies_only_days_fills_to_the_anchor(self):
        prior = {"AAA": {"shares": 100}, "BBB": {"shares": 50}}
        # Day's fills: trim AAA by 30, fully exit BBB, open a new DDD.
        deltas = {"AAA": -30, "BBB": -50, "DDD": 25}
        held, used_fallback = synthesize_positions(prior, deltas)
        assert held == {"AAA": 70, "DDD": 25}  # BBB exited → dropped
        assert used_fallback is False

    def test_does_not_drift_across_days_vs_free_running_replay(self):
        # Anchoring is immune to the cumulative-net drift the issue describes.
        # Build a ledger where a long history of opens+closes does NOT fully net
        # (the broker book closed a name the ledger still shows partially open).
        conn = _seed_conn()
        # Ledger: AAA opened 100 long ago, partially reduced 40, then a stray
        # mismatched REDUCE 30 — free-running replay would show AAA=30 (wrong);
        # the broker-reconciled prior snapshot is authoritative: AAA fully out.
        _trade(conn, tid="1", date="2026-06-01", ticker="AAA", action="ENTER", shares=100)
        _trade(conn, tid="2", date="2026-06-10", ticker="AAA", action="REDUCE", shares=40)
        _trade(conn, tid="3", date="2026-06-20", ticker="AAA", action="REDUCE", shares=30)
        free_running = replay_positions(conn, "2026-06-24")
        assert free_running == {"AAA": 30}  # drifted: shows a phantom position
        # The reconciled prior snapshot (broker book) says AAA is fully closed.
        prior = {"BBB": {"shares": 50}}  # only BBB actually held per broker
        held, used_fallback = synthesize_positions(
            prior, day_deltas={}, conn=conn, as_of_date="2026-06-24"
        )
        assert held == {"BBB": 50}  # anchored: no phantom AAA, no drift
        assert used_fallback is False

    def test_missing_anchor_falls_back_to_replay_and_flags(self):
        conn = _seed_conn()
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        held, used_fallback = synthesize_positions(
            {}, day_deltas={}, conn=conn, as_of_date="2026-06-24"
        )
        assert held == {"AAA": 100}      # the legacy replay result
        assert used_fallback is True     # flagged un-anchored

    def test_no_anchor_no_conn_is_empty_not_fallback(self):
        held, used_fallback = synthesize_positions({}, day_deltas={})
        assert held == {} and used_fallback is False

    def test_anchor_ignores_non_positive_or_bad_prior_shares(self):
        prior = {"AAA": {"shares": 100}, "BBB": {"shares": 0}, "CCC": {"shares": None}}
        held, _ = synthesize_positions(prior, day_deltas={})
        assert held == {"AAA": 100}


# ── check_position_divergence guard (config#1281/#1276) ───────────────────────


class TestDivergenceGuard:
    def test_no_trade_day_does_not_diverge(self):
        prior = {f"T{i}": {"shares": 10} for i in range(7)}
        diverged, n_s, n_p = check_position_divergence({f"T{i}": 10 for i in range(7)}, prior)
        assert diverged is False and n_s == 7 and n_p == 7

    def test_small_change_does_not_diverge(self):
        prior = {f"T{i}": {"shares": 10} for i in range(7)}
        synth = {f"T{i}": 10 for i in range(8)}  # one new name
        diverged, _, _ = check_position_divergence(synth, prior)
        assert diverged is False

    def test_blowup_diverges(self):
        # The exact 7→20 failure mode from config#1276.
        prior = {f"T{i}": {"shares": 10} for i in range(7)}
        synth = {f"T{i}": 10 for i in range(20)}
        diverged, n_s, n_p = check_position_divergence(synth, prior)
        assert diverged is True and n_s == 20 and n_p == 7

    def test_cold_start_no_prior_does_not_diverge(self):
        diverged, _, n_p = check_position_divergence({"AAA": 1, "BBB": 2}, {})
        assert diverged is False and n_p == 0


# ── backfill orchestration: anchored synthesis + guard (config#1281) ──────────


class TestBackfillAnchoredOrchestration:
    def _patch(self, stack, conn, closes):
        stack.enter_context(patch(
            "executor.backfill_eod_pnl.load_config",
            return_value={"db_path": ":memory:", "trades_bucket": "b", "aws_region": "us-east-1"},
        ))
        stack.enter_context(patch("executor.backfill_eod_pnl.init_db", return_value=conn))
        stack.enter_context(patch("executor.backfill_eod_pnl._read_closes_for_date", return_value=closes))
        stack.enter_context(patch("executor.snapshot_capturer.load_snapshot", return_value=None))

    def test_anchored_no_trade_day_reproduces_prior_positions(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-23", cash=5000.0,
                 positions={"AAA": {"shares": 100, "avg_cost": 9.0},
                            "BBB": {"shares": 50, "avg_cost": 19.0}})
        # A noisy ledger that a free-running replay would mis-net is irrelevant now.
        _trade(conn, tid="1", date="2026-06-01", ticker="ZZZ", action="ENTER", shares=999)
        with ExitStack() as stack:
            self._patch(stack, conn, {"AAA": 11.0, "BBB": 20.0})
            result = backfill("2026-06-24", dry_run=True)
        assert result["anchored_on_prior_snapshot"] is True
        assert result["n_positions"] == 2                       # AAA, BBB — not ZZZ
        assert set(result["snapshot_preview"]["positions"]) == {"AAA", "BBB"}
        # NAV = cash + 100*11 + 50*20 = 5000 + 1100 + 1000 = 7100
        assert result["synthesized_nav"] == pytest.approx(7100.0)

    def test_anchored_applies_days_fills(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-23", cash=5000.0,
                 positions={"AAA": {"shares": 100, "avg_cost": 9.0}})
        # Day's trade: buy 50 more AAA @ 10  → cash -500, shares 150.
        _trade(conn, tid="1", date="2026-06-24", ticker="AAA", action="ENTER", shares=50, fill_price=10.0)
        with ExitStack() as stack:
            self._patch(stack, conn, {"AAA": 11.0})
            result = backfill("2026-06-24", dry_run=True)
        assert result["cash_today"] == pytest.approx(4500.0)
        assert result["snapshot_preview"]["positions"]["AAA"]["shares"] == 150
        assert result["synthesized_nav"] == pytest.approx(4500.0 + 150 * 11.0)

    def test_divergence_guard_refuses_to_write(self):
        conn = _seed_conn()
        _eod_row(conn, date="2026-06-23", cash=5000.0,
                 positions={"AAA": {"shares": 100}})  # prior = 1 position
        # The day's fills open 10 brand-new names → 11 positions vs prior 1.
        for i in range(10):
            _trade(conn, tid=f"n{i}", date="2026-06-24", ticker=f"NEW{i}",
                   action="ENTER", shares=10, fill_price=5.0)
        closes = {"AAA": 11.0, **{f"NEW{i}": 5.0 for i in range(10)}}
        with ExitStack() as stack:
            self._patch(stack, conn, closes)
            with pytest.raises(RuntimeError, match="diverges materially"):
                backfill("2026-06-24", dry_run=True)

    def test_missing_anchor_fallback_is_flagged_in_summary(self):
        conn = _seed_conn()
        # Prior eod row has cash but an EMPTY positions_snapshot → no anchor.
        _eod_row(conn, date="2026-06-23", cash=5000.0, positions={})
        _trade(conn, tid="1", date="2026-06-20", ticker="AAA", action="ENTER", shares=100)
        with ExitStack() as stack:
            self._patch(stack, conn, {"AAA": 11.0})
            result = backfill("2026-06-24", dry_run=True)
        assert result["anchored_on_prior_snapshot"] is False
        assert "warning" in result and "fallback" in result["warning"]
        assert result["n_positions"] == 1                       # the replay result
