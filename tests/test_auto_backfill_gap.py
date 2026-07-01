"""Tests for executor/auto_backfill_gap.py — verified-zero-fill carry-forward
+ reprice auto-backfill of a skipped-session eod_pnl gap (config#1454).

Locks the strict 3-part gate (zero fills / prior snapshot exists /
authoritative closes complete) — every gate failure must still flag the gap
for manual review, and the happy path must produce a schema-identical
eod_pnl row stamped with ``_reconstructed`` provenance.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from executor.auto_backfill_gap import (
    _has_zero_fills,
    attempt_auto_backfill,
    build_reconstructed_snapshot,
    check_gate,
)
from executor.trade_logger import init_db


def _trade(conn, *, tid, date, ticker, action, shares, status="Filled",
           filled=None, fill_time=None):
    conn.execute(
        "INSERT INTO trades (trade_id, date, ticker, action, shares, "
        "filled_shares, status, fill_time, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (tid, date, ticker, action, shares, filled, status, fill_time,
         f"{date}T12:00:00"),
    )
    conn.commit()


def _eod_row(conn, *, date, cash=100_000.0, accrued=0.0, positions=None):
    conn.execute(
        "INSERT OR REPLACE INTO eod_pnl (date, portfolio_nav, total_cash, "
        "accrued_interest, positions_snapshot, created_at) VALUES (?,?,?,?,?,?)",
        (date, None, cash, accrued, json.dumps(positions or {}), f"{date}T20:00:00"),
    )
    conn.commit()


class _FakeLib:
    """Stub ArcticDB library — reads return a canned Close-frame per ticker."""

    def __init__(self, closes_by_ticker: dict[str, dict[str, float]]):
        self._closes = closes_by_ticker

    def read(self, ticker):
        import pandas as pd
        rows = self._closes.get(ticker)
        if rows is None:
            raise KeyError(ticker)
        idx = pd.to_datetime(list(rows.keys()))
        df = pd.DataFrame({"Close": list(rows.values())}, index=idx)
        return MagicMock(data=df)


# ── gate (a): zero fills ──────────────────────────────────────────────────


class TestHasZeroFills:
    def test_true_when_no_trades_at_all(self):
        conn = init_db(":memory:")
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is True and offending == []

    def test_false_when_a_trade_dated_on_gap_day_filled(self):
        conn = init_db(":memory:")
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=10, filled=10, fill_time="2026-06-24T14:00:00")
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is False
        assert offending[0]["ticker"] == "AAPL"

    def test_uses_fill_time_not_date_tag_for_the_gap_day(self):
        # config#1454 axis 2: a trade tagged trading_day/date = 2026-06-23 but
        # whose ACTUAL fill lands on the gap date (2026-06-24, e.g. an
        # overnight/next-morning execution) must still count as a fill ON
        # the gap date — using date alone would wrongly call this a
        # zero-fill day.
        conn = init_db(":memory:")
        _trade(conn, tid="1", date="2026-06-23", ticker="AAPL", action="ENTER",
               shares=10, filled=10, fill_time="2026-06-24T09:31:00")
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is False
        assert offending[0]["fill_time"] == "2026-06-24T09:31:00"

    def test_fill_time_can_also_clear_a_gap_day(self):
        # A trade dated/tagged on the gap day but whose real fill lands the
        # NEXT day (fill_time D+1) must NOT count as a fill on the gap day.
        conn = init_db(":memory:")
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=10, filled=10, fill_time="2026-06-25T09:31:00")
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is True and offending == []

    def test_rejected_and_cancelled_do_not_count_as_fills(self):
        conn = init_db(":memory:")
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=10, filled=10, status="Rejected", fill_time="2026-06-24T14:00:00")
        _trade(conn, tid="2", date="2026-06-24", ticker="MSFT", action="ENTER",
               shares=5, filled=5, status="cancelled", fill_time="2026-06-24T14:00:00")
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is True and offending == []

    def test_legacy_row_with_no_fill_time_falls_back_to_date(self):
        conn = init_db(":memory:")
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=10, filled=10, fill_time=None)
        zero, offending = _has_zero_fills(conn, "2026-06-24")
        assert zero is False


# ── full gate: check_gate ──────────────────────────────────────────────────


class TestCheckGate:
    def test_eligible_when_all_three_conditions_pass(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", cash=50_000.0,
                 positions={"AAPL": {"shares": 100, "avg_cost": 150.0}})
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-24": 152.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})):
            gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is True
        assert gate["reason"] is None
        assert gate["prior_date"] == "2026-06-23"
        assert gate["closes"] == {"AAPL": 152.0}

    def test_gate_a_fails_when_a_fill_executed_on_gap_date(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", positions={"AAPL": {"shares": 100}})
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=5, filled=5, fill_time="2026-06-24T10:00:00")
        gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is False
        assert "not a zero-fill day" in gate["reason"]
        assert gate["offending_fills"]

    def test_gate_b_fails_when_no_prior_snapshot(self):
        conn = init_db(":memory:")  # no eod_pnl rows at all
        gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is False
        assert "cold start" in gate["reason"]

    def test_gate_b_fails_when_prior_row_has_null_positions_snapshot(self):
        conn = init_db(":memory:")
        conn.execute(
            "INSERT INTO eod_pnl (date, portfolio_nav, created_at) VALUES (?,?,?)",
            ("2026-06-23", 900_000.0, "2026-06-23T20:00:00"),
        )
        conn.commit()
        gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is False
        assert "cold start" in gate["reason"]

    def test_gate_c_fails_when_a_held_ticker_close_is_missing(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", positions={
            "AAPL": {"shares": 100}, "MSFT": {"shares": 50},
        })
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-24": 152.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})):
            gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is False
        assert "MSFT" in gate["reason"]
        assert "Missing authoritative ArcticDB close" in gate["reason"]

    def test_gate_c_fails_when_close_exists_but_not_for_gap_date(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", positions={"AAPL": {"shares": 100}})
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-20": 149.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})):
            gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is False
        assert "AAPL" in gate["reason"]

    def test_empty_prior_book_is_trivially_eligible(self):
        # Prior day held nothing (fully in cash) — no tickers to price, gate
        # (c) is vacuously satisfied.
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", cash=1_000_000.0, positions={})
        gate = check_gate(conn, "some-bucket", "2026-06-24")
        assert gate["eligible"] is True
        assert gate["closes"] == {}


# ── build_reconstructed_snapshot ────────────────────────────────────────────


class TestBuildReconstructedSnapshot:
    def test_carries_positions_forward_unchanged_and_reprices(self):
        prior = {
            "date": "2026-06-23",
            "positions": {"AAPL": {"shares": 100, "avg_cost": 150.0, "sector": "Tech"}},
            "total_cash": 50_000.0,
            "accrued_interest": 12.5,
        }
        snap = build_reconstructed_snapshot("2026-06-24", prior, {"AAPL": 152.0})
        assert snap["positions"]["AAPL"]["shares"] == 100
        assert snap["positions"]["AAPL"]["market_value"] == 15_200.0
        assert snap["positions"]["AAPL"]["closing_price"] == 152.0
        assert snap["positions"]["AAPL"]["avg_cost"] == 150.0
        assert snap["positions"]["AAPL"]["sector"] == "Tech"  # carried, not dropped
        assert snap["account"]["total_cash"] == 50_000.0
        assert snap["account"]["net_liquidation"] == 50_000.0 + 15_200.0
        assert snap["account"]["accrued_interest"] == 12.5

    def test_stamps_reconstructed_provenance(self):
        prior = {"date": "2026-06-23", "positions": {}, "total_cash": 1_000.0,
                 "accrued_interest": None}
        snap = build_reconstructed_snapshot("2026-06-24", prior, {})
        prov = snap["_reconstructed"]
        assert prov["method"] == "verified_zero_fill_carry_forward_reprice"
        assert prov["config_ref"] == "config#1454"
        assert prov["prior_trading_day"] == "2026-06-23"
        assert prov["gate"] == {
            "zero_fills_confirmed": True,
            "prior_snapshot_exists": True,
            "authoritative_closes_complete": True,
        }

    def test_empty_book_reconstructs_to_pure_cash_nav(self):
        prior = {"date": "2026-06-23", "positions": {}, "total_cash": 984_303.41,
                 "accrued_interest": 0.0}
        snap = build_reconstructed_snapshot("2026-06-24", prior, {})
        assert snap["account"]["net_liquidation"] == 984_303.41
        assert snap["positions"] == {}

    def test_does_not_carry_forward_stale_derived_fields(self):
        # The prior day's snapshot position dict also carries fields DERIVED
        # against THAT day's close (market_value, unrealized_pnl,
        # daily_return_pct/usd, alpha_contribution_*) — these must NOT bleed
        # into the reconstructed gap-day snapshot verbatim; only the
        # IDENTITY fields (shares/avg_cost/sector) are carried, and
        # market_value/closing_price are freshly recomputed against the
        # gap-date close.
        prior = {
            "date": "2026-06-23",
            "positions": {"AAPL": {
                "shares": 100, "avg_cost": 150.0, "sector": "Tech",
                "market_value": 15_000.0, "unrealized_pnl": 500.0,
                "daily_return_pct": 1.23, "daily_return_usd": 45.0,
                "alpha_contribution_pct": 0.5, "closing_price": 150.0,
            }},
            "total_cash": 50_000.0, "accrued_interest": 0.0,
        }
        snap = build_reconstructed_snapshot("2026-06-24", prior, {"AAPL": 200.0})
        pos = snap["positions"]["AAPL"]
        assert pos["market_value"] == 100 * 200.0  # freshly recomputed
        assert pos["closing_price"] == 200.0
        assert "unrealized_pnl" not in pos
        assert "daily_return_pct" not in pos
        assert "daily_return_usd" not in pos
        assert "alpha_contribution_pct" not in pos

    def test_none_shares_and_avg_cost_do_not_crash(self):
        # dict.get(key, default) only applies the default when the key is
        # ABSENT, not when present-but-None — a malformed/legacy prior
        # snapshot with explicit nulls must not raise.
        prior = {
            "date": "2026-06-23",
            "positions": {"AAPL": {"shares": None, "avg_cost": None}},
            "total_cash": 10_000.0, "accrued_interest": None,
        }
        snap = build_reconstructed_snapshot("2026-06-24", prior, {"AAPL": 100.0})
        pos = snap["positions"]["AAPL"]
        assert pos["shares"] == 0
        assert pos["market_value"] == 0.0
        assert pos["avg_cost"] == 100.0  # falls back to the gap-date close


# ── attempt_auto_backfill (end-to-end, mocked I/O) ─────────────────────────


def _s3_no_existing_snapshot() -> MagicMock:
    """A MagicMock s3 client whose get_object raises a NoSuchKey-style error
    (the common "no snapshot written yet" case the pre-write guard treats as
    safe to proceed past)."""
    s3 = MagicMock()
    s3.get_object.side_effect = Exception("NoSuchKey: not found")
    return s3


class TestAttemptAutoBackfill:
    def test_backfills_and_runs_canonical_reconcile_when_gate_passes(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", cash=884_303.41,
                 positions={"AAPL": {"shares": 100, "avg_cost": 150.0}})
        s3 = _s3_no_existing_snapshot()
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-24": 1000.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})), \
             patch("executor.eod_reconcile.run") as eod_run_mock:
            result = attempt_auto_backfill(
                conn, gap_date="2026-06-24", trades_bucket="some-bucket",
                s3_client=s3,
            )
        assert result["backfilled"] is True
        assert result["reason"] is None
        s3.put_object.assert_called_once()
        _, kwargs = s3.put_object.call_args
        assert kwargs["Key"] == "trades/snapshots/2026-06-24.json"
        written = json.loads(kwargs["Body"].decode())
        assert written["_reconstructed"]["config_ref"] == "config#1454"
        assert written["account"]["net_liquidation"] == 884_303.41 + 100 * 1000.0
        eod_run_mock.assert_called_once_with(
            "2026-06-24", send_email=False, run_audit=False,
        )

    def test_does_not_write_or_reconcile_when_gate_fails(self):
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23", positions={"AAPL": {"shares": 100}})
        _trade(conn, tid="1", date="2026-06-24", ticker="AAPL", action="ENTER",
               shares=5, filled=5, fill_time="2026-06-24T10:00:00")
        s3 = MagicMock()
        with patch("executor.eod_reconcile.run") as eod_run_mock:
            result = attempt_auto_backfill(
                conn, gap_date="2026-06-24", trades_bucket="some-bucket",
                s3_client=s3,
            )
        assert result["backfilled"] is False
        assert "not a zero-fill day" in result["reason"]
        s3.put_object.assert_not_called()
        eod_run_mock.assert_not_called()

    def test_does_not_write_or_reconcile_on_cold_start(self):
        conn = init_db(":memory:")  # no prior snapshot at all
        s3 = MagicMock()
        with patch("executor.eod_reconcile.run") as eod_run_mock:
            result = attempt_auto_backfill(
                conn, gap_date="2026-06-24", trades_bucket="some-bucket",
                s3_client=s3,
            )
        assert result["backfilled"] is False
        s3.put_object.assert_not_called()
        eod_run_mock.assert_not_called()

    def test_refuses_to_overwrite_a_real_pre_existing_snapshot(self):
        # A REAL (non-reconstructed) snapshot already at gap_date's key with
        # no eod_pnl row means eod_reconcile itself failed after
        # CaptureSnapshot — a different failure mode than a skipped session.
        # Must not silently clobber it with a carry-forward guess.
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23",
                 positions={"AAPL": {"shares": 100, "avg_cost": 150.0}})
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "run_date": "2026-06-24", "account": {"net_liquidation": 1.0},
                "positions": {},
            }).encode()),
        }
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-24": 1000.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})), \
             patch("executor.eod_reconcile.run") as eod_run_mock:
            result = attempt_auto_backfill(
                conn, gap_date="2026-06-24", trades_bucket="some-bucket",
                s3_client=s3,
            )
        assert result["backfilled"] is False
        assert "real (non-reconstructed) snapshot already exists" in result["reason"]
        s3.put_object.assert_not_called()
        eod_run_mock.assert_not_called()

    def test_overwrites_a_prior_reconstructed_snapshot(self):
        # A PRIOR auto-backfill attempt that wrote a _reconstructed snapshot
        # but then failed downstream (e.g. eod_run raised) is safe to retry.
        conn = init_db(":memory:")
        _eod_row(conn, date="2026-06-23",
                 positions={"AAPL": {"shares": 100, "avg_cost": 150.0}})
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "run_date": "2026-06-24", "_reconstructed": {"config_ref": "config#1454"},
                "account": {"net_liquidation": 1.0}, "positions": {},
            }).encode()),
        }
        with patch("executor.price_cache._open_universe_library",
                   return_value=_FakeLib({"AAPL": {"2026-06-24": 1000.0}})), \
             patch("executor.price_cache._open_macro_library", return_value=_FakeLib({})), \
             patch("executor.eod_reconcile.run") as eod_run_mock:
            result = attempt_auto_backfill(
                conn, gap_date="2026-06-24", trades_bucket="some-bucket",
                s3_client=s3,
            )
        assert result["backfilled"] is True
        s3.put_object.assert_called_once()
        eod_run_mock.assert_called_once()
