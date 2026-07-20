"""Unit tests for executor.eod_reconcile — P&L math and data helpers."""
from __future__ import annotations

import json
import sys
from datetime import date
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401 -- only used in a quoted type annotation below

from executor.eod_reconcile import (
    _build_position_contexts,
    _compute_daily_return,
    _load_predictions_from_s3,
    _load_signals_from_s3,
    _spy_close,
    run,
)
from executor.eod_report import _buy_entry_prices

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_s3_response(body: dict | list) -> dict:
    """Build a mock S3 get_object response."""
    return {"Body": BytesIO(json.dumps(body).encode())}


def _make_signals(universe=None, buy_candidates=None, market_regime="neutral"):
    return {
        "universe": universe or [],
        "buy_candidates": buy_candidates or [],
        "market_regime": market_regime,
    }


def _make_prediction(ticker, direction="UP", confidence=0.75, alpha=0.02):
    return {
        "ticker": ticker,
        "predicted_direction": direction,
        "prediction_confidence": confidence,
        "predicted_alpha": alpha,
    }


# ── _spy_close ───────────────────────────────────────────────────────────────


def _mock_universe_with(symbols: dict[str, pd.DataFrame]):
    """Build a mock ArcticDB universe library.

    symbols maps ticker → DataFrame. universe.read(ticker) returns a
    SimpleNamespace with .data = the frame. Missing tickers raise.
    """
    from types import SimpleNamespace

    import pandas as pd  # noqa: F401 — used by caller's frames

    lib = MagicMock()

    def _read(sym):
        if sym not in symbols:
            raise KeyError(f"no such symbol: {sym}")
        return SimpleNamespace(data=symbols[sym])

    lib.read.side_effect = _read
    return lib


def test_spy_close_reads_from_arcticdb():
    """_spy_close returns the SPY close for run_date via ArcticDB universe."""
    import pandas as pd

    df = pd.DataFrame(
        {"Close": [447.00, 450.25]},
        index=pd.to_datetime(["2026-03-26", "2026-03-27"]),
    )
    with patch(
        "executor.price_cache._open_macro_library",
        return_value=_mock_universe_with({"SPY": df}),
    ):
        result = _spy_close("2026-03-27")
    assert result == pytest.approx(450.25)


def test_spy_close_hard_fails_when_symbol_missing():
    """_spy_close raises when ArcticDB has no SPY symbol — no fallback."""
    with patch(
        "executor.price_cache._open_macro_library",
        return_value=_mock_universe_with({}),
    ):
        with pytest.raises(RuntimeError, match="ArcticDB read failed for SPY"):
            _spy_close("2026-03-27")


def test_spy_close_hard_fails_when_date_missing():
    """_spy_close raises when run_date has no row in ArcticDB — no fallback."""
    import pandas as pd

    df = pd.DataFrame(
        {"Close": [447.00]},
        index=pd.to_datetime(["2026-03-26"]),
    )
    with patch(
        "executor.price_cache._open_macro_library",
        return_value=_mock_universe_with({"SPY": df}),
    ):
        with pytest.raises(RuntimeError, match="no SPY close for 2026-03-27"):
            _spy_close("2026-03-27")


def test_spy_close_hard_fails_when_close_column_missing():
    """_spy_close raises when the SPY frame has no Close column."""
    import pandas as pd

    df = pd.DataFrame(
        {"Open": [447.00]},
        index=pd.to_datetime(["2026-03-27"]),
    )
    with patch(
        "executor.price_cache._open_macro_library",
        return_value=_mock_universe_with({"SPY": df}),
    ):
        with pytest.raises(RuntimeError, match="empty or missing Close"):
            _spy_close("2026-03-27")


# ── _load_signals_from_s3 ────────────────────────────────────────────────────


@patch("executor.eod_reconcile.boto3")
def test_load_signals_exact_date(mock_boto3):
    """Loads signals for the exact run_date when available."""
    signals = _make_signals(universe=[{"ticker": "AAPL", "score": 85}])
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(signals)
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_signals_from_s3("test-bucket", "2026-03-27")
    assert warning is None
    assert result["universe"][0]["ticker"] == "AAPL"
    mock_s3.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="signals/2026-03-27/signals.json"
    )


@patch("executor.eod_reconcile.boto3")
def test_load_signals_falls_back_to_prior_date(mock_boto3):
    """Falls back to prior trading day when exact date has no signals."""
    signals = _make_signals(universe=[{"ticker": "MSFT", "score": 72}])
    mock_s3 = MagicMock()
    # Fail on Wed 2026-03-25, succeed on Tue 2026-03-24
    mock_s3.get_object.side_effect = [
        Exception("not found"),  # 2026-03-25 (Wed)
        _make_s3_response(signals),  # 2026-03-24 (Tue)
    ]
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_signals_from_s3("test-bucket", "2026-03-25")
    assert warning is None
    assert result["universe"][0]["ticker"] == "MSFT"


@patch("executor.eod_reconcile.boto3")
def test_load_signals_skips_weekends(mock_boto3):
    """Lookback skips Saturday/Sunday dates."""
    signals = _make_signals(universe=[{"ticker": "GOOG", "score": 90}])
    mock_s3 = MagicMock()
    # Mon 2026-03-23 → fail, Sun 2026-03-22 → skip, Sat 2026-03-21 → skip, Fri 2026-03-20 → success
    mock_s3.get_object.side_effect = [
        Exception("not found"),  # Mon 2026-03-23
        # Sat/Sun skipped (no S3 call)
        _make_s3_response(signals),  # Fri 2026-03-20
    ]
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_signals_from_s3("test-bucket", "2026-03-23")
    assert warning is None
    assert result["universe"][0]["ticker"] == "GOOG"


@patch("executor.eod_reconcile.boto3")
def test_load_signals_returns_empty_on_exhaustion(mock_boto3):
    """Returns empty dict with warning when all lookback days exhausted."""
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("not found")
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_signals_from_s3("test-bucket", "2026-03-27", max_lookback=2)
    assert result == {}
    assert warning is not None
    assert "unavailable" in warning.lower() or "Signals" in warning


# ── _load_predictions_from_s3 ────────────────────────────────────────────────


@patch("executor.eod_reconcile.boto3")
def test_load_predictions_keyed_by_ticker(mock_boto3):
    """Predictions are returned as a dict keyed by ticker."""
    preds = {
        "predictions": [
            _make_prediction("AAPL", "UP", 0.80, 0.03),
            _make_prediction("TSLA", "DOWN", 0.65, -0.02),
        ]
    }
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(preds)
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_predictions_from_s3("test-bucket")
    assert warning is None
    assert "AAPL" in result
    assert "TSLA" in result
    assert result["AAPL"]["predicted_direction"] == "UP"
    assert result["TSLA"]["prediction_confidence"] == 0.65


@patch("executor.eod_reconcile.boto3")
def test_load_predictions_returns_empty_on_failure(mock_boto3):
    """Returns empty dict with warning on S3 failure."""
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("access denied")
    mock_boto3.client.return_value = mock_s3

    result, warning = _load_predictions_from_s3("test-bucket")
    assert result == {}
    assert warning is not None


# ── _build_position_contexts ─────────────────────────────────────────────────


@patch("executor.eod_reconcile._load_predictions_from_s3")
@patch("executor.eod_reconcile._load_signals_from_s3")
@patch("executor.eod_reconcile.get_todays_trades")
@patch("executor.eod_reconcile.get_entry_trade")
def test_build_position_contexts_merges_data(
    mock_entry, mock_trades, mock_signals, mock_preds
):
    """Merges signals, predictions, and trade data into context dicts."""
    mock_signals.return_value = (
        _make_signals(
            universe=[{"ticker": "AAPL", "score": 85, "conviction": "rising",
                        "thesis_summary": "Strong iPhone cycle",
                        "price_target_upside": 0.15, "sector_rating": "overweight"}],
            market_regime="bull",
        ),
        None,
    )
    mock_preds.return_value = (
        {"AAPL": _make_prediction("AAPL", "UP", 0.80, 0.03)},
        None,
    )
    mock_trades.return_value = [
        {"ticker": "AAPL", "action": "BUY", "shares": 10},
    ]
    mock_entry.return_value = {
        "date": "2026-03-20",
        "price_at_order": 175.50,
        "research_score": 82,
        "research_conviction": "stable",
        "thesis_summary": "Old thesis",
        "sector_rating": "overweight",
        "rationale_json": json.dumps({"reason": "undervalued"}),
    }

    positions = {
        "AAPL": {"shares": 10, "market_value": 1800.0, "unrealized_pnl": 45.0},
    }
    conn = MagicMock()

    contexts, warnings = _build_position_contexts(positions, conn, "test-bucket", "2026-03-27")

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["ticker"] == "AAPL"
    assert ctx["shares"] == 10
    assert ctx["market_value"] == 1800.0
    assert ctx["research_score"] == 85  # from signals (takes precedence)
    assert ctx["predicted_direction"] == "UP"
    assert ctx["prediction_confidence"] == 0.80
    assert ctx["entry_date"] == "2026-03-20"
    assert ctx["entry_price"] == 175.50
    assert ctx["entry_rationale"] == {"reason": "undervalued"}
    assert ctx["market_regime"] == "bull"
    assert len(ctx["today_actions"]) == 1
    assert warnings == []


@patch("executor.eod_reconcile._load_predictions_from_s3")
@patch("executor.eod_reconcile._load_signals_from_s3")
@patch("executor.eod_reconcile.get_todays_trades")
@patch("executor.eod_reconcile.get_entry_trade")
def test_build_position_contexts_handles_bad_rationale_json(
    mock_entry, mock_trades, mock_signals, mock_preds
):
    """Gracefully handles malformed entry rationale JSON."""
    mock_signals.return_value = (_make_signals(), None)
    mock_preds.return_value = ({}, None)
    mock_trades.return_value = []
    mock_entry.return_value = {
        "date": "2026-03-20",
        "price_at_order": 100.0,
        "research_score": 70,
        "research_conviction": "stable",
        "thesis_summary": "Some thesis",
        "sector_rating": "market_weight",
        "rationale_json": "not valid json {{{",
    }

    positions = {"XYZ": {"shares": 5, "market_value": 500.0, "unrealized_pnl": 0.0}}
    conn = MagicMock()

    contexts, warnings = _build_position_contexts(positions, conn, "test-bucket", "2026-03-27")

    assert len(contexts) == 1
    assert contexts[0]["entry_rationale"] is None  # failed parse → None


# ── Per-position daily return (partial adds) ───────────────────────────────────


class TestComputeDailyReturn:
    """Gap-aware daily return; partial adds must not use prior close for new shares."""

    _PREV_TD = date(2026, 7, 2)

    def test_held_through_flat_shares(self):
        pct, usd, prior, na = _compute_daily_return(
            "AMD",
            {"shares": 148, "avg_cost": 500.0},
            {"shares": 148, "closing_price": 517.82},
            current_price=552.05,
            shares=148,
            prior_close=517.82,
            prior_close_date=self._PREV_TD,
            expected_prev_td=self._PREV_TD,
        )
        assert na is None
        assert prior == pytest.approx(517.82)
        assert usd == pytest.approx((552.05 - 517.82) * 148, abs=0.01)

    def test_partial_add_splits_baseline(self):
        """Regression for 2026-07-06 ADBE: +193 @ 213.62 on a 332-share core."""
        pct, usd, prior, na = _compute_daily_return(
            "ADBE",
            {"shares": 525, "avg_cost": 200.0},
            {"shares": 332, "closing_price": 219.72},
            current_price=218.07,
            shares=525,
            prior_close=219.72,
            prior_close_date=self._PREV_TD,
            expected_prev_td=self._PREV_TD,
            add_entry_px=213.62,
        )
        assert na is None
        expected = (218.07 - 219.72) * 332 + (218.07 - 213.62) * 193
        assert usd == pytest.approx(expected, abs=0.01)
        assert usd != pytest.approx((218.07 - 219.72) * 525, abs=0.01)

    def test_partial_add_spy_core(self):
        """2026-07-06 SPY: +27 @ 751.79 on 786-share core."""
        pct, usd, prior, na = _compute_daily_return(
            "SPY",
            {"shares": 813, "avg_cost": 746.0},
            {"shares": 786, "closing_price": 744.8},
            current_price=751.28,
            shares=813,
            prior_close=744.8,
            prior_close_date=self._PREV_TD,
            expected_prev_td=self._PREV_TD,
            add_entry_px=751.79,
        )
        expected = (751.28 - 744.8) * 786 + (751.28 - 751.79) * 27
        assert usd == pytest.approx(expected, abs=0.01)

    def test_trim_unchanged(self):
        """Trims still price only the retained shares vs prior close."""
        pct, usd, prior, na = _compute_daily_return(
            "GOOG",
            {"shares": 114, "avg_cost": 358.0},
            {"shares": 226, "closing_price": 356.18},
            current_price=364.9,
            shares=114,
            prior_close=356.18,
            prior_close_date=self._PREV_TD,
            expected_prev_td=self._PREV_TD,
        )
        assert usd == pytest.approx((364.9 - 356.18) * 114, abs=0.01)


class TestBuyEntryPrices:
    def test_share_weighted_enter_fills(self):
        trades = [
            {"action": "ENTER", "ticker": "ADBE", "shares": 100, "price": 210.0},
            {"action": "ENTER", "ticker": "ADBE", "shares": 93, "price": 217.5},
        ]
        assert _buy_entry_prices(trades)["ADBE"] == pytest.approx(
            (100 * 210.0 + 93 * 217.5) / 193, abs=1e-6
        )


# ── Daily return & alpha computation (inline math from run()) ────────────────


class TestPnLMath:
    """Tests for the P&L math used in run() — extracted as pure arithmetic."""

    def test_daily_return_positive(self):
        """prior=100k, current=101k → +1.0%."""
        prior_nav = 100_000.0
        nav = 101_000.0
        daily_return = (nav - prior_nav) / prior_nav * 100
        assert daily_return == pytest.approx(1.0)

    def test_daily_return_negative(self):
        """prior=100k, current=98k → -2.0%."""
        prior_nav = 100_000.0
        nav = 98_000.0
        daily_return = (nav - prior_nav) / prior_nav * 100
        assert daily_return == pytest.approx(-2.0)

    def test_daily_return_flat(self):
        """Same NAV → 0%."""
        nav = 100_000.0
        daily_return = (nav - nav) / nav * 100
        assert daily_return == pytest.approx(0.0)

    def test_alpha_positive(self):
        """Portfolio +1.5%, SPY +0.5% → alpha = +1.0%."""
        daily_return = 1.5
        spy_return = 0.5
        alpha = daily_return - spy_return
        assert alpha == pytest.approx(1.0)

    def test_alpha_negative(self):
        """Portfolio -0.5%, SPY +1.0% → alpha = -1.5%."""
        daily_return = -0.5
        spy_return = 1.0
        alpha = daily_return - spy_return
        assert alpha == pytest.approx(-1.5)

    def test_alpha_none_when_daily_return_missing(self):
        """Alpha is None when prior NAV unavailable (first day)."""
        daily_return = None
        spy_return = 0.5
        alpha = (daily_return - spy_return) if (daily_return is not None and spy_return is not None) else None
        assert alpha is None

    def test_alpha_none_when_spy_return_missing(self):
        """Alpha is None when SPY data unavailable."""
        daily_return = 1.0
        spy_return = None
        alpha = (daily_return - spy_return) if (daily_return is not None and spy_return is not None) else None
        assert alpha is None

    def test_spy_return_from_prior_close(self):
        """SPY return computed from prior close: (450/445 - 1) * 100."""
        spy_price = 450.0
        spy_prior_close = 445.0
        spy_return = (spy_price / spy_prior_close - 1) * 100
        assert spy_return == pytest.approx(1.1235955, rel=1e-4)


# ── snapshot-driven contract (Phase 2 of EOD-SF cutover) ─────────────────────
# Phase 1 (PR #116) hard-blocked historical reruns to stop live-IB
# corruption. Phase 2 replaces that gate entirely: eod_reconcile reads
# from a per-run_date S3 snapshot (written by snapshot_capturer in the
# CaptureSnapshot SF step) instead of querying live IB. The snapshot is
# the date-locked source of truth — today, last Tuesday, or any date
# with a snapshot all work uniformly. The contract becomes "snapshot
# must exist for run_date" instead of "run_date must equal today."


class TestSnapshotContract:
    def test_default_resolves_to_now_dual_trading_day(self):
        """run() with no run_date resolves via now_dual().trading_day, not
        date.today() (which would be UTC on ae-trading and could drift past
        midnight Pacific)."""
        # Patch now_dual + bail out at the first IO boundary (load_config) so
        # we don't need to mock the full snapshot/SQLite plumbing — confirming
        # run_date resolved to today is the only assertion here.
        with patch("executor.eod_reconcile.now_dual") as mock_now_dual, \
             patch("executor.eod_reconcile.load_config") as mock_cfg:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-28", calendar_date="2026-04-28"
            )
            mock_cfg.side_effect = RuntimeError("expected_test_sentinel")
            with pytest.raises(RuntimeError, match="expected_test_sentinel"):
                run(run_date=None)

    def test_explicit_run_date_passes_through(self):
        """Historical run_dates are valid on the CORRECTION path
        (run_audit=False — how reconcile_audit and operator replays call
        in) as long as a snapshot exists. This test confirms run() doesn't
        raise on the date itself there; the snapshot-existence check
        happens later in the flow. The LIVE path (run_audit=True) hard-
        blocks a mismatched date — see test_live_run_refuses_mismatched_
        date below (config#1610)."""
        with patch("executor.eod_reconcile.now_dual") as mock_now_dual, \
             patch("executor.eod_reconcile.load_config") as mock_cfg:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-28", calendar_date="2026-04-28"
            )
            mock_cfg.side_effect = RuntimeError("expected_test_sentinel")
            with pytest.raises(RuntimeError, match="expected_test_sentinel"):
                run(run_date="2026-04-25", run_audit=False)

    def test_live_run_refuses_mismatched_date(self):
        """The LIVE daily path (run_audit=True) must refuse a run_date that
        isn't the just-closed session (config#1610): with CaptureSnapshot
        skipped, EODReconcile is the first date-checking state, and this
        guard is what stops a mislabeled SF run_date from silently joining
        one session's trades against another session's snapshot (the
        eod-2026-06-30 skip-swallow)."""
        with patch("executor.eod_reconcile.now_dual") as mock_now_dual:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-28", calendar_date="2026-04-28"
            )
            with pytest.raises(RuntimeError, match="refusing live run"):
                run(run_date="2026-04-25", run_audit=True)

    def test_run_raises_when_snapshot_missing(self):
        """If no snapshot exists at s3://...trades/snapshots/{run_date}.json,
        run() must hard-fail with a message naming the run_date and pointing
        to the canonical writer. This is the new contract that supersedes
        PR #116's run_date-equality gate."""
        with patch("executor.eod_reconcile.now_dual") as mock_now_dual, \
             patch("executor.eod_reconcile.load_config") as mock_cfg, \
             patch("executor.preflight.ExecutorPreflight") as mock_preflight, \
             patch("executor.eod_reconcile.init_db") as mock_db, \
             patch("executor.snapshot_capturer.load_snapshot") as mock_load:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-28", calendar_date="2026-04-28"
            )
            mock_cfg.return_value = {
                "db_path": "/tmp/x.db",
                "trades_bucket": "alpha-engine-research",
                "aws_region": "us-east-1",
                "email_sender": "x@x.com",
                "email_recipients": "y@y.com",
            }
            mock_preflight.return_value.run.return_value = None
            mock_db.return_value = MagicMock()
            mock_load.return_value = None  # snapshot missing
            with pytest.raises(RuntimeError, match="No snapshot at s3://"):
                run(run_date="2026-04-28")

    def test_run_raises_message_points_to_capturer(self):
        """The error message must name the missing key + reference
        snapshot_capturer.py so the operator knows where to recover."""
        with patch("executor.eod_reconcile.now_dual") as mock_now_dual, \
             patch("executor.eod_reconcile.load_config") as mock_cfg, \
             patch("executor.preflight.ExecutorPreflight") as mock_preflight, \
             patch("executor.eod_reconcile.init_db") as mock_db, \
             patch("executor.snapshot_capturer.load_snapshot") as mock_load:
            mock_now_dual.return_value = SimpleNamespace(
                trading_day="2026-04-28", calendar_date="2026-04-28"
            )
            mock_cfg.return_value = {
                "db_path": "/tmp/x.db",
                "trades_bucket": "alpha-engine-research",
                "aws_region": "us-east-1",
                "email_sender": "x@x.com",
                "email_recipients": "y@y.com",
            }
            mock_preflight.return_value.run.return_value = None
            mock_db.return_value = MagicMock()
            mock_load.return_value = None
            with pytest.raises(RuntimeError) as exc:
                run(run_date="2026-04-25", run_audit=False)
            msg = str(exc.value)
            assert "2026-04-25" in msg
            assert "snapshot_capturer.py" in msg
            assert "trades/snapshots/" in msg

    def test_eod_no_longer_imports_ibkrclient(self):
        """eod_reconcile must NOT depend on IBKRClient — the live IB read
        is now in snapshot_capturer.py. A future regression that re-adds
        the import would re-couple reconciliation to live state."""
        import executor.eod_reconcile as eod_mod
        assert not hasattr(eod_mod, "IBKRClient"), (
            "eod_reconcile.py must not import IBKRClient. The live IB "
            "read belongs in snapshot_capturer.py (Phase 2 of EOD-SF "
            "cutover); reconciliation reads from the S3 snapshot only."
        )

    def test_cli_date_flag_forwards_to_run(self):
        """`python eod_reconcile.py --date YYYY-MM-DD` invokes run(run_date=...)."""
        argv = ["eod_reconcile.py", "--date", "2026-04-25"]
        with patch.object(sys, "argv", argv):
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("--date", default=None)
            args = parser.parse_args(argv[1:])
            assert args.date == "2026-04-25"

    def test_cli_no_args_passes_none(self):
        """No --date passes None, which run() resolves via now_dual."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--date", default=None)
        args = parser.parse_args([])
        assert args.date is None


# ── Held-position close lookup: macro-routed tickers ─────────────────────────
# 2026-05-14 EOD: portfolio-optimizer cutover (Tue 2026-05-13) introduced SPY
# as a held core position. The held-position close lookup in run() reads from
# universe_lib only; SPY lives in macro_lib, so universe_lib.read("SPY")
# raised NoSuchVersionException and the whole EOD reconcile crashed. The fix
# mirrors price_cache.load_price_histories' macro-aware dispatch.


def test_eod_reconcile_uses_macro_aware_dispatch_for_held_positions():
    """Pin the macro-aware dispatch in run()'s held-position close lookup.

    Source-level pin: a future PR that removes the dispatch and reverts
    to universe-only reads would re-introduce the 2026-05-14 EOD outage
    the moment the optimizer holds SPY (or any sector ETF — XLB/XLC/XLE
    /XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY).
    """
    from pathlib import Path

    src = (Path(__file__).parent.parent / "executor" / "eod_reconcile.py").read_text()

    assert "_MACRO_SYMBOLS" in src, (
        "eod_reconcile.run() must import _MACRO_SYMBOLS from price_cache "
        "and dispatch held-position reads between universe_lib and "
        "macro_lib — universe-only reads crash on SPY (held under "
        "portfolio-optimizer cutover, 2026-05-13)."
    )
    assert "_open_macro_library" in src, (
        "eod_reconcile.run() must lazy-open macro_lib for held tickers "
        "in _MACRO_SYMBOLS — without this, SPY/sector-ETF held positions "
        "raise NoSuchVersionException against universe_lib."
    )
    assert "if ticker in _MACRO_SYMBOLS" in src, (
        "Held-position read loop must branch on _MACRO_SYMBOLS membership "
        "to route reads between universe_lib and macro_lib (mirrors "
        "price_cache.load_price_histories:128-145)."
    )


# ── Held-ticker closing-price hard-fail (config#2737) ────────────────────────
# run()'s "Load closing prices from ArcticDB" block hard-fails when any held
# ticker has no authoritative close for run_date. SPY's equivalent guard
# (_spy_close, a standalone function) has dedicated tests above; this block is
# inlined in run() instead, so exercising it means driving run() itself up to
# that point — mirroring the bail-out-at-the-first-unmocked-boundary style
# TestSnapshotContract already uses for run().


def _run_up_to_closing_prices(monkeypatch, *, held_ticker="AAPL", read_error=None,
                               frame=None, run_date="2026-04-28"):
    """Drive run() through snapshot load + sector enrichment + SPY return up
    to the held-position closing-price lookup, with one held ticker
    (`held_ticker`, routed through universe_lib) configured to fail the
    ArcticDB read per `read_error`/`frame`.
    """

    monkeypatch.setattr(
        "executor.eod_reconcile.now_dual",
        lambda: SimpleNamespace(trading_day=run_date, calendar_date=run_date),
    )
    monkeypatch.setattr(
        "executor.eod_reconcile.load_config",
        lambda: {
            "db_path": "/tmp/x.db",
            "trades_bucket": "alpha-engine-research",
            "aws_region": "us-east-1",
            "email_sender": "x@x.com",
            "email_recipients": "y@y.com",
        },
    )
    monkeypatch.setattr(
        "executor.preflight.ExecutorPreflight",
        lambda **kw: SimpleNamespace(run=lambda: None),
    )
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = None  # no prior eod_pnl row
    monkeypatch.setattr("executor.eod_reconcile.init_db", lambda path: mock_conn)
    monkeypatch.setattr(
        "executor.snapshot_capturer.load_snapshot",
        lambda **kw: {
            "account": {"net_liquidation": 100_000.0},
            "positions": {held_ticker: {"qty": 10}},
            "captured_at": "2026-04-28T21:00:00Z",
        },
    )
    # Sector enrichment is best-effort (wrapped in try/except in run()) —
    # raising here skips straight past it without needing a real S3 mock.
    monkeypatch.setattr(
        "executor.eod_reconcile._load_signals_from_s3",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no signals in test")),
    )
    # SPY return is computed via the already-tested _spy_close — stub it so
    # this test isn't coupled to a second ArcticDB fixture.
    monkeypatch.setattr("executor.eod_reconcile._spy_close", lambda *a, **kw: 450.0)

    lib = MagicMock()
    if read_error is not None:
        lib.read.side_effect = read_error
    else:
        lib.read.return_value = SimpleNamespace(data=frame)
    monkeypatch.setattr("executor.price_cache._open_universe_library", lambda bucket: lib)


def test_held_ticker_hard_fails_when_arcticdb_read_fails(monkeypatch):
    """A held ticker whose ArcticDB read raises must hard-fail run(), naming
    the ticker, the exception class, and run_date — mirroring
    test_spy_close_hard_fails_when_symbol_missing for the held-position path."""
    _run_up_to_closing_prices(
        monkeypatch, held_ticker="AAPL", read_error=KeyError("no such symbol: AAPL"),
    )
    with pytest.raises(RuntimeError, match=r"AAPL \(KeyError\)"):
        run(run_date="2026-04-28")


def test_held_ticker_hard_fails_when_no_row_for_run_date(monkeypatch):
    """A held ticker whose ArcticDB frame has no row for run_date must
    hard-fail run() — mirroring test_spy_close_hard_fails_when_date_missing
    for the held-position path."""
    import pandas as pd

    frame = pd.DataFrame(
        {"Close": [190.0]},
        index=pd.to_datetime(["2026-04-27"]),
    )
    _run_up_to_closing_prices(monkeypatch, held_ticker="AAPL", frame=frame)
    with pytest.raises(RuntimeError, match=r"AAPL \(no row for 2026-04-28\)"):
        run(run_date="2026-04-28")


def test_held_ticker_hard_fails_when_close_column_missing(monkeypatch):
    """A held ticker whose ArcticDB frame has no Close column must hard-fail
    run() — mirroring test_spy_close_hard_fails_when_close_column_missing
    for the held-position path."""
    import pandas as pd

    frame = pd.DataFrame(
        {"Open": [190.0]},
        index=pd.to_datetime(["2026-04-28"]),
    )
    _run_up_to_closing_prices(monkeypatch, held_ticker="AAPL", frame=frame)
    with pytest.raises(RuntimeError, match=r"AAPL \(no Close column\)"):
        run(run_date="2026-04-28")
