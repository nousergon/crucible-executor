"""Unit tests for executor.eod_reconcile — P&L math and data helpers."""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from executor.eod_reconcile import (
    _spy_close,
    _load_signals_from_s3,
    _load_predictions_from_s3,
    _build_position_contexts,
)


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


@patch("executor.eod_reconcile.yf")
def test_spy_close_returns_float(mock_yf):
    """_spy_close returns the SPY closing price as a float."""
    import pandas as pd
    import numpy as np

    mock_hist = pd.DataFrame({"Close": [450.25]})
    mock_yf.download.return_value = mock_hist
    result = _spy_close("2026-03-27")
    assert result == pytest.approx(450.25)
    mock_yf.download.assert_called_once()


@patch("executor.eod_reconcile.yf")
def test_spy_close_returns_none_on_empty(mock_yf):
    """_spy_close returns None when yfinance returns no data."""
    import pandas as pd

    mock_yf.download.return_value = pd.DataFrame()
    result = _spy_close("2026-03-27")
    assert result is None


@patch("executor.eod_reconcile.yf")
def test_spy_close_returns_none_on_exception(mock_yf):
    """_spy_close returns None when yfinance raises."""
    mock_yf.download.side_effect = Exception("network error")
    result = _spy_close("2026-03-27")
    assert result is None


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
