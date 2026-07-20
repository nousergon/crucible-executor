"""Unit tests for executor.signal_reader coverage guard.

Guards against the Research↔Predictor coverage gap first observed 2026-04-20.
The guard refuses to size positions when buy_candidates has tickers missing
from predictions.json — otherwise the GBM veto gate is unreachable for those
tickers, routing position sizing around a risk control.
"""
from unittest.mock import patch

import pytest

from executor.signal_reader import (
    UnscoredBuyCandidatesError,
    assert_predictions_cover_buy_candidates,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _signals(buy_tickers: list[str]) -> dict:
    return {
        "date": "2026-04-20",
        "buy_candidates": [{"ticker": t, "signal": "ENTER"} for t in buy_tickers],
        "universe": [{"ticker": t, "signal": "ENTER"} for t in buy_tickers],
    }


def _preds(pred_tickers: list[str]) -> dict:
    return {t: {"ticker": t, "predicted_alpha": 0.01} for t in pred_tickers}


# ── Happy path ───────────────────────────────────────────────────────────────

@patch("executor.signal_reader.boto3")
def test_passes_when_all_buy_candidates_scored(mock_boto3):
    signals = _signals(["AAPL", "MSFT"])
    preds = _preds(["AAPL", "MSFT"])
    # Should not raise
    assert_predictions_cover_buy_candidates(signals, preds)
    # Metric emitted with value 0 (continuous baseline)
    mock_boto3.client.return_value.put_metric_data.assert_called_once()
    call = mock_boto3.client.return_value.put_metric_data.call_args
    assert call.kwargs["MetricData"][0]["Value"] == 0.0


@patch("executor.signal_reader.boto3")
def test_passes_when_predictions_is_superset(mock_boto3):
    # Executor may have extra predictions (e.g. from a prior run's universe);
    # only buy_candidate ⊆ predictions matters.
    signals = _signals(["AAPL"])
    preds = _preds(["AAPL", "MSFT", "GOOG"])
    assert_predictions_cover_buy_candidates(signals, preds)


@patch("executor.signal_reader.boto3")
def test_empty_buy_candidates_is_no_op(mock_boto3):
    signals = _signals([])
    preds = _preds(["AAPL"])
    assert_predictions_cover_buy_candidates(signals, preds)


# ── Failure paths ────────────────────────────────────────────────────────────

@patch("executor.signal_reader.boto3")
def test_raises_with_missing_tickers_in_message(mock_boto3):
    # Today's bug reproduced: 4 buy_candidates missing from predictions.
    signals = _signals(["AAPL", "SNDK", "WDC", "BIIB", "XEL", "CTAS"])
    preds = _preds(["AAPL", "CTAS"])
    with pytest.raises(UnscoredBuyCandidatesError) as exc:
        assert_predictions_cover_buy_candidates(signals, preds)
    msg = str(exc.value)
    assert "SNDK" in msg and "WDC" in msg and "BIIB" in msg and "XEL" in msg
    assert "4" in msg  # count of missing
    assert exc.value.missing == ["BIIB", "SNDK", "WDC", "XEL"]  # sorted
    assert exc.value.n_buy == 6
    assert exc.value.n_preds == 2


@patch("executor.signal_reader.boto3")
def test_metric_emitted_even_when_raising(mock_boto3):
    signals = _signals(["AAPL", "SNDK"])
    preds = _preds(["AAPL"])
    with pytest.raises(UnscoredBuyCandidatesError):
        assert_predictions_cover_buy_candidates(signals, preds)
    # Metric must still be emitted before the raise (alarm relies on it)
    mock_boto3.client.return_value.put_metric_data.assert_called_once()
    call = mock_boto3.client.return_value.put_metric_data.call_args
    assert call.kwargs["MetricData"][0]["Value"] == 1.0
    assert call.kwargs["Namespace"] == "AlphaEngine/Predictor"


@patch("executor.signal_reader.boto3")
def test_case_insensitive_ticker_comparison(mock_boto3):
    signals = {
        "buy_candidates": [{"ticker": "aapl"}, {"ticker": "MSFT"}],
        "universe": [],
    }
    preds = {"AAPL": {"predicted_alpha": 0.01}, "msft": {"predicted_alpha": 0.02}}
    # Both should match despite case differences
    assert_predictions_cover_buy_candidates(signals, preds)


# ── Observability is best-effort ─────────────────────────────────────────────

@patch("executor.signal_reader.boto3")
def test_metric_emission_failure_does_not_block_success(mock_boto3):
    # CloudWatch put_metric_data can fail (IAM, network) — must never block
    # the executor's trading path.
    mock_boto3.client.return_value.put_metric_data.side_effect = Exception("cw down")
    signals = _signals(["AAPL"])
    preds = _preds(["AAPL"])
    # Happy path still succeeds despite CloudWatch failure
    assert_predictions_cover_buy_candidates(signals, preds)


@patch("executor.signal_reader.boto3")
def test_metric_emission_failure_does_not_mask_hard_fail(mock_boto3):
    mock_boto3.client.return_value.put_metric_data.side_effect = Exception("cw down")
    signals = _signals(["AAPL", "SNDK"])
    preds = _preds(["AAPL"])
    # Coverage gap still raises even though CloudWatch failed
    with pytest.raises(UnscoredBuyCandidatesError):
        assert_predictions_cover_buy_candidates(signals, preds)
