"""merge_champion_predictions + the paginated constituents sector-map load
(2026-10-01: champion picks reached the optimizer at alpha_hat 0.0 although
the predictor had scored them; the sector map was read from 2026-07-23)."""
from unittest.mock import MagicMock, patch

from executor.champion import merge_champion_predictions


def _no_alpha_row():
    return {"predicted_alpha": None, "predicted_direction": None,
            "prediction_confidence": 0.0, "shadow_arm": "attractiveness_20",
            "arm_score": 0.9}


def test_no_alpha_injection_keeps_the_real_prediction():
    real = {"NVDA": {"predicted_alpha": -0.0107, "prediction_confidence": 0.19,
                     "predicted_direction": "DOWN"}}
    merged = merge_champion_predictions(real, {"NVDA": _no_alpha_row()})
    assert merged["NVDA"]["predicted_alpha"] == -0.0107
    assert merged["NVDA"]["prediction_confidence"] == 0.19
    assert merged["NVDA"]["shadow_arm"] == "attractiveness_20"
    assert merged["NVDA"]["arm_score"] == 0.9


def test_no_alpha_injection_fills_a_gap():
    merged = merge_champion_predictions({}, {"NEW": _no_alpha_row()})
    assert merged["NEW"]["predicted_alpha"] is None
    assert merged["NEW"]["prediction_confidence"] == 0.0


def test_injected_numeric_alpha_still_replaces_the_real_row():
    real = {"MSFT": {"predicted_alpha": 0.01}}
    injected = {"MSFT": {"predicted_alpha": 0.03, "prediction_confidence": 0.0}}
    assert merge_champion_predictions(real, injected)["MSFT"]["predicted_alpha"] == 0.03


def test_real_row_with_no_alpha_is_replaced():
    real = {"X": {"predicted_alpha": None, "prediction_confidence": 0.7}}
    merged = merge_champion_predictions(real, {"X": _no_alpha_row()})
    assert merged["X"]["prediction_confidence"] == 0.0


def test_inputs_are_not_mutated():
    real = {"NVDA": {"predicted_alpha": -0.01}}
    merge_champion_predictions(real, {"NVDA": _no_alpha_row()})
    assert real == {"NVDA": {"predicted_alpha": -0.01}}


def test_constituents_sector_map_reads_past_the_first_list_page():
    from executor import eod_reconcile

    pages = [
        {"Contents": [{"Key": "market_data/weekly/2026-07-23/constituents.json"}]},
        {"Contents": [{"Key": "market_data/weekly/2026-09-28/constituents.json"},
                      {"Key": "market_data/weekly/2026-09-30/manifest.json"}]},
    ]
    s3 = MagicMock()
    s3.get_paginator.return_value.paginate.return_value = pages
    s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: b'{"sector_map": {"VYLR": "Materials"}}')
    }
    with patch.object(eod_reconcile.boto3, "client", return_value=s3):
        out = eod_reconcile._load_constituents_sector_map("bucket")
    assert out == {"VYLR": "Materials"}
    s3.get_object.assert_called_once_with(
        Bucket="bucket", Key="market_data/weekly/2026-09-28/constituents.json",
    )
