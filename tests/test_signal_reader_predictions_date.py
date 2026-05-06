"""Tests that read_predictions surfaces the predictions/{date}.json
filename date alongside the per-ticker dict.

The date is read off latest.json's top-level ``date`` field. It's
distinct from today's date when the latest pointer resolves to a prior
trading day's predictions (Saturday window, holiday, weekday SF
delayed). Trade logging needs the actual filename for the lineage
column; surfacing run_date instead would silently misattribute.

Closes the artifact-lineage producer side of the Phase 2
transparency-inventory ROADMAP item — the daemon's log_trade call
inherits this value through entries_with_meta + urgent_exits_with_meta.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from botocore.exceptions import ClientError

from executor.signal_reader import read_predictions


def _fake_s3_client_with_payload(payload: dict):
    """Return a MagicMock that mimics boto3.client('s3') and replies to
    get_object with ``payload`` as the JSON body."""
    from unittest.mock import MagicMock

    s3 = MagicMock()
    s3.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(payload).encode()),
    }
    return s3


def test_read_predictions_returns_dict_and_date():
    """Happy-path: latest.json with a top-level date and per-ticker
    predictions list. read_predictions surfaces both."""
    payload = {
        "date": "2026-05-06",
        "model_version": "v3.0-meta",
        "predictions": [
            {"ticker": "NVDA", "predicted_direction": "UP", "p_up": 0.62},
            {"ticker": "AAPL", "predicted_direction": "DOWN", "p_up": 0.38},
        ],
    }
    s3 = _fake_s3_client_with_payload(payload)
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        result, date = read_predictions("alpha-engine-research")

    assert date == "2026-05-06"
    assert set(result.keys()) == {"NVDA", "AAPL"}
    assert result["NVDA"]["predicted_direction"] == "UP"


def test_read_predictions_date_is_none_when_payload_missing_field():
    """Defensive: if the predictor's payload regresses and stops
    emitting `date` at top level, the executor still loads the
    per-ticker dict (lineage just shows NULL). Better than hard-failing
    the whole executor on a non-load-bearing field."""
    payload = {
        "predictions": [
            {"ticker": "NVDA", "predicted_direction": "UP"},
        ],
    }
    s3 = _fake_s3_client_with_payload(payload)
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        result, date = read_predictions("alpha-engine-research")

    assert date is None
    assert "NVDA" in result


def test_read_predictions_returns_empty_and_none_on_no_such_key():
    """latest.json missing — daemon runs without GBM (degraded mode).
    Both halves of the tuple stay NULL/empty so callers can branch."""
    from unittest.mock import MagicMock

    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        error_response={"Error": {"Code": "NoSuchKey"}},
        operation_name="GetObject",
    )
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        result, date = read_predictions("alpha-engine-research")

    assert result == {}
    assert date is None
