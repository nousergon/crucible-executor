"""Consumer-side contract tests for the executor's two cross-repo inputs (L4520).

The executor CONSUMES:
  - signals.json   (produced by alpha-engine-research)  via read_signals /
    get_actionable_signals
  - predictions.json (produced by alpha-engine-predictor) via read_predictions

Both reads are deliberately FAIL-SOFT (``.get(...)`` defaults / a graceful
prior-day fallback) so a missing upstream artifact degrades rather than halting
trading. That same softness means a producer dropping a field is SILENT — which
is why the loud catch lives in the PRODUCERS' own CI (research
test_signals_producer_contract / predictor test_predictions_producer_contract).

These tests pin the CONSUMER side of the same contract: the exact field set the
executor reads, and the *intended* fail-soft defaults. They fail loud if a
future executor change stops consuming a contract field or silently changes the
degrade posture — without converting the deliberate graceful-degrade reads to
raise (the prior-day fallback is load-bearing).

Contract SoT: alpha-engine-config/private-docs/PIPELINE_CONTRACT.yaml boundaries
`signals` and `predictions` (drift-proof lib-hosted binding is a filed follow-up).
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from executor.signal_reader import (
    get_actionable_signals,
    read_predictions,
    read_signals,
)


# ── signals.json boundary (research → executor) ──────────────────────────────

# Per-ticker fields get_actionable_signals relies on (PIPELINE_CONTRACT `signals`).
_SIGNALS_PER_ITEM = {"ticker", "signal"}
# Top-level fields the executor reads (with intentional fail-soft defaults).
_SIGNALS_TOP_LEVEL = {"universe", "buy_candidates", "market_regime", "sector_ratings"}


def _complete_signals() -> dict:
    return {
        "date": "2026-06-15",
        "market_regime": "bull",
        "sector_modifiers": {"Technology": 1.1},
        "sector_ratings": {"Technology": {"rating": "overweight", "modifier": 1.1}},
        "signals": {},
        "population": ["AAA", "BBB", "CCC"],
        "universe": [
            {"ticker": "AAA", "signal": "ENTER", "score": 80, "conviction": "rising",
             "sector": "Technology", "sector_rating": "overweight", "price_target_upside": 0.2},
            {"ticker": "BBB", "signal": "HOLD", "score": 60, "conviction": "stable",
             "sector": "Technology", "sector_rating": "market_weight", "price_target_upside": 0.0},
            {"ticker": "CCC", "signal": "EXIT", "score": 30, "conviction": "declining",
             "sector": "Technology", "sector_rating": "underweight", "price_target_upside": 0.0},
        ],
        "buy_candidates": [
            {"ticker": "AAA", "signal": "ENTER", "score": 80, "conviction": "rising",
             "sector": "Technology", "sector_rating": "overweight", "price_target_upside": 0.2},
        ],
    }


def _fake_s3(payload: dict) -> MagicMock:
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode())}
    return s3


def test_read_signals_consumes_the_contract_top_level_fields():
    s3 = _fake_s3(_complete_signals())
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        data = read_signals("alpha-engine-research", "2026-06-15")
    # The executor relies on these top-level keys downstream; pin their presence.
    for field in _SIGNALS_TOP_LEVEL:
        assert field in data, f"read_signals output must carry contract field {field!r}"


def test_get_actionable_signals_reads_signal_and_ticker_per_item():
    actionable = get_actionable_signals(_complete_signals())
    # Bucketing by `signal` proves the consumer reads the per-ticker `signal`...
    assert [s["ticker"] for s in actionable["enter"]] == ["AAA"]
    assert [s["ticker"] for s in actionable["exit"]] == ["CCC"]
    assert [s["ticker"] for s in actionable["hold"]] == ["BBB"]
    # ...and every bucketed entry retains the per-item contract fields.
    for bucket in ("enter", "exit", "hold"):
        for entry in actionable[bucket]:
            assert _SIGNALS_PER_ITEM <= entry.keys()
    # Top-level regime + sector_ratings are surfaced (consumed for sizing).
    assert actionable["market_regime"] == "bull"
    assert actionable["sector_ratings"]["Technology"]["rating"] == "overweight"


def test_get_actionable_signals_fail_soft_defaults_are_intentional():
    # Contract posture: a payload missing market_regime / sector_ratings must
    # DEGRADE (regime→"neutral", ratings→{}), NOT raise — the executor trades
    # on last-known/neutral rather than halting. Pin the deliberate default.
    minimal = {"universe": [], "buy_candidates": []}
    actionable = get_actionable_signals(minimal)
    assert actionable["market_regime"] == "neutral"
    assert actionable["sector_ratings"] == {}


# ── predictions.json boundary (predictor → executor) ─────────────────────────


def test_read_predictions_keys_by_ticker_and_surfaces_date():
    payload = {
        "date": "2026-06-15",
        "n_predictions": 2,
        "predictions": [
            {"ticker": "AAA", "predicted_direction": "UP", "prediction_confidence": 0.6,
             "predicted_alpha": 0.03, "combined_rank": 1},
            {"ticker": "BBB", "predicted_direction": "DOWN", "prediction_confidence": 0.7,
             "predicted_alpha": -0.02, "combined_rank": 2},
        ],
    }
    s3 = _fake_s3(payload)
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        result, pdate = read_predictions("alpha-engine-research")
    assert pdate == "2026-06-15"
    assert set(result) == {"AAA", "BBB"}
    # Per-ticker contract fields are preserved through the keying transform.
    for fields in result.values():
        assert {"ticker", "predicted_direction", "prediction_confidence",
                "predicted_alpha", "combined_rank"} <= fields.keys()


def test_read_predictions_drops_ticker_less_entries_fail_soft():
    # Contract posture: a per-ticker entry missing `ticker` is silently dropped
    # (the `if "ticker" in p` guard), NOT raised — pin that deliberate softness.
    payload = {"date": "2026-06-15", "predictions": [
        {"ticker": "AAA", "predicted_direction": "UP"},
        {"predicted_direction": "DOWN"},  # no ticker → dropped
    ]}
    s3 = _fake_s3(payload)
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        result, _ = read_predictions("alpha-engine-research")
    assert set(result) == {"AAA"}
