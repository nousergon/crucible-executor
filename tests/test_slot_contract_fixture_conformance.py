"""Consumer-side conformance: the executor's contract-test fixtures must be
valid Slot R / Slot M artifacts (M0, config#989).

The executor is the canonical consumer of both slot contracts. Its consumer
contract tests (``test_signal_reader_consumer_contract.py``) exercise
``signal_reader`` against fixtures — if those fixtures drift from the real
producer shape, the consumer tests prove nothing about the actual boundary.
This pins fixture <-> contract agreement via ``alpha_engine_lib.contracts``
(lib >= 0.59.1), the same versioned schemas the producers validate against
(research / predictor CI).

Fail-soft tolerance is NOT weakened: signal_reader may still degrade gracefully
on partial payloads at runtime; this only asserts that the *reference fixtures*
used to certify the consumer are contract-complete.
"""

from __future__ import annotations

import pytest

contracts = pytest.importorskip(
    "alpha_engine_lib.contracts",
    reason="needs alpha-engine-lib[contracts] >= 0.59.1",
)

from tests.test_signal_reader_consumer_contract import _complete_signals


def _complete_predictions() -> dict:
    """Reference Slot M payload mirroring read_predictions' expectations."""
    return {
        "date": "2026-06-15",
        "model_version": "v3.0-test",
        "n_predictions": 2,
        "predictions": [
            {
                "ticker": "AAA",
                "predicted_direction": "UP",
                "prediction_confidence": 0.7,
                "predicted_alpha": 0.04,
                "combined_rank": 1,
                "gbm_veto": False,
                "momentum_veto": False,
                "barrier_win_prob": 0.6,
                "predicted_alpha_std": 0.01,
            },
            {
                "ticker": "BBB",
                "predicted_direction": "DOWN",
                "prediction_confidence": 0.8,
                "predicted_alpha": -0.05,
                "combined_rank": 2,
                "gbm_veto": True,
                "momentum_veto": True,
            },
        ],
    }


class TestConsumerFixturesConformToSlotContracts:
    def test_signals_fixture_is_a_valid_slot_r_artifact(self):
        contracts.validate("signals", _complete_signals())

    def test_predictions_fixture_is_a_valid_slot_m_artifact(self):
        contracts.validate("predictions", _complete_predictions())

    def test_red_fixture_fails_loud(self):
        """config#989 closes-when demo on the consumer side."""
        broken = _complete_signals()
        del broken["universe"][0]["sector_rating"]
        errors = contracts.conformance_errors("signals", broken)
        assert errors and "sector_rating" in " ".join(errors)


class TestReaderAcceptsContractShapedPayloads:
    """The reader must consume a fully contract-conform payload end-to-end —
    guards against the reader growing an undocumented reliance the contract
    doesn't promise."""

    def test_get_actionable_signals_on_conform_payload(self):
        from executor.signal_reader import get_actionable_signals

        payload = _complete_signals()
        contracts.validate("signals", payload)
        result = get_actionable_signals(payload)
        tickers = {
            e["ticker"] for k in ("enter", "exit", "reduce", "hold") for e in result[k]
        }
        assert "AAA" in tickers
