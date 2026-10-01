"""negative_alpha_exit_below: a name the predictor says will underperform is
pinned to zero, held or not, so its exit is mandatory turnover the conviction
gate cannot throttle (2026-10-01)."""
import numpy as np

from executor.optimizer_shadow import _build_eligibility

TICKERS = ["HL", "MRNA", "NEW", "NOPRED", "SPY", "CASH"]
SPY, CASH = 4, 5
SIGNALS = {t: {"signal": "HOLD", "score": 80} for t in TICKERS[:4]}
PREDS = {
    "HL": {"predicted_alpha": -0.032},
    "MRNA": {"predicted_alpha": 0.074},
    "NEW": {"predicted_alpha": -0.004},
    "NOPRED": {"predicted_alpha": None, "prediction_confidence": 0.0},
}
HELD = {"HL": {}, "MRNA": {}, "NOPRED": {}}


def _run(threshold):
    cfg = {"min_score_to_enter": 30}
    if threshold is not None:
        cfg["negative_alpha_exit_below"] = threshold
    return _build_eligibility(TICKERS, SIGNALS, PREDS, HELD, cfg, SPY, CASH)


def test_off_by_default_keeps_held_negative_alpha_eligible():
    elig, reasons = _run(None)
    assert elig.all()
    assert reasons == [None] * len(TICKERS)


def test_held_name_below_threshold_is_pinned_out():
    elig, reasons = _run(-0.01)
    assert not elig[0] and reasons[0] == "negative_alpha"
    assert elig[1]


def test_threshold_is_strict_and_applies_to_unheld_names():
    elig, reasons = _run(0.0)
    assert reasons[2] == "negative_alpha"
    elig, _ = _run(-0.004)
    assert elig[2]


def test_no_numeric_alpha_is_never_pinned_by_this_rule():
    elig, _ = _run(0.0)
    assert elig[3]


def test_benchmark_and_cash_are_exempt():
    elig, _ = _run(1.0)
    assert elig[SPY] and elig[CASH]
    assert not np.any(elig[:3])
