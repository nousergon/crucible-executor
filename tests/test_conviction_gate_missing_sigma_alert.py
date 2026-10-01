"""alpha-engine-config-I11791: a live budget losing its conviction gate to a
missing sigma is unthrottled (by design) but never silent."""
from __future__ import annotations

import logging

import numpy as np

from executor.portfolio_optimizer import compute_conviction_budget_multiplier

_ALPHA = np.array([0.02, -0.01, 0.03, 0.0, 0.0])
_LIVE = {"max_daily_turnover": 0.10}


def _errors(caplog):
    return [r for r in caplog.records if r.levelno == logging.ERROR]


def test_missing_sigma_vector_alerts_and_stays_unthrottled(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="executor.portfolio_optimizer"):
        out = compute_conviction_budget_multiplier(_ALPHA, None, None, 3, 4, _LIVE)
    assert out["conviction_budget_multiplier"] == 1.0
    assert out["conviction_gate_reason"] == "no_alpha_uncertainty_vector"
    assert any("I11791" in r.getMessage() for r in _errors(caplog))


def test_all_nan_sigma_alerts(caplog) -> None:
    sigma = np.full(5, np.nan)
    with caplog.at_level(logging.ERROR, logger="executor.portfolio_optimizer"):
        out = compute_conviction_budget_multiplier(_ALPHA, sigma, None, 3, 4, _LIVE)
    assert out["conviction_gate_reason"] == "no_usable_alpha_uncertainty"
    assert _errors(caplog)


def test_no_alert_when_the_budget_is_off(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="executor.portfolio_optimizer"):
        compute_conviction_budget_multiplier(_ALPHA, None, None, 3, 4, {})
    assert not _errors(caplog)


def test_no_alert_when_sigma_is_present(caplog) -> None:
    sigma = np.full(5, 0.1)
    with caplog.at_level(logging.ERROR, logger="executor.portfolio_optimizer"):
        out = compute_conviction_budget_multiplier(_ALPHA, sigma, None, 3, 4, _LIVE)
    assert out["conviction_ir_xs"] is not None
    assert not _errors(caplog)
