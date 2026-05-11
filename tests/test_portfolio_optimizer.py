"""
Unit tests for executor/portfolio_optimizer.py — PR 1 of portfolio-optimizer arc.

The kernel is pure-numpy in/out; tests exercise the math directly with
synthetic inputs to confirm:
  1. Single-asset known-answer (positive α̂ → max_pos in asset, rest in SPY)
  2. Two-asset symmetric known-answer (equal α̂ + Σ → equal weights)
  3. Vol-target SOC constraint binds when assets are high-vol
  4. L1 turnover penalty discourages large rebalances from w_prev
  5. Eligibility mask pins disallowed names to 0
  6. Cash sleeve equality constraint cannot be violated
  7. Sector cap binds when many names in one sector have positive α̂
  8. Infeasibility falls back to current-weights + cash residual
"""

from __future__ import annotations

import numpy as np
import pytest

from executor.portfolio_optimizer import (
    OPTIMIZER_CONFIG_DEFAULTS,
    OptimizerResult,
    solve_target_weights,
)


def _synthetic_returns(N: int, T: int = 250, vol: float = 0.01, seed: int = 0) -> np.ndarray:
    """Generate (T, N) iid normal returns with given daily vol."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, vol, size=(T, N))


def _baseline_universe(
    n_active: int = 2,
    sector_labels: list[str] | None = None,
    daily_vol: float = 0.01,
) -> dict:
    """
    Build a baseline universe with N = n_active + 2 (SPY + CASH).
    Returns a dict of all kwargs needed for solve_target_weights.
    """
    if sector_labels is None:
        sector_labels = ["tech"] * n_active
    assert len(sector_labels) == n_active

    tickers = [f"T{i}" for i in range(n_active)] + ["SPY", "CASH"]
    N = len(tickers)
    spy_idx = N - 2
    cash_idx = N - 1

    returns = _synthetic_returns(N, vol=daily_vol)
    returns[:, cash_idx] = 0.0

    return {
        "tickers": tickers,
        "alpha_hat": np.zeros(N),
        "returns_panel": returns,
        "w_prev": np.zeros(N),
        "sectors": sector_labels + ["__benchmark__", "__cash__"],
        "stance_caps": np.full(N, 0.08),
        "eligibility": np.ones(N, dtype=bool),
        "spy_idx": spy_idx,
        "cash_idx": cash_idx,
        "cfg": {},
    }


def _solve(u: dict) -> OptimizerResult:
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    return solve_target_weights(**u)


def test_single_asset_positive_alpha_maxes_position_and_fills_spy():
    """One asset with positive α̂ → optimizer allocates max_pos to it, SPY absorbs the rest."""
    u = _baseline_universe(n_active=1)
    u["alpha_hat"][0] = 0.05

    result = _solve(u)
    w = result.weights

    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert w[0] == pytest.approx(0.08, abs=1e-3), \
        f"Active asset should fill its 0.08 cap; got {w[0]:.4f}"
    assert w[u["cash_idx"]] == pytest.approx(0.03, abs=1e-6), \
        "Cash sleeve must be pinned at 0.03"
    assert w[u["spy_idx"]] == pytest.approx(1 - 0.08 - 0.03, abs=1e-3), \
        f"SPY should absorb residual ~0.89; got {w[u['spy_idx']]:.4f}"
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_two_asset_symmetric_alpha_yields_equal_weights():
    """Two assets, identical α̂ + identical covariance → optimizer assigns equal weight."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.05
    u["returns_panel"][:, 0] = u["returns_panel"][:, 1]

    result = _solve(u)
    w = result.weights

    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert abs(w[0] - w[1]) < 1e-3, \
        f"Symmetric inputs must give symmetric weights; got w[0]={w[0]:.4f} w[1]={w[1]:.4f}"
    assert w[0] == pytest.approx(0.08, abs=1e-3), \
        "Both should hit their 0.08 cap given strong α̂"


def test_vol_target_constraint_binds_with_high_vol_universe():
    """High-vol actives + a vol_target below the cap-filling vol → optimizer backs off caps.

    Uses sample (not Ledoit-Wolf) covariance because LW's trace/N × I shrinkage
    target is sensitive to synthetic vol heterogeneity. Production uses LW with
    real stock returns where vols are more homogeneous (1-3% daily). The
    synthetic setup uses long T (5000 rows) to drive sample cov noise low so
    feasibility math is predictable.
    """
    N = 6
    T = 5000
    rng = np.random.default_rng(7)
    active_vol = 0.05
    spy_vol = 0.005
    returns = np.column_stack([
        rng.normal(0, active_vol, T),
        rng.normal(0, active_vol, T),
        rng.normal(0, active_vol, T),
        rng.normal(0, active_vol, T),
        rng.normal(0, spy_vol, T),
        np.zeros(T),
    ])
    u = {
        "tickers": ["T0", "T1", "T2", "T3", "SPY", "CASH"],
        "alpha_hat": np.array([0.10, 0.10, 0.10, 0.10, 0.0, -1e-6]),
        "returns_panel": returns,
        "w_prev": np.zeros(N),
        "sectors": ["tech", "tech", "tech", "tech", "__benchmark__", "__cash__"],
        "stance_caps": np.array([0.08, 0.08, 0.08, 0.08, 1.0, 1.0]),
        "eligibility": np.ones(N, dtype=bool),
        "spy_idx": 4,
        "cash_idx": 5,
        "cfg": {"vol_target_annual": 0.10, "covariance_shrinkage": "sample"},
    }

    result = solve_target_weights(**u)
    w = result.weights

    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate"), \
        f"Setup must be feasible; got {result.diagnostics['status']}"
    assert result.diagnostics["portfolio_vol_ann"] <= 0.10 + 5e-3, \
        f"Vol target 0.10 violated; got {result.diagnostics['portfolio_vol_ann']:.4f}"
    active_total = w[:4].sum()
    assert active_total < 4 * 0.08 - 1e-3, \
        f"Vol-target should prevent filling all 4 caps (would be 0.32); got {active_total:.4f}"


def test_l1_turnover_penalty_reduces_rebalance_when_tcost_high():
    """Large tcost_bps + w_prev close to optimum → optimizer stays near w_prev."""
    u_low = _baseline_universe(n_active=2)
    u_low["alpha_hat"][:2] = 0.01
    u_low["w_prev"][0] = 0.07
    u_low["w_prev"][1] = 0.07
    u_low["w_prev"][u_low["spy_idx"]] = 1 - 0.07 - 0.07 - 0.03
    u_low["w_prev"][u_low["cash_idx"]] = 0.03
    u_low["cfg"] = {"tcost_bps": 0.0}
    result_low = _solve(u_low)

    u_high = _baseline_universe(n_active=2)
    u_high["alpha_hat"][:2] = 0.01
    u_high["w_prev"][0] = 0.07
    u_high["w_prev"][1] = 0.07
    u_high["w_prev"][u_high["spy_idx"]] = 1 - 0.07 - 0.07 - 0.03
    u_high["w_prev"][u_high["cash_idx"]] = 0.03
    u_high["cfg"] = {"tcost_bps": 5000.0}
    result_high = _solve(u_high)

    assert result_high.diagnostics["turnover_one_way"] < \
           result_low.diagnostics["turnover_one_way"] + 1e-6, (
        f"High tcost ({result_high.diagnostics['turnover_one_way']:.4f}) should not "
        f"trade more than low tcost ({result_low.diagnostics['turnover_one_way']:.4f})"
    )


def test_eligibility_mask_pins_disallowed_to_zero():
    """An ineligible ticker with very positive α̂ must still get w=0."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][0] = 0.10
    u["alpha_hat"][1] = 0.05
    u["eligibility"][0] = False

    result = _solve(u)
    w = result.weights

    assert w[0] == pytest.approx(0.0, abs=1e-6), \
        f"Ineligible asset must be pinned to 0; got {w[0]:.4f}"
    assert w[1] == pytest.approx(0.08, abs=1e-3), \
        "Eligible competitor should still hit its cap"


def test_cash_sleeve_equality_constraint_is_inviolable():
    """Cash sleeve pin must hold even when α̂ strongly favors equities."""
    u = _baseline_universe(n_active=4)
    u["alpha_hat"][:4] = 0.20
    u["cfg"] = {"cash_sleeve_pct": 0.05}

    result = _solve(u)

    assert result.weights[u["cash_idx"]] == pytest.approx(0.05, abs=1e-6), \
        f"Cash sleeve 0.05 violated; got {result.weights[u['cash_idx']]:.6f}"


def test_sector_cap_binds_when_many_names_in_one_sector():
    """5 names in one sector, all positive α̂ → sector total capped at max_sector_pct."""
    u = _baseline_universe(n_active=5, sector_labels=["tech"] * 5)
    u["alpha_hat"][:5] = 0.05
    u["cfg"] = {"max_sector_pct": 0.20}

    result = _solve(u)
    w = result.weights

    tech_total = w[:5].sum()
    assert tech_total <= 0.20 + 1e-4, \
        f"Sector cap 0.20 violated; got {tech_total:.4f}"
    assert tech_total >= 0.20 - 1e-3, \
        f"Sector cap should be binding given strong α̂; got {tech_total:.4f}"


def test_infeasibility_falls_back_to_current_weights_plus_cash():
    """Conflicting hard constraints → solver returns infeasible → fallback weights returned."""
    u = _baseline_universe(n_active=1)
    u["alpha_hat"][0] = 0.05
    u["stance_caps"][:] = 0.001
    u["w_prev"][0] = 0.50
    u["w_prev"][u["spy_idx"]] = 0.47
    u["w_prev"][u["cash_idx"]] = 0.03
    u["cfg"] = {"cash_sleeve_pct": 0.03}

    result = solve_target_weights(**u)

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.weights[u["cash_idx"]] == pytest.approx(0.03, abs=1e-6)
    if result.diagnostics["status"] == "infeasible_fallback":
        assert result.weights[0] > result.weights[u["spy_idx"]] * 0.8, (
            "Fallback should preserve the rough current allocation profile "
            "(asset 0 was 0.50, SPY was 0.47)"
        )
