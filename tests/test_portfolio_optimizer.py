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
    TurnoverBudgetError,
    _apply_turnover_governor,
    _clip_and_renormalize,
    _mandatory_turnover_floor,
    _solve_with_fallback,
    _turnover_diagnostics,
    compute_conviction_budget_multiplier,
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
        # These fixtures test the MVO TARGET (w_prev=zeros → full target).
        # Disable the turnover governor (a separate concern — it caps the
        # STEP toward the target) so target assertions stay bit-identical;
        # the governor is exercised directly in TestTurnoverGovernor.
        "cfg": {"max_daily_turnover": None},
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


# ─── A.1 horizon-scaling tests ──────────────────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §A.1
#
# Σ is configurable at horizon H (default 1 = daily). The optimizer's three
# Σ touchpoints (objective, vol-target SOC, diagnostics) must all consume
# the same horizon; default H=1 preserves bit-identical legacy behavior.


def _estimate_cov_via_solve(u: dict) -> np.ndarray:
    """Helper to extract the Σ used inside solve via _estimate_covariance."""
    from executor.portfolio_optimizer import (
        OPTIMIZER_CONFIG_DEFAULTS,
        _estimate_covariance,
    )
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS, **u["cfg"]}
    return _estimate_covariance(u["returns_panel"], cfg)


def test_default_horizon_preserves_legacy_behavior():
    """Default cfg (no sigma_horizon_days) must match explicit H=1 bit-identical."""
    u_default = _baseline_universe(n_active=2)
    u_default["alpha_hat"][:2] = 0.05
    u_default["cfg"] = {"covariance_shrinkage": "sample"}

    u_h1 = _baseline_universe(n_active=2)
    u_h1["alpha_hat"][:2] = 0.05
    u_h1["returns_panel"] = u_default["returns_panel"].copy()
    u_h1["cfg"] = {"covariance_shrinkage": "sample", "sigma_horizon_days": 1}

    r_default = _solve(u_default)
    r_h1 = _solve(u_h1)

    np.testing.assert_allclose(r_default.weights, r_h1.weights, atol=1e-8)
    assert r_default.diagnostics["portfolio_vol_ann"] == pytest.approx(
        r_h1.diagnostics["portfolio_vol_ann"], abs=1e-10
    )


def test_sigma_scales_linearly_with_horizon():
    """Σ_H = H · Σ_daily — covariance matrix scales linearly in horizon-days."""
    u_h1 = _baseline_universe(n_active=3)
    u_h1["cfg"] = {"covariance_shrinkage": "sample", "sigma_horizon_days": 1}
    sigma_1 = _estimate_cov_via_solve(u_h1)

    u_h21 = _baseline_universe(n_active=3)
    u_h21["returns_panel"] = u_h1["returns_panel"].copy()
    u_h21["cfg"] = {"covariance_shrinkage": "sample", "sigma_horizon_days": 21}
    sigma_21 = _estimate_cov_via_solve(u_h21)

    np.testing.assert_allclose(sigma_21, 21.0 * sigma_1, rtol=1e-10)


def test_scaling_invariance_horizon_with_compensating_lambda():
    """Mathematical invariance: solving with (Σ_H, λ_old/H) yields same weights as (Σ_1, λ_old).

    Proves the load-bearing claim that absorbing horizon into λ is mathematically
    equivalent to scaling Σ by H and rescaling λ. This is the SOTA-rationale gate
    from the plan doc §A.1 — without this proof, the horizon switch would silently
    change optimum weights.
    """
    u_base = _baseline_universe(n_active=3)
    u_base["alpha_hat"][:3] = np.array([0.03, 0.05, 0.02])
    lambda_base = 5.0

    u_h1 = {**u_base, "cfg": {
        "covariance_shrinkage": "sample",
        "sigma_horizon_days": 1,
        "risk_aversion": lambda_base,
    }}
    u_h21 = {**u_base, "cfg": {
        "covariance_shrinkage": "sample",
        "sigma_horizon_days": 21,
        "risk_aversion": lambda_base / 21.0,  # compensating rescale
    }}

    r_h1 = _solve(u_h1)
    r_h21 = _solve(u_h21)

    np.testing.assert_allclose(r_h1.weights, r_h21.weights, atol=1e-5)


def test_vol_ann_diagnostic_horizon_invariant():
    """Same portfolio under H=1 and H=21 must produce the same annualized vol diagnostic."""
    u_h1 = _baseline_universe(n_active=2)
    u_h1["alpha_hat"][:2] = 0.05
    u_h1["cfg"] = {"covariance_shrinkage": "sample", "sigma_horizon_days": 1}
    r_h1 = _solve(u_h1)

    u_h21 = _baseline_universe(n_active=2)
    u_h21["alpha_hat"][:2] = 0.05
    u_h21["returns_panel"] = u_h1["returns_panel"].copy()
    u_h21["cfg"] = {
        "covariance_shrinkage": "sample",
        "sigma_horizon_days": 21,
        "risk_aversion": OPTIMIZER_CONFIG_DEFAULTS["risk_aversion"] / 21.0,
    }
    r_h21 = _solve(u_h21)

    assert r_h21.diagnostics["portfolio_vol_ann"] == pytest.approx(
        r_h1.diagnostics["portfolio_vol_ann"], rel=1e-5
    )


def test_vol_target_soc_horizon_aware():
    """vol_target_annual must bind to the same annualized cap regardless of Σ horizon.

    A binding 10% annual vol cap on a high-vol universe should produce the same
    portfolio annualized vol whether Σ is daily (H=1) or 21d (H=21).
    """
    N = 6
    T = 5000
    rng = np.random.default_rng(11)
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
    base_u = {
        "tickers": ["T0", "T1", "T2", "T3", "SPY", "CASH"],
        "alpha_hat": np.array([0.10, 0.10, 0.10, 0.10, 0.0, -1e-6]),
        "returns_panel": returns,
        "w_prev": np.zeros(N),
        "sectors": ["tech", "tech", "tech", "tech", "__benchmark__", "__cash__"],
        "stance_caps": np.array([0.08, 0.08, 0.08, 0.08, 1.0, 1.0]),
        "eligibility": np.ones(N, dtype=bool),
        "spy_idx": 4,
        "cash_idx": 5,
    }
    cfg_common = {"vol_target_annual": 0.10, "covariance_shrinkage": "sample"}

    r_h1 = solve_target_weights(**{**base_u, "cfg": {**cfg_common, "sigma_horizon_days": 1}})
    r_h21 = solve_target_weights(**{
        **base_u, "cfg": {**cfg_common, "sigma_horizon_days": 21, "risk_aversion": 5.0 / 21.0},
    })

    assert r_h1.diagnostics["portfolio_vol_ann"] <= 0.10 + 5e-3
    assert r_h21.diagnostics["portfolio_vol_ann"] <= 0.10 + 5e-3
    assert r_h1.diagnostics["portfolio_vol_ann"] == pytest.approx(
        r_h21.diagnostics["portfolio_vol_ann"], rel=1e-3,
    )


def test_sigma_horizon_days_below_one_raises():
    """sigma_horizon_days < 1 is a config error — raise loud per no-silent-fails."""
    u = _baseline_universe(n_active=1)
    u["alpha_hat"][0] = 0.05
    u["cfg"] = {"sigma_horizon_days": 0, "covariance_shrinkage": "sample"}

    with pytest.raises(ValueError, match="sigma_horizon_days must be ≥ 1"):
        _solve(u)


# ─── A.2 EWMA covariance tests ──────────────────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §A.2
#
# RiskMetrics 1996 EWMA with zero-mean assumption. New estimator option
# "ewma" + cfg["ewma_lambda_decay"] (default 0.94). Default estimator
# (ledoit_wolf) is unchanged; EWMA is opt-in.


def test_ewma_concentrates_on_recent_regime():
    """Two-regime synthetic panel: EWMA should be closer to recent regime's
    sample cov than to a pooled sample cov.

    Construct T=500 daily returns: first 250 rows have N=2 with vol=0.005
    and ρ=+0.8 (calm regime); last 250 rows have vol=0.03 and ρ=-0.3
    (stress regime). EWMA(λ=0.94, half-life≈11d) should weight the stress
    regime heavily because its window is much shorter than 250 days.
    """
    from executor.portfolio_optimizer import _ewma_covariance

    rng = np.random.default_rng(42)
    T_per = 250

    # Calm regime: vol 0.005, correlation +0.8
    calm_cov = np.array([[0.005**2, 0.8 * 0.005**2], [0.8 * 0.005**2, 0.005**2]])
    calm = rng.multivariate_normal([0, 0], calm_cov, size=T_per)

    # Stress regime: vol 0.03, correlation -0.3 (decorrelating in a crash)
    stress_cov = np.array([[0.03**2, -0.3 * 0.03**2], [-0.3 * 0.03**2, 0.03**2]])
    stress = rng.multivariate_normal([0, 0], stress_cov, size=T_per)

    panel = np.vstack([calm, stress])  # calm first, stress last (most recent)

    sigma_sample = np.cov(panel, rowvar=False)
    sigma_ewma = _ewma_covariance(panel, lambda_decay=0.94)

    # EWMA diagonals should be far closer to stress vol² than to the pooled
    # ~mean of the two regimes' vol². Pooled vol² ≈ (0.005² + 0.03²) / 2 ≈ 4.6e-4.
    pooled_var_avg = (0.005**2 + 0.03**2) / 2
    stress_var = 0.03**2
    ewma_var_avg = (sigma_ewma[0, 0] + sigma_ewma[1, 1]) / 2

    assert abs(ewma_var_avg - stress_var) < abs(ewma_var_avg - pooled_var_avg), (
        f"EWMA should track recent regime: ewma_var_avg={ewma_var_avg:.6f}, "
        f"stress_var={stress_var:.6f}, pooled_var={pooled_var_avg:.6f}"
    )
    # And sample-cov should be in-between (averages across both regimes)
    sample_var_avg = (sigma_sample[0, 0] + sigma_sample[1, 1]) / 2
    assert abs(sample_var_avg - pooled_var_avg) < abs(sample_var_avg - stress_var), (
        f"Sample cov should be closer to pooled than to stress; "
        f"sample_var_avg={sample_var_avg:.6f}"
    )


def test_ewma_lambda_one_degenerates_to_uniform_weighted_cov():
    """λ=1.0 → uniform weights → cov matches (R.T @ R)/T (zero-mean assumption)."""
    from executor.portfolio_optimizer import _ewma_covariance

    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(300, 3))

    sigma_ewma_lambda1 = _ewma_covariance(returns, lambda_decay=1.0)
    sigma_uniform = (returns.T @ returns) / returns.shape[0]

    np.testing.assert_allclose(sigma_ewma_lambda1, sigma_uniform, rtol=1e-10)


def test_ewma_weights_normalize_to_one():
    """The EWMA weights must sum to 1 — sanity check that finite-T normalization
    is correct so total variance scale is preserved."""
    from executor.portfolio_optimizer import _ewma_covariance

    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=(500, 4))

    sigma = _ewma_covariance(returns, lambda_decay=0.94)
    # If weights summed wrong, the diagonal magnitude would be off by O(1/T)
    # vs the true variance. Compare to uniform cov for plausibility check.
    uniform_var_avg = float(np.mean(np.diag((returns.T @ returns) / 500)))
    ewma_var_avg = float(np.mean(np.diag(sigma)))
    # Both should be O(0.01²) = 1e-4. EWMA can deviate in either direction
    # due to recent-window noise but must be in the same order of magnitude.
    assert 0.1 * uniform_var_avg < ewma_var_avg < 10 * uniform_var_avg


def test_ewma_invalid_lambda_raises():
    """λ outside [0.5, 1.0] is a config error — RiskMetrics canonical range."""
    from executor.portfolio_optimizer import _ewma_covariance

    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, size=(100, 2))

    with pytest.raises(ValueError, match="ewma_lambda_decay must be in"):
        _ewma_covariance(returns, lambda_decay=0.3)
    with pytest.raises(ValueError, match="ewma_lambda_decay must be in"):
        _ewma_covariance(returns, lambda_decay=1.1)


def test_ewma_estimator_integrates_with_solve_target_weights():
    """End-to-end: covariance_shrinkage="ewma" produces a valid optimization."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "ewma", "ewma_lambda_decay": 0.94}

    result = _solve(u)

    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.weights[u["cash_idx"]] == pytest.approx(0.03, abs=1e-6)
    # Conviction picks should hit their cap given strong α̂
    assert result.weights[0] == pytest.approx(0.08, abs=1e-3)
    assert result.weights[1] == pytest.approx(0.08, abs=1e-3)


def test_ewma_composes_with_sigma_horizon_days():
    """EWMA Σ at H=21 = 21 × EWMA Σ at H=1 — composition with A.1."""
    from executor.portfolio_optimizer import OPTIMIZER_CONFIG_DEFAULTS, _estimate_covariance

    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, size=(300, 4))

    cfg_h1 = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "ewma",
              "ewma_lambda_decay": 0.94, "sigma_horizon_days": 1}
    cfg_h21 = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "ewma",
               "ewma_lambda_decay": 0.94, "sigma_horizon_days": 21}

    sigma_h1 = _estimate_covariance(returns, cfg_h1)
    sigma_h21 = _estimate_covariance(returns, cfg_h21)

    np.testing.assert_allclose(sigma_h21, 21.0 * sigma_h1, rtol=1e-10)


def test_default_estimator_unchanged_after_ewma_addition():
    """Adding EWMA must NOT change behavior when covariance_shrinkage is unset
    or set to ledoit_wolf — no silent regression of the production path."""
    u_default = _baseline_universe(n_active=2)
    u_default["alpha_hat"][:2] = 0.05
    u_default["cfg"] = {}  # default everything

    u_explicit_lw = _baseline_universe(n_active=2)
    u_explicit_lw["alpha_hat"][:2] = 0.05
    u_explicit_lw["returns_panel"] = u_default["returns_panel"].copy()
    u_explicit_lw["cfg"] = {"covariance_shrinkage": "ledoit_wolf"}

    r_default = _solve(u_default)
    r_lw = _solve(u_explicit_lw)

    np.testing.assert_allclose(r_default.weights, r_lw.weights, atol=1e-8)


# ─── A.3 OAS estimator tests ────────────────────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §A.3
#
# Chen et al. 2010 Oracle Approximating Shrinkage. Drop-in alongside LW;
# sklearn.covariance.OAS shares the .fit().covariance_ interface.


def test_oas_estimator_produces_valid_psd_matrix():
    """OAS Σ must be symmetric PSD; same shape as input."""
    from executor.portfolio_optimizer import OPTIMIZER_CONFIG_DEFAULTS, _estimate_covariance

    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, size=(252, 5))
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "oas"}

    sigma = _estimate_covariance(returns, cfg)

    assert sigma.shape == (5, 5)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() >= -1e-10, f"OAS Σ must be PSD; min eigval={eigvals.min()}"


def test_oas_estimator_integrates_with_solve_target_weights():
    """End-to-end: covariance_shrinkage="oas" produces a valid optimization."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "oas"}

    result = _solve(u)

    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.weights[u["cash_idx"]] == pytest.approx(0.03, abs=1e-6)
    assert result.weights[0] == pytest.approx(0.08, abs=1e-3)
    assert result.weights[1] == pytest.approx(0.08, abs=1e-3)


def test_oas_composes_with_sigma_horizon_days():
    """OAS Σ at H=21 = 21 × OAS Σ at H=1 — composition with A.1."""
    from executor.portfolio_optimizer import OPTIMIZER_CONFIG_DEFAULTS, _estimate_covariance

    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, size=(252, 4))

    cfg_h1 = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "oas",
              "sigma_horizon_days": 1}
    cfg_h21 = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "oas",
               "sigma_horizon_days": 21}

    sigma_h1 = _estimate_covariance(returns, cfg_h1)
    sigma_h21 = _estimate_covariance(returns, cfg_h21)

    np.testing.assert_allclose(sigma_h21, 21.0 * sigma_h1, rtol=1e-10)


def test_oas_distinct_from_lw_on_correlated_small_sample():
    """OAS and LW should produce different Σ when shrinkage intensity differs.

    With i.i.d. zero-correlation data both estimators correctly shrink fully
    to scaled-identity. Need data with real correlation structure so the
    intensity formulas (which differ between LW and OAS) yield distinct Σ.
    Confirms OAS is actually wired (not silently aliasing to LW)."""
    from executor.portfolio_optimizer import OPTIMIZER_CONFIG_DEFAULTS, _estimate_covariance

    rng = np.random.default_rng(11)
    N = 10
    T = 40  # small T/N where shrinkage intensity matters most
    # Build a correlated panel: each return = common factor + idiosyncratic noise
    common_factor = rng.normal(0, 0.01, size=T)
    idiosyncratic = rng.normal(0, 0.005, size=(T, N))
    returns = common_factor[:, None] + idiosyncratic  # broadcast; introduces ρ≈0.8

    cfg_lw = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "ledoit_wolf"}
    cfg_oas = {**OPTIMIZER_CONFIG_DEFAULTS, "covariance_shrinkage": "oas"}

    sigma_lw = _estimate_covariance(returns, cfg_lw)
    sigma_oas = _estimate_covariance(returns, cfg_oas)

    # Distinct: different shrinkage intensities → different off-diagonal magnitudes
    assert not np.allclose(sigma_lw, sigma_oas, atol=1e-7), (
        "OAS should differ from LW on small T/N correlated data — if these "
        "match, OAS may be silently aliasing to LW"
    )


# ─── B.3 α̂-uncertainty penalty tests ────────────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.3
#
# Adds γ · sum_i(σ_ε_i² · w_i²) to the MVO objective when γ > 0 and a USABLE
# epistemic vector is provided. Garlappi-Uppal-Wang 2007 diagonal-Ω form.
# Default OFF (γ=0) preserves bit-identical legacy MVO behavior.
#
# alpha-engine-config-I9452 changed WHICH vector feeds Ω. It is
# `alpha_uncertainty_epistemic` (= predicted_alpha_std_epistemic, sqrt(xᵀΣ_w x))
# and never `alpha_uncertainty` (the total predictive std). Passing only the
# total leaves the term INOPERATIVE with a recorded reason — these tests were
# migrated to the new kwarg, and two of them assert the DELIBERATE REVERSAL of
# an earlier property: a cross-sectionally flat Ω is now reported inoperative
# instead of being applied as a uniform ridge.


def test_default_alpha_uncertainty_penalty_is_off_and_bit_identical():
    """With cfg["alpha_uncertainty_penalty"] unset (default 0.0) AND no
    alpha_uncertainty arg, behavior must match legacy MVO bit-identical —
    a silent change to the production path would be the worst-case
    failure mode of this PR."""
    u_legacy = _baseline_universe(n_active=2)
    u_legacy["alpha_hat"][:2] = 0.05
    u_legacy["cfg"] = {"covariance_shrinkage": "sample"}

    u_b3 = _baseline_universe(n_active=2)
    u_b3["alpha_hat"][:2] = 0.05
    u_b3["returns_panel"] = u_legacy["returns_panel"].copy()
    u_b3["cfg"] = {"covariance_shrinkage": "sample"}  # no alpha_uncertainty_penalty key

    r_legacy = _solve(u_legacy)
    r_b3 = _solve(u_b3)
    np.testing.assert_allclose(r_legacy.weights, r_b3.weights, atol=1e-12)
    assert r_b3.diagnostics["alpha_uncertainty_penalty_used"] is False
    # The reason is populated on EVERY solve, including the ones where nothing
    # was wrong — a field that only appears on the throttled path cannot be
    # used to tell "off by configuration" from "off because it broke".
    assert r_b3.diagnostics["alpha_uncertainty_inoperative_reason"] == "gamma_zero"


def test_alpha_uncertainty_penalty_off_when_arg_none_even_if_gamma_positive():
    """γ > 0 but no epistemic vector → penalty skipped, reason recorded.

    Covers every stored artifact predating crucible-predictor PR596 and any
    champion with no learned noise precision."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}

    result = _solve(u)
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert result.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "epistemic_field_absent"
    )
    # Same as legacy MVO — both names hit the cap
    assert result.weights[0] == pytest.approx(0.08, abs=1e-3)
    assert result.weights[1] == pytest.approx(0.08, abs=1e-3)


def test_alpha_uncertainty_penalty_off_when_all_std_zero_or_nan():
    """alpha_uncertainty provided but ALL entries are NaN (full legacy-Ridge
    fallback case) → penalty term skipped."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    all_nan = np.full(N, np.nan)
    result = solve_target_weights(**u, alpha_uncertainty_epistemic=all_nan)
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert result.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "epistemic_field_absent"
    )


def test_high_uncertainty_pick_shrinks_relative_to_confident_pick():
    """Two equal-α̂ picks: T0 has high σ (0.04), T1 has low σ (0.002).
    With γ large enough, T0 shrinks below its cap while T1 stays at cap.

    This is the load-bearing property of B.3 — confident picks size up,
    diffuse picks size down. Per Garlappi-Uppal-Wang 2007 / plan §B.3."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    unc[0] = 0.04  # diffuse
    unc[1] = 0.002  # confident

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is True
    assert result.diagnostics["alpha_uncertainty_vintage"] == "epistemic"
    assert result.weights[0] < 0.05, (
        f"High-σ pick should shrink below cap; got {result.weights[0]:.4f}"
    )
    assert result.weights[1] == pytest.approx(0.08, abs=1e-3), (
        f"Low-σ pick should stay at cap; got {result.weights[1]:.4f}"
    )
    # And the high-σ pick must shrink BELOW the low-σ pick
    assert result.weights[0] < result.weights[1]


def test_uniform_uncertainty_is_reported_inoperative_not_applied():
    """DELIBERATE REVERSAL of the original B.3 property (I9452 deliverable 3).

    B.3 originally asserted that a uniformly high σ shrinks every conviction
    pick proportionally. It does — and that is precisely the failure mode: a
    flat Ω discriminates between nothing, so it is not robust sizing, it is an
    undeclared risk-aversion increase applied through a knob nobody is reading
    as one. Measured, that was not a corner case: over every stored session
    from the 2026-06-01 BayesianRidge cutover to 2026-08-31 the emitted TOTAL
    σ had a cross-name CV of at most 0.008, so this branch was the only branch
    the production wiring could ever have taken.

    A flat Ω is now REPORTED inoperative and the solve is bit-identical to
    γ=0 — the operator sees a named reason instead of an unexplained shrink.
    """
    u_baseline = _baseline_universe(n_active=3)
    u_baseline["alpha_hat"][:3] = 0.05
    u_baseline["cfg"] = {"covariance_shrinkage": "sample"}

    u_flat = _baseline_universe(n_active=3)
    u_flat["alpha_hat"][:3] = 0.05
    u_flat["returns_panel"] = u_baseline["returns_panel"].copy()
    u_flat["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 500.0}
    u_flat["stance_caps"][u_flat["spy_idx"]] = 1.0
    u_flat["stance_caps"][u_flat["cash_idx"]] = 1.0

    r_baseline = _solve(u_baseline)
    N = len(u_flat["tickers"])
    unc = np.zeros(N)
    unc[:3] = 0.05  # identical across every discretionary name
    r_flat = solve_target_weights(**u_flat, alpha_uncertainty_epistemic=unc)

    assert r_flat.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert r_flat.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "cross_section_below_floor"
    )
    assert r_flat.diagnostics["alpha_uncertainty_epistemic_cv"] == pytest.approx(0.0)
    np.testing.assert_allclose(r_baseline.weights, r_flat.weights, atol=1e-6)


def test_large_but_flat_omega_is_caught_by_cv_not_by_magnitude():
    """A magnitude test could not have caught the real degenerate regime.

    Champion ``v3.0-meta-2026-08-21-7d3d1cce`` served 2026-08-24..28 with an
    epistemic vector of 0.207-0.230 — an order of magnitude LARGER than the
    healthy champion's 0.009-0.023 — and a cross-sectional CV of 0.00007-0.00025
    because its posterior never left its prior. Big and useless. Only the CV
    floor separates it.
    """
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][:3] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 500.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    # Real reconstructed values from predictor/optimizer_shadow/2026-08-27.json
    # against that champion's learned noise precision (alpha_ = 93.18750247).
    unc[:3] = [0.23019744, 0.23019854, 0.23021499]

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    assert max(unc) > 10 * 0.023, "fixture must be LARGE, or it tests nothing"
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert result.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "cross_section_below_floor"
    )


def test_real_healthy_session_clears_the_floor():
    """The counterpart: the healthy champion's own numbers must stay operative,
    or the floor is a permanent false positive rather than a detector.

    Reconstructed from predictor/optimizer_shadow/2026-08-31.json against
    champion ``v3.0-meta-2026-08-14-119e069b`` (alpha_ = 107.28362438).
    """
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][:3] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 500.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    unc[:3] = [0.01342714, 0.01872044, 0.01263466]

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is True
    assert result.diagnostics["alpha_uncertainty_epistemic_cv"] > 0.01


def test_the_total_could_never_have_cleared_the_floor():
    """Pins the finding that made this change necessary (I9446).

    These are the ACTUAL emitted ``predicted_alpha_std`` values for three names
    on 2026-08-31. Built into Ω they are a uniform ridge, and the CV floor says
    so — so the "fall back to the total when the epistemic field is absent"
    path would have been dead code that only ever produced an inoperative
    report. That is why there is no fallback.
    """
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][:3] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 500.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    totals = np.zeros(N)
    totals[:3] = [0.097475, 0.098344, 0.097369]

    # Passed as the TOTAL (its real role) → the term never arms.
    r_total_only = solve_target_weights(**u, alpha_uncertainty=totals)
    assert r_total_only.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert r_total_only.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "epistemic_field_absent"
    )

    # And even if something DID hand the total to Ω, the floor rejects it.
    r_forced = solve_target_weights(**u, alpha_uncertainty_epistemic=totals)
    assert r_forced.diagnostics["alpha_uncertainty_penalty_used"] is False
    assert r_forced.diagnostics["alpha_uncertainty_inoperative_reason"] == (
        "cross_section_below_floor"
    )


def test_nan_entries_treated_as_zero_uncertainty():
    """Partial-rollout case: some tickers carry the epistemic field, one does
    not (legacy Ridge → None → NaN). NaN entries get zero penalty (no info)."""
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][:3] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.full(N, np.nan)
    unc[0] = 0.04   # T0: diffuse
    unc[1] = 0.004  # T1: confident
    # T2 stays NaN → no penalty for it

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is True
    # T0 shrinks hardest, T2 (no info) stays at cap.
    assert result.weights[0] < result.weights[1]
    assert result.weights[2] == pytest.approx(0.08, abs=1e-3)


def test_negative_alpha_uncertainty_coerced_to_zero_with_warning(caplog):
    """Negative σ is an upstream contract violation but we don't crash the
    morning planner over it — log loud, coerce to 0."""
    import logging
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][:3] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    unc[0] = -0.05  # invalid
    unc[1] = 0.002
    unc[2] = 0.030

    with caplog.at_level(logging.WARNING):
        result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    assert any("negative entries" in rec.message for rec in caplog.records)
    # Solve still completes; T0 gets zero penalty (coerced), so it stays at cap
    assert result.weights[0] == pytest.approx(0.08, abs=1e-3)


def test_alpha_uncertainty_wrong_shape_raises():
    """Shape mismatch IS a load-bearing programming bug — raise loud."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 1000.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    wrong_shape = np.array([0.01, 0.02])  # N=4 expected
    with pytest.raises(ValueError, match="alpha_uncertainty_epistemic shape"):
        solve_target_weights(**u, alpha_uncertainty_epistemic=wrong_shape)


def test_uncertainty_penalty_diagnostics_populated():
    """Diagnostics surface mean_alpha_std_active + penalty_contribution
    for operator readability when penalty is active."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {"covariance_shrinkage": "sample", "alpha_uncertainty_penalty": 500.0}
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    unc[0] = 0.03
    unc[1] = 0.01

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    diag = result.diagnostics
    assert diag["alpha_uncertainty_penalty_used"] is True
    assert diag["alpha_uncertainty_vintage"] == "epistemic"
    assert diag["alpha_uncertainty_inoperative_reason"] is None
    assert diag["alpha_uncertainty_epistemic_cv"] > 0.01
    assert "mean_alpha_std_active" in diag
    assert diag["mean_alpha_std_active"] > 0
    assert "alpha_uncertainty_penalty_contribution" in diag
    assert diag["alpha_uncertainty_penalty_contribution"] > 0


def test_uncertainty_penalty_composes_with_horizon_and_ewma():
    """B.3 must compose with A.1 (sigma_horizon_days) and A.2 (EWMA) —
    the full SOTA stack should work together, not just in isolation."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][:2] = 0.05
    u["cfg"] = {
        "covariance_shrinkage": "ewma",
        "ewma_lambda_decay": 0.94,
        "sigma_horizon_days": 21,
        "risk_aversion": 5.0 / 21.0,  # compensating rescale per A.1
        "alpha_uncertainty_penalty": 1000.0,
    }
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0

    N = len(u["tickers"])
    unc = np.zeros(N)
    unc[0] = 0.04
    unc[1] = 0.002

    result = solve_target_weights(**u, alpha_uncertainty_epistemic=unc)
    # Solve succeeds AND uncertainty penalty active AND differentiates picks
    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.diagnostics["alpha_uncertainty_penalty_used"] is True
    assert result.weights[0] < result.weights[1]


# ── Covariance-injection re-solve path (intraday reconcile) ──────────────────
# The daemon's intraday re-solve reuses the morning DAILY covariance Σ cached in
# the optimizer shadow log instead of re-estimating from a returns panel. These
# tests pin that the injected path is mechanism-identical to the estimated path.

from executor.portfolio_optimizer import (  # noqa: E402
    _estimate_covariance_daily,
    _validate_covariance,
)


def _solved_pair(u: dict):
    """Solve once estimating Σ from the panel, once injecting Σ_daily from the
    SAME panel. Returns (estimated_result, injected_result)."""
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    cfg_full = {**OPTIMIZER_CONFIG_DEFAULTS, **u["cfg"]}
    sigma_daily = _estimate_covariance_daily(u["returns_panel"], cfg_full)
    est = solve_target_weights(**u)
    inj_kwargs = {**u, "returns_panel": None, "covariance": sigma_daily}
    inj = solve_target_weights(**inj_kwargs)
    return est, inj, sigma_daily


def test_covariance_injection_matches_estimated_path():
    """covariance=None vs covariance=Σ_daily (same panel) → identical output."""
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.03
    u["alpha_hat"][2] = -0.02
    est, inj, _ = _solved_pair(u)
    np.testing.assert_allclose(est.weights, inj.weights, atol=1e-9)
    assert inj.diagnostics["portfolio_vol_ann"] == pytest.approx(
        est.diagnostics["portfolio_vol_ann"], rel=1e-9,
    )
    assert inj.diagnostics["status"] == est.diagnostics["status"]


def test_covariance_injection_horizon_aware():
    """Injected Σ_daily honors sigma_horizon_days exactly like the estimator."""
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.03
    u["cfg"] = {"sigma_horizon_days": 21}
    est, inj, _ = _solved_pair(u)
    np.testing.assert_allclose(est.weights, inj.weights, atol=1e-9)
    assert inj.diagnostics["portfolio_vol_ann"] == pytest.approx(
        est.diagnostics["portfolio_vol_ann"], rel=1e-9,
    )


def test_covariance_injection_json_roundtrip():
    """Σ persisted to JSON (shadow log) and reloaded still re-solves identically."""
    import json
    u = _baseline_universe(n_active=3)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.03
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    cfg_full = {**OPTIMIZER_CONFIG_DEFAULTS, **u["cfg"]}
    sigma_daily = _estimate_covariance_daily(u["returns_panel"], cfg_full)

    in_mem = solve_target_weights(**{**u, "returns_panel": None, "covariance": sigma_daily})
    serialized = json.loads(json.dumps([[float(x) for x in row] for row in sigma_daily]))
    reloaded = np.asarray(serialized, dtype=float)
    round_tripped = solve_target_weights(**{**u, "returns_panel": None, "covariance": reloaded})
    np.testing.assert_allclose(in_mem.weights, round_tripped.weights, atol=1e-9)


def test_validate_covariance_rejects_bad_shape():
    with pytest.raises(ValueError, match="covariance shape"):
        _validate_covariance(np.eye(4), 5)


def test_validate_covariance_rejects_non_finite():
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate_covariance(bad, 3)


def test_validate_covariance_rejects_non_psd():
    # Symmetric but indefinite (negative eigenvalue).
    bad = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="not PSD"):
        _validate_covariance(bad, 3)


def test_validate_covariance_symmetrizes_tiny_asymmetry():
    sym = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
    nearly = sym.copy()
    nearly[0, 1] += 1e-12  # JSON-roundtrip-scale asymmetry
    out = _validate_covariance(nearly, 3)
    np.testing.assert_allclose(out, out.T, atol=0.0)  # exactly symmetric


def test_covariance_provided_allows_none_returns_panel():
    """returns_panel=None is valid when covariance is supplied."""
    u = _baseline_universe(n_active=2)
    u["alpha_hat"][0] = 0.04
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    cfg_full = {**OPTIMIZER_CONFIG_DEFAULTS, **u["cfg"]}
    sigma_daily = _estimate_covariance_daily(u["returns_panel"], cfg_full)
    res = solve_target_weights(**{**u, "returns_panel": None, "covariance": sigma_daily})
    assert res.diagnostics["status"] in ("optimal", "optimal_inaccurate")


def test_missing_returns_panel_without_covariance_raises():
    u = _baseline_universe(n_active=2)
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    with pytest.raises(ValueError, match="returns_panel is required"):
        solve_target_weights(**{**u, "returns_panel": None})


class TestTurnoverBudgetConstraint:
    """The daily turnover budget is a CONSTRAINT inside the convex program,
    not a post-solve uniform shrink (alpha-engine-config-I7346).

    The distinction is the whole point: a shrink scales EVERY name's delta by
    ``cap / requested``, including names whose entire delta is an intended new
    position, so a hard enough shrink pushes a whole entry cohort under the
    downstream rebalance band and deletes it unnamed. A constraint makes the
    optimizer CHOOSE which trades fit the budget, and every surviving name
    lands at the size it actually wants.
    See executor/portfolio_optimizer.py::_apply_turnover_constraint.
    """

    def _solve_from_prev(self, cap=0.20, flag=0.35):
        # Six attractive names, each capped at 0.08, against an all-SPY book.
        # The unconstrained MVO target is ~0.48 one-way away, so a 0.20 budget
        # genuinely binds. w_prev sums to 1 and sits exactly ON the cash sleeve
        # and inside every stance cap — the MANDATORY floor is therefore zero
        # and the budget is purely discretionary, which is the regime these
        # assertions are about.
        u = _baseline_universe(n_active=6)
        u["alpha_hat"][:6] = [0.08, 0.078, 0.076, 0.074, 0.072, 0.070]
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[u["spy_idx"]] = 0.97
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["cfg"] = {"max_daily_turnover": cap, "large_move_turnover_flag": flag}
        return _solve(u)

    def test_small_move_below_budget_is_untouched(self):
        # Already near target → budget never binds → no flag.
        u = _baseline_universe(n_active=1)
        u["alpha_hat"][0] = 0.08
        # Seed w_prev close to the expected target (0.08 / 0.89 SPY / 0.03 cash).
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[0] = 0.075
        w_prev[u["spy_idx"]] = 0.895
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["cfg"] = {"max_daily_turnover": 0.20, "large_move_turnover_flag": 0.35}
        result = _solve(u)
        assert result.diagnostics["turnover_constraint_applied"] is True
        assert result.diagnostics["turnover_constraint_binding"] is False
        assert result.diagnostics["turnover_capped"] is False
        assert result.diagnostics["large_move_flagged"] is False
        assert result.diagnostics["turnover_one_way"] < 0.20

    def test_solved_vector_satisfies_the_budget(self):
        # The solver, not a post-hoc shrink, keeps executed turnover at/below
        # the cap. (Universe: T0, SPY, CASH.)
        result = self._solve_from_prev(cap=0.20)
        d = result.diagnostics
        assert d["turnover_constraint_binding"] is True
        assert d["turnover_capped"] is True
        assert d["turnover_one_way"] <= 0.20 + 1e-3
        assert d["turnover_one_way"] == pytest.approx(0.20, abs=1e-2)

    def test_no_post_hoc_scale_factor_is_reported(self):
        # `turnover_scale_applied` was the shrink's signature. Nothing is
        # scaled any more, so the key must be gone rather than reported as 1.0
        # (a reported 1.0 would read as "a shrink ran and was a no-op").
        result = self._solve_from_prev(cap=0.20)
        assert "turnover_scale_applied" not in result.diagnostics

    def test_budget_shadow_price_is_published(self):
        # The dual of the turnover constraint: the marginal objective value of
        # one more unit of budget. It is what the restraint COST, and it
        # replaces the old "requested vs executed" gap as the evidence that
        # the optimizer wanted to move further.
        result = self._solve_from_prev(cap=0.20)
        sp = result.diagnostics["turnover_constraint_shadow_price"]
        assert sp is not None and sp > 0.0

    def test_binding_budget_sets_large_move_flag(self):
        # A binding budget means the optimizer wanted more than it was
        # allowed. Under the constraint construction executed turnover can no
        # longer exceed the flag on a capped day, so without this branch the
        # large-move detector would have gone permanently silent.
        result = self._solve_from_prev(cap=0.20, flag=0.35)
        assert result.diagnostics["large_move_flagged"] is True
        assert result.diagnostics["large_move_reason"] == "turnover_budget_binding"

    def test_flag_without_budget_flags_but_does_not_constrain(self):
        # flag set, budget disabled → flagged on the raw comparison, weights
        # unconstrained (flagging never substitutes for the budget). The flag
        # is set below the fixture's unconstrained move so the raw comparison
        # is what fires, with no budget present to bind.
        result = self._solve_from_prev(cap=None, flag=0.10)
        assert result.diagnostics["large_move_flagged"] is True
        assert result.diagnostics["large_move_reason"] == "executed_turnover_above_flag"
        assert result.diagnostics["turnover_constraint_applied"] is False
        assert result.diagnostics["turnover_capped"] is False

    def test_budget_disabled_is_bit_identical(self):
        # max_daily_turnover=None → no constraint, no flag, legacy behaviour.
        result = self._solve_from_prev(cap=None, flag=None)
        assert result.diagnostics["turnover_constraint_applied"] is False
        assert result.diagnostics["turnover_constraint_cap"] is None
        assert result.diagnostics["turnover_capped"] is False
        assert result.diagnostics["large_move_flagged"] is False

    def test_turnover_fields_present_on_every_solve(self):
        # A field that appears only on the interesting path is
        # indistinguishable from a dead emitter.
        for cap in (0.20, None):
            d = self._solve_from_prev(cap=cap).diagnostics
            for key in (
                "requested_turnover_one_way",
                "turnover_constraint_applied",
                "turnover_constraint_cap",
                "turnover_mandatory_floor",
                "turnover_constraint_binding",
                "turnover_constraint_shadow_price",
                "turnover_capped",
                "large_move_flagged",
                "large_move_reason",
            ):
                assert key in d, f"{key} missing with cap={cap}"

    def test_weights_stay_feasible_under_the_budget(self):
        result = self._solve_from_prev(cap=0.20)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(result.weights >= -1e-9)


class TestTurnoverBudgetDoesNotDeleteAnEntryCohort:
    """The regression the issue is about (alpha-engine-config-I7346).

    Shape: a book far from target (one oversized legacy holding to unwind)
    plus SEVERAL new small candidates. Under the old post-solve uniform
    shrink, the whole step scaled by ``cap / requested``; with requested
    turnover large the scale factor is small enough that every new position's
    delta lands under the downstream ``rebalance_band_pct`` and the entire
    entry cohort is deleted while the solve still reports ``optimal``.

    Under the constraint construction the optimizer spends the budget on the
    trades it most wants, so whatever it does enter, it enters at a TRADEABLE
    size — never a uniformly-shrunk sliver.
    """

    BAND = 0.005  # executor/optimizer_shadow.py rebalance_band_pct

    N_ENTRIES = 8

    def _far_from_target(self, cap):
        # Eight new candidates against a book parked almost entirely in SPY —
        # the live 2026-08-14 shape (SPY 0.832, one held name, eight fresh
        # entries). The unconstrained MVO wants ~0.5 one-way, so a 0.20 budget
        # binds hard, and every entry is a NEW position: exactly the cohort a
        # uniform shrink scales toward zero.
        u = _baseline_universe(n_active=self.N_ENTRIES)
        u["alpha_hat"][: self.N_ENTRIES] = np.linspace(0.08, 0.06, self.N_ENTRIES)
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[u["spy_idx"]] = 0.97
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["cfg"] = {"max_daily_turnover": cap, "min_position_pct": 0.005}
        return u

    @staticmethod
    def _uniform_shrink(weights, w_prev, cap):
        """The OLD mechanism, reproduced locally so the test states what it is
        protecting against rather than merely asserting today's numbers:

            w_exec = w_prev + (w_target − w_prev) · (cap / requested)
        """
        requested = float(np.sum(np.abs(weights - w_prev)) / 2)
        if requested <= cap:
            return weights
        return w_prev + (weights - w_prev) * (cap / requested)

    def test_new_positions_survive_the_band_at_tradeable_sizes(self):
        cap = 0.20
        u = self._far_from_target(cap)
        w_prev = u["w_prev"].copy()
        result = _solve(u)

        assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
        assert result.diagnostics["turnover_one_way"] <= cap + 1e-3
        assert result.diagnostics["turnover_constraint_binding"] is True

        entered = [
            i for i in range(self.N_ENTRIES)
            if w_prev[i] == 0.0 and result.weights[i] > 0.0
        ]
        assert entered, "the solve entered nothing — fixture no longer exercises entries"
        for i in entered:
            delta = float(result.weights[i] - w_prev[i])
            assert delta >= self.BAND, (
                f"entry T{i} sized at {delta:.5f}, under the {self.BAND} "
                "rebalance band — it would be deleted downstream, unnamed"
            )

    def test_the_old_uniform_shrink_deletes_an_entry_cohort(self):
        # The counterfactual that makes the fix load-bearing. Stated as
        # arithmetic on the shrink formula rather than as a second solve: the
        # failure is a property of scaling every delta by cap/requested, not
        # of any particular solver output, and a test that depends on the
        # solver landing on a specific vector would rot into a tautology.
        #
        # Book: 0.85 in one legacy name to be unwound, plus eight intended
        # entries of 0.02 each. One-way requested = 0.85, cap 0.20 →
        # scale 0.235 → each 0.02 entry becomes 0.0047, under the 0.005 band.
        n = self.N_ENTRIES
        size = 0.02
        w_prev = np.zeros(n + 2)
        w_prev[n] = 0.85      # legacy holding, index n
        w_prev[n + 1] = 0.12  # SPY-ish remainder
        target = np.zeros(n + 2)
        target[:n] = size
        target[n] = 0.0
        target[n + 1] = 1.0 - n * size

        requested = float(np.sum(np.abs(target - w_prev)) / 2)
        cap = 0.20
        assert requested > cap
        shrunk = self._uniform_shrink(target, w_prev, cap)

        killed = [i for i in range(n) if abs(shrunk[i] - w_prev[i]) < self.BAND]
        assert len(killed) == n, (
            f"only {len(killed)}/{n} entries fell under the band — the fixture "
            "no longer demonstrates the failure mode"
        )
        # And the same shrink applied to the LARGE leg leaves it comfortably
        # tradeable: the cohort is deleted while the rebalance survives, which
        # is why the solve still looked healthy.
        assert abs(shrunk[n] - w_prev[n]) > self.BAND

    def test_budget_is_not_starved_by_a_mandated_exit(self):
        # A held name that goes INELIGIBLE is pinned to w=0, which mandates
        # turnover whether or not there is budget for it. If the constraint's
        # RHS were the raw config cap, the program would be infeasible and the
        # whole book would fall to the hold path — a new failure mode
        # introduced by the fix, on the day a forced exit is what must happen.
        u = _baseline_universe(n_active=2)
        u["alpha_hat"][:2] = [0.05, 0.04]
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[0] = 0.70  # held, and about to be gated off
        w_prev[u["spy_idx"]] = 0.27
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["eligibility"] = np.ones(len(u["tickers"]), dtype=bool)
        u["eligibility"][0] = False
        # 0.70 must go to zero → 0.35 one-way, far above a 0.05 budget.
        u["cfg"] = {"max_daily_turnover": 0.05}
        result = _solve(u)

        d = result.diagnostics
        assert d["status"] in ("optimal", "optimal_inaccurate"), (
            "the mandated exit made the program infeasible — the turnover "
            "budget starved a non-discretionary trade"
        )
        assert result.weights[0] == pytest.approx(0.0, abs=1e-6)
        assert d["turnover_mandatory_floor"] >= 0.30
        assert d["turnover_constraint_cap"] >= d["turnover_mandatory_floor"]


class TestTurnoverBudgetAssertion:
    """``_apply_turnover_governor`` no longer shrinks — it asserts.

    It asserts the SOLVER's vector against the budget (the only quantity the
    convex constraint governed) and the post-solve clip's MEASURED
    displacement against its own mutations, separately
    (alpha-engine-config-I11368).
    """

    CAP_META = {
        "turnover_constraint_applied": True,
        "turnover_constraint_cap": 0.20,
        "turnover_mandatory_floor": 0.0,
        "turnover_constraint": None,
    }

    @staticmethod
    def _clip(one_way=0.0, negatives=0.0, box=0.0, dust=0.0):
        return {
            "clip_turnover_one_way": one_way,
            "clip_negative_mass": negatives,
            "clip_box_excess": box,
            "clip_mass_zeroed": dust,
            "clip_renorm_scale": 1.0,
        }

    def test_conforming_vector_passes_and_is_unmodified(self):
        w_prev = np.array([0.10, 0.87, 0.03])
        weights = np.array([0.15, 0.82, 0.03])  # one-way 0.05
        out, gov = _apply_turnover_governor(
            weights, w_prev, {"large_move_turnover_flag": None},
            turnover_meta=self.CAP_META, clip_meta=self._clip(),
            solver_status="optimal",
        )
        assert out is weights
        assert gov["requested_turnover_one_way"] == pytest.approx(0.05)
        assert gov["turnover_pre_clip_one_way"] == pytest.approx(0.05)

    def test_violation_raises_rather_than_shrinking(self):
        w_prev = np.array([0.10, 0.87, 0.03])
        weights = np.array([0.60, 0.37, 0.03])  # one-way 0.50, cap 0.20
        with pytest.raises(TurnoverBudgetError, match="SOLVER assertion"):
            _apply_turnover_governor(
                weights, w_prev, {}, turnover_meta=self.CAP_META,
                clip_meta=self._clip(), solver_status="optimal",
            )

    def test_the_clip_displacement_is_subtracted_not_used_as_a_tolerance(self):
        # Executed 0.20 against a 0.19 cap. The old contract asked whether
        # `clip_mass_zeroed` was large enough to excuse the 0.01 excess. The
        # new one subtracts the clip's MEASURED displacement to recover the
        # solved vector's own turnover, and tests THAT.
        w_prev = np.array([0.10, 0.87, 0.03])
        weights = np.array([0.30, 0.67, 0.03])  # one-way 0.20 exactly
        meta = {"turnover_constraint_cap": 0.19, "turnover_constraint": None}
        # Clip moved 0.015 → solved turnover 0.185, under the cap. Passes.
        _apply_turnover_governor(
            weights, w_prev, {}, turnover_meta=meta,
            clip_meta=self._clip(one_way=0.015, dust=0.02),
            solver_status="optimal",
        )
        # Clip moved 0.001 → solved turnover 0.199, over the cap. Raises,
        # even though the dust mass alone (0.02) would have excused it before.
        with pytest.raises(TurnoverBudgetError, match="SOLVER assertion"):
            _apply_turnover_governor(
                weights, w_prev, {}, turnover_meta=meta,
                clip_meta=self._clip(one_way=0.001, dust=0.02),
                solver_status="optimal",
            )

    def test_optimal_inaccurate_earns_a_wider_feasibility_tolerance(self):
        # The 2026-09-22 shape (alpha-engine-config-I11368), scaled to this
        # fixture: a solved point 9.4e-4 over the cap. Under `optimal` that is
        # a solver returning an infeasible point and must raise. Under
        # `optimal_inaccurate` the solver never claimed convergence, and
        # holding the entire book over its declared residual was the defect.
        w_prev = np.array([0.10, 0.87, 0.03])
        excess = 9.4e-4
        cap = 0.0162
        weights = np.array([0.10 + cap + excess, 0.87 - cap - excess, 0.03])
        meta = {"turnover_constraint_cap": cap, "turnover_constraint": None}
        with pytest.raises(TurnoverBudgetError, match="SOLVER assertion"):
            _apply_turnover_governor(
                weights, w_prev, {}, turnover_meta=meta,
                clip_meta=self._clip(), solver_status="optimal",
            )
        _, gov = _apply_turnover_governor(
            weights, w_prev, {}, turnover_meta=meta,
            clip_meta=self._clip(), solver_status="optimal_inaccurate",
        )
        assert gov["turnover_solver_status"] == "optimal_inaccurate"
        assert gov["turnover_solver_feasibility_tol"] > excess

    def test_an_unaccountable_clip_displacement_raises_naming_the_clip(self):
        # The clip says it moved 0.05 of one-way turnover while its own three
        # mutations total 0.001 — an accounting defect in the clip, not a
        # solver problem, and the message must say so.
        w_prev = np.array([0.10, 0.87, 0.03])
        weights = np.array([0.15, 0.82, 0.03])
        with pytest.raises(TurnoverBudgetError, match="CLIP assertion"):
            _apply_turnover_governor(
                weights, w_prev, {}, turnover_meta=self.CAP_META,
                clip_meta=self._clip(one_way=0.05, dust=0.001),
                solver_status="optimal",
            )

    def test_disabled_budget_never_raises(self):
        w_prev = np.array([0.10, 0.87, 0.03])
        weights = np.array([0.90, 0.07, 0.03])
        meta = {"turnover_constraint_applied": False, "turnover_constraint_cap": None}
        out, gov = _apply_turnover_governor(
            weights, w_prev, {}, turnover_meta=meta, clip_meta=self._clip(),
            solver_status="optimal",
        )
        assert out is weights
        assert gov["turnover_capped"] is False


class TestClipAccounting:
    """``_clip_and_renormalize`` reports all three of its mutations.

    Reporting only the dust drop is what let 9.4e-4 of negative-clamp +
    renormalize turnover be charged to the solver on 2026-09-22
    (alpha-engine-config-I11368).
    """

    CFG = {"min_position_pct": 0.01}

    def test_clamped_negatives_are_measured(self):
        weights = np.array([-0.004, 0.974, 0.03])
        caps = np.array([1.0, 1.0, 1.0])
        out, meta = _clip_and_renormalize(weights, caps, cash_idx=2, cfg=self.CFG)
        assert meta["clip_negative_mass"] == pytest.approx(0.004)
        assert out.sum() == pytest.approx(1.0)
        # The displacement it actually caused is reported, not inferred.
        assert meta["clip_turnover_one_way"] > 0.0

    def test_box_excess_is_measured(self):
        weights = np.array([0.50, 0.47, 0.03])
        caps = np.array([0.40, 1.0, 1.0])
        _, meta = _clip_and_renormalize(weights, caps, cash_idx=2, cfg=self.CFG)
        assert meta["clip_box_excess"] == pytest.approx(0.10, abs=1e-6)

    def test_dust_keeps_its_old_name_and_meaning(self):
        weights = np.array([0.005, 0.965, 0.03])
        caps = np.array([1.0, 1.0, 1.0])
        _, meta = _clip_and_renormalize(weights, caps, cash_idx=2, cfg=self.CFG)
        assert meta["clip_mass_zeroed"] == pytest.approx(0.005)

    def test_a_clean_vector_reports_a_zero_footprint(self):
        weights = np.array([0.20, 0.77, 0.03])
        caps = np.array([1.0, 1.0, 1.0])
        out, meta = _clip_and_renormalize(weights, caps, cash_idx=2, cfg=self.CFG)
        assert meta["clip_turnover_one_way"] == pytest.approx(0.0, abs=1e-12)
        assert meta["clip_negative_mass"] == 0.0
        assert meta["clip_box_excess"] == pytest.approx(0.0, abs=1e-12)
        assert meta["clip_mass_zeroed"] == 0.0
        assert out == pytest.approx(weights)

    def test_the_measured_displacement_bounds_what_the_assertion_subtracts(self):
        # End-to-end of the two pieces: whatever the clip reports moving, the
        # governor must be able to subtract it and land back on a solved
        # turnover that the clip's own mutations explain.
        weights = np.array([-0.004, 0.969, 0.005, 0.03])
        caps = np.array([1.0, 1.0, 1.0, 1.0])
        out, meta = _clip_and_renormalize(
            weights, caps, cash_idx=3, cfg={"min_position_pct": 0.01},
        )
        budget = 2.0 * (
            meta["clip_negative_mass"]
            + meta["clip_box_excess"]
            + meta["clip_mass_zeroed"]
        ) + 1e-9
        assert meta["clip_turnover_one_way"] <= budget


class TestSolverPreference:
    """``_solve_with_fallback`` prefers a CERTIFIED optimal point."""

    class _FakeProblem:
        def __init__(self, statuses, values):
            self._statuses = list(statuses)
            self._values = list(values)
            self.status = None
            self.solved = []

        def solve(self, solver=None):
            self.solved.append(solver)
            self.status = self._statuses.pop(0)
            self._var.value = self._values.pop(0)

    class _FakeVar:
        value = None

    def _run(self, statuses, values, monkeypatch):
        import cvxpy as cp
        monkeypatch.setattr(
            cp, "installed_solvers", lambda: ["CLARABEL", "SCS", "OSQP"],
        )
        prob = self._FakeProblem(statuses, values)
        var = self._FakeVar()
        prob._var = var
        return _solve_with_fallback(prob, var, {}), prob

    def test_an_optimal_first_solver_short_circuits(self, monkeypatch):
        (w, status, name), prob = self._run(
            ["optimal"], [np.array([1.0])], monkeypatch,
        )
        assert status == "optimal" and name == "CLARABEL"
        assert prob.solved == ["CLARABEL"]

    def test_inaccurate_is_held_back_while_a_certified_point_is_sought(
        self, monkeypatch,
    ):
        (w, status, name), prob = self._run(
            ["optimal_inaccurate", "optimal"],
            [np.array([1.0]), np.array([2.0])],
            monkeypatch,
        )
        assert prob.solved == ["CLARABEL", "SCS"]
        assert status == "optimal" and name == "SCS"
        assert w == pytest.approx(np.array([2.0]))

    def test_inaccurate_is_accepted_when_nothing_converges(self, monkeypatch):
        (w, status, name), prob = self._run(
            ["optimal_inaccurate", "infeasible", "infeasible"],
            [np.array([1.0]), np.array([9.0]), np.array([9.0])],
            monkeypatch,
        )
        assert prob.solved == ["CLARABEL", "SCS", "OSQP"]
        assert status == "optimal_inaccurate" and name == "CLARABEL"
        # The snapshot survives the later solves that overwrote w.value.
        assert w == pytest.approx(np.array([1.0]))

    def test_no_solution_at_all_returns_a_three_tuple(self, monkeypatch):
        (w, status, name), _ = self._run(
            ["infeasible", "infeasible", "infeasible"],
            [None, None, None],
            monkeypatch,
        )
        assert w is None and name is None and status == "infeasible"


class TestMandatoryTurnoverFloor:
    def test_zero_when_w_prev_already_satisfies_the_box(self):
        w_prev = np.array([0.10, 0.87, 0.03])
        caps = np.array([0.50, 1.0, 1.0])
        floor = _mandatory_turnover_floor(
            w_prev, caps, cash_idx=2, cfg={"cash_sleeve_pct": 0.03},
        )
        assert floor == pytest.approx(0.0, abs=1e-12)

    def test_counts_a_forced_exit(self):
        # A held 0.40 pinned to zero forces 0.40 of L1 out and 0.40 back in
        # elsewhere → 0.40 one-way.
        w_prev = np.array([0.40, 0.57, 0.03])
        caps = np.array([0.0, 1.0, 1.0])  # name 0 gated off
        floor = _mandatory_turnover_floor(
            w_prev, caps, cash_idx=2, cfg={"cash_sleeve_pct": 0.03},
        )
        assert floor == pytest.approx(0.40, abs=1e-9)

    def test_counts_the_cash_sleeve_pin(self):
        # Cash at 0.00 must reach the 0.03 sleeve: 0.03 in, 0.03 out.
        w_prev = np.array([0.10, 0.90, 0.00])
        caps = np.array([0.50, 1.0, 1.0])
        floor = _mandatory_turnover_floor(
            w_prev, caps, cash_idx=2, cfg={"cash_sleeve_pct": 0.03},
        )
        assert floor == pytest.approx(0.03, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# Participation-aware √-impact transaction cost + max-%-ADV constraint
# (tradeability arc, config#1401 — consumes nousergon_lib #144 TransactionCostModel)
# ═══════════════════════════════════════════════════════════════════════════════


def _feasible_universe(n_active=2, daily_vol=0.01):
    """Baseline universe with SPY/CASH stance caps → 1.0 (as ``_solve`` does),
    so the budget constraint is feasible when calling solve_target_weights
    directly (the kwargs-splat path used by the ADV tests below)."""
    u = _baseline_universe(n_active=n_active, daily_vol=daily_vol)
    u["stance_caps"][u["spy_idx"]] = 1.0
    u["stance_caps"][u["cash_idx"]] = 1.0
    return u


def _adv_universe(n_active=2, daily_vol=0.01):
    """Feasible baseline + an ADV$ vector (SPY/CASH NaN) for the impact term."""
    u = _feasible_universe(n_active=n_active, daily_vol=daily_vol)
    N = len(u["tickers"])
    adv = np.full(N, np.nan)
    for i in range(n_active):
        adv[i] = 50_000_000.0  # liquid default; per-test overrides
    return u, adv


def test_sqrt_impact_term_produces_valid_weight_vector():
    """With adv_usd + portfolio_notional, the participation-aware term is used
    and the solve still returns a valid (sums-to-1, sleeve-pinned, capped)
    weight vector."""
    u, adv = _adv_universe(n_active=2)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.04
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=1_000_000.0,
    )
    w = result.weights
    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.diagnostics["tcost_term_mode"] == "sqrt_impact"
    assert result.diagnostics["tcost_n_names_with_adv"] == 2
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w[u["cash_idx"]] == pytest.approx(0.03, abs=1e-6)
    assert np.all(w >= -1e-8)
    assert np.all(w <= u["stance_caps"] + 1e-6)


def test_sqrt_impact_penalizes_illiquid_name_more_than_liquid():
    """Two names with IDENTICAL α̂ but different ADV: the illiquid name (thin
    ADV → higher √-impact cost to trade INTO from w_prev=0) is sized smaller."""
    u, adv = _adv_universe(n_active=2)
    u["alpha_hat"][0] = 0.05  # liquid
    u["alpha_hat"][1] = 0.05  # illiquid, same alpha
    adv[0] = 500_000_000.0    # very liquid
    adv[1] = 2_000_000.0      # thin — costly to trade into
    # Impact coefficient large enough that the illiquid name's marginal impact
    # cost overtakes its α̂ BEFORE the 0.08 stance cap binds, so the two names
    # separate (a modest coef leaves both pinned at the cap).
    u["cfg"] = {**u["cfg"], "transaction_cost": {"impact_coef_bps": 3000.0},
                "max_pct_adv": None}
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=5_000_000.0,
    )
    w = result.weights
    assert result.diagnostics["tcost_term_mode"] == "sqrt_impact"
    assert w[0] > w[1] + 1e-4, (
        f"Liquid name should size larger than the equally-attractive illiquid "
        f"name; got w_liquid={w[0]:.4f} w_illiquid={w[1]:.4f}"
    )


def test_max_pct_adv_constraint_binds_and_caps_participation():
    """The max-%-ADV constraint bounds |Δw|·NAV ≤ max_pct_adv·ADV per name.
    With a thin ADV the target trade is clipped to the participation bound."""
    u, adv = _adv_universe(n_active=1)
    u["alpha_hat"][0] = 0.20  # strongly wants the full 0.08 cap
    adv[0] = 1_000_000.0      # thin
    nav = 10_000_000.0
    max_pct_adv = 0.05
    # Bound in weight space: 0.05 * 1e6 / 1e7 = 0.005 → the name can move at most
    # 0.5% of NAV from w_prev=0 in a single solve.
    u["cfg"] = {**u["cfg"], "max_pct_adv": max_pct_adv,
                "transaction_cost": {"impact_coef_bps": 0.0}}  # isolate constraint
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=nav,
    )
    w = result.weights
    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.diagnostics["max_pct_adv_applied"] is True
    bound_weight = max_pct_adv * adv[0] / nav  # 0.005
    assert w[0] <= bound_weight + 1e-5, (
        f"Trade must respect the {max_pct_adv:.0%}-of-ADV participation cap "
        f"(bound {bound_weight:.4f}); got w={w[0]:.4f}"
    )
    assert w[0] == pytest.approx(bound_weight, abs=5e-4)


def test_max_pct_adv_constraint_off_when_disabled():
    """max_pct_adv=None → no participation constraint; the name reaches its cap."""
    u, adv = _adv_universe(n_active=1)
    u["alpha_hat"][0] = 0.20
    adv[0] = 1_000_000.0
    u["cfg"] = {**u["cfg"], "max_pct_adv": None,
                "transaction_cost": {"impact_coef_bps": 0.0}}
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=10_000_000.0,
    )
    assert result.diagnostics["max_pct_adv_applied"] is False
    assert result.weights[0] == pytest.approx(0.08, abs=1e-3)


def test_tcost_failsoft_to_flat_l1_when_adv_absent():
    """No adv_usd → the cost term degrades to the flat L1 penalty and the
    participation constraint is skipped (bit-identical pre-tradeability)."""
    u = _feasible_universe(n_active=2)
    u["alpha_hat"][0] = 0.05
    # No adv_usd, no portfolio_notional passed.
    result = solve_target_weights(**u)
    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.diagnostics["tcost_term_mode"] == "flat_l1"
    assert result.diagnostics["tcost_fallback_reason"] == "no_portfolio_notional"
    assert result.diagnostics["max_pct_adv_applied"] is False
    assert result.diagnostics["max_pct_adv_reason"] == "no_portfolio_notional"


def test_tcost_failsoft_when_notional_present_but_no_adv_coverage():
    """portfolio_notional given but every name's ADV is NaN → still flat L1."""
    u = _feasible_universe(n_active=2)
    u["alpha_hat"][0] = 0.05
    N = len(u["tickers"])
    result = solve_target_weights(
        **u, adv_usd=np.full(N, np.nan), portfolio_notional=1_000_000.0,
    )
    assert result.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    assert result.diagnostics["tcost_term_mode"] == "flat_l1"
    assert result.diagnostics["tcost_fallback_reason"] == "no_adv_coverage"
    assert result.diagnostics["max_pct_adv_reason"] == "no_adv_coverage"


def test_tcost_flat_l1_matches_legacy_bit_identical():
    """sqrt_impact fail-soft to flat_l1 reproduces the LEGACY flat-penalty solve
    exactly (same weights as the pre-1401 path with tcost_mode absent)."""
    u1 = _feasible_universe(n_active=3)
    u1["alpha_hat"][:3] = [0.05, 0.03, 0.01]
    r_legacy = solve_target_weights(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                                       for k, v in u1.items()})
    assert r_legacy.diagnostics["status"] in ("optimal", "optimal_inaccurate")
    u2 = _feasible_universe(n_active=3)
    u2["alpha_hat"][:3] = [0.05, 0.03, 0.01]
    r_failsoft = solve_target_weights(
        **u2, adv_usd=None, portfolio_notional=None,
    )
    np.testing.assert_allclose(r_legacy.weights, r_failsoft.weights, atol=1e-9)


def test_sigma_scaling_engages_with_name_sigma():
    """Passing name_sigma engages the Almgren-Chriss σ-scaling (diagnostic flag),
    and the solve remains valid."""
    u, adv = _adv_universe(n_active=2)
    u["alpha_hat"][0] = 0.05
    u["alpha_hat"][1] = 0.05
    N = len(u["tickers"])
    name_sigma = np.full(N, np.nan)
    name_sigma[0] = 0.01
    name_sigma[1] = 0.04  # 4× more volatile → higher impact cost
    u["cfg"] = {**u["cfg"], "transaction_cost": {"impact_coef_bps": 400.0},
                "max_pct_adv": None}
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=5_000_000.0, name_sigma=name_sigma,
    )
    assert result.diagnostics["tcost_sigma_scaled"] is True
    # The higher-σ name (index 1) costs more to trade into → sized smaller.
    assert result.weights[0] >= result.weights[1] - 1e-6


def test_adv_partial_coverage_mixes_impact_and_floor():
    """When only SOME names have ADV, the covered ones use √-impact and the
    uncovered ones fall to the half-spread+commission floor; still valid."""
    u, adv = _adv_universe(n_active=3)
    u["alpha_hat"][:3] = [0.05, 0.05, 0.05]
    adv[0] = 50_000_000.0
    adv[1] = np.nan          # uncovered → floor
    adv[2] = 3_000_000.0     # thin → penalized
    u["cfg"] = {**u["cfg"], "max_pct_adv": None,
                "transaction_cost": {"impact_coef_bps": 300.0}}
    result = solve_target_weights(
        **u, adv_usd=adv, portfolio_notional=5_000_000.0,
    )
    assert result.diagnostics["tcost_term_mode"] == "sqrt_impact"
    assert result.diagnostics["tcost_n_names_with_adv"] == 2  # names 0 and 2
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestConvictionBudgetGate:
    """alpha-engine-config-I9315 — the discretionary turnover budget is gated
    on measured signal quality, not only on a fixed cap.

    Measured driver (20 stored optimizer_shadow artifacts, 2026-07-31 ..
    2026-08-31): from 2026-08-17 the optimizer sat at the 20%/day budget on 12
    consecutive sessions. Replaying each day's stored solve with the budget
    removed showed the UNCONSTRAINED optimum itself moving 11%-59% one-way per
    day, so the walk could never converge. One-at-a-time input attribution put
    0.110 of the mean 0.148 daily target move on alpha_hat and exactly 0.000 on
    the covariance and on eligibility. Over the same window the cross-sectional
    alpha spread was 0.025x-0.221x the predictor's own median per-name
    sigma_alpha: the optimizer was ranking names it could not tell apart.
    """

    @staticmethod
    def _gate(alpha, sigma, cfg=None, n=None):
        n = n if n is not None else len(alpha)
        return compute_conviction_budget_multiplier(
            np.array(alpha, dtype=float),
            None if sigma is None else np.array(sigma, dtype=float),
            np.ones(n, dtype=bool),
            spy_idx=n - 2,
            cash_idx=n - 1,
            cfg={**OPTIMIZER_CONFIG_DEFAULTS, **(cfg or {})},
        )

    def test_high_conviction_is_not_throttled(self):
        # Healthy regime: spread well above the model's own error bar.
        out = self._gate([0.10, -0.10, 0.05, -0.05, 0.0, 0.0], [0.02] * 6)
        assert out["conviction_ir_xs"] > 0.75
        assert out["conviction_budget_multiplier"] == 1.0
        assert out["conviction_gate_applied"] is False
        assert out["conviction_gate_reason"] == "signal_quality_ok"

    def test_tied_names_are_throttled_to_the_floor(self):
        # The live 2026-08-24..28 regime: spread ~1/30th of sigma_alpha.
        out = self._gate([0.001, 0.002, 0.0015, 0.0012, 0.0, 0.0], [0.24] * 6)
        assert out["conviction_ir_xs"] < 0.05
        assert out["conviction_budget_multiplier"] == pytest.approx(0.05)
        assert out["conviction_gate_applied"] is True
        assert out["conviction_gate_reason"] == "alpha_spread_below_own_noise"

    def test_multiplier_is_monotone_in_signal_quality(self):
        qs = [
            self._gate([s, -s, s / 2, -s / 2, 0.0, 0.0], [0.10] * 6)[
                "conviction_budget_multiplier"
            ]
            for s in (0.01, 0.05, 0.10, 0.30)
        ]
        assert qs == sorted(qs)
        assert qs[0] < 1.0 and qs[-1] == 1.0

    def test_missing_uncertainty_never_tightens_the_budget(self):
        # Missing data must never produce a budget TIGHTER than configured: a
        # gate that halts the book on an input outage is a worse failure than
        # the churn it exists to stop. The reason is recorded, not swallowed.
        for sigma, reason in (
            (None, "no_alpha_uncertainty_vector"),
            ([0.0] * 6, "no_usable_alpha_uncertainty"),
            ([float("nan")] * 6, "no_usable_alpha_uncertainty"),
        ):
            out = self._gate([0.001, 0.002, 0.0015, 0.0012, 0.0, 0.0], sigma)
            assert out["conviction_budget_multiplier"] == 1.0
            assert out["conviction_gate_applied"] is False
            assert out["conviction_gate_reason"] == reason

    def test_disabled_gate_is_inert_and_says_so(self):
        out = self._gate(
            [0.001, 0.002, 0.0015, 0.0012, 0.0, 0.0], [0.24] * 6,
            cfg={"conviction_budget_gate_enabled": False},
        )
        assert out["conviction_budget_multiplier"] == 1.0
        assert out["conviction_gate_reason"] == "disabled"

    def test_invalid_band_does_not_become_an_arbitrary_throttle(self):
        out = self._gate(
            [0.001, 0.002, 0.0015, 0.0012, 0.0, 0.0], [0.24] * 6,
            cfg={"conviction_ir_floor": 0.9, "conviction_ir_full": 0.4},
        )
        assert out["conviction_budget_multiplier"] == 1.0
        assert out["conviction_gate_reason"] == "invalid_ir_band"

    def test_too_few_discretionary_names_is_inert(self):
        out = compute_conviction_budget_multiplier(
            np.array([0.01, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]),
            np.ones(3, dtype=bool), spy_idx=1, cash_idx=2,
            cfg=OPTIMIZER_CONFIG_DEFAULTS,
        )
        assert out["conviction_budget_multiplier"] == 1.0
        assert out["conviction_gate_reason"] == "too_few_discretionary_names"

    # ── end-to-end through the solve ─────────────────────────────────────

    @staticmethod
    def _solve_tied(gate=True, cap=0.20):
        # Six statistically tied names against an all-SPY book: alphas differ
        # by ~1/50th of their own sigma. Ungated, the optimizer spends the
        # whole 20% budget reshuffling them.
        u = _baseline_universe(n_active=6)
        u["alpha_hat"][:6] = [0.0100, 0.0102, 0.0104, 0.0106, 0.0108, 0.0110]
        u["alpha_uncertainty"] = np.full(len(u["tickers"]), 0.25)
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[u["spy_idx"]] = 0.97
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["cfg"] = {
            "max_daily_turnover": cap,
            "large_move_turnover_flag": 0.35,
            "conviction_budget_gate_enabled": gate,
        }
        return _solve(u)

    def test_gate_collapses_turnover_on_tied_names(self):
        ungated = self._solve_tied(gate=False).diagnostics
        gated = self._solve_tied(gate=True).diagnostics
        assert ungated["turnover_one_way"] == pytest.approx(0.20, abs=1e-2)
        assert gated["turnover_one_way"] < 0.05
        assert gated["conviction_gate_applied"] is True
        assert gated["turnover_budget_configured"] == pytest.approx(0.20)
        assert gated["turnover_budget_discretionary"] == pytest.approx(0.01)

    def test_gated_binding_does_not_raise_the_large_move_flag(self):
        # The alert this arc exists to end: a guard doing its job is not an
        # incident. The budget still binds (on the throttled budget) and the
        # reason SAYS which, but no operator alert is raised.
        gated = self._solve_tied(gate=True).diagnostics
        assert gated["turnover_constraint_binding"] is True
        assert gated["large_move_reason"] == "conviction_throttled_budget_binding"
        assert gated["large_move_flagged"] is False

    def test_high_conviction_still_flags_a_binding_budget(self):
        # The detector must not be killed by the fix: on a day the model
        # stands behind, a binding budget is still a real operator fact.
        u = _baseline_universe(n_active=6)
        u["alpha_hat"][:6] = [0.08, 0.078, 0.076, 0.074, 0.072, 0.070]
        # sigma_alpha well under the cross-sectional spread (0.0034) -> IR ~1.7,
        # above the 0.75 full-budget mark: the model stands behind the ranking.
        u["alpha_uncertainty"] = np.full(len(u["tickers"]), 0.002)
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[u["spy_idx"]] = 0.97
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["cfg"] = {"max_daily_turnover": 0.20, "large_move_turnover_flag": 0.35}
        d = _solve(u).diagnostics
        assert d["conviction_gate_applied"] is False
        assert d["turnover_constraint_binding"] is True
        assert d["large_move_flagged"] is True
        assert d["large_move_reason"] == "turnover_budget_binding"

    def test_conviction_fields_present_on_every_solve(self):
        for cap in (0.20, None):
            for gate in (True, False):
                d = self._solve_tied(gate=gate, cap=cap).diagnostics
                for key in (
                    "conviction_gate_applied",
                    "conviction_ir_xs",
                    "conviction_alpha_dispersion",
                    "conviction_alpha_noise",
                    "conviction_n_names",
                    "conviction_budget_multiplier",
                    "conviction_gate_reason",
                    "turnover_budget_configured",
                    "turnover_budget_discretionary",
                ):
                    assert key in d, f"{key} missing cap={cap} gate={gate}"

    def test_mandatory_turnover_is_never_starved_by_the_gate(self):
        # A forced exit is not discretionary. With the gate at its floor
        # (0.01 discretionary budget), an ineligible position carrying ~10%
        # of the book must still be fully exited.
        u = _baseline_universe(n_active=6)
        u["alpha_hat"][:6] = [0.0100, 0.0102, 0.0104, 0.0106, 0.0108, 0.0110]
        u["alpha_uncertainty"] = np.full(len(u["tickers"]), 0.25)
        w_prev = np.zeros(len(u["tickers"]))
        w_prev[0] = 0.10
        w_prev[u["spy_idx"]] = 0.87
        w_prev[u["cash_idx"]] = 0.03
        u["w_prev"] = w_prev
        u["eligibility"] = np.ones(len(u["tickers"]), dtype=bool)
        u["eligibility"][0] = False
        u["cfg"] = {"max_daily_turnover": 0.20, "large_move_turnover_flag": 0.35}
        result = _solve(u)
        d = result.diagnostics
        assert d["conviction_gate_applied"] is True
        assert d["turnover_budget_discretionary"] == pytest.approx(0.01)
        # The forced exit sets the floor, and the effective budget is raised
        # to it rather than clipped to the throttled discretionary number.
        assert d["turnover_mandatory_floor"] >= 0.09
        assert d["turnover_constraint_cap"] >= d["turnover_mandatory_floor"]
        assert result.weights[0] == pytest.approx(0.0, abs=1e-6)


class TestBudgetBindingComplementarySlackness:
    """alpha-engine-config-I9315 — a binding verdict needs BOTH halves.

    Measured on the live book: 2026-08-21 dual 4.2e-9 at 14.98% executed,
    2026-08-27 dual 2.4e-8 at 16.99%, 2026-08-28 dual 1.6e-8 at 12.27%, all
    against a 20% cap with up to 7.7 points unspent. All three were reported
    binding and all three published an operator alert stating the budget
    "BOUND the solve". The old test was `dual > 1e-9`, which is far below the
    numerical dust an interior-point solver returns for an INACTIVE constraint.
    """

    @staticmethod
    def _diag(executed, cap, dual):
        w_prev = np.zeros(4)
        w_prev[0] = 1.0
        weights = w_prev.copy()
        weights[0] -= executed
        weights[1] += executed

        class _C:
            dual_value = dual

        return _turnover_diagnostics(
            weights, w_prev,
            {"turnover_constraint_applied": True,
             "turnover_constraint_cap": cap,
             "turnover_constraint": _C()},
        )

    def test_dual_dust_below_the_bound_is_not_binding(self):
        for executed, dual in ((0.1498, 4.2e-9), (0.1699, 2.4e-8), (0.1227, 1.6e-8)):
            d = self._diag(executed, 0.20, dual)
            assert d["turnover_constraint_binding"] is False, executed
            assert d["turnover_capped"] is False

    def test_post_clip_drift_at_the_bound_is_still_binding(self):
        # The 2026-08-14 case the dual-only test was introduced to catch:
        # solver at the 0.2000 cap, post-clip-and-renormalize 0.19978.
        d = self._diag(0.19978, 0.20, 0.00926)
        assert d["turnover_constraint_binding"] is True
        assert d["turnover_capped"] is True

    def test_real_dual_at_the_bound_is_binding(self):
        d = self._diag(0.20, 0.20, 0.0256)
        assert d["turnover_constraint_binding"] is True

    def test_no_dual_falls_back_to_the_primal_and_records_it(self):
        d = self._diag(0.20, 0.20, None)
        assert d["turnover_constraint_binding"] is True
        assert d["turnover_binding_test"] == "primal_only_no_dual"

    def test_binding_test_is_named_in_the_diagnostics(self):
        d = self._diag(0.20, 0.20, 0.0256)
        assert d["turnover_binding_test"] == "complementary_slackness"
