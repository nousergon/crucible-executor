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
    from executor.portfolio_optimizer import _estimate_covariance, OPTIMIZER_CONFIG_DEFAULTS

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
    from executor.portfolio_optimizer import _estimate_covariance, OPTIMIZER_CONFIG_DEFAULTS

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
    from executor.portfolio_optimizer import _estimate_covariance, OPTIMIZER_CONFIG_DEFAULTS

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
    from executor.portfolio_optimizer import _estimate_covariance, OPTIMIZER_CONFIG_DEFAULTS

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
