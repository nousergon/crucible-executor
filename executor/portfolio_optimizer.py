"""
Constrained mean-variance portfolio optimizer — PR 1 of the portfolio-optimizer
arc (plan: `alpha-engine-docs/private/portfolio-optimizer-260511.md`).

The institutional benchmark-as-null pattern: SPY (the benchmark) is the
no-conviction fill, cash is a pinned operational sleeve, conviction picks
express deviation from SPY within sector + position + vol-target constraints.

Math:
    maximize   wᵀα̂  −  λ · wᵀΣw  −  τ · ‖w − w_prev‖₁
    s.t.       Σwᵢ = 1                                    (budget)
               w[CASH] = cash_sleeve                       (sleeve pin)
               0 ≤ wᵢ ≤ stance_capᵢ                       (per-name cap)
               Σ_{i∈sector S} wᵢ ≤ max_sector_pct          (sector cap)
               wᵢ = 0 for i with eligibility=False         (gate mask)
               wᵀΣw ≤ σ²_target_daily                      (vol-target SOC)

This module is a pure function over numpy inputs. It does no I/O, no logging
config side effects, no S3 calls — easy to unit-test (PR 1) and easy to wire
into shadow mode (PR 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

_CLARABEL = "CLARABEL"
_FALLBACK_SOLVERS = ("SCS", "OSQP")


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    diagnostics: dict


def solve_target_weights(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
    cfg: dict,
) -> OptimizerResult:
    """
    Solve the constrained MVO and return target weights + diagnostics.

    Args:
        tickers: length-N universe. Must contain SPY (benchmark fill) and a
            CASH sentinel ticker.
        alpha_hat: shape (N,) predicted alpha vector. Convention: SPY entry is
            0.0 (benchmark = null hypothesis), CASH entry is a small negative
            number so the optimizer prefers SPY over cash when ε-indifferent.
        returns_panel: shape (T, N) daily returns history for covariance
            estimation. Rows with NaN are dropped pre-shrinkage. Caller is
            responsible for ensuring CASH column is ~0 (no return) and SPY
            column has real history.
        w_prev: shape (N,) current portfolio weights (positions / NAV). Used
            for the L1 turnover penalty.
        sectors: length-N sector labels. Use a stable string like "tech",
            "healthcare". SPY and CASH should have unique sentinel sectors
            (e.g., "__benchmark__", "__cash__") so they're not summed into
            real sector caps.
        stance_caps: shape (N,) per-name upper bound on weight. The caller
            composes this from base max_pos × stance multiplier × drawdown
            tier × earnings × coverage. For SPY use a high cap (e.g., 1.0);
            for CASH the cap is overridden by the equality pin.
        eligibility: shape (N,) bool. Names with eligibility=False are pinned
            to w_i = 0. SPY and CASH must be eligibility=True.
        spy_idx, cash_idx: positions in tickers list.
        cfg: dict with optimizer parameters. See OPTIMIZER_CONFIG_DEFAULTS.

    Returns:
        OptimizerResult with weights (length N, sums to 1, sleeve pinned) and
        diagnostics dict including solver status, portfolio vol, active share
        vs SPY, and n_active.

    On infeasibility, returns the fallback weights (current weights with cash
    absorbing the residual) and diagnostics["status"] = "infeasible_fallback".
    """
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS, **cfg}
    _validate_inputs(
        tickers, alpha_hat, returns_panel, w_prev,
        sectors, stance_caps, eligibility, spy_idx, cash_idx,
    )

    N = len(tickers)
    sigma = _estimate_covariance(returns_panel, cfg)

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for portfolio_optimizer. Install via "
            "`pip install 'cvxpy>=1.4,<1.8'`. See requirements.txt."
        ) from e

    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(N)

    objective = cp.Maximize(
        alpha_hat @ w
        - cfg["risk_aversion"] * cp.quad_form(w, sigma_psd)
        - (cfg["tcost_bps"] / 1e4) * cp.norm(w - w_prev, 1)
    )

    eligibility_idx = np.where(~eligibility)[0]
    effective_caps = np.where(eligibility, stance_caps, 0.0)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        w <= effective_caps,
        w[cash_idx] == cfg["cash_sleeve_pct"],
    ]
    if cfg.get("vol_target_annual") is not None:
        sigma_target_daily = cfg["vol_target_annual"] / np.sqrt(252)
        constraints.append(cp.quad_form(w, sigma_psd) <= sigma_target_daily ** 2)
    if eligibility_idx.size > 0:
        constraints.append(w[eligibility_idx] == 0)

    for sector_label in _real_sectors(sectors):
        idx = [i for i, s in enumerate(sectors) if s == sector_label]
        constraints.append(cp.sum(w[idx]) <= cfg["max_sector_pct"])

    problem = cp.Problem(objective, constraints)
    weights, status = _solve_with_fallback(problem, w, cfg)

    if weights is None:
        weights = _fallback_weights(w_prev, cash_idx, cfg["cash_sleeve_pct"])
        diagnostics = _build_diagnostics(
            weights, w_prev, sigma, alpha_hat, spy_idx, "infeasible_fallback", cfg,
        )
        return OptimizerResult(weights=weights, diagnostics=diagnostics)

    weights = _clip_and_renormalize(weights, effective_caps, cash_idx, cfg)
    diagnostics = _build_diagnostics(
        weights, w_prev, sigma, alpha_hat, spy_idx, status, cfg,
    )
    return OptimizerResult(weights=weights, diagnostics=diagnostics)


_VOL_TARGET_COMMENT = """
vol_target_annual default is None (no SOC constraint). For a long-only
benchmark-aware portfolio that uses SPY as the no-conviction fill, the
portfolio's natural volatility is bounded below by SPY's vol (≈16% annual),
since SPY absorbs ~89% of the book on conviction-light days. Setting
vol_target_annual below SPY vol is structurally infeasible without bonds.
Set explicitly (e.g., 0.25) to enable a stress-regime cap that only binds
during high-vol periods. Reserved for v2 multi-asset / risk-parity layer.
""".strip()


OPTIMIZER_CONFIG_DEFAULTS: dict = {
    "vol_target_annual": None,
    "risk_aversion": 5.0,
    "tcost_bps": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "covariance_shrinkage": "ledoit_wolf",
    "min_position_pct": 0.005,
}


def _validate_inputs(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
) -> None:
    N = len(tickers)
    if N == 0:
        raise ValueError("Empty universe — cannot optimize")
    for name, arr in (
        ("alpha_hat", alpha_hat),
        ("w_prev", w_prev),
        ("stance_caps", stance_caps),
        ("eligibility", eligibility),
    ):
        if arr.shape != (N,):
            raise ValueError(f"{name} shape {arr.shape} != ({N},)")
    if returns_panel.ndim != 2 or returns_panel.shape[1] != N:
        raise ValueError(
            f"returns_panel shape {returns_panel.shape} incompatible with N={N}"
        )
    if len(sectors) != N:
        raise ValueError(f"sectors length {len(sectors)} != N={N}")
    if not (0 <= spy_idx < N) or not (0 <= cash_idx < N):
        raise ValueError(f"spy_idx={spy_idx} cash_idx={cash_idx} out of range [0,{N})")
    if not eligibility[spy_idx]:
        raise ValueError("SPY must be eligible (benchmark fill)")
    if not eligibility[cash_idx]:
        raise ValueError("CASH must be eligible (sleeve pin)")


def _estimate_covariance(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    clean = returns_panel[~np.isnan(returns_panel).any(axis=1)]
    if clean.shape[0] < 20:
        raise ValueError(
            f"Need ≥20 clean return rows for covariance; got {clean.shape[0]}"
        )
    if cfg["covariance_shrinkage"] == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for Ledoit-Wolf shrinkage. Install "
                "via `pip install 'scikit-learn>=1.3,<1.6'`."
            ) from e
        return LedoitWolf().fit(clean).covariance_
    if cfg["covariance_shrinkage"] == "sample":
        return np.cov(clean, rowvar=False)
    raise ValueError(f"Unknown covariance_shrinkage: {cfg['covariance_shrinkage']}")


def _real_sectors(sectors: list[str]) -> set[str]:
    return {s for s in sectors if not (s.startswith("__") and s.endswith("__"))}


def _solve_with_fallback(problem, w, cfg: dict):
    import cvxpy as cp
    for solver in (_CLARABEL, *_FALLBACK_SOLVERS):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except (cp.error.SolverError, ValueError) as e:
            logger.warning(f"Solver {solver} raised {e!r}, trying next")
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(w.value, dtype=float), problem.status
        logger.warning(
            f"Solver {solver} returned status={problem.status}, trying next"
        )
    return None, problem.status if problem.status else "no_solver_available"


def _fallback_weights(
    w_prev: np.ndarray, cash_idx: int, cash_sleeve_pct: float,
) -> np.ndarray:
    weights = np.maximum(w_prev.copy(), 0.0)
    weights[cash_idx] = 0.0
    equity_sum = weights.sum()
    target_equity = 1.0 - cash_sleeve_pct
    if equity_sum > 0:
        weights *= target_equity / equity_sum
    weights[cash_idx] = cash_sleeve_pct
    return weights


def _clip_and_renormalize(
    weights: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    weights = np.minimum(weights, effective_caps + 1e-8)
    small = (weights < cfg["min_position_pct"]) & (np.arange(len(weights)) != cash_idx)
    weights = np.where(small, 0.0, weights)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def _build_diagnostics(
    weights: np.ndarray,
    w_prev: np.ndarray,
    sigma: np.ndarray,
    alpha_hat: np.ndarray,
    spy_idx: int,
    status: str,
    cfg: dict,
) -> dict:
    daily_var = float(weights @ sigma @ weights)
    daily_var = max(daily_var, 0.0)
    vol_ann = float(np.sqrt(252 * daily_var))
    spy_only = np.zeros_like(weights)
    spy_only[spy_idx] = 1.0 - cfg["cash_sleeve_pct"]
    active_share = float(np.sum(np.abs(weights - spy_only)) / 2)
    n_active = int(np.sum(weights > cfg["min_position_pct"]))
    turnover = float(np.sum(np.abs(weights - w_prev)) / 2)
    return {
        "status": status,
        "portfolio_vol_ann": vol_ann,
        "active_share_vs_spy": active_share,
        "n_active_positions": n_active,
        "turnover_one_way": turnover,
        "expected_alpha": float(weights @ alpha_hat),
        "weight_sum": float(weights.sum()),
    }


def make_cash_sentinel_returns(n_rows: int) -> np.ndarray:
    """Helper for callers: cash has zero return (treated as risk-free at sleeve)."""
    return np.zeros(n_rows)
