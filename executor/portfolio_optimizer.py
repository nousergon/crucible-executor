"""
Constrained mean-variance portfolio optimizer — PR 1 of the portfolio-optimizer
arc (plan: `alpha-engine-docs/private/portfolio-optimizer-260511.md`).

The institutional benchmark-as-null pattern: SPY (the benchmark) is the
no-conviction fill, cash is a pinned operational sleeve, conviction picks
express deviation from SPY within sector + position + vol-target constraints.

Math:
    maximize   wᵀα̂  −  λ · wᵀΣ_H w  −  γ · wᵀΩw  −  τ · ‖w − w_prev‖₁
    s.t.       Σwᵢ = 1                                    (budget)
               w[CASH] = cash_sleeve                       (sleeve pin)
               0 ≤ wᵢ ≤ stance_capᵢ                       (per-name cap)
               Σ_{i∈sector S} wᵢ ≤ max_sector_pct          (sector cap)
               wᵢ = 0 for i with eligibility=False         (gate mask)
               wᵀΣ_H w ≤ σ²_target_H                       (vol-target SOC)

Horizon convention: Σ_H is the H-day covariance, where H is set via
``cfg["sigma_horizon_days"]`` (default 1 = daily, preserves legacy behavior).
Under i.i.d. log-return assumption, Σ_H = H · Σ_daily — see
`alpha-engine-docs/private/optimizer-sota-upgrades-260526.md` §A.1 for the
rationale (align Σ horizon with the canonical 21d log-domain α̂).

α̂-uncertainty term (workstream B.3): Ω = diag(σ_α̂²) penalizes positions
in proportion to per-name predictor variance — Garlappi-Uppal-Wang 2007
diagonal-Ω form. γ = cfg["alpha_uncertainty_penalty"] (default 0.0 = OFF,
preserves bit-identical legacy MVO). σ_α̂ comes from the predictor's
BayesianRidge posterior (`predicted_alpha_std` in predictions JSON, shipped
in alpha-engine-predictor B.1 #199). When `alpha_uncertainty=None` or all
entries are NaN, the term is skipped regardless of γ — covers the 1-week
soak window before the next training cycle promotes a BayesianRidge model.

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
    *,
    alpha_uncertainty: np.ndarray | None = None,
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
        alpha_uncertainty: optional shape (N,) array of σ_α̂ per ticker —
            the BayesianRidge posterior std emitted by the predictor
            (predicted_alpha_std field in predictions JSON, B.1). When
            provided AND cfg["alpha_uncertainty_penalty"] > 0, adds the
            Garlappi-Uppal-Wang 2007 diagonal-Ω penalty term to the MVO
            objective so noisy picks size down proportionally. NaN entries
            are treated as zero uncertainty (no penalty for that name);
            covers the partial-rollout case during the 1-week soak between
            B.1 landing and the first BayesianRidge model being promoted
            in production. None ↔ no penalty regardless of γ.

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
    omega_diag, alpha_unc_used = _resolve_alpha_uncertainty(alpha_uncertainty, N, cfg)

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for portfolio_optimizer. Install via "
            "`pip install 'cvxpy>=1.4,<1.8'`. See requirements.txt."
        ) from e

    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(N)

    objective_terms = [
        alpha_hat @ w,
        - cfg["risk_aversion"] * cp.quad_form(w, sigma_psd),
        - (cfg["tcost_bps"] / 1e4) * cp.norm(w - w_prev, 1),
    ]
    if alpha_unc_used:
        # γ · sum_i (σ_α̂_i² · w_i²) — diagonal-Ω Garlappi-Uppal-Wang penalty.
        # cp.square(w) on a Variable is convex; sum with non-negative weights
        # remains convex; negated in a Maximize is concave (well-formed).
        gamma = float(cfg["alpha_uncertainty_penalty"])
        objective_terms.append(- gamma * (omega_diag @ cp.square(w)))
    objective = cp.Maximize(sum(objective_terms))

    eligibility_idx = np.where(~eligibility)[0]
    effective_caps = np.where(eligibility, stance_caps, 0.0)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        w <= effective_caps,
        w[cash_idx] == cfg["cash_sleeve_pct"],
    ]
    if cfg.get("vol_target_annual") is not None:
        # Σ is at horizon H. Under i.i.d. log-returns, Var_ann = Var_H · (252/H),
        # so the H-day variance budget that corresponds to annual vol_target is
        # vol_target² · H/252. At default H=1 this reduces to (vol_target/√252)².
        horizon = int(cfg.get("sigma_horizon_days", 1))
        sigma_target_squared = (cfg["vol_target_annual"] ** 2) * horizon / 252
        constraints.append(cp.quad_form(w, sigma_psd) <= sigma_target_squared)
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
            omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
        )
        return OptimizerResult(weights=weights, diagnostics=diagnostics)

    weights = _clip_and_renormalize(weights, effective_caps, cash_idx, cfg)
    diagnostics = _build_diagnostics(
        weights, w_prev, sigma, alpha_hat, spy_idx, status, cfg,
        omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
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
    # Horizon (trading days) at which Σ is expressed. 1 = legacy daily Σ
    # (bit-identical to pre-260526 behavior); set to 21 to align Σ with the
    # canonical 21d log-domain α̂. See optimizer-sota-upgrades-260526.md §A.1.
    "sigma_horizon_days": 1,
    # EWMA decay for ``covariance_shrinkage="ewma"``. RiskMetrics 1996
    # canonical value 0.94 ↔ ~11d half-life; 0.97 ↔ ~23d half-life (closer
    # to canonical 21d α̂ horizon). See optimizer-sota-upgrades-260526.md §A.2.
    "ewma_lambda_decay": 0.94,
    # γ for the Garlappi-Uppal-Wang 2007 α̂-uncertainty penalty term
    # γ · sum_i(σ_α̂_i² · w_i²). 0.0 (default) disables the term and
    # preserves bit-identical legacy MVO behavior. Backtester-tunable.
    # See optimizer-sota-upgrades-260526.md §B.3.
    "alpha_uncertainty_penalty": 0.0,
}


def _resolve_alpha_uncertainty(
    alpha_uncertainty: np.ndarray | None,
    N: int,
    cfg: dict,
) -> tuple[np.ndarray, bool]:
    """Build omega_diag = σ_α̂² and decide whether the penalty term is
    active for this solve.

    Returns (omega_diag, used) where ``used`` is True iff γ > 0 AND at
    least one σ_α̂ entry is finite AND non-zero. On used=False the caller
    skips the penalty term, preserving bit-identical legacy behavior.

    Negative or non-finite σ_α̂ entries are coerced to 0 (no penalty for
    that name) so partial-rollout (legacy Ridge std=None → NaN) does not
    raise. Caller's alpha_uncertainty contract is "predictor posterior
    std or NaN per ticker"; we enforce the σ ≥ 0 invariant defensively
    here too — a negative entry IS an upstream bug (BR posterior is
    always positive), but the optimizer is the wrong place to crash the
    morning planner over it. Log loud, treat as missing.
    """
    gamma = float(cfg.get("alpha_uncertainty_penalty", 0.0))
    if alpha_uncertainty is None or gamma <= 0.0:
        return np.zeros(N), False
    arr = np.asarray(alpha_uncertainty, dtype=np.float64).ravel()
    if arr.shape != (N,):
        raise ValueError(
            f"alpha_uncertainty shape {arr.shape} != ({N},) — must be one entry per ticker"
        )
    # Any negative entry is an upstream contract violation. Don't crash the
    # morning planner — log loud, coerce to 0 (per partial-rollout policy).
    if np.any(arr[np.isfinite(arr)] < 0.0):
        n_bad = int(np.sum((arr < 0.0) & np.isfinite(arr)))
        logger.warning(
            "alpha_uncertainty has %d negative entries — coercing to 0. "
            "Predictor BayesianRidge posterior is always positive; investigate "
            "upstream (B.1 #199 wiring).", n_bad,
        )
    # NaN / inf / negative → 0 → no penalty contribution
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    omega = arr ** 2
    used = bool(np.any(omega > 0.0))
    return omega, used


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


def _ewma_covariance(returns: np.ndarray, lambda_decay: float) -> np.ndarray:
    """RiskMetrics 1996 EWMA covariance with zero-mean assumption.

    Σ_EWMA = (1−λ) · Σ_{k=0}^{T-1} λ^k · r_{t-k} r_{t-k}ᵀ, normalized so weights
    sum to 1 over the finite window. The zero-mean simplification is standard
    for daily equity returns (E[r] ≪ σ); RiskMetrics 1996 §5.3.2.

    With λ=0.94 the effective half-life is log(0.5)/log(0.94) ≈ 11.2 trading days
    (RiskMetrics canonical); 0.97 → ~22.8 days (closer to 21d α̂ horizon).

    Degenerate at λ=1.0: weights become uniform 1/T → reduces to (unbiased
    only up to the 1/T vs 1/(T-1) factor) sample covariance. Tested.
    """
    if not 0.5 <= lambda_decay <= 1.0:
        raise ValueError(
            f"ewma_lambda_decay must be in [0.5, 1.0]; got {lambda_decay}. "
            f"RiskMetrics 1996 canonical is 0.94 (daily) or 0.97 (monthly)."
        )
    T = returns.shape[0]
    if lambda_decay >= 1.0 - 1e-12:
        # Uniform weights (degenerate). Treat λ=1 as plain sample-cov-equivalent.
        return (returns.T @ returns) / T
    # Newest observation first; row 0 carries the largest weight.
    R = returns[::-1]
    weights = (1.0 - lambda_decay) * lambda_decay ** np.arange(T)
    weights /= weights.sum()  # normalize for finite-window truncation
    return (R.T * weights) @ R


def _estimate_covariance(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    """Return covariance at horizon ``cfg["sigma_horizon_days"]``.

    Estimates Σ_daily via the configured estimator, then scales by horizon-days
    under i.i.d. log-return assumption: Σ_H = H · Σ_daily. Default H=1 preserves
    legacy daily Σ bit-identical (1 × Σ = Σ).

    Estimators (cfg["covariance_shrinkage"]):
      * "ledoit_wolf" (default): Ledoit-Wolf 2004 constant-correlation shrinkage
        on equal-weighted samples. Institutional default.
      * "oas": Chen et al. 2010 Oracle Approximating Shrinkage. Lower-MSE than
        LW when T/N is small (our universe ~27 × T~252 → T/N≈9 is modestly
        small-sample). Drop-in alternative; same shrinkage-target family
        (multiple of identity). See optimizer-sota-upgrades-260526.md §A.3.
      * "sample": raw sample covariance, no shrinkage. Test-only.
      * "ewma": RiskMetrics 1996 EWMA with cfg["ewma_lambda_decay"] (default
        0.94). Captures vol-clustering; weights recent observations more.
        See optimizer-sota-upgrades-260526.md §A.2.
    """
    clean = returns_panel[~np.isnan(returns_panel).any(axis=1)]
    if clean.shape[0] < 20:
        raise ValueError(
            f"Need ≥20 clean return rows for covariance; got {clean.shape[0]}"
        )
    estimator = cfg["covariance_shrinkage"]
    if estimator == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for Ledoit-Wolf shrinkage. Install "
                "via `pip install 'scikit-learn>=1.3,<1.6'`."
            ) from e
        sigma_daily = LedoitWolf().fit(clean).covariance_
    elif estimator == "oas":
        try:
            from sklearn.covariance import OAS
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for OAS shrinkage. Install via "
                "`pip install 'scikit-learn>=1.3,<1.6'`."
            ) from e
        sigma_daily = OAS().fit(clean).covariance_
    elif estimator == "sample":
        sigma_daily = np.cov(clean, rowvar=False)
    elif estimator == "ewma":
        sigma_daily = _ewma_covariance(clean, float(cfg.get("ewma_lambda_decay", 0.94)))
    else:
        raise ValueError(f"Unknown covariance_shrinkage: {estimator}")

    horizon = int(cfg.get("sigma_horizon_days", 1))
    if horizon < 1:
        raise ValueError(f"sigma_horizon_days must be ≥ 1; got {horizon}")
    return horizon * sigma_daily


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
    *,
    omega_diag: np.ndarray | None = None,
    alpha_unc_used: bool = False,
) -> dict:
    # sigma is at horizon H per _estimate_covariance. Annualize:
    # Var_ann = Var_H · (252/H) → vol_ann = √(252/H · Var_H). At default
    # H=1 this is the legacy √(252 · daily_var).
    horizon = int(cfg.get("sigma_horizon_days", 1))
    horizon_var = float(weights @ sigma @ weights)
    horizon_var = max(horizon_var, 0.0)
    vol_ann = float(np.sqrt((252 / horizon) * horizon_var))
    spy_only = np.zeros_like(weights)
    spy_only[spy_idx] = 1.0 - cfg["cash_sleeve_pct"]
    active_share = float(np.sum(np.abs(weights - spy_only)) / 2)
    n_active = int(np.sum(weights > cfg["min_position_pct"]))
    turnover = float(np.sum(np.abs(weights - w_prev)) / 2)
    out = {
        "status": status,
        "portfolio_vol_ann": vol_ann,
        "active_share_vs_spy": active_share,
        "n_active_positions": n_active,
        "turnover_one_way": turnover,
        "expected_alpha": float(weights @ alpha_hat),
        "weight_sum": float(weights.sum()),
        "alpha_uncertainty_penalty_used": alpha_unc_used,
    }
    # α̂-uncertainty observability (workstream B.3). Mean σ_α̂ across the
    # active book (omega_diag = σ²) — operator-readable signal for how
    # confident the predictor is on the names being sized today.
    if omega_diag is not None and np.any(omega_diag > 0.0):
        active_mask = weights > cfg["min_position_pct"]
        active_omega = omega_diag[active_mask]
        if active_omega.size > 0:
            out["mean_alpha_std_active"] = float(np.sqrt(active_omega.mean()))
            out["alpha_uncertainty_penalty_contribution"] = float(
                cfg.get("alpha_uncertainty_penalty", 0.0) * (omega_diag @ (weights ** 2))
            )
    return out


def make_cash_sentinel_returns(n_rows: int) -> np.ndarray:
    """Helper for callers: cash has zero return (treated as risk-free at sleeve)."""
    return np.zeros(n_rows)
