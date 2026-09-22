"""
Constrained mean-variance portfolio optimizer — PR 1 of the portfolio-optimizer
arc (plan: `alpha-engine-docs/private/portfolio-optimizer-260511.md`).

The institutional benchmark-as-null pattern: SPY (the benchmark) is the
no-conviction fill, cash is a pinned operational sleeve, conviction picks
express deviation from SPY within sector + position + vol-target constraints.

Math:
    maximize   wᵀα̂  −  λ · wᵀΣ_H w  −  γ · wᵀΩw  −  C(w − w_prev)/NAV
    s.t.       Σwᵢ = 1                                    (budget)
               w[CASH] = cash_sleeve                       (sleeve pin)
               0 ≤ wᵢ ≤ stance_capᵢ                       (per-name cap)
               Σ_{i∈sector S} wᵢ ≤ max_sector_pct          (sector cap)
               |wᵢ − w_prevᵢ| · NAV ≤ max_pct_adv · ADVᵢ   (participation cap)
               wᵢ = 0 for i with eligibility=False         (gate mask)
               wᵀΣ_H w ≤ σ²_target_H                       (vol-target SOC)

Transaction-cost term (tradeability arc, §43 — config#1401): the objective's
cost term is the participation-aware **square-root market-impact** cost from
the fleet's ONE shared engine (``nousergon_lib.quant.transaction_cost``,
lib#144), NOT a flat L1 turnover penalty. Per-name one-side dollar cost is

    C_i(Δwᵢ) = |Δwᵢ|·NAV · (half_spread + commission)/1e4              (linear)
             + impact_coef/1e4 · NAV^{1.5} · |Δwᵢ|^{1.5} / √ADVᵢ        (impact)

i.e. cost ∝ half_spread + c·√(participation) per Almgren-Chriss/Kissell, so the
DOLLAR cost scales as |Δwᵢ|^{1.5} (participation^{1.5}) — CONVEX in the trade
size, so ``−Σ C_i / NAV`` is concave and the objective stays DCP. Keying the
impact term on per-name ADV$ (from the scanner tradeability artifact) makes
turnover cost participation-aware: rebalancing an illiquid name is penalized far
more than a liquid one, where the flat 5bps L1 penalty was liquidity-blind. When
a name has no ADV coverage the impact term drops to the half-spread+commission
floor (the lib's conservative fallback) — never an error, never a silent zero.
The impact coefficient is a literature default (impact_coef_bps≈10 at 100%
participation) and configurable via the ``transaction_cost`` config block; a
TCA calibration loop against realized slippage_vs_signal (daemon.py:1570) is a
documented FOLLOW-ON, not built here. When ``adv_usd`` is None (or all-NaN, or
``portfolio_notional`` is None — pre-tradeability-artifact rollout) the term
degrades to the legacy flat L1 turnover penalty (``tcost_bps``), preserving
bit-identical fail-soft behavior. See §43 + optimizer-sota-upgrades-260526.md.

Horizon convention: Σ_H is the H-day covariance, where H is set via
``cfg["sigma_horizon_days"]`` (default 1 = daily, preserves legacy behavior).
Under i.i.d. log-return assumption, Σ_H = H · Σ_daily — see
`alpha-engine-docs/private/optimizer-sota-upgrades-260526.md` §A.1 for the
rationale (align Σ horizon with the canonical 21d log-domain α̂).

α̂-uncertainty term (workstream B.3): Ω = diag(σ_ε²) penalizes positions in
proportion to per-name ESTIMATION variance — Garlappi-Uppal-Wang 2007
diagonal-Ω form. γ = cfg["alpha_uncertainty_penalty"].

WHICH VINTAGE OF THE UNCERTAINTY FIELD, AND WHY (alpha-engine-config-I9452).
Ω is built from `predicted_alpha_std_epistemic` — sqrt(xᵀΣ_w x), the
estimation-error std of α̂, shipped by crucible-predictor PR596 — and never
from `predicted_alpha_std`. The latter is the BayesianRidge PREDICTIVE std,
1/α̂ + xᵀΣ_w x, whose first term is a scalar learned at fit time. GUW's Ω is
defined as the covariance of the estimation error of μ̂, so including the
observation-noise term both double-counts risk Σ already carries and adds a
per-batch constant that annihilates the cross-section. Measured over the 62
stored predictions/{date}.json artifacts from the 2026-06-01 BayesianRidge
cutover to 2026-08-31, that constant carried 90–98% of the predictive
variance and the total's cross-name CV never exceeded 0.008 — so between
2026-06-01 and this change the term was a uniform ridge (γ was 0.0 in
production throughout, so nothing traded on it).

When the epistemic vector is absent, unusable, or itself cross-sectionally
flat (CV below cfg["alpha_uncertainty_min_cv"]), the term is declared
INOPERATIVE and the reason is written to the solve diagnostics. It is never
silently re-pointed at the total. See `_resolve_alpha_uncertainty`.

`alpha_uncertainty` (the total) is still a parameter, and still the input to
the CONVICTION GATE on the discretionary turnover budget, whose IR band was
derived against the total's scale (PR518). The two knobs read different
vintages ON PURPOSE: swapping the gate's denominator to the epistemic half
would multiply IR_xs by ~6x and silently move its operating point.

This module is a pure function over numpy inputs. It does no I/O, no logging
config side effects, no S3 calls — easy to unit-test (PR 1) and easy to wire
into shadow mode (PR 2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Cross-sectional CV floor for the GUW Ω input, mirroring the producer's
# crucible-predictor/monitoring/drift_detector.py::ALPHA_UNCERTAINTY_MIN_CV.
# Overridable via cfg["alpha_uncertainty_min_cv"].
_ALPHA_UNCERTAINTY_MIN_CV = 0.01

_CLARABEL = "CLARABEL"
_FALLBACK_SOLVERS = ("SCS", "OSQP")


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    diagnostics: dict


def solve_target_weights(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray | None,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
    cfg: dict,
    *,
    alpha_uncertainty: np.ndarray | None = None,
    alpha_uncertainty_epistemic: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    adv_usd: np.ndarray | None = None,
    portfolio_notional: float | None = None,
    name_sigma: np.ndarray | None = None,
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
        alpha_uncertainty: optional shape (N,) array of the predictor's
            TOTAL predictive std per ticker (`predicted_alpha_std`, B.1).
            This is the CONVICTION GATE's input — the gate's IR band was
            derived against this field's scale (PR518) and must keep reading
            it. It does NOT enter the GUW Ω; see the module docstring.
        alpha_uncertainty_epistemic: optional shape (N,) array of
            sqrt(xᵀΣ_w x) per ticker (`predicted_alpha_std_epistemic`,
            crucible-predictor PR596) — the estimation-error std, and the
            ONLY vector the Garlappi-Uppal-Wang Ω is built from. NaN entries
            are treated as zero uncertainty (no penalty for that name).
            None, unusable, or cross-sectionally flat ↔ the penalty is
            declared inoperative with a recorded reason, regardless of γ.
        covariance: optional shape (N,N) DAILY covariance matrix Σ_daily
            (pre-horizon-scaling). When provided, the returns-panel estimator
            step is skipped and this matrix is used directly (horizon scaling
            still applied: Σ_H = H · Σ_daily). This is the intraday-re-solve
            path: the daemon reuses the morning Σ (cached in the optimizer
            shadow log) so an event-driven re-solve after a hard-risk exit is
            mechanism-identical to the morning solve and adds zero alpha
            look-ahead (Σ is daily-stable). ``returns_panel`` may be None when
            ``covariance`` is provided. None ↔ estimate Σ from returns_panel
            as before.
        adv_usd: optional shape (N,) per-name average daily DOLLAR volume
            (price × shares), read from the scanner tradeability artifact's
            ``tradeability.adv_usd`` block (crucible-research#343). Drives BOTH
            the participation-aware √-impact cost term AND the max-%-ADV
            participation constraint. Entries that are NaN / ≤0 (coverage gap)
            fall back to the half-spread+commission cost floor and are exempt
            from the participation constraint (no ADV → no participation bound
            can be formed — conservative degrade, never a crash). SPY and CASH
            entries are ignored (benchmark fill / sleeve carry no market
            impact). None ↔ no ADV info → the cost term degrades to the legacy
            flat ``tcost_bps`` L1 penalty and the participation constraint is
            skipped (bit-identical pre-tradeability behavior).
        portfolio_notional: optional book size in dollars (NAV) — required to
            convert weight deltas to trade notionals for the √-impact term and
            the max-%-ADV constraint. When None (or ≤0) the participation-aware
            path is disabled and the optimizer falls back to the legacy flat
            ``tcost_bps`` penalty, preserving fail-soft behavior.
        name_sigma: optional shape (N,) per-name daily return volatility used
            for the Almgren-Chriss σ-scaling of the impact term (σᵢ/refσ, where
            refσ = the cross-sectional median). None ↔ σ-agnostic √-impact (the
            lib default), which is the safe institutional baseline.

    Returns:
        OptimizerResult with weights (length N, sums to 1, sleeve pinned) and
        diagnostics dict including solver status, portfolio vol, active share
        vs SPY, and n_active.

    On infeasibility, returns the fallback weights (current weights with cash
    absorbing the residual) and diagnostics["status"] = "infeasible_fallback".
    """
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS, **cfg}
    N = len(tickers)
    _validate_inputs(
        tickers, alpha_hat, returns_panel, w_prev,
        sectors, stance_caps, eligibility, spy_idx, cash_idx,
        covariance_provided=covariance is not None,
    )

    if covariance is not None:
        # Intraday re-solve: reuse the morning DAILY Σ instead of re-estimating.
        # Apply the SAME horizon scaling the estimator path applies (Σ_H =
        # H · Σ_daily); persisting/injecting an already-horizon-scaled Σ here
        # would double-scale and silently corrupt every vol diagnostic + the
        # vol-target SOC constraint — guarded by _validate_covariance + the
        # caller's vol-parity assertion.
        sigma_daily = _validate_covariance(covariance, N)
        horizon = int(cfg.get("sigma_horizon_days", 1))
        if horizon < 1:
            raise ValueError(f"sigma_horizon_days must be ≥ 1; got {horizon}")
        sigma = horizon * sigma_daily
    else:
        sigma = _estimate_covariance(returns_panel, cfg)
    omega_diag, alpha_unc_used, alpha_unc_meta = _resolve_alpha_uncertainty(
        alpha_uncertainty, alpha_uncertainty_epistemic, N, cfg,
    )

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for portfolio_optimizer. Install via "
            "`pip install 'cvxpy>=1.9.2,<1.10'`. See requirements.txt."
        ) from e

    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(N)

    tcost = _build_tcost_term(
        cp, w, w_prev, adv_usd, portfolio_notional, name_sigma,
        spy_idx, cash_idx, cfg,
    )
    objective_terms = [
        alpha_hat @ w,
        - cfg["risk_aversion"] * cp.quad_form(w, sigma_psd),
        tcost.objective_term,
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

    # ── max-%-ADV participation constraint (tradeability arc, config#1401) ──
    # Bound the single-solve trade in each name to a fraction of its average
    # daily dollar volume: |Δwᵢ|·NAV ≤ max_pct_adv · ADVᵢ. This is the HARD
    # capacity guardrail the √-impact objective term complements — the cost
    # term prices participation, this constraint refuses to trade a name so
    # thin that even a small book move would move the market. Only applied to
    # names with usable ADV coverage (NaN/≤0 → no bound can be formed → exempt,
    # conservative degrade) and only when a book notional is known; SPY/CASH
    # (benchmark fill / sleeve) carry no market impact and are always exempt.
    adv_cap_meta = _apply_max_pct_adv_constraint(
        cp, w, w_prev, constraints, adv_usd, portfolio_notional,
        spy_idx, cash_idx, cfg,
    )

    # ── daily turnover budget, as a CONSTRAINT (alpha-engine-config-I7346) ──
    # ‖w − w_prev‖₁ / 2 ≤ max_daily_turnover. cvxpy expresses the L1 norm
    # directly and the problem stays DCP (a norm is convex, so a ≤ bound is a
    # convex set). The optimizer therefore CHOOSES which trades fit the budget
    # and each surviving name lands at a size it actually wants.
    #
    # This replaces a post-solve uniform shrink
    # (w_exec = w_prev + (w_target − w_prev)·cap/requested), which answered
    # "how much may I trade" by trading a bit less of EVERYTHING — including
    # names whose entire delta IS an intended new position. Downstream,
    # optimizer_shadow's rebalance band drops any |Δw| < rebalance_band_pct,
    # so a hard enough shrink could push an entire entry cohort under the band
    # and delete it while the solve still reported `optimal`. A budget spent
    # by the objective cannot do that: nothing is scaled after the fact.
    # The exact minimum one-way turnover the constraints so far already
    # force, measured before the budget is added to them (I11368).
    _min_turnover, _min_status = _solve_min_attainable_turnover(
        cp, w, w_prev, constraints,
    )
    turnover_meta = _apply_turnover_constraint(
        cp, w, w_prev, constraints, effective_caps, cash_idx, cfg,
        eligibility=eligibility,
        alpha_hat=alpha_hat,
        alpha_uncertainty=alpha_uncertainty,
        spy_idx=spy_idx,
        min_attainable_turnover=_min_turnover,
        min_attainable_status=_min_status,
    )

    problem = cp.Problem(objective, constraints)
    weights, status, solver_name = _solve_with_fallback(problem, w, cfg)

    if weights is None:
        weights = _fallback_weights(w_prev, cash_idx, cfg["cash_sleeve_pct"])
        diagnostics = _build_diagnostics(
            weights, w_prev, sigma, alpha_hat, spy_idx, "infeasible_fallback", cfg,
            omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
            alpha_unc_meta=alpha_unc_meta,
        )
        diagnostics["solver_name"] = solver_name
        diagnostics.update(tcost.diagnostics)
        diagnostics.update(adv_cap_meta)
        diagnostics.update(_turnover_diagnostics(weights, w_prev, turnover_meta))
        return OptimizerResult(weights=weights, diagnostics=diagnostics)

    weights, clip_meta = _clip_and_renormalize(
        weights, effective_caps, cash_idx, cfg
    )
    weights, governor = _apply_turnover_governor(
        weights, w_prev, cfg,
        turnover_meta=turnover_meta,
        clip_meta=clip_meta,
        solver_status=status,
        solver_name=solver_name,
    )
    diagnostics = _build_diagnostics(
        weights, w_prev, sigma, alpha_hat, spy_idx, status, cfg,
        omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
        alpha_unc_meta=alpha_unc_meta,
    )
    diagnostics["solver_name"] = solver_name
    diagnostics.update(governor)
    diagnostics.update(tcost.diagnostics)
    diagnostics.update(adv_cap_meta)
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
    # ── Participation-aware transaction cost (tradeability arc, config#1401) ─
    # ``tcost_mode`` selects the objective's turnover-cost term:
    #   "sqrt_impact" (default) — the canonical participation-aware √-impact
    #     dollar cost from nousergon_lib.quant.transaction_cost (lib#144),
    #     keyed on per-name ADV$. Requires adv_usd + portfolio_notional; when
    #     either is absent it AUTOMATICALLY degrades to the flat L1 penalty
    #     (fail-soft). This is the institutional-correct construction cost.
    #   "flat_l1" — the legacy flat ``tcost_bps`` L1 penalty (liquidity-blind).
    #     Kept for A/B and as the explicit fallback the auto-degrade lands on.
    # The impact COEFFICIENT (impact_coef_bps, half_spread_bps, commission_bps)
    # lives in the ``transaction_cost`` config block consumed by
    # TransactionCostModel.from_config — literature defaults today; a TCA
    # calibration loop against realized slippage_vs_signal (daemon.py:1570) is
    # the documented FOLLOW-ON that will tune impact_coef_bps. See §43.
    "tcost_mode": "sqrt_impact",
    # ── max-%-ADV participation constraint (config#1401) ─────────────────────
    # HARD capacity guardrail: |Δwᵢ|·NAV ≤ max_pct_adv · ADVᵢ per name per
    # solve. None → constraint OFF (bit-identical legacy behavior). 0.05 = a
    # single-solve trade may consume at most 5% of a name's average daily
    # dollar volume — a conservative institutional participation ceiling that
    # keeps the √-impact objective term honest (the cost prices participation;
    # this refuses to trade a name so thin the model can't be trusted). Only
    # binds on names with ADV coverage; requires portfolio_notional to convert
    # weight deltas to notionals — skipped (with a diagnostic) when absent.
    "max_pct_adv": 0.05,
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
    # Cross-sectional CV floor below which Ω is declared INOPERATIVE rather
    # than applied. Mirrors the producer's
    # crucible-predictor/monitoring/drift_detector.py::ALPHA_UNCERTAINTY_MIN_CV
    # so the consumer stops using a channel on exactly the sessions the
    # producer's detector calls dead. See _resolve_alpha_uncertainty.
    "alpha_uncertainty_min_cv": 0.01,
    # ── Turnover governor (gradual-rebalance guardrail) ──────────────────
    # SAFETY guardrail — NOT an alpha knob, NOT backtester-tuned. Caps the
    # one-way turnover the book may execute in a single day by scaling the
    # step from w_prev toward the optimizer's target, so the portfolio WALKS
    # to the target over several daily re-solves instead of jumping in one
    # session. Institutional books rebalance gradually; a large single-day
    # reallocation is the rare exception that should be operator-reviewed,
    # not the default. Defaults ON (unlike the optional α̂-uncertainty term)
    # because it's a fail-safe — a too-tight cap only slows rebalancing, it
    # can never produce a worse trade.
    #   max_daily_turnover: one-way turnover cap/day (None → governor OFF,
    #     bit-identical legacy behavior).
    #   large_move_turnover_flag: when REQUESTED (uncapped) one-way turnover
    #     exceeds this, the solve sets large_move_flagged so the planner
    #     alerts for approval. The move is STILL executed gradually under the
    #     cap — flagging never bypasses the cap, and the cap never waits on
    #     the flag.
    "max_daily_turnover": 0.20,
    "large_move_turnover_flag": 0.35,
    # ── Turnover tripwire (L4515) ─────────────────────────────────────────
    # SAFETY alarm — NOT an alpha knob, NOT backtester-tuned. Band-checks the
    # EXECUTED one-way turnover daily in the planner and pages on breach
    # (executor/turnover_tripwire.py): daily = cap × multiple at ERROR (the
    # governor should make a breach impossible, so one means the cap was
    # bypassed/disabled); rolling = sum over the last N sessions at WARN
    # (churn-by-a-thousand-cuts — each day under the cap, week abnormal; the
    # signature of the 5/29, 6/01, 6/04 incidents this generalizes).
    # ── Conviction gate on the DISCRETIONARY turnover budget (I9315) ─────
    # SAFETY guardrail — NOT an alpha knob. The daily turnover budget answers
    # "how far may the book move today". It has never answered "is today's
    # target worth moving toward at all", and between 2026-08-17 and 2026-08-31
    # that gap put the optimizer at the 20%/day cap on 12 consecutive sessions:
    # the predictor's cross-sectional α̂ spread had collapsed to ~1/30th of its
    # own published per-name σ_α̂, so the ranking driving the target was noise,
    # and the target itself moved 11–59% one-way PER DAY (measured by replaying
    # each day's stored solve with the budget removed). A budget alone cannot
    # see that: it caps the step, never the reason for the step.
    #
    # The gate scales the DISCRETIONARY budget by a measured signal-quality
    # multiplier q ∈ [min_multiple, 1]:
    #
    #     IR_xs = stdev_cross_section(α̂_eligible) / median(σ_α̂_eligible)
    #     q     = clip((IR_xs − ir_floor) / (ir_full − ir_floor), min, 1)
    #
    # IR_xs is the cross-sectional information ratio of the alpha vector — how
    # large the spread the optimizer is ranking on is, relative to the error
    # bar the model itself puts on each element of it. Below ir_floor the names
    # are statistically tied and rebalancing between them is a pure transaction
    # cost.
    #
    # The band is DERIVED, not chosen. Measured over 2026-07-31..2026-08-31
    # (20 sessions of stored optimizer_shadow artifacts): IR_xs ∈ [1.01, 5.20]
    # on every session of the healthy regime (n=7, executed turnover 0.4%–8.1%)
    # and IR_xs ∈ [0.025, 0.320] on every session of the churning regime (n=13,
    # executed turnover 10.3%–20.0%). [0.35, 0.75] sits inside that empirical
    # gap and touches neither side, so the gate is inert over the whole healthy
    # sample and fully engaged over the whole degraded one.
    #
    # The MANDATORY floor is unaffected: `_apply_turnover_constraint` still
    # raises the effective budget to the forced-exit / ineligibility / cash-pin
    # floor. A hard-risk exit is not discretionary and is never starved of
    # budget by this gate — measured floors up to 13.2% one-way in the same
    # window still execute in full.
    #
    # min_multiple is deliberately non-zero: a book that can NEVER trade
    # discretionarily cannot recover from small drift, and a hard zero makes
    # the constraint's feasible set degenerate. 0.05 × 0.20 = 1%/day one-way,
    # i.e. 5%/5d against a 60%/5d tripwire band.
    #
    # Degradation is loud, not silent: when σ_α̂ is absent, non-finite, or
    # zero, the gate CANNOT be evaluated, q is 1.0 (no throttle — never a
    # tighter budget from missing data) and `conviction_gate_reason` names the
    # reason in the diagnostics and the shadow artifact.
    "conviction_budget_gate_enabled": True,
    "conviction_ir_floor": 0.35,
    "conviction_ir_full": 0.75,
    "conviction_budget_min_multiple": 0.05,
    "conviction_gate_min_names": 3,
    "turnover_tripwire_enabled": True,
    "turnover_tripwire_daily_multiple": 1.25,
    "turnover_tripwire_rolling_days": 5,
    "turnover_tripwire_rolling_sum_band": 0.60,
}


def _resolve_alpha_uncertainty(
    alpha_uncertainty: np.ndarray | None,
    alpha_uncertainty_epistemic: np.ndarray | None,
    N: int,
    cfg: dict,
) -> tuple[np.ndarray, bool, dict]:
    """Build Ω = diag(σ_ε²) for the GUW term and decide whether it is OPERATIVE.

    Ω is the covariance of the **estimation error of μ̂** (Garlappi, Uppal &
    Wang 2007). The predictor's ``predicted_alpha_std`` is the BayesianRidge
    *predictive* std,

        σ_pred(x)² = 1/α̂  +  xᵀ Σ_w x
                     ^^^^     ^^^^^^^^^^
                     scalar   per-name

    whose first term is learned once at fit time and is the SAME number for
    every name in a batch. Feeding the total to Ω therefore (a) double-counts
    observation risk the covariance matrix Σ already carries and (b) adds a
    per-batch constant that annihilates the cross-section the penalty exists
    to exploit. Measured over the 62 stored ``predictions/{date}.json``
    artifacts from the 2026-06-01 BayesianRidge cutover to 2026-08-31, the
    noise term carried 90–98% of σ_pred² and the cross-name coefficient of
    variation of the total NEVER exceeded 0.008 — the term has been a uniform
    ridge for three months (alpha-engine-config-I9446 / I9452).

    So Ω is built from ``predicted_alpha_std_epistemic`` (= sqrt(xᵀΣ_w x),
    shipped by ``crucible-predictor`` PR596) and from NOTHING ELSE. The three
    ways it can be inoperative are all RECORDED, never silent:

    ``gamma_zero``                γ ≤ 0 — the term is configured off.
    ``epistemic_field_absent``    the caller passed no epistemic vector, or
                                  every entry is unusable. Covers every stored
                                  artifact written before 2026-08-31 and any
                                  champion family with no scalar noise
                                  variance (a legacy Ridge pickle). It does
                                  NOT fall back to ``predicted_alpha_std``:
                                  that would silently reinstate the uniform
                                  ridge this change exists to remove, with
                                  nothing on the artifact saying so. It also
                                  does not raise — a degraded sizing knob must
                                  not halt the morning planner.
    ``cross_section_below_floor`` the epistemic vector is present but its
                                  cross-sectional CV is below
                                  ``alpha_uncertainty_min_cv``, i.e. it too is
                                  a uniform ridge. Same floor (0.01) the
                                  producer's
                                  ``drift_detector.ALPHA_UNCERTAINTY_MIN_CV``
                                  is calibrated on, and the same regime it
                                  separates: a champion whose posterior never
                                  left its prior emits a LARGE but flat
                                  epistemic vector (measured 0.207–0.230 with
                                  CV 0.00007–0.00025 on
                                  ``v3.0-meta-2026-08-21-7d3d1cce``), which a
                                  magnitude test cannot catch.

    Returns ``(omega_diag, used, meta)``. ``meta`` is merged into the solve
    diagnostics and is populated on EVERY path — a field that appears only
    when the penalty engages is indistinguishable from a dead penalty.

    ``alpha_uncertainty`` (the total) is accepted but never enters Ω. It stays
    in the signature because it remains the CONVICTION GATE's input, whose
    band was derived against the total's scale (``crucible-executor`` PR518,
    alpha-engine-config-I9447), and because a shape mismatch on it is still
    a caller bug worth raising on.
    """
    gamma = float(cfg.get("alpha_uncertainty_penalty", 0.0))
    meta: dict = {
        "alpha_uncertainty_vintage": "epistemic",
        "alpha_uncertainty_inoperative_reason": None,
        "alpha_uncertainty_epistemic_cv": None,
        "alpha_uncertainty_min_cv": float(cfg.get("alpha_uncertainty_min_cv", _ALPHA_UNCERTAINTY_MIN_CV)),
        "alpha_uncertainty_n_usable": 0,
    }
    if gamma <= 0.0:
        meta["alpha_uncertainty_inoperative_reason"] = "gamma_zero"
        return np.zeros(N), False, meta
    if alpha_uncertainty_epistemic is None:
        meta["alpha_uncertainty_inoperative_reason"] = "epistemic_field_absent"
        logger.warning(
            "GUW alpha-uncertainty penalty INOPERATIVE (gamma=%.4g): no "
            "predicted_alpha_std_epistemic vector was supplied. The total "
            "predicted_alpha_std is NOT substituted — it is cross-sectionally "
            "flat (CV <= 0.008 on every measured session) and Omega built from "
            "it is a uniform ridge that double-counts observation risk already "
            "carried by Sigma. Sizing proceeds without the penalty; see "
            "alpha-engine-config-I9452.",
            gamma,
        )
        return np.zeros(N), False, meta

    arr = np.asarray(alpha_uncertainty_epistemic, dtype=np.float64).ravel()
    if arr.shape != (N,):
        raise ValueError(
            f"alpha_uncertainty_epistemic shape {arr.shape} != ({N},) — "
            "must be one entry per ticker"
        )
    # A negative entry is an upstream contract violation (an estimation std is
    # non-negative by construction). Log loud, coerce to 0 — the optimizer is
    # the wrong place to crash the morning planner over a sizing input.
    if np.any(arr[np.isfinite(arr)] < 0.0):
        n_bad = int(np.sum((arr < 0.0) & np.isfinite(arr)))
        logger.warning(
            "alpha_uncertainty_epistemic has %d negative entries — coercing to "
            "0. sqrt(x'Sigma_w x) is non-negative by construction; investigate "
            "the producer (crucible-predictor decompose_alpha_std).", n_bad,
        )
    # NaN / inf / negative → 0 → no penalty contribution for that name.
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    usable = arr[arr > 0.0]
    meta["alpha_uncertainty_n_usable"] = int(usable.size)
    if usable.size < 2:
        meta["alpha_uncertainty_inoperative_reason"] = "epistemic_field_absent"
        logger.warning(
            "GUW alpha-uncertainty penalty INOPERATIVE (gamma=%.4g): the "
            "epistemic vector carries %d usable entries. Not substituting the "
            "total; see alpha-engine-config-I9452.",
            gamma, int(usable.size),
        )
        return np.zeros(N), False, meta

    cv = float(usable.std() / usable.mean()) if usable.mean() > 0.0 else 0.0
    meta["alpha_uncertainty_epistemic_cv"] = cv
    if cv < meta["alpha_uncertainty_min_cv"]:
        meta["alpha_uncertainty_inoperative_reason"] = "cross_section_below_floor"
        logger.warning(
            "GUW alpha-uncertainty penalty INOPERATIVE (gamma=%.4g): the "
            "epistemic cross-section is flat (CV %.6f < floor %.4f) over %d "
            "names, median %.6f. Omega would be a uniform ridge — it would "
            "shrink every name identically and discriminate between nothing. "
            "This is the signature of a champion whose posterior never left "
            "its prior (measured on v3.0-meta-2026-08-21-7d3d1cce, "
            "2026-08-24..28), which a magnitude test cannot see because the "
            "vector is LARGE as well as flat.",
            gamma, cv, meta["alpha_uncertainty_min_cv"],
            int(usable.size), float(np.median(usable)),
        )
        return np.zeros(N), False, meta

    return arr ** 2, True, meta


@dataclass(frozen=True)
class _TCostTerm:
    """The optimizer objective's turnover-cost term + its observability."""
    objective_term: object          # a cvxpy expression (concave, DCP-safe)
    diagnostics: dict


def _clean_adv(
    adv_usd: np.ndarray | None, N: int, spy_idx: int, cash_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an ADV$ vector → (adv, usable_mask).

    ``adv`` has NaN/≤0/non-finite entries coerced to 0.0 (no coverage).
    ``usable_mask`` is True only for real names (not SPY/CASH) with ADV>0.
    SPY (benchmark fill) and CASH (sleeve) carry no market impact and are
    always excluded from both the cost term and the participation constraint.
    ``adv_usd=None`` → all-zero adv, all-False mask (no ADV info at all).
    """
    if adv_usd is None:
        return np.zeros(N), np.zeros(N, dtype=bool)
    adv = np.asarray(adv_usd, dtype=np.float64).ravel()
    if adv.shape != (N,):
        raise ValueError(
            f"adv_usd shape {adv.shape} != ({N},) — one ADV$ entry per ticker"
        )
    adv = np.where(np.isfinite(adv) & (adv > 0.0), adv, 0.0)
    usable = adv > 0.0
    usable[spy_idx] = False
    usable[cash_idx] = False
    return adv, usable


def _resolve_ref_sigma(
    name_sigma: np.ndarray | None, usable_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Return (sigma_used, ref_sigma) for the Almgren-Chriss σ-scaling.

    ref_sigma is the cross-sectional MEDIAN σ over usable names (the lib's
    self-calibrating reference: the median-vol name reproduces the σ-agnostic
    cost). Returns (None, None) — σ-agnostic — when no per-name σ is supplied
    or no usable finite-positive σ exists (safe institutional default).
    """
    if name_sigma is None:
        return None, None
    sig = np.asarray(name_sigma, dtype=np.float64).ravel()
    finite_pos = np.isfinite(sig) & (sig > 0.0) & usable_mask
    if not np.any(finite_pos):
        return None, None
    ref = float(np.median(sig[finite_pos]))
    if not (ref > 0.0):
        return None, None
    return sig, ref


def _build_tcost_term(
    cp, w, w_prev: np.ndarray,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    name_sigma: np.ndarray | None,
    spy_idx: int, cash_idx: int,
    cfg: dict,
) -> _TCostTerm:
    """Build the objective's turnover-cost term.

    Consumes the fleet's canonical √-impact ``TransactionCostModel`` (lib#144)
    for the coefficients — the executor never re-derives the impact math. The
    per-name one-side DOLLAR cost decomposes into a cvxpy-DCP-safe convex sum:

        C_i = (half_spread+commission)/1e4 · NAV · |Δwᵢ|          (linear)
            + impact_coef·(σᵢ/refσ)/1e4 · NAV^{1.5}/√ADVᵢ · |Δwᵢ|^{1.5}

    |Δwᵢ| is convex; |Δwᵢ|^{1.5} = power(|Δwᵢ|,1.5) is convex; both enter the
    objective as ``−ΣC_i/NAV`` (concave → valid in cp.Maximize). Dividing by
    NAV keeps the term commensurate with the (weight-space) α̂ and risk terms.

    Fail-soft: when tcost_mode="flat_l1", OR portfolio_notional is missing/≤0,
    OR no name has usable ADV coverage, the term degrades to the legacy flat
    ``tcost_bps`` L1 penalty ``−(tcost_bps/1e4)·‖w−w_prev‖₁`` — bit-identical
    to pre-1401 behavior. The chosen mode is surfaced in diagnostics.
    """
    flat_l1 = - (cfg["tcost_bps"] / 1e4) * cp.norm(w - w_prev, 1)
    mode = str(cfg.get("tcost_mode", "sqrt_impact"))
    N = w.shape[0]
    adv, usable = _clean_adv(adv_usd, N, spy_idx, cash_idx)
    n_usable = int(np.sum(usable))

    def _flat(reason: str) -> _TCostTerm:
        return _TCostTerm(
            objective_term=flat_l1,
            diagnostics={
                "tcost_term_mode": "flat_l1",
                "tcost_fallback_reason": reason,
                "tcost_n_names_with_adv": n_usable,
            },
        )

    if mode == "flat_l1":
        return _flat("configured_flat_l1")
    if mode != "sqrt_impact":
        raise ValueError(f"Unknown tcost_mode: {mode!r} (expected sqrt_impact|flat_l1)")
    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        return _flat("no_portfolio_notional")
    if n_usable == 0:
        # No ADV coverage anywhere (pre-tradeability-artifact rollout, or an
        # all-gap universe) → cannot form the participation-aware term.
        return _flat("no_adv_coverage")

    try:
        from nousergon_lib.quant.transaction_cost import TransactionCostModel
    except ImportError as e:  # pragma: no cover - pin guarantees availability
        logger.warning(
            "nousergon_lib.quant.transaction_cost unavailable (%s) — falling "
            "back to flat L1 turnover penalty. Bump the nousergon-lib pin to "
            ">=v0.75.0 (lib#144).", e,
        )
        return _flat("lib_unavailable")

    model = TransactionCostModel.from_config(cfg)
    nav = float(portfolio_notional)
    sig_used, ref_sigma = _resolve_ref_sigma(name_sigma, usable)

    # Linear (half-spread + commission) part — applies to EVERY name that
    # trades, ADV-covered or not: it's the spread/commission floor, not impact.
    linear_bps = model.half_spread_bps + model.commission_bps
    dw = w - w_prev
    linear_cost = (linear_bps / 1e4) * nav * cp.abs(dw)  # per-name $, vector

    # Impact part — only names with ADV coverage. k_i · |Δwᵢ|^{1.5}, where
    # k_i = impact_coef·(σᵢ/refσ)/1e4 · NAV^{1.5} / √ADVᵢ.
    impact_terms = []
    for i in np.where(usable)[0]:
        sigma_scale = 1.0
        if sig_used is not None and ref_sigma:
            s = sig_used[i]
            if np.isfinite(s) and s > 0.0:
                sigma_scale = float(s) / ref_sigma
        k_i = (
            model.impact_coef_bps * sigma_scale / 1e4
            * (nav ** 1.5) / math.sqrt(adv[i])
        )
        if k_i > 0.0:
            impact_terms.append(k_i * cp.power(cp.abs(dw[i]), 1.5))

    total_cost = cp.sum(linear_cost)
    if impact_terms:
        total_cost = total_cost + cp.sum(impact_terms)
    # Normalize dollar cost back to weight units (÷NAV) so the cost term is
    # commensurate with the weight-space α̂ / risk / uncertainty terms.
    objective_term = - total_cost / nav
    return _TCostTerm(
        objective_term=objective_term,
        diagnostics={
            "tcost_term_mode": "sqrt_impact",
            "tcost_n_names_with_adv": n_usable,
            "tcost_impact_coef_bps": float(model.impact_coef_bps),
            "tcost_sigma_scaled": bool(sig_used is not None and ref_sigma),
            "tcost_portfolio_notional": nav,
        },
    )


def _apply_max_pct_adv_constraint(
    cp, w, w_prev: np.ndarray, constraints: list,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    spy_idx: int, cash_idx: int,
    cfg: dict,
) -> dict:
    """Append the per-name max-%-ADV participation constraint, in place.

    ``|wᵢ − w_prevᵢ| · NAV ≤ max_pct_adv · ADVᵢ`` for every real name with
    usable ADV coverage. cvxpy encodes ``|Δwᵢ| ≤ bound_i`` as the linear pair
    ``Δwᵢ ≤ bound_i``, ``−Δwᵢ ≤ bound_i`` (kept affine so the LP/SOCP stays
    convex). Skipped (returns a diagnostic) when the cap is disabled, no book
    notional is known, or no name has ADV coverage — fail-soft, never a crash.

    Returns a diagnostics dict recording whether the cap was applied + to how
    many names, so an operator can see the constraint is (or isn't) live.
    """
    cap = cfg.get("max_pct_adv")
    N = w.shape[0]
    adv, usable = _clean_adv(adv_usd, N, spy_idx, cash_idx)
    n_usable = int(np.sum(usable))
    if cap is None or not (float(cap) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "disabled"}
    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_portfolio_notional"}
    if n_usable == 0:
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_adv_coverage"}

    nav = float(portfolio_notional)
    cap = float(cap)
    idx = np.where(usable)[0]
    # Per-name weight-space bound: max_pct_adv·ADVᵢ / NAV.
    bounds = cap * adv[idx] / nav
    dw = w[idx] - w_prev[idx]
    constraints.append(dw <= bounds)
    constraints.append(-dw <= bounds)
    return {
        "max_pct_adv_applied": True,
        "max_pct_adv": cap,
        "max_pct_adv_n_names_constrained": int(len(idx)),
        "max_pct_adv_min_bound_weight": float(np.min(bounds)) if len(bounds) else None,
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
    covariance_provided: bool = False,
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
    # ``returns_panel`` is only required when estimating Σ from history. In the
    # intraday re-solve path the caller supplies a precomputed covariance and
    # may pass returns_panel=None.
    if not covariance_provided:
        if returns_panel is None:
            raise ValueError("returns_panel is required when covariance is not provided")
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


def _validate_covariance(cov: np.ndarray, N: int) -> np.ndarray:
    """Validate + symmetrize a precomputed DAILY covariance for the re-solve.

    JSON round-tripping can introduce tiny asymmetry / non-PSD perturbations.
    Symmetrize (0.5·(Σ+Σᵀ)) and fail LOUD on shape mismatch, non-finite
    entries, or a materially negative eigenvalue — a silently mis-shaped Σ
    would corrupt the entire vol math (vol-target SOC + every vol diagnostic).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (N, N):
        raise ValueError(f"covariance shape {cov.shape} != ({N}, {N})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite entries")
    cov = 0.5 * (cov + cov.T)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    tol = -1e-8 * max(1.0, float(np.trace(cov)) / N)
    if min_eig < tol:
        raise ValueError(
            f"covariance is not PSD (min eigenvalue {min_eig:.3e} < tol {tol:.3e})"
        )
    return cov


def _estimate_covariance_daily(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    """Estimate the DAILY covariance Σ_daily (pre-horizon-scaling).

    Split out of ``_estimate_covariance`` so the optimizer shadow log can
    persist Σ_daily for an intraday re-solve (see ``solve_target_weights``'s
    ``covariance`` argument). Horizon scaling lives in ``_estimate_covariance``.

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

    return sigma_daily


def _estimate_covariance(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    """Return covariance at horizon ``cfg["sigma_horizon_days"]``.

    Estimates Σ_daily via ``_estimate_covariance_daily``, then scales by
    horizon-days under i.i.d. log-return assumption: Σ_H = H · Σ_daily.
    Default H=1 preserves legacy daily Σ bit-identical (1 × Σ = Σ).
    """
    sigma_daily = _estimate_covariance_daily(returns_panel, cfg)
    horizon = int(cfg.get("sigma_horizon_days", 1))
    if horizon < 1:
        raise ValueError(f"sigma_horizon_days must be ≥ 1; got {horizon}")
    return horizon * sigma_daily


def _real_sectors(sectors: list[str]) -> set[str]:
    return {s for s in sectors if not (s.startswith("__") and s.endswith("__"))}


# Feasibility slack the post-solve assertion grants the SOLVER's own vector,
# by solve status. An `optimal` solve from an interior-point method satisfies
# its constraints to ~1e-9; `optimal_inaccurate` means the solver stopped
# before its convergence criteria were met and makes no such promise — its
# residuals are routinely 1e-3-scale. Granting both the same 1e-6 was the
# defect behind alpha-engine-config-I11368: a genuinely uncertified point was
# read as a solver bug and held the whole book.
#
# These are declared tolerances, not tuned ones. Anything past them is still
# unexplained and still raises.
# Slack granted above the mandatory turnover floor when the floor sets the cap.
# Declared, and sized to the loosest solver in the fallback chain rather than
# picked: SCS converges to ~1e-4, so a set narrower than that is one no solver
# in the chain can be asked to land inside.
_TURNOVER_FLOOR_REL_SLACK = 1e-3
_TURNOVER_FLOOR_ABS_SLACK = 1e-4

# Keyed on (solver, status), NOT status alone. `optimal` is the solver's own
# verdict against its own convergence criteria, and those criteria differ by
# two orders of magnitude across this chain: Clarabel is an interior-point
# method converging to ~1e-9, SCS is a first-order ADMM method whose DEFAULT
# eps_abs/eps_rel is 1e-4 — an SCS `optimal` routinely carries 1e-4-scale
# primal residuals and is not lying when it does.
#
# This is exactly the 2026-09-22 path (alpha-engine-config-I11368): Clarabel
# returned `infeasible`, SCS picked it up, and its point sat 9.4e-4 outside a
# constraint the assertion was holding to 1e-6. A status-only tolerance would
# still have held the book.
#
# Declared tolerances, not tuned ones. Anything past them is still unexplained
# and still raises.
_SOLVER_FEASIBILITY_TOL = {
    ("CLARABEL", "optimal"): 1e-6,
    ("CLARABEL", "optimal_inaccurate"): 1e-3,
    ("SCS", "optimal"): 2e-3,
    ("SCS", "optimal_inaccurate"): 5e-3,
    ("OSQP", "optimal"): 2e-3,
    ("OSQP", "optimal_inaccurate"): 5e-3,
}
# An unknown solver is held to the strictest value, so adding one to the chain
# cannot silently widen what this module will trade.
_DEFAULT_SOLVER_FEASIBILITY_TOL = 1e-6


def _solver_feasibility_tol(solver_name: str | None, status: str | None) -> float:
    return _SOLVER_FEASIBILITY_TOL.get(
        (solver_name or "", status or ""), _DEFAULT_SOLVER_FEASIBILITY_TOL
    )


def _solve_with_fallback(problem, w, cfg: dict):
    """Solve, preferring a CERTIFIED `optimal` point over an inaccurate one.

    Historically this returned the first solver whose status was `optimal`
    OR `optimal_inaccurate`, so a Clarabel run that stopped early
    short-circuited SCS and OSQP — both of which might have converged. With a
    hard L1 turnover budget in the program that is not a cosmetic difference:
    an inaccurate point can violate the budget by 1e-3 of NAV, and the
    downstream assertion then cannot tell "the solver gave up" from "this
    module has a bug" (alpha-engine-config-I11368).

    So: take the first `optimal`. Remember the first `optimal_inaccurate` and
    fall back to it only when NO solver converges — degrading to an
    uncertified point is still better than holding the book on a day the
    program is perfectly feasible, provided the status travels with it so the
    assertion and the operator surface can both account for it.

    Returns ``(weights, status, solver_name)``.
    """
    import cvxpy as cp
    inaccurate: tuple[np.ndarray, str, str] | None = None
    last_status: str | None = None
    for solver in (_CLARABEL, *_FALLBACK_SOLVERS):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except (cp.error.SolverError, ValueError) as e:
            logger.warning(f"Solver {solver} raised {e!r}, trying next")
            continue
        last_status = problem.status
        if problem.status == "optimal":
            return np.asarray(w.value, dtype=float), problem.status, solver
        if problem.status == "optimal_inaccurate" and inaccurate is None:
            # Snapshot it: a later solver's solve() overwrites w.value.
            inaccurate = (np.asarray(w.value, dtype=float), problem.status, solver)
            logger.warning(
                f"Solver {solver} returned optimal_inaccurate; trying the next "
                f"solver for a certified point before accepting it"
            )
            continue
        logger.warning(
            f"Solver {solver} returned status={problem.status}, trying next"
        )
    if inaccurate is not None:
        logger.warning(
            "No solver converged; accepting the %s optimal_inaccurate point "
            "with a widened feasibility tolerance (I11368)", inaccurate[2],
        )
        return inaccurate
    return None, last_status if last_status else "no_solver_available", None


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
) -> tuple[np.ndarray, dict]:
    """Clip to the box, drop sub-``min_position_pct`` dust, renormalize.

    Returns ``(weights, clip_meta)``.

    This step does THREE things to the solver's vector, not one: it clamps
    negatives to zero, it clips down to the box, and only then does it drop
    dust and renormalize. The previous contract reported ``mass_zeroed`` —
    the dust alone — and ``_apply_turnover_governor`` used it as its whole
    tolerance. That accounting was the defect behind
    alpha-engine-config-I11368: the negative clamp and the box clip both move
    the vector, the renormalize then rescales EVERY name by the total mass
    all three steps moved, and none of that appeared in the tolerance. On
    2026-09-22 it left 9.4e-4 of one-way turnover unexplained and held the
    entire book.

    So this now reports what actually happened rather than a proxy for it:

      * ``clip_turnover_one_way`` — ``‖w_after − w_before‖₁ / 2``, the EXACT
        one-way turnover this step introduced. Not a bound. With it the
        caller can assert the SOLVER's vector against the budget directly,
        which is the only quantity the convex constraint ever governed.
      * ``clip_negative_mass`` / ``clip_box_excess`` / ``clip_mass_zeroed`` —
        the three mutations separately, so a future breach names its stage
        instead of being attributed to whichever one happens to be measured.
      * ``clip_renorm_scale`` — the ``1/total`` actually applied.

    ``clip_mass_zeroed`` keeps its old name and meaning for every existing
    reader.
    """
    before = np.asarray(weights, dtype=float).copy()
    weights = np.maximum(weights, 0.0)
    negative_mass = float(np.sum(np.abs(before[before < 0.0])))
    upper = effective_caps + 1e-8
    box_excess = float(np.sum(np.maximum(weights - upper, 0.0)))
    weights = np.minimum(weights, upper)
    small = (weights < cfg["min_position_pct"]) & (np.arange(len(weights)) != cash_idx)
    mass_zeroed = float(np.sum(weights[small]))
    weights = np.where(small, 0.0, weights)
    total = float(weights.sum())
    renorm_scale = 1.0
    if total > 0:
        renorm_scale = 1.0 / total
        weights = weights / total
    clip_meta = {
        "clip_mass_zeroed": mass_zeroed,
        "clip_negative_mass": negative_mass,
        "clip_box_excess": box_excess,
        "clip_renorm_scale": renorm_scale,
        "clip_turnover_one_way": float(np.sum(np.abs(weights - before)) / 2),
    }
    return weights, clip_meta


def _mandatory_turnover_floor(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> float:
    """One-way turnover that the OTHER constraints force, regardless of alpha.

    Adding ``‖w − w_prev‖₁/2 ≤ cap`` can only be safe if the feasible set is
    still non-empty. ``w_prev`` has turnover 0 and so always satisfies the
    turnover constraint itself — but it does not necessarily satisfy the rest
    of the program: a held name that went ineligible is pinned to 0, and the
    cash sleeve is pinned to ``cash_sleeve_pct``. Those pins MANDATE movement.
    If that mandated movement exceeds the budget, the program is infeasible
    and the whole book falls to the hold path — a new failure mode, introduced
    by the fix, on a day when a forced exit is exactly what must happen.

    So the budget governs DISCRETIONARY trading only. This returns a bound on
    the forced movement; the caller raises the constraint's right-hand side to
    it when it is larger than the configured cap, which makes the feasible set
    non-empty BY CONSTRUCTION rather than by assumption:

      * ``d_i`` = distance from ``w_prev_i`` to its box ``[l_i, u_i]``. The
        projection ``p`` of ``w_prev`` into the box costs ``Σ d_i`` of L1.
      * ``p`` need not sum to 1; the residual ``r = 1 − Σp`` must be absorbed
        by names with slack (SPY, the unconstrained benchmark fill, always
        has some), costing a further ``|r|`` of L1.

    ``(Σd + |r|) / 2`` is therefore an attainable one-way turnover for a point
    satisfying the box, the sleeve pin and the budget identity. Sector and
    participation caps can only be satisfied at that point or need more
    movement; they are inequality caps on sums that the projection does not
    tighten, so this is the operative bound in practice, and any residual
    infeasibility still lands on the pre-existing ``_fallback_weights`` path
    rather than anywhere new.
    """
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    sleeve = float(cfg["cash_sleeve_pct"])
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    forced_l1 = float(np.sum(np.abs(projected - w_prev)))
    residual = abs(1.0 - float(projected.sum()))
    return (forced_l1 + residual) / 2.0


def _decompose_mandatory_turnover(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    eligibility: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> dict:
    """Which constraint forced each unit of the mandatory turnover floor.

    alpha-engine-config-I8753. ``turnover_mandatory_floor`` has been emitted
    since the turnover constraint shipped, and it answers "how much of today's
    trading was forced" — but not BY WHAT, and the three causes have entirely
    different fixes:

    * ``cash_sleeve_pin``   — the sleeve is an equality pin; drift into or out
      of it is mandated every session and is not a defect.
    * ``ineligibility_pin`` — a held name went ineligible (research EXIT, GBM
      veto, score gate) and is pinned to zero. A forced exit is the system
      working.
    * ``position_cap``      — a held name sits above its per-name cap
      ``max_position_pct × stance_multiplier``. The stance is re-derived every
      morning with no persistence, so a name whose stance flips
      momentum→quality must be cut from 10% of NAV to 4% THAT MORNING,
      regardless of its alpha, and rebuilt when the stance flips back.

    Measured 2026-08-27 across the nine sessions since the v2 cutover
    (``predictor/optimizer_shadow/{date}.json``, read-only): cap cuts on held
    names mandated **0.278 of NAV in one-way selling**, ~3.1% per session,
    entirely alpha-independent. On 2026-08-27 the floor was 0.105 against an
    executed one-way turnover of 0.170 — 62% of the day's trading forced before
    the objective was consulted.

    That total was legible only by reading nine artifacts and diffing
    ``stance_caps`` by hand, which is what this function removes. Whether the
    right fix is hold-side stance hysteresis, a different cap schedule, or
    nothing at all is NOT decided here — it is decided by watching this
    decomposition accrue. Nine sessions gave only four cap flips with a full
    three-session look-ahead (two reverted), which is too thin to set a
    hysteresis default on.

    Emitted on every solve, healthy included: a component emitting nothing is
    not healthy, it is unobserved.

    Returns per-cause one-way turnover plus ``total``, which reconciles to
    ``_mandatory_turnover_floor`` to within floating-point tolerance. The
    ``renormalization`` term is the residual the projection leaves behind —
    the mass that must be absorbed by names with slack once every pin is
    satisfied — and it belongs to no single name.
    """
    sleeve = float(cfg["cash_sleeve_pct"])
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    per_name = np.abs(projected - w_prev)

    idx = np.arange(len(w_prev))
    is_cash = idx == cash_idx
    # An ineligible name carries an effective cap of 0, so its whole holding is
    # forced out. Attributing that to "position_cap" would read as a sizing
    # artifact when it is a deliberate exit — the two have opposite responses.
    is_ineligible = (~np.asarray(eligibility, dtype=bool)) & (~is_cash)

    sleeve_l1 = float(per_name[is_cash].sum())
    ineligible_l1 = float(per_name[is_ineligible].sum())
    cap_l1 = float(per_name[~is_cash & ~is_ineligible].sum())
    residual = abs(1.0 - float(projected.sum()))

    names_at_cap = [
        int(i) for i in idx
        if not is_cash[i] and not is_ineligible[i] and per_name[i] > 1e-9
    ]
    names_pinned_out = [int(i) for i in idx if is_ineligible[i] and per_name[i] > 1e-9]

    return {
        "cash_sleeve_pin": sleeve_l1 / 2.0,
        "ineligibility_pin": ineligible_l1 / 2.0,
        "position_cap": cap_l1 / 2.0,
        "renormalization": residual / 2.0,
        "total": (sleeve_l1 + ineligible_l1 + cap_l1 + residual) / 2.0,
        "n_names_over_cap": len(names_at_cap),
        "n_names_pinned_out": len(names_pinned_out),
    }


def compute_conviction_budget_multiplier(
    alpha_hat: np.ndarray,
    alpha_uncertainty: np.ndarray | None,
    eligibility: np.ndarray | None,
    spy_idx: int,
    cash_idx: int,
    cfg: dict,
) -> dict:
    """Signal-quality multiplier on the DISCRETIONARY daily turnover budget.

    Returns a block that is emitted on EVERY solve — gate on, gate off, gate
    inevaluable — because a field that appears only when the gate engages is
    indistinguishable from a dead gate. Keys:

    ``conviction_gate_applied``  bool — the multiplier actually scaled the budget
    ``conviction_ir_xs``         float|None — stdev_xs(α̂) / median(σ_α̂)
    ``conviction_alpha_dispersion`` float|None — the numerator, on its own
    ``conviction_alpha_noise``   float|None — the denominator, on its own
    ``conviction_n_names``       int — discretionary names the statistic used
    ``conviction_budget_multiplier`` float — q ∈ [min_multiple, 1.0]
    ``conviction_gate_reason``   str — why q is what it is, always populated

    Never raises: every degradation path returns q = 1.0 (the UNTHROTTLED
    budget) with a reason. Missing data must never produce a TIGHTER budget
    than the operator configured — a gate that silently stops the book on an
    input outage is a worse failure than the churn it exists to stop.
    """
    out: dict = {
        "conviction_gate_applied": False,
        "conviction_ir_xs": None,
        "conviction_alpha_dispersion": None,
        "conviction_alpha_noise": None,
        "conviction_n_names": 0,
        "conviction_budget_multiplier": 1.0,
        "conviction_gate_reason": "disabled",
    }
    if not cfg.get("conviction_budget_gate_enabled", True):
        return out
    if alpha_uncertainty is None:
        out["conviction_gate_reason"] = "no_alpha_uncertainty_vector"
        return out

    alpha = np.asarray(alpha_hat, dtype=float)
    sigma = np.asarray(alpha_uncertainty, dtype=float)
    n = alpha.shape[0]
    if sigma.shape[0] != n:
        out["conviction_gate_reason"] = "alpha_uncertainty_length_mismatch"
        return out

    # Discretionary names only: SPY is the benchmark fill and CASH the sleeve —
    # neither carries a predicted alpha, and both would drag the dispersion
    # toward a number that says nothing about the ranking being traded on.
    mask = np.ones(n, dtype=bool)
    for i in (spy_idx, cash_idx):
        if 0 <= i < n:
            mask[i] = False
    if eligibility is not None:
        elig = np.asarray(eligibility, dtype=bool)
        if elig.shape[0] == n:
            mask &= elig
    mask &= np.isfinite(alpha)

    min_names = int(cfg.get("conviction_gate_min_names", 3))
    if int(mask.sum()) < max(2, min_names):
        out["conviction_n_names"] = int(mask.sum())
        out["conviction_gate_reason"] = "too_few_discretionary_names"
        return out

    sig_ok = mask & np.isfinite(sigma) & (sigma > 0)
    if int(sig_ok.sum()) < max(2, min_names):
        out["conviction_n_names"] = int(mask.sum())
        out["conviction_gate_reason"] = "no_usable_alpha_uncertainty"
        return out

    dispersion = float(np.std(alpha[mask]))
    noise = float(np.median(sigma[sig_ok]))
    out["conviction_alpha_dispersion"] = dispersion
    out["conviction_alpha_noise"] = noise
    out["conviction_n_names"] = int(mask.sum())
    if not np.isfinite(dispersion) or not np.isfinite(noise) or noise <= 0:
        out["conviction_gate_reason"] = "non_finite_statistic"
        return out

    ir = dispersion / noise
    out["conviction_ir_xs"] = float(ir)

    lo = float(cfg.get("conviction_ir_floor", 0.35))
    hi = float(cfg.get("conviction_ir_full", 0.75))
    q_min = float(cfg.get("conviction_budget_min_multiple", 0.05))
    q_min = min(max(q_min, 0.0), 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # A misconfigured band must not silently become an arbitrary throttle.
        out["conviction_gate_reason"] = "invalid_ir_band"
        return out

    q = (ir - lo) / (hi - lo)
    q = float(min(max(q, q_min), 1.0))
    out["conviction_budget_multiplier"] = q
    out["conviction_gate_applied"] = bool(q < 1.0)
    out["conviction_gate_reason"] = (
        "signal_quality_ok" if q >= 1.0 else "alpha_spread_below_own_noise"
    )
    return out


def _solve_min_attainable_turnover(cp, w, w_prev: np.ndarray, constraints: list):
    """The EXACT minimum one-way turnover the non-budget constraints force.

    ``_mandatory_turnover_floor`` estimates this by projecting ``w_prev`` into
    the box and absorbing the residual. Its own docstring concedes the gap:
    "Sector and participation caps can only be satisfied at that point or need
    more movement". On 2026-09-22 they needed more movement
    (alpha-engine-config-I11368). The projection said 0.016209 was attainable,
    the true minimum was above it, and the budget constraint pinned at the
    projection bound made the whole program INFEASIBLE — Clarabel said so, in
    the log, and SCS then returned a near-feasible point 9.4e-4 outside the
    constraint, which the post-solve assertion correctly refused to trade.
    The book was held for a day because a bound was mistaken for an optimum.

    So stop estimating it. Minimising ``‖w − w_prev‖₁ / 2`` subject to every
    constraint EXCEPT the turnover budget is a small LP over a feasible set we
    already know is non-empty, and its optimum is the answer exactly. One
    extra solve, once per session, against a day held.

    Returns ``(value, status)``; ``(None, status)`` when it does not solve, in
    which case the caller keeps the projection bound and the pre-existing
    infeasible-fallback path still catches whatever follows.
    """
    try:
        problem = cp.Problem(cp.Minimize(cp.norm(w - w_prev, 1) / 2), constraints)
        value, status, _ = _solve_with_fallback(problem, w, {})
    except Exception as e:  # noqa: BLE001 - see below
        # Never let the floor probe break a solve that would otherwise run:
        # its whole job is to WIDEN a cap, so falling back to the projection
        # bound is the pre-existing behaviour, not a degraded one. Recorded at
        # WARNING and carried in `turnover_floor_probe_status`.
        logger.warning("min-attainable-turnover probe raised %r; keeping the "
                       "projection bound", e)
        return None, f"raised:{type(e).__name__}"
    if value is None:
        logger.warning(
            "min-attainable-turnover probe did not solve (status=%s); keeping "
            "the projection bound", status,
        )
        return None, status
    return float(np.sum(np.abs(value - w_prev)) / 2), status


def _apply_turnover_constraint(
    cp,
    w,
    w_prev: np.ndarray,
    constraints: list,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
    eligibility: np.ndarray | None = None,
    *,
    alpha_hat: np.ndarray | None = None,
    alpha_uncertainty: np.ndarray | None = None,
    spy_idx: int | None = None,
    min_attainable_turnover: float | None = None,
    min_attainable_status: str | None = None,
) -> dict:
    """Append the L1 daily-turnover budget to ``constraints``.

    Returns metadata the diagnostics and the post-solve assertion both read:
    the configured cap, the mandatory floor, and the EFFECTIVE right-hand side
    actually imposed. ``max_daily_turnover=None`` disables the budget entirely
    (legacy behaviour, bit-identical).
    """
    cap = cfg.get("max_daily_turnover")
    # The conviction block is computed and emitted even when the budget is OFF:
    # "how good was today's signal" is an operator fact in its own right, and a
    # statistic that only exists on the throttled path cannot be used to judge
    # whether the throttle was right.
    conviction = (
        compute_conviction_budget_multiplier(
            alpha_hat, alpha_uncertainty, eligibility,
            spy_idx if spy_idx is not None else -1, cash_idx, cfg,
        )
        if alpha_hat is not None
        else {
            "conviction_gate_applied": False,
            "conviction_ir_xs": None,
            "conviction_alpha_dispersion": None,
            "conviction_alpha_noise": None,
            "conviction_n_names": 0,
            "conviction_budget_multiplier": 1.0,
            "conviction_gate_reason": "no_alpha_vector_supplied",
        }
    )
    meta: dict = {
        "turnover_constraint_applied": False,
        "turnover_constraint_cap": None,
        "turnover_budget_configured": None if cap is None else float(cap),
        "turnover_budget_discretionary": None,
        "turnover_mandatory_floor": None,
        "turnover_mandatory_floor_by_cause": None,
        "turnover_constraint": None,
        **conviction,
    }
    if cap is None or cap <= 0:
        return meta
    floor = _mandatory_turnover_floor(w_prev, effective_caps, cash_idx, cfg)
    # alpha-engine-config-I8753 — WHICH constraint forced each unit of it.
    # `eligibility` is optional so every existing caller keeps working; without
    # it the ineligibility pin cannot be separated from the position cap, and
    # the decomposition says so rather than guessing.
    by_cause = (
        _decompose_mandatory_turnover(
            w_prev, effective_caps, eligibility, cash_idx, cfg,
        )
        if eligibility is not None
        else None
    )
    # A small slack above the floor: the floor is a bound on an attainable
    # point, and pinning the RHS exactly to it would leave a feasible set of
    # measure ~zero that an interior-point solver reports as infeasible.
    # The conviction gate scales the DISCRETIONARY budget only. The mandatory
    # floor is applied AFTER it, so a throttled budget can never starve a
    # forced exit — the two are different kinds of trading and the `max` below
    # keeps them ordered correctly.
    q = float(conviction.get("conviction_budget_multiplier", 1.0))
    discretionary = float(cap) * q
    meta["turnover_budget_discretionary"] = discretionary
    # The floor that goes into the cap is the LARGER of the projection bound
    # and the LP's exact minimum, when the LP solved. The projection ignores
    # the sector and %ADV caps and can therefore under-state what is forced
    # (alpha-engine-config-I11368).
    binding_floor = float(floor)
    if min_attainable_turnover is not None:
        binding_floor = max(binding_floor, float(min_attainable_turnover))
    # The slack above the floor exists so an interior-point solver is not
    # handed a feasible set of measure ~zero. `(1 + 1e-6)` was that slack and
    # it is 1.6e-8 of weight on a 0.0162 floor — indistinguishable from
    # pinning the constraint exactly, which is how 2026-09-22 reached
    # Clarabel as `infeasible`. Sized now to the convergence tolerance of the
    # loosest solver in the fallback chain (SCS, eps ~1e-4), which is what
    # actually has to land inside the set.
    effective_cap = max(
        discretionary, binding_floor * (1.0 + _TURNOVER_FLOOR_REL_SLACK)
        + _TURNOVER_FLOOR_ABS_SLACK,
    )
    constraint = cp.norm(w - w_prev, 1) / 2 <= effective_cap
    constraints.append(constraint)
    meta.update({
        "turnover_constraint_applied": True,
        "turnover_constraint_cap": float(effective_cap),
        "turnover_mandatory_floor": float(floor),
        "turnover_mandatory_floor_by_cause": by_cause,
        "turnover_min_attainable": (
            None if min_attainable_turnover is None
            else float(min_attainable_turnover)
        ),
        "turnover_floor_probe_status": min_attainable_status,
        # Which term set the cap. A floor-bound cap means the day's movement
        # is FORCED, not discretionary — a materially different operator fact
        # from a budget the conviction gate throttled.
        "turnover_cap_source": (
            "discretionary" if discretionary >= effective_cap else "mandatory_floor"
        ),
        "turnover_constraint": constraint,
    })
    if q < 1.0:
        logger.warning(
            "conviction gate THROTTLED the discretionary turnover budget from "
            "%.4f to %.4f (multiplier %.3f): the cross-sectional alpha spread "
            "is %.5f against a median per-name sigma_alpha of %.5f, i.e. an "
            "information ratio of %.3f over %d discretionary names, below the "
            "%.2f floor. The names the optimizer is ranking are not "
            "statistically distinguishable from one another; rebalancing "
            "between them is a transaction cost, not a trade. Mandatory "
            "turnover (%.4f) is unaffected and still executes.",
            float(cap), discretionary, q,
            conviction.get("conviction_alpha_dispersion") or float("nan"),
            conviction.get("conviction_alpha_noise") or float("nan"),
            conviction.get("conviction_ir_xs") or float("nan"),
            conviction.get("conviction_n_names") or 0,
            float(cfg.get("conviction_ir_floor", 0.35)),
            float(floor),
        )
    if effective_cap > float(cap) + 1e-9:
        logger.warning(
            "turnover budget RAISED from the configured %.4f to %.4f: the "
            "eligibility mask and cash-sleeve pin mandate %.4f one-way "
            "turnover on their own. The budget governs discretionary trading; "
            "a forced exit is not discretionary and must not be starved of "
            "budget.",
            float(cap), effective_cap, floor,
        )
    return meta


# Complementary-slackness tolerances for the turnover-budget binding test.
# _DUAL_ACTIVE_TOL sits six orders of magnitude above the numerical dust an
# interior-point solver returns for an INACTIVE constraint (measured 1.6e-8 …
# 4.2e-9 on the live book) and four below the smallest genuine active dual in
# the same window (5.4e-3). _PRIMAL_AT_BOUND_REL_TOL is wide enough to admit
# the post-clip-and-renormalize drift (measured 0.19978 against a 0.2000 cap).
_DUAL_ACTIVE_TOL = 1e-6
_PRIMAL_AT_BOUND_REL_TOL = 5e-3


def _turnover_diagnostics(
    weights: np.ndarray, w_prev: np.ndarray, turnover_meta: dict
) -> dict:
    """Turnover observability, emitted on EVERY solve including the fallback
    and the budget-disabled path — a field that appears only on the
    interesting path is indistinguishable from a dead emitter.

    ``requested_turnover_one_way`` deliberately equals the EXECUTED one-way
    turnover now. Under the old post-solve shrink the two differed, and the
    gap was the only published evidence that the optimizer wanted to move
    more than it was allowed to. That evidence has not been dropped, it has
    moved to a better instrument: ``turnover_constraint_binding`` says the
    budget bound the solve, and ``turnover_constraint_shadow_price`` is the
    constraint's dual — the marginal objective value of one more unit of
    turnover budget, i.e. exactly what the restraint cost.
    """
    executed = float(np.sum(np.abs(weights - w_prev)) / 2)
    cap = turnover_meta.get("turnover_constraint_cap")
    out: dict = {
        "requested_turnover_one_way": executed,
        "turnover_constraint_applied": bool(
            turnover_meta.get("turnover_constraint_applied")
        ),
        "turnover_constraint_cap": cap,
        "turnover_mandatory_floor": turnover_meta.get("turnover_mandatory_floor"),
        # alpha-engine-config-I8753 — the floor split by the constraint that
        # forced it. Emitted on every solve, healthy included: "how much of
        # today's trading was forced, and by what" was previously answerable
        # only by reading nine artifacts and diffing stance_caps by hand.
        "turnover_mandatory_floor_by_cause": turnover_meta.get(
            "turnover_mandatory_floor_by_cause"
        ),
        "turnover_constraint_binding": False,
        "turnover_constraint_shadow_price": None,
        # Conviction gate (I9315) — emitted on every solve, throttled or not.
        "turnover_budget_configured": turnover_meta.get("turnover_budget_configured"),
        "turnover_budget_discretionary": turnover_meta.get(
            "turnover_budget_discretionary"
        ),
        "conviction_gate_applied": turnover_meta.get("conviction_gate_applied", False),
        "conviction_ir_xs": turnover_meta.get("conviction_ir_xs"),
        "conviction_alpha_dispersion": turnover_meta.get("conviction_alpha_dispersion"),
        "conviction_alpha_noise": turnover_meta.get("conviction_alpha_noise"),
        "conviction_n_names": turnover_meta.get("conviction_n_names", 0),
        "conviction_budget_multiplier": turnover_meta.get(
            "conviction_budget_multiplier", 1.0
        ),
        "conviction_gate_reason": turnover_meta.get("conviction_gate_reason"),
    }
    if cap is not None:
        constraint = turnover_meta.get("turnover_constraint")
        dual = getattr(constraint, "dual_value", None) if constraint is not None else None
        if dual is not None:
            try:
                out["turnover_constraint_shadow_price"] = float(np.ravel(dual)[0])
            except (TypeError, ValueError, IndexError):
                out["turnover_constraint_shadow_price"] = None
        # COMPLEMENTARY SLACKNESS, both halves (I9315). A constraint is active
        # iff its dual is strictly positive AND its primal sits at the bound.
        # Testing only one half is wrong in a different direction each way:
        #
        #  * Primal only — `weights` here is post-clip-and-renormalize, so its
        #    turnover is a few bp off the solver's own. Measured 2026-08-14 on
        #    the live book: solver at the 0.2000 cap, post-clip 0.19978, which a
        #    ">= cap · (1 − 1e-3)" test reads as NOT binding while the dual is
        #    0.00926. The relative tolerance below is widened to 5e-3 so that
        #    real case stays inside it.
        #
        #  * Dual only — an interior-point solver returns numerical dust for an
        #    INACTIVE constraint, and a 1e-9 threshold is far below that dust.
        #    Measured on the live book: 2026-08-21 dual 4.2e-9 with executed
        #    turnover 14.98%, 2026-08-27 dual 2.4e-8 at 16.99%, 2026-08-28 dual
        #    1.6e-8 at 12.27% — all against a 20% cap with up to 7.7 points of
        #    it unspent. All three were reported binding and all three fired an
        #    operator alert stating that the budget "BOUND the solve". A
        #    detector that says the opposite of the truth on 3 of 12 sessions is
        #    worse than no detector.
        #
        # Requiring both halves accepts the 2026-08-14 case and rejects all
        # three false positives. The dual tolerance is 1e-6 — six orders of
        # magnitude above the observed dust and four below the smallest genuine
        # dual measured in the same window (5.4e-3).
        shadow = out["turnover_constraint_shadow_price"]
        at_bound = bool(executed >= float(cap) * (1.0 - _PRIMAL_AT_BOUND_REL_TOL) - 1e-6)
        if shadow is not None:
            out["turnover_constraint_binding"] = bool(
                shadow > _DUAL_ACTIVE_TOL and at_bound
            )
        else:
            # No dual available (fallback solver / non-optimal status): the
            # primal test is all there is. Recorded so the ambiguity is visible.
            out["turnover_constraint_binding"] = at_bound
            out["turnover_binding_test"] = "primal_only_no_dual"
        out.setdefault("turnover_binding_test", "complementary_slackness")
    # ``turnover_capped`` keeps its consumer-facing meaning — "the daily
    # turnover budget bound this solve" — which is now the binding test
    # rather than "a post-hoc shrink was applied".
    out["turnover_capped"] = out["turnover_constraint_binding"]
    return out


class TurnoverBudgetError(RuntimeError):
    """The solved weight vector exceeds the daily turnover budget.

    Raised, not silently corrected. Under the constraint construction
    (alpha-engine-config-I7346) the budget is enforced INSIDE the convex
    program, so a vector that violates it means the solver returned a point
    that does not satisfy a constraint it was given, or the post-solve
    clip/renormalize moved it further than that step can account for. Both
    are bugs in this module, not market conditions.

    RAISE rather than degrade, deliberately. ``run_shadow_optimizer`` catches
    it, writes the ``shadow_status: "failed"`` sentinel, and
    ``optimizer_cutover.is_log_usable`` then returns False, so the planner
    falls back to an EMPTY order book: the book is HELD for the session. The
    alternative — shrinking the vector back under the budget — is precisely
    the mechanism this change removes, and it would restore the failure it
    fixed while now also hiding a solver bug behind a plausible-looking
    order book. Holding a book for one session is recoverable and loud;
    trading a book the optimizer did not sanction is neither.
    """


def _apply_turnover_governor(
    weights: np.ndarray,
    w_prev: np.ndarray,
    cfg: dict,
    *,
    turnover_meta: dict | None = None,
    clip_meta: dict | None = None,
    solver_status: str | None = None,
    solver_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Post-solve ASSERTION that the daily turnover budget held.

    Historically this function ENFORCED the budget, by scaling the whole step
    ``w_prev → weights`` uniformly by ``cap / requested``. That is no longer
    its job: the budget is a constraint inside the convex program
    (``_apply_turnover_constraint``), so the solver returns a vector that
    already satisfies it. This function's role is inverted — it measures and
    raises, and NEVER modifies ``weights``.

    **It asserts the SOLVER's vector, not the clipped one**
    (alpha-engine-config-I11368). The convex constraint governed exactly one
    quantity: ``‖w_solved − w_prev‖₁ / 2``. ``_clip_and_renormalize`` runs
    afterwards and moves the vector further by an amount it now MEASURES
    (``clip_turnover_one_way``), so the pre-clip turnover is recoverable
    exactly and there is no reason to test a composite and then try to
    subtract a proxy for one of its parts. The previous form tested the
    executed vector against ``cap + clip_mass_zeroed``, which counted only
    the dust drop and silently ignored the negative clamp, the box clip and
    the renormalize — and on 2026-09-22 it charged 9.4e-4 of solver
    inaccuracy to this module and held the whole book.

    Two independent assertions now, each naming one stage:

      * **solver** — the solved point must satisfy the budget to within the
        feasibility tolerance its own SOLVER and status earn — see
        ``_SOLVER_FEASIBILITY_TOL``, which is keyed on both because an SCS
        ``optimal`` and a Clarabel ``optimal`` differ by three orders of
        magnitude. Past that, the solver returned a point violating a
        constraint it was given.
      * **clip** — the post-solve step's measured displacement must not
        exceed what its own three mutations can account for. Past that, the
        accounting in ``_clip_and_renormalize`` is wrong.

    Both raise ``TurnoverBudgetError`` rather than degrading, per I7346. The
    message names which assertion fired, so the sentinel is diagnostic on its
    own.

    ``max_daily_turnover=None`` disables the budget; the function then only
    reports, exactly as before.
    """
    meta = turnover_meta or {}
    requested = float(np.sum(np.abs(weights - w_prev)) / 2)
    flag = cfg.get("large_move_turnover_flag")
    gov = _turnover_diagnostics(weights, w_prev, meta)
    binding = bool(gov["turnover_constraint_binding"])
    above_flag = bool(flag is not None and requested > flag)
    gov["large_move_flagged"] = bool(above_flag or binding)
    # Why it was flagged, so the operator alert can say the true thing. Under
    # the constraint construction the executed turnover can no longer exceed
    # the flag on a capped day, so a flag driven ONLY by the raw comparison
    # would go permanently silent — a detector killed by a fix is a worse
    # outcome than the fix is good. `binding` is the honest successor signal:
    # the optimizer wanted to move more than the budget allowed.
    #
    # When the conviction gate is throttling, a binding budget is the guard
    # DOING ITS JOB, not a large move: the optimizer wants to churn a book of
    # statistically tied names and the gate is refusing. Flagging that would
    # reproduce the exact alert this arc exists to end — one that fires every
    # session on a healthy state and asks a human to approve something no
    # human has a channel to approve. The fact is not lost: the whole
    # conviction block is in the diagnostics and the shadow artifact, and the
    # rolling tripwire reads it and reports the throttle as the DRIVER.
    gate_on = bool(gov.get("conviction_gate_applied"))
    if above_flag:
        gov["large_move_reason"] = "executed_turnover_above_flag"
    elif binding and gate_on:
        gov["large_move_reason"] = "conviction_throttled_budget_binding"
        gov["large_move_flagged"] = bool(above_flag)
    elif binding:
        gov["large_move_reason"] = "turnover_budget_binding"
    else:
        gov["large_move_reason"] = None

    clip = clip_meta or {}
    clip_turnover = float(clip.get("clip_turnover_one_way", 0.0))
    # ‖w_solved − w_prev‖₁/2 recovered exactly: the clip's measured
    # displacement is the only thing between the solved point and this one.
    # The triangle inequality makes this a BOUND on the solved turnover
    # (|a| ≥ |b| − |a−b|), which is the direction the assertion needs — it can
    # never understate a real solver breach.
    pre_clip = max(requested - clip_turnover, 0.0)
    solver_tol = _solver_feasibility_tol(solver_name, solver_status)
    # The clip may legitimately displace at most what its own three mutations
    # move: the clamped negatives, the mass shaved off the box, the dust
    # dropped, and the renormalize that redistributes all of it (≤ the same
    # again). `_clip_and_renormalize` measures each; this is the sum, not an
    # epsilon, and 1e-9 covers float accumulation over N names only.
    clip_budget = 2.0 * (
        float(clip.get("clip_negative_mass", 0.0))
        + float(clip.get("clip_box_excess", 0.0))
        + float(clip.get("clip_mass_zeroed", 0.0))
    ) + 1e-9

    gov["turnover_pre_clip_one_way"] = pre_clip
    gov["turnover_solver_status"] = solver_status
    gov["turnover_solver_name"] = solver_name
    gov["turnover_solver_feasibility_tol"] = solver_tol
    gov.update({k: float(v) for k, v in clip.items()})

    cap = meta.get("turnover_constraint_cap")

    def _raise(msg: str) -> None:
        # The governor block travels WITH the exception, so the failure
        # sentinel can record what the solve actually looked like
        # (alpha-engine-config-I11369). Attaching it here rather than
        # re-deriving it at the catch site keeps the numbers and the verdict
        # from ever disagreeing.
        err = TurnoverBudgetError(msg)
        err.diagnostics = {
            **{k: v for k, v in gov.items() if not hasattr(v, "dual_value")},
            "turnover_constraint_cap": cap,
        }
        raise err

    if cap is not None:
        if pre_clip > float(cap) + solver_tol:
            _raise(
                f"SOLVER assertion: the solved one-way turnover {pre_clip:.6f} "
                f"exceeds the daily budget {float(cap):.6f} beyond the "
                f"feasibility tolerance {solver_tol:.6g} that a "
                f"{solver_name} status={solver_status!r} solve earns (executed "
                f"{requested:.6f}, of which the post-solve clip measurably "
                f"contributed {clip_turnover:.6f}). The budget is a constraint "
                f"inside the convex program, so the solver returned a point "
                f"violating a constraint it was given. Holding the book for "
                f"this session rather than trading an unsanctioned vector "
                f"(alpha-engine-config-I7346 / I11368)."
            )
        if clip_turnover > clip_budget:
            _raise(
                f"CLIP assertion: the post-solve clip displaced "
                f"{clip_turnover:.6f} of one-way turnover, more than its own "
                f"mutations can account for (budget {clip_budget:.6g} = 2 × "
                f"[negatives {float(clip.get('clip_negative_mass', 0.0)):.6g} + "
                f"box {float(clip.get('clip_box_excess', 0.0)):.6g} + dust "
                f"{float(clip.get('clip_mass_zeroed', 0.0)):.6g}]). This is an "
                f"accounting defect in _clip_and_renormalize, not a market "
                f"condition. Holding the book for this session "
                f"(alpha-engine-config-I11368)."
            )
    return weights, gov


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
    alpha_unc_meta: dict | None = None,
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
        # I9452: which vintage of the predictor's uncertainty field built Ω,
        # and — when it did not — why. Emitted on EVERY solve, including the
        # infeasible-fallback path: a reason that only appears when the term
        # engages is indistinguishable from a term that is quietly dead.
        **(alpha_unc_meta or {}),
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
