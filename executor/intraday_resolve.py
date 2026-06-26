"""Intraday reconcile-to-target: real-time drawdown overlay + event-driven
optimizer re-solve to redeploy cash freed by hard-risk exits.

Background (incident 2026-05-29): under ``use_portfolio_optimizer`` the
optimizer plans a fully-invested book ONCE in the morning, but hard-risk
exits fire intraday and free cash with no same-day path to redeploy it — the
portfolio drifted to 22% idle cash with SPY at 0. This module is the daemon's
"execution + real-time risk" layer:

  1. ``compute_drawdown_overlay`` — reuses the SAME
     ``risk_guard.compute_drawdown_multiplier`` the morning planner uses
     (no divergent logic), evaluated live each tick. Drawdown tiers force
     de-risking exits AND suppress redeploy (freed cash stays cash — the
     point of a circuit breaker).
  2. ``solve_redeploy`` — event-driven optimizer re-solve reusing the MORNING
     alpha_hat + DAILY covariance cached in the optimizer shadow log (both
     daily-stable → zero new alpha look-ahead), against LIVE positions, to
     redeploy excess cash back toward the 3% sleeve.

All functions here are pure (no IB / no S3) so the daemon stays a thin wiring
layer and the logic is unit-testable. See plan toasty-brewing-glacier.md.
"""

from __future__ import annotations

import logging

import numpy as np

from executor.portfolio_optimizer import solve_target_weights
from executor.risk_guard import compute_drawdown_multiplier

logger = logging.getLogger(__name__)

_SPY = "SPY"
_CASH = "CASH"
_OK_STATUSES = {"optimal", "optimal_inaccurate"}


def compute_drawdown_overlay(
    nav: float,
    peak_nav: float,
    config: dict,
    strategy_config: dict,
    regime_intensity_z: float | None = None,
) -> dict:
    """Return the intraday drawdown posture.

    Reuses ``compute_drawdown_multiplier`` verbatim so morning-planner and
    intraday behaviour are identical. ``forced_exit_count`` mirrors the
    morning §2g tiering. ``redeploy_suppressed`` is True in any drawdown tier
    (multiplier < 1.0): while de-risking, cash freed by hard-risk exits must
    stay cash rather than be redeployed by the re-solve.
    """
    multiplier, tier_desc = compute_drawdown_multiplier(
        portfolio_nav=nav,
        peak_nav=peak_nav,
        config=config,
        regime_intensity_z=regime_intensity_z,
    )
    forced_exit_count = 0
    if strategy_config.get("drawdown_forced_exit_enabled", True) and multiplier < 1.0:
        if multiplier <= 0.25:
            forced_exit_count = int(strategy_config.get("drawdown_forced_exit_tier3_count", 2))
        elif multiplier <= 0.50:
            forced_exit_count = int(strategy_config.get("drawdown_forced_exit_tier2_count", 1))
    return {
        "multiplier": float(multiplier),
        "tier_desc": tier_desc,
        "forced_exit_count": forced_exit_count,
        "redeploy_suppressed": multiplier < 1.0,
    }


def build_conviction_map(signals_payload: dict | None) -> dict[str, dict]:
    """Flatten a ``signals.json`` payload into a ``{ticker: record}`` map for
    forced-exit conviction ranking (config#844).

    Records are drawn from the ``universe`` + ``buy_candidates`` lists (mirroring
    main.py §2d's ``signals_by_ticker`` construction); the first record wins on a
    duplicate ticker. Returns an empty map for a missing/empty/malformed payload
    so callers degrade to the smallest-position-first fallback in
    :func:`select_forced_exits` rather than raising.
    """
    out: dict[str, dict] = {}
    if not signals_payload:
        return out
    records = (signals_payload.get("universe") or []) + (signals_payload.get("buy_candidates") or [])
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ticker = rec.get("ticker")
        if ticker and ticker not in out:
            out[ticker] = rec
    return out


def select_forced_exits(
    current_positions: dict[str, dict],
    signals_by_ticker: dict[str, dict],
    already_exited: set[str],
    target_count: int,
) -> list[dict]:
    """Lowest-conviction held names to force-exit, mirroring main.py §2g.

    Idempotent: returns only NEW names beyond ``already_exited``, up to the
    remaining headroom to reach ``target_count`` total forced exits.
    """
    if target_count <= 0:
        return []
    remaining = target_count - len(already_exited)
    if remaining <= 0:
        return []

    def _rank(item):
        t, pos = item
        score = (signals_by_ticker.get(t, {}) or {}).get("score") or 50
        return (score, pos.get("market_value", 0))

    out: list[dict] = []
    for t, pos in sorted(current_positions.items(), key=_rank):
        if t in already_exited:
            continue
        shares = int(pos.get("shares", 0))
        if shares > 0:
            out.append({
                "ticker": t,
                "action": "EXIT",
                "shares": shares,
                "reason": "drawdown_forced_exit",
                "detail": f"intraday drawdown de-risk ({target_count} forced exit(s) for tier)",
            })
        if len(out) >= remaining:
            break
    return out


def w_prev_from_live(
    tickers: list[str],
    current_positions: dict[str, dict],
    nav: float,
    cash_idx: int,
) -> np.ndarray:
    """Current weights from LIVE positions (self-correcting idempotency).

    Already-filled redeploy buys show up here, so a subsequent re-solve won't
    re-buy them (the turnover band suppresses the now-zero delta).
    """
    w = np.zeros(len(tickers))
    if nav <= 0:
        w[cash_idx] = 1.0
        return w
    invested = 0.0
    for i, t in enumerate(tickers):
        if i == cash_idx:
            continue
        pos = current_positions.get(t)
        if pos:
            wi = float(pos.get("market_value", 0.0)) / nav
            w[i] = wi
            invested += wi
    w[cash_idx] = max(0.0, 1.0 - invested)
    return w


def available_redeploy_cash(
    tickers: list[str],
    current_positions: dict[str, dict],
    nav: float,
    sleeve_pct: float,
    pending_entry_dollars: float,
) -> float:
    """Cash above the target sleeve that is genuinely free to redeploy.

    Subtracts still-pending entry dollars (morning entries AND already-enqueued
    redeploy entries) so (a) we don't over-deploy past the sleeve while morning
    entries are unfilled, and (b) the redeploy is idempotent — once buys are
    enqueued they count here and the gate closes on the next tick.
    """
    cash_idx = tickers.index(_CASH)
    w = w_prev_from_live(tickers, current_positions, nav, cash_idx)
    excess = (w[cash_idx] - sleeve_pct) * nav
    return excess - max(0.0, pending_entry_dollars)


def solve_redeploy(
    *,
    shadow_log: dict,
    current_positions: dict[str, dict],
    nav: float,
    stopped_out: set[str],
) -> dict:
    """Re-solve the optimizer reusing the morning Σ + alpha_hat (cached in the
    shadow log) against LIVE positions, to redeploy freed cash to the sleeve.

    Returns ``{status, buys, target_weights, vol_ann}``. ``buys`` is a list of
    ``{ticker, delta_weight, delta_dollars, target_weight}`` for names whose
    target exceeds the current live weight by more than the rebalance band. On
    a non-optimal solve, ``buys`` is empty — the caller MUST leave cash idle
    and log loud (the re-solve is load-bearing; it never silently no-ops as if
    it had succeeded).
    """
    tickers = list(shadow_log["tickers"])
    spy_idx = tickers.index(_SPY)
    cash_idx = tickers.index(_CASH)

    alpha_hat = np.asarray(shadow_log["alpha_hat"], dtype=float)
    eligibility = np.asarray(shadow_log["eligibility"], dtype=bool).copy()
    stance_caps = np.asarray(shadow_log["stance_caps"], dtype=float)
    sectors = list(shadow_log["sectors"])
    covariance = np.asarray(shadow_log["covariance_daily"], dtype=float)
    cfg = dict(shadow_log.get("optimizer_cfg") or {})

    au = shadow_log.get("alpha_uncertainty")
    alpha_uncertainty = None
    if au is not None:
        alpha_uncertainty = np.asarray(
            [np.nan if x is None else float(x) for x in au], dtype=float,
        )

    # Exclude same-day stopped-out names: the morning alpha is still positive
    # for a name a gap stop just exited, so without this the re-solve would
    # immediately re-buy it and undo the stop. Reuses the eligibility pin.
    for t in stopped_out:
        if t in tickers:
            eligibility[tickers.index(t)] = False

    w_prev = w_prev_from_live(tickers, current_positions, nav, cash_idx)

    result = solve_target_weights(
        tickers=tickers,
        alpha_hat=alpha_hat,
        returns_panel=None,
        w_prev=w_prev,
        sectors=sectors,
        stance_caps=stance_caps,
        eligibility=eligibility,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
        cfg=cfg,
        alpha_uncertainty=alpha_uncertainty,
        covariance=covariance,
    )
    status = result.diagnostics.get("status")
    vol_ann = result.diagnostics.get("portfolio_vol_ann")
    if status not in _OK_STATUSES:
        return {"status": status, "buys": [], "target_weights": None, "vol_ann": vol_ann}

    band = float(cfg.get("rebalance_band_pct", 0.005))
    buys: list[dict] = []
    for i, t in enumerate(tickers):
        if i == cash_idx:
            continue
        dw = float(result.weights[i] - w_prev[i])
        if dw > band:
            buys.append({
                "ticker": t,
                "delta_weight": dw,
                "delta_dollars": dw * nav,
                "target_weight": float(result.weights[i]),
            })
    return {
        "status": status,
        "buys": buys,
        "target_weights": [float(x) for x in result.weights],
        "vol_ann": vol_ann,
    }


def build_redeploy_entry(
    ticker: str,
    shares: int,
    price: float,
    target_weight: float,
    run_date: str,
    pullback_pct: float = 0.02,
) -> dict:
    """Build a minimal order-book entry for an intraday redeploy buy.

    Fills via the normal entry-trigger path (pullback / graduated / 3:55 ET
    time-expiry) — NOT an immediate market order — so we don't chase into a
    volatile name; the expiry guarantees same-day deployment.
    """
    return {
        "ticker": ticker,
        "signal": "ENTER",
        "shares": int(shares),
        "current_price": float(price),
        "triggers": {
            "pullback_pct": float(pullback_pct),
            "vwap_discount": None,
            "vwap": None,
            "support_level": None,
        },
        "sizing_source": "optimizer_redeploy",
        "sizing_factors": {"optimizer_target_weight": float(target_weight)},
        "status": "pending",
    }
