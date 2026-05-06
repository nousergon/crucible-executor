"""
Hard rule enforcement. All orders must clear this before reaching IBKR.

Rules are evaluated in order — first failure blocks the order.
All thresholds loaded from config/risk.yaml.

Graduated drawdown response (added 2026-03-14):
  Instead of a binary -8% circuit breaker, position sizing scales down
  through tiers as drawdown deepens. The hard halt at -8% is preserved
  as the final tier.

Structured risk-event emission (added 2026-05-06, ROADMAP Phase 2 transparency
inventory — *risk decisions* row):
  `check_order` and `compute_drawdown_multiplier` accept an optional
  `events: list[dict] | None` kwarg. When provided, each veto/halt/throttle
  appends a structured dict (rule + value + threshold + reason) to the list,
  alongside the existing free-text reason returned to the caller. The
  caller persists each event via `trade_logger.log_risk_event`. Default
  `events=None` preserves the existing 2-tuple return contract.
"""

from __future__ import annotations

import logging

import pandas as pd

from executor.strategies.config import load_strategy_config

logger = logging.getLogger(__name__)


def _emit(events: list[dict] | None, event: dict) -> None:
    """Append a structured event to the caller's sink, or no-op if None."""
    if events is not None:
        events.append(event)


def check_correlation(
    ticker: str,
    current_positions: dict[str, dict],
    price_histories: dict[str, pd.DataFrame],
    config: dict,
    events: list[dict] | None = None,
) -> tuple[bool, str]:
    """
    Check if a new entry is too correlated with existing same-sector positions.

    Computes 60-day rolling Pearson correlation between candidate's daily returns
    and each held position's daily returns. Blocks if mean pairwise correlation
    with same-sector positions exceeds threshold.

    Returns:
        (approved, reason)
    """
    if not config.get("correlation_block_enabled", True):
        return True, "correlation check disabled"

    threshold = config.get("correlation_block_threshold", 0.80)
    lookback = config.get("correlation_lookback_days", 60)

    candidate_history = price_histories.get(ticker)
    if candidate_history is None or len(candidate_history) < lookback:
        n = 0 if candidate_history is None else len(candidate_history)
        return True, f"insufficient price history for {ticker} ({n} < {lookback})"

    # Get candidate's sector
    candidate_sector = None
    for t, pos in current_positions.items():
        if t == ticker:
            candidate_sector = pos.get("sector", "")
            break

    # Compute daily returns for candidate (vectorized; drops the N=1 NaN
    # produced by pct_change at the head)
    candidate_returns = (
        candidate_history["close"].iloc[-lookback:].pct_change().dropna()
    )
    if candidate_returns.empty:
        return True, "no returns computed for candidate"

    # Compare with same-sector held positions
    correlations: list[tuple[str, float]] = []
    for held_ticker, pos in current_positions.items():
        if held_ticker == ticker:
            continue
        held_sector = pos.get("sector", "")
        if candidate_sector and held_sector != candidate_sector:
            continue  # only compare within same sector

        held_history = price_histories.get(held_ticker)
        if held_history is None or len(held_history) < lookback:
            continue

        held_returns = held_history["close"].iloc[-lookback:].pct_change().dropna()

        # Align lengths (and indices — the two series may have different
        # last bars if one ticker has missing data). Pearson on aligned
        # tail of length ≥ 10.
        min_len = min(len(candidate_returns), len(held_returns))
        if min_len < 10:
            continue

        cr = candidate_returns.iloc[-min_len:].reset_index(drop=True)
        hr = held_returns.iloc[-min_len:].reset_index(drop=True)

        # Pearson correlation via pandas (NaN if degenerate variance)
        corr = cr.corr(hr)
        if pd.notna(corr):
            correlations.append((held_ticker, float(corr)))

    if not correlations:
        return True, "no same-sector positions to compare"

    mean_corr = sum(c for _, c in correlations) / len(correlations)

    if mean_corr > threshold:
        tickers_str = ", ".join(f"{t}({c:.2f})" for t, c in correlations)
        reason = (
            f"Mean correlation {mean_corr:.2f} > {threshold:.2f} with same-sector positions: {tickers_str}"
        )
        _emit(events, {
            "event_type": "veto",
            "rule": "correlation",
            "ticker": ticker,
            "sector": candidate_sector,
            "reason": reason,
            "value": float(mean_corr),
            "threshold": float(threshold),
            "context": {"per_ticker": [(t, round(c, 4)) for t, c in correlations]},
        })
        return False, reason

    return True, f"correlation check passed (mean={mean_corr:.2f}, threshold={threshold:.2f})"


def compute_drawdown_multiplier(
    portfolio_nav: float,
    peak_nav: float,
    config: dict,
    events: list[dict] | None = None,
) -> tuple[float, str]:
    """
    Compute position sizing multiplier based on current drawdown tier.

    Returns:
        (multiplier, description)
        multiplier=0.0 means full halt (circuit breaker).

    When `events` is supplied, appends a structured `halt` event on
    circuit-breaker fire or a `throttle` event when an active tier is
    reducing sizing below 1.0. No event is emitted at full sizing.
    """
    if peak_nav <= 0:
        return 1.0, "no peak NAV recorded"

    drawdown = (portfolio_nav - peak_nav) / peak_nav  # negative number

    strategy_cfg = load_strategy_config(config)

    if not strategy_cfg.get("graduated_drawdown_enabled", True):
        # Fall back to original binary circuit breaker
        threshold = -config.get("drawdown_circuit_breaker", 0.08)
        if drawdown <= threshold:
            reason = f"circuit breaker: {drawdown:.1%} from peak (limit {threshold:.1%})"
            _emit(events, {
                "event_type": "halt",
                "rule": "drawdown_halt",
                "reason": reason,
                "value": float(drawdown),
                "threshold": float(threshold),
                "context": {"graduated_disabled": True},
            })
            return 0.0, reason
        return 1.0, f"drawdown {drawdown:.1%} — within limit"

    tiers = strategy_cfg.get("drawdown_tiers", [])

    # Tiers are sorted by threshold ascending (most negative last).
    # Walk through tiers; the last tier whose threshold is breached applies.
    active_multiplier = 1.0
    active_desc = f"drawdown {drawdown:.1%} — full sizing"
    active_threshold: float | None = None
    active_tier_desc: str | None = None

    for threshold, multiplier, description in tiers:
        if drawdown <= threshold:
            active_multiplier = multiplier
            active_desc = f"drawdown {drawdown:.1%} — {description} (multiplier={multiplier})"
            active_threshold = float(threshold)
            active_tier_desc = description

    # Hard halt at the deepest configured tier (circuit breaker preserved)
    hard_halt = -config.get("drawdown_circuit_breaker", 0.08)
    if drawdown <= hard_halt:
        reason = f"circuit breaker: {drawdown:.1%} from peak (limit {hard_halt:.1%})"
        _emit(events, {
            "event_type": "halt",
            "rule": "drawdown_halt",
            "reason": reason,
            "value": float(drawdown),
            "threshold": float(hard_halt),
        })
        return 0.0, reason

    if active_multiplier < 1.0:
        _emit(events, {
            "event_type": "throttle",
            "rule": "drawdown_tier_throttle",
            "reason": active_desc,
            "value": float(drawdown),
            "threshold": active_threshold,
            "context": {
                "multiplier": float(active_multiplier),
                "tier_description": active_tier_desc,
            },
        })

    return active_multiplier, active_desc


def check_order(
    ticker: str,
    action: str,
    dollar_size: float,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict[str, dict],
    sector: str,
    market_regime: str,
    signal: dict,
    config: dict,
    price_histories: dict[str, pd.DataFrame] | None = None,
    events: list[dict] | None = None,
) -> tuple[bool, str]:
    """
    Validate an order against all risk rules.

    Args:
        ticker: stock symbol
        action: "ENTER" | "REDUCE" | "EXIT"
        dollar_size: proposed dollar value of the order
        portfolio_nav: current net asset value
        peak_nav: highest NAV recorded in trades.db (for drawdown check)
        current_positions: {ticker: {"market_value": float, "sector": str}}
        sector: sector of the ticker (from signals.json)
        market_regime: "bull" | "neutral" | "bear"
        signal: full signal entry from signals.json
        config: loaded from config/risk.yaml
        events: optional sink for structured veto/halt/throttle events. The
            callable persists each entry via `trade_logger.log_risk_event`.

    Returns:
        (approved, reason)
    """
    base_event_ctx = {
        "ticker": ticker,
        "sector": sector,
        "market_regime": market_regime,
    }

    if portfolio_nav <= 0:
        reason = "Portfolio NAV is zero or negative"
        _emit(events, {**base_event_ctx,
            "event_type": "veto", "rule": "nav_nonpositive",
            "reason": reason, "value": float(portfolio_nav), "threshold": 0.0,
        })
        return False, reason

    position_pct = dollar_size / portfolio_nav

    # EXIT and REDUCE always pass risk guard — we're reducing exposure
    if action in ("EXIT", "REDUCE"):
        return True, f"{action} — reducing exposure, risk rules bypassed"

    # ── ENTER rules ───────────────────────────────────────────────────────────

    # 1. Score minimum
    score = signal.get("score") or 0
    min_score = config.get("min_score_to_enter", 70)
    if score < min_score:
        reason = f"Score {score:.1f} < minimum {min_score}"
        _emit(events, {**base_event_ctx,
            "event_type": "veto", "rule": "min_score",
            "reason": reason, "value": float(score), "threshold": float(min_score),
        })
        return False, reason

    # 2. Conviction: no hard gate — declining conviction is handled by position
    #    sizer (0.7x multiplier).  Research conviction is weekly, too stale to
    #    block daily entries that the predictor scores positively.

    # 3. Graduated drawdown response (replaces binary circuit breaker).
    #    Drawdown halt/throttle is portfolio-state — the caller emits the
    #    structured event ONCE per planning cycle (see main.py's call to
    #    compute_drawdown_multiplier at the top of the planner). Don't
    #    propagate `events` here, or we'd append one halt/throttle event
    #    per ticker checked.
    dd_multiplier, dd_reason = compute_drawdown_multiplier(
        portfolio_nav, peak_nav, config,
    )
    if dd_multiplier <= 0.0:
        return False, f"Drawdown halt: {dd_reason}"
    if dd_multiplier < 1.0:
        logger.info(f"Drawdown tier active for {ticker}: {dd_reason}")

    # 4. Max single position size
    effective_max_pct = (
        config.get("bear_max_position_pct", 0.025)
        if market_regime == "bear"
        else config.get("max_position_pct", 0.05)
    )
    if position_pct > effective_max_pct:
        reason = f"Position size {position_pct:.1%} exceeds max {effective_max_pct:.1%}"
        _emit(events, {**base_event_ctx,
            "event_type": "veto", "rule": "max_position",
            "reason": reason, "value": float(position_pct),
            "threshold": float(effective_max_pct),
            "context": {"dollar_size": float(dollar_size), "portfolio_nav": float(portfolio_nav)},
        })
        return False, reason

    # 5. Bear regime: block new entries in underweight sectors
    if market_regime == "bear" and config.get("bear_block_underweight", True):
        sector_rating_str = signal.get("sector_rating", "market_weight")
        if sector_rating_str == "underweight":
            reason = f"Bear regime: new entries blocked in underweight sector ({sector})"
            _emit(events, {**base_event_ctx,
                "event_type": "veto", "rule": "bear_underweight",
                "reason": reason,
                "context": {"sector_rating": sector_rating_str},
            })
            return False, reason

    # 6. Max sector exposure
    sector_exposure = sum(
        pos["market_value"]
        for pos in current_positions.values()
        if pos.get("sector") == sector
    )
    sector_pct = (sector_exposure + dollar_size) / portfolio_nav
    max_sector = config.get("max_sector_pct", 0.25)
    if sector_pct > max_sector:
        reason = f"Sector exposure {sector_pct:.1%} would exceed max {max_sector:.1%} for {sector}"
        _emit(events, {**base_event_ctx,
            "event_type": "veto", "rule": "max_sector",
            "reason": reason, "value": float(sector_pct),
            "threshold": float(max_sector),
            "context": {
                "existing_sector_exposure": float(sector_exposure),
                "added_dollar_size": float(dollar_size),
            },
        })
        return False, reason

    # 7. Max total equity exposure
    total_equity = sum(pos["market_value"] for pos in current_positions.values())
    equity_pct = (total_equity + dollar_size) / portfolio_nav
    max_equity = config.get("max_equity_pct", 0.90)
    if equity_pct > max_equity:
        reason = f"Total equity exposure {equity_pct:.1%} would exceed max {max_equity:.1%}"
        _emit(events, {**base_event_ctx,
            "event_type": "veto", "rule": "max_equity",
            "reason": reason, "value": float(equity_pct),
            "threshold": float(max_equity),
            "context": {"existing_equity": float(total_equity)},
        })
        return False, reason

    # 8. Cross-ticker correlation
    if price_histories is not None:
        corr_approved, corr_reason = check_correlation(
            ticker, current_positions, price_histories, config, events=events,
        )
        if not corr_approved:
            return False, corr_reason

    conviction = signal.get("conviction", "stable")
    return True, (
        f"ENTER approved | score={score:.1f} conviction={conviction} "
        f"size={position_pct:.1%} NAV | dd_multiplier={dd_multiplier}"
    )
