"""
Hard rule enforcement. All orders must clear this before reaching IBKR.

Rules are evaluated in order — first failure blocks the order.
All thresholds loaded from config/risk.yaml.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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

    Returns:
        (approved, reason)
    """
    if portfolio_nav <= 0:
        return False, "Portfolio NAV is zero or negative"

    position_pct = dollar_size / portfolio_nav

    # EXIT and REDUCE always pass risk guard — we're reducing exposure
    if action in ("EXIT", "REDUCE"):
        return True, f"{action} — reducing exposure, risk rules bypassed"

    # ── ENTER rules ───────────────────────────────────────────────────────────

    # 1. Score minimum
    score = signal.get("score", 0)
    min_score = config.get("min_score_to_enter", 70)
    if score < min_score:
        return False, f"Score {score:.1f} < minimum {min_score}"

    # 2. Conviction gate
    conviction = signal.get("conviction", "stable")
    allowed_convictions = config.get("min_conviction_to_enter", ["rising", "stable"])
    if conviction not in allowed_convictions:
        return False, f"Conviction '{conviction}' not in allowed set {allowed_convictions}"

    # 3. Drawdown circuit breaker
    if peak_nav > 0:
        drawdown = (portfolio_nav - peak_nav) / peak_nav
        threshold = -config.get("drawdown_circuit_breaker", 0.08)
        if drawdown <= threshold:
            return False, f"Drawdown circuit breaker: portfolio is {drawdown:.1%} from peak (limit {threshold:.1%})"

    # 4. Max single position size
    effective_max_pct = (
        config.get("bear_max_position_pct", 0.025)
        if market_regime == "bear"
        else config.get("max_position_pct", 0.05)
    )
    if position_pct > effective_max_pct:
        return False, f"Position size {position_pct:.1%} exceeds max {effective_max_pct:.1%}"

    # 5. Bear regime: block new entries in underweight sectors
    if market_regime == "bear" and config.get("bear_block_underweight", True):
        # sector_rating is passed via signal or looked up in calling code
        sector_rating_str = signal.get("sector_rating", "market_weight")
        if sector_rating_str == "underweight":
            return False, f"Bear regime: new entries blocked in underweight sector ({sector})"

    # 6. Max sector exposure
    sector_exposure = sum(
        pos["market_value"]
        for pos in current_positions.values()
        if pos.get("sector") == sector
    )
    sector_pct = (sector_exposure + dollar_size) / portfolio_nav
    max_sector = config.get("max_sector_pct", 0.25)
    if sector_pct > max_sector:
        return False, f"Sector exposure {sector_pct:.1%} would exceed max {max_sector:.1%} for {sector}"

    # 7. Max total equity exposure
    total_equity = sum(pos["market_value"] for pos in current_positions.values())
    equity_pct = (total_equity + dollar_size) / portfolio_nav
    max_equity = config.get("max_equity_pct", 0.90)
    if equity_pct > max_equity:
        return False, f"Total equity exposure {equity_pct:.1%} would exceed max {max_equity:.1%}"

    return True, (
        f"ENTER approved | score={score:.1f} conviction={conviction} "
        f"size={position_pct:.1%} NAV"
    )
