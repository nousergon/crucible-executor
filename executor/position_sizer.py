"""
Position sizing algorithm per design doc section B.3.

Inputs: portfolio NAV, signal data, sector rating, current price, config.
Output: shares, dollar_size, position_pct.

Graduated drawdown multiplier (added 2026-03-14):
  When portfolio is in a drawdown tier, all position sizes are scaled
  down by the tier's multiplier. This is applied after all other
  adjustments but before the max position cap.
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)

_DEFAULT_SECTOR_ADJ = {
    "overweight": 1.05,
    "market_weight": 1.00,
    "underweight": 0.85,
}


def compute_position_size(
    ticker: str,
    portfolio_nav: float,
    enter_signals: list[dict],
    signal: dict,
    sector_rating: str,
    current_price: float,
    config: dict,
    drawdown_multiplier: float = 1.0,
    atr_pct: float | None = None,
    prediction_confidence: float | None = None,
    p_up: float | None = None,
    signal_age_days: int | None = None,
    days_to_earnings: int | None = None,
    feature_coverage: float | None = None,
) -> dict:
    """
    Compute position size for a new ENTER order.

    Algorithm (design doc B.3 + graduated drawdown):
      1. base_weight = 1 / n_enter_signals  (equal weight across all entries today)
      2. sector_adj: from config sector_adj map (default: slight OW/UW tilt)
      3. conviction_adj: rising/stable→1.00, declining→config multiplier
      4. upside_adj: price_target_upside < min_price_target_upside → config multiplier
      5. drawdown_adj: multiplier from graduated drawdown tiers (1.0/0.50/0.25)
      6. position_weight = min(base * sector * conviction * upside * dd, max_position_pct)
      7. dollar_size = portfolio_nav * position_weight
      8. shares = floor(dollar_size / current_price)

    Returns:
        {"shares": int, "dollar_size": float, "position_pct": float}
    """
    n = max(len(enter_signals), 1)
    base_weight = 1.0 / n

    sector_adj_map = config.get("sector_adj", _DEFAULT_SECTOR_ADJ)
    sector_adj = sector_adj_map.get(sector_rating, 1.00)

    conviction = signal.get("conviction", "stable")
    conviction_decline_mult = config.get("conviction_decline_adj", 0.70)
    conviction_adj = conviction_decline_mult if conviction == "declining" else 1.00

    upside = signal.get("price_target_upside")
    min_upside = config.get("min_price_target_upside", 0.05)
    upside_fail_mult = config.get("upside_fail_adj", 0.70)
    upside_adj = upside_fail_mult if (upside is not None and upside < min_upside) else 1.00

    # ATR-based position sizing (Task 2.1)
    if config.get("atr_sizing_enabled", True) and atr_pct is not None and atr_pct > 0:
        target_risk = config.get("atr_sizing_target_risk", 0.02)
        atr_adj = max(0.5, min(target_risk / atr_pct, 1.5))
    else:
        atr_adj = 1.0

    # Confidence-weighted sizing (Task 2.2)
    if config.get("confidence_sizing_enabled", True) and prediction_confidence is not None:
        prediction_confidence = max(0.0, min(1.0, prediction_confidence))
        conf_min = config.get("confidence_sizing_min", 0.7)
        conf_range = config.get("confidence_sizing_range", 0.6)
        confidence_adj = conf_min + conf_range * prediction_confidence
    else:
        confidence_adj = 1.0

    # p_up-weighted sizing (Phase 4d): blend p_up into confidence if IC is positive
    if config.get("use_p_up_sizing") and p_up is not None:
        p_up = max(0.0, min(1.0, p_up))
        blend = config.get("p_up_sizing_blend", 0.3)
        p_up_adj = 0.7 + 0.6 * p_up  # map p_up [0,1] → [0.7, 1.3]
        confidence_adj = confidence_adj * (1 - blend) + p_up_adj * blend

    # Signal staleness discount (Task 2.3)
    # Decay applies only to age BEYOND the source's expected refresh cadence.
    # Research signals are written weekly (Saturday); a Wednesday read at age=4
    # is fresh relative to cadence, not stale. Daemon health gate already uses
    # the same grace period (192h for research at main.py:894).
    if config.get("staleness_discount_enabled", True) and signal_age_days is not None:
        cadence_grace = config.get("signal_cadence_days", 7)
        effective_age = max(0, signal_age_days - cadence_grace)
        if effective_age > 0:
            decay_rate = config.get("staleness_decay_per_day", 0.03)
            floor = config.get("staleness_floor", 0.70)
            staleness_adj = max(1.0 - decay_rate * effective_age, floor)
        else:
            staleness_adj = 1.0
    else:
        staleness_adj = 1.0

    # Earnings sizing adjustment (Task 2.4)
    if config.get("earnings_sizing_enabled", True) and days_to_earnings is not None:
        proximity_days = config.get("earnings_proximity_days", 5)
        reduction = config.get("earnings_sizing_reduction", 0.50)
        if days_to_earnings <= proximity_days:
            earnings_adj = 1.0 - reduction
        else:
            earnings_adj = 1.0
    else:
        earnings_adj = 1.0

    # Feature-coverage derate (2026-04-22). Post-PR-#78 (alpha-engine-data),
    # short-history tickers (new listings, spinoffs) land in ArcticDB with
    # partial-NaN features — e.g. SNDK with ~290 bars has NaN on every
    # 252-day rolling feature. Predictor can still score such tickers
    # (LightGBM splits on NaN natively), but sizing them at full weight
    # overstates the information we have. Derate position size by the
    # fraction of non-NaN features, floored at ``coverage_derate_floor``
    # so we never size below a meaningful threshold. Continuous (no cliff)
    # so an 87%-covered ticker gets 87% of a 100%-covered ticker's size.
    if (
        config.get("coverage_sizing_enabled", True)
        and feature_coverage is not None
    ):
        coverage_floor = config.get("coverage_derate_floor", 0.25)
        clamped_cov = max(0.0, min(1.0, feature_coverage))
        coverage_adj = max(coverage_floor, clamped_cov)
    else:
        coverage_adj = 1.0

    max_pct = config.get("max_position_pct", 0.05)
    raw_weight = (base_weight * sector_adj * conviction_adj * upside_adj
                  * drawdown_multiplier * atr_adj * confidence_adj
                  * staleness_adj * earnings_adj * coverage_adj)
    position_weight = min(raw_weight, max_pct)

    # ATR volatility cap: ensure ATR constraint is not overridden by other
    # multipliers. If ATR sizing reduced the weight, the final weight should
    # not exceed what ATR alone would have allowed.
    if config.get("atr_sizing_enabled", True) and atr_adj < 1.0:
        atr_only_weight = base_weight * atr_adj * drawdown_multiplier
        position_weight = min(position_weight, atr_only_weight, max_pct)

    dollar_size = portfolio_nav * position_weight
    shares = math.floor(dollar_size / current_price) if current_price and current_price > 0 else 0

    # Minimum position size check (Task 1.2)
    if dollar_size < config.get("min_position_dollar", 500):
        shares = 0

    logger.debug(
        f"{ticker} sizing: n={n} base={base_weight:.3f} sector_adj={sector_adj} "
        f"conviction_adj={conviction_adj} upside_adj={upside_adj} "
        f"dd_mult={drawdown_multiplier} atr_adj={atr_adj} "
        f"confidence_adj={confidence_adj} staleness_adj={staleness_adj} "
        f"earnings_adj={earnings_adj} coverage_adj={coverage_adj} "
        f"→ {position_weight:.3f} NAV = ${dollar_size:.0f} = {shares} shares"
    )

    return {
        "shares": shares,
        "dollar_size": round(dollar_size, 2),
        "position_pct": round(position_weight, 4),
        "sector_adj": sector_adj,
        "conviction_adj": conviction_adj,
        "upside_adj": upside_adj,
        "dd_multiplier": drawdown_multiplier,
        "atr_adj": atr_adj,
        "confidence_adj": confidence_adj,
        "staleness_adj": staleness_adj,
        "earnings_adj": earnings_adj,
        "coverage_adj": coverage_adj,
    }
