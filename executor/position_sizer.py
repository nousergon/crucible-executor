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


def regime_conditional_size_multiplier(
    intensity_z: float | None,
    *,
    scale: float = 0.05,
    floor: float = 0.70,
    ceil: float = 1.30,
) -> float:
    """Map regime composite intensity_z to a sizing multiplier.

    intensity_z follows the risk-on convention from the predictor's
    regime substrate (alpha-engine-predictor/regime/composite.py): positive
    values = risk-on conditions, negative = risk-off. Sizing follows the
    same direction — upweight in risk-on, downweight in risk-off.

    Linear with clamping: ``1.0 + intensity_z * scale`` then clamped to
    ``[floor, ceil]``. ``scale=0.05`` means a +1σ risk-on reading yields
    a 1.05× tilt, a -1σ risk-off reading a 0.95× tilt — small enough
    that it composes with the other (multiplicative) adjustments without
    dominating, large enough to register vs sector_adj (1.05/0.85).
    Clamp prevents pathological tails from blowing past max_position_pct.

    Returns 1.0 when ``intensity_z`` is None (substrate unavailable) —
    preserves legacy behavior so a substrate read failure does not
    silently change sizing.
    """
    if intensity_z is None:
        return 1.0
    raw = 1.0 + float(intensity_z) * float(scale)
    return max(float(floor), min(float(ceil), raw))


def compute_position_size(
    ticker: str,
    portfolio_nav: float,
    enter_signals: list[dict],
    signal: dict,
    sector_rating: str,
    current_price: float,
    config: dict,
    drawdown_multiplier: float = 1.0,
    derisk_multiplier: float = 1.0,
    atr_pct: float | None = None,
    prediction_confidence: float | None = None,
    p_up: float | None = None,
    signal_age_days: int | None = None,
    days_to_earnings: int | None = None,
    feature_coverage: float | None = None,
    stance: str | None = None,
    regime_intensity_z: float | None = None,
    barrier_win_prob: float | None = None,
    adv_usd: float | None = None,
) -> dict:
    """
    Compute position size for a new ENTER order.

    Algorithm (design doc B.3 + graduated drawdown):
      1. base_weight = 1 / n_enter_signals  (equal weight across all entries today)
      2. sector_adj: from config sector_adj map (default: slight OW/UW tilt)
      3. conviction_adj: rising/stable→1.00, declining→config multiplier
      4. upside_adj: price_target_upside < min_price_target_upside → config multiplier
      5. drawdown_adj: multiplier from graduated drawdown tiers (1.0/0.50/0.25)
      6. derisk_adj: expectancy-gated de-risk multiplier (config-I2820 / PR2071),
         1.0 unless the standing de-risk stance is active, in which case it's
         ``derisk_sizing_multiplier`` (config default 0.50) — independent of
         and multiplicatively composed with the drawdown adj, not a substitute
         for it (drawdown = realized loss; de-risk = forward expectancy gate).
      7. position_weight = min(base * sector * conviction * upside * dd * derisk, max_position_pct)
      8. ADV cap (tradeability arc, config#1401): dollar_size is additionally
         capped at ``max_pct_adv × adv_usd`` so a single new position can never
         consume more than a configured slice of the name's average daily dollar
         volume — the per-name capacity guardrail that mirrors the optimizer's
         max-%-ADV constraint. Skipped when ``adv_usd`` is missing/≤0 (coverage
         gap → conservative degrade, no cap) or the cap is disabled.
      9. dollar_size = portfolio_nav * position_weight  (then ADV-capped)
      10. shares = floor(dollar_size / current_price)

    ``adv_usd`` — the name's average daily DOLLAR volume from the scanner
    tradeability artifact (crucible-research#343, ``tradeability.adv_usd``).
    None ↔ no ADV coverage → no ADV cap applied (preserves legacy sizing).

    Returns:
        {"shares": int, "dollar_size": float, "position_pct": float, ...}
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
    # Stance-conditional sizing (stance taxonomy arc PR 4 follow-up).
    # Per-stance multipliers reflect the asymmetry between stance theses:
    #   momentum (1.0×) — trend-following, the baseline thesis
    #   value (0.7×)    — contrarian thesis carries higher uncertainty
    #                     (catching a falling knife is structurally riskier
    #                     than riding a trend), so smaller stake
    #   quality (0.8×)  — defensive names accept smaller positions in
    #                     exchange for longer hold + tighter exit gates
    #   catalyst (0.6×) — event-driven = higher variance + binary outcome,
    #                     smallest stake size
    #
    # Multipliers backtester-tunable from day 1 — once 4+ weeks of
    # stance-tagged history exists, the optimizer can move them based
    # on per-stance Sharpe / alpha attribution from
    # backtester#182's ``by_stance`` table.
    #
    # Falls through to 1.0× (no adjustment) when stance is None
    # (pre-stance-arc artifacts) — preserves legacy behavior.
    if config.get("stance_sizing_enabled", True) and stance is not None:
        stance_multipliers = {
            "momentum": config.get("stance_size_momentum", 1.0),
            "value":    config.get("stance_size_value",    0.7),
            "quality":  config.get("stance_size_quality",  0.8),
            "catalyst": config.get("stance_size_catalyst", 0.6),
        }
        stance_adj = stance_multipliers.get(stance, 1.0)
    else:
        stance_adj = 1.0

    if (
        config.get("coverage_sizing_enabled", True)
        and feature_coverage is not None
    ):
        coverage_floor = config.get("coverage_derate_floor", 0.25)
        clamped_cov = max(0.0, min(1.0, feature_coverage))
        coverage_adj = max(coverage_floor, clamped_cov)
    else:
        coverage_adj = 1.0

    # Regime sizing adjustment (Stage D' Wire 2, regime-v3-260514). Reads
    # the composite intensity_z from the regime substrate (predictor-side,
    # see regime/composite.py — positive=risk-on, negative=risk-off).
    # Default OFF so the wire ships dormant; operator flips ON via
    # risk.yaml ``regime_sizing_enabled`` after observing 4 weeks of
    # SF runs with intensity_z surfaced through the dashboard. Composes
    # multiplicatively with the other adjustments.
    if config.get("regime_sizing_enabled", False):
        regime_adj = regime_conditional_size_multiplier(
            regime_intensity_z,
            scale=config.get("regime_sizing_scale", 0.05),
            floor=config.get("regime_sizing_floor", 0.70),
            ceil=config.get("regime_sizing_ceil", 1.30),
        )
    else:
        regime_adj = 1.0

    # Barrier-win-probability sizing (Task B2 — meta-labeling consumer).
    # Reads the predictor's calibrated ``barrier_win_prob`` = P(profit/upper
    # barrier touched before stop/lower barrier). Linear map centered at the
    # 0.5 coin-flip → 1.0×, mirroring the p_up precedent above:
    #   adj = min + range * bwp  →  bwp=0.5→1.0, bwp=0→min, bwp=1→min+range.
    # Default OFF (ships DORMANT): the predictor emits the field observe-only
    # (Task B1); the operator flips ``barrier_win_prob_sizing_enabled`` ON only
    # after the field has soaked ≥1 Saturday cycle AND the backtester sweep
    # (Task B3) justifies the weight. Composes multiplicatively; clamped so a
    # bad/extreme probability cannot blow up size (and the final
    # ``min(raw_weight, max_pct)`` caps it regardless). Falls through to 1.0×
    # when the field is absent (pre-B1 predictions) — graceful degrade.
    if (
        config.get("barrier_win_prob_sizing_enabled", False)
        and barrier_win_prob is not None
    ):
        bwp = max(0.0, min(1.0, barrier_win_prob))
        bwp_min = config.get("barrier_win_prob_sizing_min", 0.70)
        bwp_range = config.get("barrier_win_prob_sizing_range", 0.60)
        barrier_win_prob_adj = bwp_min + bwp_range * bwp
    else:
        barrier_win_prob_adj = 1.0

    max_pct = config.get("max_position_pct", 0.05)
    raw_weight = (base_weight * sector_adj * conviction_adj * upside_adj
                  * drawdown_multiplier * derisk_multiplier * atr_adj * confidence_adj
                  * staleness_adj * earnings_adj * coverage_adj
                  * stance_adj * regime_adj * barrier_win_prob_adj)
    position_weight = min(raw_weight, max_pct)

    # ATR volatility cap: ensure ATR constraint is not overridden by other
    # multipliers. If ATR sizing reduced the weight, the final weight should
    # not exceed what ATR alone would have allowed.
    if config.get("atr_sizing_enabled", True) and atr_adj < 1.0:
        atr_only_weight = base_weight * atr_adj * drawdown_multiplier * derisk_multiplier
        position_weight = min(position_weight, atr_only_weight, max_pct)

    dollar_size = portfolio_nav * position_weight

    # ── ADV-based size cap (tradeability arc, config#1401) ───────────────────
    # Per-name capacity guardrail: a single new position may consume at most
    # ``adv_size_cap_pct_adv`` of the name's average daily DOLLAR volume. This
    # mirrors the optimizer's max-%-ADV constraint at the position-sizer layer
    # (the legacy per-name path that runs when the optimizer isn't authoritative)
    # so an illiquid ENTER can't be sized to a notional the market can't absorb.
    # FAIL-SOFT: no ADV coverage (adv_usd None/≤0/NaN) → no cap, legacy sizing
    # preserved. The cost of thin liquidity is priced by the optimizer's √-impact
    # term; this is the HARD ceiling. Default ON at a conservative 10% of ADV —
    # a single new position is a one-side trade, so 10% leaves ample headroom
    # under the optimizer's 5%-of-ADV per-solve participation cap once cut over.
    adv_cap_applied = False
    adv_size_cap_pct = config.get("adv_size_cap_pct_adv", 0.10)
    if (
        config.get("adv_size_cap_enabled", True)
        and adv_size_cap_pct is not None and adv_size_cap_pct > 0
        and adv_usd is not None
    ):
        try:
            adv_f = float(adv_usd)
        except (TypeError, ValueError):
            adv_f = 0.0
        if adv_f > 0.0 and adv_f == adv_f:  # positive + not NaN
            adv_cap_dollars = adv_size_cap_pct * adv_f
            if dollar_size > adv_cap_dollars:
                dollar_size = adv_cap_dollars
                position_weight = (
                    dollar_size / portfolio_nav if portfolio_nav and portfolio_nav > 0
                    else position_weight
                )
                adv_cap_applied = True

    shares = math.floor(dollar_size / current_price) if current_price and current_price > 0 else 0

    # Minimum position size check (Task 1.2)
    if dollar_size < config.get("min_position_dollar", 500):
        shares = 0

    logger.debug(
        f"{ticker} sizing: n={n} base={base_weight:.3f} sector_adj={sector_adj} "
        f"conviction_adj={conviction_adj} upside_adj={upside_adj} "
        f"dd_mult={drawdown_multiplier} derisk_mult={derisk_multiplier} atr_adj={atr_adj} "
        f"confidence_adj={confidence_adj} staleness_adj={staleness_adj} "
        f"earnings_adj={earnings_adj} coverage_adj={coverage_adj} "
        f"stance_adj={stance_adj} regime_adj={regime_adj} "
        f"barrier_win_prob_adj={barrier_win_prob_adj} "
        f"adv_cap_applied={adv_cap_applied} "
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
        "derisk_multiplier": derisk_multiplier,
        "atr_adj": atr_adj,
        "confidence_adj": confidence_adj,
        "staleness_adj": staleness_adj,
        "earnings_adj": earnings_adj,
        "coverage_adj": coverage_adj,
        "stance_adj": stance_adj,
        "regime_adj": regime_adj,
        "barrier_win_prob_adj": barrier_win_prob_adj,
        "adv_cap_applied": adv_cap_applied,
    }
