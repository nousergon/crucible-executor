"""
Daily trading signal generator for population-based architecture.

Combines three inputs to produce executor-compatible signal envelopes:
  1. Investment population (weekly, from Research) — "what to consider"
  2. Technical indicators (computed here from OHLCV) — "when to trade"
  3. GBM predictions (daily, from Predictor) — advisory input + veto gate

The output format matches signal_reader.get_actionable_signals() exactly,
so all downstream logic (risk guard, position sizer, exit manager, order
placement) works unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from executor.technical_scorer import (
    compute_indicators_from_ohlcv,
    compute_momentum_percentiles,
    compute_technical_score,
)

logger = logging.getLogger(__name__)


# ── Predictions reader ────────────────────────────────────────────────────────

def read_predictions(s3_bucket: str) -> dict[str, dict]:
    """
    Read predictor/predictions/latest.json from S3.

    Returns: {ticker: {predicted_direction, prediction_confidence, predicted_alpha,
                       p_up, p_flat, p_down}}
             Empty dict if predictions not available.
    """
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(
            Bucket=s3_bucket, Key="predictor/predictions/latest.json"
        )
        data = json.loads(obj["Body"].read())
        preds = data.get("predictions", [])
        result = {p["ticker"]: p for p in preds if "ticker" in p}
        logger.info("Predictions loaded | n=%d", len(result))
        return result
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning(
                "predictions/latest.json not found — running without GBM input"
            )
            return {}
        raise
    except Exception:
        logger.warning("Failed to load predictions — running without GBM input", exc_info=True)
        return {}


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_trading_signals(
    population: list[dict],
    predictions: dict[str, dict],
    price_histories: dict[str, list[dict]],
    market_regime: str,
    sector_ratings: dict,
    config: dict,
) -> dict:
    """
    Generate daily trading signals from population + technical scores + GBM.

    For each population stock:
    1. Compute 5 technical indicators from OHLCV (price_histories)
    2. Score via compute_technical_score() → 0-100 technical score
    3. Enrich with GBM alpha (±max_enrichment pts)
    4. Apply GBM veto: DOWN + high confidence → demote to HOLD
    5. Assign signal: ENTER / EXIT / REDUCE / HOLD

    Args:
        population: list of population dicts from population_reader
        predictions: {ticker: prediction_dict} from read_predictions()
        price_histories: {ticker: [{date, open, high, low, close}, ...]}
                         from price_cache.load_price_histories()
        market_regime: 'bull' | 'neutral' | 'caution' | 'bear'
        sector_ratings: {sector: {modifier, rating, ...}}
        config: risk.yaml config dict with 'trading' section

    Returns:
        Signal envelope matching signal_reader format:
        {
            "market_regime": str,
            "sector_ratings": dict,
            "universe": [signal_dict, ...],
            "buy_candidates": [signal_dict, ...]
        }
    """
    trading_cfg = config.get("trading", {})
    min_technical_score = trading_cfg.get("min_technical_score", 60)
    gbm_veto_confidence = trading_cfg.get("gbm_veto_confidence", 0.65)
    gbm_enrichment_max = trading_cfg.get("gbm_enrichment_max", 10.0)
    exit_score_threshold = trading_cfg.get("exit_score_threshold", 30)

    # ── Step 1: Compute technical indicators for all population stocks ──
    indicators_by_ticker: dict[str, dict] = {}
    for stock in population:
        ticker = stock["ticker"]
        history = price_histories.get(ticker)
        if history:
            indicators = compute_indicators_from_ohlcv(history)
            if indicators is not None:
                indicators_by_ticker[ticker] = indicators

    # ── Step 2: Compute momentum percentiles across population ──
    momentum_data = {
        ticker: ind.get("momentum_20d")
        for ticker, ind in indicators_by_ticker.items()
    }
    momentum_percentiles = compute_momentum_percentiles(momentum_data)

    # ── Step 3: Score each stock and assign signals ──
    universe: list[dict] = []
    buy_candidates: list[dict] = []

    for stock in population:
        ticker = stock["ticker"]
        indicators = indicators_by_ticker.get(ticker)

        if indicators is None:
            # No price data — can't score, default to HOLD
            signal_dict = _build_signal_dict(
                stock=stock,
                trading_score=50.0,
                signal="HOLD",
                market_regime=market_regime,
            )
            universe.append(signal_dict)
            continue

        # Technical score
        tech_score = compute_technical_score(
            indicators,
            market_regime=market_regime,
            momentum_percentile=momentum_percentiles.get(ticker),
        )

        # GBM enrichment
        pred = predictions.get(ticker, {})
        gbm_adjustment = _compute_gbm_adjustment(pred, gbm_enrichment_max)
        trading_score = round(
            max(0.0, min(100.0, tech_score + gbm_adjustment)), 2
        )

        # GBM veto check
        gbm_vetoed = _check_gbm_veto(pred, gbm_veto_confidence)

        # Signal assignment
        signal = _assign_signal(
            stock=stock,
            trading_score=trading_score,
            gbm_vetoed=gbm_vetoed,
            min_technical_score=min_technical_score,
            exit_score_threshold=exit_score_threshold,
        )

        signal_dict = _build_signal_dict(
            stock=stock,
            trading_score=trading_score,
            signal=signal,
            market_regime=market_regime,
            technical_score=tech_score,
            gbm_adjustment=gbm_adjustment,
            gbm_vetoed=gbm_vetoed,
            prediction=pred,
        )

        if signal == "ENTER":
            buy_candidates.append(signal_dict)
        else:
            universe.append(signal_dict)

    n_enter = len(buy_candidates)
    n_exit = sum(1 for s in universe if s.get("signal") == "EXIT")
    n_hold = sum(1 for s in universe if s.get("signal") == "HOLD")
    n_reduce = sum(1 for s in universe if s.get("signal") == "REDUCE")
    logger.info(
        "Trading signals generated | regime=%s | ENTER=%d EXIT=%d REDUCE=%d HOLD=%d",
        market_regime,
        n_enter,
        n_exit,
        n_reduce,
        n_hold,
    )

    return {
        "market_regime": market_regime,
        "sector_ratings": sector_ratings,
        "universe": universe,
        "buy_candidates": buy_candidates,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_gbm_adjustment(
    prediction: dict,
    max_enrichment: float,
) -> float:
    """
    Compute GBM enrichment adjustment (±max_enrichment points).

    Uses p_up - p_down from GBM predictions, scaled by confidence.
    Returns 0.0 if prediction not available.
    """
    if not prediction:
        return 0.0

    p_up = prediction.get("p_up")
    p_down = prediction.get("p_down")
    confidence = prediction.get("prediction_confidence", 0.0)

    if p_up is None or p_down is None:
        return 0.0

    # (p_up - p_down) in [-1, +1]; scale to max ±max_enrichment pts
    direction_signal = (p_up - p_down) * max_enrichment * confidence
    return round(max(-max_enrichment, min(max_enrichment, direction_signal)), 2)


def _check_gbm_veto(prediction: dict, veto_confidence: float) -> bool:
    """
    Check if GBM veto flag is set (negative alpha + bottom-half combined rank).

    Falls back to legacy check (DOWN + high confidence) if gbm_veto flag
    is not present in the prediction (backward compatibility).
    """
    if not prediction:
        return False

    # New veto: predictor computes and sets gbm_veto directly
    if "gbm_veto" in prediction:
        return bool(prediction["gbm_veto"])

    # Legacy fallback: direction + confidence threshold
    return (
        prediction.get("predicted_direction") == "DOWN"
        and prediction.get("prediction_confidence", 0.0) >= veto_confidence
    )


def _assign_signal(
    stock: dict,
    trading_score: float,
    gbm_vetoed: bool,
    min_technical_score: float,
    exit_score_threshold: float,
) -> str:
    """
    Assign trading signal based on long-term rating + technical score + GBM veto.

    ENTER:  long_term_rating == "BUY" AND trading_score >= min_technical AND not vetoed
    EXIT:   long_term_rating == "SELL" OR trading_score < exit_threshold
    REDUCE: long_term_rating == "HOLD" AND conviction == "declining"
    HOLD:   everything else (including GBM-vetoed ENTER candidates)
    """
    lt_rating = stock.get("long_term_rating", "HOLD")
    conviction = stock.get("conviction", "stable")

    # EXIT conditions
    if lt_rating == "SELL":
        return "EXIT"
    if trading_score < exit_score_threshold:
        return "EXIT"

    # ENTER conditions
    if (
        lt_rating == "BUY"
        and trading_score >= min_technical_score
        and not gbm_vetoed
    ):
        return "ENTER"

    # REDUCE condition
    if lt_rating == "HOLD" and conviction == "declining":
        return "REDUCE"

    return "HOLD"


def _build_signal_dict(
    stock: dict,
    trading_score: float,
    signal: str,
    market_regime: str,
    technical_score: Optional[float] = None,
    gbm_adjustment: Optional[float] = None,
    gbm_vetoed: bool = False,
    prediction: Optional[dict] = None,
) -> dict:
    """
    Build a signal dict in the format expected by get_actionable_signals().

    Keys: ticker, signal, score, conviction, rating, sector,
          price_target_upside, thesis_summary, long_term_score,
          long_term_rating, predicted_direction, prediction_confidence,
          gbm_veto
    """
    pred = prediction or {}

    # Map long_term_rating to the "rating" field expected downstream
    lt_rating = stock.get("long_term_rating", "HOLD")
    rating = lt_rating  # BUY / HOLD / SELL

    return {
        "ticker": stock["ticker"],
        "signal": signal,
        "score": trading_score,
        "conviction": stock.get("conviction", "stable"),
        "rating": rating,
        "sector": stock.get("sector", "Unknown"),
        "price_target_upside": stock.get("price_target_upside"),
        "thesis_summary": stock.get("thesis_summary", ""),
        "long_term_score": stock.get("long_term_score", 50.0),
        "long_term_rating": lt_rating,
        # Technical breakdown (for logging / dashboard)
        "technical_score": technical_score,
        "gbm_adjustment": gbm_adjustment,
        "gbm_veto": gbm_vetoed,
        # GBM prediction passthrough (for trade_logger)
        "predicted_direction": pred.get("predicted_direction"),
        "prediction_confidence": pred.get("prediction_confidence"),
    }
