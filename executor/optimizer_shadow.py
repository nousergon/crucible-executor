"""
Shadow-mode portfolio optimizer wrapper — PR 2 of the portfolio-optimizer arc.

The optimizer kernel (PR 1, executor/portfolio_optimizer.py) is a pure-numpy
in/out function. This module assembles optimizer inputs from the existing
main.py state (signals, predictions, positions, price histories), calls the
kernel, and logs the resulting target weights + diagnostics to S3.

Production behaviour is unchanged — no orders are placed based on the
optimizer's output. The shadow log is the primary observability artifact for
deciding whether to cut over (PR 5 of the arc).

S3 layout:
    predictor/optimizer_shadow/{run_date}.json   ← per-day snapshot
    predictor/optimizer_shadow/latest.json       ← convenience pointer

This wrapper NEVER raises into the legacy planner path. All exceptions are
caught, logged at WARNING, and a sentinel is written to S3 so the absence
of a shadow log is itself flagged in the daily diagnostic surface.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
import pandas as pd

from executor.portfolio_optimizer import (
    OPTIMIZER_CONFIG_DEFAULTS,
    solve_target_weights,
)

logger = logging.getLogger(__name__)

_SPY = "SPY"
_CASH = "CASH"
_BENCH_SECTOR = "__benchmark__"
_CASH_SECTOR = "__cash__"
_CASH_ALPHA_HINT = -1e-6
_RETURNS_LOOKBACK_DAYS = 252
_MIN_RETURNS_FOR_COV = 60


def run_shadow_optimizer(
    signals_raw: dict,
    predictions_by_ticker: dict[str, dict],
    current_positions: dict[str, dict],
    portfolio_nav: float,
    price_histories: dict[str, pd.DataFrame],
    config: dict,
    signals_bucket: str,
    run_date: str,
    legacy_orders: list[dict] | None = None,
    s3_client=None,
) -> dict | None:
    """
    Run the optimizer in shadow mode and write the result to S3.

    Returns the shadow log dict on success, or None on any failure. Never
    raises — exceptions are caught + logged + a sentinel written to S3 so
    the absence of a real shadow log is itself observable.
    """
    try:
        log = _build_and_solve(
            signals_raw=signals_raw,
            predictions_by_ticker=predictions_by_ticker,
            current_positions=current_positions,
            portfolio_nav=portfolio_nav,
            price_histories=price_histories,
            config=config,
            run_date=run_date,
            legacy_orders=legacy_orders or [],
        )
        _write_shadow_log_to_s3(log, signals_bucket, run_date, s3_client)
        logger.info(
            f"Shadow optimizer OK: status={log['diagnostics']['status']} "
            f"n_active={log['diagnostics']['n_active_positions']} "
            f"vol_ann={log['diagnostics']['portfolio_vol_ann']:.3f} "
            f"active_share={log['diagnostics']['active_share_vs_spy']:.3f}"
        )
        return log
    except Exception as e:
        logger.warning(f"Shadow optimizer failed (non-blocking): {e}", exc_info=True)
        sentinel = {
            "run_date": run_date,
            "shadow_status": "failed",
            "error": repr(e),
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_shadow_log_to_s3(sentinel, signals_bucket, run_date, s3_client)
        except Exception as inner:
            logger.warning(f"Shadow sentinel write also failed: {inner}")
        return None


def _build_and_solve(
    signals_raw: dict,
    predictions_by_ticker: dict[str, dict],
    current_positions: dict[str, dict],
    portfolio_nav: float,
    price_histories: dict[str, pd.DataFrame],
    config: dict,
    run_date: str,
    legacy_orders: list[dict],
) -> dict:
    optimizer_cfg = {
        **OPTIMIZER_CONFIG_DEFAULTS,
        **config.get("portfolio_optimizer", {}),
    }
    tickers = _build_universe(
        signals_raw, predictions_by_ticker, current_positions, price_histories,
    )
    N = len(tickers)
    spy_idx = tickers.index(_SPY)
    cash_idx = tickers.index(_CASH)

    signals_by_ticker = signals_raw.get("signals", {})
    alpha_hat = _build_alpha_hat(tickers, predictions_by_ticker, spy_idx, cash_idx)
    returns_panel = _build_returns_panel(tickers, price_histories, cash_idx)
    w_prev = _build_w_prev(tickers, current_positions, portfolio_nav, cash_idx, optimizer_cfg)
    sectors = _build_sectors(tickers, signals_by_ticker, spy_idx, cash_idx)
    stance_caps = _build_stance_caps(
        tickers, signals_by_ticker, predictions_by_ticker,
        config, optimizer_cfg, spy_idx, cash_idx,
    )
    eligibility = _build_eligibility(
        tickers, signals_by_ticker, predictions_by_ticker,
        current_positions, config, spy_idx, cash_idx,
    )

    result = solve_target_weights(
        tickers=tickers,
        alpha_hat=alpha_hat,
        returns_panel=returns_panel,
        w_prev=w_prev,
        sectors=sectors,
        stance_caps=stance_caps,
        eligibility=eligibility,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
        cfg=optimizer_cfg,
    )

    would_be_trades = _compute_trade_deltas(
        tickers, result.weights, w_prev, portfolio_nav, optimizer_cfg,
    )

    return {
        "run_date": run_date,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "shadow_status": "ok",
        "portfolio_nav": float(portfolio_nav),
        "n_tickers": N,
        "tickers": tickers,
        "target_weights": [float(x) for x in result.weights],
        "current_weights": [float(x) for x in w_prev],
        "alpha_hat": [float(x) for x in alpha_hat],
        "eligibility": [bool(x) for x in eligibility],
        "stance_caps": [float(x) for x in stance_caps],
        "sectors": sectors,
        "would_be_trades": would_be_trades,
        "diagnostics": result.diagnostics,
        "legacy_orders": [_redact_order(o) for o in legacy_orders],
        "optimizer_cfg": optimizer_cfg,
    }


def _build_universe(
    signals_raw: dict,
    predictions_by_ticker: dict,
    current_positions: dict,
    price_histories: dict[str, pd.DataFrame],
) -> list[str]:
    candidates: set[str] = set()
    candidates.update(predictions_by_ticker.keys())
    candidates.update(current_positions.keys())
    candidates.update(_extract_universe_tickers(signals_raw.get("universe", [])))

    candidates.discard(_SPY)
    candidates.discard(_CASH)
    eligible = sorted(t for t in candidates if _has_usable_history(t, price_histories))

    if _SPY not in price_histories or not _has_usable_history(_SPY, price_histories):
        raise RuntimeError(
            "Shadow optimizer requires SPY price history; not found in price_histories. "
            "Confirm executor's load_price_histories includes SPY (line ~1096 in main.py)."
        )

    return eligible + [_SPY, _CASH]


def _extract_universe_tickers(universe_list: Any) -> list[str]:
    """Normalize `signals_raw['universe']` to a list of ticker strings.

    Production signals.json emits the universe as a list of per-ticker dicts
    (`{"ticker": "COST", "signal": "ENTER", "score": 55.3, ...}`); legacy /
    minimal payloads emit a flat list of ticker strings. Accept both shapes —
    unknown shapes are skipped silently so the wrapper degrades to a
    smaller universe rather than failing the whole optimizer call.
    """
    if not isinstance(universe_list, list):
        return []
    out: list[str] = []
    for el in universe_list:
        if isinstance(el, str):
            out.append(el)
        elif isinstance(el, dict):
            t = el.get("ticker")
            if isinstance(t, str) and t:
                out.append(t)
    return out


def _has_usable_history(ticker: str, price_histories: dict[str, pd.DataFrame]) -> bool:
    df = price_histories.get(ticker)
    if df is None or len(df) < _MIN_RETURNS_FOR_COV + 1:
        return False
    if "close" not in df.columns:
        return False
    return True


def _build_alpha_hat(
    tickers: list[str],
    predictions_by_ticker: dict[str, dict],
    spy_idx: int,
    cash_idx: int,
) -> np.ndarray:
    alpha = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if i == spy_idx:
            alpha[i] = 0.0
            continue
        if i == cash_idx:
            alpha[i] = _CASH_ALPHA_HINT
            continue
        pred = predictions_by_ticker.get(t, {})
        raw_alpha = pred.get("predicted_alpha") or pred.get("canonical_predicted_alpha") or 0.0
        try:
            alpha[i] = float(raw_alpha)
        except (TypeError, ValueError):
            alpha[i] = 0.0
        if not math.isfinite(alpha[i]):
            alpha[i] = 0.0
    return alpha


def _build_returns_panel(
    tickers: list[str],
    price_histories: dict[str, pd.DataFrame],
    cash_idx: int,
) -> np.ndarray:
    """Daily LOG returns for covariance estimation.

    Convention matches the 2026-05-09 21d log-domain canonical-alpha cutover
    (alpha-engine-predictor PRs A-E + 2026-05-10 transition arc): alpha_hat
    consumed by the optimizer is 21d log alpha (predictor's predicted_alpha
    field), so the Sigma fed in must be in the same log-units family. Daily
    log variance compounds linearly to higher horizons (Var_T = T · Var_daily
    for iid log returns).
    """
    series_by_ticker: dict[str, pd.Series] = {}
    for i, t in enumerate(tickers):
        if i == cash_idx:
            continue
        df = price_histories[t]
        close = df["close"].tail(_RETURNS_LOOKBACK_DAYS + 1)
        s = np.log(close).diff().dropna()
        series_by_ticker[t] = s

    aligned = pd.DataFrame(series_by_ticker).dropna()
    if aligned.shape[0] < _MIN_RETURNS_FOR_COV:
        raise RuntimeError(
            f"Aligned returns panel has only {aligned.shape[0]} rows; "
            f"need ≥{_MIN_RETURNS_FOR_COV} for covariance estimation. "
            "Universe likely has tickers with non-overlapping histories — "
            "filter pre-call."
        )

    panel = np.zeros((aligned.shape[0], len(tickers)))
    for i, t in enumerate(tickers):
        if i == cash_idx:
            panel[:, i] = 0.0
        else:
            panel[:, i] = aligned[t].values
    return panel


def _build_w_prev(
    tickers: list[str],
    current_positions: dict[str, dict],
    portfolio_nav: float,
    cash_idx: int,
    optimizer_cfg: dict,
) -> np.ndarray:
    w = np.zeros(len(tickers))
    if portfolio_nav <= 0:
        w[cash_idx] = 1.0
        return w
    for i, t in enumerate(tickers):
        if i == cash_idx:
            continue
        pos = current_positions.get(t, {})
        mv = pos.get("market_value", 0.0) or 0.0
        try:
            w[i] = float(mv) / portfolio_nav
        except (TypeError, ValueError, ZeroDivisionError):
            w[i] = 0.0
    deployed = w.sum()
    w[cash_idx] = max(0.0, 1.0 - deployed)
    return w


def _build_sectors(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    spy_idx: int,
    cash_idx: int,
) -> list[str]:
    out: list[str] = []
    for i, t in enumerate(tickers):
        if i == spy_idx:
            out.append(_BENCH_SECTOR)
        elif i == cash_idx:
            out.append(_CASH_SECTOR)
        else:
            sector = signals_by_ticker.get(t, {}).get("sector", "Unknown")
            out.append(str(sector) if sector else "Unknown")
    return out


def _build_stance_caps(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    predictions_by_ticker: dict[str, dict],
    config: dict,
    optimizer_cfg: dict,
    spy_idx: int,
    cash_idx: int,
) -> np.ndarray:
    base_cap = float(config.get("max_position_pct", 0.08))
    stance_multipliers = {
        "momentum": float(config.get("stance_size_momentum", 1.0)),
        "value":    float(config.get("stance_size_value",    0.7)),
        "quality":  float(config.get("stance_size_quality",  0.8)),
        "catalyst": float(config.get("stance_size_catalyst", 0.6)),
    }
    caps = np.full(len(tickers), base_cap)
    caps[spy_idx] = 1.0
    caps[cash_idx] = 1.0
    for i, t in enumerate(tickers):
        if i in (spy_idx, cash_idx):
            continue
        pred = predictions_by_ticker.get(t, {})
        stance = pred.get("stance") or signals_by_ticker.get(t, {}).get("stance")
        if stance and stance in stance_multipliers:
            caps[i] = base_cap * stance_multipliers[stance]
    return caps


def _build_eligibility(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    predictions_by_ticker: dict[str, dict],
    current_positions: dict[str, dict],
    config: dict,
    spy_idx: int,
    cash_idx: int,
) -> np.ndarray:
    min_score = float(config.get("min_score_to_enter", 57))
    eligibility = np.ones(len(tickers), dtype=bool)
    for i, t in enumerate(tickers):
        if i in (spy_idx, cash_idx):
            continue
        sig = signals_by_ticker.get(t, {})
        pred = predictions_by_ticker.get(t, {})
        is_held = t in current_positions

        if sig.get("signal") == "EXIT":
            eligibility[i] = False
            continue
        if pred.get("gbm_veto") is True:
            eligibility[i] = False
            continue
        if is_held:
            continue
        score = sig.get("score")
        if score is None or float(score) < min_score:
            eligibility[i] = False
    return eligibility


def _compute_trade_deltas(
    tickers: list[str],
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    portfolio_nav: float,
    optimizer_cfg: dict,
) -> list[dict]:
    band = float(optimizer_cfg.get("rebalance_band_pct", 0.005))
    trades: list[dict] = []
    for i, t in enumerate(tickers):
        if t == _CASH:
            continue
        delta_pct = float(target_weights[i] - current_weights[i])
        if abs(delta_pct) < band:
            continue
        delta_dollars = delta_pct * float(portfolio_nav)
        trades.append({
            "ticker": t,
            "action": "BUY" if delta_pct > 0 else "SELL",
            "delta_weight": round(delta_pct, 6),
            "delta_dollars": round(delta_dollars, 2),
            "target_weight": round(float(target_weights[i]), 6),
            "current_weight": round(float(current_weights[i]), 6),
        })
    return trades


def _redact_order(order: dict) -> dict:
    keep = {"ticker", "action", "shares", "limit_price", "dollar_size",
            "position_pct", "stance", "score", "signal_type"}
    return {k: order.get(k) for k in keep if k in order}


def _write_shadow_log_to_s3(
    log: dict, bucket: str, run_date: str, s3_client=None,
) -> None:
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(log, default=str, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=f"predictor/optimizer_shadow/{run_date}.json",
        Body=body,
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket,
        Key="predictor/optimizer_shadow/latest.json",
        Body=body,
        ContentType="application/json",
    )
