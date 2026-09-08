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
from datetime import UTC, datetime
from typing import Any

import boto3
import numpy as np
import pandas as pd

from executor.alpha_contract import (
    _numeric_alpha,
    assert_optimizer_anchor,
)
from executor.portfolio_optimizer import (
    OPTIMIZER_CONFIG_DEFAULTS,
    _estimate_covariance_daily,
    solve_target_weights,
)

logger = logging.getLogger(__name__)

_SPY = "SPY"
_CASH = "CASH"
_BENCH_SECTOR = "__benchmark__"
_CASH_SECTOR = "__cash__"
_CASH_ALPHA_HINT = -1e-6
# Below this |Δweight| the optimizer asked for no trade at all (a sub-dollar
# move on any realistic book — solver noise, not an intention). Used only to
# keep `band_dropped_trades` to trades the band actually REMOVED, rather than
# to the whole universe sitting at zero on both sides.
_DUST_DELTA = 1e-6
_RETURNS_LOOKBACK_DAYS = 252
_MIN_RETURNS_FOR_COV = 60

# config#1057 inc 2: the backtester auto-tunes the MVO optimizer's OWN params
# and writes them to config/portfolio_optimizer.json. Mirror the WRITER's
# writable-set + bounds here as DEFENSE IN DEPTH — the executor never trusts the
# written value: it consumes ONLY these two knobs and re-clamps them, so even a
# corrupt/out-of-band file can't move the live solver outside a sane band.
_AUTO_TUNED_WRITABLE = ("risk_aversion", "tcost_bps")
# Read-side re-clamp band (defense in depth). Generic public default floor (3.0)
# only — the OPERATING risk_aversion floor is a private risk-policy variable,
# overridden via the private config repo
# (alpha-engine-config/executor/risk.yaml, key
# `portfolio_optimizer.tuner_risk_aversion_floor`; resolved by
# config_loader.py — see its search order) so an aggressive floor (e.g.
# 1.0) never ships in this public repo (divergence policy: alpha-bearing values
# stay private). MUST stay in lockstep with the tuner write-side override
# alpha-engine-backtester/optimizer/portfolio_optimizer_optimizer.py
# (portfolio_optimizer_tuner.risk_aversion_floor).
_AUTO_TUNED_BOUNDS = {
    "risk_aversion": (3.0, 10.0),
    "tcost_bps": (1.0, 20.0),
}


def _load_auto_tuned_optimizer_cfg(config: dict, s3_client=None) -> dict:
    """Read the backtester's auto-tuned MVO params (risk_aversion × tcost_bps)
    from ``config/portfolio_optimizer.json`` and return them ALLOWLISTED +
    CLAMPED (config#1057 inc 2).

    Fail-safe: any absence / read error / parse error returns ``{}`` so the
    solver falls back to the YAML/defaults — the auto-tuner is never a hard
    dependency. Gated by ``portfolio_optimizer.consume_auto_tuned`` (default
    True), the executor's independent kill-switch (separate from the
    backtester's apply flag)."""
    po_cfg = config.get("portfolio_optimizer", {}) or {}
    if not po_cfg.get("consume_auto_tuned", True):
        return {}
    bucket = config.get("signals_bucket")
    if not bucket:
        return {}
    try:
        import json

        s3 = s3_client or boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/portfolio_optimizer.json")
        data = json.loads(obj["Body"].read())
    except Exception as e:  # noqa: BLE001 — absence/error → fall back to defaults
        # (a) the auto-tuned optimizer params S3 object is absent or
        # unreadable — the existing comment above already names this as
        # expected (no auto-tune run has published params yet).
        # (c) not recorded elsewhere — deliberate carve-out
        # (alpha-engine-config-I10031): an expected-absence probe with a
        # defined fallback (YAML/defaults), same class as the
        # connection-teardown carve-outs.
        logger.debug("no auto-tuned optimizer params (%s) — using YAML/defaults", e)
        return {}

    out: dict = {}
    for k in _AUTO_TUNED_WRITABLE:
        if k not in data:
            continue
        try:
            v = float(data[k])
        except (TypeError, ValueError):
            logger.warning("auto-tuned %s=%r non-numeric — ignored", k, data.get(k))
            continue
        lo, hi = _AUTO_TUNED_BOUNDS[k]
        # Private risk-policy override: the operating risk_aversion floor lives in
        # the config-repo risk.yaml (alpha-engine-config/executor/risk.yaml),
        # not this public default (divergence policy).
        if k == "risk_aversion":
            _floor_override = po_cfg.get("tuner_risk_aversion_floor")
            if _floor_override is not None:
                lo = float(_floor_override)
        cv = min(max(v, lo), hi)
        if cv != v:
            logger.warning(
                "auto-tuned %s=%s clamped to %s (band [%s, %s])",
                k,
                v,
                cv,
                lo,
                hi,
            )
        out[k] = cv
    if out:
        logger.info("Consuming auto-tuned optimizer params from S3: %s", out)
    return out


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
    price_histories_requested: set[str] | None = None,
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
            signals_bucket=signals_bucket,
            price_histories_requested=price_histories_requested,
        )
        # L4515 turnover tripwire: band-check the executed turnover (daily +
        # rolling) BEFORE the artifact write so the verdict rides the daily
        # shadow log. check_turnover_tripwire never raises (sentinel on error).
        from executor.turnover_tripwire import check_turnover_tripwire

        log["turnover_tripwire"] = check_turnover_tripwire(
            log.get("diagnostics") or {},
            log.get("optimizer_cfg") or {},
            signals_bucket,
            run_date,
            s3_client,
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
            "written_at_utc": datetime.now(UTC).isoformat(),
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
    signals_bucket: str | None = None,
    price_histories_requested: set[str] | None = None,
) -> dict:
    optimizer_cfg = {
        **OPTIMIZER_CONFIG_DEFAULTS,
        **config.get("portfolio_optimizer", {}),
        # config#1057 inc 2: the backtester's auto-tuned risk_aversion × tcost_bps
        # win over the static YAML (allowlisted + re-clamped; empty on absence so
        # the solver is unchanged until a tuned config exists).
        **_load_auto_tuned_optimizer_cfg(config),
    }

    # ── Expectancy-gated de-risk floor (config-I2820 / PR2071) ───────────────
    # When the standing de-risk stance is active, the MVO risk_aversion is
    # floored at configured_risk_aversion / derisk_sizing_multiplier — applied
    # AFTER the auto-tuned override above so a de-risk breach can only make
    # the solve MORE conservative than whatever risk_aversion the YAML/tuner
    # otherwise resolved to (apply_risk_aversion_floor is a max(), never a
    # decrease). Fail-loud by construction: evaluate_derisk_gate raises
    # DeriskGateConfigError (propagated, not caught here) when the flag is
    # enabled but config/ledger is malformed — a live-sizing safety gate must
    # not silently degrade to an un-floored solve. No-op (1.0x, floor=None)
    # when the flag is absent/false, so this is bit-identical to pre-this-PR
    # behavior until an operator opts in.
    from executor.derisk_gate import apply_risk_aversion_floor, evaluate_derisk_gate

    _derisk_gate = evaluate_derisk_gate(config, bucket=signals_bucket)
    if _derisk_gate.active:
        optimizer_cfg["risk_aversion"] = apply_risk_aversion_floor(
            optimizer_cfg["risk_aversion"],
            _derisk_gate,
        )
        logger.warning(
            "[derisk_gate] optimizer risk_aversion floored at %.2f (%s)",
            optimizer_cfg["risk_aversion"],
            _derisk_gate.reason,
        )
    dropped_candidates: list[dict] = []
    tickers = _build_universe(
        signals_raw,
        predictions_by_ticker,
        current_positions,
        price_histories,
        price_histories_requested=price_histories_requested,
        dropped_out=dropped_candidates,
    )
    N = len(tickers)
    spy_idx = tickers.index(_SPY)
    cash_idx = tickers.index(_CASH)

    signals_by_ticker = signals_raw.get("signals", {})
    # Raises AlphaAnchorError on a mixed- or undeclared-anchor batch before
    # anything is solved (alpha-engine-config-I7337). The returned block is
    # published on the artifact below so the check's verdict is a number an
    # operator can read, not an inference from the absence of a traceback.
    alpha_anchor = assert_optimizer_anchor(
        tickers,
        predictions_by_ticker,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
    )
    alpha_hat = _build_alpha_hat(tickers, predictions_by_ticker, spy_idx, cash_idx)
    # TOTAL — the conviction gate's input; its band was derived at this scale.
    alpha_uncertainty = _build_alpha_uncertainty(
        tickers,
        predictions_by_ticker,
        spy_idx,
        cash_idx,
    )
    # EPISTEMIC — sqrt(x'Sigma_w x), the ONLY vector the GUW Omega is built
    # from (alpha-engine-config-I9452). All-NaN on artifacts written before
    # crucible-predictor PR596 deployed (2026-08-31); the solve then declares
    # the penalty inoperative with a reason rather than falling back to the
    # total, which is a uniform ridge that double-counts Sigma.
    alpha_uncertainty_epistemic = _build_alpha_uncertainty(
        tickers,
        predictions_by_ticker,
        spy_idx,
        cash_idx,
        field="predicted_alpha_std_epistemic",
    )
    returns_panel = _build_returns_panel(tickers, price_histories, cash_idx)
    # Σ_daily (pre-horizon) is persisted below for the daemon's intraday
    # re-solve. Computed deterministically from the SAME panel + cfg the morning
    # solve uses internally, so the cached matrix is bit-identical to what
    # solve_target_weights estimates — the re-solve is mechanism-identical.
    sigma_daily = _estimate_covariance_daily(returns_panel, optimizer_cfg)
    w_prev = _build_w_prev(tickers, current_positions, portfolio_nav, cash_idx, optimizer_cfg)
    sectors = _build_sectors(tickers, signals_by_ticker, spy_idx, cash_idx)
    stance_caps = _build_stance_caps(
        tickers,
        signals_by_ticker,
        predictions_by_ticker,
        config,
        optimizer_cfg,
        spy_idx,
        cash_idx,
    )
    eligibility, eligibility_reasons = _build_eligibility(
        tickers,
        signals_by_ticker,
        predictions_by_ticker,
        current_positions,
        config,
        spy_idx,
        cash_idx,
    )
    # Per-name ADV$ from the scanner tradeability artifact (crucible-research#343)
    # drives the participation-aware √-impact cost term + max-%-ADV constraint
    # (config#1401). Fail-soft: absent artifact → all-NaN adv_usd → the optimizer
    # degrades to the flat L1 tcost penalty (bit-identical pre-tradeability).
    adv_usd, adv_coverage = _build_adv_usd(
        tickers,
        signals_bucket,
        run_date,
        spy_idx,
        cash_idx,
    )
    # Per-name daily σ for the Almgren-Chriss σ-scaling of the impact term.
    # Reuse the returns panel we already built (no extra I/O); σ-agnostic when
    # a column has no usable history.
    name_sigma = _build_name_sigma(returns_panel, spy_idx, cash_idx)

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
        alpha_uncertainty=alpha_uncertainty,
        alpha_uncertainty_epistemic=alpha_uncertainty_epistemic,
        adv_usd=adv_usd,
        portfolio_notional=float(portfolio_nav) if portfolio_nav and portfolio_nav > 0 else None,
        name_sigma=name_sigma,
    )

    # Large-move flag: the book is moving hard today, for a reason the SOLVE
    # could not itself justify. Two reasons raise it, and the message says
    # WHICH (alpha-engine-config-I7346):
    #   * executed_turnover_above_flag — executed one-way turnover exceeds
    #     large_move_turnover_flag outright (only reachable when the daily
    #     budget is off or set above the flag).
    #   * turnover_budget_binding — the daily turnover budget bound the solve
    #     on a day the conviction gate did NOT throttle, i.e. the optimizer
    #     wanted to move more than it was allowed to on a signal the model
    #     stands behind. That is a real operator fact.
    #
    # `conviction_throttled_budget_binding` is deliberately NOT flagged
    # (alpha-engine-config-I9315). Between 2026-08-17 and 2026-08-31 this alert
    # fired on 8 of 12 sessions saying the budget bound the solve, and it was
    # RIGHT on the mechanics and useless as a signal: the budget bound because
    # the predictor's cross-sectional alpha spread had collapsed to ~1/30th of
    # its own published per-name sigma, so the optimizer was trying to reshuffle
    # a book of statistically tied names. The conviction gate now refuses that
    # move; the budget still binds, but on the THROTTLED budget, and a guard
    # working as designed is not an incident. The fact is not lost — the whole
    # conviction block is persisted in this artifact's diagnostics and the
    # rolling turnover tripwire reads it and names the throttle as the driver.
    #
    # The message carries the DRIVER it found. It does not ask the operator to
    # go and read the shadow logs, and it does not ask for an approval there is
    # no channel to give: an alert whose only instruction is unanswerable is an
    # unactionable surface (principles.md 2.3, 2.7), and a scheduled request for
    # a human decision with no state is not coverage.
    # Alert-publish is best-effort secondary observability — the budget already
    # protected the book, so a publish failure must never block the planner.
    # See [[feedback_no_silent_fails]] (recording surface = the WARN log + the
    # shadow-log turnover fields below).
    _diag = result.diagnostics
    if _diag.get("conviction_gate_applied"):
        logger.info(
            "conviction gate active (run_date=%s): IR_xs=%.4f over %d names "
            "(alpha spread %.5f vs sigma_alpha %.5f) -> discretionary budget "
            "%.4f of a configured %.4f; executed %.4f, mandatory floor %.4f. "
            "No alert: the guard refusing to churn statistically tied names is "
            "the designed behaviour.",
            run_date,
            _diag.get("conviction_ir_xs") or float("nan"),
            _diag.get("conviction_n_names") or 0,
            _diag.get("conviction_alpha_dispersion") or float("nan"),
            _diag.get("conviction_alpha_noise") or float("nan"),
            _diag.get("turnover_budget_discretionary") or float("nan"),
            _diag.get("turnover_budget_configured") or float("nan"),
            _diag.get("turnover_one_way") or float("nan"),
            _diag.get("turnover_mandatory_floor") or float("nan"),
        )
    if _diag.get("large_move_flagged"):
        _req = _diag.get("requested_turnover_one_way", 0.0)
        _cap = optimizer_cfg.get("max_daily_turnover")
        _flag = optimizer_cfg.get("large_move_turnover_flag")
        _reason = _diag.get("large_move_reason")
        _shadow = _diag.get("turnover_constraint_shadow_price")
        _floor = _diag.get("turnover_mandatory_floor")
        _ir = _diag.get("conviction_ir_xs")
        if _reason == "turnover_budget_binding":
            _forced = (
                f" Of that, {_floor:.1%} was MANDATORY (forced exits / "
                f"ineligibility / cash pin); the rest was discretionary."
                if _floor is not None
                else ""
            )
            _conv = (
                f" Signal quality passed the conviction gate "
                f"(IR_xs={_ir:.2f} over {_diag.get('conviction_n_names', 0)} "
                f"names), so the model stands behind the move it wanted."
                if _ir is not None
                else f" Conviction gate not evaluable "
                     f"({_diag.get('conviction_gate_reason')})."
            )
            _detail = (
                f"the {(_cap or 0):.0%}/day turnover budget BOUND the solve "
                f"(executed {_req:.1%}, shadow price "
                f"{'n/a' if _shadow is None else format(_shadow, '.4f')}) — the "
                f"optimizer wanted to move further and the budget stopped it."
                f"{_forced}{_conv} Every name it did trade is sized where it "
                f"wants it; the remainder is deferred to subsequent daily "
                f"re-solves, which is the designed behaviour of a gradual "
                f"rebalance. No action is required unless this repeats across "
                f"sessions — the rolling turnover tripwire owns that case."
            )
        else:
            _detail = (
                f"executed one-way turnover {_req:.1%} exceeds the "
                f"{(_flag or 0):.0%} large-move flag."
            )
        logger.warning(
            "Optimizer large-move flag (%s): %s (run_date=%s)",
            _reason,
            _detail,
            run_date,
        )
        try:
            from executor.notifier import publish_ops_alert

            publish_ops_alert(
                message=(
                    f"[executor] Optimizer large rebalance ({_reason}): "
                    f"{_detail} (run_date={run_date})"
                ),
                severity="WARN",
                source="alpha-engine/executor/optimizer_shadow.py",
                dedup_key=f"optimizer_large_move_{run_date}",
            )
        except Exception as _alert_err:  # noqa: BLE001 — secondary observability
            logger.warning(
                "large-move alert publish failed (non-fatal, cap already applied): %s",
                _alert_err,
            )

    would_be_trades, band_dropped_trades = _compute_trade_deltas(
        tickers,
        result.weights,
        w_prev,
        portfolio_nav,
        optimizer_cfg,
    )
    _dropped_entries = [t for t in band_dropped_trades if t["is_new_position"]]
    if _dropped_entries:
        logger.warning(
            "rebalance band removed %d intended NEW position(s) (band=%.4f): "
            "%s — an entry sized under the band is not drift; check the "
            "turnover budget and min_position_pct (run_date=%s)",
            len(_dropped_entries),
            float(optimizer_cfg.get("rebalance_band_pct", 0.005)),
            "; ".join(f"{t['ticker']}({t['delta']:+.5f})" for t in _dropped_entries),
            run_date,
        )

    # B.4 ablation: when the α̂-uncertainty penalty is configured ON, solve
    # a second time with γ=0 so the shadow log carries both perspectives
    # side-by-side. Operators read the ablation diff to decide whether the
    # uncertainty signal is shaping sizing in a sane way before B.5 cutover.
    # No-op when γ=0 (default) or when no per-ticker σ_α̂ is available —
    # the active solve already IS the no-penalty solve in that regime.
    ablation = _maybe_run_ablation(
        tickers=tickers,
        alpha_hat=alpha_hat,
        alpha_uncertainty=alpha_uncertainty,
        alpha_uncertainty_epistemic=alpha_uncertainty_epistemic,
        returns_panel=returns_panel,
        w_prev=w_prev,
        sectors=sectors,
        stance_caps=stance_caps,
        eligibility=eligibility,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
        optimizer_cfg=optimizer_cfg,
        active_weights=result.weights,
    )

    out: dict = {
        "run_date": run_date,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "shadow_status": "ok",
        "portfolio_nav": float(portfolio_nav),
        "n_tickers": N,
        "tickers": tickers,
        "target_weights": [float(x) for x in result.weights],
        "current_weights": [float(x) for x in w_prev],
        "alpha_hat": [float(x) for x in alpha_hat],
        # Which anchor every solved alpha declared, and how many names were
        # checked (alpha-engine-config-I7337). Emitted every run, healthy
        # included — `{"n_checked": 0}` says the contract measured nothing,
        # which is a finding, not a pass.
        "alpha_anchor": alpha_anchor,
        "alpha_uncertainty": _alpha_uncertainty_to_json(alpha_uncertainty),
        # I9452 — persisted so the daemon's intraday re-solve builds the SAME
        # Omega the morning solve did. Without it the re-solve would silently
        # run the penalty inoperative on a day the morning run had it armed,
        # and the two would size the same book differently.
        "alpha_uncertainty_epistemic": _alpha_uncertainty_to_json(
            alpha_uncertainty_epistemic
        ),
        "eligibility": [bool(x) for x in eligibility],
        "eligibility_reasons": list(eligibility_reasons),
        # Names deleted BEFORE the solve, with a typed reason each. Distinct
        # from `eligibility_reasons`, which only explains an included name's
        # zero weight — the two answer different questions and neither
        # substitutes for the other (config-I7337). Emitted every run,
        # including empty: a field that appears only on the bad path is
        # indistinguishable from a dead emitter.
        "dropped_candidates": dropped_candidates,
        "stance_caps": [float(x) for x in stance_caps],
        "sectors": sectors,
        "covariance_daily": [[float(x) for x in row] for row in sigma_daily],
        "adv_usd": [None if not np.isfinite(x) else float(x) for x in adv_usd],
        "adv_coverage": adv_coverage,
        "would_be_trades": would_be_trades,
        # Trades the anti-churn rebalance band removed, with enough to tell an
        # intended entry apart from genuine drift (`is_new_position`). Emitted
        # every run including empty (alpha-engine-config-I7346).
        "band_dropped_trades": band_dropped_trades,
        "diagnostics": result.diagnostics,
        "legacy_orders": [_redact_order(o) for o in legacy_orders],
        "optimizer_cfg": optimizer_cfg,
    }
    if ablation is not None:
        out["uncertainty_ablation"] = ablation
    return out


def _alpha_uncertainty_to_json(arr: np.ndarray) -> list:
    """Convert per-ticker σ_α̂ array to a JSON-safe list. NaN entries are
    emitted as None so consumers can distinguish 'unknown' from 0 (which
    means 'sentinel — no uncertainty' for SPY/CASH)."""
    out: list = []
    for v in arr:
        if not np.isfinite(v):
            out.append(None)
        else:
            out.append(float(v))
    return out


def _maybe_run_ablation(
    *,
    tickers: list[str],
    alpha_hat: np.ndarray,
    alpha_uncertainty: np.ndarray,
    alpha_uncertainty_epistemic: np.ndarray | None,
    returns_panel: np.ndarray,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
    optimizer_cfg: dict,
    active_weights: np.ndarray,
) -> dict | None:
    """Run a second solve with γ=0 for side-by-side comparison.

    Skipped (returns None) when γ=0 already (no difference would result)
    or when alpha_uncertainty has no usable signal (all-NaN). In both
    cases the canonical solve IS the no-penalty solve.
    """
    gamma = float(optimizer_cfg.get("alpha_uncertainty_penalty", 0.0))
    if gamma <= 0.0:
        return None
    # The ablation contrasts the CANONICAL solve against gamma=0, so it must
    # test the vector the canonical solve's Omega is built from — the
    # epistemic one. Testing the total here would run a second solve on days
    # the penalty was inoperative and report a difference of exactly zero as
    # if it were a measurement.
    if alpha_uncertainty_epistemic is None:
        return None
    has_any_signal = bool(
        np.any(np.isfinite(alpha_uncertainty_epistemic) & (alpha_uncertainty_epistemic > 0.0))
    )
    if not has_any_signal:
        return None

    no_penalty_cfg = {**optimizer_cfg, "alpha_uncertainty_penalty": 0.0}
    try:
        no_penalty_result = solve_target_weights(
            tickers=tickers,
            alpha_hat=alpha_hat,
            returns_panel=returns_panel,
            w_prev=w_prev,
            sectors=sectors,
            stance_caps=stance_caps,
            eligibility=eligibility,
            spy_idx=spy_idx,
            cash_idx=cash_idx,
            cfg=no_penalty_cfg,
            alpha_uncertainty=alpha_uncertainty,
            alpha_uncertainty_epistemic=alpha_uncertainty_epistemic,  # γ=0 → unused
        )
    except Exception as exc:
        logger.warning(
            "Uncertainty-ablation solve failed (non-blocking, shadow continues): %s",
            exc,
        )
        return None

    no_penalty_weights = np.asarray(no_penalty_result.weights, dtype=np.float64)
    active = np.asarray(active_weights, dtype=np.float64)
    deltas = active - no_penalty_weights
    # Per-ticker rows make the diff readable in the shadow JSON. Only
    # include names that actually moved (≥1bp) to keep the log compact.
    per_ticker = []
    for i, t in enumerate(tickers):
        if abs(deltas[i]) >= 1e-4:
            per_ticker.append(
                {
                    "ticker": t,
                    "with_penalty": float(active[i]),
                    "no_penalty": float(no_penalty_weights[i]),
                    "delta": float(deltas[i]),
                    "sigma_alpha": (
                        float(alpha_uncertainty[i])
                        if np.isfinite(alpha_uncertainty[i]) else None
                    ),
                    "sigma_alpha_epistemic": (
                        float(alpha_uncertainty_epistemic[i])
                        if np.isfinite(alpha_uncertainty_epistemic[i]) else None
                    ),
                }
            )
    return {
        "gamma": gamma,
        "no_penalty_weights": [float(x) for x in no_penalty_weights],
        "no_penalty_diagnostics": no_penalty_result.diagnostics,
        "l1_delta": float(np.sum(np.abs(deltas))),
        "max_abs_delta": float(np.max(np.abs(deltas))),
        "n_names_moved": len(per_ticker),
        "per_ticker_delta": per_ticker,
    }


def _classify_drop(
    ticker: str,
    price_histories: dict[str, pd.DataFrame],
    requested: set[str] | None,
) -> str:
    """Why a declared candidate did not make the solved universe.

    Three outcomes that ``_has_usable_history`` collapses into one ``False``,
    and the collapse is the whole defect (config-I7337):

    * ``no_history_loaded``    — the loader was never asked for it. A plumbing
      contradiction: this module DECLARED it a candidate and something upstream
      decided it would never have the data. Raised on, never recorded and
      swallowed.
    * ``history_absent_in_cache`` — asked for, the cache had nothing. A data
      condition (new listing, gap in the predictor cache). Legitimate; recorded.
    * ``history_too_short``    — present but under the covariance floor.
      Legitimate; recorded.

    ``requested is None`` means the caller could not say what was asked for, so
    the first case is unprovable — degrade to ``history_absent_in_cache`` rather
    than raise on a distinction we cannot make.
    """
    if requested is not None and ticker not in requested:
        return "no_history_loaded"
    df = price_histories.get(ticker)
    if df is None:
        return "history_absent_in_cache"
    if "close" not in df.columns:
        return "history_missing_close_column"
    return "history_too_short"


def _build_universe(
    signals_raw: dict,
    predictions_by_ticker: dict,
    current_positions: dict,
    price_histories: dict[str, pd.DataFrame],
    price_histories_requested: set[str] | None = None,
    dropped_out: list[dict] | None = None,
) -> list[str]:
    candidates: set[str] = set()
    candidates.update(predictions_by_ticker.keys())
    candidates.update(current_positions.keys())
    # signals.json::universe read: this is the executor sizing/exit path —
    # the ONE fleet-level exception to "resolve ticker lists from
    # decision_set, not universe" (alpha-engine-config#5809). Formal
    # policy-clause registration tracked separately, not yet landed:
    # alpha-engine-config#6448.
    candidates.update(_extract_universe_tickers(signals_raw.get("universe", [])))

    candidates.discard(_SPY)
    candidates.discard(_CASH)
    eligible = sorted(t for t in candidates if _has_usable_history(t, price_histories))

    # ── A dropped candidate is NAMED, and a plumbing bug RAISES (config-I7337) ─
    #
    # `eligibility_reasons` further down explains why an INCLUDED name got
    # weight 0. It is structurally blind to names deleted here, one step
    # earlier — so until this block, the optimizer's own artifact could not
    # distinguish "the predictor proposed nothing" from "the predictor's whole
    # cut was silently removed before the solve". Those two rendered
    # identically, which is the fleet's could-not-measure-as-found-nothing
    # class and the reason a 20-name cut produced a 2-name book unnoticed.
    #
    # Predicted names are the ones that raise. A held position or a
    # signals-universe name lacking history is ordinary; a name THIS MODULE
    # declared a candidate on line ~483 and that the loader was never asked for
    # is a contract violated inside one process, and it must not be tradeable
    # through. Fail loud and fast: the fleet default is RAISE.
    dropped = [
        {
            "ticker": t,
            "reason": _classify_drop(t, price_histories, price_histories_requested),
            "source": (
                "prediction"
                if t in predictions_by_ticker
                else "position"
                if t in current_positions
                else "signals_universe"
            ),
        }
        for t in sorted(candidates - set(eligible))
    ]
    if dropped_out is not None:
        # Grouped by (source, reason), each group carrying its FULL member list.
        # Measured on the 2026-08-14 inputs, the flat per-ticker form added ~67 KB
        # to a 10 KB artifact — 881 of the 891 records were byte-identical apart
        # from the ticker, because `candidates` also absorbs the whole 903-name
        # `signals.json::universe` sizing envelope (config#5809). Grouping keeps
        # every member name — a count without its members is the defect
        # config-I7324 exists for — and drops only the repetition of the two
        # constant fields.
        groups: dict[tuple[str, str], list[str]] = {}
        for d in dropped:
            groups.setdefault((d["source"], d["reason"]), []).append(d["ticker"])
        dropped_out.extend(
            {"source": src, "reason": rsn, "count": len(names), "tickers": names}
            for (src, rsn), names in sorted(groups.items())
        )

    plumbing_bugs = [d for d in dropped if d["reason"] == "no_history_loaded" and d["source"] == "prediction"]
    if plumbing_bugs:
        names = ", ".join(d["ticker"] for d in plumbing_bugs)
        raise RuntimeError(
            f"Shadow optimizer: {len(plumbing_bugs)} predicted ticker(s) were "
            f"declared candidates but their price histories were never "
            f"requested, so they were dropped from the solved universe with no "
            f"record: {names}. This is a data-plumbing contradiction inside one "
            f"run — `_build_universe` asked for the predictor's cut and "
            f"`executor.main` did not load it (config-I7337). Refusing to solve "
            f"over a universe that silently excludes the signal it exists to "
            f"trade. A ticker legitimately missing from the price cache is "
            f"reported as `history_absent_in_cache` and does NOT raise."
        )

    if dropped:
        logger.warning(
            "[optimizer_shadow] %d candidate(s) dropped before the solve: %s",
            len(dropped),
            "; ".join(f"{d['ticker']}({d['source']}/{d['reason']})" for d in dropped),
        )

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
    """Assemble the solve's alpha vector, on ONE declared market-relative
    anchor (alpha-engine-config-I7337).

    The anchor assertion runs FIRST, before a single value is read: the
    optimizer compares every entry against a SPY=0.0 sentinel, so a batch
    mixing a market-relative alpha with a raw one carrying the meta-L2's
    common-mode macro level cannot be solved — it flushes the whole book to
    zero weights and renders as a normal `optimizer_target_zero` decision.
    See `executor/alpha_contract.py` for the measured incident.
    """
    assert_optimizer_anchor(
        tickers,
        predictions_by_ticker,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
    )
    alpha = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if i == spy_idx:
            alpha[i] = 0.0
            continue
        if i == cash_idx:
            alpha[i] = _CASH_ALPHA_HINT
            continue
        pred = predictions_by_ticker.get(t) or {}
        # `_numeric_alpha` returns None only for a genuinely absent/None/
        # non-numeric opinion. An exact 0.0 is a real opinion and survives —
        # the previous `a or b or 0.0` chain treated 0.0 as missing and fell
        # through to the next field, which is the same falsy-`or` class as
        # the present-and-zero `pullback_pct` defect fixed in PR477.
        val = _numeric_alpha(pred)
        alpha[i] = 0.0 if val is None else val
    return alpha


def _build_alpha_uncertainty(
    tickers: list[str],
    predictions_by_ticker: dict[str, dict],
    spy_idx: int,
    cash_idx: int,
    field: str = "predicted_alpha_std",
) -> np.ndarray:
    """Read a per-ticker uncertainty field off the predictions artifact.

    ``field`` selects the vintage, and the two are NOT interchangeable
    (alpha-engine-config-I9446 / I9452):

    ``predicted_alpha_std``           the TOTAL predictive std,
                                      sqrt(1/α̂ + xᵀΣ_w x). Cross-sectionally
                                      flat by construction (the 1/α̂ term is one
                                      number for the whole batch). This is the
                                      CONVICTION GATE's input — its IR band was
                                      derived against this scale (PR518).
    ``predicted_alpha_std_epistemic`` sqrt(xᵀΣ_w x), the estimation-error std.
                                      The ONLY vector the Garlappi-Uppal-Wang Ω
                                      may be built from. Absent (None) on every
                                      artifact written before 2026-08-31 and on
                                      any champion with no learned noise
                                      precision.

    Returns a length-N array with:
      • 0.0 for SPY and CASH (sentinels — no uncertainty)
      • finite σ ≥ 0 when the predictor emitted a usable value
      • NaN when the field was missing, None, or non-numeric

    ``solve_target_weights`` treats NaN as zero-penalty for that name, and
    declares the whole term inoperative — with a reason on the artifact — when
    the epistemic vector is absent or cross-sectionally flat. It never
    substitutes one vintage for the other.
    """
    sigma = np.full(len(tickers), np.nan)
    for i, t in enumerate(tickers):
        if i == spy_idx or i == cash_idx:
            sigma[i] = 0.0
            continue
        pred = predictions_by_ticker.get(t, {})
        raw_std = pred.get(field)  # may be missing/None
        if raw_std is None:
            continue
        try:
            v = float(raw_std)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 0.0:
            sigma[i] = v
        # else leave NaN — partial-rollout case, handled downstream
    return sigma


def _build_adv_usd(
    tickers: list[str],
    signals_bucket: str | None,
    run_date: str,
    spy_idx: int,
    cash_idx: int,
) -> tuple[np.ndarray, dict]:
    """Build the per-name ADV$ vector (aligned to ``tickers``) from the scanner
    tradeability artifact + a small coverage-observability dict.

    Returns ``(adv_usd, coverage)`` where ``adv_usd[i]`` is name i's average
    daily dollar volume, or ``np.nan`` when uncovered (SPY/CASH are always NaN —
    benchmark fill / sleeve carry no market impact). ``coverage`` records how
    many real names had ADV so an operator can see whether the participation-
    aware term actually engaged.

    FAIL-SOFT end to end: no bucket, an absent artifact, or a read error →
    all-NaN vector → the optimizer degrades to the flat L1 turnover penalty.
    Never raises (the read helper already swallows S3/parse errors).
    """
    from executor.signal_reader import (
        extract_adv_usd,
        read_universe_tradeability,
    )

    adv = np.full(len(tickers), np.nan)
    n_real = sum(1 for i in range(len(tickers)) if i not in (spy_idx, cash_idx))
    if not signals_bucket:
        return adv, {"adv_names_covered": 0, "adv_names_total": n_real, "adv_source": "none_no_bucket"}
    # The reader already fails soft (returns {} on any S3/credential/parse
    # error), but wrap defensively so NO tradeability-read failure mode can
    # ever propagate up into run_shadow_optimizer and null the shadow log.
    # A None/{} map ↔ "no ADV coverage" ↔ the optimizer's flat-L1 fallback.
    try:
        tradeability = read_universe_tradeability(signals_bucket, run_date)
    except Exception as exc:  # noqa: BLE001 — construction refinement, never a gate
        logger.warning(
            "Universe tradeability read raised (%s) — ADV absent, flat-L1 tcost fallback.",
            exc,
        )
        return adv, {"adv_names_covered": 0, "adv_names_total": n_real, "adv_source": "none_read_error"}
    adv_by_ticker = extract_adv_usd(tradeability)
    covered = 0
    for i, t in enumerate(tickers):
        if i in (spy_idx, cash_idx):
            continue
        v = adv_by_ticker.get(t)
        if v is not None:
            adv[i] = v
            covered += 1
    return adv, {
        "adv_names_covered": covered,
        "adv_names_total": n_real,
        "adv_source": "scanner_universe_tradeability" if covered else "none_uncovered",
    }


def _build_name_sigma(
    returns_panel: np.ndarray,
    spy_idx: int,
    cash_idx: int,
) -> np.ndarray:
    """Per-name daily return σ from the returns panel, for the Almgren-Chriss
    σ-scaling of the impact term. SPY/CASH → NaN (excluded from the impact
    term). A column with no finite std → NaN → that name is σ-agnostic."""
    N = returns_panel.shape[1]
    sig = np.full(N, np.nan)
    for i in range(N):
        if i in (spy_idx, cash_idx):
            continue
        col = returns_panel[:, i]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            s = float(np.std(col, ddof=1))
            if np.isfinite(s) and s > 0:
                sig[i] = s
    return sig


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
        "value": float(config.get("stance_size_value", 0.7)),
        "quality": float(config.get("stance_size_quality", 0.8)),
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
) -> tuple[np.ndarray, list[str | None]]:
    """Compute per-ticker eligibility for new entries + the gate that
    excluded each ineligible ticker.

    Returns (eligibility_mask, reasons). ``reasons[i]`` is None for
    eligible tickers, otherwise a stable slug naming the gate that
    excluded the ticker — consumed downstream by
    ``order_book_rationale.build_order_book_rationale`` to answer
    "why didn't ticker X enter?" when the portfolio optimizer is the
    authoritative driver (the legacy ``_plan_entries`` path is bypassed
    and produces no ``blocked_entries`` / ``risk_events`` itself).

    Reason slugs:
      * ``"signal_exit"`` — research said EXIT.
      * ``"gbm_veto"`` — predictor's high-confidence DOWN veto fired.
      * ``"score_below_min"`` — research composite < ``min_score_to_enter``.
      * ``"no_score"`` — no research signal / score for this ticker.

    Held tickers stay eligible regardless of score (the optimizer
    decides whether to reduce them).
    """
    min_score = float(config.get("min_score_to_enter", 57))
    eligibility = np.ones(len(tickers), dtype=bool)
    reasons: list[str | None] = [None] * len(tickers)
    for i, t in enumerate(tickers):
        if i in (spy_idx, cash_idx):
            continue
        sig = signals_by_ticker.get(t, {})
        pred = predictions_by_ticker.get(t, {})
        is_held = t in current_positions

        if sig.get("signal") == "EXIT":
            eligibility[i] = False
            reasons[i] = "signal_exit"
            continue
        if pred.get("gbm_veto") is True:
            eligibility[i] = False
            reasons[i] = "gbm_veto"
            continue
        if is_held:
            continue
        score = sig.get("score")
        if score is None:
            eligibility[i] = False
            reasons[i] = "no_score"
        elif float(score) < min_score:
            eligibility[i] = False
            reasons[i] = "score_below_min"
    return eligibility, reasons


def _compute_trade_deltas(
    tickers: list[str],
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    portfolio_nav: float,
    optimizer_cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """Return ``(would_be_trades, band_dropped_trades)``.

    The anti-churn rebalance band suppresses per-name DRIFT — a position that
    has wandered a few basis points from its target is not worth a commission.
    It cannot, on its own, tell drift apart from an intended NEW position that
    something upstream shrank into the band, and until this returned its
    second list, nothing named what it removed: the solve reported ``optimal``,
    ``entries_blocked`` stayed empty, and a deleted entry cohort rendered
    identically to a quiet hold day (alpha-engine-config-I7346).

    ``band_dropped_trades`` is written to the shadow artifact on EVERY run,
    including empty — the same could-not-measure-vs-found-nothing rule the
    ``dropped_candidates`` field was given in config-I7337. A field that only
    appears on the bad path is indistinguishable from a dead emitter.

    ``is_new_position`` is the field that makes the record actionable: a
    dropped trade whose ``current_weight`` is zero is an ENTRY the band
    deleted, which is categorically different from trimming drift on a name
    already held.
    """
    band = float(optimizer_cfg.get("rebalance_band_pct", 0.005))
    trades: list[dict] = []
    band_dropped: list[dict] = []
    for i, t in enumerate(tickers):
        if t == _CASH:
            continue
        delta_pct = float(target_weights[i] - current_weights[i])
        target_w = float(target_weights[i])
        current_w = float(current_weights[i])
        if abs(delta_pct) < band:
            # Only an intended trade can be "removed by the band". A name the
            # optimizer left exactly where it was — most of the universe, at
            # weight 0 on both sides, plus any name already at its target —
            # asked for no trade at all, and recording those would bury the
            # real drops. _DUST_DELTA is a sub-dollar move on any realistic
            # book, i.e. solver noise rather than an intention.
            if abs(delta_pct) < _DUST_DELTA:
                continue
            band_dropped.append(
                {
                    "ticker": t,
                    "target_weight": round(target_w, 6),
                    "current_weight": round(current_w, 6),
                    "delta": round(delta_pct, 6),
                    "band": band,
                    "is_new_position": bool(current_w == 0.0 and target_w > 0.0),
                }
            )
            continue
        delta_dollars = delta_pct * float(portfolio_nav)
        trades.append(
            {
                "ticker": t,
                "action": "BUY" if delta_pct > 0 else "SELL",
                "delta_weight": round(delta_pct, 6),
                "delta_dollars": round(delta_dollars, 2),
                "target_weight": round(target_w, 6),
                "current_weight": round(current_w, 6),
            }
        )
    return trades, band_dropped


def _redact_order(order: dict) -> dict:
    keep = {
        "ticker",
        "action",
        "shares",
        "limit_price",
        "dollar_size",
        "position_pct",
        "stance",
        "score",
        "signal_type",
    }
    return {k: order.get(k) for k in keep if k in order}


def _write_shadow_log_to_s3(
    log: dict,
    bucket: str,
    run_date: str,
    s3_client=None,
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
