"""Executor-side decision capture — emits ``DecisionArtifact`` rows for
algorithmic (non-LLM) decisions in the intraday daemon and morning planner.

ROADMAP L2308 — fills the "Executor Components" insufficient-data gap in
the Saturday evaluator email. Per-component artifacts at
``s3://alpha-engine-research/decision_artifacts/{Y}/{M}/{D}/executor:{component}/{run_id}.json``.

Schema is the lib's ``DecisionArtifact`` with ``model_metadata=None`` and
``full_prompt_context=None`` (deterministic decision — see
``alpha_engine_lib.decision_capture`` schema_version=2 lib v0.10.0+).
Producer provenance lives in ``input_data_snapshot._producer`` +
``_producer_version`` rather than the LLM ``model_metadata.model_name``
field — the plan doc question 2 anchor.

**This module covers PR 1 of the L2308 arc: entry_triggers.** Sibling
modules / helpers for position_sizer, risk_guard, exit_rules ship in
PRs 2-4.

**Feature flag:** the capture path is gated on
``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=true`` env var (same convention
as research's sector-team capture). Default off so the trading EC2 only
opts in when its IAM grants + budget have been validated. Operator
enables by setting the env var on the trading instance.

**Hard-fail discipline:** on S3 write failure, ``DecisionCaptureWriteError``
propagates per ``feedback_no_silent_fails``. Daemon caller must decide
whether the trade-execution path catches it (best-effort semantics) or
lets it bubble up (correctness semantics). PR 1 wires this with a
try/except at the daemon call site so a transient S3 outage doesn't
kill a trade execution — capture is observability, not load-bearing for
the trade itself.

Plan doc: ``~/Development/alpha-engine-docs/private/executor-decision-capture-260511.md``.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from alpha_engine_lib.decision_capture import (
    DecisionCaptureWriteError,
    capture_decision,
)

logger = logging.getLogger(__name__)


# ── Feature flag ──────────────────────────────────────────────────────────


_ENV_VAR = "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"


def is_decision_capture_enabled() -> bool:
    """Read the env var fresh on each call (allows toggling in tests).

    Returns True iff ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` is set to
    ``"true"`` / ``"1"`` / ``"yes"`` (case-insensitive). Default off.
    """
    return os.environ.get(_ENV_VAR, "").lower() in ("true", "1", "yes")


# ── Producer identity ─────────────────────────────────────────────────────


_AGENT_ID_ENTRY_TRIGGERS = "executor:entry_triggers"
_PRODUCER_NAME_ENTRY_TRIGGERS = "alpha-engine.executor.entry_triggers"

_AGENT_ID_POSITION_SIZER = "executor:position_sizer"
_PRODUCER_NAME_POSITION_SIZER = "alpha-engine.executor.position_sizer"

_AGENT_ID_RISK_GUARD = "executor:risk_guard"
_PRODUCER_NAME_RISK_GUARD = "alpha-engine.executor.risk_guard"

_AGENT_ID_EXIT_RULES = "executor:exit_rules"
_PRODUCER_NAME_EXIT_RULES = "alpha-engine.executor.exit_rules"

# Bump when the snapshot/output shape changes; readers can filter on this
# rather than guessing from S3 timestamps. Per-component versioning so a
# bump in one producer (e.g. entry_triggers) doesn't force a re-tag of
# every other deterministic-decision producer.
_PRODUCER_VERSION = "1.0.0"


_TRIGGER_KINDS = (
    "pullback",
    "vwap_discount",
    "support_bounce",
    "graduated_entry",
    "time_expiry",
)


def _classify_trigger_kind(trigger_reason: str) -> str:
    """Map the daemon's free-text trigger reason to a canonical kind.

    Mirrors the if-chain in ``entry_triggers.py::should_enter``. Returns
    ``"unknown"`` if no kind matches — surfaces a producer-side gap rather
    than silently labeling.

    Trigger reasons from ``EntryTriggerEngine.should_enter`` follow these
    prefixes:
    - ``"pullback {pct} from high $..."`` → ``"pullback"``
    - ``"VWAP discount ..."``               → ``"vwap_discount"``
    - ``"near support ..."``                → ``"support_bounce"``
    - ``"graduated_entry (... vs morning $...)"`` → ``"graduated_entry"``
    - ``"time_expiry"``                     → ``"time_expiry"``
    """
    lower = trigger_reason.lower()
    if "graduated_entry" in lower:
        # Check before pullback since graduated_entry strings can contain
        # the substring "from" or "high" via float formatting in edge cases.
        return "graduated_entry"
    if "time_expiry" in lower:
        return "time_expiry"
    if "pullback" in lower:
        return "pullback"
    if "vwap" in lower:
        return "vwap_discount"
    if "support" in lower:
        return "support_bounce"
    return "unknown"


# ── Snapshot builder ──────────────────────────────────────────────────────


def build_entry_trigger_payload(
    *,
    entry: dict,
    price_state: dict,
    trigger_reason: str,
    strategy_config: dict,
    disabled_triggers: list[str],
    now_et_iso: str,
    fill_price: float | None = None,
    actual_shares: int | None = None,
    trade_id: str | int | None = None,
) -> tuple[dict, dict, str]:
    """Build ``(input_data_snapshot, agent_output, input_data_summary)``
    for one entry-trigger fire event.

    Snapshot captures everything the engine saw at decision time —
    config thresholds + entry context + price state — so a future
    replay or grading run has reproducibility-grade inputs without
    needing to re-query S3 for the morning order-book entry.

    ``agent_output`` records the chosen trigger + fill outcome (joins
    cleanly against ``trades.csv`` via ``trade_id`` for downstream
    grading analytics in PR 5).
    """
    triggers_cfg = entry.get("triggers", {})
    snapshot = {
        "_producer": _PRODUCER_NAME_ENTRY_TRIGGERS,
        "_producer_version": _PRODUCER_VERSION,
        "ticker": entry.get("ticker"),
        "signal": entry.get("signal"),
        "shares": entry.get("shares"),
        "signal_date": entry.get("signal_date"),
        "prediction_date": entry.get("prediction_date"),
        "morning_price": entry.get("current_price"),
        "current_price": price_state.get("last"),
        "day_high": price_state.get("high"),
        "day_low": price_state.get("low"),
        "vwap": triggers_cfg.get("vwap"),
        "support_level": triggers_cfg.get("support_level"),
        "thresholds": {
            "pullback_pct": (
                triggers_cfg.get("pullback_pct")
                or strategy_config.get("intraday_pullback_pct")
            ),
            "vwap_discount": (
                triggers_cfg.get("vwap_discount")
                or strategy_config.get("intraday_vwap_discount_pct")
            ),
            "support_pct": (
                triggers_cfg.get("support_pct")
                or strategy_config.get("intraday_support_pct")
            ),
            "graduated_max_premium": strategy_config.get(
                "intraday_graduated_max_premium_pct",
            ),
            "expiry_time": strategy_config.get("intraday_expiry_time"),
            "graduated_start_time": strategy_config.get(
                "intraday_graduated_start_time",
            ),
        },
        "disabled_triggers": list(disabled_triggers),
        "now_et": now_et_iso,
    }

    agent_output = {
        "fired_trigger": trigger_reason,
        "trigger_kind": _classify_trigger_kind(trigger_reason),
        "captured_at_fill_attempt": True,
        "fill_price": fill_price,
        "actual_shares": actual_shares,
        "trade_id": trade_id,
    }

    summary = (
        f"{entry.get('ticker') or '<unknown>'} ENTER "
        f"shares={entry.get('shares')} fired={trigger_reason}"
    )
    return snapshot, agent_output, summary


# ── Capture call (deterministic; model_metadata=None) ─────────────────────


def capture_entry_trigger(
    *,
    run_date: str,
    entry: dict,
    price_state: dict,
    trigger_reason: str,
    strategy_config: dict,
    disabled_triggers: list[str],
    now_et_iso: str,
    fill_price: float | None = None,
    actual_shares: int | None = None,
    trade_id: str | int | None = None,
    s3_client: Any | None = None,
    s3_bucket: str = "alpha-engine-research",
) -> str | None:
    """Emit one ``executor:entry_triggers`` ``DecisionArtifact`` to S3.

    Returns the S3 key on successful capture, ``None`` if the env-flag
    feature gate is off. Raises ``DecisionCaptureWriteError`` on any S3
    write failure — caller decides whether to swallow (best-effort
    observability) or propagate (strict capture).

    ``run_id`` is constructed as ``{run_date}_{ticker}_{uuid8}`` so
    multiple captures per ticker per day don't overwrite each other at
    the S3 leaf (per the plan doc question 1 anchor: per-trading-day
    run_id with per-decision suffix at the leaf).

    Per the lib v0.10.0 schema, ``model_metadata`` and
    ``full_prompt_context`` are both ``None`` — this is a deterministic
    decision, not an LLM call. The lib's ``_llm_fields_paired`` validator
    enforces both-or-neither semantics.
    """
    if not is_decision_capture_enabled():
        return None

    snapshot, agent_output, summary = build_entry_trigger_payload(
        entry=entry,
        price_state=price_state,
        trigger_reason=trigger_reason,
        strategy_config=strategy_config,
        disabled_triggers=disabled_triggers,
        now_et_iso=now_et_iso,
        fill_price=fill_price,
        actual_shares=actual_shares,
        trade_id=trade_id,
    )

    ticker = entry.get("ticker") or "unknown"
    run_id = f"{run_date}_{ticker}_{uuid.uuid4().hex[:8]}"

    s3_key = capture_decision(
        run_id=run_id,
        agent_id=_AGENT_ID_ENTRY_TRIGGERS,
        model_metadata=None,
        full_prompt_context=None,
        input_data_snapshot=snapshot,
        agent_output=agent_output,
        input_data_summary=summary,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
    )
    return s3_key


# ── Position sizer payload + capture ─────────────────────────────────────


def build_position_sizer_payload(
    *,
    ticker: str,
    signal: dict,
    sector_rating: str,
    current_price: float,
    portfolio_nav: float,
    n_enter_signals: int,
    drawdown_multiplier: float,
    atr_pct: float | None,
    prediction_confidence: float | None,
    p_up: float | None,
    signal_age_days: int | None,
    days_to_earnings: int | None,
    feature_coverage: float | None,
    stance: str | None,
    sizing_result: dict,
    sized_outcome: str,
    sized_outcome_reason: str | None,
) -> tuple[dict, dict, str]:
    """Build ``(input_data_snapshot, agent_output, input_data_summary)``
    for one ``compute_position_size`` invocation.

    Snapshot mirrors every argument passed into the sizer plus the
    portfolio context (NAV, n_entries today) so a replay can re-derive
    the sizing decision from inputs without hitting S3 again.

    ``agent_output`` records the sized result (shares + dollars + pct NAV)
    plus the per-multiplier breakdown (sector_adj, conviction_adj,
    upside_adj, drawdown_multiplier, atr_adj, confidence_adj,
    staleness_adj, earnings_adj, coverage_adj, stance_adj). Joining
    against ``trades.csv`` by ``ticker + date`` lets the backtester
    grading analytics (PR 5) measure sizing-skill — e.g. did high-
    multiplier sizes systematically outperform low-multiplier ones over
    5d.

    ``sized_outcome`` is one of ``{"approved", "shares_zero"}`` — captured
    at the call site so downstream consumers know whether the sized
    quantity became an order or got filtered. (Risk-guard vetoes and
    GBM vetoes that happen AFTER sizing are captured by PR 3
    ``executor:risk_guard``, not here.)
    """
    snapshot = {
        "_producer": _PRODUCER_NAME_POSITION_SIZER,
        "_producer_version": _PRODUCER_VERSION,
        # Inputs passed into compute_position_size
        "ticker": ticker,
        "signal": {
            "score": signal.get("score"),
            "conviction": signal.get("conviction"),
            "rating": signal.get("rating"),
            "price_target_upside": signal.get("price_target_upside"),
            "sector": signal.get("sector"),
        },
        "sector_rating": sector_rating,
        "current_price": current_price,
        "portfolio_nav": portfolio_nav,
        "n_enter_signals": n_enter_signals,
        "drawdown_multiplier": drawdown_multiplier,
        "atr_pct": atr_pct,
        "prediction_confidence": prediction_confidence,
        "p_up": p_up,
        "signal_age_days": signal_age_days,
        "days_to_earnings": days_to_earnings,
        "feature_coverage": feature_coverage,
        "stance": stance,
    }

    # The sizer return dict carries the full multiplier breakdown — copy
    # it through so grading analytics can decompose any subsequent
    # under/over-performance against any single multiplier.
    agent_output = {
        "shares": sizing_result.get("shares"),
        "dollar_size": sizing_result.get("dollar_size"),
        "position_pct": sizing_result.get("position_pct"),
        "sector_adj": sizing_result.get("sector_adj"),
        "conviction_adj": sizing_result.get("conviction_adj"),
        "upside_adj": sizing_result.get("upside_adj"),
        "dd_multiplier": sizing_result.get("dd_multiplier"),
        "atr_adj": sizing_result.get("atr_adj"),
        "confidence_adj": sizing_result.get("confidence_adj"),
        "staleness_adj": sizing_result.get("staleness_adj"),
        "earnings_adj": sizing_result.get("earnings_adj"),
        "coverage_adj": sizing_result.get("coverage_adj"),
        "stance_adj": sizing_result.get("stance_adj"),
        "sized_outcome": sized_outcome,
        "sized_outcome_reason": sized_outcome_reason,
    }

    summary = (
        f"{ticker} sizing: shares={sizing_result.get('shares')} "
        f"dollars=${sizing_result.get('dollar_size'):.0f} "
        f"pct={sizing_result.get('position_pct'):.4f} "
        f"outcome={sized_outcome}"
        if sizing_result.get("dollar_size") is not None
        and sizing_result.get("position_pct") is not None
        else f"{ticker} sizing: outcome={sized_outcome}"
    )
    return snapshot, agent_output, summary


def capture_position_sizer(
    *,
    run_date: str,
    ticker: str,
    signal: dict,
    sector_rating: str,
    current_price: float,
    portfolio_nav: float,
    n_enter_signals: int,
    drawdown_multiplier: float,
    atr_pct: float | None,
    prediction_confidence: float | None,
    p_up: float | None,
    signal_age_days: int | None,
    days_to_earnings: int | None,
    feature_coverage: float | None,
    stance: str | None,
    sizing_result: dict,
    sized_outcome: str,
    sized_outcome_reason: str | None = None,
    s3_client: Any | None = None,
    s3_bucket: str = "alpha-engine-research",
) -> str | None:
    """Emit one ``executor:position_sizer`` artifact for a single
    ``compute_position_size`` invocation.

    No-op when the env-flag feature gate is off. Raises
    ``DecisionCaptureWriteError`` on S3 failure per
    ``feedback_no_silent_fails`` — caller is responsible for the
    best-effort try/except.

    ``run_id`` is ``{run_date}_{ticker}_{uuid8}`` (per the plan doc Q1
    anchor — multiple sizing calls per ticker per day don't overwrite
    at the S3 leaf; in practice the morning planner sizes each ticker
    at most once, but the uniqueness invariant is preserved for
    correctness rather than relying on assumed call frequency).
    """
    if not is_decision_capture_enabled():
        return None

    snapshot, agent_output, summary = build_position_sizer_payload(
        ticker=ticker,
        signal=signal,
        sector_rating=sector_rating,
        current_price=current_price,
        portfolio_nav=portfolio_nav,
        n_enter_signals=n_enter_signals,
        drawdown_multiplier=drawdown_multiplier,
        atr_pct=atr_pct,
        prediction_confidence=prediction_confidence,
        p_up=p_up,
        signal_age_days=signal_age_days,
        days_to_earnings=days_to_earnings,
        feature_coverage=feature_coverage,
        stance=stance,
        sizing_result=sizing_result,
        sized_outcome=sized_outcome,
        sized_outcome_reason=sized_outcome_reason,
    )

    run_id = f"{run_date}_{ticker}_{uuid.uuid4().hex[:8]}"

    s3_key = capture_decision(
        run_id=run_id,
        agent_id=_AGENT_ID_POSITION_SIZER,
        model_metadata=None,
        full_prompt_context=None,
        input_data_snapshot=snapshot,
        agent_output=agent_output,
        input_data_summary=summary,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
    )
    return s3_key


# ── Risk guard payload + capture ─────────────────────────────────────────


def _compute_existing_sector_exposure(
    current_positions: dict, sector: str | None,
) -> float:
    """Sum market_value across current positions in the same sector.

    Mirrors the risk_guard internal computation so the capture snapshot
    carries the same view risk_guard saw at gate-evaluation time —
    important for grading the max_sector veto in particular (the
    threshold check uses `(existing + new) / nav`).
    """
    if not sector:
        return 0.0
    return float(sum(
        pos.get("market_value", 0.0) or 0.0
        for pos in (current_positions or {}).values()
        if pos.get("sector") == sector
    ))


def build_risk_guard_payload(
    *,
    ticker: str,
    action: str,
    dollar_size: float,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict,
    sector: str | None,
    market_regime: str,
    signal: dict,
    config: dict,
    approved: bool,
    reason: str,
    events: list[dict] | None,
) -> tuple[dict, dict, str]:
    """Build ``(input_data_snapshot, agent_output, input_data_summary)``
    for one ``risk_guard.check_order`` invocation.

    Captures both the **vetoed** path (artifact mirrors the risk_events
    table row with full input context) AND the **counterfactual** —
    every approved ticker also gets one artifact recording the
    "non-vetoed" path inputs so backtester grading (PR 5) can measure
    precision-of-refusal: of the entries risk_guard approved, how many
    became drawdowns? Of the ones it vetoed, how many would have been
    winners? Without the counterfactual, only one direction is gradable.

    Snapshot mirrors every variable risk_guard reads at decision time:
    portfolio state (nav, peak_nav, drawdown fraction), per-ticker
    proposal (dollar_size, sector, sector_rating), market regime, and
    every gate threshold from config that's evaluated for ENTER.

    ``agent_output.outcome`` is ``"approved"`` or ``"vetoed"``;
    ``vetoed_rule`` carries the rule name (``"min_score"``, ``"max_position"``,
    ``"bear_underweight"``, ``"max_sector"``, ``"max_equity"``,
    ``"correlation_block"``, etc.) for grading-side filtering.
    ``events`` is the raw list emitted by risk_guard for the per-rule
    audit trail (one row per veto event — usually 0 or 1, never more
    than 1 per ticker because the function short-circuits on first fail).
    """
    drawdown_frac = (
        (peak_nav - portfolio_nav) / peak_nav
        if peak_nav and peak_nav > 0 else 0.0
    )
    sector_rating = signal.get("sector_rating", "market_weight")
    existing_sector_exposure = _compute_existing_sector_exposure(
        current_positions, sector,
    )
    snapshot = {
        "_producer": _PRODUCER_NAME_RISK_GUARD,
        "_producer_version": _PRODUCER_VERSION,
        # Per-ticker proposal
        "ticker": ticker,
        "action": action,
        "dollar_size": dollar_size,
        "sector": sector,
        "sector_rating": sector_rating,
        # Portfolio state at gate-evaluation time
        "portfolio_nav": portfolio_nav,
        "peak_nav": peak_nav,
        "drawdown_fraction": drawdown_frac,
        "n_open_positions": len(current_positions or {}),
        "existing_sector_exposure": existing_sector_exposure,
        # Signal context (the values gates read against thresholds)
        "signal": {
            "score": signal.get("score"),
            "conviction": signal.get("conviction"),
            "rating": signal.get("rating"),
            "price_target_upside": signal.get("price_target_upside"),
        },
        "market_regime": market_regime,
        # Config-derived gate thresholds — every one risk_guard evaluates
        # on the ENTER path. Captured so grading can replay against a
        # different threshold set without re-running the risk gate.
        "thresholds": {
            "min_score_to_enter": config.get("min_score_to_enter", 70),
            "max_position_pct": config.get("max_position_pct", 0.05),
            "bear_max_position_pct": config.get("bear_max_position_pct", 0.025),
            "max_sector_pct": config.get("max_sector_pct", 0.25),
            "max_equity_pct": config.get("max_equity_pct", 1.00),
            "bear_block_underweight": config.get("bear_block_underweight", True),
            "drawdown_halt_pct": config.get("drawdown_halt_pct"),
            "correlation_block_threshold": config.get(
                "correlation_block_threshold",
            ),
        },
    }

    outcome = "approved" if approved else "vetoed"
    # First fired veto rule (the one that short-circuited). risk_guard
    # only emits one veto event per ticker due to short-circuit; on the
    # approved path the events list may still carry portfolio-level
    # rows from the caller, so filter to per-ticker veto events.
    veto_events = [
        ev for ev in (events or [])
        if ev.get("ticker") == ticker and ev.get("event_type") == "veto"
    ]
    vetoed_rule = veto_events[0].get("rule") if veto_events else None
    agent_output = {
        "outcome": outcome,
        "reason": reason,
        "vetoed_rule": vetoed_rule,
        "events": veto_events,
    }

    summary = f"{ticker} risk_guard: outcome={outcome} reason={reason}"
    return snapshot, agent_output, summary


def capture_risk_guard(
    *,
    run_date: str,
    ticker: str,
    action: str,
    dollar_size: float,
    portfolio_nav: float,
    peak_nav: float,
    current_positions: dict,
    sector: str | None,
    market_regime: str,
    signal: dict,
    config: dict,
    approved: bool,
    reason: str,
    events: list[dict] | None,
    s3_client: Any | None = None,
    s3_bucket: str = "alpha-engine-research",
) -> str | None:
    """Emit one ``executor:risk_guard`` artifact for a single
    ``check_order`` invocation. Captures both vetoed and approved paths
    (the counterfactual coverage that grading needs for precision-of-
    refusal analytics).

    No-op when the env-flag feature gate is off. Raises
    ``DecisionCaptureWriteError`` on S3 failure per
    ``feedback_no_silent_fails``.
    """
    if not is_decision_capture_enabled():
        return None

    snapshot, agent_output, summary = build_risk_guard_payload(
        ticker=ticker,
        action=action,
        dollar_size=dollar_size,
        portfolio_nav=portfolio_nav,
        peak_nav=peak_nav,
        current_positions=current_positions,
        sector=sector,
        market_regime=market_regime,
        signal=signal,
        config=config,
        approved=approved,
        reason=reason,
        events=events,
    )

    run_id = f"{run_date}_{ticker}_{uuid.uuid4().hex[:8]}"

    s3_key = capture_decision(
        run_id=run_id,
        agent_id=_AGENT_ID_RISK_GUARD,
        model_metadata=None,
        full_prompt_context=None,
        input_data_snapshot=snapshot,
        agent_output=agent_output,
        input_data_summary=summary,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
    )
    return s3_key


# ── Exit rules payload + capture (daemon-side intraday) ──────────────────


# Map IntradayExitManager exit-signal reason strings to canonical kinds.
# Mirrors the reason values in executor/intraday_exit_manager.py.
_INTRADAY_EXIT_REASON_TO_KIND = {
    "intraday_trailing_stop": "atr_trail",
    "intraday_profit_take": "profit_take",
    "intraday_collapse": "collapse",
}

# Planner-side fired_rule_key values from
# ``executor/strategies/exit_manager.py::_evaluate_single_position``.
# Canonical kinds for the morning-planner exit-rule decision layer.
# Where overlapping with daemon-side kinds (atr_trail, profit_take), the
# planner key maps to the same canonical kind so grading analytics can
# treat both layers consistently for those rules. Planner-only kinds
# (catalyst_hard_exit, fallback_stop, momentum_exit, time_decay,
# sector_veto_blocked) extend the vocabulary.
_PLANNER_EXIT_RULE_KEY_TO_KIND = {
    "catalyst_hard_exit": "catalyst_hard_exit",
    "atr_trailing_stop": "atr_trail",
    "sector_veto_blocked": "sector_veto_blocked",
    "fallback_stop": "fallback_stop",
    "profit_take": "profit_take",
    "momentum_exit": "momentum_exit",
    "time_decay": "time_decay",
}


def _classify_exit_rule_kind(exit_reason: str) -> str:
    """Map an exit signal's ``reason`` field to a canonical rule kind
    for daemon-side intraday captures (PR 4).

    Returns ``"unknown"`` for unmapped reasons — surfaces a producer-side
    gap rather than silently labeling.
    """
    if not exit_reason:
        return "unknown"
    return _INTRADAY_EXIT_REASON_TO_KIND.get(exit_reason, "unknown")


def _classify_planner_exit_kind(fired_rule_key: str | None) -> str:
    """Map planner-side ``fired_rule_key`` to a canonical rule kind for
    planner-side captures (PR 4b).

    ``None`` → ``"no_fire"`` (no rule fired — counterfactual coverage
    row that grading uses to measure missed-exit precision).
    Unmapped key → ``"unknown"`` (defensive; surfaces if a new rule
    branch in ``_evaluate_single_position`` ships without updating this
    map).
    """
    if fired_rule_key is None:
        return "no_fire"
    return _PLANNER_EXIT_RULE_KEY_TO_KIND.get(fired_rule_key, "unknown")


def build_exit_rule_payload(
    *,
    stop: dict,
    price_state: dict,
    exit_signal: dict,
    strategy_config: dict,
    fill_price: float | None = None,
    actual_shares_exited: int | None = None,
    trade_id: str | int | None = None,
) -> tuple[dict, dict, str]:
    """Build ``(input_data_snapshot, agent_output, input_data_summary)``
    for one ``IntradayExitManager.evaluate`` fire event (daemon-side
    intraday exit rule).

    Snapshot mirrors what the rule engine saw at decision time:
    - Position state from the stop record (entry_price, current_stop,
      trail_atr, atr_multiple, high_water, shares, entry_date, ticker)
    - Live price state (last, high, low)
    - Config thresholds (intraday_profit_take_pct, intraday_collapse_pct,
      intraday_tighten_after_days, intraday_tighten_atr_multiple)
    - Computed gain/loss vs entry, days_held — load-bearing for grading
      "did this exit fire too early / too late?"

    agent_output records the fired rule (atr_trail / profit_take /
    collapse) + the executed fill outcome, joinable against trades.csv
    by ticker + date.
    """
    entry_price = stop.get("entry_price")
    current_price = price_state.get("last")
    gain_pct: float | None = None
    if entry_price and entry_price > 0 and current_price is not None:
        gain_pct = (current_price - entry_price) / entry_price

    entry_date_str = stop.get("entry_date")
    days_held: int | None = None
    if entry_date_str:
        try:
            from datetime import date as _date
            days_held = (
                _date.today() - _date.fromisoformat(entry_date_str)
            ).days
        except (ValueError, TypeError):
            days_held = None

    snapshot = {
        "_producer": _PRODUCER_NAME_EXIT_RULES,
        "_producer_version": _PRODUCER_VERSION,
        # Position state
        "ticker": stop.get("ticker"),
        "entry_price": entry_price,
        "entry_date": entry_date_str,
        "current_stop": stop.get("current_stop"),
        "trail_atr": stop.get("trail_atr"),
        "atr_multiple": stop.get("atr_multiple"),
        "high_water": stop.get("high_water"),
        "shares_held": stop.get("shares"),
        "profit_take_executed": stop.get("profit_take_executed", False),
        # Market state
        "current_price": current_price,
        "day_high": price_state.get("high"),
        "day_low": price_state.get("low"),
        # Derived signals (load-bearing for grading)
        "gain_pct": gain_pct,
        "days_held": days_held,
        # Config thresholds evaluated by the rules
        "thresholds": {
            "intraday_profit_take_pct": strategy_config.get(
                "intraday_profit_take_pct", 0.08,
            ),
            "intraday_collapse_pct": strategy_config.get(
                "intraday_collapse_pct", 0.05,
            ),
            "intraday_tighten_after_days": strategy_config.get(
                "intraday_tighten_after_days", 3,
            ),
            "intraday_tighten_atr_multiple": strategy_config.get(
                "intraday_tighten_atr_multiple", 1.5,
            ),
        },
        # Capture layer (daemon-side intraday vs planner-side
        # evaluate_exits — PR 4b will add "planner" as a layer)
        "evaluation_layer": "daemon_intraday",
    }

    fired_reason = exit_signal.get("reason")
    agent_output = {
        "outcome": "fired",
        "action": exit_signal.get("action"),  # "EXIT" | "REDUCE"
        "fired_rule": fired_reason,
        "fired_rule_kind": _classify_exit_rule_kind(fired_reason),
        "shares_requested": exit_signal.get("shares"),
        "detail": exit_signal.get("detail"),
        "fill_price": fill_price,
        "actual_shares_exited": actual_shares_exited,
        "trade_id": trade_id,
    }

    summary = (
        f"{stop.get('ticker') or '<unknown>'} "
        f"{exit_signal.get('action', 'EXIT')} "
        f"fired={fired_reason} gain={gain_pct:.1%}"
        if gain_pct is not None
        else f"{stop.get('ticker') or '<unknown>'} "
             f"{exit_signal.get('action', 'EXIT')} fired={fired_reason}"
    )
    return snapshot, agent_output, summary


def capture_exit_rule(
    *,
    run_date: str,
    stop: dict,
    price_state: dict,
    exit_signal: dict,
    strategy_config: dict,
    fill_price: float | None = None,
    actual_shares_exited: int | None = None,
    trade_id: str | int | None = None,
    s3_client: Any | None = None,
    s3_bucket: str = "alpha-engine-research",
) -> str | None:
    """Emit one ``executor:exit_rules`` artifact for a single
    ``IntradayExitManager.evaluate`` fire event.

    No-op when the env-flag feature gate is off. Raises
    ``DecisionCaptureWriteError`` on S3 failure per
    ``feedback_no_silent_fails``.

    Planner-side ``evaluate_exits`` captures will share this same
    helper via a follow-up PR (4b) that extracts a per-position helper
    from the existing ``evaluate_exits`` loop so the input snapshot can
    include the planner's wider context (signal_at_exit, stance,
    catalyst_date, sector etc.).
    """
    if not is_decision_capture_enabled():
        return None

    snapshot, agent_output, summary = build_exit_rule_payload(
        stop=stop,
        price_state=price_state,
        exit_signal=exit_signal,
        strategy_config=strategy_config,
        fill_price=fill_price,
        actual_shares_exited=actual_shares_exited,
        trade_id=trade_id,
    )

    ticker = stop.get("ticker") or "unknown"
    run_id = f"{run_date}_{ticker}_{uuid.uuid4().hex[:8]}"

    s3_key = capture_decision(
        run_id=run_id,
        agent_id=_AGENT_ID_EXIT_RULES,
        model_metadata=None,
        full_prompt_context=None,
        input_data_snapshot=snapshot,
        agent_output=agent_output,
        input_data_summary=summary,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
    )
    return s3_key


# ── Planner-side exit rules payload + capture (PR 4b) ────────────────────


def build_planner_exit_payload(
    *,
    ticker: str,
    pos: dict,
    research_signal: dict,
    current_price: float,
    stance: str | None,
    catalyst_date,
    stance_config: dict,
    signal: dict | None,
    fired_rule_key: str | None,
) -> tuple[dict, dict, str]:
    """Build ``(input_data_snapshot, agent_output, input_data_summary)``
    for one ``evaluate_exits._evaluate_single_position`` invocation
    (planner-side morning exit-rule layer).

    Captures BOTH fired (signal non-None) and no-fire (signal is None)
    positions. The no-fire counterfactual lets grading measure missed-
    exit precision: positions where no exit rule fired in the planner
    but later produced drawdowns are gradable as missed exits.

    Snapshot mirrors what ``_evaluate_single_position`` saw at decision
    time:
    - Position state from ``pos`` (entry_date, avg_cost, sector, shares,
      stance, catalyst_date)
    - Research signal context (action, score, conviction)
    - Resolved stance_config thresholds for each rule the planner
      evaluated (atr_*, profit_take_pct, time_decay_*, momentum_*,
      catalyst_*, sector_relative_*)
    - Market state (current_price)
    """
    avg_cost = pos.get("avg_cost")
    gain_pct: float | None = None
    if avg_cost and avg_cost > 0:
        gain_pct = (current_price - avg_cost) / avg_cost

    snapshot = {
        "_producer": _PRODUCER_NAME_EXIT_RULES,
        "_producer_version": _PRODUCER_VERSION,
        "ticker": ticker,
        # Position state
        "entry_date": pos.get("entry_date"),
        "avg_cost": avg_cost,
        "shares_held": pos.get("shares"),
        "market_value": pos.get("market_value"),
        "sector": pos.get("sector"),
        "stance": stance,
        "catalyst_date": catalyst_date,
        # Market state
        "current_price": current_price,
        "gain_pct": gain_pct,
        # Research-side signal context (the planner reads this to decide
        # whether the position is still in HOLD vs already exiting)
        "research_action": research_signal.get("signal", "HOLD"),
        "research_score": research_signal.get("score"),
        "research_conviction": research_signal.get("conviction"),
        # Resolved (stance-aware) config thresholds for each rule
        # branch ``_evaluate_single_position`` evaluated. Captured so a
        # replay can re-evaluate with different thresholds without
        # re-running the planner.
        "thresholds": {
            "atr_period": stance_config.get("atr_period"),
            "atr_multiple": stance_config.get("atr_multiple"),
            "fallback_stop_enabled": stance_config.get(
                "fallback_stop_enabled", True,
            ),
            "fallback_stop_pct": stance_config.get("fallback_stop_pct"),
            "profit_take_pct": stance_config.get("profit_take_pct"),
            "time_decay_enabled": stance_config.get("time_decay_enabled"),
            "time_decay_days": stance_config.get("time_decay_days"),
            "momentum_exit_enabled": stance_config.get(
                "momentum_exit_enabled",
            ),
            "momentum_exit_threshold": stance_config.get(
                "momentum_exit_threshold",
            ),
            "catalyst_follow_through_days": stance_config.get(
                "catalyst_follow_through_days",
            ),
        },
        "evaluation_layer": "planner",
    }

    fired_rule_kind = _classify_planner_exit_kind(fired_rule_key)
    if signal is not None:
        agent_output = {
            "outcome": "fired",
            "action": signal.get("action"),  # "EXIT" | "REDUCE"
            "fired_rule": signal.get("reason"),
            "fired_rule_key": fired_rule_key,
            "fired_rule_kind": fired_rule_kind,
            "shares_requested": signal.get("shares"),
            "detail": signal.get("detail"),
        }
        summary = (
            f"{ticker} {signal.get('action', 'EXIT')} "
            f"fired={fired_rule_key} kind={fired_rule_kind}"
        )
    else:
        # No-fire counterfactual row. fired_rule_key may be None (truly
        # nothing fired) or "sector_veto_blocked" (ATR fired but was
        # suppressed by sector veto — load-bearing for grading the veto
        # decision separately from the no-fire decision).
        agent_output = {
            "outcome": "no_fire",
            "action": None,
            "fired_rule": None,
            "fired_rule_key": fired_rule_key,
            "fired_rule_kind": fired_rule_kind,
            "shares_requested": None,
            "detail": None,
        }
        summary = f"{ticker} no_fire kind={fired_rule_kind}"

    return snapshot, agent_output, summary


def capture_planner_exit(
    *,
    run_date: str,
    ticker: str,
    pos: dict,
    research_signal: dict,
    current_price: float,
    stance: str | None,
    catalyst_date,
    stance_config: dict,
    signal: dict | None,
    fired_rule_key: str | None,
    s3_client: Any | None = None,
    s3_bucket: str = "alpha-engine-research",
) -> str | None:
    """Emit one ``executor:exit_rules`` artifact for a single planner-
    side ``_evaluate_single_position`` invocation.

    Same agent_id as PR 4's daemon-side captures (``executor:exit_rules``)
    so grading analytics can read them from one S3 prefix. The
    ``evaluation_layer`` field on the snapshot distinguishes
    ``"planner"`` (PR 4b) from ``"daemon_intraday"`` (PR 4).

    No-op when the env-flag feature gate is off. Raises
    ``DecisionCaptureWriteError`` on S3 failure per
    ``feedback_no_silent_fails``.
    """
    if not is_decision_capture_enabled():
        return None

    snapshot, agent_output, summary = build_planner_exit_payload(
        ticker=ticker,
        pos=pos,
        research_signal=research_signal,
        current_price=current_price,
        stance=stance,
        catalyst_date=catalyst_date,
        stance_config=stance_config,
        signal=signal,
        fired_rule_key=fired_rule_key,
    )

    run_id = f"{run_date}_{ticker}_{uuid.uuid4().hex[:8]}"

    s3_key = capture_decision(
        run_id=run_id,
        agent_id=_AGENT_ID_EXIT_RULES,
        model_metadata=None,
        full_prompt_context=None,
        input_data_snapshot=snapshot,
        agent_output=agent_output,
        input_data_summary=summary,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
    )
    return s3_key


__all__ = [
    "DecisionCaptureWriteError",
    "build_entry_trigger_payload",
    "build_exit_rule_payload",
    "build_planner_exit_payload",
    "build_position_sizer_payload",
    "build_risk_guard_payload",
    "capture_entry_trigger",
    "capture_exit_rule",
    "capture_planner_exit",
    "capture_position_sizer",
    "capture_risk_guard",
    "is_decision_capture_enabled",
]
