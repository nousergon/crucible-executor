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
# Bump when the snapshot/output shape changes; readers can filter on this
# rather than guessing from S3 timestamps.
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


__all__ = [
    "DecisionCaptureWriteError",
    "build_entry_trigger_payload",
    "capture_entry_trigger",
    "is_decision_capture_enabled",
]
