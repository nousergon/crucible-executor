"""
order_book_rationale.py — Per-ticker daily order-book decision record.

The order-book state for a ticker on a given day (approved entry /
urgent exit / reduce / held / risk-blocked / predictor-vetoed) is the
*output* of a multi-stage decision chain: research signal read →
predictor veto → risk-guard rules → position sizer → intraday entry
trigger. The individual inputs are each observable (``signals.json``,
``predictions/{date}.json``, the ``risk_events`` SQLite table, the
order book JSON) but no single per-ticker artifact answers
*"why is ticker X in state S today?"* for the **whole considered
universe** — including the tickers that did *not* make it in (the
"why didn't Y enter" question is as operationally important as the
"why did X enter" one).

This module builds that record. It is a pure join + serialize over
structures already materialized by the morning planner — it adds **no
new instrumentation**. It reuses the ``risk_events`` rule slugs
(alpha-engine #139), the signal/prediction lineage columns
(alpha-engine #138), and the ``entries_with_meta`` sizing breakdown
the order book already carries.

The artifact is written in the canonical ``alpha_engine_lib.eval_artifacts``
shape (``{prefix}/{run_id}.json`` dated forensic artifact + a
``{prefix}/latest.json`` operator-UX sidecar) so the dashboard reads it
with the same ``load_latest_eval_artifact`` / ``list_eval_artifacts``
helpers every other eval-style artifact uses, and same-day re-runs are
preserved for audit.

The record is reproducible from the artifact alone — it carries the
research sub-scores, predictor outputs, the risk slug + threshold that
excluded a ticker, the full sizing-factor breakdown, and the intraday
trigger config. Nothing in it requires re-opening S3 or querying
SQLite to interpret.

Producer of the consumer-facing transparency surface tracked in
ROADMAP "Per-ticker daily order-book rationale record + console tab".
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# Default S3 prefix. Lives under ``trades/`` alongside the order book
# itself (``trades/order_book/{date}.json``) so the rationale and the
# book it explains are co-located in the audit namespace.
DEFAULT_S3_PREFIX = "trades/order_book_rationale"

# Terminal-state vocabulary. Ordered most-actioned → least so listings
# and the console table surface entries/exits before held/no-action.
STATE_APPROVED_ENTRY = "approved_entry"
STATE_URGENT_EXIT = "urgent_exit"
STATE_REDUCE = "reduce"
STATE_PREDICTOR_VETOED = "predictor_vetoed"
STATE_RISK_BLOCKED = "risk_blocked"
STATE_HELD = "held"
STATE_NO_ACTION = "no_action"

_STATE_ORDER = {
    STATE_APPROVED_ENTRY: 0,
    STATE_URGENT_EXIT: 1,
    STATE_REDUCE: 2,
    STATE_PREDICTOR_VETOED: 3,
    STATE_RISK_BLOCKED: 4,
    STATE_HELD: 5,
    STATE_NO_ACTION: 6,
}

# risk_events rules emitted when the *predictor* (not research/risk)
# drove the rejection — these map to STATE_PREDICTOR_VETOED so the
# console can answer "blocked by the ML layer" distinctly from
# "blocked by a hard risk rule". Sourced from deciders.py emit sites.
_PREDICTOR_RULES = {"stance_gate", "momentum_gate"}


def _research_block(sig: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the research signal fields the record exposes."""
    if not sig:
        return {}
    return {
        "signal": sig.get("signal"),
        "score": sig.get("score"),
        "conviction": sig.get("conviction"),
        "rating": sig.get("rating"),
        "sector": sig.get("sector"),
        "sector_rating": sig.get("sector_rating"),
        "price_target_upside": sig.get("price_target_upside"),
        "thesis_summary": sig.get("thesis_summary"),
    }


def _predictor_block(pred: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the predictor fields the record exposes."""
    if not pred:
        return {}
    return {
        "predicted_direction": pred.get("predicted_direction"),
        "prediction_confidence": pred.get("prediction_confidence"),
        "predicted_alpha": pred.get("predicted_alpha"),
        "stance": pred.get("stance"),
        "catalyst_date": pred.get("catalyst_date"),
    }


def build_order_book_rationale(
    *,
    signals: Mapping[str, Any],
    predictions_by_ticker: Mapping[str, Any],
    order_book_data: Mapping[str, Any],
    blocked_entries: Sequence[Mapping[str, Any]],
    risk_events: Sequence[Mapping[str, Any]],
    market_regime: str,
    run_date: str,
    signal_date: str | None,
    prediction_date: str | None,
    calendar_date: str,
    trading_day: str,
    run_id: str,
) -> dict[str, Any]:
    """Join the morning-planner decision structures into a per-ticker record.

    Pure function — no I/O, no clock reads (``run_id`` /
    ``calendar_date`` / ``trading_day`` are injected so the caller owns
    the dating discipline and tests are deterministic).

    Args:
        signals: output of ``signal_reader.get_actionable_signals`` —
            ``{"enter": [...], "exit": [...], "reduce": [...],
            "hold": [...]}`` lists of per-ticker research signal dicts.
            This is the *considered universe*.
        predictions_by_ticker: ``{ticker: prediction_dict}`` from the
            predictor inference artifact.
        order_book_data: the finalized ``OrderBook._data`` dict
            (``approved_entries`` carry ``entries_with_meta`` payloads
            incl. ``sizing_factors`` + ``triggers``; ``urgent_exits``
            carry exit/reduce/cover records).
        blocked_entries: ``plan.blocked`` — per-ENTER rejections with a
            free-text ``block_reason``.
        risk_events: ``plan.risk_events`` — structured rule log with
            ``event_type`` / ``rule`` / ``value`` / ``threshold``.
        market_regime: effective regime for the run.
        run_date / signal_date / prediction_date: lineage dates.
        calendar_date / trading_day: dual-tracked dates (LdP).
        run_id: ``YYMMDDHHMM`` from ``new_eval_run_id``.

    Returns:
        Serializable payload dict (canonical eval_artifacts shape).
    """
    enter = {s.get("ticker"): s for s in signals.get("enter", []) if s.get("ticker")}
    held = {s.get("ticker"): s for s in signals.get("hold", []) if s.get("ticker")}
    exited = {s.get("ticker"): s for s in signals.get("exit", []) if s.get("ticker")}
    reduced = {s.get("ticker"): s for s in signals.get("reduce", []) if s.get("ticker")}

    approved = {
        e.get("ticker"): e
        for e in order_book_data.get("approved_entries", [])
        if e.get("ticker")
    }
    # Urgent-exit records keyed by ticker (last write wins — dedup is
    # the order book's job; we only read the final state).
    urgent: dict[str, dict] = {}
    for rec in order_book_data.get("urgent_exits", []):
        t = rec.get("ticker")
        if t:
            urgent[t] = rec

    blocked = {
        b.get("ticker"): b for b in blocked_entries if b.get("ticker")
    }
    # First risk_event per ticker is the one that excluded it (decider
    # emits at the first failing gate then `continue`s).
    first_event: dict[str, dict] = {}
    for ev in risk_events:
        t = ev.get("ticker")
        if t and t not in first_event:
            first_event[t] = ev

    considered = (
        set(enter) | set(held) | set(exited) | set(reduced)
        | set(approved) | set(urgent) | set(blocked) | set(first_event)
    )

    records: list[dict[str, Any]] = []
    for ticker in considered:
        sig = (
            enter.get(ticker) or held.get(ticker)
            or exited.get(ticker) or reduced.get(ticker)
        )
        pred = predictions_by_ticker.get(ticker)
        ev = first_event.get(ticker)
        ob_entry = approved.get(ticker)
        ob_exit = urgent.get(ticker)

        chain: list[dict[str, Any]] = []
        exclusion: dict[str, Any] | None = None

        # Stage 1 — research signal read.
        chain.append({
            "stage": "signal_read",
            "result": (sig or {}).get("signal") or "none",
            "detail": (
                f"composite {sig.get('score')}"
                if sig and sig.get("score") is not None else None
            ),
        })

        # Stage 2 — predictor (direction/confidence; veto attribution
        # comes from the risk_event slug at stage 3).
        if pred:
            chain.append({
                "stage": "predictor",
                "result": pred.get("predicted_direction") or "none",
                "detail": (
                    f"conf {pred.get('prediction_confidence')}"
                    if pred.get("prediction_confidence") is not None else None
                ),
            })

        # Terminal-state resolution + stage 3+ chain.
        if ob_exit is not None:
            sigtype = (ob_exit.get("signal") or "").upper()
            state = STATE_REDUCE if sigtype == "REDUCE" else STATE_URGENT_EXIT
            chain.append({
                "stage": "exit_path",
                "result": sigtype or "EXIT",
                "detail": ob_exit.get("reason") or ob_exit.get("detail"),
            })
        elif ob_entry is not None:
            state = STATE_APPROVED_ENTRY
            chain.append({"stage": "risk_guard", "result": "pass"})
            sf = ob_entry.get("sizing_factors") or {}
            chain.append({
                "stage": "position_sizer",
                "result": (
                    f"{ob_entry.get('position_pct', 0) * 100:.2f}% NAV"
                    if ob_entry.get("position_pct") is not None else "sized"
                ),
                "sizing_factors": sf,
                "shares": ob_entry.get("shares"),
                "dollar_size": ob_entry.get("dollar_size"),
            })
            chain.append({
                "stage": "entry_trigger",
                "result": "pending",
                "triggers": ob_entry.get("triggers") or {},
            })
        elif ev is not None or ticker in blocked:
            rule = (ev or {}).get("rule")
            state = (
                STATE_PREDICTOR_VETOED
                if rule in _PREDICTOR_RULES
                or (ev or {}).get("event_type") == "override"
                else STATE_RISK_BLOCKED
            )
            exclusion = {
                "event_type": (ev or {}).get("event_type"),
                "rule": rule,
                "value": (ev or {}).get("value"),
                "threshold": (ev or {}).get("threshold"),
                "reason": (
                    (ev or {}).get("reason")
                    or blocked.get(ticker, {}).get("block_reason")
                ),
            }
            chain.append({
                "stage": "risk_guard",
                "result": "blocked",
                "rule": rule,
                "value": exclusion["value"],
                "threshold": exclusion["threshold"],
                "detail": exclusion["reason"],
            })
        elif ticker in held:
            state = STATE_HELD
        else:
            state = STATE_NO_ACTION

        records.append({
            "ticker": ticker,
            "terminal_state": state,
            "research": _research_block(sig),
            "predictor": _predictor_block(pred),
            "decision_chain": chain,
            "exclusion": exclusion,
            "order_book": ob_entry or ob_exit or None,
        })

    records.sort(key=lambda r: (_STATE_ORDER.get(r["terminal_state"], 9), r["ticker"]))

    summary: dict[str, int] = {"n_considered": len(records)}
    for r in records:
        summary[f"n_{r['terminal_state']}"] = summary.get(
            f"n_{r['terminal_state']}", 0
        ) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_date": run_date,
        "signal_date": signal_date,
        "prediction_date": prediction_date,
        "market_regime": market_regime,
        "summary": summary,
        "tickers": records,
    }


def write_order_book_rationale(
    payload: Mapping[str, Any],
    *,
    s3_client: Any,
    bucket: str,
    prefix: str = DEFAULT_S3_PREFIX,
    write_latest: bool = True,
) -> dict[str, str]:
    """Publish the rationale payload to S3 in canonical eval_artifacts shape.

    Writes ``{prefix}/{run_id}.json`` (forensic dated artifact, always)
    and ``{prefix}/latest.json`` (operator-UX sidecar pointer, when
    ``write_latest``). Mirrors ``regime.substrate.write_regime_substrate``
    — the single source of truth for the sidecar body shape so the
    dashboard's ``load_latest_eval_artifact`` resolves it identically.

    Returns dict with ``artifact_key`` and (when written) ``latest_key``.
    """
    from alpha_engine_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
    )

    run_id = payload["run_id"]
    artifact_key = eval_artifact_key(prefix, run_id)

    s3_client.put_object(
        Bucket=bucket,
        Key=artifact_key,
        Body=json.dumps(payload, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    result = {"artifact_key": artifact_key}

    if write_latest:
        latest_key = eval_latest_key(prefix)
        sidecar = {
            "run_id": run_id,
            "artifact_key": artifact_key,
            "calendar_date": payload["calendar_date"],
            "trading_day": payload["trading_day"],
            "schema_version": payload["schema_version"],
            "market_regime": payload.get("market_regime"),
            "n_considered": payload.get("summary", {}).get("n_considered"),
            "written_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        s3_client.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=json.dumps(sidecar).encode("utf-8"),
            ContentType="application/json",
        )
        result["latest_key"] = latest_key
        logger.info(
            "[order_book_rationale] wrote run_id=%s → s3://%s/%s (latest=%s)",
            run_id, bucket, artifact_key, latest_key,
        )
    else:
        logger.info(
            "[order_book_rationale] wrote run_id=%s → s3://%s/%s (latest skipped)",
            run_id, bucket, artifact_key,
        )

    return result
