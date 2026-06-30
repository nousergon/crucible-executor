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

The artifact is written in the canonical ``nousergon_lib.eval_artifacts``
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

SCHEMA_VERSION = "1.3.0"

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
# Generic NO_ACTION kept as a compatibility alias — the producer now
# always emits one of the two sub-states below. Consumers reading
# pre-1.2.0 artifacts may still encounter the bare slug.
STATE_NO_ACTION = "no_action"
# Sub-states (schema 1.2.0+) — the ways a ticker can land in the
# no-action bucket post-filter are:
#   1. Research ENTER + optimizer eligible + target ≈ 0 → optimizer_zero
#      (the "we looked and chose not to allocate" case — benign)
#   2. Research ENTER + optimizer assigned a NON-ZERO target but NO order
#      was created (not approved / blocked / held) → optimizer_dropped.
#      The optimizer WANTED this position and the allocation was lost
#      downstream (e.g. a price-resolve failure in optimizer_cutover).
#      This is an ERROR, not a benign no-action — surfaced distinctly so
#      the console flags it loudly. (2026-06-04: AMD, the optimizer's
#      10% top pick, was dropped this way on a transient IBKR price miss.)
#   3. No optimizer view at all (legacy non-optimizer run / signal
#      silently dropped upstream) → unknown.
# Research HOLD / EXIT / REDUCE on a non-held ticker is filtered out
# of the considered universe entirely — those rows are dead signals
# (no possible order-book interaction) and would only add noise.
STATE_NO_ACTION_OPTIMIZER_ZERO = "no_action_optimizer_zero_weight"
STATE_NO_ACTION_OPTIMIZER_DROPPED = "no_action_optimizer_dropped"
STATE_NO_ACTION_UNKNOWN = "no_action_unknown"

_NO_ACTION_STATES = frozenset({
    STATE_NO_ACTION,
    STATE_NO_ACTION_OPTIMIZER_ZERO,
    STATE_NO_ACTION_OPTIMIZER_DROPPED,
    STATE_NO_ACTION_UNKNOWN,
})

# Error-severity terminal states — an operator MUST look at these. The
# console renders them as errors (banner + red row) and they sort to the
# top of the no-action cluster.
_ERROR_STATES = frozenset({
    STATE_NO_ACTION_OPTIMIZER_DROPPED,
    STATE_NO_ACTION_UNKNOWN,
})

_STATE_ORDER = {
    STATE_APPROVED_ENTRY: 0,
    STATE_URGENT_EXIT: 1,
    STATE_REDUCE: 2,
    STATE_PREDICTOR_VETOED: 3,
    STATE_RISK_BLOCKED: 4,
    STATE_HELD: 5,
    # Error states sort FIRST within the no-action cluster so the operator
    # sees the dropped allocation before the benign optimizer-zero rows.
    STATE_NO_ACTION_OPTIMIZER_DROPPED: 6,
    STATE_NO_ACTION_UNKNOWN: 7,
    STATE_NO_ACTION_OPTIMIZER_ZERO: 8,
    # Compatibility — legacy aggregate slug sorts with the others.
    STATE_NO_ACTION: 8,
}

# risk_events rules emitted when the *predictor* (not research/risk)
# drove the rejection — these map to STATE_PREDICTOR_VETOED so the
# console can answer "blocked by the ML layer" distinctly from
# "blocked by a hard risk rule". Sourced from deciders.py emit sites.
_PREDICTOR_RULES = {"stance_gate", "momentum_gate", "gbm_veto"}

# Mapping from optimizer_shadow eligibility-rejection slugs to the
# (rule, event_type, human_message) tuple used to synthesize
# risk_events for the optimizer-authoritative path. Maps to the same
# state vocabulary as the legacy path:
#   - ``gbm_veto`` → STATE_PREDICTOR_VETOED (predictor-driven)
#   - everything else → STATE_RISK_BLOCKED (research/score-driven)
_OPTIMIZER_REJECTION_SLUGS = {
    "gbm_veto": (
        "gbm_veto",
        "override",
        "Predictor high-confidence DOWN veto fired (gbm_veto=true).",
    ),
    "score_below_min": (
        "min_score_to_enter",
        "block",
        "Research composite score below configured min_score_to_enter.",
    ),
    "no_score": (
        "missing_score",
        "block",
        "No research score available for this ticker (universe entry but signals "
        "missing).",
    ),
    # ``signal_exit`` is intentionally not surfaced as a block — research-EXIT
    # tickers flow through urgent_exits in the order book and are handled by
    # the existing exit_path branch in build_order_book_rationale.
}


def _synthesize_optimizer_rejections(
    *,
    shadow_log: Mapping[str, Any] | None,
    signals_by_ticker: Mapping[str, Mapping[str, Any]],
    current_positions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct ``blocked_entries`` + ``risk_events`` from the optimizer log.

    When ``use_portfolio_optimizer: true`` the legacy ``_plan_entries``
    path is bypassed (executor/main.py § "Process ENTER signals"), so
    ``plan.blocked`` and ``plan.risk_events`` are empty — but the
    optimizer's eligibility mask still encodes per-ticker rejection
    reasons. This function projects those reasons into the same dict
    shape the legacy path produces so
    ``build_order_book_rationale`` can answer "why didn't ticker X
    enter?" identically across both code paths.

    Returns (synthetic_blocked_entries, synthetic_risk_events). Both
    lists are stable-ordered by the optimizer's ``tickers`` order.

    Tickers that are currently held are excluded — a "rejection" for an
    already-held ticker is structurally a research-EXIT (which the
    order book carries as ``urgent_exits``), not a missed entry.
    """
    if not shadow_log:
        return [], []
    tickers = shadow_log.get("tickers") or []
    eligibility = shadow_log.get("eligibility") or []
    reasons = shadow_log.get("eligibility_reasons") or []
    opt_cfg = shadow_log.get("optimizer_cfg") or {}
    min_score_threshold = opt_cfg.get("min_score_to_enter")

    blocked: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for i, ticker in enumerate(tickers):
        if i >= len(eligibility) or eligibility[i]:
            continue
        if ticker in current_positions:
            continue
        reason = reasons[i] if i < len(reasons) else None
        mapped = _OPTIMIZER_REJECTION_SLUGS.get(reason or "")
        if mapped is None:
            continue
        rule_slug, event_type, message = mapped

        # Surface threshold + ticker's actual value when meaningful.
        # score_below_min: threshold = min_score; value = the score.
        # gbm_veto / no_score / others: best-effort, may be None.
        sig = signals_by_ticker.get(ticker, {}) or {}
        value: Any | None = None
        threshold: Any | None = None
        if reason == "score_below_min":
            value = sig.get("score")
            threshold = min_score_threshold
        elif reason == "gbm_veto":
            value = True
        elif reason == "no_score":
            value = None
            threshold = min_score_threshold

        blocked.append({"ticker": ticker, "block_reason": message})
        events.append({
            "ticker": ticker,
            "event_type": event_type,
            "rule": rule_slug,
            "value": value,
            "threshold": threshold,
            "reason": message,
        })

    return blocked, events


# Tickers with an optimizer target inside this absolute fraction of NAV
# are treated as "effectively zero target weight" when classifying the
# no-action sub-state. Mirrors what a `target * NAV` rounding-to-zero
# would surface to an operator; the rebalance_band is a stricter
# trade-emission gate downstream, not a "did the optimizer want this"
# threshold.
_OPTIMIZER_TARGET_ZERO_EPSILON = 1e-6


def _classify_no_action(
    *,
    opt_view: Mapping[str, Any] | None,
) -> tuple[str, str | None]:
    """Classify a ticker that fell through to the no-action terminal state.

    Post-filter, the only way to land here is a research ENTER signal
    that didn't make it into approved / blocked / risk-evented. The
    optimizer view (when present) supplies the explanation.

      - Optimizer view present, target ≈ 0 → optimizer_zero_weight
        (the "we looked and chose not to allocate" case — benign).
      - Optimizer view present, target ≥ ε but no order/block/held →
        optimizer_dropped — the optimizer WANTED this position and the
        allocation was lost downstream (price-resolve failure, etc.).
        An ERROR, surfaced distinctly so the console flags it loudly.
      - No optimizer view at all → unknown (legacy non-optimizer run or
        a signal silently dropped upstream).

    Returns the sub-state slug + a short human-readable detail string
    appended to the decision chain so the per-ticker drill-down
    explains the fallthrough.
    """
    if isinstance(opt_view, Mapping):
        tgt = opt_view.get("target_weight")
        if isinstance(tgt, (int, float)):
            if abs(tgt) < _OPTIMIZER_TARGET_ZERO_EPSILON:
                return (
                    STATE_NO_ACTION_OPTIMIZER_ZERO,
                    "optimizer eligible but assigned ~0 target weight",
                )
            # Non-zero target that never became an order — the allocation
            # was dropped downstream. ERROR. See [[feedback_no_silent_fails]].
            return (
                STATE_NO_ACTION_OPTIMIZER_DROPPED,
                f"ERROR: optimizer targeted {tgt * 100:.2f}% but no order was "
                f"created (not approved / blocked / held) — allocation dropped "
                f"downstream; check the AlphaEngine/Executor/"
                f"optimizer_target_dropped alarm + executor log",
            )
    return (
        STATE_NO_ACTION_UNKNOWN,
        "research ENTER signal with no order, block, or optimizer view",
    )


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


def _build_book_status(
    *,
    summary: Mapping[str, int],
    optimizer_shadow_log: Mapping[str, Any] | None,
    rebalance_band_pct: float | None,
    distribution_gate: Mapping[str, Any] | None,
    hold_book_active: bool,
    hold_book_diag: Mapping[str, Any] | None,
    predictions_by_ticker: Mapping[str, Any],
) -> dict[str, Any]:
    """One-line "why did/didn't the book move today" status for the console banner.

    A pure join over structures the planner already materialized (the
    terminal-state ``summary`` counts, the optimizer shadow-log
    diagnostics, the predictor distribution gate, and the
    ``_should_hold_book`` diagnostics). Answers the daily operator
    question — *benign HOLD vs fault* — without cross-referencing the
    planner log, the separate ``hold_book_flags`` artifact, and a
    CloudWatch metric (the three surfaces it used to take).

    ``state`` precedence, most-alarming first:

    * ``allocations_dropped`` — the optimizer targeted a non-zero weight
      but no order was created (price-resolution failure downstream). An
      ERROR; mirrors the ``no_action_optimizer_dropped`` terminal state.
    * ``hold_book_safeguard`` — the predictor gate flagged AND the
      tradable ``predicted_alpha`` collapsed, so the optimizer rebalance
      was suppressed and the current book retained.
    * ``rebalanced`` — entries and/or exits were written.
    * ``no_rebalance_at_target`` — the optimizer solved optimal with
      one-way turnover below the rebalance band; the book is unchanged by
      design (a valid HOLD, not a fault).

    Dispersion of the tradable signal is the authoritative
    ``_should_hold_book`` ``alpha_stdev`` when present (computed only when
    the gate flagged), else the same cross-sectional ``pstdev`` recomputed
    here over ``predicted_alpha``. Direction skew is an unambiguous label
    count, always computed from the batch. ``n_high_confidence`` is
    deliberately NOT recomputed here — it is the predictor's own
    threshold-dependent metric, and re-deriving it with a possibly-
    divergent threshold is the name-semantic-mismatch bug class.
    """
    import math
    import statistics

    n_entries = int(summary.get(f"n_{STATE_APPROVED_ENTRY}", 0))
    n_exits = int(summary.get(f"n_{STATE_URGENT_EXIT}", 0)) + int(
        summary.get(f"n_{STATE_REDUCE}", 0)
    )
    n_dropped = int(summary.get(f"n_{STATE_NO_ACTION_OPTIMIZER_DROPPED}", 0))

    # One-way turnover the optimizer computed for this batch (the field the
    # planner already logs as the HOLD justification).
    turnover_one_way: float | None = None
    if isinstance(optimizer_shadow_log, Mapping):
        _diag = optimizer_shadow_log.get("diagnostics") or {}
        _t = _diag.get("turnover_one_way") if isinstance(_diag, Mapping) else None
        if isinstance(_t, (int, float)) and not isinstance(_t, bool):
            turnover_one_way = float(_t)

    # Predictor dispersion — what made a low-conviction day low-conviction.
    alphas: list[float] = []
    n_up = n_down = n_flat = 0
    for p in (predictions_by_ticker or {}).values():
        if not isinstance(p, Mapping):
            continue
        d = str(p.get("predicted_direction") or "").upper()
        if d == "UP":
            n_up += 1
        elif d == "DOWN":
            n_down += 1
        elif d == "FLAT":
            n_flat += 1
        a = p.get("predicted_alpha")
        if a is None:
            a = p.get("canonical_predicted_alpha")
        if isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a):
            alphas.append(float(a))

    alpha_stdev: float | None = None
    if isinstance(hold_book_diag, Mapping) and isinstance(
        hold_book_diag.get("alpha_stdev"), (int, float)
    ):
        alpha_stdev = float(hold_book_diag["alpha_stdev"])
    elif len(alphas) >= 2:
        alpha_stdev = round(statistics.pstdev(alphas), 6)

    _degenerate = (
        hold_book_diag.get("signal_degenerate")
        if isinstance(hold_book_diag, Mapping)
        else None
    )
    dispersion = {
        "n_predictions": len(predictions_by_ticker or {}),
        "n_up": n_up,
        "n_down": n_down,
        "n_flat": n_flat,
        "alpha_stdev": alpha_stdev,
        "signal_degenerate": (None if _degenerate is None else bool(_degenerate)),
    }

    _gate = distribution_gate if isinstance(distribution_gate, Mapping) else {}
    safeguard = {
        "fired": bool(hold_book_active),
        "reason": _gate.get("reason"),
        "failed_check": _gate.get("failed_check"),
    }

    if n_dropped > 0:
        state = "allocations_dropped"
        headline = (
            f"{n_dropped} allocation(s) targeted by the optimizer but dropped "
            "before order creation — price-resolution failed. Investigate."
        )
    elif hold_book_active:
        state = "hold_book_safeguard"
        _why = safeguard["reason"] or safeguard["failed_check"] or "predictor batch flagged"
        headline = (
            "Hold-book safeguard fired — optimizer rebalance suppressed; current "
            f"book retained with stops. Trigger: {_why}."
        )
    elif n_entries > 0 or n_exits > 0:
        state = "rebalanced"
        _e = "entry" if n_entries == 1 else "entries"
        headline = f"Book rebalanced — {n_entries} {_e} + {n_exits} exit(s) written."
    else:
        state = "no_rebalance_at_target"
        _t = (
            f"{turnover_one_way * 100:.2f}%"
            if turnover_one_way is not None
            else "below threshold"
        )
        headline = (
            "No rebalance — the optimizer solved optimal and the current portfolio "
            f"already matches target (one-way turnover {_t}, below the rebalance "
            "band). Existing positions retained with stops. Valid HOLD, not a fault."
        )

    return {
        "state": state,
        "headline": headline,
        "n_entries": n_entries,
        "n_exits": n_exits,
        "n_dropped": n_dropped,
        "turnover_one_way": turnover_one_way,
        "rebalance_band_pct": rebalance_band_pct,
        "safeguard": safeguard,
        "dispersion": dispersion,
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
    optimizer_shadow_log: Mapping[str, Any] | None = None,
    current_positions: Mapping[str, Any] | None = None,
    distribution_gate: Mapping[str, Any] | None = None,
    hold_book_active: bool = False,
    hold_book_diag: Mapping[str, Any] | None = None,
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
            free-text ``block_reason``. EMPTY when the portfolio
            optimizer is the authoritative entry driver (legacy
            ``_plan_entries`` bypassed); the missing information is
            then sourced from ``optimizer_shadow_log``.
        risk_events: ``plan.risk_events`` — structured rule log with
            ``event_type`` / ``rule`` / ``value`` / ``threshold``.
            Same EMPTY-on-optimizer-path caveat as ``blocked_entries``.
        market_regime: effective regime for the run.
        run_date / signal_date / prediction_date: lineage dates.
        calendar_date / trading_day: dual-tracked dates (LdP).
        run_id: ``YYMMDDHHMM`` from ``new_eval_run_id``.
        optimizer_shadow_log: when present, sources synthetic blocked /
            risk_event records from ``log["eligibility"]`` +
            ``log["eligibility_reasons"]`` so the rationale answers
            "why didn't ticker X enter?" for the optimizer-driven path
            (``use_portfolio_optimizer: true``). Synthetic records are
            appended to ``blocked_entries`` / ``risk_events`` — the
            legacy lists win on conflicts (legacy path is more
            authoritative when it ran).
        current_positions: ``{ticker: position_dict}`` of held tickers
            from the IB portfolio. This is the **authoritative source**
            for whether a ticker is currently held — NOT the research
            ``signals.json["hold"]`` bucket (which is a research
            recommendation, not portfolio truth). Used to (a) set the
            ``STATE_HELD`` terminal state and the per-record ``held``
            boolean, (b) extend the considered universe so held tickers
            appear in the table even when research is silent on them,
            and (c) suppress synthetic optimizer rejections for tickers
            already in the portfolio (those exit via ``urgent_exits``).

    Returns:
        Serializable payload dict (canonical eval_artifacts shape).
    """
    # Optimizer-aware augmentation — see _synthesize_optimizer_rejections
    # docstring. Synthesized records extend the legacy lists; existing
    # legacy entries take precedence (the legacy gate, when it ran, is
    # the authoritative source).
    _signals_by_ticker_for_synth: dict[str, dict[str, Any]] = {}
    for bucket in ("enter", "hold", "exit", "reduce"):
        for sig in signals.get(bucket, []) or []:
            t = sig.get("ticker") if isinstance(sig, Mapping) else None
            if t:
                _signals_by_ticker_for_synth[t] = dict(sig)
    _held_set = (
        {t for t in (current_positions or {}).keys() if t}
        if current_positions is not None
        else set()
    )
    _legacy_blocked_tickers = {
        (b.get("ticker") if isinstance(b, Mapping) else None)
        for b in (blocked_entries or [])
    }
    _legacy_event_tickers = {
        (ev.get("ticker") if isinstance(ev, Mapping) else None)
        for ev in (risk_events or [])
    }
    _synth_blocked, _synth_events = _synthesize_optimizer_rejections(
        shadow_log=optimizer_shadow_log,
        signals_by_ticker=_signals_by_ticker_for_synth,
        current_positions=_held_set,
    )
    # De-duplicate against legacy entries by ticker so a ticker the
    # legacy path already explained doesn't get a second synthetic row.
    blocked_entries = list(blocked_entries or []) + [
        b for b in _synth_blocked
        if b.get("ticker") not in _legacy_blocked_tickers
    ]
    risk_events = list(risk_events or []) + [
        e for e in _synth_events
        if e.get("ticker") not in _legacy_event_tickers
    ]
    enter = {s.get("ticker"): s for s in signals.get("enter", []) if s.get("ticker")}
    research_hold = {
        s.get("ticker"): s for s in signals.get("hold", []) if s.get("ticker")
    }
    exited = {s.get("ticker"): s for s in signals.get("exit", []) if s.get("ticker")}
    reduced = {s.get("ticker"): s for s in signals.get("reduce", []) if s.get("ticker")}

    # Per-ticker optimizer view, sourced from the shadow log's parallel
    # lists (tickers / current_weights / target_weights / alpha_hat /
    # eligibility). Empty dict for legacy non-optimizer runs.
    optimizer_view: dict[str, dict[str, Any]] = {}
    if isinstance(optimizer_shadow_log, Mapping):
        _tks = optimizer_shadow_log.get("tickers") or []
        _cw = optimizer_shadow_log.get("current_weights") or []
        _tw = optimizer_shadow_log.get("target_weights") or []
        _ah = optimizer_shadow_log.get("alpha_hat") or []
        _el = optimizer_shadow_log.get("eligibility") or []
        for i, tk in enumerate(_tks):
            if not tk:
                continue
            optimizer_view[tk] = {
                "current_weight": _cw[i] if i < len(_cw) else None,
                "target_weight": _tw[i] if i < len(_tw) else None,
                "alpha_hat": _ah[i] if i < len(_ah) else None,
                "eligible": (
                    bool(_el[i]) if i < len(_el) and _el[i] is not None else None
                ),
            }

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

    # The considered universe is "tickers with at least one possible
    # order-book interaction today." Research HOLD / EXIT / REDUCE on a
    # ticker we do NOT currently hold is an informational research view
    # with no actionable side — there is nothing to sell, and HOLD by
    # definition does not change position. Such signals are dead for
    # the order-book rationale and are filtered out so they don't bulk
    # up the table with rows the operator cannot act on. Research HOLD /
    # EXIT / REDUCE on a held ticker is still surfaced — those tickers
    # are in _held_set and pick up the STATE_HELD / urgent_exit /
    # reduce path naturally.
    considered = (
        set(enter)
        | set(approved) | set(urgent) | set(blocked) | set(first_event)
        | _held_set
    )

    records: list[dict[str, Any]] = []
    for ticker in considered:
        sig = (
            enter.get(ticker) or research_hold.get(ticker)
            or exited.get(ticker) or reduced.get(ticker)
        )
        pred = predictions_by_ticker.get(ticker)
        ev = first_event.get(ticker)
        ob_entry = approved.get(ticker)
        ob_exit = urgent.get(ticker)
        is_held = ticker in _held_set
        opt_view = optimizer_view.get(ticker)

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
        elif is_held:
            state = STATE_HELD
            # Surface the optimizer's maintain-decision for held tickers
            # so the chain explains *why* the position is held today
            # (target weight ≈ current weight, no rebalance trade).
            if opt_view is not None:
                _cw = opt_view.get("current_weight")
                _tw = opt_view.get("target_weight")
                chain.append({
                    "stage": "optimizer",
                    "result": "maintain",
                    "detail": (
                        f"tgt {_tw * 100:.2f}% / cur {_cw * 100:.2f}%"
                        if _cw is not None and _tw is not None else None
                    ),
                })
        else:
            state, _na_detail = _classify_no_action(opt_view=opt_view)
            if state == STATE_NO_ACTION_OPTIMIZER_DROPPED:
                # Recording surface #2 (the producer-side CW alarm in
                # optimizer_cutover is #1): an ERROR in the planner log so
                # the dropped allocation is never silent. [[feedback_no_silent_fails]]
                logger.error(
                    "order_book_rationale: %s — optimizer targeted %s but no "
                    "order was created (allocation DROPPED). %s",
                    ticker,
                    (opt_view or {}).get("target_weight"),
                    _na_detail,
                )
            chain.append({
                "stage": "no_action",
                "result": state,
                "detail": _na_detail,
            })

        records.append({
            "ticker": ticker,
            "terminal_state": state,
            "held": is_held,
            "research": _research_block(sig),
            "predictor": _predictor_block(pred),
            "optimizer": opt_view,
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

    # Optimizer reconciliation projection — surfaces target-vs-current
    # NAV state + the rebalance band so consumers can render the
    # three-way "target / current / planned trade / residual gap" view
    # without re-reading the optimizer shadow log. All four fields are
    # None on legacy non-optimizer runs (the shadow log is absent).
    portfolio_nav: float | None = None
    optimizer_trades: list[dict[str, Any]] | None = None
    rebalance_band_pct: float | None = None
    if isinstance(optimizer_shadow_log, Mapping):
        _nav = optimizer_shadow_log.get("portfolio_nav")
        if isinstance(_nav, (int, float)):
            portfolio_nav = float(_nav)
        _trades = optimizer_shadow_log.get("would_be_trades")
        if isinstance(_trades, list):
            optimizer_trades = [dict(t) for t in _trades if isinstance(t, Mapping)]
        _cfg = optimizer_shadow_log.get("optimizer_cfg") or {}
        _band = _cfg.get("rebalance_band_pct") if isinstance(_cfg, Mapping) else None
        if isinstance(_band, (int, float)):
            rebalance_band_pct = float(_band)

    # Single "why did/didn't the book move today" status (schema 1.3.0+).
    # Joined from the summary counts + optimizer diagnostics + the
    # distribution gate / hold-book decision already in hand at the call
    # site — the daily HOLD-vs-fault answer in one field. [[feedback_no_silent_fails]]
    book_status = _build_book_status(
        summary=summary,
        optimizer_shadow_log=optimizer_shadow_log,
        rebalance_band_pct=rebalance_band_pct,
        distribution_gate=distribution_gate,
        hold_book_active=hold_book_active,
        hold_book_diag=hold_book_diag,
        predictions_by_ticker=predictions_by_ticker,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_date": run_date,
        "signal_date": signal_date,
        "prediction_date": prediction_date,
        "market_regime": market_regime,
        "portfolio_nav": portfolio_nav,
        "optimizer_trades": optimizer_trades,
        "rebalance_band_pct": rebalance_band_pct,
        "book_status": book_status,
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
    from nousergon_lib.eval_artifacts import (
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
