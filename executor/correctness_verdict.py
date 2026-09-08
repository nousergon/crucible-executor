"""correctness_verdict.py — the executor's §2.3a consumer half.

Tracked as ``alpha-engine-config-I9466``.

WHY THIS EXISTS
---------------
``sf-pipeline-policy.md`` §2.3a rule 1 is explicit:

    *the verdict is consumed by every stage whose output depends on it being
    true — not merely by whatever happens to read it today. A stage that
    reports, grades, promotes or acts on the run's numbers depends on those
    numbers being uncontaminated, whether or not it currently reads the check
    that says so.*

Measured 2026-08-31: ``crucible-executor`` contained **no reference to an
attestation, a verdict, or a correctness artifact anywhere in its tree**, and
``ne-preopen-trading-pipeline`` (84 states) gated on ``MarketHoursGate``,
``DeployDriftGate``, ``TradingDayGate``, ``CodeFreshnessGate``,
``CheckPredictorCoverage``/``FinalCoverageGate`` and per-stage skip/retry
checks — **not one of which reads a §2.3a verdict**.

The executor is the sharpest instance of rule 1 in the fleet. The Director
files issues off unattested numbers; the executor **places orders** off them.
``config/executor_params.json`` — ``min_score``, ``max_position_pct``,
``atr_multiplier``, the time-decay ladder — is produced by the weekly
backtester's optimizer from the very simulation arithmetic that
``backtest/{run_date}/attestation.json`` exists to attest. The system has been
trading on parameters whose arithmetic may never have been checked, and
nothing anywhere said so.

WHAT A VERDICT MAY AND MAY NOT DO HERE
--------------------------------------
§2.3a rule 4 (Brian ruling 2026-08-22, ``alpha-engine-config-I8187``): what a
non-PASS verdict withholds is **mutating authority, scoped by mutation class**
— never a code path and never the consumer's whole cycle. The test is *what
authority does this action exercise*, not *was the run clean*.

Applied here, the split is not close:

===========================  ===========  =====================================
Action                       Class        Under a non-PASS verdict
===========================  ===========  =====================================
Place a NEW ENTRY order      MUTATES IB   **WITHHELD.** A new position is a
                                          durable write into the broker, sized
                                          by the parameters in question.
Place an EXIT / stop /       MUTATES IB   **RUNS.** Never gated. See below.
urgent-exit / cover order
Read signals, predictions,   READ         Runs.
prices
Write the order book,        OWN LEDGER   Runs.
decision-capture rows,
risk-event rows
Send the morning email       ANNOTATE     Runs, **marked** with the verdict.
===========================  ===========  =====================================

**Exits are never gated, and this is the load-bearing carve-out.** Blocking an
exit does not withhold a guarantee, it withholds risk management: it leaves
open positions with no stop, no time-decay exit and no urgent-exit path, on a
book that was sized by exactly the parameters now in doubt. §1.2 of the same
policy makes the cost of *not* trading a first-class input, and three
consecutive lost sessions (2026-08-05–08-07) are the measured instance. A gate
that stops the repair actions is the failure mode §2.3a rule 4's amendment
record was written about — over-broad withholding is not the safe direction.

So the withheld set is exactly one action: **opening a new position.**

WHICH UPSTREAM VERDICTS GATE A TRADING DECISION
-----------------------------------------------
Stated in code (:data:`UPSTREAM_VERDICTS`) rather than inferred, because §2.3a
rule 4's first obligation is *name the actions, not the pass* — a dependency
that lives only in a reader's head grows silently.

- ``backtest/latest/attestation.json`` — **GATING.** It attests the simulation
  arithmetic (fills, fees, NAV marking, classification counts) that produced
  ``config/executor_params.json``. This is the only artifact on the daily path
  whose falsity would make today's *sizing* wrong.
- ``director/latest/action_plan.json`` — **ADVISORY, conditionally.** The
  Director's plan is prose Brian reads; the executor does not trade on it.
  But ``executor/derisk_gate.py`` reads ``director/carryover_ledger.json`` and
  composes a **sizing multiplier** from it, so the moment
  ``derisk_on_expectancy_enabled`` is set true in ``risk.yaml`` the Director
  becomes a gating upstream of an entry decision. That flag is ``False`` today
  (``DERISK_ON_EXPECTANCY_ENABLED_DEFAULT``), which is why this is recorded as
  a **conditional** row with its own predicate rather than as advisory full
  stop. Reading the current flag value is not the same as reading the
  dependency, and rule 1 is about the dependency.
- ``predictor/predictions/{date}.json`` — covered by the pipeline's existing
  ``CheckPredictorCoverage``/``FinalCoverageGate``, which is a *coverage*
  check, not a correctness verdict. Named here so its absence from this
  module is a recorded decision rather than an oversight.

THE BINDING IS PROVENANCE, NOT RECENCY
---------------------------------------
``executor_params.json`` is written weekly; the preopen runs daily. Requiring
the attestation's ``run_date`` to equal *today* would fire the gate on four
days in five, which is not a guard, it is an outage on a timer.

The correct predicate is that the verdict must attest **the cycle that produced
the parameters actually being loaded** — ``attestation.run_date ==
executor_params.updated_at``. That is not an inherited verdict; the artifact
and its attestation are two halves of one cycle, and they travel together or
neither is used. A verdict stamped with any other ``run_date`` is
:data:`STALE_BINDING` and blocks, because it says nothing about the numbers in
the file the executor just read.

FAIL-CLOSED, AND THE THREE NON-PASS STATES STAY DISTINCT
---------------------------------------------------------
Only the literal string ``"PASS"`` grants the guarantee. ``"ok"``, ``"pass"``,
``True``, ``1`` and a missing key are **not** a pass — the truthiness read is
the bug this module exists to make unwritable. An absent object is
:data:`UNKNOWN`, never "an older artifact, proceed".

``FAIL``, ``UNKNOWN`` and ``STALE_BINDING`` all block and are reported
separately, because they call for different operator actions: ``FAIL`` means
the arithmetic is wrong and the week's parameters must be rolled back;
``UNKNOWN`` means nobody checked and the producer needs re-running;
``STALE_BINDING`` means the two artifacts came from different cycles, which is
a pipeline-ordering defect rather than a numbers defect.

OBSERVE-FIRST (§7a) — THIS SHIPS NOT ENFORCING, ON PURPOSE
-----------------------------------------------------------
``sf-pipeline-policy.md`` §7a (clause ``SFP-7a-new-guard-observes-first``): a
check newly added to a scheduled pipeline path whose verdict can halt a stage
**runs in observe mode for a declared number of cycles, carries its promotion
criterion in its own module, and is loud while observing.**

The system is trading live paper right now — IB Gateway up, orders placed every
session. A new halting guard that enforces on its first production run is
exactly the §7a pattern that lost the 2026-08-17 EOD its ArcticDB append and
its reconcile. So :data:`MODE_OBSERVE` is the default, and in observe mode the
gate **never blocks anything**: it computes the verdict, logs it at WARNING
when non-PASS, writes its artifact, and returns
``blocks_new_entries=False``.

**Promotion criterion — declared here, in the guard's own module, per §7a:**

.. data:: PROMOTION_CRITERION

Promotion is a config edit (``correctness_verdict_gate_mode: enforce`` in
``risk.yaml``), reviewable as a one-line diff, never a code change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

__all__ = [
    "ATTESTATION_KEY", "EXECUTOR_PARAMS_KEY", "PASS", "FAIL", "UNKNOWN",
    "STALE_BINDING", "MODE_OBSERVE", "MODE_ENFORCE", "PROMOTION_CRITERION",
    "UPSTREAM_VERDICTS", "WITHHELD_ACTIONS", "UNGATED_ACTIONS",
    "CorrectnessVerdictState", "evaluate_correctness_verdict",
    "operator_message", "read_verdict",
]

#: The backtester's own known-answer battery over the simulation arithmetic
#: that produced ``config/executor_params.json``. A fixed ``latest/`` pointer
#: rather than ``backtest/{date}/attestation.json``: a date-templated key
#: renders absent every day the template is not the key that exists, and
#: "absent" is a blocking state here (same reasoning that chose the
#: ``director/latest/`` pointer — nous-ergon-ops-PR614).
ATTESTATION_KEY = "backtest/latest/attestation.json"

#: The artifact whose contents the verdict above attests.
EXECUTOR_PARAMS_KEY = "config/executor_params.json"

#: The closed vocabulary. ``STALE_BINDING`` is assigned by this module alone
#: and no producer may ever write it — it is a fact about the RELATIONSHIP
#: between two artifacts, not about either one of them.
PASS = "PASS"  # noqa: S105 — a verdict value, not a credential
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
STALE_BINDING = "STALE_BINDING"

MODE_OBSERVE = "observe"
MODE_ENFORCE = "enforce"

#: §7a: the promotion criterion lives in the guard's own module, not in a plan
#: doc that the guard cannot read and that nobody re-reads.
PROMOTION_CRITERION = (
    "Promote to enforce when BOTH hold: (1) ten consecutive preopen runs have "
    "recorded verdict=PASS with blocks_new_entries=False and no STALE_BINDING "
    "observation, evidencing that the weekly->daily provenance binding "
    "(attestation.run_date == executor_params.updated_at) actually holds "
    "across a weekly boundary; and (2) at least one observed non-PASS cycle "
    "has been reconciled by hand against the weekly run that caused it, "
    "evidencing that the operator message names a real and actionable cause. "
    "Criterion (2) is the one that matters: ten green cycles prove only that "
    "the gate is quiet, which is also what a gate reading the wrong key looks "
    "like. Promotion is `correctness_verdict_gate_mode: enforce` in risk.yaml."
)

#: §2.3a rule 4, first obligation — NAME THE ACTIONS, and their mutation class
#: beside each. A gate named after a function grows silently to cover whatever
#: that function later does.
WITHHELD_ACTIONS: dict[str, str] = {
    "open_new_position": (
        "mutates IB — a new entry order is a durable write into the broker, "
        "sized by the parameters this verdict attests"
    ),
}

#: §2.3a rule 4, second obligation — EMIT BOTH SIDES. A withheld list that
#: appears only when something stopped cannot be told apart from a producer
#: that stopped emitting.
UNGATED_ACTIONS: dict[str, str] = {
    "place_exit_order": (
        "mutates IB, and runs under EVERY verdict — stops, time-decay exits, "
        "urgent exits and unintended-short covers protect capital already at "
        "risk. Withholding them would not pause the exposure, it would strip "
        "the risk management off a book sized by the very parameters in doubt"
    ),
    "cancel_order": "mutates IB, reduces exposure — never gated",
    "read_signals_predictions_prices": "reads the world",
    "write_order_book": "writes the executor's own ledger",
    "write_decision_capture": "writes the executor's own ledger",
    "log_risk_event": "writes the executor's own ledger",
    "send_morning_email": "annotates — runs, carrying the verdict state",
}

#: §2.3a rule 1 — the dependency is DECLARED, not inferred from what happens to
#: be read today. ``gating`` says whether a non-PASS verdict on this artifact
#: withholds a trading decision; ``predicate`` records what would have to
#: change for a conditional row to become gating.
UPSTREAM_VERDICTS: dict[str, dict] = {
    ATTESTATION_KEY: {
        "gating": True,
        "attests": EXECUTOR_PARAMS_KEY,
        "why": (
            "the simulation arithmetic (fills, fees, NAV marking, "
            "classification counts) the weekly optimizer derived today's "
            "entry sizing and thresholds from"
        ),
        "predicate": None,
    },
    "director/latest/action_plan.json": {
        "gating": False,
        "attests": "director/carryover_ledger.json",
        "why": (
            "the Director's plan is prose for a human and the executor does "
            "not trade on it — but executor/derisk_gate.py composes a SIZING "
            "MULTIPLIER out of director/carryover_ledger.json, so this "
            "becomes gating the moment the flag below is set"
        ),
        "predicate": "risk.yaml derisk_on_expectancy_enabled is true",
    },
}


@dataclass(frozen=True)
class CorrectnessVerdictState:
    """The §2.3a verdict for one preopen cycle.

    ``blocks_new_entries`` is the only field a caller acts on, and it is
    False in observe mode by construction (§7a).
    """

    verdict: str
    mode: str
    blocks_new_entries: bool
    attested_run_date: str | None
    params_updated_at: str | None
    reason: str
    raw_verdict: object = None
    context: dict = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        """Value comparison against the literal, never truthiness.

        ``"ok"``, ``"pass"``, ``True`` and ``1`` are all falsy here on
        purpose — the truthy read is the defect this module exists to make
        unwritable.
        """
        return self.verdict == PASS

    def to_log_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "mode": self.mode,
            "blocks_new_entries": self.blocks_new_entries,
            "attested_run_date": self.attested_run_date,
            "params_updated_at": self.params_updated_at,
            "reason": self.reason,
            "withheld_actions": sorted(WITHHELD_ACTIONS) if not self.is_pass else [],
            "ungated_actions": sorted(UNGATED_ACTIONS),
            "raw_verdict": self.raw_verdict,
            **self.context,
        }


def _get_json(bucket: str, key: str, s3_client=None) -> tuple[dict | None, str | None]:
    """Read one JSON object. Returns ``(doc, error_reason)``; never raises.

    Never returns a default document. Every failure path is carried out as an
    explicit reason so the caller resolves it to a blocking verdict — a reader
    that returns ``{}`` on a missing object is a reader that grants the
    guarantee by accident.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None, f"s3://{bucket}/{key} is ABSENT"
        return None, f"s3://{bucket}/{key} is unreadable ({code})"
    except Exception as exc:  # noqa: BLE001 — CONTRACT: never raises
        # (a) the S3 read failed for a reason other than a recognized
        # ClientError code (network, credentials, throttling, ...).
        # (c) recording surface: the error is NOT dropped — it is
        # returned in-band as the function's `error` element, which
        # `read_verdict` (the caller) propagates to ITS caller. This is
        # the documented CONTRACT: never raises, an explicit design
        # choice, not a swallow (alpha-engine-config-I10031).
        return None, f"s3://{bucket}/{key} read failed ({type(exc).__name__}: {exc})"
    try:
        doc = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        # (a) the S3 body was not valid JSON.
        # (c) recording surface: same in-band `error` return as above —
        # returned to `read_verdict`, never silently discarded
        # (alpha-engine-config-I10031).
        return None, f"s3://{bucket}/{key} body is not JSON ({type(exc).__name__})"
    if not isinstance(doc, dict):
        return None, f"s3://{bucket}/{key} body is not a JSON object"
    return doc, None


def read_verdict(bucket: str, s3_client=None) -> tuple[str, str | None, object, str]:
    """Normalize ``backtest/latest/attestation.json`` onto the vocabulary.

    Returns ``(verdict, attested_run_date, raw_verdict, reason)``. Every
    degenerate input — absent, unreadable, non-JSON, non-object, a verdict key
    that is missing or is any value other than the literal ``"PASS"`` /
    ``"FAIL"`` — resolves to a NON-PASS value with the cause recorded.
    """
    doc, err = _get_json(bucket, ATTESTATION_KEY, s3_client=s3_client)
    if doc is None:
        return UNKNOWN, None, None, f"{err} — a verdict that was never read is not a pass."
    raw = doc.get("verdict")
    run_date = doc.get("run_date")
    if raw == PASS:
        return PASS, run_date, raw, f"backtester attestation PASS for run_date {run_date!r}."
    if raw == FAIL:
        return (FAIL, run_date, raw,
                f"backtester attestation FAILED for run_date {run_date!r} — "
                f"{doc.get('n_failed')} known-answer check(s) did not pass. This "
                "cycle's simulation arithmetic is NOT trustworthy, and "
                "config/executor_params.json was derived from it.")
    return (UNKNOWN, run_date, raw,
            f"backtester attestation verdict {raw!r} is not the literal 'PASS' "
            "or 'FAIL' — treated as UNKNOWN, never as a pass.")


def evaluate_correctness_verdict(
    bucket: str, config: dict | None = None, s3_client=None,
) -> CorrectnessVerdictState:
    """Evaluate the §2.3a gate for one preopen cycle. Never raises.

    ``config`` is the merged ``risk.yaml`` mapping; only
    ``correctness_verdict_gate_mode`` is read, defaulting to
    :data:`MODE_OBSERVE` (§7a).

    Fail-closed: every path that does not end in a literal ``"PASS"`` over a
    provenance-matched pair returns a non-PASS verdict. In enforce mode that
    sets ``blocks_new_entries=True``; in observe mode it never does.
    """
    mode = (config or {}).get("correctness_verdict_gate_mode", MODE_OBSERVE)
    if mode not in (MODE_OBSERVE, MODE_ENFORCE):
        # An unrecognised mode is not silently treated as observe: that would
        # let a typo disarm the guard permanently and invisibly. It is the
        # STRICTER of the two, and it is loud.
        logger.error(
            "correctness gate: correctness_verdict_gate_mode=%r is not %r or %r "
            "— reading it as %r, the stricter value. A typo must never disarm a "
            "guard.", mode, MODE_OBSERVE, MODE_ENFORCE, MODE_ENFORCE,
        )
        mode = MODE_ENFORCE

    verdict, attested_run_date, raw, reason = read_verdict(bucket, s3_client=s3_client)

    params, params_err = _get_json(bucket, EXECUTOR_PARAMS_KEY, s3_client=s3_client)
    params_updated_at = params.get("updated_at") if params else None

    if verdict == PASS:
        if params is None:
            # The executor falls back to its bundled risk.yaml when the S3
            # params are absent, which is a legitimate configuration — but the
            # verdict then attests a file that is not in play, so it grants
            # nothing about today's decision.
            verdict = UNKNOWN
            reason = (
                f"{params_err} — the attestation PASSes for run_date "
                f"{attested_run_date!r}, but the artifact it attests is not "
                "present, so it says nothing about the parameters actually in "
                "use this session."
            )
        elif attested_run_date != params_updated_at:
            verdict = STALE_BINDING
            reason = (
                f"the attestation attests run_date {attested_run_date!r} but "
                f"{EXECUTOR_PARAMS_KEY} was written by cycle "
                f"{params_updated_at!r}. A verdict is never inherited across "
                "cycles: these two artifacts came from different weekly runs, "
                "so the parameters in use this session are unattested."
            )

    blocks = (mode == MODE_ENFORCE) and verdict != PASS

    state = CorrectnessVerdictState(
        verdict=verdict,
        mode=mode,
        blocks_new_entries=blocks,
        attested_run_date=attested_run_date,
        params_updated_at=params_updated_at,
        reason=reason,
        raw_verdict=raw,
        context={
            "attestation_key": ATTESTATION_KEY,
            "params_key": EXECUTOR_PARAMS_KEY,
            "promotion_criterion": PROMOTION_CRITERION,
        },
    )

    # Both polarities, every cycle. A line that appears only on the bad case
    # cannot be told apart from a gate that stopped running (principles.md 2.7).
    if state.is_pass:
        logger.info(
            "correctness gate [%s]: PASS — %s Withheld: none. Ungated: %s.",
            mode, reason, ", ".join(sorted(UNGATED_ACTIONS)),
        )
    else:
        logger.error(
            "correctness gate [%s]: %s — %s Withheld: %s. Ungated (running "
            "regardless): %s.",
            mode, verdict, reason, ", ".join(sorted(WITHHELD_ACTIONS)),
            ", ".join(sorted(UNGATED_ACTIONS)),
        )
    return state


def operator_message(state: CorrectnessVerdictState) -> str:
    """What the operator reads when the gate is non-PASS, with the unblocking step.

    Written here rather than at the call site so the SF notification, the
    morning email and the daemon log cannot drift into three different
    accounts of the same condition.
    """
    if state.is_pass:
        return (
            f"CORRECTNESS VERDICT PASS — s3://.../{ATTESTATION_KEY} attests "
            f"run_date {state.attested_run_date}, matching "
            f"{EXECUTOR_PARAMS_KEY} (updated_at {state.params_updated_at}). "
            "New entries permitted."
        )

    unblock = {
        FAIL: (
            "The weekly backtester's known-answer battery FAILED, so the "
            "optimizer's output is not trustworthy. UNBLOCK: roll "
            "config/executor_params.json back to the last cycle whose "
            "attestation PASSed, then re-run the preopen for today "
            "(weekday_sf_rerun.py). Do NOT clear this by re-running the "
            "backtester until the failing check is understood."
        ),
        UNKNOWN: (
            "Nobody checked this cycle's arithmetic — the attestation is "
            "absent, unreadable, or carries a verdict outside the vocabulary. "
            "UNBLOCK: re-run the weekly Backtester stage so it writes "
            "backtest/{run_date}/attestation.json and the latest/ pointer, "
            "then re-run the preopen for today. An absent verdict is not an "
            "older verdict; there is nothing to fall back to."
        ),
        STALE_BINDING: (
            "The attestation and the parameters came from different weekly "
            "cycles, which is a pipeline-ordering defect, not a numbers "
            "defect: one of the two artifacts was written without the other. "
            "UNBLOCK: re-run the weekly Backtester stage so both are written "
            "by one execution, then re-run the preopen for today."
        ),
    }[state.verdict]

    acting = (
        "NEW ENTRIES ARE BLOCKED for this session."
        if state.blocks_new_entries
        else "OBSERVE MODE — nothing is blocked; this is a report, not a halt."
    )
    return (
        f"CORRECTNESS VERDICT {state.verdict} [{state.mode}] — {acting}\n"
        f"  Why: {state.reason}\n"
        f"  Attested run_date: {state.attested_run_date}   "
        f"executor_params updated_at: {state.params_updated_at}\n"
        f"  STILL RUNNING (never gated): exits, stops, time-decay exits, "
        f"urgent exits, unintended-short covers, order cancels, the order "
        f"book, and the morning email. The book is managed as normal.\n"
        f"  {unblock}"
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI — the surface `ne-preopen-trading-pipeline`'s CorrectnessVerdictGate runs
# ════════════════════════════════════════════════════════════════════════════

#: Exit codes. The SF reads these off the SSM invocation's ``ResponseCode`` so
#: FAIL, UNKNOWN and STALE_BINDING reach the operator as three DIFFERENT
#: terminal states rather than one generic "gate failed" — they call for three
#: different actions, and collapsing them is how a pipeline teaches its
#: operator that its alerts are not worth reading.
EXIT_OK = 0
EXIT_FAIL = 3
EXIT_UNKNOWN = 4
EXIT_STALE_BINDING = 5

_EXIT_FOR = {FAIL: EXIT_FAIL, UNKNOWN: EXIT_UNKNOWN, STALE_BINDING: EXIT_STALE_BINDING}


def main(argv: list[str] | None = None) -> int:
    """Evaluate the gate and return the SF's exit code.

    ``--mode`` overrides ``risk.yaml`` so the SF state declares its own
    posture in the definition, where an operator reading the pipeline can see
    it, instead of it being invisible in a config file on a box.

    In ``observe`` mode this ALWAYS returns 0 — §7a: a guard newly added to a
    scheduled path may not take its halt branch while observing. It is loud
    regardless: the verdict and the full operator message go to stdout every
    run, in both polarities.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="executor.correctness_verdict")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--mode", choices=(MODE_OBSERVE, MODE_ENFORCE),
                        default=MODE_OBSERVE)
    args = parser.parse_args(argv)

    state = evaluate_correctness_verdict(
        args.bucket, {"correctness_verdict_gate_mode": args.mode},
    )
    print(operator_message(state))
    print(
        f"CORRECTNESS_VERDICT_GATE verdict={state.verdict} mode={state.mode} "
        f"blocks_new_entries={str(state.blocks_new_entries).lower()} "
        f"attested_run_date={state.attested_run_date} "
        f"params_updated_at={state.params_updated_at}"
    )
    if not state.blocks_new_entries:
        return EXIT_OK
    return _EXIT_FOR[state.verdict]


if __name__ == "__main__":  # pragma: no cover — exercised via main()
    raise SystemExit(main())
