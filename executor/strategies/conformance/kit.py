"""Slot S conformance checks — the validation battery behind the kit.

``conformance_errors(rule)`` returns a flat list of human-readable
violations (empty = conformant), mirroring
``nousergon_lib.contracts.conformance_errors`` for slots R/M so the
future ``ne validate`` CLI can front both with one vocabulary.

What conformance asserts (the Slot S contract, executor/strategies/contract.py):

  C1  interface  — ``key`` is a valid snake_case identifier; ``check`` is
      callable; the object satisfies the ExitRule protocol.
  C2  shape      — on every golden scenario, ``check`` returns a
      RuleOutcome whose decision (when present) is a valid ExitDecision:
      action in {EXIT, REDUCE}, reason == rule.key, ticker == ctx.ticker,
      non-empty detail.
  C3  no-crash   — no exception on any golden scenario, including the
      degenerate data-gap shapes the live executor produces.
  C4  purity     — two evaluations of the same scenario return equal
      outcomes (determinism), and the context's position dict / config
      dict / price index are not mutated.
  C5  flag use   — flags (when raised) are snake_case identifiers.
"""

from __future__ import annotations

import copy

from executor.strategies.conformance.scenarios import golden_scenarios
from executor.strategies.contract import (
    ALLOWED_ACTIONS,
    ExitRule,
    RuleOutcome,
)


def _outcome_repr(outcome: RuleOutcome) -> tuple:
    d = outcome.decision
    return (
        None if d is None else (d.ticker, d.action, d.reason, d.detail, sorted(d.extras.items())),
        outcome.flag,
    )


def conformance_errors(rule: object) -> list[str]:
    """Run the full battery against ``rule``; return violations (empty = pass)."""
    errors: list[str] = []

    # C1 — interface
    if not isinstance(rule, ExitRule):
        errors.append("C1: object does not satisfy the ExitRule protocol (needs .key and .check(ctx))")
        return errors  # nothing else is checkable
    key = getattr(rule, "key", None)
    if not isinstance(key, str) or not key.isidentifier() or key != key.lower():
        errors.append(f"C1: key must be a snake_case identifier string, got {key!r}")

    for name, ctx in golden_scenarios().items():
        pos_before = copy.deepcopy(ctx.position)
        cfg_before = copy.deepcopy(ctx.config)
        hist_len_before = None if ctx.price_history is None else len(ctx.price_history)

        # C3 — no-crash
        try:
            outcome = rule.check(ctx)
        except Exception as e:  # noqa: BLE001 — the kit's whole job is to catch these
            errors.append(f"C3[{name}]: check() raised {type(e).__name__}: {e}")
            continue

        # C2 — shape
        if not isinstance(outcome, RuleOutcome):
            errors.append(f"C2[{name}]: check() must return RuleOutcome, got {type(outcome).__name__}")
            continue
        d = outcome.decision
        if d is not None:
            if d.action not in ALLOWED_ACTIONS:
                errors.append(f"C2[{name}]: action {d.action!r} not in {ALLOWED_ACTIONS}")
            declared = getattr(rule, "reasons", None) or {rule.key}
            if d.reason not in declared:
                errors.append(
                    f"C2[{name}]: reason {d.reason!r} not in declared vocabulary {sorted(declared)}"
                )
            if d.ticker != ctx.ticker:
                errors.append(f"C2[{name}]: ticker {d.ticker!r} != ctx.ticker {ctx.ticker!r}")
            if not d.detail:
                errors.append(f"C2[{name}]: decision detail must be non-empty")

        # C5 — flag vocabulary
        if outcome.flag is not None and (
            not isinstance(outcome.flag, str) or not outcome.flag.isidentifier()
        ):
            errors.append(f"C5[{name}]: flag must be a snake_case identifier, got {outcome.flag!r}")

        # C4 — purity: determinism + no input mutation
        try:
            second = rule.check(ctx)
        except Exception as e:  # noqa: BLE001
            errors.append(f"C4[{name}]: second evaluation raised {type(e).__name__}: {e}")
            continue
        if _outcome_repr(outcome) != _outcome_repr(second):
            errors.append(f"C4[{name}]: non-deterministic — two evaluations of the same context differ")
        if ctx.position != pos_before:
            errors.append(f"C4[{name}]: check() mutated ctx.position")
        if ctx.config != cfg_before:
            errors.append(f"C4[{name}]: check() mutated ctx.config")
        if hist_len_before is not None and len(ctx.price_history) != hist_len_before:
            errors.append(f"C4[{name}]: check() mutated ctx.price_history")

    return errors
