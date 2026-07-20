"""Slot S contract — the strategy (exit-rule) plugin interface (M0 / config#990).

The harness's three experiment slots exchange PRODUCT CONTRACTS (M0
discipline, ratified 2026-06-11 — config#638). Slots R and M are JSON
artifacts with versioned schemas in ``nousergon_lib.contracts``; Slot S
is different in kind: a **Python plugin interface** for backtestable exit
strategy rules. This module is that contract's single source of truth:

- :class:`ExitContext`   — everything a rule may read (frozen; one per
  held position per evaluation).
- :class:`ExitDecision`  — the decision payload a rule returns (the
  formalization of the ad-hoc dicts the stock checks have always built).
- :class:`RuleOutcome`   — decision | flag wrapper (a rule can fire,
  decline, or decline-with-flag, e.g. ``sector_veto_blocked``).
- :class:`ExitRule`      — the runtime-checkable Protocol an
  implementation satisfies.
- :class:`ExitRuleRegistry` — ordered rule chain with the canonical
  short-circuit semantics (first decision wins; later rules can be
  skipped by earlier flags via ``skip_if_flags``).
- :func:`stock_registry` — the seven stock rules (adapters over
  ``exit_manager``'s functions, behavior-identical) registered in
  canonical order: the reference Slot S implementation.

External implementations ("bring your own strategy") subclass nothing —
they provide any object satisfying :class:`ExitRule` and validate it with
the conformance kit (``executor/strategies/conformance/`` — one pytest
command, see its README). Discovery for production use is via the
``alpha_engine.exit_rules`` entry-point group (:func:`load_entry_point_rules`),
mirroring the ``metron.plugins`` precedent.

CONTRACT EVOLUTION is additive-only (mirrors the S3 contract rule):
``ExitContext`` may gain fields; existing fields are never renamed or
removed without a major version bump of ``CONTRACT_VERSION`` and a
deprecation window. Rules MUST tolerate unknown extra context fields.

NOTE (sequencing, config#990): the live ``evaluate_exits`` path still
calls ``_evaluate_single_position`` directly — the registry is wired in
parallel and proven equivalent by ``tests/test_slot_s_contract.py``'s
equivalence battery. The one-line cutover is deliberately deferred
post-2026-06-13 per the issue's runtime-refactor gate.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

CONTRACT_VERSION = 1

#: The only actions a Slot S rule may emit. HOLD is expressed by returning
#: no decision; ENTER is not a Slot S concern (entries belong to the
#: planner/trigger layers).
ALLOWED_ACTIONS = ("EXIT", "REDUCE")


@dataclass(frozen=True)
class ExitContext:
    """Read-only evaluation context for one held position on one run.

    Field semantics (all prices in dollars, dates as ISO ``YYYY-MM-DD``):

    - ``ticker``: position symbol.
    - ``position``: the holder's position dict — at minimum ``avg_cost``
      (float | None) and ``sector`` (str); additive extra keys allowed.
    - ``research_action``: today's research signal for the ticker
      (``ENTER``/``HOLD``/``EXIT``/``REDUCE``) — ``HOLD`` when absent.
    - ``current_price``: latest price (may be None on data gaps — rules
      must tolerate).
    - ``price_history``: OHLC DataFrame (columns open/high/low/close,
      ascending DatetimeIndex) or None.
    - ``sector_etf_histories``: {etf_ticker: OHLC DataFrame} or None.
    - ``config``: the stance-resolved strategy config dict (see
      ``strategies.config.load_strategy_config`` +
      ``_resolve_strategy_config_for_stance``). Rules read their knobs
      here; unknown keys must be ignored.
    - ``catalyst_date`` / ``entry_date``: ISO dates or None.
    - ``run_date``: the trading run date.
    - ``feature_lookup``: optional precomputed-feature accelerator
      (``executor.feature_lookup.FeatureLookup``); rules MUST function
      identically (modulo speed) when it is None.
    """

    ticker: str
    position: dict[str, Any]
    research_action: str
    current_price: float | None
    price_history: Any  # pd.DataFrame | None (kept untyped: no hard pandas dep in the contract)
    sector_etf_histories: dict[str, Any] | None
    config: dict[str, Any]
    catalyst_date: str | None
    entry_date: str | None
    run_date: str
    feature_lookup: Any | None = None


@dataclass(frozen=True)
class ExitDecision:
    """A fired exit decision.

    ``reason`` MUST equal the emitting rule's ``key`` (it becomes the
    order-book exit reason, the trades.db ``trigger_type``, and the
    decision-artifact ``fired_rule`` — one canonical vocabulary).
    ``extras`` carries rule-specific diagnostics (stop levels, ATR values,
    reduce fractions…) — additive, never load-bearing for the executor.
    """

    ticker: str
    action: str
    reason: str
    detail: str
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"ExitDecision.action must be one of {ALLOWED_ACTIONS}, got {self.action!r}"
            )

    def to_signal_dict(self) -> dict[str, Any]:
        """The legacy signal-dict shape ``evaluate_exits`` consumers expect."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "reason": self.reason,
            "detail": self.detail,
            **self.extras,
        }


@dataclass(frozen=True)
class RuleOutcome:
    """What a rule evaluation produced.

    Exactly the (signal, fired_rule_key) duality the stock chain has
    always had: a rule may fire a decision, decline silently, or decline
    while raising a ``flag`` that downstream rules / capture can see
    (today's only stock flag: ``sector_veto_blocked``).
    """

    decision: ExitDecision | None = None
    flag: str | None = None

    @classmethod
    def none(cls) -> RuleOutcome:
        return cls()


@runtime_checkable
class ExitRule(Protocol):
    """The Slot S plugin interface.

    Implementations provide:

    - ``key`` — canonical snake_case identifier; the chain/grading
      identity (``fired_rule_key``, capture vocabulary).
    - ``check(ctx)`` — pure decision function: same ``ExitContext`` in,
      same :class:`RuleOutcome` out; MUST NOT mutate ``ctx`` or its
      members, perform I/O, or raise on degenerate-but-valid inputs
      (missing prices/history/avg_cost → decline, don't crash).
    - ``reasons`` (optional) — the set of decision ``reason`` strings the
      rule may emit; defaults to ``{key}``. Declare variants explicitly
      (stock example: ``time_decay`` emits ``time_decay_exit`` /
      ``time_decay_reduce`` — vocabulary predating this contract, baked
      into trades.db ``trigger_type``).
    - ``skip_if_flags`` (optional) — set of flags raised earlier in the
      chain that suppress this rule for the position (stock example:
      ``fallback_stop`` is skipped after ``sector_veto_blocked``).
    """

    key: str

    def check(self, ctx: ExitContext) -> RuleOutcome: ...


def _validate_key(rule_key: str) -> None:
    if not rule_key or not rule_key.isidentifier() or keyword.iskeyword(rule_key):
        raise ValueError(f"ExitRule.key must be a valid snake_case identifier, got {rule_key!r}")


class ExitRuleRegistry:
    """Ordered Slot S rule chain with the canonical evaluation semantics.

    Order is significance: rules are evaluated in registration order, the
    first decision wins (short-circuit), and a rule whose
    ``skip_if_flags`` intersects already-raised flags is skipped. On
    no-fire the LAST raised flag is reported (mirrors the stock chain's
    ``sector_veto_blocked`` reporting).
    """

    def __init__(self) -> None:
        self._rules: list[ExitRule] = []

    def register(self, rule: ExitRule, *, before: str | None = None) -> None:
        """Append ``rule`` (or insert before the rule keyed ``before``)."""
        if not isinstance(rule, ExitRule):
            raise TypeError(f"{rule!r} does not satisfy the ExitRule protocol")
        _validate_key(rule.key)
        if any(r.key == rule.key for r in self._rules):
            raise ValueError(f"duplicate ExitRule key {rule.key!r}")
        if before is None:
            self._rules.append(rule)
            return
        for i, existing in enumerate(self._rules):
            if existing.key == before:
                self._rules.insert(i, rule)
                return
        raise ValueError(f"no registered rule keyed {before!r} to insert before")

    @property
    def keys(self) -> list[str]:
        return [r.key for r in self._rules]

    def evaluate(self, ctx: ExitContext) -> tuple[dict[str, Any] | None, str | None]:
        """Run the chain. Returns the legacy ``(signal_dict, fired_rule_key)``.

        ``fired_rule_key`` is the firing rule's key, or the last raised
        flag on no-fire, or None — bit-identical to
        ``exit_manager._evaluate_single_position``'s return contract.
        """
        flags: set[str] = set()
        last_flag: str | None = None
        for rule in self._rules:
            skip = getattr(rule, "skip_if_flags", None) or set()
            if flags & set(skip):
                continue
            outcome = rule.check(ctx)
            if outcome.decision is not None:
                return outcome.decision.to_signal_dict(), rule.key
            if outcome.flag:
                flags.add(outcome.flag)
                last_flag = outcome.flag
        return None, last_flag


# ── Stock rules: adapters over the existing exit_manager functions ──────────
#
# Thin, behavior-identical wrappers — the 1000-line battle-tested check
# functions stay the implementation; the adapters only translate
# ExitContext into each function's historical signature. This is the
# reference Slot S implementation the conformance kit certifies.


@dataclass(frozen=True)
class _FunctionRule:
    """Adapter base: wraps a legacy ``check_*`` function as an ExitRule."""

    key: str = ""
    skip_if_flags: frozenset[str] = frozenset()
    reasons: frozenset[str] | None = None  # None -> {key}
    _call: Callable[[ExitContext], dict | None] = lambda ctx: None  # noqa: E731

    def check(self, ctx: ExitContext) -> RuleOutcome:
        raw = self._call(ctx)
        if raw is None:
            return RuleOutcome.none()
        extras = {k: v for k, v in raw.items()
                  if k not in ("ticker", "action", "reason", "detail")}
        return RuleOutcome(decision=ExitDecision(
            ticker=raw["ticker"], action=raw["action"],
            reason=raw["reason"], detail=raw.get("detail", ""), extras=extras,
        ))


class _AtrWithSectorVetoRule:
    """ATR trailing stop + the sector-relative veto, as one composite rule.

    The veto is intrinsically coupled to the ATR fire (it suppresses
    exactly that decision), so Slot S models the pair as a single rule
    that either fires ``atr_trailing_stop`` or raises the
    ``sector_veto_blocked`` flag — preserving the stock chain's exact
    semantics, including ``fallback_stop`` being skipped after a veto.
    """

    key = "atr_trailing_stop"

    def check(self, ctx: ExitContext) -> RuleOutcome:
        from executor.strategies import exit_manager as em

        raw = em.check_atr_trailing_stop(
            ticker=ctx.ticker, current_price=ctx.current_price,
            entry_date=ctx.entry_date, price_history=ctx.price_history,
            strategy_config=ctx.config, feature_lookup=ctx.feature_lookup,
            run_date=ctx.run_date,
        )
        if raw is None:
            return RuleOutcome.none()
        sector = ctx.position.get("sector", "")
        etf_ticker = em.SECTOR_ETF_MAP.get(sector, "SPY")
        etf_history = (ctx.sector_etf_histories or {}).get(etf_ticker)
        if em.check_sector_relative_veto(
            ctx.ticker, sector, ctx.price_history, etf_history, ctx.config,
        ):
            return RuleOutcome(flag="sector_veto_blocked")
        extras = {k: v for k, v in raw.items()
                  if k not in ("ticker", "action", "reason", "detail")}
        return RuleOutcome(decision=ExitDecision(
            ticker=raw["ticker"], action=raw["action"],
            reason=raw["reason"], detail=raw.get("detail", ""), extras=extras,
        ))


def stock_registry() -> ExitRuleRegistry:
    """The seven stock rules in canonical chain order — Slot S's reference
    implementation (loss floor first, decay last; matches
    ``_evaluate_single_position`` exactly, proven by the equivalence
    battery in ``tests/test_slot_s_contract.py``)."""
    from executor.strategies import exit_manager as em

    reg = ExitRuleRegistry()
    reg.register(_FunctionRule(
        key="position_loss_floor",
        _call=lambda ctx: em.check_position_loss_floor(
            ticker=ctx.ticker, current_price=ctx.current_price,
            avg_cost=ctx.position.get("avg_cost"), strategy_config=ctx.config),
    ))
    reg.register(_FunctionRule(
        key="catalyst_hard_exit",
        _call=lambda ctx: em.check_catalyst_hard_exit(
            ticker=ctx.ticker, catalyst_date=ctx.catalyst_date,
            run_date=ctx.run_date, strategy_config=ctx.config),
    ))
    reg.register(_AtrWithSectorVetoRule())
    reg.register(_FunctionRule(
        key="fallback_stop",
        skip_if_flags=frozenset({"sector_veto_blocked"}),
        _call=lambda ctx: em.check_fallback_stop(
            ticker=ctx.ticker, current_price=ctx.current_price,
            entry_price=ctx.position.get("avg_cost"), strategy_config=ctx.config),
    ))
    reg.register(_FunctionRule(
        key="profit_take",
        _call=lambda ctx: em.check_profit_take(
            ticker=ctx.ticker, current_price=ctx.current_price,
            avg_cost=ctx.position.get("avg_cost"), strategy_config=ctx.config),
    ))
    reg.register(_FunctionRule(
        key="momentum_exit",
        _call=lambda ctx: em.check_momentum_exit(
            ticker=ctx.ticker, price_history=ctx.price_history,
            strategy_config=ctx.config, feature_lookup=ctx.feature_lookup,
            run_date=ctx.run_date),
    ))
    reg.register(_FunctionRule(
        key="time_decay",
        # Pre-contract vocabulary baked into trades.db trigger_type — declared,
        # not renamed (renaming = breaking change to downstream consumers).
        reasons=frozenset({"time_decay_exit", "time_decay_reduce"}),
        _call=lambda ctx: em.check_time_decay(
            ticker=ctx.ticker, entry_date=ctx.entry_date, run_date=ctx.run_date,
            signal_action=ctx.research_action, strategy_config=ctx.config),
    ))
    return reg


def load_entry_point_rules(
    registry: ExitRuleRegistry, group: str = "alpha_engine.exit_rules",
) -> list[str]:
    """Discover and register external ExitRules from entry points.

    Each entry point must resolve to an ExitRule class or factory; loading
    is fail-loud (a broken plugin raises at load time, before any trading
    decision). Returns the registered keys. NOT yet called from the live
    path — production wiring rides the post-6/13 cutover (config#990).
    """
    from importlib.metadata import entry_points

    registered = []
    for ep in entry_points(group=group):
        obj = ep.load()
        rule = obj() if isinstance(obj, type) else obj
        registry.register(rule)
        registered.append(rule.key)
    return registered
