"""Slot S contract tests (config#990) — registry mechanics + the
EQUIVALENCE BATTERY proving ``ExitRuleRegistry(stock_registry()).evaluate``
is behavior-identical to the live ``_evaluate_single_position`` chain.

The equivalence battery is the load-bearing piece: the post-6/13 cutover
(one line — the orchestrator delegating to the registry) is only safe
because every scenario here returns bit-identical (signal, fired_rule_key)
through both paths.
"""

from __future__ import annotations

import pytest

from executor.strategies.conformance.scenarios import _ctx, _ohlc, golden_scenarios
from executor.strategies.contract import (
    ExitContext,
    ExitDecision,
    ExitRuleRegistry,
    RuleOutcome,
    stock_registry,
)
from executor.strategies.exit_manager import _evaluate_single_position

# ── ExitDecision validation ─────────────────────────────────────────────────


def test_decision_rejects_unknown_action():
    with pytest.raises(ValueError):
        ExitDecision(ticker="X", action="ENTER", reason="r", detail="d")


def test_decision_signal_dict_roundtrip():
    d = ExitDecision(ticker="X", action="EXIT", reason="my_rule",
                     detail="why", extras={"stop_level": 9.5})
    assert d.to_signal_dict() == {"ticker": "X", "action": "EXIT", "reason": "my_rule",
                                  "detail": "why", "stop_level": 9.5}


# ── registry mechanics ──────────────────────────────────────────────────────


class _StubRule:
    def __init__(self, key, outcome=None, skip_if_flags=frozenset()):
        self.key = key
        self.skip_if_flags = skip_if_flags
        self._outcome = outcome or RuleOutcome.none()
        self.calls = 0

    def check(self, ctx):
        self.calls += 1
        return self._outcome


def _fire(key):
    return RuleOutcome(decision=ExitDecision(
        ticker="TEST", action="EXIT", reason=key, detail="fired"))


def test_registry_short_circuits_on_first_decision():
    a, b = _StubRule("a", _fire("a")), _StubRule("b", _fire("b"))
    reg = ExitRuleRegistry()
    reg.register(a)
    reg.register(b)
    signal, fired = reg.evaluate(golden_scenarios()["steady_uptrend"])
    assert fired == "a" and signal["reason"] == "a"
    assert b.calls == 0


def test_registry_skip_if_flags_and_flag_reporting():
    flagger = _StubRule("flagger", RuleOutcome(flag="sector_veto_blocked"))
    skipped = _StubRule("skipped", _fire("skipped"),
                        skip_if_flags=frozenset({"sector_veto_blocked"}))
    runs = _StubRule("runs")
    reg = ExitRuleRegistry()
    for r in (flagger, skipped, runs):
        reg.register(r)
    signal, fired = reg.evaluate(golden_scenarios()["steady_uptrend"])
    assert signal is None and fired == "sector_veto_blocked"
    assert skipped.calls == 0 and runs.calls == 1


def test_registry_rejects_duplicates_bad_keys_and_non_rules():
    reg = ExitRuleRegistry()
    reg.register(_StubRule("a"))
    with pytest.raises(ValueError):
        reg.register(_StubRule("a"))
    with pytest.raises(ValueError):
        reg.register(_StubRule("not-an-identifier"))
    with pytest.raises(TypeError):
        reg.register(object())


def test_registry_insert_before():
    reg = ExitRuleRegistry()
    reg.register(_StubRule("a"))
    reg.register(_StubRule("c"))
    reg.register(_StubRule("b"), before="c")
    assert reg.keys == ["a", "b", "c"]
    with pytest.raises(ValueError):
        reg.register(_StubRule("d"), before="zzz")


def test_stock_registry_canonical_order():
    assert stock_registry().keys == [
        "position_loss_floor", "catalyst_hard_exit", "atr_trailing_stop",
        "fallback_stop", "profit_take", "momentum_exit", "time_decay",
    ]


# ── equivalence battery: registry vs _evaluate_single_position ─────────────


def _legacy(ctx: ExitContext):
    return _evaluate_single_position(
        ticker=ctx.ticker, pos=ctx.position, research_action=ctx.research_action,
        current_price=ctx.current_price, history=ctx.price_history,
        sector_etf_histories=ctx.sector_etf_histories, stance_config=ctx.config,
        catalyst_date=ctx.catalyst_date, entry_date=ctx.entry_date,
        run_date=ctx.run_date, feature_lookup=ctx.feature_lookup,
    )


def _equivalence_cases() -> dict[str, ExitContext]:
    cases = dict(golden_scenarios())
    # Targeted path-forcing additions beyond the golden battery:
    steady = [100 + 0.2 * i for i in range(45)]
    # ATR fire with an UNDERPERFORMING sector ETF (veto must NOT suppress)
    cases["atr_fire_no_veto"] = _ctx(
        price_history=_ohlc(steady[:40] + [108, 96, 90, 84, 78]), current_price=77.0,
        sector_etf_histories={"XLK": _ohlc([200 - 0.8 * i for i in range(45)]),
                              "SPY": _ohlc([400 - 0.5 * i for i in range(45)])},
    )
    # ATR fire with an OUTPERFORMING ticker vs collapsing ETF (veto path)
    cases["atr_fire_veto_candidate"] = _ctx(
        price_history=_ohlc(steady[:40] + [108, 99, 95, 92, 89]), current_price=88.0,
        sector_etf_histories={"XLK": _ohlc([200 - 2.5 * i for i in range(45)]),
                              "SPY": _ohlc([400 - 2.0 * i for i in range(45)])},
    )
    # Profit-take zone without momentum breakdown
    cases["profit_take_zone"] = _ctx(
        price_history=_ohlc([100 + 0.9 * i for i in range(45)]), current_price=141.0)
    # Old position + HOLD signal → time decay territory
    cases["time_decay_zone"] = _ctx(entry_date="2025-12-01", research_action="HOLD")
    # Old position + ENTER reaffirmation → decay reset
    cases["time_decay_reset"] = _ctx(entry_date="2025-12-01", research_action="ENTER")
    return cases


@pytest.mark.parametrize("name", sorted(_equivalence_cases().keys()))
def test_registry_equivalent_to_legacy_chain(name):
    ctx = _equivalence_cases()[name]
    legacy_signal, legacy_key = _legacy(ctx)
    reg_signal, reg_key = stock_registry().evaluate(ctx)
    assert reg_key == legacy_key, f"{name}: fired_rule_key diverged"
    assert reg_signal == legacy_signal, f"{name}: signal dict diverged"


def test_equivalence_battery_exercises_multiple_rules():
    """Guard against a vacuous battery: the cases must collectively fire
    several distinct rules (not all no-fire)."""
    fired = {stock_registry().evaluate(ctx)[1] for ctx in _equivalence_cases().values()}
    fired.discard(None)
    assert len(fired) >= 3, f"battery too weak — only fired {fired}"
