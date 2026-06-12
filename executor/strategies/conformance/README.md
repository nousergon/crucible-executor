# Slot S conformance kit

**Slot S** is the strategy slot of the Nous Ergon harness: backtestable
exit rules behind a typed Python plugin interface
(`executor/strategies/contract.py`). Slots R and M exchange JSON
artifacts validated by `alpha_engine_lib.contracts`; Slot S's contract is
this interface, and this kit is its conformance check — same role, one
vocabulary (the future `ne validate` verb fronts both).

## Validate your strategy rule

```bash
pytest executor/strategies/conformance --exit-rule "your_pkg.your_mod:YourRule"
```

Repeat `--exit-rule` for multiple rules. Run with no flag to certify the
stock rules (CI does this on every push).

## What you implement

Any object satisfying the `ExitRule` protocol:

```python
from executor.strategies.contract import ExitContext, ExitDecision, RuleOutcome

class MyDrawupRule:
    key = "my_drawup_exit"                # snake_case; becomes the exit reason

    def check(self, ctx: ExitContext) -> RuleOutcome:
        if ctx.current_price is None or not ctx.position.get("avg_cost"):
            return RuleOutcome.none()     # degenerate inputs: decline, never raise
        gain = ctx.current_price / ctx.position["avg_cost"] - 1
        if gain > ctx.config.get("my_drawup_pct", 0.25):
            return RuleOutcome(decision=ExitDecision(
                ticker=ctx.ticker, action="REDUCE", reason=self.key,
                detail=f"gain {gain:.1%} > threshold", extras={"gain_pct": gain}))
        return RuleOutcome.none()
```

## What conformance asserts

| check | meaning |
|---|---|
| C1 interface | `key` is snake_case; object satisfies the protocol |
| C2 shape | decisions are valid: action ∈ {EXIT, REDUCE}, `reason == key`, ticker echoes the context, non-empty detail |
| C3 no-crash | survives every golden scenario, including data-gap shapes (no price, no history, one bar, no avg_cost) |
| C4 purity | deterministic; never mutates the context |
| C5 flags | raised flags are snake_case identifiers |

Golden scenarios live in `scenarios.py` (deterministic, offline). The
chain semantics (ordering, short-circuit, `skip_if_flags`) live in
`ExitRuleRegistry` — your rule doesn't reimplement them.

> Production wiring of external rules (entry-point group
> `alpha_engine.exit_rules`) is deliberately not active until the
> registry cutover lands (config#990, post-2026-06-13). Conformance is
> the prerequisite, not the activation.
