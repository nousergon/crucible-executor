"""CI certification: every stock exit rule passes the Slot S conformance kit.

The kit's user-facing entry is ``pytest executor/strategies/conformance
--exit-rule ...`` (its own conftest); this bridge puts the stock-rule
certification inside the default-collected suite so a regression in any
stock rule (or in the kit itself) fails repo CI.
"""

import pytest

from executor.strategies.conformance.kit import conformance_errors
from executor.strategies.contract import stock_registry

_RULES = list(stock_registry()._rules)  # noqa: SLF001 — first-party certification


@pytest.mark.parametrize("rule", _RULES, ids=[r.key for r in _RULES])
def test_stock_rule_passes_slot_s_conformance(rule):
    errors = conformance_errors(rule)
    assert not errors, f"{rule.key} fails Slot S conformance:\n  " + "\n  ".join(errors)
