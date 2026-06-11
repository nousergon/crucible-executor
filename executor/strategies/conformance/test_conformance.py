"""Slot S conformance kit — THE documented validation command (config#990).

    pytest executor/strategies/conformance --exit-rule "your_pkg.your_mod:YourRule"

Without ``--exit-rule`` this certifies the seven stock rules (the
reference implementations) — which is what repo CI runs.
"""

from executor.strategies.conformance.kit import conformance_errors


def test_rule_conforms_to_slot_s_contract(rule_under_test):
    errors = conformance_errors(rule_under_test)
    assert not errors, (
        f"{getattr(rule_under_test, 'key', rule_under_test)!r} fails Slot S conformance:\n  "
        + "\n  ".join(errors)
    )
