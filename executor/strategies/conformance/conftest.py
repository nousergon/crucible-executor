"""Pytest plumbing for the Slot S conformance kit.

``--exit-rule "pkg.mod:ClassName"`` points the kit at an external
implementation; omitted, the kit certifies the stock rules.
"""

from __future__ import annotations

import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--exit-rule",
        action="append",
        default=[],
        help='ExitRule under test, as "pkg.mod:ClassName" (repeatable). '
             "Omit to certify the stock rules.",
    )


def _load(spec: str):
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise pytest.UsageError(f'--exit-rule must be "pkg.mod:ClassName", got {spec!r}')
    obj = getattr(importlib.import_module(mod_name), attr)
    return obj() if isinstance(obj, type) else obj


def pytest_generate_tests(metafunc):
    if "rule_under_test" not in metafunc.fixturenames:
        return
    specs = metafunc.config.getoption("--exit-rule")
    if specs:
        rules = [_load(s) for s in specs]
    else:
        from executor.strategies.contract import stock_registry
        reg = stock_registry()
        rules = list(reg._rules)  # noqa: SLF001 — kit is first-party to the registry
    metafunc.parametrize("rule_under_test", rules, ids=lambda r: getattr(r, "key", repr(r)))
