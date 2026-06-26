"""Regression tests for the drawdown_forced_exit_* config-namespace fix (config#845).

`drawdown_forced_exit_enabled / _tier2_count / _tier3_count` are documented in
config/risk.yaml.example at the TOP level under `strategy:` (siblings of
`graduated_drawdown:` and `bracket:`). The loader historically read them from the
nested `strategy.graduated_drawdown.*` block, so the documented YAML override was
silently ignored and defaults (True/1/2) were always used.

These tests pin the read path to the documented YAML location: a non-default
override placed where risk.yaml.example puts it MUST propagate through the real
`load_strategy_config`. They fail on the pre-fix code (which reads the nested
block) and pass after the fix.
"""

import os

import yaml

from executor.strategies.config import load_strategy_config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLE_PATH = os.path.join(_REPO_ROOT, "config", "risk.yaml.example")


def test_example_yaml_places_keys_top_level_under_strategy():
    """The documented contract: keys are siblings of graduated_drawdown, not nested."""
    with open(_EXAMPLE_PATH) as f:
        cfg = yaml.safe_load(f)

    strategy = cfg["strategy"]
    # Documented location: top-level under `strategy:`.
    assert "drawdown_forced_exit_enabled" in strategy
    assert "drawdown_forced_exit_tier2_count" in strategy
    assert "drawdown_forced_exit_tier3_count" in strategy
    # And NOT nested inside graduated_drawdown (guards against silent re-nesting).
    grad = strategy.get("graduated_drawdown", {})
    assert "drawdown_forced_exit_enabled" not in grad
    assert "drawdown_forced_exit_tier2_count" not in grad
    assert "drawdown_forced_exit_tier3_count" not in grad


def test_top_level_override_propagates_via_real_loader():
    """Non-default overrides at the documented top-level location must take effect.

    Pre-fix: load_strategy_config read these from strategy.graduated_drawdown, so
    the top-level overrides below were ignored and defaults (True/1/2) returned —
    making the tier-count assertions fail. Post-fix they propagate.
    """
    config = {
        "strategy": {
            "graduated_drawdown": {"enabled": True},
            # Documented top-level location, with NON-default values.
            "drawdown_forced_exit_enabled": False,
            "drawdown_forced_exit_tier2_count": 3,
            "drawdown_forced_exit_tier3_count": 5,
        }
    }

    sc = load_strategy_config(config)

    assert sc["drawdown_forced_exit_enabled"] is False
    assert sc["drawdown_forced_exit_tier2_count"] == 3
    assert sc["drawdown_forced_exit_tier3_count"] == 5


def test_example_yaml_values_round_trip_through_loader():
    """Load the shipped risk.yaml.example and assert the loader honors its values.

    risk.yaml.example documents enabled=true / tier2=1 / tier3=2 at the top level.
    To prove the loader reads THAT location (not a coincidental default match), we
    flip the example's values to non-defaults in-memory before loading.
    """
    with open(_EXAMPLE_PATH) as f:
        cfg = yaml.safe_load(f)

    # Sanity: the example actually ships the documented top-level keys.
    assert "drawdown_forced_exit_tier2_count" in cfg["strategy"]

    # Mutate to non-default values at the documented location.
    cfg["strategy"]["drawdown_forced_exit_enabled"] = False
    cfg["strategy"]["drawdown_forced_exit_tier2_count"] = 7
    cfg["strategy"]["drawdown_forced_exit_tier3_count"] = 9

    sc = load_strategy_config(cfg)

    assert sc["drawdown_forced_exit_enabled"] is False
    assert sc["drawdown_forced_exit_tier2_count"] == 7
    assert sc["drawdown_forced_exit_tier3_count"] == 9


def test_defaults_apply_when_keys_absent():
    """No override anywhere → documented defaults (True/1/2) still apply."""
    sc = load_strategy_config({"strategy": {"graduated_drawdown": {"enabled": True}}})

    assert sc["drawdown_forced_exit_enabled"] is True
    assert sc["drawdown_forced_exit_tier2_count"] == 1
    assert sc["drawdown_forced_exit_tier3_count"] == 2
