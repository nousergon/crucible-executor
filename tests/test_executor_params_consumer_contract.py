"""L4520 — consumer contract for config/executor_params.json (cross-repo).

Pins the executor side of the boundary declared in alpha-engine-config/
private-docs/PIPELINE_CONTRACT.yaml (boundary_id: executor_params), mirroring
the backtester's tests/test_executor_params_producer_contract.py. The loader
(main._load_executor_params_from_s3) APPLIES only _PARAM_MAP + the special
non-numeric keys and silently excludes everything else, so this test is the
chokepoint that catches a producer/consumer key-set drift at PR time instead
of as a tuned-param-that-never-applied in live trading.

The declared sets are hard-coded here (per-repo CI can't import the config
repo's YAML — the test_scanner_consumer_contract.py precedent); the YAML is
the human SoT.
"""
from __future__ import annotations

from executor import main


# Params the producer (backtester optimizer assembler) can emit for the
# executor to APPLY — PIPELINE_CONTRACT.yaml "applied" sections.
PRODUCER_APPLIED_PARAMS = {
    "atr_multiplier", "time_decay_reduce_days", "time_decay_exit_days",
    "min_score", "max_position_pct", "reduce_fraction",
    "atr_sizing_target_risk", "confidence_sizing_min",
    "confidence_sizing_range", "staleness_decay_per_day",
    "earnings_sizing_reduction", "earnings_proximity_days",
    "momentum_gate_threshold", "correlation_block_threshold",
    "profit_take_pct", "momentum_exit_threshold",
    "barrier_win_prob_sizing_min", "barrier_win_prob_sizing_range",
    # L4598 (config#697): stance overlay application wired 2026-06-12 —
    # the silent-drop gap is closed; keys remain triple-gated upstream
    # (rank-IC promotion gate → assembler cutover → soak).
    "stance_size_momentum", "stance_size_value", "stance_size_quality",
    "stance_size_catalyst",
}
PRODUCER_SPECIAL_KEYS = {
    "disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend",
    "barrier_win_prob_sizing_enabled",
}
# Provenance / metadata the producer rides along — the loader must KNOW them
# (no unknown-key WARN) but never apply them.
PRODUCER_METADATA_KEYS = {
    "updated_at", "assembled_by", "fit_target", "best_sharpe", "best_alpha",
    "best_sortino", "improvement_pct", "n_combos_tested", "manual_override",
    "disabled_triggers_updated_at", "barrier_win_prob_sizing_updated_at",
    "barrier_win_prob_sizing_ic", "p_up_sizing_updated_at", "p_up_sizing_ic",
    "stance_sizing_updated_at", "stance_sizing_alpha_spread",
}
STANCE_OVERLAY_KEYS = {
    "stance_size_momentum", "stance_size_value", "stance_size_quality",
    "stance_size_catalyst",
}


def test_param_map_covers_every_applied_producer_param():
    missing = PRODUCER_APPLIED_PARAMS - set(main._PARAM_MAP)
    assert not missing, (
        f"executor _PARAM_MAP is missing producer-emitted param(s) "
        f"{sorted(missing)} — they would be silently dropped from the applied "
        f"set. Add the mapping + _PARAM_VALIDATORS entry, or remove them from "
        f"the contract (PIPELINE_CONTRACT.yaml executor_params) + producer."
    )


def test_every_applied_param_has_a_validator():
    missing = PRODUCER_APPLIED_PARAMS - set(main._PARAM_VALIDATORS)
    assert not missing, (
        f"S3-delivered param(s) {sorted(missing)} have no _PARAM_VALIDATORS "
        f"range — an out-of-range tuned value would apply unchecked."
    )


def test_advisory_known_keys_cover_producer_metadata():
    # The loader's unknown-key WARN must stay SIGNAL: every declared metadata
    # key is known, so a WARN always means genuine contract drift.
    known = main._EXECUTOR_PARAMS_KNOWN_METADATA_KEYS
    missing = PRODUCER_METADATA_KEYS - set(known) - PRODUCER_SPECIAL_KEYS
    assert not missing, (
        f"producer metadata key(s) {sorted(missing)} missing from the "
        f"loader's advisory known-keys list — every live read would log a "
        f"spurious unknown-key WARN, training operators to ignore the warn."
    )


def test_special_keys_accepted():
    # The loader's special-key passthrough must accept every declared special.
    assert PRODUCER_SPECIAL_KEYS == set(main._EXECUTOR_PARAMS_SPECIAL_KEYS)


def test_stance_overlay_wired_and_bounded():
    # L4598 (config#697): the stance overlay is APPLIED — every key maps to the
    # flat top-level config path position_sizer reads, and the validator bounds
    # mirror the producer's emit bounds (stance_sizing_optimizer
    # _SIZE_FLOOR=0.4 / _SIZE_CAP=1.1) so a widened producer emission is
    # rejected loudly (WARN + skip), never applied unchecked.
    for key in STANCE_OVERLAY_KEYS:
        assert main._PARAM_MAP.get(key) == (key,), (
            f"{key} must map to the flat top-level config key position_sizer "
            f"reads (config.get('{key}'))"
        )
        assert main._PARAM_VALIDATORS.get(key) == (float, 0.4, 1.1), (
            f"{key} validator must mirror the producer emit bounds [0.4, 1.1]"
        )
    # Applied params are not metadata — keys must not be double-declared.
    assert not STANCE_OVERLAY_KEYS & set(main._EXECUTOR_PARAMS_KNOWN_METADATA_KEYS)
    assert not STANCE_OVERLAY_KEYS & set(main._EXECUTOR_PARAMS_SPECIAL_KEYS)
