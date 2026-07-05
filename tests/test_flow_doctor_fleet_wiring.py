"""Executor flow-doctor.yaml stays aligned with nousergon_lib.flow_doctor_fleet."""

from __future__ import annotations

from pathlib import Path

import yaml

from nousergon_lib.flow_doctor_fleet import (
    EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS,
    fleet_telegram_notifier_dicts,
)


def test_executor_flow_doctor_yaml_telegram_matches_fleet_canonical():
    yaml_path = Path(__file__).resolve().parents[1] / "flow-doctor.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    telegram_blocks = [n for n in raw["notify"] if n.get("type") == "telegram"]
    expected = fleet_telegram_notifier_dicts(EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS)
    assert len(telegram_blocks) == len(expected)
    for act, exp in zip(telegram_blocks, expected):
        for key, value in exp.items():
            assert act.get(key) == value, f"telegram notifier mismatch on {key!r}"


def test_executor_flow_doctor_yaml_github_routes_to_config_backlog():
    """config#1695 — CODE/CONFIG issues file to alpha-engine-config, not local repo."""
    yaml_path = Path(__file__).resolve().parents[1] / "flow-doctor.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    github = next(n for n in raw["notify"] if n.get("type") == "github")
    assert github["repo"] == "nousergon/alpha-engine-config"
    assert github.get("notify_on_category") == ["CODE", "CONFIG"]
    assert "area:executor" in github.get("labels", [])
    ops_health = next(
        n
        for n in raw["notify"]
        if n.get("type") == "telegram"
        and n.get("message_thread_id") == "${FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH}"
    )
    assert ops_health.get("notify_on_category") == ["TRANSIENT", "EXTERNAL", "INFRA"]
