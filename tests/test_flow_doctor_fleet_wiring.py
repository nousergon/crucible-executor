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
    assert telegram_blocks == expected
