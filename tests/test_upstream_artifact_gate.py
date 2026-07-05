"""Tests for executor.upstream_artifact_gate (config#1725 Phase A)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import yaml

from executor.upstream_artifact_gate import (
    EXECUTOR_UPSTREAM_SPECS,
    check_upstream_deliverables,
)
from nousergon_lib.artifact_freshness import check_freshness


class _ClientError404(Exception):
    def __init__(self) -> None:
        super().__init__("Not Found")
        self.response = {
            "Error": {"Code": "404", "Message": "Not Found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


def _fake_s3(objects: dict[str, datetime]):
    """Minimal S3 mock for date-templated LIST probes."""

    def _paginate(*, Bucket, Prefix):
        contents = [
            {"Key": k, "LastModified": lm}
            for k, lm in objects.items()
            if k.startswith(Prefix)
        ]
        return iter([{"Contents": contents}])

    paginator = mock.Mock()
    paginator.paginate.side_effect = _paginate
    client = mock.Mock()
    client.get_paginator.return_value = paginator
    return client


def _health_ok(hours_ago: float = 1.0) -> dict:
    last = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {"module": "x", "status": "ok", "last_success": last}


# Monday 2026-06-23 14:00 UTC — weekday pre-open, after the 13:00 UTC SF tick.
_MONDAY_PREOPEN = datetime(2026, 6, 23, 14, 0, tzinfo=timezone.utc)
_FRIDAY = datetime(2026, 6, 20, 13, 30, tzinfo=timezone.utc)
_SATURDAY = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
_ANCIENT = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


class TestCheckUpstreamDeliverables:
    def test_blocks_stale_artifacts_even_when_health_self_reports_ok(self):
        """The blind spot: hollow health stamps must not waive stale inputs."""
        s3 = _fake_s3(
            {
                "signals/2026-06-21/signals.json": _ANCIENT,
                "predictor/predictions/2026-06-20.json": _ANCIENT,
                "staging/daily_closes/2026-06-20.parquet": _ANCIENT,
            }
        )
        with mock.patch("executor.health_status.read_health", return_value=_health_ok()):
            failures = check_upstream_deliverables(
                "alpha-engine-research", now=_MONDAY_PREOPEN, s3_client=s3
            )

        assert failures
        assert any("predictor_predictions" in f for f in failures)
        assert any("daily_closes_parquet" in f for f in failures)
        # research_signals uses saturday_sf 10-day window — May 1 is >10d before Jun 23
        assert any("research_signals" in f for f in failures)

    def test_passes_prior_trading_day_fallback_artifacts(self):
        """Legitimate Fri signals/predictions/closes still pass on Monday pre-open."""
        s3 = _fake_s3(
            {
                "signals/2026-06-21/signals.json": _SATURDAY,
                "predictor/predictions/2026-06-20.json": _FRIDAY,
                "staging/daily_closes/2026-06-20.parquet": _FRIDAY,
            }
        )
        failures = check_upstream_deliverables(
            "alpha-engine-research", now=_MONDAY_PREOPEN, s3_client=s3
        )
        assert failures == []

    def test_blocks_when_required_artifact_missing(self):
        s3 = _fake_s3({})
        failures = check_upstream_deliverables(
            "alpha-engine-research", now=_MONDAY_PREOPEN, s3_client=s3
        )
        assert len(failures) == 3
        assert all("missing" in f for f in failures)


_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2].parent
    / "alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml"
)


@pytest.fixture(scope="module")
def registry_rows():
    if not _REGISTRY_PATH.is_file():
        pytest.skip("ARTIFACT_REGISTRY.yaml not available locally")
    data = yaml.safe_load(_REGISTRY_PATH.read_text())
    return {row["artifact_id"]: row for row in data["artifacts"]}


class TestExecutorUpstreamSpecsRegistryParity:
    """Gate specs must stay aligned with ARTIFACT_REGISTRY.yaml."""

    @pytest.mark.parametrize("spec", EXECUTOR_UPSTREAM_SPECS, ids=lambda s: s.artifact_id)
    def test_spec_matches_registry(self, spec, registry_rows):
        row = registry_rows[spec.artifact_id]
        assert spec.s3_key_template == row["s3_key_template"]
        assert spec.cadence == row["cadence"]
        assert spec.sla_minutes_after_cron == row["sla_minutes_after_cron"]
        assert spec.severity == row["severity"]
        assert spec.owner_repo == row["owner_repo"]

    def test_each_gate_spec_is_fresh_on_prior_trading_day_fixture(self, registry_rows):
        """Sanity: the acceptance fixture is fresh per check_freshness directly."""
        s3 = _fake_s3(
            {
                "signals/2026-06-21/signals.json": _SATURDAY,
                "predictor/predictions/2026-06-20.json": _FRIDAY,
                "staging/daily_closes/2026-06-20.parquet": _FRIDAY,
            }
        )
        for spec in EXECUTOR_UPSTREAM_SPECS:
            result = check_freshness(s3, spec, _MONDAY_PREOPEN)
            assert result.state == "fresh", (
                f"{spec.artifact_id} unexpectedly {result.state}: {result.reason}"
            )
