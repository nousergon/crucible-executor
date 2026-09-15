"""Tests for executor.upstream_artifact_gate (config#1725 Phase A)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest
import yaml
from nousergon_lib.artifact_freshness import check_freshness, resolve_current_cycle

from executor.upstream_artifact_gate import (
    EXECUTOR_UPSTREAM_SPECS,
    check_upstream_deliverables,
)


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
    last = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    return {"module": "x", "status": "ok", "last_success": last}


# Monday 2026-06-23 14:00 UTC — weekday pre-open, after the 13:00 UTC SF tick.
_MONDAY_PREOPEN = datetime(2026, 6, 23, 14, 0, tzinfo=UTC)
_FRIDAY = datetime(2026, 6, 20, 13, 30, tzinfo=UTC)
_SATURDAY = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
_ANCIENT = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


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
        failures = check_upstream_deliverables(
                "alpha-engine-research", now=_MONDAY_PREOPEN, s3_client=s3
            )

        assert failures
        assert any("predictor_predictions" in f for f in failures)
        assert any("daily_closes_parquet" in f for f in failures)
        # research_signals uses eod_sf (config-I6658) — May 1 is far outside
        # the ~2-trading-day floor before Jun 23
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


class TestExecutorUpstreamSpecsBuildUnderPinnedLib:
    """alpha-engine-config-I10859: every spec the gate builds must construct
    cleanly under the pinned nousergon-lib (>=0.124.129) — in particular none
    may be refused by ``ArtifactSpec._validate_inert_sla_minutes`` (added in
    nousergon-lib PR415/416), which raises ``ValueError`` at construction for
    a declared ``sla_minutes_after_cron`` the freshness floor never consults.

    ``EXECUTOR_UPSTREAM_SPECS`` is built at module import time, so a refused
    spec would already fail collection of this whole file — this test makes
    that withholding shape explicit and names the spec that would fail,
    rather than leaving the class to an opaque collection error.
    """

    @pytest.mark.parametrize(
        "spec", EXECUTOR_UPSTREAM_SPECS, ids=lambda s: s.artifact_id
    )
    def test_spec_constructs_without_validation_error(self, spec):
        # dataclasses.replace() re-runs __post_init__ (and therefore every
        # _validate_* check, including _validate_inert_sla_minutes) against
        # the exact field values EXECUTOR_UPSTREAM_SPECS declares, under
        # whatever nousergon-lib version is pinned right now.
        rebuilt = dataclasses.replace(spec)
        assert rebuilt.artifact_id == spec.artifact_id

    def test_no_spec_declares_an_inert_sla(self):
        """None of the three gate specs use cadence='continuous', the only
        shape ``_validate_inert_sla_minutes`` polices — documents the reason
        none needed a ``deadline_local``/``sla_minutes_after_cron`` fix on
        this pin bump, rather than leaving that absence unexplained.
        """
        for spec in EXECUTOR_UPSTREAM_SPECS:
            assert spec.cadence != "continuous", (
                f"{spec.artifact_id} is cadence='continuous' — re-check it "
                "against ArtifactSpec._validate_inert_sla_minutes by hand"
            )


class TestWeekdaySfAnchorIsDstAware:
    """alpha-engine-config-I10829 gap #2: the weekday_sf cadence anchor is
    declared in America/New_York and converted to UTC per-date (nousergon-lib
    PR415), so the UTC instant moves with the DST offset instead of staying
    fixed at 13:00 UTC year-round. Exercised directly against
    ``predictor_predictions`` (cadence='weekday_sf'), one of the two specs
    this gate's DST-anchor gap named.
    """

    _PREDICTOR_SPEC = next(
        s for s in EXECUTOR_UPSTREAM_SPECS if s.artifact_id == "predictor_predictions"
    )

    def test_edt_anchor_2026_09_15(self):
        # Tuesday 2026-09-15 is EDT (UTC-4): 09:00 America/New_York == 13:00 UTC.
        now = datetime(2026, 9, 15, 15, 0, tzinfo=UTC)
        tick, _label = resolve_current_cycle(self._PREDICTOR_SPEC, now)
        assert tick == datetime(2026, 9, 15, 13, 0, tzinfo=UTC)

    def test_est_anchor_2026_11_03(self):
        # Tuesday 2026-11-03 is EST (UTC-5, post the 2026-11-01 changeover):
        # 09:00 America/New_York == 14:00 UTC.
        now = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
        tick, _label = resolve_current_cycle(self._PREDICTOR_SPEC, now)
        assert tick == datetime(2026, 11, 3, 14, 0, tzinfo=UTC)

    def test_anchor_moves_by_exactly_the_dst_offset(self):
        """The two fixtures above differ by exactly one hour — the DST
        offset — never a fixed-UTC constant that would be wrong on one side
        of the 2026-11-01 changeover.
        """
        edt_tick, _ = resolve_current_cycle(
            self._PREDICTOR_SPEC, datetime(2026, 9, 15, 15, 0, tzinfo=UTC)
        )
        est_tick, _ = resolve_current_cycle(
            self._PREDICTOR_SPEC, datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
        )
        assert (est_tick.hour - edt_tick.hour) % 24 == 1
