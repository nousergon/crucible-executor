"""Executor upstream gate — independent artifact freshness (config#1725 Phase A).

Replaces the self-reported ``health/{module}.json`` interlock with
:func:`nousergon_lib.artifact_freshness.check_freshness` on the three
upstream deliverables the morning planner requires. Self-reported health
may still be written for observability but is not consulted here (Phase C
may surface it as enrichment only).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any

import boto3
from nousergon_lib.artifact_freshness import ArtifactSpec, check_freshness

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"

# Inline mirror of ARTIFACT_REGISTRY.yaml rows the executor gate requires.
# Keep in sync — tests/test_upstream_artifact_gate.py asserts registry parity.
EXECUTOR_UPSTREAM_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        artifact_id="research_signals",
        s3_bucket=_DEFAULT_BUCKET,
        s3_key_template="signals/{trading_day}/signals.json",
        cadence="saturday_sf",
        sla_minutes_after_cron=180,
        severity="critical",
        owner_repo="alpha-engine-research",
        created_at=date(2026, 5, 27),
    ),
    ArtifactSpec(
        artifact_id="predictor_predictions",
        s3_bucket=_DEFAULT_BUCKET,
        s3_key_template="predictor/predictions/{trading_day}.json",
        cadence="weekday_sf",
        sla_minutes_after_cron=60,
        severity="critical",
        owner_repo="alpha-engine-predictor",
        created_at=date(2026, 5, 27),
    ),
    ArtifactSpec(
        artifact_id="daily_closes_parquet",
        s3_bucket=_DEFAULT_BUCKET,
        s3_key_template="staging/daily_closes/{trading_day}.parquet",
        cadence="weekday_sf",
        sla_minutes_after_cron=30,
        severity="critical",
        owner_repo="alpha-engine-data",
        created_at=date(2026, 5, 27),
    ),
)

_BLOCK_STATES = frozenset({"missing", "stale", "probe_failed"})


def check_upstream_deliverables(
    bucket: str,
    *,
    now: datetime | None = None,
    s3_client: Any | None = None,
) -> list[str]:
    """Probe required upstream deliverables; return failure lines (empty ⇒ pass)."""
    now_utc = now or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    else:
        now_utc = now_utc.astimezone(UTC)

    client = s3_client or boto3.client("s3")
    failures: list[str] = []

    for spec in EXECUTOR_UPSTREAM_SPECS:
        effective = replace(spec, s3_bucket=bucket) if spec.s3_bucket != bucket else spec
        try:
            result = check_freshness(client, effective, now_utc)
        except Exception as exc:  # noqa: BLE001 — gate must fail loud
            failures.append(
                f"{spec.artifact_id}: probe error — {type(exc).__name__}: {exc}"
            )
            logger.exception("Upstream artifact probe failed for %s", spec.artifact_id)
            continue

        if result.state in _BLOCK_STATES:
            failures.append(f"{spec.artifact_id}: {result.state} — {result.reason}")

    return failures
