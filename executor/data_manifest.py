"""Dated data-manifest writer + S3 object freshness probe (executor-local).

Health enrichment writes live in ``nousergon_lib.health`` (config#1727).
This module keeps executor-specific helpers that are not part of the shared
health schema: dated ``data_manifest/`` PUTs and ``head_object`` freshness
checks used before trusting upstream blobs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)


def write_data_manifest(
    bucket: str,
    module_name: str,
    run_date: str,
    manifest: dict,
) -> None:
    """Write a dated data manifest to S3 at data_manifest/{module}/{date}.json.

    Unlike health files (overwritten each run), manifests are dated and never
    overwritten — the collection of dated files IS the time series.
    """
    payload = {
        "module": module_name,
        "run_date": run_date,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"data_manifest/{module_name}/{run_date}.json",
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Data manifest written: %s/%s", module_name, run_date)
    except Exception as e:
        logger.warning("Failed to write data manifest for %s: %s", module_name, e)


def verify_s3_object_fresh(
    s3_client,
    bucket: str,
    key: str,
    run_date: str,
    max_age_hours: float = 12,
) -> None:
    """Assert an S3 object exists and, when run_date is today (UTC), was
    written within the last max_age_hours.

    Raises RuntimeError on missing or stale. Backfill runs (run_date in the
    past) only get the existence check — historical writes are legitimately
    old.
    """
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise RuntimeError(
            f"s3://{bucket}/{key} not found — upstream did not run or failed."
        ) from exc

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if run_date != today:
        return  # backfill: skip freshness

    age = datetime.now(timezone.utc) - resp["LastModified"]
    age_hours = age.total_seconds() / 3600
    if age_hours > max_age_hours:
        raise RuntimeError(
            f"s3://{bucket}/{key} is stale ({age_hours:.1f}h old, "
            f"max {max_age_hours:.0f}h) — upstream did not refresh today's file."
        )
