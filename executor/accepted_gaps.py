"""Accepted gaps registry — dates with permanently-unrecoverable eod_pnl gaps.

When a trading day's snapshot never existed (CaptureSnapshot failed
permanently, alpha-engine-config-I5569) and is accepted as unrecoverable by
operator ruling (alpha-engine-config-I5325), ``reconcile_audit`` must stop
paging about it every run. This registry provides the durable, queryable
record of such rulings — a liveness fact, not an inference from prose.

S3 location: s3://{trades_bucket}/config/accepted_gaps.json

Schema:
::

    {
      "schema_version": 1,
      "gaps": [
        {
          "date": "2026-07-27",
          "reason": "CaptureSnapshot failed permanently — snapshot never existed",
          "ruling": "nousergon/alpha-engine-config#5325",
          "accepted_at": "2026-07-29T00:00:00Z"
        }
      ]
    }

Adding a future accepted gap requires only a record in this file, not a code
change. Each entry is an operator action tied to a specific ruling — a gap
suppressed without a ruling is a corrupted alpha series with the evidence
deleted (see the Gotcha in alpha-engine-config#5570).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)

ACCEPTED_GAPS_KEY = "config/accepted_gaps.json"

# Declared non-session ROWS (alpha-engine-config-I9613): the opposite defect to
# a gap, a persisted row on a date that was never a session. Checked in rather
# than in S3 because the set is closed (eod_reconcile refuses a non-session
# run_date since PR531). See pnl_integrity._validate_declared_non_sessions.
DECLARED_NON_SESSIONS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "pnl_declared_non_sessions.json"
)


def load_accepted_gaps(
    trades_bucket: str,
    region: str = "us-east-1",
) -> dict[str, dict]:
    """Load the accepted-gaps registry from S3.

    Returns a dict keyed by date string (``"2026-07-27"`` → entry dict).
    Returns an empty dict if the file does not exist or is inaccessible
    — absence of the registry means no gaps have been accepted yet.
    An empty registry is a valid state; every S3 access failure is
    logged but never fatal, because the gap handler downstream degrades
    gracefully (logs at INFO, not WARNING, for accepted gaps) and a
    missing registry at worst means a gap gets flagged when it could
    have been silent — better than silently suppressing a real gap.
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        obj = s3.get_object(Bucket=trades_bucket, Key=ACCEPTED_GAPS_KEY)
        data = json.loads(obj["Body"].read())
        raw_gaps = data.get("gaps", [])
        if not isinstance(raw_gaps, list):
            logger.warning(
                "[accepted_gaps] 'gaps' key is not a list in "
                "s3://%s/%s — treating as empty.",
                trades_bucket, ACCEPTED_GAPS_KEY,
            )
            return {}
        return {g["date"]: g for g in raw_gaps if isinstance(g, dict) and "date" in g}
    except s3.exceptions.NoSuchKey:
        logger.info(
            "[accepted_gaps] no registry at s3://%s/%s — no gaps accepted.",
            trades_bucket, ACCEPTED_GAPS_KEY,
        )
        return {}
    except Exception:
        logger.warning(
            "[accepted_gaps] failed to load from s3://%s/%s — treating as empty.",
            trades_bucket, ACCEPTED_GAPS_KEY, exc_info=True,
        )
        return {}


def load_accepted_gap_entries(
    trades_bucket: str,
    region: str = "us-east-1",
) -> list:
    """Load the registry's RAW ``gaps`` list, malformed entries included.

    ``load_accepted_gaps`` drops an entry with no ``date`` and collapses a
    duplicated date, which is right for a pager but wrong for a gate: the
    session-axis contiguity gate
    (``executor.pnl_integrity.check_session_axis_coverage``,
    alpha-engine-config-I11087) must SEE a malformed entry to reject it loudly.

    Fails closed in the gate's direction: a missing or unreadable registry
    returns ``[]``, which accepts no gap, so every gap still breaches.
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        obj = s3.get_object(Bucket=trades_bucket, Key=ACCEPTED_GAPS_KEY)
        data = json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        logger.info(
            "[accepted_gaps] no registry at s3://%s/%s — no gaps accepted.",
            trades_bucket, ACCEPTED_GAPS_KEY,
        )
        return []
    except Exception:
        logger.warning(
            "[accepted_gaps] failed to load from s3://%s/%s — accepting no gaps.",
            trades_bucket, ACCEPTED_GAPS_KEY, exc_info=True,
        )
        return []
    raw_gaps = data.get("gaps", []) if isinstance(data, dict) else None
    if not isinstance(raw_gaps, list):
        logger.warning(
            "[accepted_gaps] 'gaps' is not a list in s3://%s/%s — accepting no gaps.",
            trades_bucket, ACCEPTED_GAPS_KEY,
        )
        return []
    return raw_gaps


def load_declared_non_sessions(path: Path | str = DECLARED_NON_SESSIONS_PATH) -> list[Any]:
    """Return the raw ``declarations`` list, malformed entries included.

    Fails closed: a missing or unreadable file returns ``[]``, which excuses
    nothing, so the declared row breaches again rather than passing silently.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        logger.warning("[accepted_gaps] cannot read %s — declaring no non-session rows.", path)
        return []
    raw = data.get("declarations") if isinstance(data, dict) else None
    return raw if isinstance(raw, list) else []
