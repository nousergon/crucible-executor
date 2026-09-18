"""
Reference-rate showcase artifact — component 3's OWN post-close schedule.

`alpha-engine-config-I11007`: Brian ruled R4 option (b) on 2026-09-14
(`data_collection_plan_260914.md` §7 R4) — the trader publishes
`metron/reference_rate.json` because "it is an execution output of a trader
that keeps running." The ruling was settled; the write did not have a
schedule of its own. It rode inside `executor/eod_reconcile.py`'s `run()`,
which is invoked only via `ne-postclose-trading-pipeline` (the v1 Step
Function `crucible/gate.py::_clause_old_sf_execution_count_zero` grades and
`alpha-engine-config-I10655` deliverable 3 disables at the Crucible v2 phase-4
cutover). Left alone, that cutover freezes this artifact silently —
`architecture.d/146` rule 2: each component runs on its own schedule and its
own stack; component 1's collector already made this move for the Metron
market-data spine (`nousergon-data-PR1701`), this is the same move for
component 3.

This entrypoint is SELF-SUFFICIENT — it does not depend on the v1 postclose
pipeline having fired:

  1. If today's EOD snapshot (`trades/snapshots/{run_date}.json`, written by
     `snapshot_capturer.py`) already exists, read it. During the transition
     period the v1 SF's own `CaptureSnapshot` step still writes this file
     every trading day, so the common case is a plain S3 read with no new IB
     session opened — the timer below is deliberately scheduled AFTER the
     v1 pipeline's typical `CaptureSnapshot` step so this is the normal path
     today, minimizing IB Gateway contention with the still-running v1
     pipeline during the coexistence window.
  2. If it does not exist — the case once v1 is disabled at cutover, or if
     this timer ever fires ahead of the v1 pipeline on a given day — this
     captures one itself via `snapshot_capturer.run()`, the same
     already-tested capture path `eod_reconcile.py` itself depends on
     (idempotent: "re-running on the same run_date overwrites the existing
     snapshot", per that module's own docstring).

`nav_history` is read from the durable `trades/eod_pnl.csv` export
(best-effort — its own producer schedule is a separate, already-tracked gap;
see the module docstring note below). The CURRENT positions + NAV read above
is the artifact's primary content and is never softened by that.

Fails loud, unlike the `eod_reconcile.run()` call site this replaces for
schedule purposes: there this publish is secondary observability hung off an
already-committed EOD run, so a failure there must not override the primary
deliverable. Here it has no sibling deliverable to protect — this script's
entire job is this one artifact, so a failure must raise and page
(fleet default: RAISE, no silent swallows on a producer).

Schedule: `infrastructure/systemd/alpha-engine-reference-rate-publish.timer` — weekday
21:15 UTC, chosen 10 minutes after `eod_reconcile.py`'s own documented
21:05 UTC post-close slot (see that module's docstring) so the snapshot this
reads is normally already there, and while IB Gateway is still up (stopped
only by v1's `StopTradingInstance` step during the coexistence window).

NOTE — a related, already-flagged, NOT-in-scope-here gap: `trades/eod_pnl.csv`
(this script's `nav_history` source) is itself written only inside
`eod_reconcile.run()`, which has the same "no schedule outside the v1
pipeline" problem this issue fixes for `reference_rate.json`. This script
degrades gracefully (empty `nav_history`) if that file is stale or missing;
it does not fix eod_pnl.csv's own schedule. Filed separately —
see the PR description.
"""

from __future__ import annotations

import io
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import boto3
import pandas as pd
from nousergon_lib.dates import now_dual
from nousergon_lib.logging import guard_entrypoint, setup_logging

from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)

_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging("reference-rate", flow_doctor_yaml=_FLOW_DOCTOR_YAML)
logger = logging.getLogger(__name__)

from executor import reference_rate, snapshot_capturer  # noqa: E402 -- must follow setup_logging above
from executor.config_loader import load_config  # noqa: E402


def _load_nav_history(s3, bucket: str) -> list[dict]:
    """Best-effort read of `trades/eod_pnl.csv` for the NAV-vs-SPY curve.

    Never fatal: the artifact's primary content is the CURRENT positions +
    NAV read live (or from today's snapshot), not the history curve. A
    missing or stale csv degrades to an empty history, never blocks the
    publish.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key="trades/eod_pnl.csv")
        eod_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as exc:  # noqa: BLE001 — (a) nav_history unavailable this run; (b) current positions/NAV (the primary content) are unaffected; (c) recorded here at WARNING, the app log stream is the recording surface
        logger.warning("nav_history read failed (non-fatal, current NAV/positions unaffected): %s", exc)
        return []
    return reference_rate.nav_history_from_eod_df(eod_df)


def run(run_date: str | None = None) -> None:
    """Publish `metron/reference_rate.json` for `run_date` (default: today's
    trading day), independent of whether the v1 postclose pipeline fired.
    """
    if run_date is None:
        run_date = now_dual().trading_day
        logger.info("Reference-rate publish | run_date=%s (resolved from now_dual().trading_day)", run_date)
    else:
        logger.info("Reference-rate publish | run_date=%s (explicit)", run_date)

    config = load_config()
    bucket = config["trades_bucket"]
    s3 = boto3.client("s3", region_name=config.get("aws_region", "us-east-1"))

    snapshot = snapshot_capturer.load_snapshot(bucket=bucket, run_date=run_date, region=config.get("aws_region", "us-east-1"))
    if snapshot is None:
        logger.info(
            "No snapshot at s3://%s/trades/snapshots/%s.json yet — capturing one directly "
            "(this schedule does not depend on the v1 postclose pipeline's CaptureSnapshot step)",
            bucket, run_date,
        )
        snapshot_capturer.run(run_date)
        snapshot = snapshot_capturer.load_snapshot(bucket=bucket, run_date=run_date, region=config.get("aws_region", "us-east-1"))
        if snapshot is None:
            raise RuntimeError(
                f"snapshot_capturer.run({run_date!r}) returned without raising but no "
                f"snapshot is readable back at s3://{bucket}/trades/snapshots/{run_date}.json "
                f"— refusing to publish reference_rate.json from no data."
            )

    nav_history = _load_nav_history(s3, bucket)

    payload = reference_rate.build_payload(
        positions=snapshot["positions"],
        nav=snapshot["account"]["net_liquidation"],
        nav_history=nav_history,
        run_date=run_date,
    )
    reference_rate.publish(s3, bucket, payload)
    logger.info("Reference-rate publish complete | run_date=%s", run_date)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Publish metron/reference_rate.json on component 3's own schedule, "
            "independent of the v1 postclose pipeline (alpha-engine-config-I11007)."
        )
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD; defaults to today's trading_day (via nousergon_lib.dates.now_dual).",
    )
    args = parser.parse_args()
    with guard_entrypoint():
        run(run_date=args.date)
