"""
EOD reconciliation — component 3's OWN post-close schedule (`alpha-engine-
config-I11066`), independent of the v1 `ne-postclose-trading-pipeline`.

`I11007` (`executor/publish_reference_rate.py`, `crucible-executor-PR565`)
gave `metron/reference_rate.json` its own schedule because that publish used
to ride inside `executor.eod_reconcile.run()`, reachable only via the v1
postclose SF that Crucible v2 phase 4 (`alpha-engine-config-I10655`
deliverable 3) disables at cutover. `eod_reconcile.run()` ITSELF has the
identical problem: it is invoked only via that SF's `EODReconcile` SSM step
(`executor/daemon.py::_trigger_eod_pipeline`). Left alone, cutover freezes
`trades/eod_pnl.csv`, `trades/trades_full.csv`, `trades/shadow_book.csv`, the
`eod_pnl` SQLite table (the trader's core NAV/P&L record), the EOD email, and
the durable trades.db S3 backup — invisible to the phase-4 gate because
Metron does not read `eod_pnl.csv` directly
(`crucible/gate.py::_clause_metron_reads_have_surviving_producer` never sees
it).

This entrypoint mirrors `publish_reference_rate.py`'s shape exactly (same
issue's precedent, `policy-shared-code`: mirror the SOTA pattern rather than
inventing a parallel one) — SELF-SUFFICIENT, does not depend on the v1
postclose pipeline's `CaptureSnapshot` step having fired:

  1. If today's EOD snapshot (`trades/snapshots/{run_date}.json`, written by
     `snapshot_capturer.py`) already exists, `eod_reconcile.run()` reads it
     as it always has. During the v1/v2 coexistence window the v1 SF's own
     `CaptureSnapshot` step still writes this file every trading day, so
     the common case is a plain S3 read with no new IB session opened by
     this path.
  2. If it does not exist — the case once v1 is disabled at cutover, or if
     this timer ever fires ahead of the v1 pipeline on a given day — this
     captures one itself via `snapshot_capturer.run()`, the SAME
     already-tested capture path `eod_reconcile.run()` itself has always
     depended on for its NAV/positions input (idempotent: "re-running on
     the same run_date overwrites the existing snapshot", per that module's
     own docstring).

`executor.eod_reconcile.run()` is NOT modified by this entrypoint — the v1
SF's own `EODReconcile` SSM step keeps calling `python -m
executor.eod_reconcile` exactly as it does today, unaffected by this file's
existence, so the v1 coexistence path (including the existing hard-fail on
a missing snapshot, `alpha-engine-config-I5569`/`I6705`'s irreversibility
handling and paging) is untouched.

Idempotent end to end: `eod_reconcile.run()`'s own `eod_pnl` write is
`INSERT OR REPLACE` keyed on `date` (`executor/trade_logger.py`), so a
v1-triggered `EODReconcile` and this independent timer both firing on the
same trading day during the coexistence window safely converge on one row
rather than racing or duplicating (the design question `I11066` left open —
resolved here as option (a), "run both in parallel, absorb the duplicate
via existing idempotency" — the same choice `I11007` made for
`reference_rate.json`, requiring no new gating state to build or keep in
sync with the v1 pipeline's own timing).

Fails loud: `guard_entrypoint()` propagates any raise (snapshot self-capture
failure, or any failure inside `eod_reconcile.run()` itself, which already
fails loud per its own module contract) to a non-zero systemd oneshot exit.
No silent swallow — this is the PRIMARY producer of the artifacts above, not
secondary observability hung off an already-committed run.

Schedule: `infrastructure/systemd/alpha-engine-eod-reconcile-standalone.timer`
— weekday 21:05 UTC, this module's own long-documented post-close slot (see
this file's own historical cron comment, mirrored from `eod_reconcile.py`'s
top-of-file docstring), 10 minutes ahead of
`alpha-engine-reference-rate-publish.timer`'s 21:15 UTC so that timer's
`nav_history` read of `trades/eod_pnl.csv` normally finds today's row
already written, and while IB Gateway is still up (stopped only by v1's
`StopTradingInstance` step during the coexistence window — same UNVERIFIED
live-timing caveat `I11007`'s registry row flagged; this timer is
self-sufficient either way, but a systemd timer cannot fire on a stopped
instance).
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.dates import now_dual
from nousergon_lib.logging import guard_entrypoint, setup_logging

from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)

_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging("eod-reconcile-standalone", flow_doctor_yaml=_FLOW_DOCTOR_YAML)
logger = logging.getLogger(__name__)

from executor import eod_reconcile, snapshot_capturer  # noqa: E402 -- must follow setup_logging above
from executor.config_loader import load_config  # noqa: E402


def run(run_date: str | None = None) -> None:
    """Run `eod_reconcile.run()` for `run_date` (default: today's trading
    day), self-capturing the EOD snapshot first if the v1 postclose
    pipeline's `CaptureSnapshot` step has not already written one.
    """
    if run_date is None:
        run_date = now_dual().trading_day
        logger.info("EOD reconcile (standalone) | run_date=%s (resolved from now_dual().trading_day)", run_date)
    else:
        logger.info("EOD reconcile (standalone) | run_date=%s (explicit)", run_date)

    config = load_config()
    bucket = config["trades_bucket"]

    snapshot = snapshot_capturer.load_snapshot(bucket=bucket, run_date=run_date, region=config.get("aws_region", "us-east-1"))
    if snapshot is None:
        logger.info(
            "No snapshot at s3://%s/trades/snapshots/%s.json yet — capturing one directly "
            "(this schedule does not depend on the v1 postclose pipeline's CaptureSnapshot step)",
            bucket, run_date,
        )
        snapshot_capturer.run(run_date)
        # eod_reconcile.run() re-reads the snapshot itself below; this call's
        # only job is to make sure one exists before that read. No local
        # existence re-check here — eod_reconcile.run()'s own snapshot load
        # already raises loud (with the same irreversibility framing,
        # alpha-engine-config-I5569/I6705) if it is somehow still missing.

    eod_reconcile.run(run_date=run_date, send_email=True, run_audit=True)
    logger.info("EOD reconcile (standalone) complete | run_date=%s", run_date)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run EOD reconciliation on component 3's own schedule, "
            "independent of the v1 postclose pipeline (alpha-engine-config-I11066)."
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
