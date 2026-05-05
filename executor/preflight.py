"""
Executor preflight: connectivity + safety checks run at the top of each
entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into a mode-specific sequence. See the
alpha-engine-lib README for the rationale.

Data-freshness assertions (universe + macro/SPY) live upstream in
``alpha-engine-data``'s preflight, which runs before ``RunMorningPlanner``
+ ``RunDaemon`` + EOD steps in the weekday + EOD Step Functions. If
upstream data is stale, the data step hard-fails and the SF never reaches
the executor — re-checking here was redundant.

Modes:

- ``"main"`` — ``executor/main.py``, the morning order-book planner.
- ``"daemon"`` — ``executor/daemon.py``, the sole order executor. The IB
  paper-account guard is invoked separately by the daemon after IBKRClient
  connects.
- ``"eod"`` — ``executor/eod_reconcile.py``.
"""

from __future__ import annotations

from alpha_engine_lib.preflight import BasePreflight


class ExecutorPreflight(BasePreflight):
    """Preflight checks for the three executor entrypoints."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("main", "daemon", "eod"):
            raise ValueError(f"ExecutorPreflight: unknown mode {mode!r}")
        self.mode = mode

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
