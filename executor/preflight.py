"""
Executor preflight: connectivity + safety checks run at the top of each
entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into a mode-specific sequence. See the
alpha-engine-lib README for the rationale.

Modes:

- ``"main"`` — ``executor/main.py``, the morning order-book planner.
  Reads per-ticker OHLCV for ATR sizing + macro/SPY for alpha context.
  Both ArcticDB libraries must be readable + fresh; per-ticker
  freshness scan catches the partial-write class (2026-04-21 ASGN/MOH)
  that single-symbol checks miss.
- ``"daemon"`` — ``executor/daemon.py``, the sole order executor. Same
  ArcticDB freshness gates as ``main`` plus the IB paper-account guard
  is invoked separately by the daemon after IBKRClient connects.
- ``"eod"`` — ``executor/eod_reconcile.py``. macro/SPY only — eod
  computes alpha vs SPY + reads held-ticker closes; full-universe scan
  is overkill since only the ~20 held names matter.
"""

from __future__ import annotations

from alpha_engine_lib.preflight import BasePreflight

# ArcticDB freshness thresholds for executor mode runs.
# 4 days: covers Fri→Tue long weekends (US market holidays) + 1 day
# buffer. Matches alpha-engine-data DataPreflight daily-mode thresholds.
_MACRO_MAX_STALE_DAYS = 4
_UNIVERSE_SYMBOL_MAX_STALE_DAYS = 4
# 5 days: per-ticker scan threshold. Slightly more permissive than the
# canonical-symbol check because individual tickers can legitimately
# trail SPY by 1 day (DST/cross-listing edge cases). Backtester uses
# the same 5d default for the same reason.
_UNIVERSE_PER_TICKER_MAX_STALE_DAYS = 5


class ExecutorPreflight(BasePreflight):
    """Preflight checks for the three executor entrypoints."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("main", "daemon", "eod"):
            raise ValueError(f"ExecutorPreflight: unknown mode {mode!r}")
        self.mode = mode

    def run(self) -> None:
        # Cheap-first ordering: env (~ms) → S3 HEAD (~ms) → ArcticDB
        # canonical liveness (~100ms) → universe-wide scan (~5-10s).
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()

        if self.mode in ("main", "daemon"):
            # Canonical macro liveness (existing-class check) + matching
            # universe-side liveness probe. The morning planner + daemon
            # both read per-ticker OHLCV from `universe`; macro/SPY
            # reports DataPhase1 health, universe/SPY reports daily_append
            # health on the universe library.
            self.check_arcticdb_fresh(
                "macro", "SPY", max_stale_days=_MACRO_MAX_STALE_DAYS,
            )
            self.check_arcticdb_fresh(
                "universe", "SPY",
                max_stale_days=_UNIVERSE_SYMBOL_MAX_STALE_DAYS,
            )
            # Per-ticker freshness scan — catches the partial-write class
            # (2026-04-21 ASGN/MOH) where macro.SPY + universe.SPY both
            # stay fresh while individual tickers stop receiving writes.
            # Without this, executor's load_atr_14_pct guard aborts deep
            # in the morning run.
            self.check_arcticdb_universe_fresh(
                "universe",
                max_stale_days=_UNIVERSE_PER_TICKER_MAX_STALE_DAYS,
            )

        if self.mode == "eod":
            # EOD reconcile reads macro.SPY for alpha-vs-SPY computation
            # and per-position closes from universe — but only the ~20
            # held tickers matter, not the full ~900 universe. Single-
            # symbol macro check is sufficient; per-ticker validation
            # happens at the held-position read site.
            self.check_arcticdb_fresh(
                "macro", "SPY", max_stale_days=_MACRO_MAX_STALE_DAYS,
            )
