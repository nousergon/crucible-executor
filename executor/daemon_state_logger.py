"""
Daemon decision-state capture for intraday replay parity (ROADMAP L139a).

The morning-planner stage already replays cleanly through
``alpha-engine-backtester/tests/test_parity_replay.py`` — but the
backtester has no visibility into intraday daemon decisions (urgent
exits, intraday trigger fires, ATR-trail exits, profit-take REDUCEs,
time-decay exits, COVER fills). The test_parity_replay docstring
explicitly carves these out:

  "Backtester sim runs the morning planner only — none of these exist
   at planner stage, so they're null in replay output. Including them
   as required-match fields produced spurious divergence on every
   cohort-matched ENTER (observed 2026-04-26 ROST 4/12 trade). Phase 2
   (entry_triggers.py daily-bar port) will eventually populate them in
   sim, but until then they're a pure noise floor."

This module is the substrate that closes that gap. Every intraday
decision point in ``executor/daemon.py`` calls ``record_decision()`` with
the live state at the time of the call; on daemon shutdown,
``flush_to_s3()`` writes the buffered events to
``s3://{bucket}/daemon_state/{trading_day}/intraday_decisions.jsonl``.

Backtester consumer (L139b, separate PR) reads that artifact, replays
the intraday decision rules deterministically, and the parity gate
turns from "PARITY IS OBSERVABILITY NOT A GATE" into a real gate at
predictor weight promotion. Until L139b lands, this artifact is
observability-only.

Schema — one JSON object per line:

    {
        "timestamp_utc": "2026-05-22T15:42:01.123456Z",
        "trading_day": "2026-05-22",
        "decision_type": "urgent_exit" | "intraday_exit" | "entry_trigger"
                         | "cover" | "phase0_auto_cover",
        "ticker": "AAPL",
        "action": "EXIT" | "REDUCE" | "ENTER" | "COVER",
        "shares": 100,
        "trigger_reason": "atr_trail" | "vwap_pullback" | "research_signal"
                          | "time_decay" | "profit_take" | ...,
        "trigger_price": 150.0,
        "signal_price": 149.50,    # daemon's snapshot at decision time
        "fill_price": 150.25,      # IB fill (None if order failed)
        "ib_order_id": 287,
        "retry_count": 0,
        "attempts": [...],         # from _place_order_with_retry (L133)
        "context": {...},          # decision-type-specific extras
    }

The artifact is **append-only per trading day** — each daemon run
on a given trading_day appends to the existing JSONL via S3 read +
re-upload. The 2026-05-22 EOD-SF can re-trigger the daemon mid-day for
fix-and-rerun cycles; we want every decision recorded, not just the
last invocation's.

Best-effort observability — recording or flushing failures DO NOT
block the daemon's primary order-execution path (`[[feedback_no_silent_fails]]`
secondary-observability clause). WARN logs surface the failure;
the loss is one cycle of replay coverage, not a stuck order.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_JSONL_PREFIX = "daemon_state"
_JSONL_FILENAME = "intraday_decisions.jsonl"


class DaemonDecisionLogger:
    """In-memory buffer of intraday decisions, flushed to S3 on shutdown.

    Thread-safe — the daemon's monitor + main loop both call into
    decision recording paths. Append-only buffer; no eviction.

    Singleton-ish — `get_logger()` returns the module-level instance.
    Tests construct their own instances via the class constructor to
    avoid leaking buffer state across runs.
    """

    def __init__(self) -> None:
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        decision_type: str,
        ticker: str,
        action: str | None,
        trading_day: str,
        **context: Any,
    ) -> None:
        """Record one decision. Never raises — record failures swallow.

        Required positional-keyword args lock down the schema's
        required fields; ``**context`` admits decision-type-specific
        extras (e.g. ``trigger_reason``, ``fill_price``, ``attempts``).
        """
        try:
            entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "trading_day": trading_day,
                "decision_type": decision_type,
                "ticker": ticker,
                "action": action,
                **context,
            }
            with self._lock:
                self._buffer.append(entry)
        except Exception as exc:  # noqa: BLE001 — secondary observability; primary path unaffected
            logger.warning(
                "DaemonDecisionLogger.record swallowed exception (decision_type=%s, ticker=%s): %s",
                decision_type, ticker, exc,
            )

    def snapshot(self) -> list[dict]:
        """Return a defensive copy of the current buffer."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush_to_s3(self, bucket: str, trading_day: str, s3_client=None) -> bool:
        """Write the buffered decisions to S3 as JSONL. Append semantics
        per the module docstring — fix-and-rerun cycles within a single
        trading_day must preserve prior captures rather than overwriting.

        Returns True on successful write (or no-op on empty buffer);
        False on any failure. Never raises — secondary observability.
        """
        with self._lock:
            entries = list(self._buffer)
        if not entries:
            return True
        try:
            import boto3
            s3 = s3_client or boto3.client("s3")
            key = f"{_JSONL_PREFIX}/{trading_day}/{_JSONL_FILENAME}"

            # Append semantics: read prior JSONL (or empty on miss),
            # concat new entries, re-upload. Prior decisions land first
            # in line order. For huge runs we could switch to multipart
            # append via S3 ListObjects + sequential parts, but a typical
            # daemon run is <100 decisions so single-blob is fine.
            existing_body = b""
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                existing_body = obj["Body"].read()
            except Exception:  # noqa: BLE001 — NoSuchKey or other; treat as first write
                existing_body = b""

            new_body = b"".join(
                (json.dumps(e, default=str) + "\n").encode("utf-8") for e in entries
            )
            combined = existing_body + new_body

            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=combined,
                ContentType="application/x-ndjson",
            )
            logger.info(
                "daemon_state: flushed %d intraday decisions to s3://%s/%s "
                "(combined size %d bytes)",
                len(entries), bucket, key, len(combined),
            )
            return True
        except Exception as exc:  # noqa: BLE001 — secondary observability
            logger.warning(
                "daemon_state: flush_to_s3 failed (trading_day=%s, n=%d): %s — "
                "intraday replay coverage lost for this cycle, but primary "
                "order-execution path unaffected",
                trading_day, len(entries), exc,
            )
            return False


_singleton: DaemonDecisionLogger | None = None
_singleton_lock = threading.Lock()


def get_logger() -> DaemonDecisionLogger:
    """Return the module-level singleton logger.

    Tests should NOT use this — they construct their own instance via
    ``DaemonDecisionLogger()`` to keep buffers isolated.
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = DaemonDecisionLogger()
        return _singleton


def reset_singleton_for_tests() -> None:
    """Reset the module singleton. Test-only — never call in production."""
    global _singleton
    with _singleton_lock:
        _singleton = None
