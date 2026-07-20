"""
Open-IB-orders snapshot writer for the order-book rationale reconciliation.

Publishes a single artifact from the daemon's IB order state to S3 each
poll tick:

- ``s3://{bucket}/trades/open_orders/latest.json`` — current open-order
  state at IB: per-order ticker, action, shares, remaining, order type,
  limit/aux price, status, and IB order IDs.

The dashboard's order-book reconciliation view (page 16) reads this
artifact to render the "Working $" column alongside the existing
"Planned $" — answering the operator question *"is the daemon
following through on the optimizer's plan?"*. Planned $ comes from
the morning-planner shadow log; Working $ from this artifact;
together they expose the gap between intent and live order state.

**Failure semantics.** Mirrors ``executor/intraday_snapshot.py``:
S3 writes are fire-and-forget — failures log at WARNING and never
raise. This is observability hung off the daemon's primary order-
execution loop; the primary path records its own failures via
``trade_logger`` SQLite + Telegram, so a stale snapshot here cannot
mask a real execution failure. Stale snapshots also surface to the
surveillance Lambda via daemon heartbeat staleness ([[reference: the
existing IntradaySnapshotWriter design]]).
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# Single rolling snapshot — operators want "what's open right now," not
# a per-tick archive. trades.db carries the full intraday event log for
# forensics; this artifact is a flat read for the dashboard.
OPEN_ORDERS_LATEST_KEY = "trades/open_orders/latest.json"

# IB order statuses that count as "working" (capital at risk in an open
# order). Sourced from ib_insync's OrderStatus.Statuses — the active set
# (orders the broker has accepted and is working) is everything except
# the terminal states. We enumerate explicitly so a new IB status value
# doesn't get silently re-classified.
_WORKING_STATUSES = frozenset({
    "PendingSubmit",
    "PendingCancel",
    "PreSubmitted",
    "Submitted",
    "ApiPending",
})


def _coerce_float(value: Any) -> float | None:
    """Return float for finite numeric inputs, else None.

    IB sometimes returns 0.0 or NaN for unused price fields (lmtPrice
    on a market order, auxPrice on a limit). Coerce defensively so the
    JSON payload is clean for the dashboard.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _trade_to_record(trade: Any) -> dict[str, Any] | None:
    """Project an ib_insync ``Trade`` (or duck-type) to a JSON-safe record.

    Returns None if the trade lacks the minimum identifying fields
    (ticker, action, totalQuantity) — defensive against partial Trade
    objects that show up briefly during order-edit races.
    """
    contract = getattr(trade, "contract", None)
    order = getattr(trade, "order", None)
    status_obj = getattr(trade, "orderStatus", None)
    if contract is None or order is None:
        return None
    ticker = getattr(contract, "symbol", None)
    action = getattr(order, "action", None)
    total_qty = getattr(order, "totalQuantity", None)
    if not ticker or not action or total_qty is None:
        return None

    try:
        total_shares = int(total_qty)
    except (TypeError, ValueError):
        return None
    filled = 0
    if status_obj is not None:
        try:
            filled = int(getattr(status_obj, "filled", 0) or 0)
        except (TypeError, ValueError):
            filled = 0
    remaining = max(total_shares - filled, 0)
    status = getattr(status_obj, "status", "") if status_obj is not None else ""

    return {
        "ticker": ticker,
        "action": action,
        "shares": total_shares,
        "filled": filled,
        "remaining": remaining,
        "order_type": getattr(order, "orderType", None),
        "limit_price": _coerce_float(getattr(order, "lmtPrice", None)),
        "aux_price": _coerce_float(getattr(order, "auxPrice", None)),
        "status": status,
        "ib_order_id": getattr(order, "orderId", None),
        "parent_id": getattr(order, "parentId", None) or None,
        "is_working": status in _WORKING_STATUSES,
    }


def build_open_orders_snapshot(
    open_trades: Iterable[Any],
    *,
    calendar_date: str,
    trading_day: str,
    written_at_utc: str | None = None,
    daemon_pid: int | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable open-orders snapshot payload.

    Pure function — no I/O, no clock reads when ``written_at_utc`` is
    injected. Tests pass IB-Trade duck types; production passes
    ``ibkr.ib.openTrades()`` directly.
    """
    ts = written_at_utc
    if ts is None:
        ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    records: list[dict[str, Any]] = []
    for t in open_trades:
        rec = _trade_to_record(t)
        if rec is not None:
            records.append(rec)
    # Stable sort: working orders first (those carry capital), then by
    # ticker for predictable display. The dashboard re-sorts by impact
    # anyway, but this keeps the artifact diff-friendly across ticks.
    records.sort(key=lambda r: (not r["is_working"], r["ticker"]))

    return {
        "written_at_utc": ts,
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "daemon_pid": daemon_pid if daemon_pid is not None else os.getpid(),
        "n_open_orders": len(records),
        "n_working": sum(1 for r in records if r["is_working"]),
        "open_orders": records,
    }


class OpenOrdersSnapshotWriter:
    """Writes the daemon's current open-IB-order state to S3 each tick.

    Fire-and-forget — S3 write failures log at WARNING and never raise.
    Mirrors ``IntradaySnapshotWriter`` to keep the daemon's order-loop
    observability surfaces uniform.
    """

    def __init__(
        self,
        bucket: str,
        *,
        daemon_pid: int | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._daemon_pid = daemon_pid if daemon_pid is not None else os.getpid()
        self._s3 = s3_client if s3_client is not None else boto3.client("s3")

    def write(
        self,
        open_trades: Iterable[Any],
        *,
        calendar_date: str,
        trading_day: str,
    ) -> bool:
        """Publish the snapshot to S3. Returns True on success, False on
        any boto error (logged).
        """
        payload = build_open_orders_snapshot(
            open_trades,
            calendar_date=calendar_date,
            trading_day=trading_day,
            daemon_pid=self._daemon_pid,
        )
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=OPEN_ORDERS_LATEST_KEY,
                Body=json.dumps(payload, default=str).encode("utf-8"),
                ContentType="application/json",
            )
            return True
        except (ClientError, BotoCoreError) as e:
            # Acceptable swallow per CLAUDE.md "secondary observability"
            # carve-out: the daemon's primary execution path records its
            # own failures via trade_logger SQLite + Telegram. A stale
            # snapshot surfaces to the surveillance Lambda via daemon
            # heartbeat staleness — there is no silent-failure mode.
            logger.warning(
                "open-orders snapshot write to s3://%s/%s failed (%s)",
                self._bucket, OPEN_ORDERS_LATEST_KEY, type(e).__name__,
            )
            return False
