"""
Intraday S3 snapshot writer for the executor surveillance arc.

Publishes two artifacts from the daemon's IB market-data subscription set to
S3 on every poll tick during market hours:

- ``s3://{bucket}/intraday/latest_prices.json`` — current price state for
  every ticker the daemon has live IB data for.
- ``s3://{bucket}/intraday/heartbeat.json`` — daemon liveness signal:
  timestamp, ``ib_connected``, daemon_pid, subscribed-ticker count. The
  surveillance Lambda (PR 3, alpha-engine-research) treats staleness of
  this file as a first-class daemon-down alert.

**Surveillance universe.** :func:`compute_surveillance_universe` returns
the union ``signals.signals ∪ signals.buy_candidates ∪ current_positions``
that both producers (the daemon publishing snapshots) and consumers (the
surveillance Lambda) compute identically from canonical artifacts. Universe
consistency is enforced by construction, not by discipline — there is no
``watchlist.yaml`` to drift.

**Failure semantics.** S3 writes are fire-and-forget — failures are logged
at WARNING and never raise. A failed snapshot write must never interrupt
the daemon's order-execution loop. Stale snapshots are visible to the
surveillance Lambda via heartbeat-timestamp staleness, which is itself the
designed alert signal.

ROADMAP L1067 PR 2b. Composes with ``alpha_engine_lib.telegram`` (lib v0.14.0,
PR 1) and ``executor/notifier.py`` migration (PR 2a, merged).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

LATEST_PRICES_KEY = "intraday/latest_prices.json"
HEARTBEAT_KEY = "intraday/heartbeat.json"


def compute_surveillance_universe(
    signals: dict | None,
    order_book_tickers: list[str] | None = None,
    current_positions: list[str] | None = None,
    *,
    include_spy: bool = True,
) -> list[str]:
    """Compute the union of tickers the surveillance layer should watch.

    Universe = ``signals.signals.keys() ∪ signals.buy_candidates ∪
    order_book_tickers ∪ current_positions``. Sorted deterministic output.
    Pre-population scanner output deliberately excluded — non-action there
    isn't a surveillance signal.

    :param signals: The ``signals.json`` payload as a dict (or None on read
        failure). Keys read: ``signals`` (dict of ticker → rec) and
        ``buy_candidates`` (list of ticker strings).
    :param order_book_tickers: Tickers the daemon's order book is tracking
        (held positions + active candidates). Optional; defaults to empty.
    :param current_positions: Current IB position tickers. Optional;
        defaults to empty. Belt-and-braces — positions are typically already
        in ``order_book_tickers`` via the morning planner.
    :param include_spy: If ``True`` (default), SPY is always included for
        roundtrip benchmarking — matches existing daemon behavior.
    :returns: Sorted deduplicated list of tickers.
    """
    universe: set[str] = set()

    if signals:
        signals_map = signals.get("signals")
        if isinstance(signals_map, dict):
            universe.update(signals_map.keys())
        buy_cands = signals.get("buy_candidates")
        if isinstance(buy_cands, list):
            universe.update(t for t in buy_cands if isinstance(t, str))

    if order_book_tickers:
        universe.update(order_book_tickers)

    if current_positions:
        universe.update(current_positions)

    if include_spy:
        universe.add("SPY")

    universe.discard("")
    return sorted(universe)


class IntradaySnapshotWriter:
    """Writes daemon IB price snapshots + heartbeats to S3 each poll tick.

    Fire-and-forget — S3 write failures are logged at WARNING and never
    raise. Surveillance Lambda treats heartbeat staleness as a daemon-down
    alert, so write failures naturally surface as surveillance signals
    rather than silently lost data.
    """

    def __init__(
        self,
        bucket: str,
        *,
        daemon_pid: int | None = None,
        s3_client: Any | None = None,
    ) -> None:
        """:param bucket: S3 bucket to write under (typically
            ``alpha-engine-research``).
        :param daemon_pid: Stamped into heartbeat for ops triage. Defaults
            to ``os.getpid()``.
        :param s3_client: Inject a pre-built boto3 client (tests). Defaults
            to a freshly-constructed ``boto3.client("s3")``.
        """
        self._bucket = bucket
        self._daemon_pid = daemon_pid if daemon_pid is not None else os.getpid()
        self._s3 = s3_client if s3_client is not None else boto3.client("s3")

    def write(
        self,
        prices: dict[str, dict],
        *,
        ib_connected: bool,
        subscribed_tickers: list[str],
    ) -> bool:
        """Publish latest_prices + heartbeat artifacts to S3.

        :param prices: Current price-state dict from ``PriceMonitor.prices``
            (or any mapping with the same shape).
        :param ib_connected: ``ibkr.ib.isConnected()`` — stamped on
            heartbeat so a stale-but-present heartbeat with
            ``ib_connected=False`` is distinguishable from a daemon-dead
            scenario.
        :param subscribed_tickers: The full surveillance universe the
            daemon is subscribed to. Stamped on heartbeat for coverage
            audit.
        :returns: ``True`` if both artifacts wrote successfully, ``False``
            otherwise (logged).
        """
        now = datetime.utcnow()
        timestamp_iso = now.isoformat() + "Z"

        prices_payload = {
            "timestamp": timestamp_iso,
            "prices": dict(prices),
        }
        heartbeat_payload = {
            "timestamp": timestamp_iso,
            "ib_connected": ib_connected,
            "daemon_pid": self._daemon_pid,
            "subscribed_tickers": list(subscribed_tickers),
            "subscribed_count": len(subscribed_tickers),
        }

        prices_ok = self._put_json(LATEST_PRICES_KEY, prices_payload)
        heartbeat_ok = self._put_json(HEARTBEAT_KEY, heartbeat_payload)
        return prices_ok and heartbeat_ok

    def _put_json(self, key: str, payload: dict) -> bool:
        """Write a JSON dict to S3 as a single object. Fire-and-forget."""
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(payload, default=str).encode("utf-8"),
                ContentType="application/json",
            )
            return True
        except (ClientError, BotoCoreError) as e:
            logger.warning(
                "intraday snapshot write to s3://%s/%s failed (%s) — "
                "surveillance Lambda will see heartbeat staleness",
                self._bucket, key, type(e).__name__,
            )
            return False
