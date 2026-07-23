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
the union of actionable (``ENTER``/``EXIT``/``REDUCE``) tickers from
``signals.signals`` ∪ ``signals.buy_candidates`` ∪ ``current_positions``
that both producers (the daemon publishing snapshots) and consumers (the
surveillance Lambda) compute identically from canonical artifacts. Universe
consistency is enforced by construction, not by discipline — there is no
``watchlist.yaml`` to drift. ``HOLD`` entries (the bulk of the ~900-ticker
weekly-scan population) are excluded — see :func:`compute_surveillance_universe`
for the incident that made this filter load-bearing.

**Failure semantics.** S3 writes are fire-and-forget — failures are logged
at WARNING and never raise. A failed snapshot write must never interrupt
the daemon's order-execution loop. Stale snapshots are visible to the
surveillance Lambda via heartbeat-timestamp staleness, which is itself the
designed alert signal.

ROADMAP L1067 PR 2b. Composes with ``nousergon_lib.telegram`` (lib v0.14.0,
PR 1) and ``executor/notifier.py`` migration (PR 2a, merged).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

LATEST_PRICES_KEY = "intraday/latest_prices.json"
HEARTBEAT_KEY = "intraday/heartbeat.json"
NAV_KEY = "intraday/nav.json"
NAV_SERIES_PREFIX = "intraday/nav_series/"

# Defensive cap on per-day series length. At a 60s poll the daemon writes
# ~390 points across a session; 2000 leaves generous headroom for a faster
# poll interval while bounding the read-modify-write object size.
_MAX_SERIES_POINTS = 2000
_ET = ZoneInfo("America/New_York")
_NYSE_CLOSE_ET = time(16, 0)


def _log_nav_series_session_refusal(trading_day: str, axis_err: ValueError, now_utc: datetime) -> None:
    """Log a refused nav_series point at the right severity.

    Post-close wind-down (daemon still polling with a frozen run_date while
    ``session_date`` has rolled to the next session, but still on the labeled
    session's calendar day) is expected — INFO only. A stale-label mis-key
    (e.g. D-1 run_date during a live session) stays ERROR.
    """
    try:
        from nousergon_lib.dates import session_date

        labeled = date.fromisoformat(trading_day)
        actual = session_date(now_utc)
        now_et = now_utc.astimezone(_ET)
        post_close_wind_down = actual != labeled and now_et.date() == labeled and now_et.time() > _NYSE_CLOSE_ET
    except (ImportError, ValueError):
        post_close_wind_down = False

    if post_close_wind_down:
        logger.info("nav_series point skipped (post-close) — %s", axis_err)
    else:
        logger.error("nav_series point refused — %s", axis_err)


def _put_json_to_s3(s3: Any, bucket: str, key: str, payload: dict) -> bool:
    """Write a JSON dict to S3 as a single object. Fire-and-forget.

    Shared by all intraday writers: S3 errors are logged at WARNING and
    swallowed (return ``False``) so a failed observability write never
    interrupts the daemon's order-execution loop. The recording surface
    for a sustained failure is heartbeat-timestamp staleness, which the
    surveillance Lambda treats as a first-class daemon-down alert.
    """
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except (ClientError, BotoCoreError) as e:
        logger.warning(
            "intraday write to s3://%s/%s failed (%s) — surveillance Lambda will see heartbeat staleness",
            bucket,
            key,
            type(e).__name__,
        )
        return False


def _get_json_from_s3(s3: Any, bucket: str, key: str) -> tuple[dict | None, str]:
    """GET + parse a JSON object. Returns ``(payload, status)``.

    ``status`` is one of:
    - ``"ok"`` — parsed dict returned;
    - ``"missing"`` — object absent (NoSuchKey/404), an expected state
      (e.g. the first tick of a trading day);
    - ``"error"`` — a transient failure; callers doing read-modify-write
      MUST NOT treat this as "empty" and clobber the existing object.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read()), "ok"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None, "missing"
        logger.warning("intraday read s3://%s/%s failed (%s)", bucket, key, code)
        return None, "error"
    except (BotoCoreError, ValueError) as e:
        logger.warning("intraday read s3://%s/%s failed (%s)", bucket, key, type(e).__name__)
        return None, "error"


_ACTIONABLE_SIGNALS = frozenset({"ENTER", "EXIT", "REDUCE"})


def compute_surveillance_universe(
    signals: dict | None,
    order_book_tickers: list[str] | None = None,
    current_positions: list[str] | None = None,
    *,
    include_spy: bool = True,
) -> list[str]:
    """Compute the union of tickers the surveillance layer should watch.

    Universe = actionable entries of ``signals.signals`` (``ENTER``/``EXIT``/
    ``REDUCE`` only — ``HOLD`` is explicitly non-action, per the
    ``signals.json`` contract's ``signal`` field) ∪ ``signals.buy_candidates``
    ∪ ``order_book_tickers`` ∪ ``current_positions``. Sorted deterministic
    output. Pre-population scanner output (the ~900-ticker weekly-scan
    population, virtually all ``HOLD``) is deliberately excluded — non-action
    there isn't a surveillance signal, and requesting live IB market-data
    lines for the full population blows through the account's concurrent
    market-data-line cap (incident 2026-07-20/21, IBKR error 101 "Max number
    of tickers has been reached" cascading across every alphabetically-later
    symbol once the cap was hit).

    :param signals: The ``signals.json`` payload as a dict (or None on read
        failure). Keys read: ``universe`` (dict of ticker → rec, each with a
        ``signal`` field) and ``buy_candidates`` (list of ticker strings,
        already actionable by construction — included unconditionally).
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
        signals_list = signals.get("universe")
        if isinstance(signals_list, list):
            universe.update(
                rec.get("ticker")
                for rec in signals_list
                if isinstance(rec, dict) and rec.get("signal") in _ACTIONABLE_SIGNALS and rec.get("ticker")
            )
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
        return _put_json_to_s3(self._s3, self._bucket, key, payload)


class IntradayNavWriter:
    """Publishes a live portfolio NAV snapshot to ``intraday/nav.json``.

    Powers the live.nousergon.ai intraday header (current NAV, today's
    return, today's alpha vs SPY). The daemon already holds an IB
    account-summary subscription, so ``NetLiquidation`` is IB's own
    real-time, fill-inclusive ground truth — more accurate than marking
    a stale position list dashboard-side.

    Producer publishes RAW marks only (NAV, cash, SPY last); the consumer
    derives today's return/alpha against the prior EOD baseline it already
    loads from ``eod_pnl.csv``. Keeping the math consumer-side means the
    display convention can change without redeploying the trading box.

    Fire-and-forget — S3 failures are logged at WARNING and never raise.
    A failed NAV write must never interrupt the order-execution loop;
    sustained failure surfaces via heartbeat staleness like the price
    writer.
    """

    def __init__(
        self,
        bucket: str,
        *,
        s3_client: Any | None = None,
    ) -> None:
        """:param bucket: S3 bucket to write under (the daemon's
            ``signals_bucket``).
        :param s3_client: Inject a pre-built boto3 client (tests). Defaults
            to a freshly-constructed ``boto3.client("s3")``.
        """
        self._bucket = bucket
        self._s3 = s3_client if s3_client is not None else boto3.client("s3")

    def write(
        self,
        account_snapshot: dict,
        *,
        spy_last: float | None,
        ib_connected: bool,
    ) -> bool:
        """Publish the NAV snapshot artifact to S3.

        :param account_snapshot: The dict returned by
            ``ibkr.get_account_snapshot()`` (``net_liquidation``,
            ``total_cash``, ``gross_position_value``, ``unrealized_pnl``,
            etc.). Missing fields are tolerated — published as ``None``.
        :param spy_last: Current SPY last price from the daemon's price
            monitor (``monitor.prices["SPY"]["last"]``), or ``None`` if SPY
            has no live tick yet. The consumer needs it to compute today's
            alpha vs SPY.
        :param ib_connected: ``ibkr.ib.isConnected()`` — stamped so the
            consumer can distinguish a fresh-but-disconnected snapshot from
            a stale one.
        :returns: ``True`` on successful write, ``False`` otherwise
            (logged).
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ib_connected": ib_connected,
            "net_liquidation": account_snapshot.get("net_liquidation"),
            "total_cash": account_snapshot.get("total_cash"),
            "gross_position_value": account_snapshot.get("gross_position_value"),
            "unrealized_pnl": account_snapshot.get("unrealized_pnl"),
            "spy_last": spy_last,
        }
        return _put_json_to_s3(self._s3, self._bucket, NAV_KEY, payload)


class IntradayNavSeriesWriter:
    """Appends a per-tick (NAV, SPY) point to a per-day intraday series.

    Powers the intraday portfolio-vs-SPY curve on live.nousergon.ai. Unlike
    ``IntradayNavWriter`` (a single overwritten snapshot for the live
    header numbers), this accumulates the day's points into
    ``intraday/nav_series/{trading_day}.json`` so the consumer can draw a
    line, not just a number.

    **Single-writer read-modify-write.** The daemon is the only writer of
    this object and its poll loop is single-threaded, so RMW is race-free.
    A transient READ error skips the append rather than clobbering the
    day's history with a fresh list; a missing object (first tick of the
    day) starts a new series. Fire-and-forget on write like the other
    intraday writers — a failed point is a dropped chart sample, never an
    interruption to the order loop.
    """

    def __init__(
        self,
        bucket: str,
        *,
        s3_client: Any | None = None,
        max_points: int = _MAX_SERIES_POINTS,
    ) -> None:
        self._bucket = bucket
        self._s3 = s3_client if s3_client is not None else boto3.client("s3")
        self._max_points = max_points

    @staticmethod
    def key_for(trading_day: str) -> str:
        return f"{NAV_SERIES_PREFIX}{trading_day}.json"

    def write(
        self,
        trading_day: str,
        account_snapshot: dict,
        *,
        spy_last: float | None,
    ) -> bool:
        """Append today's current (NAV, SPY) point to the per-day series.

        :param trading_day: The series key date (the daemon's run_date).
        :param account_snapshot: ``ibkr.get_account_snapshot()`` — only
            ``net_liquidation`` is charted; a point with no NAV is skipped.
        :param spy_last: SPY mark from the price monitor (may be ``None``
            before SPY has a live tick; stored as-is for the consumer).
        :returns: ``True`` on a successful append+write, ``False`` if the
            point was skipped (no NAV / transient read error) or the write
            failed.
        """
        nav = account_snapshot.get("net_liquidation")
        if nav is None:
            return False

        # Content-vs-key guard (config#1610): a NAV point ticks during the
        # session labeled on the file — a point timestamped in a different
        # session than its key is the exact mislabel that mis-joined the EOD
        # reconcile (nav_series/2026-07-01.json full of 07-02 timestamps).
        # Refuse the mis-keyed point (skipped write; severity depends on
        # whether this is post-close wind-down vs a true stale-label bug)
        # rather than raising into the order loop's fire-and-forget except.
        now_utc = datetime.now(UTC)
        try:
            from nousergon_lib.dates import assert_within_session

            assert_within_session(now_utc, trading_day)
        except ValueError as _axis_err:
            _log_nav_series_session_refusal(trading_day, _axis_err, now_utc)
            return False
        except ImportError:
            pass  # lib not yet bumped on this deploy — guard is best-effort

        key = self.key_for(trading_day)
        existing, status = _get_json_from_s3(self._s3, self._bucket, key)
        if status == "error":
            # Don't clobber the day's history on a transient read failure.
            return False

        points: list = []
        if status == "ok" and isinstance(existing, dict):
            prior = existing.get("points")
            if isinstance(prior, list):
                points = prior

        now_iso = now_utc.replace(tzinfo=None).isoformat() + "Z"
        points.append({"t": now_iso, "nav": nav, "spy": spy_last})
        if len(points) > self._max_points:
            points = points[-self._max_points :]

        payload = {
            # Historical field name; holds the SESSION the curve belongs to
            # (session axis, config#1610). `session_date` states it
            # explicitly; `trading_day` is retained for consumers
            # (additive-only S3 contract).
            "trading_day": trading_day,
            "session_date": trading_day,
            "updated_at": now_iso,
            "points": points,
        }
        return _put_json_to_s3(self._s3, self._bucket, key, payload)
