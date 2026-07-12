"""
Alpha Engine Intraday Daemon — sole order executor during market hours.

Runs from ~6:45 AM to 4:00 PM ET on trading days. Uses 15-minute delayed
streaming data from IB Gateway (free, no subscription required).

Architecture:
  - Morning batch (main.py) writes the order book: approved entries, urgent
    exits/reduces, and active stop records. It places NO orders.
  - Daemon is the sole order executor:
    Phase 0: Execute urgent exits/reduces immediately (no trigger delay)
    Phase 1: Monitor entries for technical triggers (pullback, VWAP, support, expiry)
    Phase 2: Monitor stops for exit rules (trailing stop, profit-take, collapse)
  - All trades logged to trades.db with source="intraday_daemon"
  - Telegram notifications sent for each trade

Usage:
    python -m executor.daemon              # run until market close
    python -m executor.daemon --dry-run    # log triggers without placing orders

The daemon uses clientId=2 to avoid conflicts with the morning batch (clientId=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time as _time  # aliased to avoid shadowing by local 'time' variables
from datetime import date, datetime

import pytz
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.decision_capture import (
    DecisionCaptureWriteError,
    capture_entry_trigger,
    capture_exit_rule,
    is_decision_capture_enabled,
)
from executor.entry_triggers import EntryTriggerEngine
from executor.ibkr import IBKRClient
from executor.intraday_exit_manager import IntradayExitManager
from executor.intraday_resolve import (
    available_redeploy_cash,
    build_conviction_map,
    build_redeploy_entry,
    compute_drawdown_overlay,
    select_forced_exits,
    solve_redeploy,
)
from executor.intraday_snapshot import (
    IntradayNavSeriesWriter,
    IntradayNavWriter,
    IntradaySnapshotWriter,
    compute_surveillance_universe,
)
from executor.open_orders_artifact import OpenOrdersSnapshotWriter
from executor.daemon_state_logger import get_logger as _get_decision_logger
from executor.market_hours import is_market_hours
from executor.notifier import send_daemon_status, send_trade_alert
from executor.order_book import OrderBook, build_stop_record
from executor.price_monitor import PriceMonitor
from executor.polygon_price_monitor import make_price_monitor
from executor.strategies.config import load_strategy_config
from executor.trade_logger import (
    init_db, log_trade, get_unmatched_entry, log_risk_event,
    get_executed_entry_tickers,
)

from nousergon_lib.logging import setup_logging, guard_entrypoint
# See executor/main.py for the rationale on IB Error 10197 / 10349 suppression.
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
from executor.config_loader import get_flow_doctor_yaml_path  # noqa: E402 (must precede setup_logging)
_FLOW_DOCTOR_YAML = get_flow_doctor_yaml_path()  # experiment-package-first (config#1042)
setup_logging("daemon", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

# Terminology:
#   "status" — IB order execution status: "Filled", "Rejected", "Timeout", etc.
#   "signal" — trading action type from Research/strategy: "ENTER", "EXIT", "REDUCE", "COVER"

from executor.config_loader import get_config_path

# Order retry policy — applied uniformly to all order types (urgent exits, intraday exits, entries)
MAX_ORDER_RETRIES = 3
ORDER_RETRY_DELAYS = [0, 2, 5]  # seconds between attempts

# Market timing constants (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30

# US Eastern timezone — hoisted to module scope so helper functions like
# ``_execute_entry`` (decision-capture wiring, L2308) can read it without
# threading the tz through every call site.
_ET = pytz.timezone("US/Eastern")


def _within_resolve_window(cutoff_et: str) -> bool:
    """True if the current ET time is before the intraday re-solve cutoff.

    Past the cutoff, freed cash can't reliably fill before the 3:55 PM ET
    time-expiry, so we stop re-solving and let it carry to tomorrow's planner.
    """
    try:
        hh, mm = (int(x) for x in cutoff_et.split(":"))
    except (ValueError, AttributeError):
        hh, mm = 15, 30
    now = datetime.now(_ET)
    return (now.hour, now.minute) < (hh, mm)


def _load_optimizer_shadow_log(bucket: str) -> dict | None:
    """Load the morning optimizer shadow log (target weights + cached daily Σ
    + alpha_hat) for the intraday re-solve. Returns None on ANY failure — the
    caller treats a missing log as 'cannot redeploy' and leaves cash idle (the
    re-solve is load-bearing; it never silently no-ops as if it succeeded)."""
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(
            Bucket=bucket, Key="predictor/optimizer_shadow/latest.json",
        )
        return json.loads(obj["Body"].read())
    except Exception as e:
        logger.error(
            "Could not load optimizer shadow log for intraday re-solve "
            "(redeploy disabled, freed cash stays idle until next planner): %s",
            e,
        )
        return None


# Connection retry limits
MAX_RECONNECT_BACKOFF_SECS = 300
DEFAULT_CONNECT_BACKOFF_BASE = 30

# Exception types that indicate a dropped IB Gateway connection
try:
    from asyncio import IncompleteReadError, TimeoutError as AsyncTimeoutError
    asyncio_exceptions = (IncompleteReadError, AsyncTimeoutError)
except ImportError:
    asyncio_exceptions = ()

_shutdown_requested = False
_midday_backup_done = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (%s)", signum)
    _shutdown_requested = True


def _cleanup_connections(
    monitor: "PriceMonitor | None",
    ibkr: "IBKRClient | None",
) -> None:
    """Best-effort cleanup of IB connections."""
    if monitor:
        try:
            monitor.unsubscribe_all()
        except Exception:
            logger.debug("monitor.unsubscribe_all failed during cleanup", exc_info=True)
    if ibkr:
        try:
            ibkr.disconnect()
        except Exception:
            logger.debug("ibkr.disconnect failed during cleanup", exc_info=True)


def _reconnect(
    ibkr: IBKRClient,
    monitor: "PriceMonitor",
    order_book: "OrderBook",
    config: dict,
    client_id: int,
    max_reconnect_attempts: int = 10,
    backoff_base: int = 30,
) -> tuple[IBKRClient, PriceMonitor]:
    """Reconnect to IB Gateway after a connection drop.

    Returns (new_ibkr, new_monitor) tuple. Raises after max_reconnect_attempts.
    """
    _cleanup_connections(monitor, ibkr)

    send_daemon_status(
        "\u26a0\ufe0f *IB Gateway connection lost* — attempting reconnect..."
    )

    for attempt in range(1, max_reconnect_attempts + 1):
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown during reconnect")
        wait = min(backoff_base * attempt, MAX_RECONNECT_BACKOFF_SECS)
        logger.info("Reconnect attempt %d/%d — waiting %ds...", attempt, max_reconnect_attempts, wait)
        _time.sleep(wait)
        try:
            new_ibkr = IBKRClient(
                host=config["ibkr_host"],
                port=config["ibkr_port"],
                client_id=client_id,
                reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
            )
            new_monitor = make_price_monitor(new_ibkr.ib)
            new_monitor.subscribe(order_book.all_tickers())
            logger.info("Reconnected to IB Gateway successfully")
            send_daemon_status("\u2705 *IB Gateway reconnected*")
            return new_ibkr, new_monitor
        except Exception as e:
            logger.warning("Reconnect attempt %d/%d failed: %s", attempt, max_reconnect_attempts, e)
            if attempt == max_reconnect_attempts:
                send_daemon_status(
                    f"\u274c *IB Gateway reconnect failed after {max_reconnect_attempts} attempts* — daemon exiting"
                )
                raise


_allow_shorts: bool = False  # Set from config in run_daemon(); default: never short


def _validate_sell_shares(
    positions: dict,
    ticker: str,
    shares: int,
    action: str,
    context: str,
    pending_sell_shares: int = 0,
) -> int | None:
    """Validate and cap sell shares against held position minus in-flight sells.

    Returns adjusted share count, or None if the sell should be skipped
    (no capacity to sell — selling would go short).

    ``pending_sell_shares`` is the total un-filled SELL quantity already working
    at the broker for this ticker. Passing it in lets us cap against
    held - in-flight so a retry/duplicate can't cumulatively blow past the
    position. (PFE incident 2026-04-22: three duplicate SELL 77s each passed
    this check individually against held=155, summing to 231 → short 76.)

    When ``_allow_shorts`` is True (set via ``allow_shorts: true`` in
    risk.yaml), the guard is bypassed and sells are allowed to create
    short positions.
    """
    if _allow_shorts:
        return shares
    held = int(positions.get(ticker, {}).get("shares", 0))
    available = held - int(pending_sell_shares)
    if available <= 0:
        logger.warning(
            "SKIP %s %s %s: hold %d shares, in-flight SELL %d — no capacity",
            context, action, ticker, held, pending_sell_shares,
        )
        return None
    if shares > available:
        logger.warning(
            "CAPPING %s %s %s: requested %d but available %d (held %d minus in-flight SELL %d)",
            context, action, ticker, shares, available, held, pending_sell_shares,
        )
        return available
    return shares


def _validate_buy_not_duplicate(
    ibkr: IBKRClient,
    ticker: str,
    context: str,
) -> bool:
    """Broker-truth backstop against a crash-restart double BUY (config#2328).

    Symmetric to ``_validate_sell_shares``: before placing an ENTER, confirm
    the broker does not already show a position or a working BUY for this
    ticker. Returns True if it is safe to place the BUY, False to SKIP it.

    The daemon holds at most one position per ticker and its intraday entries
    are for currently-unheld names, so an existing position or working BUY at
    this point means the entry already executed (the exact double-buy this
    guards) — refuse the duplicate.

    FAIL-CLOSED: if broker state cannot be read (IB API error), SKIP the buy.
    A blocked entry is recoverable on the next tick / next planner run; a
    double position is real capital at risk. (Same asymmetry that makes the
    sell guard cap rather than trust the in-memory view.)
    """
    try:
        positions = ibkr.get_positions()
        working_buy = ibkr.get_open_buy_shares(ticker)
    except Exception as e:  # noqa: BLE001 — any broker read failure fails closed
        logger.error(
            "SKIP %s BUY %s: broker state unreadable (%s) — failing closed to "
            "avoid a possible double buy", context, ticker, e,
        )
        send_daemon_status(
            f"⚠️ *BUY {ticker} SKIPPED*: broker state unreadable — "
            f"failing closed (config#2328)"
        )
        return False
    held = int(positions.get(ticker, {}).get("shares", 0))
    if held > 0:
        logger.warning(
            "SKIP %s BUY %s: already hold %d shares — refusing duplicate entry",
            context, ticker, held,
        )
        send_daemon_status(
            f"⚠️ *BUY {ticker} SKIPPED*: already hold {held} shares "
            f"(dup-entry guard, config#2328)"
        )
        return False
    if working_buy > 0:
        logger.warning(
            "SKIP %s BUY %s: %d shares already working at broker — refusing "
            "duplicate entry", context, ticker, working_buy,
        )
        send_daemon_status(
            f"⚠️ *BUY {ticker} SKIPPED*: {working_buy} shares already "
            f"working (dup-entry guard, config#2328)"
        )
        return False
    return True


def _reconcile_executing_entries(
    ibkr: IBKRClient,
    order_book: OrderBook,
    dry_run: bool,
) -> None:
    """Resolve write-ahead ``executing`` entries left by a crash (config#2328).

    An entry in the ``executing`` state means the pre-crash daemon marked its
    intent, saved the book, then died at or after IB order placement without
    finalizing. Reconcile each against broker truth:

      * position exists OR a BUY is working  → the order landed; finalize it
        (``mark_entry_executed``) so it is never re-placed.
      * no position and no working order      → the order never reached IB;
        revert to ``pending`` so the legitimate entry re-drives through the
        guarded path (idempotent — the pre-BUY check re-verifies at placement).
      * broker state unreadable               → leave ``executing`` (fail-safe);
        it stays out of ``pending_entries`` so it cannot be re-bought, and
        surfaces to the operator / EOD reconcile.

    A no-op when there are no executing entries (the common case).
    """
    executing = order_book.executing_entries()
    if not executing:
        return
    if dry_run:
        logger.info(
            "[DRY RUN] %d executing entries found — leaving untouched", len(executing),
        )
        return
    try:
        positions = ibkr.get_positions()
    except Exception as e:  # noqa: BLE001 — fail-safe: cannot verify, touch nothing
        logger.error(
            "Startup reconcile: broker positions unreadable (%s) — leaving %d "
            "executing entries in-doubt (fail-safe, will not re-buy)",
            e, len(executing),
        )
        send_daemon_status(
            f"⚠️ *Crash-recovery: {len(executing)} in-doubt entries* — "
            f"broker unreadable, left blocked (config#2328)"
        )
        return
    resolved, reverted, blocked = [], [], []
    for entry in executing:
        ticker = entry["ticker"]
        held = int(positions.get(ticker, {}).get("shares", 0))
        try:
            working_buy = ibkr.get_open_buy_shares(ticker)
        except Exception as e:  # noqa: BLE001 — per-ticker read failure fails safe
            logger.error(
                "Startup reconcile %s: open-order read failed (%s) — leaving "
                "in-doubt", ticker, e,
            )
            blocked.append(ticker)
            continue
        if held > 0 or working_buy > 0:
            order_book.mark_entry_executed(
                ticker, entry.get("trigger_reason", "crash_recovery"),
            )
            resolved.append(ticker)
        else:
            order_book.revert_entry_to_pending(ticker)
            reverted.append(ticker)
    order_book.save()
    logger.warning(
        "Startup reconcile of executing WAL: finalized %s, reverted-to-pending "
        "%s, left-in-doubt %s", resolved or "[]", reverted or "[]", blocked or "[]",
    )
    send_daemon_status(
        f"\U0001f501 *Crash-recovery reconcile (config#2328)*\n"
        f"Finalized (already at broker): {', '.join(resolved) or 'none'}\n"
        f"Reverted to pending (never placed): {', '.join(reverted) or 'none'}\n"
        f"Left in-doubt (unreadable): {', '.join(blocked) or 'none'}"
    )


def _place_order_with_retry(
    ibkr: IBKRClient,
    ticker: str,
    side: str,
    shares: int,
    label: str,
    use_bracket: bool = False,
    bracket_kwargs: dict | None = None,
) -> dict:
    """Place a market order with retry on Rejected/Timeout.

    Returns the order result dict. Logs retries and final failure.

    Retry rules:
      * Rejected → real failure (Cancelled/Inactive/ApiCancelled). Retry.
      * Timeout  → no answer from IB Gateway at all. Cancel prior orderId
                   (best-effort) to prevent a stale duplicate, then retry.
      * Working  → IB accepted the order and is holding it (PreSubmitted at
                   pre-open, Submitted routing, etc.). Do NOT retry — it will
                   fill when the hold releases. Retrying duplicates the order
                   (PFE incident 2026-04-22).
      * Filled / PartialFill → done.

    L133 (2026-05-22): the returned dict carries two new audit-trail
    fields the caller embeds into ``rationale_json``:
      * ``retry_count`` (int) — 0 if the first attempt succeeded.
      * ``attempts`` (list of dict) — per-attempt
        ``{attempt, status, ib_order_id, retry_reason}`` records.
    Closes the home-endnote gap that PR #100 had to reword from
    "every order, fill, retry, and exit decision recorded with
    rationale" — the retry chain is now persisted alongside the
    final fill state, not lost on the floor.
    """
    order_result = None
    attempts: list[dict] = []
    for attempt in range(MAX_ORDER_RETRIES):
        retry_reason: str | None = None
        if attempt > 0:
            prior_id = order_result.get("ib_order_id") if order_result else None
            prior_status = order_result.get("status") if order_result else None
            retry_reason = prior_status  # "Timeout" / "Rejected" → why we're retrying
            if prior_status == "Timeout" and prior_id is not None:
                try:
                    ibkr.cancel_order(prior_id)
                except Exception as exc:
                    logger.warning("cancel_order(%s) raised: %s", prior_id, exc)
            _time.sleep(ORDER_RETRY_DELAYS[attempt])
            logger.info("Retry %d/%d: %s %s", attempt + 1, MAX_ORDER_RETRIES, label, ticker)
        if use_bracket and bracket_kwargs:
            from executor.bracket_orders import place_bracket_with_stop
            order_result = place_bracket_with_stop(ibkr, ticker, shares, **bracket_kwargs)
        else:
            order_result = ibkr.place_market_order(ticker, side, shares)
        attempts.append({
            "attempt": attempt + 1,
            "status": order_result.get("status"),
            "ib_order_id": order_result.get("ib_order_id"),
            "retry_reason": retry_reason,
        })
        if order_result["status"] not in ("Rejected", "Timeout"):
            break
    # Embed the audit trail on the returned dict so log_trade callers
    # can include it verbatim in rationale_json. Routing decisions
    # never read these — purely audit.
    order_result["retry_count"] = len(attempts) - 1
    order_result["attempts"] = attempts
    return order_result


def _enqueue_cover_for_unintended_shorts(
    positions: dict,
    order_book: "OrderBook",
    run_date: str,
) -> list[str]:
    """Scan IB positions; enqueue an URGENT COVER for any short.

    Runs at the top of Phase 0 when ``allow_shorts=False`` (the default).
    Any negative position is treated as unintended — we did not knowingly
    open it — and must be flattened immediately at market open.

    Mirror of urgent_exit: emits a COVER urgent into the order book so the
    existing Phase 0 BUY path executes it. Dedupe is handled by
    ``OrderBook.add_urgent_exit`` (ticker+signal), so calling this twice in
    a session is safe.

    Returns the list of tickers that had an auto-cover enqueued.
    """
    if _allow_shorts:
        return []
    covered = []
    for ticker, pos in positions.items():
        shares = int(pos.get("shares", 0))
        if shares >= 0:
            continue
        qty = abs(shares)
        logger.error(
            "AUTO-COVER %s: detected short position %d — enqueuing URGENT COVER %d",
            ticker, shares, qty,
        )
        order_book.add_urgent_exit({
            "ticker": ticker,
            "signal": "COVER",
            "shares": qty,
            "reason": "auto_cover_unintended_short",
            "detail": f"position={shares} at Phase 0 open; allow_shorts=False",
            "date": run_date,
        })
        covered.append(ticker)
    return covered


def load_config() -> dict:
    with open(get_config_path()) as f:
        return yaml.safe_load(f)


def _signals_fingerprint(signals: dict | None) -> str | None:
    """Return a stable content fingerprint for a signals payload (config#897).

    Used to detect a mid-session signals.json refresh (e.g. a manual
    Saturday-SF re-run during a weekday session) so the daemon only recomputes
    the surveillance universe and touches IB subscriptions when the payload
    actually changed. ``None`` when signals are unavailable, so a failed read
    never spuriously looks like a change.
    """
    if not signals:
        return None
    try:
        return hashlib.sha256(
            json.dumps(signals, sort_keys=True, default=str).encode()
        ).hexdigest()
    except (TypeError, ValueError):  # unserializable payload — treat as opaque
        return None


def _refresh_surveillance_universe(
    monitor,
    *,
    config: dict,
    run_date: str,
    order_book: "OrderBook",
    ibkr,
    dry_run: bool,
    last_fingerprint: str | None,
    current_tickers: list[str],
) -> tuple[list[str], str | None]:
    """Re-derive the IB surveillance universe if signals.json changed (config#897).

    Re-reads signals.json, and — only when its content fingerprint differs from
    ``last_fingerprint`` — recomputes the surveillance universe and diff-applies
    it to the monitor (subscribe newly-added, cancel newly-removed). Preserves
    the existing universe on an unchanged payload (zero IB churn) and fails soft
    to the current universe on a read error (logs a warning, never crashes the
    daemon or drops live subscriptions).

    Returns ``(tickers, fingerprint)`` — the universe now in force and the
    fingerprint to carry into the next tick. On no-op / failure these are the
    unchanged current values.
    """
    try:
        from executor.signal_reader import read_signals_with_fallback
        signals = read_signals_with_fallback(config["signals_bucket"], run_date)
    except Exception as sig_err:  # noqa: BLE001
        logger.warning(
            "surveillance refresh: read_signals_with_fallback failed (%s) — "
            "keeping current universe (%d tickers)",
            sig_err, len(current_tickers),
        )
        return current_tickers, last_fingerprint

    fingerprint = _signals_fingerprint(signals)
    if fingerprint is not None and fingerprint == last_fingerprint:
        # Unchanged signals.json — no recompute, no IB churn.
        return current_tickers, last_fingerprint

    try:
        positions = list(ibkr.get_positions().keys()) if not dry_run else []
    except Exception as pos_err:  # noqa: BLE001
        logger.warning(
            "surveillance refresh: get_positions failed (%s) — universe degrades",
            pos_err,
        )
        positions = []

    new_tickers = compute_surveillance_universe(
        signals,
        order_book_tickers=order_book.all_tickers(),
        current_positions=positions,
    )

    if set(new_tickers) == set(current_tickers):
        # Payload changed but the derived universe did not (e.g. a rewrite that
        # didn't add/remove names). Adopt the new fingerprint, skip IB work.
        return new_tickers, fingerprint

    added, removed = monitor.resubscribe(new_tickers)
    logger.info(
        "surveillance universe refreshed mid-session (config#897): "
        "+%s -%s → %d tickers",
        sorted(added) or "[]", sorted(removed) or "[]", len(new_tickers),
    )
    return new_tickers, fingerprint


def run_daemon(dry_run: bool = False) -> None:
    """Main daemon loop — runs until market close or shutdown signal."""
    global _shutdown_requested

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = load_config()
    strategy_config = load_strategy_config(config)

    # Preflight: AWS_REGION + S3 bucket reachable. The check_ib_paper_account
    # primitive on the returned preflight instance is reused after IBKRClient
    # connects below (replaces the inline live-account SAFETY HALT).
    from executor.preflight import ExecutorPreflight
    preflight = ExecutorPreflight(bucket=config["signals_bucket"], mode="daemon")
    preflight.run()

    # Flow Doctor: retrieve the shared instance set up at module import
    from nousergon_lib.logging import get_flow_doctor
    fd = get_flow_doctor()

    global _allow_shorts
    _allow_shorts = config.get("allow_shorts", False)
    if _allow_shorts:
        logger.warning("allow_shorts=true — short-sell prevention is DISABLED")

    if not strategy_config.get("intraday_enabled", False):
        logger.info("Intraday daemon is disabled in config — exiting")
        return

    client_id = strategy_config.get("intraday_client_id", 2)
    poll_interval = strategy_config.get("intraday_poll_interval_sec", 60)
    # Session axis (config#1610; supersedes the config#1016 rationale):
    # run_date keys the daemon's EVENT artifacts — trades.date, nav_series,
    # decision capture, the EOD SF trigger — so it must be the SESSION BEING
    # TRADED (session_date), NOT now_dual().trading_day. trading_day is the
    # last *closed* session (= D-1 during a live session; #1016's comment
    # believed otherwise), which mislabeled every session from 6/23–7/2 and
    # mis-joined EOD reconciles against the wrong day's snapshot.
    # trading_day remains the knowledge axis: log_trade derives it per fill
    # (D-1, "what closed data was this trade acting on"). run_date must use
    # the SAME axis as main.py's order-book `date` field or the order-book
    # freshness match (order_book.py) discards the morning book — both
    # flipped together in this change. Frozen once at startup: stays correct
    # through the close, and the post-close shutdown path can't drift onto
    # the next session.
    from nousergon_lib.dates import session_date
    try:
        run_date = session_date(strict=True).isoformat()
    except ValueError as _axis_err:
        # Weekend/holiday/post-close start — no session to trade. Exit
        # loudly-but-cleanly (mirrors the intraday_enabled early-return);
        # silently attributing this start to the NEXT session would
        # re-create the mislabel bug in the other direction.
        logger.error("No live NYSE session to trade at startup: %s", _axis_err)
        return

    # ── Intraday reconcile-to-target state (optimizer authority) ─────────────
    # When the optimizer owns the book, hard-risk exits (catastrophic gap stop,
    # drawdown forced-exit) free cash intraday with no same-day redeploy path —
    # the bug that left the book at 22% cash / SPY 0 on 2026-05-29. This state
    # drives the real-time drawdown overlay + event-driven re-solve that
    # redeploys freed cash back to the sleeve. See executor/intraday_resolve.py.
    use_optimizer = bool(config.get("use_portfolio_optimizer", False))
    _opt_cfg = config.get("portfolio_optimizer", {}) or {}
    resolve_enabled = use_optimizer and bool(_opt_cfg.get("intraday_resolve_enabled", True))
    overlay_enabled = bool(_opt_cfg.get("intraday_drawdown_overlay_enabled", True))
    resolve_min_freed_pct = float(_opt_cfg.get("intraday_resolve_min_freed_cash_pct", 0.01))
    resolve_cutoff_et = str(_opt_cfg.get("intraday_resolve_cutoff_et", "15:30"))
    resolve_max_per_day = int(_opt_cfg.get("intraday_resolve_max_per_day", 5))
    _stopped_out_today: set[str] = set()   # gap-stopped + dd-forced names (no same-day rebuy)
    _dd_forced_today: set[str] = set()     # subset force-exited by the drawdown overlay
    _hard_risk_exit_seen = False           # event flag: a hard-risk exit freed cash → re-solve
    _resolve_count = 0
    _shadow_log_cache: dict | None = None
    # Per-position catastrophic-gap-stop-watch cohort (config#846): worst
    # intraday drop each gap-only position experienced vs its reference,
    # whether the stop fired, and the threshold in effect. Flushed once per
    # session to risk_events in the finally block so the offline
    # catastrophic_gap_stop_pct tuner has a realized-outcome cohort to join.
    _gap_watch: dict[str, dict] = {}
    if resolve_enabled:
        logger.info(
            "Intraday reconcile ENABLED — drawdown overlay + event-driven "
            "re-solve (min_freed=%.1f%% NAV, cutoff=%s ET, max/day=%d)",
            resolve_min_freed_pct * 100, resolve_cutoff_et, resolve_max_per_day,
        )

    logger.info(
        "Intraday daemon starting | date=%s | dry_run=%s | clientId=%d | poll=%ds",
        run_date, dry_run, client_id, poll_interval,
    )

    # Wait for order book — polls every 2 minutes until one appears or market closes.
    # This allows the daemon to recover from a late predictor inference or morning batch.
    # _ET timezone is module-scoped now (decision-capture wiring needs it from
    # _execute_entry); rebinding here would shadow the module constant.
    order_book_poll_secs = strategy_config.get("order_book_poll_interval_sec", 120)
    order_book = OrderBook.load()
    notified_no_order_book = False

    while not order_book.has_content() and not _shutdown_requested:
        now_et = datetime.now(_ET)

        # Notify once at market open that we have no order book
        if not notified_no_order_book and (now_et.hour > MARKET_OPEN_HOUR or (now_et.hour == MARKET_OPEN_HOUR and now_et.minute >= MARKET_OPEN_MINUTE)):
            send_daemon_status(
                "\u26a0\ufe0f *No order book at market open*\n"
                f"Date: {run_date}\n"
                "Waiting for morning batch to write order book..."
            )
            notified_no_order_book = True

        # Give up once market session ends (4:15 PM ET, accounts for 15-min data delay)
        if not is_market_hours(now_et) and (now_et.hour > MARKET_OPEN_HOUR or (now_et.hour == MARKET_OPEN_HOUR and now_et.minute >= MARKET_OPEN_MINUTE)):
            send_daemon_status(
                "\u274c *No order book received today*\n"
                f"Date: {run_date}\n"
                "Market closed — daemon exiting."
            )
            logger.info("Order book never arrived and market has closed — exiting.")
            return

        logger.info(
            "Order book is empty — waiting %ds for morning batch to write it...",
            order_book_poll_secs,
        )
        _time.sleep(order_book_poll_secs)
        order_book = OrderBook.load()

    if _shutdown_requested:
        return

    # Notify that order book has arrived (especially useful on late-start days)
    if notified_no_order_book:
        send_daemon_status(
            "\u2705 *Order book received (late start)*\n"
            f"Date: {run_date}\n"
            f"Entries: {len(order_book.pending_entries())}\n"
            f"Urgent exits: {len(order_book.pending_urgent_exits())}\n"
            f"Stops: {len(order_book.active_stops())}"
        )
        logger.info("Order book arrived after market open — proceeding with late start")

    # Connect to IB Gateway (with retry)
    db_path = config["db_path"]
    conn = init_db(db_path)

    max_connect_attempts = config.get("ibkr_daemon_max_connect_attempts", 10)
    connect_backoff_base = DEFAULT_CONNECT_BACKOFF_BASE

    def _connect_ibkr() -> IBKRClient:
        """Connect to IB Gateway with exponential backoff. Raises after max attempts."""
        for attempt in range(1, max_connect_attempts + 1):
            try:
                client = IBKRClient(
                    host=config["ibkr_host"],
                    port=config["ibkr_port"],
                    client_id=client_id,
                    reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
                )
                # Paper account safety check — delegate the "starts with D"
                # rule to the shared preflight primitive so every module uses
                # the same definition of "paper account."
                try:
                    accounts = client.ib.managedAccounts()
                    acct = accounts[0] if accounts else ""
                    try:
                        preflight.check_ib_paper_account(acct)
                    except RuntimeError as exc:
                        logger.critical("SAFETY HALT: %s — daemon refusing to trade.", exc)
                        client.disconnect()
                        raise SystemExit(1) from exc
                    logger.info("Paper account verified: %s", acct)
                except SystemExit:
                    raise
                except Exception as e:
                    logger.error(
                        "Paper account verification failed on attempt %d: %s — will retry",
                        attempt, e,
                    )
                    client.disconnect()
                    raise  # let outer retry loop reconnect and re-verify
                return client
            except SystemExit:
                raise
            except Exception as e:
                wait = min(connect_backoff_base * attempt, MAX_RECONNECT_BACKOFF_SECS)
                logger.warning(
                    "IB Gateway connection attempt %d/%d failed: %s — retrying in %ds",
                    attempt, max_connect_attempts, e, wait,
                )
                if attempt == max_connect_attempts:
                    raise
                _time.sleep(wait)
                if _shutdown_requested:
                    raise KeyboardInterrupt("Shutdown during reconnect")

    ibkr = _connect_ibkr()

    # Crash-recovery: resolve any write-ahead 'executing' entries left by a
    # previous daemon that died between order placement and finalization
    # (config#2328). Must run after IB connects (needs broker truth) and
    # before executed_tickers is seeded below (may finalize entries into
    # executed_today). No-op on a clean boot.
    _reconcile_executing_entries(ibkr, order_book, dry_run)

    monitor = make_price_monitor(ibkr.ib)
    exit_mgr = IntradayExitManager(strategy_config)
    entry_engine = EntryTriggerEngine(strategy_config)

    # Surveillance universe — signals.signals ∪ buy_candidates ∪ order_book ∪
    # current_positions (+ SPY). Same universe is derived independently by
    # the surveillance Lambda from the same canonical artifacts, so producer
    # and consumer agree by construction (ROADMAP L1067). Best-effort:
    # signals.json read failure degrades to order_book + positions only, so
    # daemon still trades the order book even if signals are unreadable.
    try:
        from executor.signal_reader import read_signals_with_fallback
        signals_for_surveillance = read_signals_with_fallback(
            config["signals_bucket"], run_date,
        )
    except Exception as _sig_err:  # noqa: BLE001
        logger.warning(
            "surveillance: read_signals_with_fallback failed (%s) — "
            "universe degrades to order_book + positions only",
            _sig_err,
        )
        signals_for_surveillance = None
    try:
        positions_for_surveillance = (
            list(ibkr.get_positions().keys()) if not dry_run else []
        )
    except Exception as _pos_err:  # noqa: BLE001
        logger.warning("surveillance: get_positions failed (%s) — universe degrades", _pos_err)
        positions_for_surveillance = []

    tickers = compute_surveillance_universe(
        signals_for_surveillance,
        order_book_tickers=order_book.all_tickers(),
        current_positions=positions_for_surveillance,
    )
    monitor.subscribe(tickers)

    # Fingerprint of the signals payload the current IB subscription was
    # derived from. The poll loop re-reads signals.json each tick and, when
    # this fingerprint changes (a mid-session Research re-run), diff-applies
    # the new surveillance universe to the monitor instead of waiting for a
    # daemon restart (config#897). None when signals were unreadable at boot.
    surveillance_fingerprint = _signals_fingerprint(signals_for_surveillance)

    # Research-conviction map for intraday drawdown forced-exit ranking
    # (config#844). Built once from the already-loaded signals payload so
    # select_forced_exits ranks lowest-conviction-first instead of degrading
    # to smallest-position-first. Empty map (the prior fallback) if signals
    # were unreadable, so de-risking still works without it.
    forced_exit_signals_by_ticker = build_conviction_map(signals_for_surveillance)

    # S3 snapshot writer — publishes latest_prices + heartbeat at every poll
    # tick. Surveillance Lambda treats heartbeat staleness as daemon-down.
    snapshot_writer = IntradaySnapshotWriter(
        bucket=config["signals_bucket"],
        daemon_pid=os.getpid(),
    )

    # Open-IB-orders snapshot writer — publishes trades/open_orders/latest.json
    # each tick so the dashboard's reconciliation table can render the
    # "Working $" column alongside the optimizer's "Planned $". Same
    # fire-and-forget contract as IntradaySnapshotWriter.
    open_orders_writer = OpenOrdersSnapshotWriter(
        bucket=config["signals_bucket"],
        daemon_pid=os.getpid(),
    )

    # Live-NAV snapshot writer — publishes intraday/nav.json each tick so
    # live.nousergon.ai can render a live intraday header (current NAV,
    # today's return + alpha vs SPY) instead of only the last EOD close.
    # Same fire-and-forget contract as the writers above.
    nav_writer = IntradayNavWriter(bucket=config["signals_bucket"])

    # Per-day NAV series writer — appends each tick's (NAV, SPY) point to
    # intraday/nav_series/{run_date}.json so the live site can draw an
    # intraday portfolio-vs-SPY curve, not just the latest number.
    nav_series_writer = IntradayNavSeriesWriter(bucket=config["signals_bucket"])

    n_urgent = len(order_book.pending_urgent_exits())
    send_daemon_status(
        f"\u2705 *Daemon started*\n"
        f"Date: {run_date}\n"
        f"Monitoring: {len(tickers)} tickers\n"
        f"Urgent exits: {n_urgent}\n"
        f"Entries: {len(order_book.pending_entries())}\n"
        f"Stops: {len(order_book.active_stops())}"
    )

    trades_executed = 0
    # Rehydrate the already-traded set from durable state so a crash-restart
    # does not re-place a BUY the pre-crash process already sent (config#2328).
    # Three unioned sources, each closing a different crash window:
    #   * order_book.executed_today  — clean executions that reached book save.
    #   * order_book executing WAL    — orders sent to IB but not finalized;
    #     _reconcile_executing_entries above already resolved these (finalized
    #     ones are now in executed_today; unreadable ones stay 'executing' and
    #     are seeded here so they are never re-bought).
    #   * trades.db ENTER fills today — belt-and-suspenders: a fill logged to
    #     trades.db whose following book save never landed.
    executed_tickers: set = set()  # tracks tickers already traded today
    executed_tickers.update(
        e["ticker"] for e in order_book.data.get("executed_today", [])
        if e.get("ticker")
    )
    executed_tickers.update(e["ticker"] for e in order_book.executing_entries())
    if not dry_run:
        try:
            executed_tickers.update(get_executed_entry_tickers(conn, run_date))
        except Exception as _seed_err:  # noqa: BLE001 — seeding is defensive
            logger.warning(
                "Could not seed executed_tickers from trades.db (%s) — relying "
                "on order-book state only", _seed_err,
            )
    if executed_tickers:
        logger.info(
            "Seeded executed_tickers from durable state (%d): %s",
            len(executed_tickers), sorted(executed_tickers),
        )

    # Track whether we reached the live trading window so the finally block
    # can decide whether it's safe to fire the EOD pipeline. Pre-market exits
    # (shutdown signal, early crash, etc.) must NOT trigger EOD — market-close
    # side-effects like stopping the trading EC2 instance would then run
    # before the market ever opened.
    market_opened = False

    try:
        # ── Wait for market open (daemon may start before 9:30 AM ET) ────
        if not is_market_hours():
            logger.info("Market not yet open — waiting for 9:30 AM ET...")
            while not _shutdown_requested and not is_market_hours():
                ibkr.ib.sleep(15)
            if _shutdown_requested:
                return
            logger.info("Market is open — proceeding")

        # Guard: Phase 0 urgent exits + live IB order placement must only run
        # once the market is actually open. Reaching this line means the
        # wait-for-open loop above exited on is_market_hours() == True.
        market_opened = True

        # ── Phase 0: Execute urgent exits/covers immediately (no trigger delay) ──
        # Fetch current positions once for short-sell prevention checks
        _phase0_positions = ibkr.get_positions() if not dry_run else {}

        # Auto-cover: any short position at Phase 0 open is unintended
        # (allow_shorts=False is the configured invariant). Enqueue URGENT
        # COVERs into the order book before the normal loop so the existing
        # BUY path flattens them. Covers the residue from bugs like the PFE
        # retry-duplicate incident 2026-04-22.
        if not dry_run:
            auto_covered = _enqueue_cover_for_unintended_shorts(
                _phase0_positions, order_book, run_date,
            )
            if auto_covered:
                send_daemon_status(
                    f"⚠️ *AUTO-COVER enqueued for unintended shorts*\n"
                    f"Tickers: {', '.join(auto_covered)}"
                )
                order_book.save()

        for urgent in order_book.pending_urgent_exits():
            ticker = urgent["ticker"]
            action = urgent["signal"]  # "EXIT", "REDUCE", or "COVER"
            shares = urgent["shares"]
            reason = urgent.get("reason", "research_signal")

            # COVER = buy to close a short position
            if action == "COVER":
                side = "BUY"
            else:
                side = "SELL"
                # Short-sell prevention: cap sell shares at held minus in-flight.
                # In-flight check defends against the PFE incident 2026-04-22
                # where a retry loop issued three duplicate SELL 77s that each
                # individually passed held=155, summing to 231 → short 76.
                pending = 0
                if not dry_run:
                    try:
                        pending = ibkr.get_open_sell_shares(ticker)
                    except Exception as exc:
                        logger.warning("get_open_sell_shares(%s) failed: %s — treating as 0", ticker, exc)
                validated = _validate_sell_shares(
                    _phase0_positions, ticker, shares, action, "URGENT",
                    pending_sell_shares=pending,
                )
                if validated is None:
                    order_book.mark_urgent_executed(ticker, action)
                    continue
                shares = validated

            logger.info(
                "%sURGENT %s %s: %s %d shares | reason: %s",
                "[DRY RUN] " if dry_run else "",
                action, ticker, side, shares, reason,
            )

            if not dry_run:
                order_result = _place_order_with_retry(ibkr, ticker, side, shares, f"URGENT {action}")
                if order_result["status"] in ("Rejected", "Timeout"):
                    logger.error("URGENT %s %s FAILED after %d attempts: %s", action, ticker, MAX_ORDER_RETRIES, order_result["status"])
                    send_daemon_status(
                        f"\u26a0\ufe0f *URGENT {action} {ticker} FAILED*\n"
                        f"Status: {order_result['status']} after {MAX_ORDER_RETRIES} retries\n"
                        f"Shares: {shares} | Reason: {reason}"
                    )
                    continue

                # Use actual filled quantity (handles PartialFill)
                actual_shares = order_result.get("filled_shares") or shares
                if actual_shares != shares:
                    logger.warning(
                        "URGENT %s %s partial fill: requested %d, filled %d",
                        action, ticker, shares, actual_shares,
                    )

                fill_price = order_result.get("fill_price") or ibkr.get_current_price(ticker) or 0

                # ── Roundtrip linkage for urgent exits ──
                _entry = get_unmatched_entry(conn, ticker)
                _entry_id = _entry["trade_id"] if _entry else None
                _entry_fill = _entry["fill_price"] if _entry else None
                _entry_spy = _entry.get("spy_price_at_order") if _entry else None
                _entry_date = _entry["date"] if _entry else None
                _spy_now = None
                _spy_state = monitor.get_price("SPY")
                if _spy_state:
                    _spy_now = _spy_state.get("last")
                _rpnl = ((fill_price - _entry_fill) * actual_shares) if _entry_fill else None
                _rpct = ((fill_price / _entry_fill) - 1) * 100 if _entry_fill else None
                _spy_ret = ((_spy_now / _entry_spy) - 1) * 100 if (_spy_now and _entry_spy) else None
                _ralpha = (_rpct - _spy_ret) if (_rpct is not None and _spy_ret is not None) else None
                _dheld = (date.fromisoformat(run_date) - date.fromisoformat(_entry_date)).days if _entry_date else None

                log_trade(conn, {
                    "date": run_date,
                    "ticker": ticker,
                    "action": action,
                    "shares": actual_shares,
                    "price_at_order": fill_price,
                    "portfolio_nav_at_order": None,
                    "position_pct": None,
                    "ib_order_id": order_result.get("ib_order_id"),
                    "fill_price": fill_price,
                    "fill_time": order_result.get("fill_time"),
                    "filled_shares": order_result.get("filled_shares"),
                    "status": order_result.get("status"),
                    "research_score": urgent.get("research_score"),
                    "research_conviction": urgent.get("research_conviction"),
                    "research_rating": urgent.get("research_rating"),
                    "sector_rating": urgent.get("sector_rating"),
                    "market_regime": urgent.get("market_regime"),
                    "predicted_direction": urgent.get("predicted_direction"),
                    "prediction_confidence": urgent.get("prediction_confidence"),
                    "exit_reason": reason,
                    "rationale_json": json.dumps({
                        "action": action,
                        "exit_reason": reason,
                        "exit_detail": urgent.get("detail", ""),
                        "source": "intraday_daemon",
                        "phase": "urgent",
                        # L133 — universal context on every exit, not
                        # just edge cases. Closes home endnote bullet 3.
                        "signal_context": {
                            "research_score": urgent.get("research_score"),
                            "research_rating": urgent.get("research_rating"),
                            "research_conviction": urgent.get("research_conviction"),
                            "sector": urgent.get("sector"),
                            "sector_rating": urgent.get("sector_rating"),
                            "market_regime": urgent.get("market_regime"),
                            "predicted_direction": urgent.get("predicted_direction"),
                            "prediction_confidence": urgent.get("prediction_confidence"),
                        },
                        # L133 — retry chain audit trail from
                        # _place_order_with_retry. retry_count=0 +
                        # single-entry attempts list = first-attempt
                        # success.
                        "retry_count": order_result.get("retry_count", 0),
                        "attempts": order_result.get("attempts", []),
                    }),
                    "entry_trade_id": _entry_id,
                    "trigger_price": fill_price,
                    "trigger_type": reason,
                    "spy_price_at_order": _spy_now,
                    "realized_pnl": _rpnl,
                    "realized_return_pct": _rpct,
                    "spy_return_during_hold": _spy_ret,
                    "realized_alpha_pct": _ralpha,
                    "days_held": _dheld,
                    # Phase 2 lineage — signal_date / prediction_date are
                    # the artifact filename dates the urgent_exits_with_meta
                    # record sourced from. Both default to None for COVER
                    # orders generated outside the deciders path.
                    "signal_date": urgent.get("signal_date"),
                    "prediction_date": urgent.get("prediction_date"),
                })

                order_book.mark_urgent_executed(ticker, action)
                if action == "EXIT":
                    order_book.remove_stop(ticker)

                # Update cached positions after execution
                if action == "COVER":
                    _phase0_positions.pop(ticker, None)
                elif ticker in _phase0_positions:
                    held_after = int(_phase0_positions[ticker].get("shares", 0)) - shares
                    _phase0_positions[ticker]["shares"] = held_after

                # L165 (2026-05-22): pass the SEMANTIC action ("EXIT" /
                # "REDUCE" / "COVER"), not the IB side ("SELL" / "BUY").
                # The intraday `_execute_exit` path at L1094 already does
                # this correctly; the prior asymmetric spelling caused
                # "Telegram shows SELL but page 16 shows REDUCE" for
                # morning urgent REDUCEs. The action label IS the
                # alert's semantic payload — losing it silently fails
                # the alert's purpose ([[feedback_no_silent_fails]]).
                send_trade_alert(
                    action=action,
                    ticker=ticker,
                    shares=shares,
                    price=fill_price,
                    trigger=f"urgent_{reason}",
                    source="daemon",
                )

                # L139(a) — daemon-stage intraday replay capture. The
                # urgent_exits loop is the morning-planner-to-daemon
                # handoff; recording it here gives the backtester an
                # entry point into the intraday decision stream for
                # eventual gate-enforcement (L139b).
                _get_decision_logger().record(
                    decision_type=(
                        "phase0_auto_cover" if action == "COVER"
                        else "urgent_exit"
                    ),
                    ticker=ticker,
                    action=action,
                    trading_day=run_date,
                    shares=shares,
                    trigger_reason=reason,
                    fill_price=fill_price,
                    ib_order_id=order_result.get("ib_order_id"),
                    status=order_result.get("status"),
                    retry_count=order_result.get("retry_count", 0),
                    attempts=order_result.get("attempts", []),
                    context={
                        "exit_detail": urgent.get("detail", ""),
                        "research_score": urgent.get("research_score"),
                        "research_conviction": urgent.get("research_conviction"),
                        "research_rating": urgent.get("research_rating"),
                        "sector": urgent.get("sector"),
                        "sector_rating": urgent.get("sector_rating"),
                        "market_regime": urgent.get("market_regime"),
                        "predicted_direction": urgent.get("predicted_direction"),
                        "prediction_confidence": urgent.get("prediction_confidence"),
                    },
                )

                trades_executed += 1
                # COVER trades shouldn't prevent new ENTER for the same ticker
                if action != "COVER":
                    executed_tickers.add(ticker)

        if n_urgent > 0:
            order_book.save()
            logger.info("Phase 0 complete: %d urgent exits processed", n_urgent)

        # ── Phase 1+2: Monitor entries and exits ──────────────────────────
        while not _shutdown_requested:
            # Check market hours
            if not is_market_hours():
                logger.info("Market closed — daemon shutting down")
                break

            try:
                # Let ib_insync process events in short bursts so SIGTERM
                # is checked promptly (avoids 60s blocking sleep that caused
                # SIGKILL and dirty IB disconnects / competing sessions).
                _poll_remaining = poll_interval
                while _poll_remaining > 0 and not _shutdown_requested:
                    _chunk = min(_poll_remaining, 5)
                    ibkr.ib.sleep(_chunk)
                    _poll_remaining -= _chunk
                if _shutdown_requested:
                    break
            except (ConnectionError, OSError, asyncio_exceptions) as e:
                logger.warning("IB Gateway connection lost during poll: %s — reconnecting", e)
                ibkr, monitor = _reconnect(ibkr, monitor, order_book, config, client_id)
                continue

            # Reload order book in case morning batch updated it
            order_book = OrderBook.load()
            order_book.merge_executed(executed_tickers)

            # Re-derive the IB surveillance universe if signals.json was
            # refreshed mid-session (config#897). Re-reads signals.json and
            # only recomputes + diff-applies subscriptions when the payload
            # changed — unchanged signals produce zero IB churn, and a read
            # failure fails soft to the current universe. Runs after the
            # order_book reload so newly-tracked book names are included.
            tickers, surveillance_fingerprint = _refresh_surveillance_universe(
                monitor,
                config=config,
                run_date=run_date,
                order_book=order_book,
                ibkr=ibkr,
                dry_run=dry_run,
                last_fingerprint=surveillance_fingerprint,
                current_tickers=tickers,
            )

            # Per-tick structured log line consumed by uptime_tracker.
            # Format is stable — parsers match on the DAEMON_TICK prefix.
            logger.info("DAEMON_TICK ib_connected=%s", str(ibkr.ib.isConnected()).lower())

            # ── Intraday S3 snapshot (surveillance Lambda producer) ───
            # Fire-and-forget; failures log a warning. Surveillance Lambda
            # treats heartbeat staleness as daemon-down (ROADMAP L1067).
            try:
                snapshot_writer.write(
                    monitor.prices,
                    ib_connected=ibkr.ib.isConnected(),
                    subscribed_tickers=tickers,
                )
            except Exception as _snap_err:  # noqa: BLE001
                # Defensive — IntradaySnapshotWriter.write should already
                # swallow S3 errors. This catch ensures no unexpected
                # exception from the writer leaks into the trade loop.
                logger.warning("intraday snapshot writer raised: %s", _snap_err)

            # ── Live-NAV snapshot + series ───────────────────────────
            # Publish intraday/nav.json (latest NetLiquidation + SPY mark)
            # for the live header, and append the same point to the per-day
            # nav_series for the intraday curve. Only when connected — a NAV
            # read off a dead session is meaningless. One get_account_snapshot
            # call feeds both writers. Fire-and-forget; same defensive belt.
            try:
                if ibkr.ib.isConnected():
                    spy_last = (monitor.prices.get("SPY") or {}).get("last")
                    account_snapshot = ibkr.get_account_snapshot()
                    nav_writer.write(
                        account_snapshot,
                        spy_last=spy_last,
                        ib_connected=True,
                    )
                    nav_series_writer.write(
                        run_date,
                        account_snapshot,
                        spy_last=spy_last,
                    )
            except Exception as _nav_err:  # noqa: BLE001
                logger.warning("nav snapshot writer raised: %s", _nav_err)

            # ── Open-IB-orders snapshot ──────────────────────────────
            # Refresh trades/open_orders/latest.json each tick so the
            # dashboard's reconciliation view sees the current working-
            # order state. Fire-and-forget; same defensive try/except
            # belt as the price snapshot writer above.
            try:
                if ibkr.ib.isConnected():
                    open_orders_writer.write(
                        ibkr.ib.openTrades(),
                        calendar_date=run_date,
                        trading_day=run_date,
                    )
            except Exception as _oo_err:  # noqa: BLE001
                logger.warning(
                    "open-orders snapshot writer raised: %s", _oo_err,
                )

            # ── Mid-day backup (noon ET) ─────────────────────────────
            global _midday_backup_done
            _now_bk = datetime.now(_ET)
            if not _midday_backup_done and _now_bk.hour == 12 and _now_bk.minute < 5:
                try:
                    from executor.trade_logger import backup_to_s3 as _midday_bk
                    _midday_bk(db_path, _now_bk.strftime("%Y-%m-%d"), config["signals_bucket"])
                    logger.info("Mid-day trades.db backup completed")
                    _midday_backup_done = True
                except Exception as _bk_err:
                    logger.warning("Mid-day backup failed: %s", _bk_err)

            # ── Check exits ──────────────────────────────────────────
            for stop in order_book.active_stops():
                ticker = stop["ticker"]
                price_state = monitor.get_price(ticker)
                if not price_state:
                    continue

                # Dispatch on book authority. Under the optimizer the alpha
                # rules (trailing-stop / profit-take / 5% collapse) and the
                # trail ratchet are retired — ONLY the hard-risk catastrophic
                # gap stop runs. The authority is the RUN-level ``use_optimizer``
                # flag (belt); ``build_stop_record`` also stamps each record's
                # ``stop_kind`` (suspenders). Keying the dispatch off the run
                # config means an un-stamped record from a forgotten/new
                # producer can NEVER reintroduce same-day churn — it fails SAFE
                # toward no-churn. (WDAY 2026-06-05: a daemon-entered stop
                # lacked stop_kind, fell through here, and the collapse rule
                # force-sold it the same day.)
                stop_kind = stop.get("stop_kind")
                if use_optimizer or stop_kind == "catastrophic_gap_only":
                    if use_optimizer and stop_kind != "catastrophic_gap_only":
                        logger.warning(
                            "stop_kind=%r under optimizer authority for %s — "
                            "forcing catastrophic_gap_only (producer did not "
                            "stamp the record; failing safe toward no-churn)",
                            stop_kind, ticker,
                        )
                    exit_signal = exit_mgr.check_catastrophic_gap(stop, price_state)
                    # Accumulate the gap-stop-watch cohort (config#846): the
                    # worst drop this position reaches vs its reference, and
                    # whether the stop fired — the counterfactual the offline
                    # catastrophic_gap_stop_pct tuner needs (firings alone can't
                    # tell you whether a tighter/looser threshold would help).
                    # Pure observability: best-effort, never perturbs the exit
                    # path, flushed once at session end.
                    try:
                        _gap = exit_mgr.catastrophic_gap_drop(stop, price_state)
                        if _gap is not None:
                            _w = _gap_watch.get(ticker)
                            if _w is None:
                                _w = {
                                    "max_drop": _gap["drop"],
                                    "reference_price": _gap["reference"],
                                    "price_at_max_drop": _gap["current"],
                                    "threshold": _gap["threshold"],
                                    "fired": False,
                                }
                                _gap_watch[ticker] = _w
                            elif _gap["drop"] > _w["max_drop"]:
                                _w["max_drop"] = _gap["drop"]
                                _w["price_at_max_drop"] = _gap["current"]
                            _w["fired"] = _w["fired"] or _gap["fired"]
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "gap-watch accumulation failed for %s", ticker,
                            exc_info=True,
                        )
                else:
                    # Check for trail update first
                    trail_update = exit_mgr.should_update_trail(stop, price_state["last"])
                    if trail_update:
                        new_high, new_stop = trail_update
                        order_book.update_stop_high_water(ticker, new_high, new_stop)
                        order_book.save()
                        logger.debug(
                            "Trail updated %s: high=$%.2f stop=$%.2f",
                            ticker, new_high, new_stop,
                        )

                    # Check exit rules
                    exit_signal = exit_mgr.evaluate(stop, price_state)
                if exit_signal:
                    try:
                        _execute_exit(
                            ibkr, conn, order_book, exit_signal, price_state,
                            run_date, dry_run, monitor=monitor,
                        )
                        if not dry_run:
                            trades_executed += 1
                            executed_tickers.add(exit_signal.get("ticker"))
                            # A catastrophic gap stop is a hard-risk exit: the
                            # name must NOT be re-bought same-day (morning alpha
                            # is still positive), and the freed cash triggers an
                            # event-driven re-solve in the reconcile block below.
                            if exit_signal.get("reason") == "catastrophic_gap_stop":
                                _stopped_out_today.add(exit_signal.get("ticker"))
                                _hard_risk_exit_seen = True
                            # Emit executor:exit_rules DecisionArtifact
                            # (L2308 PR 4 — daemon-side intraday exits).
                            # Lands AFTER the fill succeeded (mirrors PR 1
                            # entry-trigger capture pattern). Best-effort:
                            # capture failure must never kill subsequent
                            # exit executions. Skipped naturally on
                            # dry_run via the not-dry_run guard above.
                            if is_decision_capture_enabled():
                                try:
                                    capture_exit_rule(
                                        run_date=run_date,
                                        stop=stop,
                                        price_state=price_state,
                                        exit_signal=exit_signal,
                                        strategy_config=strategy_config,
                                    )
                                except DecisionCaptureWriteError as _cap_exc:
                                    logger.warning(
                                        "decision_capture S3 write failed "
                                        "for EXIT %s — continuing daemon "
                                        "(capture is observability, not "
                                        "load-bearing): %s",
                                        exit_signal.get("ticker"), _cap_exc,
                                    )
                                except Exception:  # noqa: BLE001
                                    logger.exception(
                                        "decision_capture raised unexpected "
                                        "exception for EXIT %s — continuing "
                                        "daemon",
                                        exit_signal.get("ticker"),
                                    )
                    except (ConnectionError, OSError, asyncio_exceptions) as e:
                        logger.warning("Connection lost during exit %s: %s — reconnecting", exit_signal.get("ticker"), e)
                        ibkr, monitor = _reconnect(ibkr, monitor, order_book, config, client_id)
                        break

            # ── Intraday reconcile-to-target (optimizer owns the book) ────────
            # Real-time drawdown overlay (force de-risk + suppress redeploy) and
            # event-driven re-solve that redeploys cash freed by hard-risk exits
            # back to the sleeve same-day. Reuses the morning Σ + alpha_hat from
            # the optimizer shadow log → zero new alpha look-ahead. Guarded so a
            # reconcile failure never kills the daemon (cash just carries to the
            # next morning planner); the re-solve itself fails loud (no silent
            # no-op-as-success). Runs after exits / before entries so freshly
            # enqueued redeploy buys get a fill attempt this same tick.
            if resolve_enabled and not dry_run:
                try:
                    nav = ibkr.get_portfolio_nav()
                    try:
                        peak = ibkr.get_peak_nav(conn) or nav
                    except Exception:
                        peak = nav
                    positions = ibkr.get_positions()
                    overlay = compute_drawdown_overlay(nav, peak, config, strategy_config)
                    if not overlay_enabled:
                        # Overlay off: no intraday forced exits, no drawdown
                        # redeploy suppression — redeploy purely event-driven.
                        overlay = {**overlay, "forced_exit_count": 0, "redeploy_suppressed": False}

                    # (a) Drawdown de-risk: force-exit lowest-conviction names.
                    # Independent of the shadow log so de-risking works even if
                    # the log is missing. Ranked by research conviction
                    # (forced_exit_signals_by_ticker, built once at startup from
                    # the signals payload, config#844); if signals were
                    # unreadable the map is empty and ranking falls back to
                    # smallest-position-first (the prior conservative behavior).
                    if overlay["forced_exit_count"] > 0:
                        for fx in select_forced_exits(
                            positions, forced_exit_signals_by_ticker,
                            _stopped_out_today | _dd_forced_today,
                            overlay["forced_exit_count"],
                        ):
                            ps = monitor.get_price(fx["ticker"])
                            if not ps:
                                continue
                            _execute_exit(ibkr, conn, order_book, fx, ps, run_date,
                                          dry_run, monitor=monitor)
                            _dd_forced_today.add(fx["ticker"])
                            _stopped_out_today.add(fx["ticker"])
                            _hard_risk_exit_seen = True
                            trades_executed += 1
                            executed_tickers.add(fx["ticker"])
                            logger.warning(
                                "INTRADAY DRAWDOWN FORCED EXIT: %s (%s)",
                                fx["ticker"], overlay["tier_desc"],
                            )

                    # (b) Redeploy freed cash — only when NOT de-risking, only in
                    # response to a hard-risk exit, within the fill window, and
                    # under the per-day cap.
                    if (
                        _hard_risk_exit_seen
                        and not overlay["redeploy_suppressed"]
                        and _resolve_count < resolve_max_per_day
                        and _within_resolve_window(resolve_cutoff_et)
                    ):
                        if _shadow_log_cache is None:
                            _shadow_log_cache = _load_optimizer_shadow_log(config["signals_bucket"])
                        if _shadow_log_cache is not None:
                            sleeve = float((_shadow_log_cache.get("optimizer_cfg") or {}).get("cash_sleeve_pct", 0.03))
                            pending_dollars = sum(
                                (e.get("dollar_size") or (e.get("shares", 0) * (e.get("current_price") or 0)))
                                for e in order_book.pending_entries()
                            )
                            avail = available_redeploy_cash(
                                _shadow_log_cache["tickers"], positions, nav,
                                sleeve, pending_dollars,
                            )
                            if avail > resolve_min_freed_pct * nav:
                                res = solve_redeploy(
                                    shadow_log=_shadow_log_cache,
                                    current_positions=positions,
                                    nav=nav,
                                    stopped_out=_stopped_out_today,
                                )
                                _resolve_count += 1
                                # Event consumed: only re-solve again when a NEW
                                # hard-risk exit fires (coalesces same-tick stops,
                                # prevents per-tick churn).
                                _hard_risk_exit_seen = False
                                n_enq = 0
                                if res["status"] not in ("optimal", "optimal_inaccurate"):
                                    logger.error(
                                        "Intraday re-solve status=%r — NOT redeploying; "
                                        "~$%.0f freed cash left idle for next planner",
                                        res["status"], avail,
                                    )
                                else:
                                    for b in res["buys"]:
                                        if b["ticker"] in _stopped_out_today:
                                            continue
                                        ps = monitor.get_price(b["ticker"])
                                        px = ps.get("last") if ps else None
                                        if not px or px <= 0:
                                            continue
                                        sh = int(b["delta_dollars"] // px)
                                        if sh <= 0:
                                            continue
                                        order_book.add_entry(build_redeploy_entry(
                                            b["ticker"], sh, px, b["target_weight"], run_date,
                                            pullback_pct=strategy_config.get("intraday_pullback_pct", 0.02),
                                        ))
                                        n_enq += 1
                                    if n_enq:
                                        order_book.save()
                                    logger.info(
                                        "Intraday re-solve #%d: enqueued %d redeploy buy(s) "
                                        "for ~$%.0f freed cash (vol_ann=%s)",
                                        _resolve_count, n_enq, avail, res.get("vol_ann"),
                                    )
                                # Log the cash-resolution event (config#846) so the
                                # intraday_resolve_* thresholds can be tuned offline
                                # against realized outcomes: freed cash, solver
                                # status, redeploy count, and the window/cap params
                                # in effect. Best-effort observability.
                                try:
                                    log_risk_event(conn, {
                                        "date": run_date,
                                        "event_type": "intraday_resolve",
                                        "rule": "intraday_resolve",
                                        "value": round(float(avail), 2),
                                        "threshold": round(float(resolve_min_freed_pct * nav), 2),
                                        "reason": res.get("status"),
                                        "context": {
                                            "resolve_count": _resolve_count,
                                            "n_redeployed": n_enq,
                                            "solve_status": res.get("status"),
                                            "vol_ann": res.get("vol_ann"),
                                            "nav": round(float(nav), 2),
                                            "min_freed_cash_pct": resolve_min_freed_pct,
                                            "cutoff_et": resolve_cutoff_et,
                                            "max_per_day": resolve_max_per_day,
                                            "redeploy_tickers": [
                                                b["ticker"] for b in res.get("buys", [])
                                            ],
                                        },
                                    })
                                except Exception:  # noqa: BLE001
                                    logger.debug(
                                        "intraday_resolve event log failed",
                                        exc_info=True,
                                    )
                except Exception as _rec_err:
                    logger.error(
                        "Intraday reconcile error (freed cash may stay idle until "
                        "the next morning planner): %s", _rec_err, exc_info=True,
                    )

            # ── Check entries ────────────────────────────────────────
            if strategy_config.get("intraday_entry_triggers_enabled", True):
                for entry in order_book.pending_entries():
                    ticker = entry["ticker"]
                    price_state = monitor.get_price(ticker)
                    if not price_state:
                        continue

                    should_enter, reason = entry_engine.should_enter(entry, price_state)
                    if should_enter:
                        try:
                            _execute_entry(
                                ibkr, conn, order_book, entry, price_state, reason,
                                run_date, strategy_config, dry_run, monitor=monitor,
                                use_optimizer=use_optimizer,
                            )
                            if not dry_run:
                                trades_executed += 1
                                executed_tickers.add(ticker)
                        except (ConnectionError, OSError, asyncio_exceptions) as e:
                            logger.warning("Connection lost during entry %s: %s — reconnecting", ticker, e)
                            ibkr, monitor = _reconnect(ibkr, monitor, order_book, config, client_id)
                            break

    except Exception as _exc:
        logger.exception("Daemon error")
        if fd:
            fd.report(_exc, severity="critical", context={
                "site": "daemon_main", "dry_run": dry_run, "run_date": run_date})
        send_daemon_status("\u274c *Daemon crashed* — check logs")
        raise
    finally:
        # ── L139(a) — flush intraday decision capture for replay parity ──
        # Best-effort; flush failures WARN-log only (the primary trade
        # path already completed via log_trade / send_trade_alert).
        # Append semantics handle fix-and-rerun cycles within a single
        # trading_day.
        try:
            _get_decision_logger().flush_to_s3(
                bucket=config.get("signals_bucket", "alpha-engine-research"),
                trading_day=run_date,
            )
        except Exception:
            logger.debug("daemon_state flush failed", exc_info=True)

        # ── Data manifest ──────────────────────────────────────────────────
        try:
            from executor.data_manifest import write_data_manifest
            write_data_manifest(
                bucket=config.get("signals_bucket", "alpha-engine-research"),
                module_name="daemon",
                run_date=run_date,
                manifest={
                    "trades_executed": trades_executed,
                    "tickers_monitored": len(order_book.all_tickers()) if order_book else 0,
                },
            )
        except Exception:
            logger.debug("Data manifest write failed", exc_info=True)

        # ── Flush the catastrophic-gap-stop-watch cohort (config#846) ──────
        # One risk_events row per gap-only position: worst intraday drop vs
        # reference, whether the stop fired, and the threshold in effect —
        # the realized-outcome cohort the offline catastrophic_gap_stop_pct
        # tuner joins to score_performance_outcomes. Best-effort; a logging
        # failure must never mask a trading-session error.
        if conn is not None and _gap_watch:
            for _tk, _w in _gap_watch.items():
                try:
                    log_risk_event(conn, {
                        "date": run_date,
                        "event_type": "catastrophic_gap_watch",
                        "rule": "catastrophic_gap_stop",
                        "ticker": _tk,
                        "value": round(float(_w["max_drop"]), 6),
                        "threshold": round(float(_w["threshold"]), 6),
                        "reason": "fired" if _w["fired"] else "watched",
                        "context": {
                            "max_drop": _w["max_drop"],
                            "reference_price": _w["reference_price"],
                            "price_at_max_drop": _w["price_at_max_drop"],
                            "fired": _w["fired"],
                        },
                    })
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "gap-watch flush failed for %s", _tk, exc_info=True,
                    )

        _cleanup_connections(monitor, ibkr)
        if conn:
            conn.close()
        if fd:
            fd.log_summary(logger)
        send_daemon_status(
            f"\u23f9 *Daemon stopped*\n"
            f"Trades executed: {trades_executed}"
        )
        logger.info("Daemon shutdown complete | trades=%d", trades_executed)

        # Trigger EOD pipeline Step Function only when two conditions hold
        # at exit time:
        #   1. market_opened: the daemon actually entered the live trading
        #      window. Pre-market exits (crash, signal) must not fire EOD.
        #   2. not is_market_hours(): the market is closed RIGHT NOW. This
        #      makes SIGTERM-driven mid-session restarts (systemctl restart,
        #      maintenance) safe — the daemon exits, market is still open,
        #      no EOD fires, instance stays up.
        # Checking state at exit (instead of tracking an exit-reason flag)
        # handles the race where the market closes between the last loop
        # iteration and the finally block.
        #
        # 2026-04-28: this is the canonical EOD trigger after Phase 1 of the
        # EOD-SF cutover. Replaced systemd timer + EventBridge cron with a
        # single daemon-driven path per "no redundant paths." Removed by
        # PR #94 on 2026-04-22 (which kept the systemd timer instead) —
        # restored here because the timer was retired in PR #117 and the
        # SF is the architecturally correct caller.
        if not dry_run and market_opened and not is_market_hours():
            _trigger_eod_pipeline(config, run_date)
        elif not dry_run:
            logger.warning(
                "Skipping EOD pipeline trigger: market_opened=%s market_open_now=%s",
                market_opened, is_market_hours(),
            )


def _trigger_eod_pipeline(config: dict, run_date: str) -> None:
    """Start the EOD Step Function pipeline after daemon shutdown.

    Input shape matches the SF's expectations:
    - ``trading_instance_id`` (array for SSM sendCommand) — PostMarketData,
      CaptureSnapshot, EODReconcile, StopTradingInstance all target this.
    - ``ec2_instance_id`` (array) — DailySubstrateHealthCheck targets the
      dashboard EC2 (where alpha-engine-dashboard + lib pin are installed).
    - ``sns_topic_arn`` — HandleFailure publish.

    Extra fields (``run_date``, ``triggered_by``) are harmless — the SF
    ignores them but they're useful for audit / debugging in execution
    history.

    Failures are non-fatal so a transient SF / IAM hiccup doesn't crash
    the daemon mid-shutdown — BUT they are no longer silent. A failure to
    start the SF means NO EOD pipeline runs at all (no NAV/alpha row, no
    EOD email, trading box left up), and unlike a mid-pipeline failure
    there is no SNS HandleFailure branch to fire because the SF never
    started. So on failure we record the swallow on two surfaces per the
    no-silent-fails rule: (1) a named CloudWatch metric
    ``eod_trigger_failure`` (alarmable), and (2) a Telegram operator alert
    via ``send_daemon_status``. This closed the silent-EOD-gap that bit
    2026-06-30 (config#1447): the ne-postclose rename left the executor
    IAM unapplied, ``start_execution`` returned AccessDenied, and the bare
    WARN here meant the missing EOD went unnoticed until the next morning.
    """
    try:
        import boto3 as _b3_sf
        sfn = _b3_sf.client("stepfunctions", region_name="us-east-1")
        state_machine_arn = "arn:aws:states:us-east-1:711398986525:stateMachine:ne-postclose-trading-pipeline"
        trading_instance_id = "i-018eb3307a21329bf"
        dashboard_instance_id = "i-09b539c844515d549"
        sns_topic_arn = config.get(
            "sns_topic_arn",
            "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts",
        )
        import json as _json_sf
        sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"eod-{run_date}-{int(__import__('time').time())}",
            input=_json_sf.dumps({
                "trading_instance_id": [trading_instance_id],
                "ec2_instance_id": [dashboard_instance_id],
                "sns_topic_arn": sns_topic_arn,
                "run_date": run_date,
                "triggered_by": "daemon_shutdown",
                # pipeline_role tag (Option-D 2026-05-25) — page 25 filters
                # EOD section to role="eod" by default. Operator-initiated
                # EOD replays MUST set their own pipeline_role per the
                # taxonomy in pipeline-reporting-revamp-260524.md §6.
                "pipeline_role": "eod",
            }),
        )
        logger.info("EOD pipeline triggered: %s", state_machine_arn)
    except Exception as exc:
        logger.warning("Failed to trigger EOD pipeline (non-fatal): %s", exc)
        # No-silent-fails: the SF never started, so there is no SNS
        # HandleFailure path to surface this. Record it loudly on two
        # independent surfaces (config#1447). Each is best-effort and must
        # not raise — daemon shutdown has already done its job.
        try:
            import boto3 as _b3_cw
            _b3_cw.client("cloudwatch", region_name="us-east-1").put_metric_data(
                Namespace="AlphaEngine/Executor",
                MetricData=[{
                    "MetricName": "eod_trigger_failure",
                    "Value": 1.0,
                    "Unit": "Count",
                }],
            )
        except Exception as _metric_exc:
            logger.warning(
                "CloudWatch eod_trigger_failure metric failed: %s", _metric_exc,
            )
        try:
            send_daemon_status(
                f"\U0001f6a8 *EOD pipeline trigger FAILED* — no EOD will run "
                f"for {run_date} (NAV/alpha/email missing, box left up). "
                f"Error: {exc}"
            )
        except Exception as _notify_exc:
            logger.warning(
                "Telegram EOD-trigger-failure alert failed: %s", _notify_exc,
            )


def _execute_exit(
    ibkr: IBKRClient,
    conn: "sqlite3.Connection",
    order_book: OrderBook,
    exit_signal: dict,
    price_state: dict,
    run_date: str,
    dry_run: bool,
    monitor: "PriceMonitor | None" = None,
) -> None:
    """Execute an intraday exit (SELL or REDUCE)."""
    ticker = exit_signal["ticker"]
    action = exit_signal["action"]  # "EXIT" or "REDUCE"
    shares = exit_signal["shares"]
    sell_action = "SELL"
    current_price = price_state.get("last", 0)

    # Short-sell prevention: verify we hold enough shares net of in-flight sells.
    if not dry_run:
        positions = ibkr.get_positions()
        pending = 0
        try:
            pending = ibkr.get_open_sell_shares(ticker)
        except Exception as exc:
            logger.warning("get_open_sell_shares(%s) failed: %s — treating as 0", ticker, exc)
        validated = _validate_sell_shares(
            positions, ticker, shares, action, "intraday",
            pending_sell_shares=pending,
        )
        if validated is None:
            order_book.remove_stop(ticker)
            order_book.save()
            return
        shares = validated

    logger.info(
        "%s%s %s: %d shares @ ~$%.2f | %s",
        "[DRY RUN] " if dry_run else "",
        action, ticker, shares, current_price, exit_signal.get("detail", ""),
    )

    if dry_run:
        return

    order_result = _place_order_with_retry(ibkr, ticker, sell_action, shares, action)
    if order_result["status"] in ("Rejected", "Timeout"):
        logger.error("%s %s FAILED after %d attempts: %s", action, ticker, MAX_ORDER_RETRIES, order_result["status"])
        send_daemon_status(f"\u26a0\ufe0f *{action} {ticker} FAILED*: {order_result['status']}")
        return

    # Use actual filled quantity (handles PartialFill)
    actual_shares = order_result.get("filled_shares") or shares
    if actual_shares != shares:
        logger.warning(
            "%s %s partial fill: requested %d, filled %d",
            action, ticker, shares, actual_shares,
        )

    fill_price = order_result.get("fill_price") or current_price

    # ── Roundtrip linkage ──
    _entry = get_unmatched_entry(conn, ticker)
    _entry_id = _entry["trade_id"] if _entry else None
    _entry_fill = _entry["fill_price"] if _entry else None
    _entry_spy = _entry.get("spy_price_at_order") if _entry else None
    _entry_date = _entry["date"] if _entry else None
    _spy_now = None
    if monitor:
        _spy_state = monitor.get_price("SPY")
        if _spy_state:
            _spy_now = _spy_state.get("last")
    _rpnl = ((fill_price - _entry_fill) * actual_shares) if _entry_fill else None
    _rpct = ((fill_price / _entry_fill) - 1) * 100 if _entry_fill else None
    _spy_ret = ((_spy_now / _entry_spy) - 1) * 100 if (_spy_now and _entry_spy) else None
    _ralpha = (_rpct - _spy_ret) if (_rpct is not None and _spy_ret is not None) else None
    _dheld = (date.fromisoformat(run_date) - date.fromisoformat(_entry_date)).days if _entry_date else None

    log_trade(conn, {
        "date": run_date,
        "ticker": ticker,
        "action": action,
        "shares": actual_shares,
        "price_at_order": current_price,
        "portfolio_nav_at_order": None,
        "position_pct": None,
        "ib_order_id": order_result.get("ib_order_id"),
        "fill_price": fill_price,
        "fill_time": order_result.get("fill_time"),
        "filled_shares": order_result.get("filled_shares"),
        "status": order_result.get("status"),
        "exit_reason": exit_signal.get("reason"),
        "rationale_json": json.dumps({
            "action": action,
            "exit_reason": exit_signal.get("reason"),
            "exit_detail": exit_signal.get("detail"),
            "source": "intraday_daemon",
            # L133 — signal context at exit time + retry chain audit.
            # ``exit_signal`` carries scores the intraday exit_manager
            # used to decide (profit-take threshold, time-decay tier,
            # ATR multiplier, etc.); pass through verbatim so trades.db
            # records WHY this specific exit fired, not just THAT it did.
            "signal_context": {
                "research_score": exit_signal.get("research_score"),
                "research_rating": exit_signal.get("research_rating"),
                "research_conviction": exit_signal.get("research_conviction"),
                "sector": exit_signal.get("sector"),
                "sector_rating": exit_signal.get("sector_rating"),
                "market_regime": exit_signal.get("market_regime"),
                "predicted_direction": exit_signal.get("predicted_direction"),
                "prediction_confidence": exit_signal.get("prediction_confidence"),
                "exit_gate": exit_signal.get("gate"),  # e.g. "atr_trail", "time_decay", "profit_take"
            },
            "retry_count": order_result.get("retry_count", 0),
            "attempts": order_result.get("attempts", []),
        }),
        "entry_trade_id": _entry_id,
        "trigger_price": current_price,
        "trigger_type": exit_signal.get("reason"),
        "spy_price_at_order": _spy_now,
        "realized_pnl": _rpnl,
        "realized_return_pct": _rpct,
        "spy_return_during_hold": _spy_ret,
        "realized_alpha_pct": _ralpha,
        "days_held": _dheld,
    })

    # Update order book
    if action == "EXIT":
        order_book.remove_stop(ticker)
    elif action == "REDUCE":
        # Update stop record shares to reflect remaining position
        remaining = int(positions.get(ticker, {}).get("shares", 0))
        if remaining > 0:
            order_book.update_stop_shares(ticker, remaining)
            # Mark profit-take as executed so it doesn't fire again
            if exit_signal.get("reason") == "intraday_profit_take":
                order_book.mark_profit_take_executed(ticker)
        else:
            order_book.remove_stop(ticker)
    order_book.save()

    send_trade_alert(
        action=action,
        ticker=ticker,
        shares=shares,
        price=fill_price,
        trigger=exit_signal.get("reason", ""),
        source="daemon",
    )

    # L139(a) — intraday exit decision capture for replay parity.
    _get_decision_logger().record(
        decision_type="intraday_exit",
        ticker=ticker,
        action=action,
        trading_day=run_date,
        shares=shares,
        trigger_reason=exit_signal.get("reason", ""),
        trigger_price=current_price,
        fill_price=fill_price,
        ib_order_id=order_result.get("ib_order_id"),
        status=order_result.get("status"),
        retry_count=order_result.get("retry_count", 0),
        attempts=order_result.get("attempts", []),
        entry_trade_id=_entry_id,
        spy_price_at_order=_spy_now,
        context={
            "exit_detail": exit_signal.get("detail"),
            "exit_gate": exit_signal.get("gate"),
            "research_score": exit_signal.get("research_score"),
            "research_conviction": exit_signal.get("research_conviction"),
            "sector": exit_signal.get("sector"),
            "market_regime": exit_signal.get("market_regime"),
            "predicted_direction": exit_signal.get("predicted_direction"),
            "prediction_confidence": exit_signal.get("prediction_confidence"),
            "realized_pnl": _rpnl,
            "realized_return_pct": _rpct,
            "spy_return_during_hold": _spy_ret,
            "realized_alpha_pct": _ralpha,
            "days_held": _dheld,
        },
    )


def _execute_entry(
    ibkr: IBKRClient,
    conn: "sqlite3.Connection",
    order_book: OrderBook,
    entry: dict,
    price_state: dict,
    trigger_reason: str,
    run_date: str,
    strategy_config: dict,
    dry_run: bool,
    monitor: "PriceMonitor | None" = None,
    use_optimizer: bool = False,
) -> None:
    """Execute an intraday entry (BUY) with bracket stop.

    ``use_optimizer`` carries the book authority into the stop record so a
    daemon-entered position gets the SAME ``stop_kind`` the morning planner
    stamps on held positions. Without it, intraday entries silently defaulted
    to the alpha exit rules and were churned same-day by the 5% intraday
    collapse rule (WDAY 2026-06-05) — see order_book.build_stop_record.
    """
    ticker = entry["ticker"]
    shares = entry.get("shares", 0)
    current_price = price_state.get("last", 0)

    if shares <= 0:
        return

    logger.info(
        "%sBUY %s: %d shares @ ~$%.2f | trigger: %s",
        "[DRY RUN] " if dry_run else "",
        ticker, shares, current_price, trigger_reason,
    )

    if dry_run:
        return

    # Broker-truth pre-BUY duplicate guard (config#2328). Symmetric to the
    # SELL side's held-minus-in-flight cap: refuse the ENTER if the broker
    # already shows a position or a working BUY for this ticker \u2014 the
    # authoritative backstop against a crash-restart re-placing a BUY the
    # pre-crash process already sent. Fails CLOSED on any broker read error.
    if not _validate_buy_not_duplicate(ibkr, ticker, context="daemon_entry"):
        return

    # Try bracket order if ATR is available in the entry
    atr_value = entry.get("atr_value")
    use_bracket = atr_value and atr_value > 0 and strategy_config.get("bracket_stop_enabled", True)
    bracket_mult = strategy_config.get("bracket_trail_atr_multiple", 2.0) if use_bracket else None

    # Write-ahead the intent to disk BEFORE placing the order (config#2328).
    # If the daemon crashes anywhere after this point, the entry is durably
    # 'executing' (not 'pending'), so a restarted daemon will not re-place it
    # via the naive entry loop \u2014 startup reconciliation resolves it instead.
    order_book.mark_entry_executing(ticker, trigger_reason)
    order_book.save()

    _t0_order = _time.time()
    order_result = _place_order_with_retry(
        ibkr, ticker, "BUY", shares, "ENTER",
        use_bracket=bool(use_bracket),
        bracket_kwargs={"atr_value": atr_value, "atr_multiple": bracket_mult} if use_bracket else None,
    )

    if order_result["status"] == "Rejected":
        # Definitive no-fill \u2014 the order was cancelled/rejected at the broker.
        # Roll the WAL entry back to 'pending' so the legitimate entry can
        # retry on a later tick.
        logger.error("ENTER %s REJECTED after %d attempts", ticker, MAX_ORDER_RETRIES)
        send_daemon_status(f"\u26a0\ufe0f *ENTER {ticker} FAILED*: Rejected")
        order_book.revert_entry_to_pending(ticker)
        order_book.save()
        return

    if order_result["status"] == "Timeout":
        # In-doubt \u2014 IB never answered, the order may or may not be working.
        # Leave the entry 'executing' (fail-safe): it stays out of
        # pending_entries so it is not re-placed, and startup reconciliation /
        # EOD reconcile resolves it against broker truth.
        logger.error(
            "ENTER %s TIMEOUT after %d attempts \u2014 leaving in-doubt (executing), "
            "will reconcile against broker", ticker, MAX_ORDER_RETRIES,
        )
        send_daemon_status(
            f"\u26a0\ufe0f *ENTER {ticker} TIMEOUT* \u2014 in-doubt, left blocked "
            f"pending reconcile (config#2328)"
        )
        order_book.save()
        return

    # Use actual filled quantity (handles PartialFill)
    actual_shares = order_result.get("filled_shares") or shares
    if actual_shares != shares:
        logger.warning(
            "ENTER %s partial fill: requested %d, filled %d",
            ticker, shares, actual_shares,
        )

    fill_price = order_result.get("fill_price") or current_price

    # ── Roundtrip fields for entry ──
    _signal_price = entry.get("current_price")  # morning plan price
    _trigger_price = current_price               # price at trigger time
    _spy_now = None
    if monitor:
        _spy_state = monitor.get_price("SPY")
        if _spy_state:
            _spy_now = _spy_state.get("last")
    _slippage = ((fill_price - _signal_price) / _signal_price) if _signal_price else None
    _latency_ms = int((_time.time() - _t0_order) * 1000)

    trade_id = log_trade(conn, {
        "date": run_date,
        "ticker": ticker,
        "action": "ENTER",
        "shares": actual_shares,
        "price_at_order": current_price,
        "portfolio_nav_at_order": None,
        "position_pct": entry.get("position_pct"),
        "ib_order_id": order_result.get("ib_order_id"),
        "fill_price": fill_price,
        "fill_time": order_result.get("fill_time"),
        "filled_shares": order_result.get("filled_shares"),
        "status": order_result.get("status"),
        "research_score": entry.get("research_score"),
        "research_conviction": entry.get("research_conviction"),
        "research_rating": entry.get("research_rating"),
        "sector": entry.get("sector"),
        "sector_rating": entry.get("sector_rating"),
        "market_regime": entry.get("market_regime"),
        "price_target_upside": entry.get("price_target_upside"),
        "predicted_direction": entry.get("predicted_direction"),
        "prediction_confidence": entry.get("prediction_confidence"),
        "rationale_json": json.dumps({
            "action": "ENTER",
            "trigger_reason": trigger_reason,
            "source": "intraday_daemon",
            "planned_price": _signal_price,
            "sizing_factors": entry.get("sizing_factors"),
            "predicted_alpha": entry.get("predicted_alpha"),
            # L133 — full signal context on every ENTER so the
            # trade-by-trade audit log can answer "why did we enter
            # AAPL today?" without joining with signals.json + the
            # predictor archive at review time.
            "signal_context": {
                "research_score": entry.get("research_score"),
                "research_rating": entry.get("research_rating"),
                "research_conviction": entry.get("research_conviction"),
                "sector": entry.get("sector"),
                "sector_rating": entry.get("sector_rating"),
                "market_regime": entry.get("market_regime"),
                "price_target_upside": entry.get("price_target_upside"),
                "predicted_direction": entry.get("predicted_direction"),
                "prediction_confidence": entry.get("prediction_confidence"),
                "position_pct": entry.get("position_pct"),
            },
            # L133 retry chain audit — see _place_order_with_retry docstring.
            "retry_count": order_result.get("retry_count", 0),
            "attempts": order_result.get("attempts", []),
        }),
        "execution_latency_ms": _latency_ms,
        "signal_price": _signal_price,
        "trigger_price": _trigger_price,
        "trigger_type": trigger_reason,
        # entry_trigger duplicates trigger_reason here intentionally —
        # see _TRADES_MIGRATIONS for the canonical-name rationale (the
        # substrate inventory expects entry_trigger; trigger_type is
        # also populated on exits with the exit reason, so the two
        # cannot be merged).
        "entry_trigger": trigger_reason,
        "spy_price_at_order": _spy_now,
        "slippage_vs_signal": _slippage,
        # Date-convention dual-tracking. trading_day is the last completed
        # NYSE session at fill time (populated by log_trade fallback if
        # omitted); signal_trading_day links back to the signals.json that
        # originated this entry — threaded through OrderBook from main.py's
        # _read_signals(). See alpha-engine-docs/private/DATE_CONVENTIONS.md.
        # signal_date and prediction_date are the artifact filename dates
        # (Phase 2 transparency-inventory; ROADMAP entry 2026-05-05).
        "signal_trading_day": entry.get("signal_date"),
        "signal_date": entry.get("signal_date"),
        "prediction_date": entry.get("prediction_date"),
        # Stance taxonomy arc (2026-05-11) — denormalize stance + catalyst_date
        # onto the trade row so exit_manager.evaluate_exits can read them via
        # trade_logger.get_entry_stance_and_catalyst at exit time. Stance routes
        # ATR-multiplier override, time-decay disable, and catalyst hard exit.
        # NULL on entries from planners that haven't been bumped to surface
        # these fields yet — exit_manager falls through to baseline behavior.
        "stance": entry.get("stance"),
        "catalyst_date": entry.get("catalyst_date"),
    })

    # Mark entry as executed in order book
    order_book.mark_entry_executed(ticker, trigger_reason)

    # Emit executor:entry_triggers DecisionArtifact (L2308 PR 1).
    # Best-effort: capture is observability, not load-bearing for the trade
    # itself. Swallow DecisionCaptureWriteError + log loudly so a transient
    # S3 outage doesn't kill subsequent trade executions. No-op when the
    # ALPHA_ENGINE_DECISION_CAPTURE_ENABLED env var is unset (default off).
    if is_decision_capture_enabled():
        try:
            capture_entry_trigger(
                run_date=run_date,
                entry=entry,
                price_state=price_state,
                trigger_reason=trigger_reason,
                strategy_config=strategy_config,
                disabled_triggers=list(strategy_config.get("disabled_triggers", [])),
                now_et_iso=datetime.now(_ET).isoformat(),
                fill_price=fill_price,
                actual_shares=actual_shares,
                trade_id=trade_id,
            )
        except DecisionCaptureWriteError as _cap_exc:
            logger.warning(
                "decision_capture S3 write failed for ENTER %s — continuing "
                "trade flow (capture is observability, not load-bearing): %s",
                ticker, _cap_exc,
            )
        except Exception:  # noqa: BLE001 — capture must never kill trading
            logger.exception(
                "decision_capture raised unexpected exception for ENTER %s "
                "— continuing trade flow", ticker,
            )

    # Add stop record for the new position (skip if ATR unavailable)
    trail_atr = entry.get("atr_value", 0)
    atr_mult = strategy_config.get("intraday_trailing_stop_atr_multiple", 2.0)
    if trail_atr and trail_atr > 0:
        stop_price = round(fill_price - trail_atr * atr_mult, 2)
        order_book.add_stop(build_stop_record(
            ticker=ticker,
            entry_price=fill_price,
            current_stop=stop_price,
            trail_atr=trail_atr,
            atr_multiple=atr_mult,
            high_water=fill_price,
            entry_date=run_date,
            shares=actual_shares,
            use_optimizer=use_optimizer,
            entry_trade_id=trade_id,
        ))
    else:
        fallback_enabled = strategy_config.get("fallback_stop_enabled", True)
        fallback_pct = strategy_config.get("fallback_stop_pct", 0.10)
        if fallback_enabled:
            stop_price = round(fill_price * (1 - fallback_pct), 2)
            logger.warning(
                "No ATR for %s — using %.0f%% fallback stop at $%.2f",
                ticker, fallback_pct * 100, stop_price,
            )
            order_book.add_stop(build_stop_record(
                ticker=ticker,
                entry_price=fill_price,
                current_stop=stop_price,
                trail_atr=0,
                atr_multiple=0,
                high_water=fill_price,
                entry_date=run_date,
                shares=actual_shares,
                use_optimizer=use_optimizer,
                entry_trade_id=trade_id,
            ))
        else:
            logger.warning("No ATR for %s — fallback stop disabled, position has no stop", ticker)
    order_book.save()

    send_trade_alert(
        action="BUY",
        ticker=ticker,
        shares=shares,
        price=fill_price,
        trigger=trigger_reason,
        source="daemon",
    )

    # L139(a) — entry trigger decision capture for replay parity.
    _get_decision_logger().record(
        decision_type="entry_trigger",
        ticker=ticker,
        action="ENTER",
        trading_day=run_date,
        shares=shares,
        trigger_reason=trigger_reason,
        signal_price=_signal_price,
        trigger_price=_trigger_price,
        fill_price=fill_price,
        ib_order_id=order_result.get("ib_order_id"),
        status=order_result.get("status"),
        retry_count=order_result.get("retry_count", 0),
        attempts=order_result.get("attempts", []),
        execution_latency_ms=_latency_ms,
        spy_price_at_order=_spy_now,
        context={
            "predicted_alpha": entry.get("predicted_alpha"),
            "research_score": entry.get("research_score"),
            "research_conviction": entry.get("research_conviction"),
            "research_rating": entry.get("research_rating"),
            "sector": entry.get("sector"),
            "sector_rating": entry.get("sector_rating"),
            "market_regime": entry.get("market_regime"),
            "predicted_direction": entry.get("predicted_direction"),
            "prediction_confidence": entry.get("prediction_confidence"),
            "position_pct": entry.get("position_pct"),
            "sizing_factors": entry.get("sizing_factors"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Intraday Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Log triggers without placing orders")
    args = parser.parse_args()
    # Capture an uncaught daemon crash via flow-doctor before re-raising
    # (no-ops when flow-doctor is inactive).
    with guard_entrypoint():
        run_daemon(dry_run=args.dry_run)
