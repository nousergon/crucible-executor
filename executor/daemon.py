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

from ssm_secrets import load_secrets
load_secrets()

from executor.entry_triggers import EntryTriggerEngine
from executor.ibkr import IBKRClient
from executor.intraday_exit_manager import IntradayExitManager
from executor.market_hours import is_market_hours
from executor.notifier import send_daemon_status, send_trade_alert
from executor.order_book import OrderBook
from executor.price_monitor import PriceMonitor
from executor.strategies.config import load_strategy_config
from executor.trade_logger import init_db, log_trade, get_unmatched_entry

from alpha_engine_lib.logging import setup_logging
# See executor/main.py for the rationale on IB Error 10197 / 10349 suppression.
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
_FLOW_DOCTOR_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow-doctor.yaml")
setup_logging("daemon", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

# Terminology:
#   "status" — IB order execution status: "Filled", "Rejected", "Timeout", etc.
#   "signal" — trading action type from Research/strategy: "ENTER", "EXIT", "REDUCE", "COVER"

from executor.config_loader import CONFIG_PATH

# Order retry policy — applied uniformly to all order types (urgent exits, intraday exits, entries)
MAX_ORDER_RETRIES = 3
ORDER_RETRY_DELAYS = [0, 2, 5]  # seconds between attempts

# Market timing constants (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30

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
            new_monitor = PriceMonitor(new_ibkr.ib)
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
) -> int | None:
    """Validate and cap sell shares against held position.

    Returns adjusted share count, or None if the sell should be skipped
    (no position held — selling would go short).

    When ``_allow_shorts`` is True (set via ``allow_shorts: true`` in
    risk.yaml), the guard is bypassed and sells are allowed to create
    short positions.
    """
    if _allow_shorts:
        return shares
    held = int(positions.get(ticker, {}).get("shares", 0))
    if held <= 0:
        logger.warning(
            "SKIP %s %s %s: hold %d shares — selling would go short",
            context, action, ticker, held,
        )
        return None
    if shares > held:
        logger.warning(
            "CAPPING %s %s %s: requested %d but hold %d — capping to avoid short",
            context, action, ticker, shares, held,
        )
        return held
    return shares


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
    """
    order_result = None
    for attempt in range(MAX_ORDER_RETRIES):
        if attempt > 0:
            _time.sleep(ORDER_RETRY_DELAYS[attempt])
            logger.info("Retry %d/%d: %s %s", attempt + 1, MAX_ORDER_RETRIES, label, ticker)
        if use_bracket and bracket_kwargs:
            from executor.bracket_orders import place_bracket_with_stop
            order_result = place_bracket_with_stop(ibkr, ticker, shares, **bracket_kwargs)
        else:
            order_result = ibkr.place_market_order(ticker, side, shares)
        if order_result["status"] not in ("Rejected", "Timeout"):
            break
    return order_result


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    from alpha_engine_lib.logging import get_flow_doctor
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
    run_date = date.today().isoformat()

    logger.info(
        "Intraday daemon starting | date=%s | dry_run=%s | clientId=%d | poll=%ds",
        run_date, dry_run, client_id, poll_interval,
    )

    # Wait for order book — polls every 2 minutes until one appears or market closes.
    # This allows the daemon to recover from a late predictor inference or morning batch.
    _ET = pytz.timezone("US/Eastern")
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

    monitor = PriceMonitor(ibkr.ib)
    exit_mgr = IntradayExitManager(strategy_config)
    entry_engine = EntryTriggerEngine(strategy_config)

    # Subscribe to all tickers in the order book (+ SPY for roundtrip benchmarking)
    tickers = order_book.all_tickers()
    if "SPY" not in tickers:
        tickers.append("SPY")
    monitor.subscribe(tickers)

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
    executed_tickers: set = set()  # tracks tickers already traded today

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
                # Short-sell prevention: cap sell shares at current held position
                validated = _validate_sell_shares(_phase0_positions, ticker, shares, action, "URGENT")
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

                send_trade_alert(
                    action=side,
                    ticker=ticker,
                    shares=shares,
                    price=fill_price,
                    trigger=f"urgent_{reason}",
                    source="daemon",
                )

                trades_executed += 1
                # COVER trades shouldn't prevent new ENTER for the same ticker
                if action != "COVER":
                    executed_tickers.add(ticker)

        if n_urgent > 0:
            order_book.save()
            logger.info("Phase 0 complete: %d urgent exits processed", n_urgent)

        # ── Phase 1+2: Monitor entries and exits ──────────────────────────
        _last_heartbeat = _time.time()
        _HEARTBEAT_INTERVAL = strategy_config.get("heartbeat_interval_sec", 3600)

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

            # Per-tick structured log line consumed by uptime_tracker.
            # Format is stable — parsers match on the DAEMON_TICK prefix.
            logger.info("DAEMON_TICK ib_connected=%s", str(ibkr.ib.isConnected()).lower())

            # ── Heartbeat ─────────────────────────────────────────────
            _elapsed = _time.time() - _last_heartbeat
            if _elapsed >= _HEARTBEAT_INTERVAL:
                n_pending = len(order_book.pending_entries())
                n_stops = len(order_book.active_stops())
                n_positions = len(ibkr.get_positions())
                msg = (
                    f"\U0001f49a *Daemon heartbeat*\n"
                    f"Positions: {n_positions} | Stops: {n_stops} | "
                    f"Pending entries: {n_pending}\n"
                    f"Trades today: {trades_executed}"
                )
                ok = send_daemon_status(msg)
                logger.info("Heartbeat sent (ok=%s) after %.0fs | pos=%d stops=%d pending=%d trades=%d", ok, _elapsed, n_positions, n_stops, n_pending, trades_executed)
                _last_heartbeat = _time.time()

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
                    except (ConnectionError, OSError, asyncio_exceptions) as e:
                        logger.warning("Connection lost during exit %s: %s — reconnecting", exit_signal.get("ticker"), e)
                        ibkr, monitor = _reconnect(ibkr, monitor, order_book, config, client_id)
                        break

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
        # ── Data manifest ──────────────────────────────────────────────────
        try:
            from executor.health_status import write_data_manifest
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
        if not dry_run and market_opened and not is_market_hours():
            _trigger_eod_pipeline(config, run_date)
        elif not dry_run:
            logger.warning(
                "Skipping EOD pipeline trigger: market_opened=%s market_open_now=%s",
                market_opened, is_market_hours(),
            )


def _trigger_eod_pipeline(config: dict, run_date: str) -> None:
    """Start the EOD Step Function pipeline after daemon shutdown."""
    try:
        import boto3 as _b3_sf
        sfn = _b3_sf.client("stepfunctions", region_name="us-east-1")
        state_machine_arn = "arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-eod-pipeline"
        micro_instance_id = "i-09b539c844515d549"
        trading_instance_id = "i-018eb3307a21329bf"
        sns_topic_arn = config.get("sns_topic_arn", "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts")
        import json as _json_sf
        sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"eod-{run_date}-{int(__import__('time').time())}",
            input=_json_sf.dumps({
                "ec2_instance_id": [micro_instance_id],
                "trading_instance_id": [trading_instance_id],
                "sns_topic_arn": sns_topic_arn,
                "run_date": run_date,
                "triggered_by": "daemon_shutdown",
            }),
        )
        logger.info("EOD pipeline triggered: %s", state_machine_arn)
    except Exception as exc:
        logger.warning("Failed to trigger EOD pipeline (non-fatal): %s", exc)


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

    # Short-sell prevention: verify we hold enough shares before selling
    if not dry_run:
        positions = ibkr.get_positions()
        validated = _validate_sell_shares(positions, ticker, shares, action, "intraday")
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
) -> None:
    """Execute an intraday entry (BUY) with bracket stop."""
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

    # Try bracket order if ATR is available in the entry
    atr_value = entry.get("atr_value")
    use_bracket = atr_value and atr_value > 0 and strategy_config.get("bracket_stop_enabled", True)
    bracket_mult = strategy_config.get("bracket_trail_atr_multiple", 2.0) if use_bracket else None

    _t0_order = _time.time()
    order_result = _place_order_with_retry(
        ibkr, ticker, "BUY", shares, "ENTER",
        use_bracket=bool(use_bracket),
        bracket_kwargs={"atr_value": atr_value, "atr_multiple": bracket_mult} if use_bracket else None,
    )

    if order_result["status"] in ("Rejected", "Timeout"):
        logger.error("ENTER %s FAILED after %d attempts: %s", ticker, MAX_ORDER_RETRIES, order_result["status"])
        send_daemon_status(f"\u26a0\ufe0f *ENTER {ticker} FAILED*: {order_result['status']}")
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
        }),
        "execution_latency_ms": _latency_ms,
        "signal_price": _signal_price,
        "trigger_price": _trigger_price,
        "trigger_type": trigger_reason,
        "spy_price_at_order": _spy_now,
        "slippage_vs_signal": _slippage,
    })

    # Mark entry as executed in order book
    order_book.mark_entry_executed(ticker, trigger_reason)

    # Add stop record for the new position (skip if ATR unavailable)
    trail_atr = entry.get("atr_value", 0)
    atr_mult = strategy_config.get("intraday_trailing_stop_atr_multiple", 2.0)
    if trail_atr and trail_atr > 0:
        stop_price = round(fill_price - trail_atr * atr_mult, 2)
        order_book.add_stop({
            "ticker": ticker,
            "entry_price": fill_price,
            "current_stop": stop_price,
            "trail_atr": trail_atr,
            "atr_multiple": atr_mult,
            "high_water": fill_price,
            "entry_date": run_date,
            "shares": actual_shares,
            "entry_trade_id": trade_id,
        })
    else:
        fallback_enabled = strategy_config.get("fallback_stop_enabled", True)
        fallback_pct = strategy_config.get("fallback_stop_pct", 0.10)
        if fallback_enabled:
            stop_price = round(fill_price * (1 - fallback_pct), 2)
            logger.warning(
                "No ATR for %s — using %.0f%% fallback stop at $%.2f",
                ticker, fallback_pct * 100, stop_price,
            )
            order_book.add_stop({
                "ticker": ticker,
                "entry_price": fill_price,
                "current_stop": stop_price,
                "trail_atr": 0,
                "atr_multiple": 0,
                "high_water": fill_price,
                "entry_date": run_date,
                "shares": actual_shares,
                "entry_trade_id": trade_id,
            })
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Intraday Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Log triggers without placing orders")
    args = parser.parse_args()
    run_daemon(dry_run=args.dry_run)
