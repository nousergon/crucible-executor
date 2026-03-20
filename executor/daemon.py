"""
Alpha Engine Intraday Daemon — monitors positions and entries during market hours.

Runs from ~6:45 AM to 4:00 PM ET on trading days. Uses 15-minute delayed
streaming data from IB Gateway (free, no subscription required).

Architecture:
  - Morning batch (main.py) writes approved entries and stop state to order_book.json
  - Daemon reads the order book, subscribes to prices, and evaluates:
    1. Entry triggers (pullback, VWAP, support, time expiry) → BUY
    2. Exit rules (trailing stop, profit-take, collapse) → SELL
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
import time as _time
from datetime import date, datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.entry_triggers import EntryTriggerEngine
from executor.ibkr import IBKRClient
from executor.intraday_exit_manager import IntradayExitManager
from executor.market_hours import is_market_hours
from executor.notifier import send_daemon_status, send_trade_alert
from executor.order_book import OrderBook
from executor.price_monitor import PriceMonitor
from executor.strategies.config import load_strategy_config
from executor.trade_logger import init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [daemon] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (%s)", signum)
    _shutdown_requested = True


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

    # Load order book
    order_book = OrderBook.load()
    if not order_book.all_tickers():
        logger.info("Order book is empty — nothing to monitor. Exiting.")
        return

    # Connect to IB Gateway
    db_path = config["db_path"]
    conn = init_db(db_path)

    ibkr = IBKRClient(
        host=config["ibkr_host"],
        port=config["ibkr_port"],
        client_id=client_id,
        reconnect_attempts=config.get("ibkr_reconnect_attempts", 3),
    )

    monitor = PriceMonitor(ibkr.ib)
    exit_mgr = IntradayExitManager(strategy_config)
    entry_engine = EntryTriggerEngine(strategy_config)

    # Subscribe to all tickers in the order book
    tickers = order_book.all_tickers()
    monitor.subscribe(tickers)

    send_daemon_status(
        f"\u2705 *Daemon started*\n"
        f"Date: {run_date}\n"
        f"Monitoring: {len(tickers)} tickers\n"
        f"Entries: {len(order_book.pending_entries())}\n"
        f"Stops: {len(order_book.active_stops())}"
    )

    trades_executed = 0

    try:
        while not _shutdown_requested:
            # Check market hours
            if not is_market_hours():
                logger.info("Market closed — daemon shutting down")
                break

            # Let ib_insync process events and update tickers
            ibkr.ib.sleep(poll_interval)

            # Reload order book in case morning batch updated it
            order_book = OrderBook.load()

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
                    _execute_exit(
                        ibkr, conn, order_book, exit_signal, price_state,
                        run_date, dry_run,
                    )
                    trades_executed += 1

            # ── Check entries ────────────────────────────────────────
            if strategy_config.get("intraday_entry_triggers_enabled", True):
                for entry in order_book.pending_entries():
                    ticker = entry["ticker"]
                    price_state = monitor.get_price(ticker)
                    if not price_state:
                        continue

                    should_enter, reason = entry_engine.should_enter(entry, price_state)
                    if should_enter:
                        _execute_entry(
                            ibkr, conn, order_book, entry, price_state, reason,
                            run_date, strategy_config, dry_run,
                        )
                        trades_executed += 1

    except Exception:
        logger.exception("Daemon error")
        send_daemon_status("\u274c *Daemon crashed* — check logs")
        raise
    finally:
        monitor.unsubscribe_all()
        ibkr.disconnect()
        if conn:
            conn.close()
        send_daemon_status(
            f"\u23f9 *Daemon stopped*\n"
            f"Trades executed: {trades_executed}"
        )
        logger.info("Daemon shutdown complete | trades=%d", trades_executed)


def _execute_exit(
    ibkr: IBKRClient,
    conn,
    order_book: OrderBook,
    exit_signal: dict,
    price_state: dict,
    run_date: str,
    dry_run: bool,
) -> None:
    """Execute an intraday exit (SELL or REDUCE)."""
    ticker = exit_signal["ticker"]
    action = exit_signal["action"]  # "EXIT" or "REDUCE"
    shares = exit_signal["shares"]
    sell_action = "SELL"
    current_price = price_state.get("last", 0)

    logger.info(
        "%s%s %s: %d shares @ ~$%.2f | %s",
        "[DRY RUN] " if dry_run else "",
        action, ticker, shares, current_price, exit_signal.get("detail", ""),
    )

    if dry_run:
        return

    order_result = ibkr.place_market_order(ticker, sell_action, shares)
    if order_result["status"] in ("Rejected", "Timeout"):
        logger.warning("%s %s order %s", action, ticker, order_result["status"])
        return

    fill_price = order_result.get("fill_price") or current_price

    log_trade(conn, {
        "date": run_date,
        "ticker": ticker,
        "action": action,
        "shares": shares,
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
        "execution_latency_ms": None,
    })

    # Update order book
    if action == "EXIT":
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
    conn,
    order_book: OrderBook,
    entry: dict,
    price_state: dict,
    trigger_reason: str,
    run_date: str,
    strategy_config: dict,
    dry_run: bool,
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
    if atr_value and atr_value > 0 and strategy_config.get("bracket_stop_enabled", True):
        from executor.bracket_orders import place_bracket_with_stop
        bracket_mult = strategy_config.get("bracket_trail_atr_multiple", 2.0)
        order_result = place_bracket_with_stop(
            ibkr, ticker, shares,
            atr_value=atr_value,
            atr_multiple=bracket_mult,
        )
    else:
        order_result = ibkr.place_market_order(ticker, "BUY", shares)

    if order_result["status"] in ("Rejected", "Timeout"):
        logger.warning("ENTER %s order %s", ticker, order_result["status"])
        return

    fill_price = order_result.get("fill_price") or current_price

    log_trade(conn, {
        "date": run_date,
        "ticker": ticker,
        "action": "ENTER",
        "shares": shares,
        "price_at_order": current_price,
        "portfolio_nav_at_order": None,
        "position_pct": None,
        "ib_order_id": order_result.get("ib_order_id"),
        "fill_price": fill_price,
        "fill_time": order_result.get("fill_time"),
        "filled_shares": order_result.get("filled_shares"),
        "status": order_result.get("status"),
        "rationale_json": json.dumps({
            "action": "ENTER",
            "trigger_reason": trigger_reason,
            "source": "intraday_daemon",
        }),
        "execution_latency_ms": None,
    })

    # Mark entry as executed in order book
    order_book.mark_entry_executed(ticker, trigger_reason)

    # Add stop record for the new position
    trail_atr = entry.get("atr_value", 0)
    atr_mult = strategy_config.get("intraday_trailing_stop_atr_multiple", 2.0)
    stop_price = round(fill_price - trail_atr * atr_mult, 2) if trail_atr else 0
    order_book.add_stop({
        "ticker": ticker,
        "entry_price": fill_price,
        "current_stop": stop_price,
        "trail_atr": trail_atr,
        "atr_multiple": atr_mult,
        "high_water": fill_price,
        "entry_date": run_date,
        "shares": shares,
    })
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
