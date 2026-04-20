"""
Emergency shutdown — cancel all orders, close all positions, stop the daemon.

Usage:
    python executor/emergency_shutdown.py                        # dry-run (report only)
    python executor/emergency_shutdown.py --execute              # cancel + liquidate + stop daemon
    python executor/emergency_shutdown.py --execute --stop-instance  # also stop EC2

IB Gateway must be running locally on port 4002 in paper mode.
Uses clientId=3 to avoid conflicts with main (1) and daemon (2).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timezone

import yaml

from executor.ibkr import IBKRClient
from executor.trade_logger import backup_to_s3, init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [EMERGENCY] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from executor.config_loader import load_config as _load_config


def emergency_shutdown(execute: bool, stop_instance: bool) -> None:
    config = _load_config()
    run_date = date.today().isoformat()

    # ── Connect to IB Gateway ──────────────────────────────────────────
    client = IBKRClient(
        host=config.get("ib_host", "127.0.0.1"),
        port=config.get("ib_port", 4002),
        client_id=3,  # Avoid conflicts with main(1) and daemon(2)
    )

    # ── Paper account safety check ─────────────────────────────────────
    try:
        accounts = client.ib.managedAccounts()
        if accounts and not accounts[0].startswith("D"):
            logger.critical("LIVE ACCOUNT DETECTED (%s) — aborting emergency shutdown", accounts[0])
            client.disconnect()
            sys.exit(99)
        logger.info("Paper account confirmed: %s", accounts[0] if accounts else "unknown")
    except Exception as e:
        logger.warning("Could not verify account type: %s", e)

    # ── Report current state ───────────────────────────────────────────
    nav = client.get_portfolio_nav()
    positions = client.get_positions()
    open_orders = client.get_open_orders()

    logger.info("NAV: $%.2f", nav)
    logger.info("Open positions: %d", len(positions))
    for ticker, pos in positions.items():
        logger.info("  %s: %d shares ($%.2f)", ticker, pos["shares"], pos.get("market_value", 0))
    logger.info("Open orders: %d", len(open_orders))

    if not execute:
        logger.info("DRY RUN — no actions taken. Use --execute to proceed.")
        client.disconnect()
        return

    # ── Step 1: Cancel all open orders ─────────────────────────────────
    logger.info("Step 1: Cancelling all open orders...")
    try:
        client.cancel_all_orders()
        logger.info("All orders cancelled")
    except Exception as e:
        logger.error("Order cancellation failed: %s — continuing with liquidation", e)

    # ── Step 2: Close all positions ────────────────────────────────────
    logger.info("Step 2: Closing all positions...")
    db_path = config.get("db_path", "/home/ec2-user/alpha-engine/trades.db")
    conn = init_db(db_path)

    for ticker, pos in positions.items():
        shares = pos["shares"]
        if shares <= 0:
            continue
        logger.info("Selling %d shares of %s at market...", shares, ticker)
        try:
            result = client.place_market_order(ticker, "SELL", shares)
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "EMERGENCY_SELL",
                "shares": shares,
                "price_at_order": client.get_current_price(ticker),
                "fill_price": result.get("fill_price"),
                "filled_shares": result.get("filled_shares"),
                "fill_time": result.get("fill_time"),
                "ib_order_id": result.get("ib_order_id"),
                "status": result.get("status", "Unknown"),
                "source": "emergency_shutdown",
                "portfolio_nav_at_order": nav,
            })
            logger.info("  %s: %s (fill=$%.2f)", ticker, result.get("status"),
                        result.get("fill_price") or 0)
        except Exception as e:
            logger.error("  %s: SELL FAILED — %s", ticker, e)

    conn.commit()

    # ── Step 3: Stop the daemon ────────────────────────────────────────
    logger.info("Step 3: Stopping daemon...")
    try:
        subprocess.run(["sudo", "systemctl", "stop", "alpha-engine-daemon"],
                       timeout=30, capture_output=True)
        logger.info("Daemon stopped")
    except Exception as e:
        logger.warning("Daemon stop failed (may not be running): %s", e)

    # ── Step 4: Backup trades.db ───────────────────────────────────────
    logger.info("Step 4: Backing up trades.db...")
    try:
        s3_bucket = config.get("signals_bucket", "alpha-engine-research")
        backup_to_s3(db_path, run_date, s3_bucket)
        logger.info("Backup complete")
    except Exception as e:
        logger.error("Backup failed: %s", e)

    # ── Step 5: Send notification ──────────────────────────────────────
    logger.info("Step 5: Sending notification...")
    try:
        import boto3
        sns = boto3.client("sns", region_name="us-east-1")
        topic_arn = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts")
        sns.publish(
            TopicArn=topic_arn,
            Subject="Alpha Engine — EMERGENCY SHUTDOWN EXECUTED",
            Message=(
                f"Emergency shutdown completed at {datetime.now(timezone.utc).isoformat()}\n\n"
                f"Actions taken:\n"
                f"  - Cancelled all open orders\n"
                f"  - Closed {len(positions)} positions\n"
                f"  - Stopped daemon\n"
                f"  - Backed up trades.db\n\n"
                f"NAV at shutdown: ${nav:.2f}"
            ),
        )
        logger.info("Notification sent")
    except Exception as e:
        logger.warning("Notification failed: %s", e)

    # ── Step 6: Stop EC2 instance (optional) ───────────────────────────
    if stop_instance:
        logger.info("Step 6: Stopping EC2 instance...")
        try:
            import boto3
            ec2 = boto3.client("ec2", region_name="us-east-1")
            # Get this instance's ID from metadata
            import urllib.request
            token_req = urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                method="PUT",
            )
            token = urllib.request.urlopen(token_req, timeout=2).read().decode()
            id_req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
            )
            instance_id = urllib.request.urlopen(id_req, timeout=2).read().decode()
            ec2.stop_instances(InstanceIds=[instance_id])
            logger.info("EC2 stop-instances sent for %s", instance_id)
        except Exception as e:
            logger.error("EC2 stop failed: %s", e)

    conn.close()
    client.disconnect()
    logger.info("Emergency shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Emergency shutdown — cancel orders + liquidate + stop")
    parser.add_argument("--execute", action="store_true", help="Actually execute (default is dry-run)")
    parser.add_argument("--stop-instance", action="store_true", help="Also stop the EC2 instance")
    args = parser.parse_args()

    if args.execute:
        logger.warning("=" * 60)
        logger.warning("  EMERGENCY SHUTDOWN — EXECUTING")
        logger.warning("=" * 60)

    emergency_shutdown(execute=args.execute, stop_instance=args.stop_instance)


if __name__ == "__main__":
    main()
