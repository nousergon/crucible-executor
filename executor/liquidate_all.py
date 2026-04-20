"""
Liquidate all open positions in the paper account — operational tool for
paper account resets. Not part of the automated trading pipeline.

Usage:
    python liquidate_all.py                    # dry-run (shows what would be sold)
    python liquidate_all.py --execute          # places real market SELL orders
    python liquidate_all.py --execute --yes    # skip confirmation prompt

IB Gateway must be running locally on port 4002 in paper mode.
Trades are logged to trades.db and the DB is backed up to S3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone

import yaml

from ibkr import IBKRClient
from trade_logger import backup_to_s3, init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from executor.config_loader import load_config


def liquidate(execute: bool, skip_confirm: bool) -> None:
    config = load_config()

    client = IBKRClient(
        host=config.get("ibkr_host", "127.0.0.1"),
        port=config.get("ibkr_port", 4002),
        client_id=config.get("ibkr_client_id", 1),
    )

    try:
        nav = client.get_portfolio_nav()
        positions = client.get_positions()

        if not positions:
            logger.info("No open positions — nothing to liquidate.")
            return

        # ── Summary ───────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  LIQUIDATION SUMMARY")
        print("=" * 60)
        print(f"  Portfolio NAV : ${nav:>12,.2f}")
        print(f"  Open positions: {len(positions)}")
        print()
        print(f"  {'Ticker':<8}  {'Shares':>8}  {'Market Value':>14}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*14}")
        total_mv = 0.0
        for ticker, pos in sorted(positions.items()):
            mv = pos.get("market_value", 0)
            total_mv += mv
            print(f"  {ticker:<8}  {pos['shares']:>8,}  ${mv:>13,.2f}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*14}")
        print(f"  {'TOTAL':<8}  {'':>8}  ${total_mv:>13,.2f}")
        print()

        if not execute:
            print("  DRY-RUN mode — no orders placed.")
            print("  Re-run with --execute to place real orders.")
            print("=" * 60)
            return

        if not skip_confirm:
            answer = input(
                f"  Sell ALL {len(positions)} position(s) at market? [yes/no]: "
            ).strip().lower()
            if answer != "yes":
                print("  Aborted.")
                return

        print("=" * 60)
        print()

        # ── Place orders ──────────────────────────────────────────────────────
        run_date = str(date.today())
        db_conn = init_db(config.get("db_path", "trades.db"))

        for ticker, pos in sorted(positions.items()):
            shares = pos["shares"]
            if shares <= 0:
                logger.warning(f"Skipping {ticker} — shares={shares}")
                continue

            price = client.get_current_price(ticker)

            logger.info(f"Selling {shares:,} shares of {ticker} at market...")
            result = client.place_market_order(ticker, "SELL", shares)

            log_trade(
                db_conn,
                {
                    "date": run_date,
                    "ticker": ticker,
                    "action": "LIQUIDATION_SELL",
                    "shares": shares,
                    "price_at_order": price,
                    "portfolio_nav_at_order": nav,
                    "position_pct": round(pos.get("market_value", 0) / nav, 4) if nav else None,
                    "market_regime": "liquidation",
                    "thesis_summary": "Manual liquidation before funding reset",
                    "ib_order_id": result.get("ib_order_id"),
                },
            )

        db_conn.close()

        # ── Backup DB to S3 ───────────────────────────────────────────────────
        try:
            backup_to_s3(
                config.get("db_path", "trades.db"),
                run_date,
                config["trades_bucket"],
            )
        except Exception as exc:
            logger.warning(f"S3 backup failed (non-fatal): {exc}")

        logger.info(
            f"Liquidation complete — {len(positions)} position(s) submitted. "
            "Verify fills in TWS / IB Gateway before resetting funds."
        )

    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sell all open positions before a paper-account funding reset."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Place real SELL orders (default is dry-run).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (use with --execute).",
    )
    args = parser.parse_args()

    if args.yes and not args.execute:
        parser.error("--yes requires --execute")

    liquidate(execute=args.execute, skip_confirm=args.yes)


if __name__ == "__main__":
    main()
