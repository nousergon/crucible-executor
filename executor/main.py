"""
Alpha Engine Executor — daily morning trading loop.

Reads signals.json from S3, applies risk rules and position sizing,
places market orders via IB Gateway (paper trading).

Cron (EC2, America/Los_Angeles):
    30 6 * * 1-5  python /home/ec2-user/alpha-engine/executor/main.py >> /var/log/executor.log 2>&1

Usage:
    python main.py              # live paper trading
    python main.py --dry-run    # print orders without placing them
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.ibkr import IBKRClient
from executor.position_sizer import compute_position_size
from executor.risk_guard import check_order
from executor.signal_reader import get_actionable_signals, read_signals
from executor.trade_logger import backup_to_s3, init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run(dry_run: bool = False):
    run_date = str(date.today())
    logger.info(f"Executor starting | date={run_date} | dry_run={dry_run}")

    config = load_config()
    db_path = config["db_path"]
    signals_bucket = config["signals_bucket"]
    trades_bucket = config["trades_bucket"]

    conn = init_db(db_path)

    # ── 1. Read signals from S3 ──────────────────────────────────────────────
    signals_raw = read_signals(signals_bucket, run_date)
    signals = get_actionable_signals(signals_raw)
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]

    logger.info(
        f"Signals | regime={market_regime} "
        f"| ENTER={len(signals['enter'])} EXIT={len(signals['exit'])} "
        f"REDUCE={len(signals['reduce'])} HOLD={len(signals['hold'])}"
    )

    # ── 2. Connect to IBKR ───────────────────────────────────────────────────
    ibkr = IBKRClient(
        host=config["ibkr_host"],
        port=config["ibkr_port"],
        client_id=config["ibkr_client_id"],
    )
    portfolio_nav = ibkr.get_portfolio_nav()
    current_positions = ibkr.get_positions()
    peak_nav = ibkr.get_peak_nav(conn)

    # Enrich positions with sector data from signals
    universe_sectors = {
        s["ticker"]: s.get("sector", "")
        for s in signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])
        if s.get("ticker")
    }
    for ticker, pos in current_positions.items():
        pos["sector"] = universe_sectors.get(ticker, "")

    enter_signals = signals["enter"]

    # ── 3. Process ENTER signals ─────────────────────────────────────────────
    for sig in enter_signals:
        ticker = sig["ticker"]
        sector = sig.get("sector", "Technology")
        sector_info = sector_ratings.get(sector, {})
        sector_rating_str = sector_info.get("rating", "market_weight")

        if ticker in current_positions:
            logger.info(f"SKIP ENTER {ticker} — already in portfolio")
            continue

        current_price = ibkr.get_current_price(ticker)
        if not current_price:
            logger.warning(f"SKIP ENTER {ticker} — no price available")
            continue

        sizing = compute_position_size(
            ticker=ticker,
            portfolio_nav=portfolio_nav,
            enter_signals=enter_signals,
            signal=sig,
            sector_rating=sector_rating_str,
            current_price=current_price,
            config=config,
        )

        if sizing["shares"] == 0:
            logger.info(f"SKIP ENTER {ticker} — position too small (${sizing['dollar_size']:.0f})")
            continue

        # Inject sector_rating into signal for risk guard
        sig_with_sector = {**sig, "sector_rating": sector_rating_str}

        approved, reason = check_order(
            ticker=ticker,
            action="ENTER",
            dollar_size=sizing["dollar_size"],
            portfolio_nav=portfolio_nav,
            peak_nav=peak_nav,
            current_positions=current_positions,
            sector=sector,
            market_regime=market_regime,
            signal=sig_with_sector,
            config=config,
        )

        if not approved:
            logger.info(f"BLOCKED {ticker} — {reason}")
            continue

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER ENTER {ticker} "
            f"{sizing['shares']} shares @ ~${current_price:.2f} "
            f"(${sizing['dollar_size']:.0f}, {sizing['position_pct']*100:.1f}% NAV)"
        )

        if not dry_run:
            order_result = ibkr.place_market_order(ticker, "BUY", sizing["shares"])
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "ENTER",
                "shares": sizing["shares"],
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": sizing["position_pct"],
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": sector_rating_str,
                "market_regime": market_regime,
                "price_target_upside": sig.get("price_target_upside"),
                "thesis_summary": sig.get("thesis_summary"),
                "ib_order_id": order_result.get("ib_order_id"),
            })

    # ── 4. Process EXIT signals ──────────────────────────────────────────────
    for sig in signals["exit"]:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            logger.info(f"SKIP EXIT {ticker} — not in portfolio")
            continue

        shares_held = int(current_positions[ticker]["shares"])
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}ORDER EXIT {ticker} {shares_held} shares")

        if not dry_run:
            current_price = ibkr.get_current_price(ticker)
            order_result = ibkr.place_market_order(ticker, "SELL", shares_held)
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "EXIT",
                "shares": shares_held,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": 0.0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "ib_order_id": order_result.get("ib_order_id"),
            })

    # ── 5. Process REDUCE signals ────────────────────────────────────────────
    for sig in signals["reduce"]:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            continue

        shares_held = int(current_positions[ticker]["shares"])
        shares_to_sell = shares_held // 2
        if shares_to_sell == 0:
            logger.info(f"SKIP REDUCE {ticker} — position too small to halve")
            continue

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER REDUCE {ticker} "
            f"{shares_to_sell} shares (50% reduction)"
        )

        if not dry_run:
            current_price = ibkr.get_current_price(ticker)
            order_result = ibkr.place_market_order(ticker, "SELL", shares_to_sell)
            remaining_value = (shares_held - shares_to_sell) * (current_price or 0)
            log_trade(conn, {
                "date": run_date,
                "ticker": ticker,
                "action": "REDUCE",
                "shares": shares_to_sell,
                "price_at_order": current_price,
                "portfolio_nav_at_order": portfolio_nav,
                "position_pct": remaining_value / portfolio_nav if portfolio_nav else 0,
                "research_score": sig.get("score"),
                "research_conviction": sig.get("conviction"),
                "research_rating": sig.get("rating"),
                "sector_rating": current_positions[ticker].get("sector", ""),
                "market_regime": market_regime,
                "ib_order_id": order_result.get("ib_order_id"),
            })

    # ── 6. Backup and disconnect ─────────────────────────────────────────────
    if not dry_run:
        backup_to_s3(db_path, run_date, trades_bucket)

    ibkr.disconnect()
    conn.close()
    logger.info(f"Executor complete | dry_run={dry_run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orders without placing them",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
