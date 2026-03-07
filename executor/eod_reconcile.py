"""
EOD reconciliation — runs at 4:05pm ET after market close.

Captures portfolio NAV, computes daily return vs. SPY, writes to eod_pnl table.

Cron:  5 21 * * 1-5  python /home/ec2-user/alpha-engine/executor/eod_reconcile.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date

import yaml
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.eod_emailer import send_eod_email
from executor.ibkr import IBKRClient
from executor.trade_logger import init_db, log_eod, backup_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")


def _spy_close(run_date: str) -> float | None:
    """Fetch SPY closing price for run_date via yfinance."""
    try:
        hist = yf.download("SPY", start=run_date, end=run_date, progress=False, auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].iloc[0])
    except Exception as e:
        logger.warning(f"Could not fetch SPY price: {e}")
    return None


def run(run_date: str | None = None):
    run_date = run_date or str(date.today())
    logger.info(f"EOD reconciliation | date={run_date}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]

    conn = init_db(db_path)
    ibkr = IBKRClient(
        host=config["ibkr_host"],
        port=config["ibkr_port"],
        client_id=config["ibkr_client_id"],
    )

    # Current NAV and positions
    nav = ibkr.get_portfolio_nav()
    positions = ibkr.get_positions()
    ibkr.disconnect()

    # Prior day's NAV (to compute daily return)
    prior_row = conn.execute(
        "SELECT portfolio_nav FROM eod_pnl ORDER BY date DESC LIMIT 1"
    ).fetchone()
    prior_nav = prior_row[0] if prior_row else None

    daily_return = ((nav - prior_nav) / prior_nav * 100) if prior_nav else None

    # SPY return for the day
    spy_price = _spy_close(run_date)
    spy_prior_row = conn.execute(
        "SELECT spy_return_pct, portfolio_nav FROM eod_pnl ORDER BY date DESC LIMIT 1"
    ).fetchone()
    spy_return = None
    if spy_price and spy_prior_row:
        # We don't store spy price directly — use accumulated return approach
        # (simplified: just log SPY daily return from yfinance)
        try:
            hist = yf.download("SPY", period="2d", progress=False, auto_adjust=True)
            if len(hist) >= 2:
                spy_return = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100)
        except Exception as e:
            logger.warning(f"SPY return calc failed: {e}")

    alpha = (daily_return - spy_return) if (daily_return is not None and spy_return is not None) else None

    logger.info(
        f"NAV=${nav:,.2f} | daily={daily_return:.2f}% | "
        f"SPY={spy_return:.2f}% | alpha={alpha:.2f}%"
        if all(x is not None for x in [daily_return, spy_return, alpha])
        else f"NAV=${nav:,.2f} | prior_nav={prior_nav}"
    )

    log_eod(conn, {
        "date": run_date,
        "portfolio_nav": nav,
        "daily_return_pct": daily_return,
        "spy_return_pct": spy_return,
        "daily_alpha_pct": alpha,
        "positions_snapshot": positions,
    })

    backup_to_s3(db_path, run_date, trades_bucket)

    send_eod_email(
        run_date=run_date,
        nav=nav,
        daily_return=daily_return,
        spy_return=spy_return,
        alpha=alpha,
        positions=positions,
        conn=conn,
        sender=config["email_sender"],
        recipients=config["email_recipients"],
    )

    conn.close()
    logger.info("EOD reconciliation complete")


if __name__ == "__main__":
    run()
