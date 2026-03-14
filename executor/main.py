"""
Alpha Engine Executor — daily morning trading loop.

Reads signals.json from S3, applies risk rules and position sizing,
places market orders via IB Gateway (paper trading).

Strategy layer (added 2026-03-14):
  - Graduated drawdown response scales position sizes by drawdown tier
  - ATR trailing stops exit positions when price falls below trailing stop
  - Time-based decay reduces/exits stale positions after N trading days

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

from executor.ibkr import IBKRClient, SimulatedIBKRClient
from executor.position_sizer import compute_position_size
from executor.risk_guard import check_order, compute_drawdown_multiplier
from executor.signal_reader import get_actionable_signals, read_signals_with_fallback
from executor.strategies.config import load_strategy_config
from executor.strategies.exit_manager import evaluate_exits
from executor.price_cache import load_price_histories
from executor.trade_logger import backup_to_s3, get_entry_dates, init_db, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run(
    dry_run: bool = False,
    simulate: bool = False,
    ibkr_client=None,           # injected by backtester when simulate=True
    signals_override: dict = None,  # injected signals dict (skips S3 read)
    price_histories: dict = None,   # injected by backtester for exit manager
) -> "list[dict] | None":
    """
    Returns list of order dicts when simulate=True, else None.
    All other behaviour (risk guard, position sizer, trade logger) is unchanged.
    """
    orders = []
    run_date = str(date.today())
    logger.info(f"Executor starting | date={run_date} | dry_run={dry_run} | simulate={simulate}")

    config = load_config()
    db_path = config["db_path"]
    signals_bucket = config["signals_bucket"]
    trades_bucket = config["trades_bucket"]

    conn = None if simulate else init_db(db_path)

    # ── 1. Read signals from S3 (or use injected override) ──────────────────
    if signals_override is not None:
        signals_raw = signals_override
        run_date = signals_raw.get("date", run_date)
    else:
        try:
            signals_raw = read_signals_with_fallback(signals_bucket, run_date)
        except RuntimeError as e:
            logger.error(f"Cannot proceed without signals: {e}")
            if conn:
                conn.close()
            return
    signals = get_actionable_signals(signals_raw)
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]

    logger.info(
        f"Signals | regime={market_regime} "
        f"| ENTER={len(signals['enter'])} EXIT={len(signals['exit'])} "
        f"REDUCE={len(signals['reduce'])} HOLD={len(signals['hold'])}"
    )

    # ── 2. Connect to IBKR (or use injected simulated client) ───────────────
    if simulate:
        ibkr = ibkr_client
    else:
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

    # ── 2b. Enrich positions with entry_date from trades.db ──────────────────
    if conn and current_positions:
        entry_dates = get_entry_dates(conn, list(current_positions.keys()))
        for ticker, pos in current_positions.items():
            pos["entry_date"] = entry_dates.get(ticker)
        logger.info(f"Entry dates resolved for {len(entry_dates)}/{len(current_positions)} positions")

    # ── 2c. Compute graduated drawdown multiplier ──────────────────────────
    dd_multiplier, dd_reason = compute_drawdown_multiplier(portfolio_nav, peak_nav, config)
    if dd_multiplier < 1.0:
        logger.info(f"Drawdown tier active: {dd_reason}")

    # ── 2d. Strategy layer: evaluate exit rules on held positions ──────────
    strategy_config = load_strategy_config(config)

    # Build signals lookup for exit manager
    signals_by_ticker = {}
    for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
        t = s.get("ticker")
        if t and t not in signals_by_ticker:
            signals_by_ticker[t] = s

    # Load price histories from predictor S3 cache (unless injected by backtester)
    if price_histories is None and current_positions:
        price_histories = load_price_histories(
            tickers=list(current_positions.keys()),
            signals_bucket=signals_bucket,
        )

    strategy_exits = evaluate_exits(
        current_positions=current_positions,
        signals_by_ticker=signals_by_ticker,
        run_date=run_date,
        price_histories=price_histories or {},
        ibkr_client=ibkr,
        strategy_config=strategy_config,
    )

    if strategy_exits:
        logger.info(
            f"Strategy layer generated {len(strategy_exits)} exit signal(s): "
            + ", ".join(f"{s['ticker']}({s['action']}: {s['reason']})" for s in strategy_exits)
        )

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
            drawdown_multiplier=dd_multiplier,
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

        if simulate:
            orders.append({
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
            })
        elif not dry_run:
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

    # ── 4. Process EXIT signals (Research + Strategy) ───────────────────────
    # Merge Research exits with strategy-generated exits (deduplicate by ticker)
    all_exit_tickers = set()
    all_exits = []
    for sig in signals["exit"]:
        t = sig["ticker"]
        if t not in all_exit_tickers:
            all_exit_tickers.add(t)
            all_exits.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig["action"] == "EXIT" and strat_sig["ticker"] not in all_exit_tickers:
            all_exit_tickers.add(strat_sig["ticker"])
            all_exits.append(strat_sig)

    for sig in all_exits:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            logger.info(f"SKIP EXIT {ticker} — not in portfolio")
            continue

        shares_held = int(current_positions[ticker]["shares"])
        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}ORDER EXIT {ticker} {shares_held} shares{reason_tag}")

        if simulate:
            current_price = ibkr.get_current_price(ticker)
            orders.append({
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
                "exit_reason": sig.get("reason"),
            })
        elif not dry_run:
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

    # ── 5. Process REDUCE signals (Research + Strategy) ─────────────────────
    all_reduce_tickers = set()
    all_reduces = []
    for sig in signals["reduce"]:
        t = sig["ticker"]
        if t not in all_reduce_tickers:
            all_reduce_tickers.add(t)
            all_reduces.append(sig)
    for strat_sig in strategy_exits:
        if strat_sig["action"] == "REDUCE" and strat_sig["ticker"] not in all_reduce_tickers:
            # Also skip if we already have an EXIT for this ticker
            if strat_sig["ticker"] not in all_exit_tickers:
                all_reduce_tickers.add(strat_sig["ticker"])
                all_reduces.append(strat_sig)

    for sig in all_reduces:
        ticker = sig["ticker"]
        if ticker not in current_positions:
            continue

        shares_held = int(current_positions[ticker]["shares"])
        shares_to_sell = shares_held // 2
        if shares_to_sell == 0:
            logger.info(f"SKIP REDUCE {ticker} — position too small to halve")
            continue

        reason_tag = f" ({sig.get('reason', 'research')})" if sig.get("reason") else ""
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER REDUCE {ticker} "
            f"{shares_to_sell} shares (50% reduction){reason_tag}"
        )

        if simulate:
            current_price = ibkr.get_current_price(ticker)
            remaining_value = (shares_held - shares_to_sell) * (current_price or 0)
            orders.append({
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
                "exit_reason": sig.get("reason"),
            })
        elif not dry_run:
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
    if not dry_run and not simulate:
        backup_to_s3(db_path, run_date, trades_bucket)

    ibkr.disconnect()
    if conn:
        conn.close()
    logger.info(f"Executor complete | dry_run={dry_run} | simulate={simulate}")

    if simulate:
        return orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Engine Executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orders without placing them",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
