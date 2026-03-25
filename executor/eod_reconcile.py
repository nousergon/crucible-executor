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
import time as _time
from datetime import date, timedelta

import boto3
import pandas as pd
import yaml
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.eod_emailer import send_eod_email
from executor.ibkr import IBKRClient
from executor.trade_logger import (
    init_db, log_eod, backup_to_s3, get_entry_trade, get_todays_trades,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk.yaml")


def _spy_close(run_date: str) -> float | None:
    """Fetch SPY closing price for run_date via yfinance."""
    try:
        end_date = (date.fromisoformat(run_date) + timedelta(days=1)).isoformat()
        hist = yf.download("SPY", start=run_date, end=end_date, progress=False, auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].values.flat[0])
    except Exception as e:
        logger.warning(f"Could not fetch SPY price: {e}")
    return None


def _load_signals_from_s3(bucket: str, run_date: str, max_lookback: int = 5) -> tuple[dict, str | None]:
    """Load signals.json from S3, falling back to prior trading days.

    Research only runs on Mondays, so on Tue–Fri the exact run_date won't
    have a signals file.  Walk back up to *max_lookback* calendar days
    (skipping weekends) before giving up.
    """
    s3 = boto3.client("s3")
    start = date.fromisoformat(run_date)
    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
        if candidate.weekday() >= 5:  # skip Sat/Sun
            continue
        dt = str(candidate)
        try:
            obj = s3.get_object(Bucket=bucket, Key=f"signals/{dt}/signals.json")
            if days_back > 0:
                logger.info("No signals for %s — using %s (%d day(s) old)", run_date, dt, days_back)
            return json.loads(obj["Body"].read()), None
        except Exception:
            continue
    logger.error("No signals found within %d days of %s", max_lookback, run_date)
    return {}, f"Signals unavailable from S3 for {run_date} (checked {max_lookback} days back)"


def _load_predictions_from_s3(bucket: str) -> tuple[dict, str | None]:
    """Load latest predictions from S3. Returns ({ticker: pred_dict}, warning_msg) on failure."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="predictor/predictions/latest.json")
        data = json.loads(obj["Body"].read())
        return {p["ticker"]: p for p in data.get("predictions", []) if "ticker" in p}, None
    except Exception as e:
        logger.error("Failed to load predictions from S3: %s", e)
        return {}, "Predictions unavailable from S3"


def _build_position_contexts(
    positions: dict,
    conn,
    signals_bucket: str,
    run_date: str,
) -> tuple[list[dict], list[str]]:
    """Assemble per-position context for rationale synthesis.

    Returns (contexts, data_warnings).
    """
    data_warnings: list[str] = []
    signals_data, sig_warn = _load_signals_from_s3(signals_bucket, run_date)
    predictions, pred_warn = _load_predictions_from_s3(signals_bucket)
    if sig_warn:
        data_warnings.append(sig_warn)
    if pred_warn:
        data_warnings.append(pred_warn)

    # Build signals lookup
    signals_by_ticker = {}
    for s in (signals_data.get("universe", []) + signals_data.get("buy_candidates", [])):
        t = s.get("ticker")
        if t:
            signals_by_ticker[t] = s

    todays_trades = get_todays_trades(conn, run_date)
    trades_by_ticker = {}
    for t in todays_trades:
        trades_by_ticker.setdefault(t["ticker"], []).append(t)

    contexts = []
    for ticker, pos in sorted(positions.items()):
        entry = get_entry_trade(conn, ticker)
        sig = signals_by_ticker.get(ticker, {})
        pred = predictions.get(ticker, {})
        today_actions = trades_by_ticker.get(ticker, [])

        entry_rationale = None
        if entry and entry.get("rationale_json"):
            try:
                entry_rationale = json.loads(entry["rationale_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        ctx = {
            "ticker": ticker,
            "shares": pos.get("shares"),
            "market_value": pos.get("market_value"),
            "unrealized_pnl": pos.get("unrealized_pnl"),
            "entry_date": entry["date"] if entry else None,
            "entry_price": entry["price_at_order"] if entry else None,
            "research_score": sig.get("score") or (entry["research_score"] if entry else None),
            "conviction": sig.get("conviction") or (entry["research_conviction"] if entry else None),
            "thesis_summary": sig.get("thesis_summary") or (entry["thesis_summary"] if entry else None),
            "price_target_upside": sig.get("price_target_upside"),
            "sector_rating": sig.get("sector_rating") or (entry["sector_rating"] if entry else None),
            "market_regime": signals_data.get("market_regime"),
            "predicted_direction": pred.get("predicted_direction"),
            "prediction_confidence": pred.get("prediction_confidence"),
            "predicted_alpha": pred.get("predicted_alpha"),
            "today_actions": [
                {"action": t["action"], "shares": t["shares"]}
                for t in today_actions
            ],
            "entry_rationale": entry_rationale,
        }
        contexts.append(ctx)

    return contexts, data_warnings


def _synthesize_rationales(contexts: list[dict]) -> dict[str, str]:
    """Call Haiku to synthesize per-position narratives. Falls back to templates."""
    if not contexts:
        return {}

    # Try LLM synthesis
    try:
        import anthropic
        client = anthropic.Anthropic()

        prompt = (
            "You are a portfolio analyst writing concise position rationales for an end-of-day report.\n"
            "For each position below, write 2-3 sentences explaining why it is held, "
            "focusing on near-term catalysts (research thesis, technical signals, GBM predictions). "
            "If a trade was made today, explain why. Be specific about numbers.\n\n"
            "Return valid JSON only: {\"narratives\": [{\"ticker\": \"XXX\", \"narrative\": \"...\"}]}\n\n"
            f"Positions:\n{json.dumps(contexts, indent=2, default=str)}"
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.content[0].text)
        return {n["ticker"]: n["narrative"] for n in result.get("narratives", [])}
    except Exception as e:
        logger.warning(f"LLM rationale synthesis failed: {e} — using template fallback")

    # Template fallback
    narratives = {}
    for ctx in contexts:
        parts = []
        ticker = ctx["ticker"]

        if ctx.get("entry_date") and ctx.get("entry_price"):
            parts.append(f"Entered {ctx['entry_date']} at ${ctx['entry_price']:.2f}.")

        if ctx.get("research_score") is not None:
            conv = ctx.get("conviction", "stable")
            parts.append(f"Research score {ctx['research_score']:.0f}/100 ({conv}).")

        if ctx.get("predicted_direction"):
            conf = ctx.get("prediction_confidence")
            conf_str = f" ({conf*100:.0f}% conf)" if conf else ""
            alpha = ctx.get("predicted_alpha")
            alpha_str = f", α={alpha*100:.2f}%" if alpha else ""
            parts.append(f"GBM: {ctx['predicted_direction']}{conf_str}{alpha_str}.")

        if ctx.get("thesis_summary"):
            thesis = ctx["thesis_summary"]
            if len(thesis) > 120:
                thesis = thesis[:117] + "..."
            parts.append(thesis)

        if ctx.get("today_actions"):
            actions = ", ".join(f"{a['action']} {a['shares']} shares" for a in ctx["today_actions"])
            parts.append(f"Today: {actions}.")

        narratives[ticker] = " ".join(parts) if parts else "No rationale data available."

    return narratives


def run(run_date: str | None = None):
    run_date = run_date or str(date.today())
    _health_start = _time.time()
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

    # Enrich positions with sector from signals.json
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    try:
        sig_data, _ = _load_signals_from_s3(signals_bucket, run_date)
        sector_lookup = {}
        for s in (sig_data.get("universe", []) + sig_data.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and s.get("sector"):
                sector_lookup[t] = s["sector"]
        for ticker in positions:
            if not positions[ticker].get("sector") and ticker in sector_lookup:
                positions[ticker]["sector"] = sector_lookup[ticker]
    except Exception as e:
        logger.warning(f"Sector enrichment failed: {e}")

    # Prior day's NAV (to compute daily return)
    prior_row = conn.execute(
        "SELECT portfolio_nav FROM eod_pnl ORDER BY date DESC LIMIT 1"
    ).fetchone()
    prior_nav = prior_row[0] if prior_row else None

    if prior_nav is None:
        logger.info("First trading day — no prior NAV, daily return unavailable")
        daily_return = None
    else:
        daily_return = ((nav - prior_nav) / prior_nav * 100)

    # SPY return for the day
    spy_price = _spy_close(run_date)
    spy_return = None
    if spy_price:
        # Try cached prior SPY close from eod_pnl first
        spy_prior_row = conn.execute(
            "SELECT spy_close FROM eod_pnl WHERE spy_close IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if spy_prior_row and spy_prior_row[0]:
            spy_return = (spy_price / spy_prior_row[0] - 1) * 100
        else:
            # Fallback: yfinance 2d download
            try:
                hist = yf.download("SPY", period="2d", progress=False, auto_adjust=True)
                if len(hist) >= 2:
                    spy_return = float((hist["Close"].values.flat[-1] / hist["Close"].values.flat[-2] - 1) * 100)
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
        "spy_close": spy_price,
    })

    # ── Sector attribution ──────────────────────────────────────────────────
    sector_attribution = {}
    if positions and nav > 0:
        for ticker, pos in positions.items():
            sector = pos.get("sector", "Unknown")
            mv = pos.get("market_value", 0)
            weight = mv / nav
            # Use unrealized PnL as proxy for daily contribution
            unrealized = pos.get("unrealized_pnl", 0)
            daily_contrib = (unrealized / nav * 100) if nav else 0
            if sector not in sector_attribution:
                sector_attribution[sector] = {"weight": 0.0, "contribution": 0.0, "positions": 0}
            sector_attribution[sector]["weight"] += weight
            sector_attribution[sector]["contribution"] += daily_contrib
            sector_attribution[sector]["positions"] += 1
        logger.info(f"Sector attribution: {sector_attribution}")

    # Export full history CSVs for dashboard consumption
    trades_df = pd.read_sql("SELECT * FROM trades ORDER BY date, created_at", conn)
    eod_df = pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn)
    s3 = boto3.client("s3")
    for df, key in [
        (trades_df, "trades/trades_full.csv"),
        (eod_df, "trades/eod_pnl.csv"),
    ]:
        try:
            buf = df.to_csv(index=False).encode()
            s3.put_object(Bucket=trades_bucket, Key=key, Body=buf)
            logger.info(f"Exported {key} ({len(df)} rows) to s3://{trades_bucket}/{key}")
        except Exception as e:
            logger.warning(f"S3 CSV export failed for {key}: {e}")

    backup_to_s3(db_path, run_date, trades_bucket)

    # Build position rationale narratives
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    position_narratives = {}
    data_warnings: list[str] = []
    try:
        if positions:
            contexts, data_warnings = _build_position_contexts(positions, conn, signals_bucket, run_date)
            position_narratives = _synthesize_rationales(contexts)
            logger.info(f"Position narratives generated for {len(position_narratives)} tickers")
    except Exception as e:
        logger.warning(f"Position rationale generation failed: {e}")

    try:
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
            position_narratives=position_narratives,
            sector_attribution=sector_attribution,
            data_warnings=data_warnings,
        )
    except Exception as e:
        logger.error(f"EOD email failed: {e}")

    # Write health status
    try:
        from executor.health_status import write_health
        write_health(
            bucket=trades_bucket,
            module_name="eod_reconcile",
            status="ok",
            run_date=run_date,
            duration_seconds=_time.time() - _health_start,
            summary={
                "nav": round(nav, 2),
                "daily_return": round(daily_return, 4) if daily_return is not None else None,
                "alpha": round(alpha, 4) if alpha is not None else None,
                "n_positions": len(positions),
            },
        )
    except Exception as _he:
        logger.warning("Health status write failed: %s", _he)

    conn.close()
    logger.info("EOD reconciliation complete")


if __name__ == "__main__":
    run()
