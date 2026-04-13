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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ssm_secrets import load_secrets
load_secrets()

from executor.eod_emailer import send_eod_email
from executor.ibkr import IBKRClient
from executor.trade_logger import (
    init_db, log_eod, backup_to_s3, get_entry_trade, get_todays_trades,
)

from executor.log_config import setup_logging
setup_logging("eod")
logger = logging.getLogger(__name__)

from executor.config_loader import CONFIG_PATH


def _spy_close(run_date: str, config: dict | None = None) -> float | None:
    """Fetch SPY closing price for run_date.

    Priority: S3 daily_closes (written by post-market data step) → polygon → yfinance.
    """
    # Try S3 daily_closes first (most reliable — written by PostMarketData step)
    try:
        import io
        bucket = (config or {}).get("trades_bucket", "alpha-engine-research")
        s3 = boto3.client("s3")
        key = f"predictor/daily_closes/{run_date}.parquet"
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        if "SPY" in df.index and "Close" in df.columns:
            close = float(df.loc["SPY", "Close"])
            logger.info("SPY close from daily_closes/%s: $%.2f", run_date, close)
            return close
    except Exception as e:
        logger.debug("daily_closes SPY lookup failed: %s", e)

    # Fallback: polygon
    try:
        from polygon_client import polygon_client
        close = polygon_client().get_single_close("SPY", run_date)
        if close is not None:
            return close
    except Exception as e:
        logger.warning("Polygon SPY fetch failed, trying yfinance: %s", e)
    # Fallback: yfinance
    try:
        import yfinance as yf
        end_date = (date.fromisoformat(run_date) + timedelta(days=1)).isoformat()
        hist = yf.download("SPY", start=run_date, end=end_date, progress=False, auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].values.flat[0])
    except Exception as e:
        logger.warning("Could not fetch SPY price: %s", e)
    return None


def _load_signals_from_s3(bucket: str, run_date: str, max_lookback: int = 5) -> tuple[dict, str | None]:
    """Load signals.json from S3, falling back to prior days.

    Research runs on Saturday and writes signals with the Saturday date,
    so the lookback must include weekends.
    """
    s3 = boto3.client("s3")
    start = date.fromisoformat(run_date)
    for days_back in range(max_lookback + 1):
        candidate = start - timedelta(days=days_back)
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
                logger.debug("Could not parse entry rationale JSON for %s", ticker)

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


def run(run_date: str | None = None) -> None:
    run_date = run_date or str(date.today())
    _health_start = _time.time()
    logger.info(f"EOD reconciliation | date={run_date}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]

    # Flow Doctor: retrieve the shared instance owned by log_config
    from executor.log_config import get_flow_doctor
    fd = get_flow_doctor()

    if not config.get("email_sender") or not config.get("email_recipients"):
        logger.warning(
            "Email not configured (email_sender/email_recipients missing from risk.yaml) "
            "— EOD email will be skipped"
        )

    conn = init_db(db_path)

    # Connect to IB Gateway with retry (transient failures at EOD are common)
    max_eod_attempts = 3
    nav = None
    positions = None
    for attempt in range(1, max_eod_attempts + 1):
        try:
            ibkr = IBKRClient(
                host=config["ibkr_host"],
                port=config["ibkr_port"],
                client_id=config["ibkr_client_id"],
            )
            account = ibkr.get_account_snapshot()
            nav = account["net_liquidation"]
            positions = ibkr.get_positions()
            ibkr.disconnect()
            break
        except Exception as e:
            if attempt == max_eod_attempts:
                logger.error(
                    "EOD: IB Gateway connection failed after %d attempts: %s",
                    max_eod_attempts, e,
                )
                if fd:
                    fd.report(e, severity="critical", context={
                        "site": "eod_ibkr_connect", "run_date": run_date,
                        "attempts": max_eod_attempts})
                raise
            wait = 30 * attempt
            logger.warning(
                "EOD: IB Gateway attempt %d/%d failed: %s — retrying in %ds",
                attempt, max_eod_attempts, e, wait,
            )
            _time.sleep(wait)

    # Enrich positions with sector — signals.json first, entry-trade fallback.
    # A missing sector is an observability failure (blank rows in sector
    # attribution), not a hard error — log loudly and continue with "Unknown".
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    try:
        sig_data, _ = _load_signals_from_s3(signals_bucket, run_date)
        sector_lookup = {}
        for s in (sig_data.get("universe", []) + sig_data.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and s.get("sector"):
                sector_lookup[t] = s["sector"]
        for ticker in positions:
            if positions[ticker].get("sector"):
                continue
            if ticker in sector_lookup:
                positions[ticker]["sector"] = sector_lookup[ticker]
                continue
            entry = get_entry_trade(conn, ticker)
            if entry and entry.get("sector"):
                positions[ticker]["sector"] = entry["sector"]
                continue
            logger.error(
                "Sector unknown for %s — missing from signals.json and entry trade. "
                "Sector attribution will be incomplete.", ticker,
            )
            positions[ticker]["sector"] = "Unknown"
    except Exception as e:
        logger.error(f"Sector enrichment failed: {e}")

    # Prior day's NAV (to compute daily return)
    prior_row = conn.execute(
        "SELECT portfolio_nav FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    prior_nav = prior_row[0] if prior_row else None

    if prior_nav is None:
        logger.info("First trading day — no prior NAV, daily return unavailable")
        daily_return = None
    else:
        daily_return = ((nav - prior_nav) / prior_nav * 100)

    # SPY return for the day
    spy_price = _spy_close(run_date, config)
    spy_return = None
    if spy_price:
        # Try cached prior SPY close from eod_pnl first
        spy_prior_row = conn.execute(
            "SELECT spy_close FROM eod_pnl WHERE spy_close IS NOT NULL AND date < ? ORDER BY date DESC LIMIT 1",
            (run_date,),
        ).fetchone()
        if spy_prior_row and spy_prior_row[0]:
            spy_return = (spy_price / spy_prior_row[0] - 1) * 100
        else:
            # Fallback: fetch SPY close for the actual prior eod_pnl date
            # (avoids period="2d" which only gets 1 day regardless of gaps)
            prior_date_row = conn.execute(
                "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
                (run_date,),
            ).fetchone()
            if prior_date_row:
                prior_spy = _spy_close(prior_date_row[0])
                if prior_spy:
                    spy_return = (spy_price / prior_spy - 1) * 100
                else:
                    logger.warning("Could not fetch SPY close for prior date %s", prior_date_row[0])
            else:
                logger.warning("No prior eod_pnl row — cannot compute SPY return")

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
        "total_cash": account.get("total_cash"),
        "accrued_interest": account.get("accrued_interest"),
        "unrealized_pnl": account.get("unrealized_pnl"),
        "realized_pnl": account.get("realized_pnl"),
    })

    # ── Load closing prices from daily_closes for accurate per-position returns ──
    closing_prices: dict[str, float] = {}
    try:
        import io as _io_dc
        dc_key = f"predictor/daily_closes/{run_date}.parquet"
        dc_obj = boto3.client("s3").get_object(Bucket=trades_bucket, Key=dc_key)
        dc_df = pd.read_parquet(_io_dc.BytesIO(dc_obj["Body"].read()))
        for ticker_idx, row in dc_df.iterrows():
            if "Close" in row and pd.notna(row["Close"]):
                closing_prices[str(ticker_idx)] = float(row["Close"])
        if closing_prices:
            logger.info("Loaded %d closing prices from daily_closes/%s", len(closing_prices), run_date)
    except Exception as _dc_exc:
        logger.debug("daily_closes not available for %s — using IB Gateway prices: %s", run_date, _dc_exc)

    # ── Per-position daily return & alpha contribution ──────────────────────
    # Look up prior day's positions_snapshot to get yesterday's price per ticker
    prior_snapshot_row = conn.execute(
        "SELECT positions_snapshot FROM eod_pnl WHERE positions_snapshot IS NOT NULL AND date < ? ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    prior_positions = {}
    if prior_snapshot_row and prior_snapshot_row[0]:
        try:
            prior_positions = json.loads(prior_snapshot_row[0])
        except (json.JSONDecodeError, TypeError):
            pass

    for ticker, pos in positions.items():
        shares = pos.get("shares", 0)
        mv = pos.get("market_value", 0)
        current_price = mv / shares if shares else 0

        # Prefer closing price from daily_closes over IB Gateway's delayed data
        if ticker in closing_prices:
            current_price = closing_prices[ticker]
            pos["market_value"] = current_price * shares
            mv = pos["market_value"]

        # Daily return: today's price vs yesterday's price (or entry price if new today)
        prior_pos = prior_positions.get(ticker)
        if prior_pos:
            prior_mv = prior_pos.get("market_value", 0)
            prior_shares = prior_pos.get("shares", 0)
            prior_price = prior_mv / prior_shares if prior_shares else 0
        else:
            # New position — use avg_cost (entry price) as the baseline
            prior_price = pos.get("avg_cost", current_price)

        if prior_price and prior_price > 0:
            pos["daily_return_pct"] = (current_price / prior_price - 1) * 100
            pos["daily_return_usd"] = (current_price - prior_price) * shares
        else:
            pos["daily_return_pct"] = 0.0
            pos["daily_return_usd"] = 0.0

        # Alpha contribution: (weight * position_return) - (weight * SPY_return)
        weight = mv / nav if nav else 0
        pos_spy = spy_return if spy_return is not None else 0
        pos["alpha_contribution_pct"] = weight * (pos["daily_return_pct"] - pos_spy)
        pos["alpha_contribution_usd"] = pos["alpha_contribution_pct"] / 100 * nav if nav else 0

    # ── Sector attribution ──────────────────────────────────────────────────
    # Daily contribution = today's per-position P&L as % of NAV (not cumulative
    # unrealized, which has no relationship to the day's return).
    sector_attribution = {}
    if positions and nav > 0:
        for ticker, pos in positions.items():
            sector = pos.get("sector", "Unknown")
            mv = pos.get("market_value", 0)
            weight = mv / nav
            daily_usd = pos.get("daily_return_usd", 0)
            daily_contrib = (daily_usd / nav * 100) if nav else 0
            if sector not in sector_attribution:
                sector_attribution[sector] = {"weight": 0.0, "contribution": 0.0, "positions": 0}
            sector_attribution[sector]["weight"] += weight
            sector_attribution[sector]["contribution"] += daily_contrib
            sector_attribution[sector]["positions"] += 1
        logger.info(f"Sector attribution: {sector_attribution}")

    # Export full history CSVs for dashboard consumption
    trades_df = pd.read_sql("SELECT * FROM trades ORDER BY date, created_at", conn)
    eod_df = pd.read_sql("SELECT * FROM eod_pnl ORDER BY date", conn)
    shadow_df = pd.read_sql("SELECT * FROM executor_shadow_book ORDER BY date, created_at", conn)
    s3 = boto3.client("s3")
    for df, key in [
        (trades_df, "trades/trades_full.csv"),
        (eod_df, "trades/eod_pnl.csv"),
        (shadow_df, "trades/shadow_book.csv"),
    ]:
        try:
            buf = df.to_csv(index=False).encode()
            s3.put_object(Bucket=trades_bucket, Key=key, Body=buf)
            logger.info(f"Exported {key} ({len(df)} rows) to s3://{trades_bucket}/{key}")
        except Exception as e:
            logger.warning(f"S3 CSV export failed for {key}: {e}")

    backup_to_s3(db_path, run_date, trades_bucket)

    # Backup daemon and executor logs to S3 (before EC2 shuts down at 1:30 PM)
    for log_file, s3_key in [
        ("/var/log/daemon.log", f"trades/logs/{run_date}/daemon.log"),
        ("/var/log/executor.log", f"trades/logs/{run_date}/executor.log"),
    ]:
        try:
            if os.path.exists(log_file):
                s3.upload_file(log_file, trades_bucket, s3_key)
                logger.info("Log backed up to s3://%s/%s", trades_bucket, s3_key)
        except Exception as e:
            logger.debug("Log backup failed for %s: %s", log_file, e)

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

    # ── Roundtrip stats (for trades with entry-exit linkage) ──────────────
    roundtrip_stats = None
    try:
        rt_row = conn.execute("""
            SELECT COUNT(*) as n,
                   AVG(realized_return_pct) as avg_ret,
                   AVG(realized_alpha_pct) as avg_alpha,
                   AVG(days_held) as avg_hold,
                   SUM(CASE WHEN realized_alpha_pct > 0 THEN 1 ELSE 0 END) as n_beat_spy
            FROM trades
            WHERE entry_trade_id IS NOT NULL
              AND realized_return_pct IS NOT NULL
        """).fetchone()
        if rt_row and rt_row[0] > 0:
            roundtrip_stats = {
                "n_roundtrips": rt_row[0],
                "avg_return_pct": round(rt_row[1], 2) if rt_row[1] else None,
                "avg_alpha_pct": round(rt_row[2], 2) if rt_row[2] else None,
                "avg_hold_days": round(rt_row[3], 1) if rt_row[3] else None,
                "n_beat_spy": rt_row[4] or 0,
                "win_rate_vs_spy": round(rt_row[4] / rt_row[0] * 100, 1) if rt_row[4] else 0,
            }
            logger.info("Roundtrip stats: %s", roundtrip_stats)
    except Exception as e:
        logger.warning("Roundtrip stats query failed: %s", e)

    # ── Execution quality monitoring ──────────────────────────────────────
    execution_quality = None
    try:
        eq_rows = conn.execute("""
            SELECT trigger_type, slippage_vs_signal, execution_latency_ms,
                   signal_price, fill_price
            FROM trades
            WHERE date = ? AND fill_price IS NOT NULL AND action = 'ENTER'
        """, (run_date,)).fetchall()
        if eq_rows:
            slippage_by_trigger: dict[str, list[float]] = {}
            all_slippage = []
            all_latency = []
            for row in eq_rows:
                trigger = row[0] or "unknown"
                slip = row[1]
                latency = row[2]
                if slip is not None:
                    slippage_by_trigger.setdefault(trigger, []).append(slip)
                    all_slippage.append(slip)
                if latency is not None:
                    all_latency.append(latency)
            execution_quality = {
                "date": run_date,
                "n_entries": len(eq_rows),
                "avg_slippage_pct": round(sum(all_slippage) / len(all_slippage), 4) if all_slippage else None,
                "avg_latency_ms": round(sum(all_latency) / len(all_latency), 0) if all_latency else None,
                "slippage_by_trigger": {
                    t: {"avg": round(sum(v) / len(v), 4), "n": len(v)}
                    for t, v in slippage_by_trigger.items()
                },
            }
            logger.info("Execution quality: %s", execution_quality)
            # Write to S3
            try:
                s3 = boto3.client("s3")
                s3.put_object(
                    Bucket=trades_bucket,
                    Key=f"trades/execution_quality/{run_date}.json",
                    Body=json.dumps(execution_quality, indent=2).encode(),
                    ContentType="application/json",
                )
            except Exception as _eq_s3:
                logger.warning("Execution quality S3 write failed: %s", _eq_s3)
    except Exception as e:
        logger.warning("Execution quality query failed: %s", e)

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
            roundtrip_stats=roundtrip_stats,
            trades_bucket=trades_bucket,
            account_snapshot=account,
        )
    except Exception as e:
        logger.error(f"EOD email failed: {e}")
        if fd:
            fd.report(e, severity="error", context={
                "site": "eod_email", "run_date": run_date})

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

    # ── Data manifest ──────────────────────────────────────────────────────
    try:
        from executor.health_status import write_data_manifest
        trades_today_count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE date=?", (run_date,)
        ).fetchone()[0]
        write_data_manifest(
            bucket=trades_bucket,
            module_name="eod_reconcile",
            run_date=run_date,
            manifest={
                "nav": round(nav, 2),
                "n_positions": len(positions),
                "daily_return_pct": round(daily_return, 4) if daily_return is not None else None,
                "spy_return_pct": round(spy_return, 4) if spy_return is not None else None,
                "alpha_pct": round(alpha, 4) if alpha is not None else None,
                "trades_today": trades_today_count,
                "roundtrip_stats": roundtrip_stats,
            },
        )
    except Exception as _me:
        logger.warning("Data manifest write failed: %s", _me)

    # ── Uptime metrics ─────────────────────────────────────────────────────
    try:
        from executor import uptime_tracker
        metrics = uptime_tracker.run(bucket=trades_bucket)
        logger.info(
            "Uptime: active=%d/%d connected=%d crashes=%d uptime=%.1f%%",
            metrics.get("active_minutes", 0),
            metrics.get("market_minutes", 0),
            metrics.get("connected_minutes", 0),
            metrics.get("crashes", 0),
            metrics.get("uptime_pct", 0) * 100,
        )
    except Exception as _ue:
        logger.warning("Uptime tracker failed: %s", _ue)

    if fd:
        fd.log_summary(logger)
    conn.close()
    logger.info("EOD reconciliation complete")


if __name__ == "__main__":
    run()
