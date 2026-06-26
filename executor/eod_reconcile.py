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

from executor import reference_rate
from executor.eod_emailer import send_eod_email
from executor.eod_report import build_eod_report, write_eod_report
from executor.trade_logger import (
    init_db, log_eod, backup_to_s3, get_entry_trade, get_todays_trades,
)

from nousergon_lib.dates import now_dual
from nousergon_lib.trading_calendar import previous_trading_day
from nousergon_lib.logging import setup_logging, guard_entrypoint
# See executor/main.py for the rationale on IB Error 10197 / 10349 suppression.
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [r"Error 10197", r"Error 10349"]
_FLOW_DOCTOR_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow-doctor.yaml")
setup_logging("eod", flow_doctor_yaml=_FLOW_DOCTOR_YAML, exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS)
logger = logging.getLogger(__name__)

from executor.config_loader import load_config


def _compute_unattributed_residual_pct(
    unattributed_usd: float | None,
    nav: float | None,
) -> float | None:
    """Headline metric for the *P&L attribution* row of the Phase 2
    transparency-inventory: ``unattributed_usd / portfolio_nav × 100``.

    Returns None when either input is None or when nav is zero/falsy
    (a divide-by-zero protection — a NAV of 0 means we can't compute a
    meaningful residual % regardless of the dollar amount).

    The inventory gate is ≤1%. Sign is preserved — a negative value
    means position-pnl + interest exceeded the actual NAV change
    (typically an unaccounted fee). Consumers should compare on
    absolute value when alarming.
    """
    if unattributed_usd is None or not nav:
        return None
    return (unattributed_usd / nav) * 100.0


def _spy_close(run_date: str, config: dict | None = None) -> float:
    """Fetch SPY close for run_date from ArcticDB macro library.

    SPY lives in the `macro` library (per alpha-engine-data's
    daily_append writer), NOT the `universe` library. Reading from
    universe was a bug: universe has only the per-ticker watchlist
    symbols, no index ETFs.

    ArcticDB is the single source of truth — no parquet, polygon, or
    yfinance fallback. Hard-fails if SPY is missing, stale, or has no
    close for run_date, because EOD alpha is meaningless without a
    reliable SPY reference.
    """
    from executor.price_cache import _open_macro_library
    bucket = (config or {}).get("trades_bucket", "alpha-engine-research")
    macro = _open_macro_library(bucket)
    try:
        df = macro.read("SPY").data
    except Exception as e:
        raise RuntimeError(f"ArcticDB read failed for SPY: {e}") from e
    if df.empty or "Close" not in df.columns:
        raise RuntimeError("ArcticDB SPY frame empty or missing Close column")
    target = pd.Timestamp(run_date).normalize()
    idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
    matches = df[idx == target]
    if matches.empty:
        raise RuntimeError(
            f"ArcticDB has no SPY close for {run_date} (latest: "
            f"{pd.Timestamp(df.index[-1]).date()})"
        )
    close = float(matches["Close"].iloc[-1])
    logger.info("[data_source=arcticdb] SPY close for %s: $%.2f", run_date, close)
    return close


def _load_signals_from_s3(bucket: str, run_date: str, max_lookback: int = 14) -> tuple[dict, str | None]:
    """Load signals.json from S3, falling back to prior days.

    Research runs weekly (Saturday) and writes signals with the trading-day
    date, so the lookback must span weekends + a missed-cycle buffer. The
    14-day default matches the morning planner's
    ``read_signals_with_fallback`` window — EOD reconciliation should use the
    same last-good signals the planner traded on, not a tighter window. (The
    prior 5-day window paged a misleading flow-doctor ERROR on 2026-06-25
    when the only staleness was a missed Saturday cycle the planner had
    already tolerated.) Whether the required signals.json was actually
    PRODUCED is owned by the central artifact-freshness monitor
    (research_signals in ARTIFACT_REGISTRY.yaml), not this consumer.
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
    # WARNING not ERROR: EOD reconcile degrades gracefully on absent signals
    # (returns {} + a warning the caller surfaces in the EOD report). The
    # operator-paging "signals.json was not produced" alert is owned by the
    # central artifact-freshness monitor (research_signals), so this consumer
    # must not also page flow-doctor for the same upstream condition.
    logger.warning("No signals found within %d days of %s", max_lookback, run_date)
    return {}, f"Signals unavailable from S3 for {run_date} (checked {max_lookback} days back)"


def _load_constituents_sector_map(bucket: str) -> dict[str, str]:
    """Return the latest S&P 500+400 ticker→GICS sector map from S3.

    Reads ``market_data/weekly/{YYYY-MM-DD}/constituents.json`` written by
    the alpha-engine-data weekly collector. Used as the final sector
    lookup fallback in EOD reconcile — catches legacy/fractional-share
    positions whose ticker isn't in today's research universe or whose
    entry_trade row predates reliable sector population.

    Returns an empty dict on miss so the caller can fall through to the
    "Unknown" sentinel without raising.
    """
    s3 = boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="market_data/weekly/")
        keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith("/constituents.json")
        ]
        if not keys:
            logger.warning("No constituents.json under market_data/weekly/ in %s", bucket)
            return {}
        latest = max(keys)
        obj = s3.get_object(Bucket=bucket, Key=latest)
        data = json.loads(obj["Body"].read())
        sector_map = data.get("sector_map", {}) or {}
        logger.info(
            "Loaded sector_map from %s (%d tickers)", latest, len(sector_map),
        )
        return sector_map
    except Exception as e:
        # WARNING not ERROR: graceful fallthrough to the "Unknown" sentinel
        # (return {}). Upstream constituents.json freshness is owned by the
        # central artifact-freshness monitor, not this best-effort lookup.
        logger.warning("Failed to load constituents sector_map: %s", e)
        return {}


# Broad-market index/ETF core positions are not GICS sector constituents.
# Since the portfolio-optimizer cutover (use_portfolio_optimizer: true,
# 2026-05-13) SPY is held as the enhanced-index core position. SPY has no
# `sector` field in signals.json, no entry_trade.sector, and is not in the
# S&P 500+400 constituents map, so the normal lookup chain misses it and it
# renders as a bare "—"/"Unknown" on the public site — reads as missing data
# rather than "this is the broad-market core." Tag it explicitly so every
# downstream consumer (public site, private console, sector attribution)
# inherits a meaningful label. New core ETFs the optimizer may substitute
# get added to ``_INDEX_ETF_TICKERS``.
_INDEX_ETF_SECTOR = "Broad Market / Index"
_INDEX_ETF_TICKERS = frozenset({"SPY", "VOO", "IVV", "SPLG"})


def _index_etf_sector(ticker: str) -> str | None:
    """Return the broad-market sector label for index/ETF core positions.

    Returns ``"Broad Market / Index"`` for known broad-market ETFs held as
    the enhanced-index core (SPY and S&P 500 trackers the optimizer may
    substitute), else ``None`` so the caller falls through to the normal
    signals.json / entry-trade / S&P-constituents lookup chain.
    """
    return _INDEX_ETF_SECTOR if ticker in _INDEX_ETF_TICKERS else None


def _load_predictions_from_s3(bucket: str) -> tuple[dict, str | None]:
    """Load latest predictions from S3. Returns ({ticker: pred_dict}, warning_msg) on failure."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="predictor/predictions/latest.json")
        data = json.loads(obj["Body"].read())
        return {p["ticker"]: p for p in data.get("predictions", []) if "ticker" in p}, None
    except Exception as e:
        # WARNING not ERROR: EOD reconcile degrades gracefully without
        # predictions (returns {} + a warning the caller surfaces). Upstream
        # predictions.json freshness is owned by the central artifact-
        # freshness monitor (predictor_predictions), not this consumer.
        logger.warning("Failed to load predictions from S3: %s", e)
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
    """Build per-position rationales from the context dict for the EOD email.

    **Zero LLM exposure — hard architectural guardrail.** Per
    ``[[preference_llm_calls_confined_to_research_module]]``, executor
    never invokes an LLM. Trading execution is the most operationally
    critical surface; introducing any external-API dependency (even
    gated) couples trading-day reliability to upstream availability.
    If a future surface genuinely needs LLM-synthesized prose, the call
    goes in research and produces a frozen artifact executor reads.

    Output is mechanical synthesis from the context dict: entry / score
    / GBM prediction / thesis summary / today's actions, joined with
    spaces. Same dict-shape the EOD emailer has always consumed.

    History: an Anthropic-Haiku-backed synthesis path shipped
    2026-03-17 (commit 58dcb9b) and ran for ~10 weeks before being
    nuked outright 2026-05-25 after Brian flagged the unmandated LLM
    exposure. Cost-telemetry wiring + WARN-above-ceiling instrumentation
    + an opt-in kill-switch flag (PRs #210/#211/#212 same session) all
    went with it — the rule is "no LLM in executor, period" rather than
    "LLM with substrate."
    """
    if not contexts:
        return {}

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


def _resolve_prior_price(
    prior_pos: dict | None,
    pos: dict,
    current_price: float,
) -> float:
    """Pick the right prior-day price for daily return computation.

    Phase 3+ snapshots store an explicit `closing_price` from daily_closes,
    which is the same source today's reconcile uses for current_price —
    eliminating the IB-MV-vs-daily-closes mismatch that was dumping noise
    into the cash residual. Falls back to MV/shares for legacy snapshots
    and to avg_cost for positions opened today.
    """
    if prior_pos:
        cp = prior_pos.get("closing_price")
        if cp is not None:
            return float(cp)
        prior_mv = prior_pos.get("market_value", 0)
        prior_shares = prior_pos.get("shares", 0)
        if prior_shares:
            return prior_mv / prior_shares
    # No prior snapshot — position opened today, use avg_cost
    return pos.get("avg_cost", current_price)


def _compute_daily_return(
    ticker: str,
    pos: dict,
    prior_pos: dict | None,
    current_price: float,
    shares: float,
    prior_close: float | None,
    prior_close_date: "date | None",
    expected_prev_td: "date",
) -> tuple[float, float, float | None, str | None]:
    """Gap-aware per-position daily return.

    The prior-day baseline for a position HELD on the previous trading day
    must be the close on that previous trading day, read authoritatively
    from ArcticDB (``prior_close`` at ``prior_close_date``) — NOT the stored
    snapshot's ``closing_price``, which can be days stale whenever a
    weekday/EOD Step Function was skipped. In the normal case the two are
    identical (the snapshot persists the same daily_closes-sourced close),
    so behavior is unchanged; the difference only bites across a gap.

    Concrete bug this closes: the 2026-06-24 SF halt left no 06-24
    snapshot, so the 06-25 reconcile picked the 06-23 snapshot as
    "yesterday" and reported RGEN at ``145.23/126.37-1 = +14.92%`` — a
    two-session move mislabeled as one day, which drove the entire
    headline (config#1228).

    Returns ``(daily_return_pct, daily_return_usd, prior_price, na_reason)``:
      * held-through, ArcticDB's prior row IS the previous trading day →
        true one-session return; ``na_reason`` is None.
      * held-through but ArcticDB's latest prior row predates the previous
        trading day (market-data gap not yet healed) → ``(0.0, 0.0, None,
        reason)``: we refuse to compute a return against a stale baseline,
        flag it N/A, and let the position's P&L surface in the NAV
        unattributed bucket rather than fabricate an inflated number.
      * opened since the prior trading day (absent from the prior snapshot)
        → return vs entry ``avg_cost`` (unchanged legacy behavior).
    """
    held_through = prior_pos is not None
    if not held_through:
        # Opened today / during a gap — baseline is the entry price.
        prior_price = pos.get("avg_cost", current_price)
    elif prior_close is not None and prior_close_date == expected_prev_td:
        # Authoritative: the previous trading day's close from ArcticDB.
        prior_price = float(prior_close)
    elif (
        prior_close is not None
        and prior_close_date is not None
        and prior_close_date < expected_prev_td
    ):
        # ArcticDB's latest prior row predates the previous trading day — a
        # weekday/EOD SF was skipped and the market-data gap is not yet
        # healed. Do NOT report a multi-session move as a one-day return.
        reason = (
            f"{ticker}: previous-trading-day ({expected_prev_td}) close unavailable "
            f"in ArcticDB (latest prior row = {prior_close_date}). Daily return marked "
            f"N/A rather than computed against a stale baseline; the position's P&L "
            f"surfaces in the NAV unattributed bucket. Heal the market-data gap "
            f"(config#1228)."
        )
        return 0.0, 0.0, None, reason
    else:
        # No ArcticDB prior close at all (e.g. brand-new listing) — fall back
        # to the legacy snapshot/avg_cost resolution.
        prior_price = _resolve_prior_price(prior_pos, pos, current_price)

    if prior_price and prior_price > 0:
        return (
            (current_price / prior_price - 1) * 100,
            (current_price - prior_price) * shares,
            prior_price,
            None,
        )
    return 0.0, 0.0, None, None


def _apply_dividend_delta(
    pos: dict,
    prior_pos: dict | None,
    prior_price: float,
    shares: int,
) -> None:
    """Attribute today's dividend accrual to the position.

    Only positive accrual deltas (ex-dividend earnings) are added to
    daily_return_usd — these represent new economic value earned today.

    Negative deltas (accrual → cash reclassification on payout day) are
    recorded in pos['dividend_paid_usd'] but NOT subtracted from position
    P&L. The dividend was already earned on ex-dividend day; the payout
    is a bookkeeping transfer that raises cash without changing portfolio
    value. The reconciliation bucket uses dividend_paid_usd to explain
    the cash inflow on the payout day.
    """
    today_div = float(pos.get("accrued_dividend", 0.0) or 0.0)
    prior_div = float((prior_pos or {}).get("accrued_dividend", 0.0) or 0.0)
    div_delta = today_div - prior_div
    if div_delta > 0:
        pos["dividend_usd"] = div_delta
        pos["daily_return_usd"] = pos.get("daily_return_usd", 0.0) + div_delta
        prior_mv = prior_price * shares if prior_price else 0
        if prior_mv > 0:
            pos["daily_return_pct"] = (pos["daily_return_usd"] / prior_mv) * 100
    elif div_delta < 0:
        # Accrual dropped — payout to cash. Don't double-count as position loss.
        pos["dividend_paid_usd"] = -div_delta


def run(
    run_date: str | None = None,
    *,
    send_email: bool = True,
    run_audit: bool = True,
) -> None:
    """Reconcile EOD P&L for ``run_date`` against the settled ArcticDB closes.

    ``send_email``: when False, the outbound EOD email is suppressed. A
    re-reconcile / correction pass (``reconcile_audit``) re-runs this for a
    PAST date to fix a value frozen pre-settlement — it must re-emit the
    ``eod_report.json`` artifact (kept) but must NOT resend that day's email.

    ``run_audit``: when True (the live daily run), the trailing-window
    ``reconcile_audit`` self-heal pass fires at the end. The audit pass itself
    calls ``run(..., run_audit=False)`` so the re-reconcile can't recurse.
    """
    today_trading_day = now_dual().trading_day
    if run_date is None:
        run_date = today_trading_day
        logger.info(
            "EOD reconciliation | date=%s (resolved from now_dual().trading_day)",
            run_date,
        )
    else:
        logger.info("EOD reconciliation | date=%s (explicit)", run_date)
    # Previous NYSE trading day — the baseline every "daily" figure must be
    # measured against. Used to detect skipped-SF gaps (config#1228).
    expected_prev_td = previous_trading_day(date.fromisoformat(run_date))
    # data_warnings is populated through the run (gap flags, NAV residual)
    # and surfaced in the EOD email + report artifact.
    data_warnings: list[str] = []
    _health_start = _time.time()

    config = load_config()

    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]

    # Preflight: AWS_REGION + S3 bucket reachable. Fail fast so a
    # misconfigured env surfaces immediately instead of deeper down.
    from executor.preflight import ExecutorPreflight
    ExecutorPreflight(bucket=trades_bucket, mode="eod").run()

    # Flow Doctor: retrieve the shared instance set up at module import
    from nousergon_lib.logging import get_flow_doctor
    fd = get_flow_doctor()

    if not config.get("email_sender") or not config.get("email_recipients"):
        logger.warning(
            "Email not configured (email_sender/email_recipients missing from risk.yaml) "
            "— EOD email will be skipped"
        )

    conn = init_db(db_path)

    # Load EOD state from S3 snapshot keyed by run_date.
    #
    # 2026-04-28 (Phase 2 of EOD-SF cutover): replaced the live IB-read
    # block (`get_account_snapshot` + `get_positions` +
    # `get_accrued_dividends_by_symbol`) with a snapshot read. The
    # snapshot is written by `executor/snapshot_capturer.py` running as
    # the SF's `CaptureSnapshot` step before this step. The snapshot
    # decouples capture from reconciliation — the row keyed by
    # `run_date=X` is now built from observations made at time X, not
    # from now-as-of state at write-time. PR #116's `run_date != today`
    # hard-block is no longer needed: the snapshot-existence check is
    # the new contract, and snapshot existence is what makes the run
    # safe (today, last Tuesday, or any other date with a snapshot).
    from executor.snapshot_capturer import load_snapshot
    snapshot = load_snapshot(
        bucket=trades_bucket,
        run_date=run_date,
        region=config.get("aws_region", "us-east-1"),
    )
    if snapshot is None:
        msg = (
            f"No snapshot at s3://{trades_bucket}/trades/snapshots/{run_date}.json — "
            f"`executor/snapshot_capturer.py` must run before "
            f"`executor/eod_reconcile.py` so the row keyed by run_date={run_date!r} "
            f"sources its inputs from observations made at time {run_date!r} "
            f"(not from now-as-of IB state). The CaptureSnapshot SF step is the "
            f"canonical writer; for manual recovery, run "
            f"`python executor/snapshot_capturer.py --date {run_date}` while IB "
            f"Gateway is up on ae-trading."
        )
        if fd:
            fd.report(
                RuntimeError(msg),
                severity="critical",
                context={"site": "eod_load_snapshot", "run_date": run_date},
            )
        raise RuntimeError(msg)

    account = snapshot["account"]
    nav = account["net_liquidation"]
    positions = snapshot["positions"]
    dividends_by_symbol = snapshot.get("accrued_dividends", {})
    for _tkr, _accrued in dividends_by_symbol.items():
        if _tkr in positions:
            positions[_tkr]["accrued_dividend"] = _accrued
    logger.info(
        "EOD: snapshot loaded | NAV=$%.2f positions=%d dividends=%d captured_at=%s",
        nav,
        len(positions),
        len(dividends_by_symbol),
        snapshot.get("captured_at"),
    )

    # Enrich positions with sector. Lookup chain:
    #   0. index/ETF core (SPY etc.) → "Broad Market / Index" — not a GICS
    #      constituent, so it must short-circuit before the sector lookups
    #      (an index ETF must never be mislabeled with a sector even if it
    #      somehow appears in a lookup table).
    #   1. signals.json today (universe + buy_candidates)
    #   2. trades.db entry_trade.sector
    #   3. S&P 500+400 constituents.json (latest weekly snapshot) — catches
    #      legacy/fractional-share positions whose ticker has fallen out of
    #      today's research universe (e.g. dividend-reinvestment remnants).
    # A missing sector is an observability failure (blank rows in sector
    # attribution), not a hard error — log loudly and continue with "Unknown"
    # only when all sources miss.
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    try:
        sig_data, _ = _load_signals_from_s3(signals_bucket, run_date)
        sector_lookup = {}
        for s in (sig_data.get("universe", []) + sig_data.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and s.get("sector"):
                sector_lookup[t] = s["sector"]
        constituents_lookup: dict[str, str] | None = None
        for ticker in positions:
            if positions[ticker].get("sector"):
                continue
            etf_sector = _index_etf_sector(ticker)
            if etf_sector:
                positions[ticker]["sector"] = etf_sector
                continue
            if ticker in sector_lookup:
                positions[ticker]["sector"] = sector_lookup[ticker]
                continue
            entry = get_entry_trade(conn, ticker)
            if entry and entry.get("sector"):
                positions[ticker]["sector"] = entry["sector"]
                continue
            if constituents_lookup is None:
                constituents_lookup = _load_constituents_sector_map(signals_bucket)
            if ticker in constituents_lookup:
                positions[ticker]["sector"] = constituents_lookup[ticker]
                continue
            logger.error(
                "Sector unknown for %s — missing from signals.json, entry trade, "
                "and S&P 500+400 constituents. Sector attribution will be incomplete.",
                ticker,
            )
            positions[ticker]["sector"] = "Unknown"
    except Exception as e:
        logger.error(f"Sector enrichment failed: {e}")

    # Prior day's NAV (to compute daily return). Also capture its DATE so we
    # can detect when it is not the previous trading day — i.e. an eod_pnl row
    # is missing because a weekday/EOD SF was skipped, which makes the
    # headline NAV daily return span multiple sessions.
    prior_row = conn.execute(
        "SELECT date, portfolio_nav FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    prior_eod_date = (
        date.fromisoformat(prior_row[0]) if prior_row and prior_row[0] else None
    )
    prior_nav = prior_row[1] if prior_row else None

    if prior_nav is None:
        logger.info("First trading day — no prior NAV, daily return unavailable")
        daily_return = None
    else:
        daily_return = ((nav - prior_nav) / prior_nav * 100)

    # Headline gap guard: if the prior eod_pnl row is not the previous trading
    # day, the NAV-level daily return / alpha span more than one session. Per-
    # position returns are gap-corrected from ArcticDB, but the NAV baseline
    # stays stale until the missing row is backfilled (config#1229 / Phase 2).
    if prior_nav is not None and prior_eod_date is not None and prior_eod_date != expected_prev_td:
        hdr_warn = (
            f"Headline daily return/alpha span multiple sessions: prior eod_pnl row is "
            f"{prior_eod_date} but the previous trading day is {expected_prev_td} (an "
            f"eod_pnl row is missing — a weekday/EOD SF was skipped). Per-position "
            f"returns are gap-corrected from ArcticDB; the NAV-level baseline is stale "
            f"until the missing row is backfilled (config#1229)."
        )
        logger.warning(hdr_warn)
        data_warnings.append(hdr_warn)

    # SPY return for the day.
    #
    # Both legs are read from the SETTLED ArcticDB macro source — the prior
    # leg is NOT taken from the stored ``eod_pnl.spy_close`` (config#1276).
    # Freezing the prior close meant that any day whose stored close was a
    # pre-settlement value (captured at same-day ~4:20pm ET, before the
    # official close lands in ArcticDB) silently corrupted the NEXT day's
    # spy_return as the denominator — and never self-healed when ArcticDB
    # later corrected. Windowing stays gap-consistent: we still span to the
    # prior *eod_pnl* DATE (the same baseline ``prior_nav`` uses, so SPY and
    # the portfolio measure the same interval across a skipped session), but
    # the close VALUE is always the authoritative settled close for that date.
    spy_price = _spy_close(run_date, config)
    spy_return = None
    if spy_price:
        prior_date_row = conn.execute(
            "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
            (run_date,),
        ).fetchone()
        if prior_date_row:
            prior_spy = _spy_close(prior_date_row[0], config)
            if prior_spy:
                spy_return = (spy_price / prior_spy - 1) * 100
            else:
                logger.warning("Could not fetch settled SPY close for prior date %s", prior_date_row[0])
        else:
            logger.warning("No prior eod_pnl row — cannot compute SPY return")

    alpha = (daily_return - spy_return) if (daily_return is not None and spy_return is not None) else None

    # Same-day EOD reads run_date's SPY close from ArcticDB at ~4:20pm ET,
    # which can still be pre-settlement. Mark the artifact provisional so the
    # console flags it and the T+1 reconcile_audit pass re-finalizes it from
    # the settled close. A re-reconcile of a PAST date (run_date earlier than
    # today's trading_day) is by definition post-settlement → final.
    spy_close_provisional = run_date == today_trading_day

    logger.info(
        f"NAV=${nav:,.2f} | daily={daily_return:.2f}% | "
        f"SPY={spy_return:.2f}% | alpha={alpha:.2f}%"
        if all(x is not None for x in [daily_return, spy_return, alpha])
        else f"NAV=${nav:,.2f} | prior_nav={prior_nav}"
    )

    # ── Load closing prices from ArcticDB ──────────────────────────────────
    # Hard-fails on any miss: EOD reconcile must reconcile against an
    # authoritative price source, not IB Gateway's delayed intraday data.
    #
    # Macro-routed held positions (sector ETFs / VIX / TNX / etc.) live in
    # the `macro` library, NOT `universe`. The portfolio-optimizer cutover
    # (2026-05-13) made SPY a held core position; its first EOD on
    # 2026-05-14 raised NoSuchVersionException because reconcile was
    # universe-only. SPY-as-held is now read from `universe` directly
    # (alpha-engine-data #245 lifted SPY to a full universe member via
    # `_UNIVERSE_EXTRA`); only the remaining macro-only-Close symbols still
    # need the macro-lib dispatch. Mirror `price_cache.load_price_histories`
    # (executor/price_cache.py, `_MACRO_SYMBOLS`).
    from executor.price_cache import (
        _open_universe_library,
        _open_macro_library,
        _MACRO_SYMBOLS,
    )
    universe_lib = _open_universe_library(trades_bucket)
    macro_lib = None  # lazy-open only if a macro-routed held ticker appears
    target_ts = pd.Timestamp(run_date).normalize()
    closing_prices: dict[str, float] = {}
    # Authoritative prior-day baseline: the last ArcticDB row strictly before
    # run_date, with its date, so daily returns are measured against the real
    # previous trading day rather than a possibly-stale snapshot (config#1228).
    prior_closes: dict[str, float] = {}
    prior_close_dates: dict[str, date] = {}
    missing: list[str] = []
    for ticker in positions.keys():
        if ticker in _MACRO_SYMBOLS:
            if macro_lib is None:
                macro_lib = _open_macro_library(trades_bucket)
            lib = macro_lib
        else:
            lib = universe_lib
        try:
            df = lib.read(ticker).data
        except Exception as e:
            missing.append(f"{ticker} ({e.__class__.__name__})")
            continue
        if df.empty or "Close" not in df.columns:
            missing.append(f"{ticker} (no Close column)")
            continue
        idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        match = df[idx == target_ts]
        if match.empty:
            missing.append(f"{ticker} (no row for {run_date})")
            continue
        closing_prices[ticker] = float(match["Close"].iloc[-1])
        # Capture the previous available close (last row strictly before
        # run_date) + its date — the gap-aware daily-return baseline.
        prior_mask = idx < target_ts
        if prior_mask.any():
            prior_rows = df[prior_mask]
            prior_closes[ticker] = float(prior_rows["Close"].iloc[-1])
            prior_close_dates[ticker] = (
                pd.Timestamp(prior_rows.index[-1]).normalize().date()
            )
    if missing:
        raise RuntimeError(
            f"ArcticDB closing-price lookup failed for {len(missing)} "
            f"held ticker(s) on {run_date}: {missing}. EOD reconcile cannot "
            "proceed without authoritative closes."
        )
    logger.info(
        "[data_source=arcticdb] Loaded closing prices for %d/%d held tickers on %s",
        len(closing_prices), len(positions), run_date,
    )

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
        # Persist the canonical close so tomorrow's reconcile reads the same
        # source for prior_price (not derived from possibly-stale IB MV).
        pos["closing_price"] = current_price

        # Daily return — gap-aware. Held-through positions price against the
        # previous TRADING day's ArcticDB close (config#1228); a stale
        # snapshot baseline previously inflated returns across a skipped-SF
        # gap (RGEN +14.92% on 2026-06-25 vs the 06-23, not 06-24, close).
        prior_pos = prior_positions.get(ticker)
        daily_pct, daily_usd, prior_price, na_reason = _compute_daily_return(
            ticker, pos, prior_pos, current_price, shares,
            prior_closes.get(ticker), prior_close_dates.get(ticker),
            expected_prev_td,
        )
        pos["daily_return_pct"] = daily_pct
        pos["daily_return_usd"] = daily_usd
        if na_reason:
            # Fail loud: the figure is an explicit N/A, not a silent zero.
            pos["daily_return_na"] = True
            pos["daily_return_na_reason"] = na_reason
            logger.warning("Daily-return N/A | %s", na_reason)
            data_warnings.append(na_reason)

        # Dividend attribution: today's accrued dividend for this ticker vs
        # yesterday's snapshot. Delta is the day's dividend income (or its
        # reversal when paid to cash). Flows into position α instead of
        # leaking into the cash residual. Skipped when the baseline is N/A
        # (no valid prior price to express the accrual against).
        if prior_price is not None:
            _apply_dividend_delta(pos, prior_pos, prior_price, shares)

        # Alpha contribution: (weight * position_return) - (weight * SPY_return)
        weight = mv / nav if nav else 0
        pos_spy = spy_return if spy_return is not None else 0
        pos["alpha_contribution_pct"] = weight * (pos["daily_return_pct"] - pos_spy)
        pos["alpha_contribution_usd"] = pos["alpha_contribution_pct"] / 100 * nav if nav else 0

    # data_warnings was initialized at the top of run() and accumulates gap
    # flags (per-position N/A, headline multi-session) plus the NAV residual
    # appended below; it is also extended by _build_position_contexts.

    # ── NAV change reconciliation ───────────────────────────────────────────
    # Every dollar of NAV change must be attributable to a source: position
    # MTM, interest, dividends, or (flagged) unattributed. Anything in the
    # unattributed bucket indicates a pricing/snapshot mismatch, fee, FX,
    # corporate action, or similar — surface it loudly instead of burying it
    # in cash return.
    nav_reconciliation: dict = {}
    if prior_nav is not None:
        total_nav_change = nav - prior_nav
        total_day_usd = sum(p.get("daily_return_usd", 0) for p in positions.values())

        # Interest: day-over-day delta in IB's AccruedCash
        prior_accrued_row = conn.execute(
            "SELECT accrued_interest FROM eod_pnl WHERE accrued_interest IS NOT NULL AND date < ? ORDER BY date DESC LIMIT 1",
            (run_date,),
        ).fetchone()
        today_accrued = account.get("accrued_interest")
        if today_accrued is not None and prior_accrued_row and prior_accrued_row[0] is not None:
            interest_usd = float(today_accrued) - float(prior_accrued_row[0])
        else:
            interest_usd = 0.0

        # Dividends earned today (accrual increase) are already added into
        # each position's daily_return_usd, so they flow through total_day_usd.
        # Payout-day cash inflow is exactly offset by accrual drop in IB's
        # NetLiquidation, so NAV doesn't move from the payout itself — no
        # reconciliation term needed. dividend_usd here is informational
        # only, summing positive accrual deltas for the email.
        dividend_usd = sum(p.get("dividend_usd", 0.0) for p in positions.values())

        unattributed_usd = total_nav_change - total_day_usd - interest_usd
        nav_reconciliation = {
            "nav_change_usd": total_nav_change,
            "position_pnl_usd": total_day_usd,
            "interest_usd": interest_usd,
            "dividend_usd": dividend_usd,
            "unattributed_usd": unattributed_usd,
        }
        logger.info(
            "NAV recon: Δ=$%.0f | positions=$%.0f | interest=$%.0f | "
            "dividends=$%.0f | unattributed=$%.0f",
            total_nav_change, total_day_usd, interest_usd,
            dividend_usd, unattributed_usd,
        )
        # Warn if unattributed is material (> max($100, 0.05% NAV)). Surface
        # the gap in data_warnings so it appears in the EOD email, not only
        # in server logs.
        if nav and abs(unattributed_usd) > max(100.0, 0.0005 * nav):
            msg = (
                f"NAV reconciliation gap: ${unattributed_usd:+,.0f} unattributed "
                f"({unattributed_usd / nav * 100:+.3f}% of NAV). Likely causes: "
                "stale prior-day prices, untracked corporate action, fees, or FX."
            )
            logger.warning(msg)
            data_warnings.append(msg)

    # ── Daemon-vs-IB reconciliation-integrity audit (config#859) ──
    # Secondary observability hung off the primary EOD path: a failure here
    # must NOT abort EOD reconcile (the NAV log + email are the primary
    # deliverable), and the failure IS recorded — (a) swallowed: audit
    # build/write error; (c) recording surface: the WARN below + the
    # report-card reconciliation_integrity component shows N/A when the
    # artifact is absent. Per the feedback_no_silent_fails secondary-
    # observability carve-out.
    try:
        from executor.reconciliation_audit import (
            build_reconciliation_audit,
            write_reconciliation_audit,
        )

        _recon_audit = build_reconciliation_audit(
            conn,
            today_positions=positions,
            prior_positions=prior_positions,
            run_date=run_date,
            ib_nav=nav,
        )
        _recon_key = write_reconciliation_audit(
            _recon_audit,
            bucket=trades_bucket,
            run_date=run_date,
            region=config.get("aws_region", "us-east-1"),
        )
        logger.info(
            "[reconciliation_audit] match_rate=%.3f status=%s positions=%d "
            "mismatched=%d -> s3://%s/%s",
            _recon_audit["reconciliation_match_rate"], _recon_audit["status"],
            _recon_audit["n_positions"], _recon_audit["n_mismatched"],
            trades_bucket, _recon_key,
        )
    except Exception as _recon_err:  # noqa: BLE001 — secondary observability (see comment above)
        logger.warning(
            "[reconciliation_audit] FAILED to build/write reconciliation "
            "audit for run_date=%s: %s (report card reconciliation_integrity "
            "shows N/A this cycle)", run_date, _recon_err,
        )

    # Persist EOD snapshot AFTER positions are enriched with closing prices,
    # accrued dividends, and per-position returns. Yesterday's reconcile now
    # reads this snapshot via closing_price (same source as today's
    # daily_closes), closing the source-mismatch gap that was causing NAV
    # residuals to land in cash.
    #
    # Phase 2 transparency-inventory: persist the NAV-reconciliation
    # waterfall (nav_change / position_pnl / interest / dividend /
    # unattributed) and the headline residual % as named fields. Closes
    # the *P&L attribution* row in the gate checklist — until now these
    # values existed in logs + the email body but weren't queryable from
    # eod_pnl.csv. nav_reconciliation can be {} when prior_nav is None
    # (first-ever EOD run); .get() defaults to None for those columns.
    unattributed_for_log = nav_reconciliation.get("unattributed_usd")
    unattributed_pct_for_log = _compute_unattributed_residual_pct(
        unattributed_for_log, nav,
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
        "nav_change_usd": nav_reconciliation.get("nav_change_usd"),
        "position_pnl_usd": nav_reconciliation.get("position_pnl_usd"),
        "interest_usd": nav_reconciliation.get("interest_usd"),
        "dividend_usd": nav_reconciliation.get("dividend_usd"),
        "unattributed_usd": unattributed_for_log,
        "unattributed_residual_pct": unattributed_pct_for_log,
    })

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

    # ── Reference-rate showcase artifact (metron/reference_rate.json) ─────────
    # Publish the illustrative-only Reference Rate contract artifact Metron renders
    # as a demo portfolio. Best-effort: it is secondary observability hung off the
    # already-committed eod_pnl + S3 CSV exports (recording surface = this WARN), so
    # a publish failure must never override the EOD run's primary deliverables.
    try:
        ref_payload = reference_rate.build_payload(
            positions=positions,
            nav=nav,
            nav_history=reference_rate.nav_history_from_eod_df(eod_df),
            run_date=run_date,
        )
        reference_rate.publish(s3, trades_bucket, ref_payload)
    except Exception as e:  # noqa: BLE001 — best-effort secondary path; never fatal
        logger.warning("Reference-rate artifact publish failed (non-fatal): %s", e)

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

    # Build position rationale narratives — mechanical synthesis from
    # the context dict. No LLM exposure in executor per
    # [[preference_llm_calls_confined_to_research_module]].
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    position_narratives = {}
    try:
        if positions:
            contexts, ctx_warnings = _build_position_contexts(positions, conn, signals_bucket, run_date)
            position_narratives = _synthesize_rationales(contexts)
            logger.info(
                f"Position narratives generated for {len(position_narratives)} tickers"
            )
            data_warnings.extend(ctx_warnings)
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

    # ── Build + write the structured EOD report artifact ──────────────────
    # consolidated/{date}/eod_report.json is the single source of truth for
    # the console EOD Report page. The alpha attribution here is the
    # prior-NAV-basis decomposition that ties to the headline alpha exactly
    # (executor/eod_report.py) — it replaces the old emailer's sign-flipping
    # "α % of Total" column and the positions-table total that never
    # reconciled with the NAV-based headline.
    try:
        report = build_eod_report(
            run_date=run_date,
            nav=nav,
            prior_nav=prior_nav,
            daily_return=daily_return,
            spy_return=spy_return,
            alpha=alpha,
            positions=positions,
            prior_positions=prior_positions,
            conn=conn,
            account_snapshot=account,
            nav_reconciliation=nav_reconciliation,
            position_narratives=position_narratives,
            sector_attribution=sector_attribution,
            roundtrip_stats=roundtrip_stats,
            data_warnings=data_warnings,
            generated_at=snapshot.get("captured_at"),
            spy_close_provisional=spy_close_provisional,
        )
        attribution = report.get("alpha_attribution")
        if attribution is not None and not attribution.get("ties_to_headline"):
            logger.warning(
                "EOD alpha attribution did not tie to headline (residual=$%.2f) "
                "on %s — investigate before trusting per-position contributions.",
                attribution.get("residual_usd", 0.0), run_date,
            )
        write_eod_report(report, trades_bucket=trades_bucket, run_date=run_date)
    except Exception as e:
        logger.error("EOD report artifact build/write failed: %s", e)
        if fd:
            fd.report(e, severity="error", context={
                "site": "eod_report_artifact", "run_date": run_date})

    if not send_email:
        logger.info(
            "send_email=False — skipping EOD email for %s (re-reconcile / "
            "reconcile_audit correction pass; artifact re-emitted, no resend).",
            run_date,
        )
    else:
        try:
            send_eod_email(
                run_date=run_date,
                nav=nav,
                daily_return=daily_return,
                spy_return=spy_return,
                alpha=alpha,
                sender=config["email_sender"],
                recipients=config["email_recipients"],
                account_snapshot=account,
                data_warnings=data_warnings,
                console_base_url=config.get("console_base_url"),
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

    # ── T+1 self-heal: re-reconcile any prior day whose stored SPY close has
    # since diverged from the now-settled ArcticDB close (config#1276). Cheap
    # in the common case (a few ArcticDB reads, no re-reconcile when clean).
    # Fail-soft: a correction-pass error must never break the primary EOD.
    # ``run_audit=False`` on the live run is how the audit's own re-reconciles
    # avoid recursing back into the audit.
    if run_audit:
        try:
            from executor.reconcile_audit import audit_window
            _audit = audit_window(exclude_dates={run_date}, send_email=False)
            if _audit.get("corrected"):
                logger.warning(
                    "[reconcile_audit] corrected %d prior day(s) against settled "
                    "ArcticDB: %s", len(_audit["corrected"]),
                    [c["date"] for c in _audit["corrected"]],
                )
            else:
                logger.info(
                    "[reconcile_audit] clean — %d trailing day(s) checked, all "
                    "stored SPY closes match settled ArcticDB within tolerance.",
                    _audit.get("checked", 0),
                )
        except Exception as _ae:  # noqa: BLE001 — self-heal is secondary; primary EOD already wrote
            logger.warning("[reconcile_audit] trailing self-heal FAILED (non-fatal): %s", _ae)
            if fd:
                fd.report(_ae, severity="warning", context={
                    "site": "reconcile_audit_selfheal", "run_date": run_date})

    if fd:
        fd.log_summary(logger)
    conn.close()
    logger.info("EOD reconciliation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "EOD reconciliation. Defaults to today's trading_day "
            "(via nousergon_lib.dates.now_dual). A past --date is SAFE: since "
            "the 2026-04-28 snapshot cutover, the row keyed by run_date sources "
            "its NAV/positions from the durable S3 snapshot for that date (not "
            "now-as-of IB state) and re-prices from settled ArcticDB, so "
            "re-reconciling a past day is the canonical correction path "
            "(config#1276). Requires a snapshot for the date."
        )
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD; defaults to today's trading_day. A past date re-reconciles from its snapshot.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Suppress the EOD email (for a manual re-reconcile / correction of a past day).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the trailing reconcile_audit self-heal pass.",
    )
    args = parser.parse_args()
    # Capture an uncaught crash via flow-doctor before re-raising
    # (no-ops when flow-doctor is inactive).
    with guard_entrypoint():
        run(run_date=args.date, send_email=not args.no_email, run_audit=not args.no_audit)
