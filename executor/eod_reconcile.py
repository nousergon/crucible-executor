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
from pydantic import BaseModel, Field, ValidationError
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.eod_emailer import send_eod_email
from executor.trade_logger import (
    init_db, log_eod, backup_to_s3, get_entry_trade, get_todays_trades,
)

from alpha_engine_lib.dates import now_dual
from alpha_engine_lib.logging import setup_logging
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
        logger.error("Failed to load constituents sector_map: %s", e)
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


class _Narrative(BaseModel):
    """One per-position rationale."""

    ticker: str = Field(..., description="Position ticker symbol (e.g. AAPL).")
    narrative: str = Field(
        ...,
        description=(
            "2-3 sentences explaining why this position is held today, citing the "
            "research thesis, technical signals, and GBM predictions where relevant. "
            "If a trade was made today, the narrative also explains why."
        ),
    )


class _RationalesResponse(BaseModel):
    """Tool-use payload for the EOD rationale synthesis call. The Anthropic
    SDK validates this shape at the tool-use layer; Pydantic re-validates it
    here for type safety + strict-field enforcement. Replaces the legacy
    "ask for JSON in the prompt and json.loads the text" pattern that L1248
    / L2669 documented as recurrence-prone (markdown fences, preamble,
    trailing text — string-pattern whack-a-mole)."""

    narratives: list[_Narrative] = Field(
        ...,
        description="One narrative per position in the input list.",
    )


_RATIONALES_TOOL = {
    "name": "emit_rationales",
    "description": (
        "Emit per-position rationales for the EOD report. Call this tool exactly "
        "once with the full list — one narrative per input position."
    ),
    "input_schema": _RationalesResponse.model_json_schema(),
}


_COST_TELEMETRY_BUCKET = "alpha-engine-research"
_COST_TELEMETRY_PREFIX = "decision_artifacts/_cost_raw"

# Phase 4 #1 — cost-budget WARN threshold. Shared env var with
# alpha-engine-research's ``llm_cost_tracker.RunBudgetExceededError`` +
# alpha-engine-data's ``_cost_telemetry.CostBudgetExceededError`` so a
# single operator knob ceilings cost across all SF entry points.
# Executor's posture is WARN-only (NOT raise) per [[feedback_no_silent_fails]]
# applied carefully: the EOD report is operator-critical and survives
# a cost-sink miss; we cannot let a single anomalous narrative call
# take down the trading-day reconcile. The WARN log + the per-call
# JSONL row on S3 are the operator-facing signals.
_COST_BUDGET_ENV_VAR = "ALPHA_ENGINE_RUN_BUDGET_USD"
_COST_BUDGET_DEFAULT_USD = 100.0


def _resolve_cost_budget_ceiling() -> float:
    """Read ``ALPHA_ENGINE_RUN_BUDGET_USD`` (shared with research + data).

    Returns the positive threshold to WARN above, or 0.0 to disable.
    Parse failure → disable (don't let a malformed env take down EOD).
    """
    import os
    raw = os.environ.get(_COST_BUDGET_ENV_VAR, "")
    if not raw:
        return _COST_BUDGET_DEFAULT_USD
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[cost_telemetry] ALPHA_ENGINE_RUN_BUDGET_USD=%r is not a "
            "number; disabling cost-budget WARN", raw,
        )
        return 0.0

# Anthropic snapshot suffix: -YYYYMMDD on family names (e.g.
# "claude-haiku-4-5-20251001"). Cost-telemetry rate cards are family-
# keyed, so strip the suffix before pricing lookup. Mirrors the research-
# side normalization at ``alpha-engine-research/graph/llm_cost_tracker.py``
# (the 2026-05-02 SF halt that motivated that fix). Lift candidate:
# both research + executor now do this — when a 3rd consumer adopts the
# same pin pattern, fold into ``alpha_engine_lib.cost``'s SDK adapter.
import re as _re
_SNAPSHOT_SUFFIX_RE = _re.compile(r"-\d{8}$")


def _record_eod_narrative_cost(response, run_date: str) -> None:
    """Price + persist one cost-telemetry JSONL row for the EOD narrative.

    Best-effort: any failure is logged and swallowed so the EOD report
    (the primary deliverable) survives. The cost-telemetry surface
    itself fails loud at flush time per
    ``[[feedback_no_silent_fails]]`` — but since the executor's EOD
    runs nightly outside the research SF, we cannot let a cost-sink
    miss take down the trading-day report. The WARN log is the
    operator-facing signal that the row was lost.

    Writes to the same partition the research-side
    ``aggregate_costs.py`` already scans, so the daily parquet rolls
    executor + data + research rows up under one ``by_agent_id``
    breakdown — single source of truth for the cost dashboard.
    """
    try:
        from alpha_engine_lib.cost import record_anthropic_call

        # Normalize snapshot suffix off the SDK-reported model so the
        # family-keyed rate card matches (see _SNAPSHOT_SUFFIX_RE above).
        family_model = _SNAPSHOT_SUFFIX_RE.sub("", getattr(response, "model", ""))
        record = record_anthropic_call(
            response,
            model_name=family_model or None,
            extra_fields={
                "run_id": run_date,
                "agent_id": "executor:eod_narrative",
            },
        )
        key = (
            f"{_COST_TELEMETRY_PREFIX}/{run_date}/{run_date}/"
            f"executor:eod_narrative.jsonl"
        )
        body = (json.dumps(record, default=str) + "\n").encode("utf-8")
        boto3.client("s3").put_object(
            Bucket=_COST_TELEMETRY_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        cost_usd = float(record.get("cost_usd", 0))
        logger.info(
            "[cost_telemetry] EOD narrative cost recorded: $%.4f → "
            "s3://%s/%s",
            cost_usd,
            _COST_TELEMETRY_BUCKET, key,
        )

        # Phase 4 #1 — WARN if this single call alone exceeds the run
        # budget (signal that a narrative prompt expansion or runaway
        # loop pushed one call into the danger zone). Doesn't raise:
        # EOD report continues. Operators see the WARN + dashboard's
        # next refresh shows the spike in the "Recent calls" tab.
        ceiling = _resolve_cost_budget_ceiling()
        if ceiling > 0 and cost_usd > ceiling:
            logger.warning(
                "[cost_telemetry] EOD narrative single call exceeded "
                "cost budget: $%.4f > ceiling=$%.4f (env var "
                "ALPHA_ENGINE_RUN_BUDGET_USD). Investigate the prompt "
                "or position-count expansion — typical EOD call is "
                "under $0.01. JSONL row preserved on S3 for diagnosis.",
                cost_usd, ceiling,
            )
    except Exception as exc:
        # Best-effort. Operator-facing WARN; EOD report continues.
        logger.warning(
            "[cost_telemetry] EOD narrative cost record FAILED for "
            "run_date=%s — cost row LOST: %s",
            run_date, exc,
        )


def _synthesize_rationales(
    contexts: list[dict],
    run_date: str | None = None,
    *,
    llm_enabled: bool = False,
) -> dict[str, str]:
    """Synthesize per-position narratives for the EOD report.

    **Default posture: template-only (NO LLM call).** The executor's
    standing architectural rule is that LLM calls live in the research
    module; the EOD report's "research thesis + GBM signals + trades"
    summary is mechanically derivable from the context dict without
    LLM synthesis. The Haiku path is opt-in via
    ``config.get("eod_narrative_llm_enabled", False)``.

    When ``llm_enabled=True``: calls Haiku via Anthropic tool-use +
    Pydantic validation. L1248 / L2669: previous implementation read
    Haiku's freeform text and tried to ``json.loads`` it —
    recurrence-prone (markdown fences / preamble / trailing text).
    Tool-use makes the parse failure mode structurally impossible:
    Haiku returns a typed ``tool_use`` block whose ``input`` is
    schema-validated by the SDK *before* it lands here. Falls back to
    templates on any failure.

    ``run_date``: when provided AND ``llm_enabled=True``, the response's
    tokens + tool-fee usage are priced via
    ``alpha_engine_lib.cost.record_anthropic_call`` and written to S3.
    When LLM is disabled there's no API call → cost telemetry is
    naturally skipped.
    """
    if not contexts:
        return {}

    if not llm_enabled:
        # Template-only path — same shape as the LLM branch's fallback.
        # Operator-preferred default per the no-LLM-outside-research-
        # module architectural rule (see [[preference_llm_calls_confined_to_research_module]]).
        return _template_rationales(contexts)

    # Try LLM synthesis
    try:
        import anthropic
        client = anthropic.Anthropic()

        prompt = (
            "You are a portfolio analyst writing concise position rationales for an end-of-day report.\n"
            "For each position below, write 2-3 sentences explaining why it is held, "
            "focusing on near-term catalysts (research thesis, technical signals, GBM predictions). "
            "If a trade was made today, explain why. Be specific about numbers.\n\n"
            "Call the emit_rationales tool exactly once with one narrative per position.\n\n"
            f"Positions:\n{json.dumps(contexts, indent=2, default=str)}"
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            tools=[_RATIONALES_TOOL],
            tool_choice={"type": "tool", "name": "emit_rationales"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Phase 0.2 cost telemetry — best-effort capture. Failures here
        # MUST NOT break narrative synthesis (the primary deliverable);
        # log + continue. Per [[feedback_no_silent_fails]] we still WARN
        # loud so missed rows surface in CloudWatch.
        if run_date is not None:
            _record_eod_narrative_cost(response, run_date)
        # tool_choice={"type": "tool", "name": ...} forces Haiku to emit a
        # tool_use block — but Anthropic still allows additional text blocks
        # alongside it. Pick the tool_use block explicitly.
        tool_use = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use is None:
            raise RuntimeError(
                "Haiku response missing the forced emit_rationales tool_use block — "
                f"stop_reason={response.stop_reason!r}"
            )
        try:
            parsed = _RationalesResponse.model_validate(tool_use.input)
        except ValidationError as e:
            logger.warning(
                f"LLM rationale tool_use failed Pydantic validation: {e} — "
                f"input={tool_use.input!r}"
            )
            raise
        return {n.ticker: n.narrative for n in parsed.narratives}
    except Exception as e:
        logger.warning(f"LLM rationale synthesis failed: {e} — using template fallback")

    return _template_rationales(contexts)


def _template_rationales(contexts: list[dict]) -> dict[str, str]:
    """Build per-position rationales from the context dict — no LLM call.

    Default path when ``eod_narrative_llm_enabled`` is false (the
    standing-policy default) AND the deterministic fallback when the
    LLM path raises. Same output shape either way so the EOD emailer
    consumes one contract.
    """
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


def run(run_date: str | None = None) -> None:
    if run_date is None:
        run_date = now_dual().trading_day
        logger.info(
            "EOD reconciliation | date=%s (resolved from now_dual().trading_day)",
            run_date,
        )
    else:
        logger.info("EOD reconciliation | date=%s (explicit)", run_date)
    _health_start = _time.time()

    config = load_config()

    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]

    # Preflight: AWS_REGION + S3 bucket reachable. Fail fast so a
    # misconfigured env surfaces immediately instead of deeper down.
    from executor.preflight import ExecutorPreflight
    ExecutorPreflight(bucket=trades_bucket, mode="eod").run()

    # Flow Doctor: retrieve the shared instance set up at module import
    from alpha_engine_lib.logging import get_flow_doctor
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

    # ── Load closing prices from ArcticDB ──────────────────────────────────
    # Hard-fails on any miss: EOD reconcile must reconcile against an
    # authoritative price source, not IB Gateway's delayed intraday data.
    #
    # Macro-routed held positions (SPY/sector ETFs/etc.) live in the
    # `macro` library, NOT `universe`. The portfolio-optimizer cutover
    # (2026-05-13) made SPY a held core position; its first EOD on
    # 2026-05-14 raised NoSuchVersionException because reconcile was
    # universe-only. Mirror price_cache.load_price_histories' macro-aware
    # dispatch (executor/price_cache.py:128-145, _MACRO_SYMBOLS).
    from executor.price_cache import (
        _open_universe_library,
        _open_macro_library,
        _MACRO_SYMBOLS,
    )
    universe_lib = _open_universe_library(trades_bucket)
    macro_lib = None  # lazy-open only if a macro-routed held ticker appears
    target_ts = pd.Timestamp(run_date).normalize()
    closing_prices: dict[str, float] = {}
    missing: list[str] = []
    # L1346 (c) second-half routing post-#245: SPY removed from macro-routed
    # set since universe.SPY now carries full OHLCV. Defensive macro fallback
    # for SPY preserves backwards compat — mirrors price_cache.py + predictor
    # #196 pattern.
    _MACRO_SYMBOLS_NO_OHLCV = _MACRO_SYMBOLS - {"SPY"}
    for ticker in positions.keys():
        if ticker in _MACRO_SYMBOLS_NO_OHLCV:
            if macro_lib is None:
                macro_lib = _open_macro_library(trades_bucket)
            lib = macro_lib
        else:
            lib = universe_lib
        try:
            df = lib.read(ticker).data
        except Exception as e:
            # SPY-specific defensive fallback to macro.SPY if universe.SPY
            # unreadable. Removed once L1346 (b)+(c) soak clean ≥1 cycle.
            if ticker == "SPY" and lib is universe_lib:
                if macro_lib is None:
                    macro_lib = _open_macro_library(trades_bucket)
                try:
                    df = macro_lib.read(ticker).data
                except Exception as e2:
                    missing.append(
                        f"{ticker} (universe={e.__class__.__name__}, "
                        f"macro={e2.__class__.__name__})"
                    )
                    continue
            else:
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

        # Daily return: today's price vs yesterday's price (or entry price if new today)
        prior_pos = prior_positions.get(ticker)
        prior_price = _resolve_prior_price(prior_pos, pos, current_price)

        if prior_price and prior_price > 0:
            pos["daily_return_pct"] = (current_price / prior_price - 1) * 100
            pos["daily_return_usd"] = (current_price - prior_price) * shares
        else:
            pos["daily_return_pct"] = 0.0
            pos["daily_return_usd"] = 0.0

        # Dividend attribution: today's accrued dividend for this ticker vs
        # yesterday's snapshot. Delta is the day's dividend income (or its
        # reversal when paid to cash). Flows into position α instead of
        # leaking into the cash residual.
        _apply_dividend_delta(pos, prior_pos, prior_price, shares)

        # Alpha contribution: (weight * position_return) - (weight * SPY_return)
        weight = mv / nav if nav else 0
        pos_spy = spy_return if spy_return is not None else 0
        pos["alpha_contribution_pct"] = weight * (pos["daily_return_pct"] - pos_spy)
        pos["alpha_contribution_usd"] = pos["alpha_contribution_pct"] / 100 * nav if nav else 0

    # data_warnings is appended to here (NAV gap) and by _build_position_contexts
    data_warnings: list[str] = []

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

    # Build position rationale narratives — default = template-only (NO
    # LLM call). Operators wanting the polished Haiku-synthesized prose
    # opt in via ``eod_narrative_llm_enabled: true`` in risk.yaml.
    # See [[preference_llm_calls_confined_to_research_module]] for the
    # standing architectural rule.
    signals_bucket = config.get("signals_bucket", "alpha-engine-research")
    llm_enabled = bool(config.get("eod_narrative_llm_enabled", False))
    position_narratives = {}
    try:
        if positions:
            contexts, ctx_warnings = _build_position_contexts(positions, conn, signals_bucket, run_date)
            position_narratives = _synthesize_rationales(
                contexts, run_date=run_date, llm_enabled=llm_enabled,
            )
            logger.info(
                f"Position narratives generated for {len(position_narratives)} "
                f"tickers (llm_enabled={llm_enabled})"
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
            nav_reconciliation=nav_reconciliation,
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
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "EOD reconciliation. Defaults to today's trading_day "
            "(via alpha_engine_lib.dates.now_dual). Hard-fails on any "
            "explicit --date that isn't today: live IB state would corrupt "
            "the historical row's NAV/positions."
        )
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD; must equal today's trading_day or the run aborts.",
    )
    args = parser.parse_args()
    run(run_date=args.date)
