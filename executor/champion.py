"""Champion candidate-source adapter (config#2364 / config#2366).

The champion-promotion loop lets the trading system switch its ENTRY
candidate source between two arms without touching the exit/risk stack:

  * ``agentic``               — today's research pipeline (signals.json
                                 buy_candidates, unchanged).
  * ``scanner_predictor_direct`` — the "measured" arm: entries synthesized
                                 directly from the research-free predictor's
                                 outcome parquet, ranked by predicted_alpha.

Design correction (2026-07-11, config#2366): this is a PLANNER-LAYER
candidate-source adapter, NOT an ``executor/strategies/`` Slot-S plugin —
that contract is EXIT RULES ONLY (``ALLOWED_ACTIONS = ("EXIT", "REDUCE")``
in ``executor/strategies/contract.py``; entries are explicitly out of
scope there). The champion switch instead rewrites
``signals_raw["buy_candidates"]`` before the existing universe/coverage
gates run, so synthesized entries are subject to the SAME risk-rule path
as agentic ones.

Fail-loud posture: the ONLY silent default in this module is the S3 404
on the champion pointer itself (pre-bootstrap state, unambiguous — no
promotion has ever been written). Every other ambiguous condition
(malformed pointer JSON, unknown champion value, stale predictor cohort)
raises. No trading day should start — or silently mis-trade — on an
ambiguous champion selection; the pointer is customer-visible via
Metron's Showcase Portfolio.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date, timedelta

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

CHAMPION_POINTER_KEY = "config/producer_champion.json"
RESEARCH_FREE_PARQUET_KEY = (
    "predictor/research_free_backfill/predictor_outcomes_research_free.parquet"
)

VALID_CHAMPIONS = ("agentic", "scanner_predictor_direct")

# Pre-bootstrap default: no promotion has ever been written, so the pointer
# key legitimately does not exist yet. This is the one unambiguous silent
# default in the whole module — every other failure mode below raises.
_DEFAULT_POINTER = {
    "schema_version": 1,
    "champion": "agentic",
    "promotion_source": "default_pre_bootstrap",
}


class ChampionPointerError(RuntimeError):
    """Raised when the champion pointer is present but unreadable/ambiguous.

    Deliberately NOT raised on a clean S3 404 (see ``_DEFAULT_POINTER``) —
    only on malformed JSON, an unknown ``champion`` value, or any other
    S3 error besides "the key doesn't exist yet".
    """


class StaleChampionFeedError(RuntimeError):
    """Raised when the research-free predictor cohort is older than the
    configured freshness window. A stale champion feed must not trade
    silently on data that no longer reflects current market state."""


def load_champion_pointer(bucket: str, s3_client=None) -> dict:
    """Read ``s3://{bucket}/config/producer_champion.json``.

    Semantics:
      * S3 404 / NoSuchKey → ``_DEFAULT_POINTER`` (agentic, pre-bootstrap).
        This is the one legitimate "pointer doesn't exist yet" case.
      * Any other S3 read error, malformed JSON, or an unknown ``champion``
        value → raise ``ChampionPointerError``. An ambiguous champion must
        never resolve to a silent default — that would risk starting a
        trading day (or worse, silently mixing arms) on a corrupt pointer.

    Pointer schema (written independently by config#2367 in
    crucible-backtester): ``{schema_version: 1, champion: "agentic" |
    "scanner_predictor_direct", promoted_at: <iso8601>, promotion_source:
    <str>}``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=CHAMPION_POINTER_KEY)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            logger.info(
                "No champion pointer at s3://%s/%s — defaulting to agentic "
                "(pre-bootstrap state)", bucket, CHAMPION_POINTER_KEY,
            )
            return dict(_DEFAULT_POINTER)
        raise ChampionPointerError(
            f"Failed to read champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY}: {e}"
        ) from e

    try:
        raw = obj["Body"].read()
        pointer = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
        raise ChampionPointerError(
            f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} is malformed: {e}"
        ) from e

    if not isinstance(pointer, dict):
        raise ChampionPointerError(
            f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} did not "
            f"parse to a JSON object (got {type(pointer).__name__})"
        )

    champion = pointer.get("champion")
    if champion not in VALID_CHAMPIONS:
        raise ChampionPointerError(
            f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} has "
            f"unknown champion={champion!r} — expected one of {VALID_CHAMPIONS}. "
            "Refusing to start a trading day on an ambiguous champion."
        )

    return pointer


def _rank_to_score(rank_fraction: float, floor: float, ceiling: float) -> float:
    """Map a within-cohort rank fraction in [0, 1] (0 = best) onto
    ``[floor, ceiling]``, best rank → ceiling, worst rank → floor.

    Monotonic and deterministic — preserves the predictor's relative
    ordering through ``decide_entries``' score gates (``min_score_to_enter``
    etc.) without hand-tuning a score per name.
    """
    if ceiling <= floor:
        raise ValueError(f"champion_score_ceiling ({ceiling}) must exceed champion_score_floor ({floor})")
    rank_fraction = min(max(rank_fraction, 0.0), 1.0)
    return ceiling - rank_fraction * (ceiling - floor)


def _load_research_free_cohort(bucket: str, s3_client=None) -> pd.DataFrame:
    """Read the research-free predictor outcomes parquet and return the
    LATEST ``prediction_date`` cohort as a DataFrame.

    Schema (crucible-backtester PR#486/#482, already live): ``ticker,
    prediction_date, predicted_alpha, n_research_features_missing``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=RESEARCH_FREE_PARQUET_KEY)
        body = obj["Body"].read()
    except ClientError as e:
        raise ChampionPointerError(
            f"scanner_predictor_direct champion selected but research-free "
            f"parquet s3://{bucket}/{RESEARCH_FREE_PARQUET_KEY} is unreadable: {e}. "
            "Refusing to trade on a missing champion feed."
        ) from e

    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as e:  # noqa: BLE001 — any parse failure must raise, not silently no-op
        raise ChampionPointerError(
            f"Failed to parse research-free parquet s3://{bucket}/"
            f"{RESEARCH_FREE_PARQUET_KEY}: {e}"
        ) from e

    required_cols = {"ticker", "prediction_date", "predicted_alpha"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ChampionPointerError(
            f"research-free parquet missing required column(s) {sorted(missing_cols)} "
            f"— got columns {sorted(df.columns)}"
        )

    if df.empty:
        raise ChampionPointerError(
            "research-free parquet is empty — scanner_predictor_direct champion "
            "has no candidates to select from."
        )

    latest_date = df["prediction_date"].max()
    cohort = df[df["prediction_date"] == latest_date].copy()
    return cohort


def _check_freshness(prediction_date, run_date: str, max_days: int) -> None:
    """Raise ``StaleChampionFeedError`` if ``prediction_date`` is more than
    ``max_days`` calendar days older than ``run_date``."""
    pred_d = pd.Timestamp(prediction_date).date()
    run_d = date.fromisoformat(run_date)
    age_days = (run_d - pred_d).days
    if age_days > max_days:
        raise StaleChampionFeedError(
            f"scanner_predictor_direct champion cohort is stale: "
            f"prediction_date={pred_d} is {age_days} calendar day(s) before "
            f"run_date={run_date} (max allowed {max_days}). "
            "A stale champion feed must not trade silently."
        )
    if age_days < 0:
        raise StaleChampionFeedError(
            f"scanner_predictor_direct champion cohort prediction_date="
            f"{pred_d} is AFTER run_date={run_date} ({-age_days} day(s) in "
            "the future) — refusing to trade on an inconsistent artifact."
        )


def apply_champion_selection(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client=None,
    pointer: dict | None = None,
) -> tuple[dict, dict]:
    """Apply the champion candidate-source switch to ``signals_raw``.

    No-op passthrough when the resolved champion is ``agentic`` (including
    the pre-bootstrap default) — returns ``(signals_raw, predictions_by_ticker)``
    unchanged, same objects, zero mutation.

    When ``scanner_predictor_direct``:
      * Loads the latest research-free predictor cohort (freshness-gated:
        ``run_date - prediction_date <= champion_freshness_max_days``, else
        raises — a stale champion feed must not trade silently).
      * Selects top-N by ``predicted_alpha`` where N = the current
        ``buy_candidates`` count (count-match preserves entry-budget parity
        with the measured counterfactual) when that count is > 0, else the
        ``champion_top_n_default`` config knob.
      * Replaces ``signals_raw["buy_candidates"]`` with synthesized ENTER
        entries. The ``universe`` list (held/EXIT/REDUCE population) is left
        untouched — exits stay managed for all holdings regardless of
        champion.
      * Injects research-free predictions into ``predictions_by_ticker`` for
        the selected tickers (deliberately ``prediction_confidence: 0.0`` —
        keeps the high-confidence-DOWN veto and the hold-book dispersion
        gate neutral on injected entries) so
        ``assert_predictions_cover_buy_candidates`` passes downstream.

    ``pointer``: pass an already-resolved pointer dict (from a prior
    ``load_champion_pointer`` call) to avoid a second S3 round-trip on the
    same key — callers that need to branch on the champion arm BEFORE
    calling this function (e.g. to decide whether to pay for a sector-map
    load) already have the pointer in hand. When omitted, this function
    resolves it itself (single-read convenience for simpler callers/tests).

    Caller contract (config#2366 ordering constraint): this must run BEFORE
    ``filter_buy_candidates_to_universe`` / ``filter_buy_candidates_by_coverage``
    / ``assert_predictions_cover_buy_candidates`` so synthesized candidates
    flow through the same gates as agentic ones — wired inside
    ``executor.main._read_signals``.
    """
    if pointer is None:
        pointer = load_champion_pointer(bucket, s3_client=s3_client)
    champion = pointer["champion"]

    if champion == "agentic":
        return signals_raw, predictions_by_ticker

    if champion != "scanner_predictor_direct":
        # Unreachable in practice — load_champion_pointer already validates
        # against VALID_CHAMPIONS — but fail loud rather than silently
        # falling through if a new champion value is ever added to the
        # pointer schema without a matching branch here.
        raise ChampionPointerError(
            f"apply_champion_selection has no handling for champion={champion!r}"
        )

    max_days = int(config.get("champion_freshness_max_days", 8))
    cohort = _load_research_free_cohort(bucket, s3_client=s3_client)
    latest_date = cohort["prediction_date"].iloc[0]
    _check_freshness(latest_date, run_date, max_days)

    n_buy_candidates = len(signals_raw.get("buy_candidates") or [])
    n = n_buy_candidates if n_buy_candidates > 0 else int(
        config.get("champion_top_n_default", 10)
    )

    cohort_sorted = cohort.sort_values("predicted_alpha", ascending=False).reset_index(drop=True)
    top_n = cohort_sorted.head(n)

    score_floor = float(config.get("champion_score_floor", 60))
    score_ceiling = float(config.get("champion_score_ceiling", 95))
    cohort_size = len(cohort_sorted)
    sector_map = sector_map or {}

    synthesized: list[dict] = []
    injected_predictions: dict[str, dict] = {}
    for rank, row in top_n.iterrows():
        ticker = row["ticker"]
        predicted_alpha = float(row["predicted_alpha"])
        # rank_fraction: 0.0 for the best name (rank 0), approaching 1.0 for
        # the worst — computed against the FULL cohort size so the score
        # band reflects the name's standing in the whole scored universe,
        # not just within the top-N cut.
        rank_fraction = rank / max(cohort_size - 1, 1)
        score = _rank_to_score(rank_fraction, score_floor, score_ceiling)
        predicted_direction = "up" if predicted_alpha >= 0 else "down"

        entry = {
            "signal": "ENTER",
            "ticker": ticker,
            "date": run_date,
            "sector": sector_map.get(ticker, "Unknown"),
            "score": score,
            "conviction": "medium",
            "stance": None,
            "price_target_upside": None,
            "catalyst_date": None,
            "thesis_summary": "research-free predictor champion (config#2364)",
            "champion_arm": "scanner_predictor_direct",
        }
        synthesized.append(entry)

        injected_predictions[ticker] = {
            "predicted_alpha": predicted_alpha,
            "predicted_direction": predicted_direction,
            # Deliberately neutral: the high-confidence-DOWN veto and the
            # hold-book alpha-dispersion gate must not fire off an
            # arbitrarily-assigned confidence for injected entries.
            "prediction_confidence": 0.0,
            "research_free": True,
        }

    logger.info(
        "[champion] scanner_predictor_direct selected %d/%d candidate(s) from "
        "cohort=%s (n_buy_candidates=%d, cohort_size=%d)",
        len(synthesized), n, latest_date, n_buy_candidates, cohort_size,
    )

    new_signals_raw = dict(signals_raw)
    new_signals_raw["buy_candidates"] = synthesized
    new_signals_raw["champion"] = champion
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")

    new_predictions_by_ticker = dict(predictions_by_ticker)
    new_predictions_by_ticker.update(injected_predictions)

    return new_signals_raw, new_predictions_by_ticker
