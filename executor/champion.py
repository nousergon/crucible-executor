"""Champion candidate-source adapter (config#2364 / config#2366 /
alpha-engine-config-I2518 / I2515).

The champion-promotion loop lets the trading system switch its ENTRY
candidate source between arms without touching the exit/risk stack:

  * ``agentic``               — today's research pipeline (signals.json
                                 buy_candidates, unchanged).
  * ``scanner_predictor_direct`` — the "measured" arm: entries synthesized
                                 directly from the research-free predictor's
                                 outcome parquet, ranked by predicted_alpha.
  * ``thinktank_coverage``    — the Think Tank challenger arm (epic I2515):
                                 entries synthesized from the Think Tank
                                 challenger-selection artifact
                                 (``thinktank/challenger_selection/latest.json``,
                                 crucible-research PR#427), ranked by the
                                 analyst's own independent 0-100 rating.
                                 Brian's ruling (config-I2518, 2026-07-14):
                                 champion and challenger run side by side,
                                 whichever performs best in a given week is
                                 promoted at that time — this module is what
                                 lets the pointer actually EXECUTE when it
                                 flips to this arm.

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
(malformed pointer JSON, unknown champion value, stale predictor cohort,
missing/incomplete/stale challenger-selection artifact) raises. No trading
day should start — or silently mis-trade — on an ambiguous champion
selection; the pointer is customer-visible via Metron's Showcase Portfolio.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

CHAMPION_POINTER_KEY = "config/producer_champion.json"
RESEARCH_FREE_PARQUET_KEY = (
    "predictor/research_free_backfill/predictor_outcomes_research_free.parquet"
)
# Think Tank's challenger-arm submission (crucible-research thinktank/__init__.py
# CHALLENGER_SELECTION_LATEST_KEY — kept as a literal here rather than an
# import to avoid a cross-repo package dependency from crucible-executor on
# crucible-research; the key is a stable S3 contract, not shared code).
CHALLENGER_SELECTION_LATEST_KEY = "thinktank/challenger_selection/latest.json"

VALID_CHAMPIONS = ("agentic", "scanner_predictor_direct", "thinktank_coverage")

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
    "scanner_predictor_direct" | "thinktank_coverage", promoted_at:
    <iso8601>, promotion_source: <str>}``.
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


def _load_challenger_selection(bucket: str, s3_client=None) -> dict:
    """Read the Think Tank challenger-selection artifact
    (``thinktank/challenger_selection/latest.json``, crucible-research
    PR#427) and return it as a validated dict.

    Mirrors ``_load_research_free_cohort``'s failure-mode convention
    exactly: missing/unreadable artifact or malformed JSON raises
    ``ChampionPointerError`` — same degrade path, same loudness as the
    scanner_predictor_direct arm's missing-parquet case (no fallback to
    the raw signals.json candidates).

    Schema (``thinktank.schemas.ChallengerSelection``, read here as a
    plain dict — no cross-repo import of crucible-research's pydantic
    model, only the stable field-name contract): ``schema_version, arm,
    trading_day, calendar_date, run_id, mode, board_date, coverage_complete,
    uncovered_count, selections: [{ticker, rating, stance, conviction,
    thesis_version, attractiveness_rank}]``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=CHALLENGER_SELECTION_LATEST_KEY)
        body = obj["Body"].read()
    except ClientError as e:
        raise ChampionPointerError(
            f"thinktank_coverage champion selected but challenger-selection "
            f"artifact s3://{bucket}/{CHALLENGER_SELECTION_LATEST_KEY} is "
            f"unreadable: {e}. Refusing to trade on a missing champion feed."
        ) from e

    try:
        selection = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} is malformed: {e}"
        ) from e

    if not isinstance(selection, dict):
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} did not parse to a JSON "
            f"object (got {type(selection).__name__})"
        )

    required_top_level = {"trading_day", "coverage_complete", "selections"}
    missing_top_level = required_top_level - set(selection.keys())
    if missing_top_level:
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} missing required top-level "
            f"key(s) {sorted(missing_top_level)} — got keys "
            f"{sorted(selection.keys())}"
        )

    selections = selection.get("selections")
    if not isinstance(selections, list):
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} 'selections' is not a list "
            f"(got {type(selections).__name__})"
        )

    required_row_keys = {"ticker", "rating"}
    for i, row in enumerate(selections):
        if not isinstance(row, dict):
            raise ChampionPointerError(
                f"Challenger-selection artifact s3://{bucket}/"
                f"{CHALLENGER_SELECTION_LATEST_KEY} selections[{i}] is not "
                f"an object (got {type(row).__name__})"
            )
        missing_row_keys = required_row_keys - set(row.keys())
        if missing_row_keys:
            raise ChampionPointerError(
                f"Challenger-selection artifact s3://{bucket}/"
                f"{CHALLENGER_SELECTION_LATEST_KEY} selections[{i}] missing "
                f"required key(s) {sorted(missing_row_keys)} — got "
                f"{sorted(row.keys())}"
            )

    return selection


def _check_freshness(
    prediction_date, run_date: str, max_days: int, *, feed_label: str = "scanner_predictor_direct champion cohort"
) -> None:
    """Raise ``StaleChampionFeedError`` if ``prediction_date`` is more than
    ``max_days`` calendar days older than ``run_date``.

    ``feed_label`` names the feed in the raised message so the same
    calendar-day-diff check (technique mirrors ``main._warn_if_stale``'s
    knowledge-day age computation, but with HARD-FAIL severity — a
    champion-arm feed must never trade silently on stale data per this
    module's fail-loud posture) reads correctly for whichever arm calls it.
    """
    pred_d = pd.Timestamp(prediction_date).date()
    run_d = date.fromisoformat(run_date)
    age_days = (run_d - pred_d).days
    if age_days > max_days:
        raise StaleChampionFeedError(
            f"{feed_label} is stale: "
            f"prediction_date={pred_d} is {age_days} calendar day(s) before "
            f"run_date={run_date} (max allowed {max_days}). "
            "A stale champion feed must not trade silently."
        )
    if age_days < 0:
        raise StaleChampionFeedError(
            f"{feed_label} prediction_date="
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

    When ``thinktank_coverage`` (epic I2515 / config-I2518): same shape,
    entries synthesized from the Think Tank challenger-selection artifact
    instead of the research-free predictor cohort — see
    ``_apply_thinktank_coverage`` for the arm-specific validity gates
    (coverage-completeness + trading_day freshness) and rank→score mapping.

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

    if champion == "scanner_predictor_direct":
        return _apply_scanner_predictor_direct(
            signals_raw, predictions_by_ticker,
            bucket=bucket, run_date=run_date, config=config,
            sector_map=sector_map, s3_client=s3_client, pointer=pointer,
        )

    if champion == "thinktank_coverage":
        return _apply_thinktank_coverage(
            signals_raw, predictions_by_ticker,
            bucket=bucket, run_date=run_date, config=config,
            sector_map=sector_map, s3_client=s3_client, pointer=pointer,
        )

    # Unreachable in practice — load_champion_pointer already validates
    # against VALID_CHAMPIONS — but fail loud rather than silently
    # falling through if a new champion value is ever added to the
    # pointer schema without a matching branch here.
    raise ChampionPointerError(
        f"apply_champion_selection has no handling for champion={champion!r}"
    )


def _apply_scanner_predictor_direct(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client,
    pointer: dict,
) -> tuple[dict, dict]:
    """``scanner_predictor_direct`` arm — see ``apply_champion_selection``
    docstring for the full contract. Extracted verbatim (config-I2518) so
    ``apply_champion_selection`` can dispatch across multiple arms without a
    single function growing without bound."""
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
    new_signals_raw["champion"] = "scanner_predictor_direct"
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")

    new_predictions_by_ticker = dict(predictions_by_ticker)
    new_predictions_by_ticker.update(injected_predictions)

    return new_signals_raw, new_predictions_by_ticker


def _apply_thinktank_coverage(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client,
    pointer: dict,
) -> tuple[dict, dict]:
    """``thinktank_coverage`` arm (epic I2515 / config-I2518) — entries
    synthesized from the Think Tank challenger-selection artifact
    (``thinktank/challenger_selection/latest.json``, crucible-research
    PR#427) instead of the research-free predictor cohort.

    HARD VALIDITY GATES (config#1580 — a champion feed must never trade
    silently on stale/invalid data; same fail-loud posture and same
    missing-artifact degrade path as ``_apply_scanner_predictor_direct``'s
    missing-parquet case — no fallback to raw signals.json candidates):

      * ``coverage_complete`` must be True. Brian's ruling (config#1580):
        the selection only counts as valid champion-arm evidence once the
        ENTIRE current-scan top-N coverage window is covered — an
        incomplete-coverage selection raises ``ChampionPointerError``
        rather than trading a partial/unrepresentative pool.
      * ``trading_day`` must be within ``champion_freshness_max_days`` of
        ``run_date`` (same knob + same calendar-day-diff technique as the
        scanner arm's ``_check_freshness``, reused rather than a new
        TT-specific staleness parameter — both represent "how stale can a
        champion feed be before we refuse to trade", not an arm-specific
        concept). Note this is a HARD gate, distinct from the artifact's
        own ``board_date`` (which the producer deliberately never
        hard-fails on — the daily Think Tank cadence legitimately reads a
        stale universe board all week, config#1580) — ``trading_day`` is
        the run identity of the challenger-selection artifact itself, the
        analogue of the scanner arm's ``prediction_date``.

    Rank → score: ``selections`` arrives PRE-SORTED best-rating-first from
    the producer (``thinktank.challenger_selection.write_challenger_selection``
    sorts by rating descending before truncating to its own top-N), but this
    is defensively re-sorted here rather than trusted, mirroring the scanner
    arm's own defensive ``sort_values`` on its cohort. rank_fraction is
    computed WITHIN the selection itself (denominator = the number of names
    Think Tank actually submitted, up to its own ``CHALLENGER_TOP_N``) —
    unlike the scanner arm, there is no larger scored population to rank
    against; the challenger-selection artifact only ever contains its own
    top-N, so "within the selection" is the correct (and only available)
    cohort for the rank-fraction denominator.

    Deliberately-neutral injected prediction fields — same intent as the
    scanner arm (keep the high-confidence-DOWN veto and hold-book dispersion
    gate authoritative, not skewed by champion-injected values), same
    ``prediction_confidence: 0.0``. ``predicted_alpha``/``predicted_direction``
    are explicitly ``None`` rather than a fabricated numeric value: Think
    Tank's rating is a subjective 0-100 score, not a log-alpha estimate, and
    inventing a fake numeric alpha would misrepresent the hold-book
    dispersion calc (``main._should_hold_book``) rather than keep it neutral
    — ``None`` is excluded from that calc's cross-sectional stdev entirely
    (its `isinstance(a, (int, float))` guard), the honest way to contribute
    zero signal-magnitude opinion.
    """
    selection = _load_challenger_selection(bucket, s3_client=s3_client)

    if not selection.get("coverage_complete"):
        raise ChampionPointerError(
            f"thinktank_coverage champion selected but the challenger-"
            f"selection artifact's coverage_complete=False "
            f"(uncovered_count={selection.get('uncovered_count')!r}, "
            f"trading_day={selection.get('trading_day')!r}) — refusing to "
            "trade on an incomplete-coverage selection (config#1580)."
        )

    max_days = int(config.get("champion_freshness_max_days", 8))
    trading_day = selection["trading_day"]
    _check_freshness(
        trading_day, run_date, max_days,
        feed_label="thinktank_coverage challenger-selection artifact",
    )

    rows = list(selection.get("selections") or [])
    if not rows:
        raise ChampionPointerError(
            "thinktank_coverage challenger-selection artifact has no "
            "selections — champion has no candidates to select from."
        )

    rows_sorted = sorted(rows, key=lambda r: r["rating"], reverse=True)
    selection_size = len(rows_sorted)

    n_buy_candidates = len(signals_raw.get("buy_candidates") or [])
    n = n_buy_candidates if n_buy_candidates > 0 else int(
        config.get("champion_top_n_default", 10)
    )
    top_n = rows_sorted[:n]

    score_floor = float(config.get("champion_score_floor", 60))
    score_ceiling = float(config.get("champion_score_ceiling", 95))
    sector_map = sector_map or {}

    synthesized: list[dict] = []
    injected_predictions: dict[str, dict] = {}
    for rank, row in enumerate(top_n):
        ticker = row["ticker"]
        # rank_fraction: 0.0 for the best-rated name, approaching 1.0 for
        # the worst — computed WITHIN the selection (see docstring: there is
        # no larger scored population to rank against for this arm).
        rank_fraction = rank / max(selection_size - 1, 1)
        score = _rank_to_score(rank_fraction, score_floor, score_ceiling)

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
            "thesis_summary": "thinktank_coverage challenger champion (config-I2518 / I2515)",
            "champion_arm": "thinktank_coverage",
        }
        synthesized.append(entry)

        injected_predictions[ticker] = {
            # Deliberately None, not a fabricated numeric alpha — see
            # docstring. Keeps this entry OUT of main._should_hold_book's
            # cross-sectional dispersion calc entirely.
            "predicted_alpha": None,
            "predicted_direction": None,
            # Same neutral value as the scanner arm — keeps the
            # high-confidence-DOWN veto and hold-book dispersion gate
            # authoritative, not skewed by champion-injected entries.
            "prediction_confidence": 0.0,
            "thinktank_coverage": True,
        }

    logger.info(
        "[champion] thinktank_coverage selected %d/%d candidate(s) from "
        "challenger-selection trading_day=%s (n_buy_candidates=%d, "
        "selection_size=%d, uncovered_count=%d)",
        len(synthesized), n, trading_day, n_buy_candidates, selection_size,
        selection.get("uncovered_count", 0),
    )

    new_signals_raw = dict(signals_raw)
    new_signals_raw["buy_candidates"] = synthesized
    new_signals_raw["champion"] = "thinktank_coverage"
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")

    new_predictions_by_ticker = dict(predictions_by_ticker)
    new_predictions_by_ticker.update(injected_predictions)

    return new_signals_raw, new_predictions_by_ticker
