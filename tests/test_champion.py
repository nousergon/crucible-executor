"""Tests for executor/champion.py — champion candidate-source adapter
(config#2364 / config#2366).

Covers:
  * load_champion_pointer: 404→agentic default; malformed JSON→raise;
    unknown champion→raise; other ClientError→raise.
  * apply_champion_selection: agentic passthrough is a true no-op; the
    scanner_predictor_direct path synthesizes ENTER entries that route to
    get_actionable_signals()["enter"], injects predictions so
    assert_predictions_cover_buy_candidates passes, leaves universe
    untouched, honors count-match, raises on a stale cohort, and produces
    a monotonic rank→score mapping.

All hermetic — S3 is a tiny in-memory fake, no real boto3/network calls.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from executor.alpha_contract import AlphaAnchorError
from executor.champion import (
    CHALLENGER_SELECTION_LATEST_KEY,
    CHAMPION_POINTER_KEY,
    COHORT_FRESH_MAX_DAYS,
    RESEARCH_FREE_PARQUET_KEY,
    SHADOW_SIGNALS_KEY_TEMPLATE,
    THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS,
    ChampionPointerError,
    StaleChampionFeedError,
    apply_champion_selection,
    evaluate_cohort_staleness,
    load_champion_pointer,
)
from executor.signal_reader import (
    assert_predictions_cover_buy_candidates,
    get_actionable_signals,
)


class _FakeS3:
    """Minimal get_object stand-in over a dict of {key: bytes}."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                operation_name="GetObject",
            )
        return {"Body": io.BytesIO(self.objects[Key])}


def _pointer_bytes(champion="scanner_predictor_direct", promotion_source="manual_test", **extra) -> bytes:
    payload = {
        "schema_version": 1,
        "champion": champion,
        "promoted_at": "2026-07-10T00:00:00Z",
        "promotion_source": promotion_source,
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def _parquet_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def _challenger_selection_bytes(
    trading_day: str = "2026-07-10",
    n: int = 5,
    coverage_complete: bool = True,
    uncovered_count: int = 0,
    presorted: bool = True,
    row_overrides: list[dict] | None = None,
) -> bytes:
    """Build a ChallengerSelection-shaped JSON payload (thinktank/
    challenger_selection/latest.json, crucible-research PR#427). Ratings
    descend by ticker index so rank order is deterministic; ``presorted``
    lets a test deliberately scramble the order to exercise the defensive
    re-sort in ``_apply_thinktank_coverage``."""
    rows = [
        {
            "ticker": f"TKR{i:03d}",
            "rating": 90 - i * 5,
            "stance": "attractive",
            "conviction": 70,
            "thesis_version": 1,
            "attractiveness_rank": i + 1,
        }
        for i in range(n)
    ]
    if row_overrides is not None:
        rows = row_overrides
    if not presorted:
        rows = list(reversed(rows))
    payload = {
        "schema_version": 1,
        "arm": "thinktank_coverage",
        "trading_day": trading_day,
        "calendar_date": trading_day,
        "run_id": "run-test-1",
        "mode": "daily",
        "board_date": trading_day,
        "coverage_complete": coverage_complete,
        "uncovered_count": uncovered_count,
        "selections": rows,
    }
    return json.dumps(payload).encode()


def _cohort_rows(date: str, n: int = 5) -> list[dict]:
    # predicted_alpha descending by ticker index so rank order is deterministic.
    return [
        {
            "ticker": f"TKR{i:03d}",
            "prediction_date": date,
            "predicted_alpha": 0.10 - i * 0.01,
            "n_research_features_missing": 0,
        }
        for i in range(n)
    ]


_CONFIG = {
    "champion_top_n_default": 10,
    "champion_score_floor": 60,
    "champion_score_ceiling": 95,
    "champion_freshness_max_days": 8,
}


# ── load_champion_pointer ────────────────────────────────────────────────


class TestLoadChampionPointer:
    def test_404_defaults_to_agentic(self):
        s3 = _FakeS3({})
        pointer = load_champion_pointer("test-bucket", s3_client=s3)
        assert pointer["champion"] == "agentic"
        assert pointer["promotion_source"] == "default_pre_bootstrap"

    def test_malformed_json_raises(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: b"{not valid json"})
        with pytest.raises(ChampionPointerError):
            load_champion_pointer("test-bucket", s3_client=s3)

    def test_unknown_champion_value_raises(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="totally_bogus_arm")})
        with pytest.raises(ChampionPointerError):
            load_champion_pointer("test-bucket", s3_client=s3)

    def test_non_dict_json_raises(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: b"[1, 2, 3]"})
        with pytest.raises(ChampionPointerError):
            load_champion_pointer("test-bucket", s3_client=s3)

    def test_other_client_error_raises(self):
        class _BrokenS3:
            def get_object(self, Bucket, Key):
                raise ClientError(
                    error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
                    operation_name="GetObject",
                )

        with pytest.raises(ChampionPointerError):
            load_champion_pointer("test-bucket", s3_client=_BrokenS3())

    def test_valid_agentic_pointer_reads_through(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="agentic")})
        pointer = load_champion_pointer("test-bucket", s3_client=s3)
        assert pointer["champion"] == "agentic"

    def test_valid_scanner_predictor_direct_pointer_reads_through(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct")})
        pointer = load_champion_pointer("test-bucket", s3_client=s3)
        assert pointer["champion"] == "scanner_predictor_direct"

    def test_valid_thinktank_coverage_pointer_reads_through(self):
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage")})
        pointer = load_champion_pointer("test-bucket", s3_client=s3)
        assert pointer["champion"] == "thinktank_coverage"


# ── apply_champion_selection: agentic passthrough ───────────────────────


class TestAgenticPassthrough:
    def test_agentic_pointer_is_true_no_op(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [{"ticker": "AAPL", "signal": "ENTER", "sector": "Technology"}],
            "universe": [{"ticker": "AAPL", "signal": "ENTER", "sector": "Technology"}],
        }
        predictions_by_ticker = {"AAPL": {"predicted_alpha": 0.02}}
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="agentic")})

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            predictions_by_ticker,
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        assert out_signals is signals_raw
        assert out_preds is predictions_by_ticker

    def test_404_pointer_default_is_true_no_op(self):
        """No pointer object at all (pre-bootstrap) behaves identically to
        an explicit agentic pointer — same no-op contract."""
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [{"ticker": "AAPL", "signal": "ENTER", "sector": "Technology"}],
            "universe": [],
        }
        predictions_by_ticker = {"AAPL": {"predicted_alpha": 0.02}}
        s3 = _FakeS3({})  # no pointer key at all

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            predictions_by_ticker,
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        assert out_signals is signals_raw
        assert out_preds is predictions_by_ticker


# ── apply_champion_selection: scanner_predictor_direct ──────────────────


class TestScannerPredictorDirect:
    def _s3(self, cohort_date="2026-07-10", n=5, champion="scanner_predictor_direct"):
        return _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion=champion),
                RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows(cohort_date, n)),
            }
        )

    def test_synthesized_entries_route_to_enter(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [],  # empty → falls back to champion_top_n_default
            "universe": [{"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology"}],
        }
        s3 = self._s3(cohort_date="2026-07-10", n=3)

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={"TKR000": "Technology"},
            s3_client=s3,
        )

        actionable = get_actionable_signals(out_signals)
        entered_tickers = {s["ticker"] for s in actionable["enter"]}
        assert entered_tickers == {"TKR000", "TKR001", "TKR002"}
        for s in actionable["enter"]:
            assert s["signal"] == "ENTER"
            assert s["champion_arm"] == "scanner_predictor_direct"

    def test_coverage_assert_passes_with_injected_predictions(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [],
            "universe": [],
        }
        s3 = self._s3(n=4)

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        # Must not raise — every synthesized buy_candidate has a prediction row.
        assert_predictions_cover_buy_candidates(out_signals, out_preds)

        for pred in out_preds.values():
            assert pred["prediction_confidence"] == 0.0
            assert pred["research_free"] is True
            assert pred["predicted_direction"] in ("up", "down")

    def test_universe_left_untouched(self):
        held = [{"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology"}]
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [{"ticker": "OLD1", "signal": "ENTER", "sector": "Technology"}],
            "universe": held,
        }
        # n=2, not 1: a single-name cohort has no definable market level, so
        # the arm now refuses it outright (alpha-engine-config-I7337). See
        # test_a_single_name_cohort_is_refused for that behaviour on purpose.
        s3 = self._s3(n=2)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        assert out_signals["universe"] == held
        assert out_signals["universe"] is signals_raw["universe"]

    def test_count_match_honored_against_nonempty_buy_candidates(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [
                {"ticker": "OLD1", "signal": "ENTER", "sector": "Technology"},
                {"ticker": "OLD2", "signal": "ENTER", "sector": "Technology"},
            ],
            "universe": [],
        }
        s3 = self._s3(n=5)  # cohort has 5, but only 2 buy_candidates → N=2

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_empty_buy_candidates_uses_top_n_default(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=5)
        cfg = dict(_CONFIG, champion_top_n_default=3)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 3

    def test_stale_cohort_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # cohort is 20 days before run_date; max allowed is 8.
        s3 = self._s3(cohort_date="2026-06-23", n=3)

        with pytest.raises(StaleChampionFeedError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_cohort_within_freshness_window_does_not_raise(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # exactly 8 days old — boundary, should be allowed (<=).
        s3 = self._s3(cohort_date="2026-07-05", n=2)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_rank_to_score_monotonic_within_cohort(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=6)
        cfg = dict(_CONFIG, champion_top_n_default=6)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )

        entries = out_signals["buy_candidates"]
        scores = [e["score"] for e in entries]
        # _cohort_rows generates strictly descending predicted_alpha by
        # ticker index, so rank order == ticker order == score order
        # (best predicted_alpha gets the highest score).
        assert scores == sorted(scores, reverse=True)
        assert max(scores) <= _CONFIG["champion_score_ceiling"]
        assert min(scores) >= _CONFIG["champion_score_floor"]
        # Best-ranked name should be at (or very near) the ceiling.
        assert scores[0] == pytest.approx(_CONFIG["champion_score_ceiling"])

    def test_missing_required_columns_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        bad_rows = [{"ticker": "TKR000", "prediction_date": "2026-07-10"}]  # no predicted_alpha
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(),
                RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(bad_rows),
            }
        )

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_missing_parquet_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes()})  # no parquet key

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_sector_map_applied_to_synthesized_entries(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=2)
        sector_map = {"TKR000": "Health Care", "TKR001": "Financials"}

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map=sector_map,
            s3_client=s3,
        )
        by_ticker = {e["ticker"]: e for e in out_signals["buy_candidates"]}
        assert by_ticker["TKR000"]["sector"] == "Health Care"
        assert by_ticker["TKR001"]["sector"] == "Financials"

    def test_stamps_champion_and_promotion_source_on_signals_raw(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=2)  # see I7337 note above — n=1 has no market level

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert out_signals["champion"] == "scanner_predictor_direct"
        assert out_signals["promotion_source"] == "manual_test"


# ── apply_champion_selection: thinktank_coverage ────────────────────────


class TestThinktankCoverage:
    def _s3(self, **kwargs):
        return _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage"),
                CHALLENGER_SELECTION_LATEST_KEY: _challenger_selection_bytes(**kwargs),
            }
        )

    def test_synthesized_entries_route_to_enter(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [],  # empty → falls back to champion_top_n_default
            "universe": [{"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology"}],
        }
        s3 = self._s3(trading_day="2026-07-10", n=3)

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={"TKR000": "Technology"},
            s3_client=s3,
        )

        actionable = get_actionable_signals(out_signals)
        entered_tickers = {s["ticker"] for s in actionable["enter"]}
        assert entered_tickers == {"TKR000", "TKR001", "TKR002"}
        for s in actionable["enter"]:
            assert s["signal"] == "ENTER"
            assert s["champion_arm"] == "thinktank_coverage"

    def test_coverage_assert_passes_with_injected_predictions(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=4)

        out_signals, out_preds = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        # Must not raise — every synthesized buy_candidate has a prediction row.
        assert_predictions_cover_buy_candidates(out_signals, out_preds)

        for pred in out_preds.values():
            assert pred["prediction_confidence"] == 0.0
            assert pred["thinktank_coverage"] is True
            # Deliberately None — no fabricated numeric alpha for a
            # subjective 0-100 rating (see _apply_thinktank_coverage docstring).
            assert pred["predicted_alpha"] is None
            assert pred["predicted_direction"] is None

    def test_universe_left_untouched(self):
        held = [{"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology"}]
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [{"ticker": "OLD1", "signal": "ENTER", "sector": "Technology"}],
            "universe": held,
        }
        # n=2, not 1: a single-name cohort has no definable market level, so
        # the arm now refuses it outright (alpha-engine-config-I7337). See
        # test_a_single_name_cohort_is_refused for that behaviour on purpose.
        s3 = self._s3(n=2)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )

        assert out_signals["universe"] == held
        assert out_signals["universe"] is signals_raw["universe"]

    def test_count_match_honored_against_nonempty_buy_candidates(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [
                {"ticker": "OLD1", "signal": "ENTER", "sector": "Technology"},
                {"ticker": "OLD2", "signal": "ENTER", "sector": "Technology"},
            ],
            "universe": [],
        }
        s3 = self._s3(n=5)  # selection has 5, but only 2 buy_candidates → N=2

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_empty_buy_candidates_uses_top_n_default(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=5)
        cfg = dict(_CONFIG, champion_top_n_default=3)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 3

    def test_n_exceeding_selection_size_returns_all_available(self):
        """Count parity mirrors the scanner arm's pandas .head(n) semantics:
        requesting more than the selection has just returns what's there,
        never an error."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=3)
        cfg = dict(_CONFIG, champion_top_n_default=50)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 3

    def test_stale_trading_day_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # trading_day is 20 days before run_date; the arm's own bound is 3.
        s3 = self._s3(trading_day="2026-06-23", n=3)

        with pytest.raises(StaleChampionFeedError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_trading_day_within_freshness_window_does_not_raise(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # exactly 3 days old — boundary of THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS,
        # should be allowed (<=). alpha-engine-config-I7232.
        s3 = self._s3(trading_day="2026-07-10", n=2)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_freshness_bound_is_3_and_independent_of_shared_config_constant(self):
        """alpha-engine-config-I7232 (Brian ruling 2026-08-14): the
        thinktank_coverage arm's freshness bound is a fixed module constant
        (3, matching crucible-research-PR630's POINTER_LAG_ERROR_DAYS), NOT
        derived from — or equal to by coincidence with — the shared
        `champion_freshness_max_days` config default (8) used by the
        scanner_predictor_direct arm. A future change to the shared constant
        must not silently change this arm's behavior."""
        assert THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS == 3
        assert THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS != _CONFIG["champion_freshness_max_days"]

        # Even if a caller's config explicitly sets champion_freshness_max_days
        # to something else entirely, the thinktank arm's bound must not move:
        # a pointer 4 days old still raises regardless of the shared knob.
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(trading_day="2026-07-09", n=2)  # 4 days old
        cfg = dict(_CONFIG, champion_freshness_max_days=30)  # would NOT raise if shared

        with pytest.raises(StaleChampionFeedError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=cfg,
                sector_map={},
                s3_client=s3,
            )

    def test_scanner_arm_still_uses_shared_config_constant_unaffected(self):
        """The scanner_predictor_direct arm's freshness bound is untouched by
        I7232 — it still reads `champion_freshness_max_days` from config
        (default 8), not THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS. A cohort 4
        days old (which now fails the thinktank arm's 3-day bound) must
        continue to pass for the scanner arm under the unchanged 8-day
        default."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
                RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows("2026-07-09", n=2)),  # 4 days old
            }
        )

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_4_day_old_pointer_now_raises_but_would_not_have_under_old_8_day_bound(self):
        """Proves the BEHAVIOR CHANGE from I7232, not just the constant: a
        challenger-selection pointer aged 4 calendar days — well within the
        old shared 8-day `champion_freshness_max_days` bound, so it would
        NOT have raised before this fix — now raises StaleChampionFeedError
        under the arm's own 3-day bound. This is exactly the frozen-pointer
        shape the ruling closes (config-I7232): a stale Think Tank selection
        that would previously have passed silently into a live champion
        feed."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(trading_day="2026-07-09", n=2)  # 4 days old

        # Sanity: 4 days is within the OLD shared 8-day bound — confirms this
        # case would have passed under the pre-fix behavior.
        assert 4 <= _CONFIG["champion_freshness_max_days"]
        # And outside the NEW arm-specific 3-day bound.
        assert 4 > THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS

        with pytest.raises(StaleChampionFeedError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_coverage_incomplete_raises(self):
        """Brian's ruling (config#1580): coverage_complete=False must never
        trade — same hard-fail loudness as a missing artifact, no fallback
        to raw signals.json candidates."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=5, coverage_complete=False, uncovered_count=7)

        with pytest.raises(ChampionPointerError, match="coverage_complete"):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_missing_artifact_raises(self):
        """Mirrors the scanner arm's missing-parquet convention exactly:
        missing challenger-selection artifact → ChampionPointerError, no
        silent fallback to agentic candidates."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage")})

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_malformed_artifact_json_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage"),
                CHALLENGER_SELECTION_LATEST_KEY: b"{not valid json",
            }
        )

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_missing_required_top_level_key_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        bad_payload = json.dumps({"schema_version": 1, "selections": []}).encode()  # no trading_day/coverage_complete
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage"),
                CHALLENGER_SELECTION_LATEST_KEY: bad_payload,
            }
        )

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_missing_required_row_key_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=1, row_overrides=[{"ticker": "TKR000"}])  # no rating

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_empty_selections_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=0)

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw,
                {},
                bucket="test-bucket",
                run_date="2026-07-13",
                config=_CONFIG,
                sector_map={},
                s3_client=s3,
            )

    def test_sector_map_applied_to_synthesized_entries(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=2)
        sector_map = {"TKR000": "Health Care", "TKR001": "Financials"}

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map=sector_map,
            s3_client=s3,
        )
        by_ticker = {e["ticker"]: e for e in out_signals["buy_candidates"]}
        assert by_ticker["TKR000"]["sector"] == "Health Care"
        assert by_ticker["TKR001"]["sector"] == "Financials"

    def test_stamps_champion_and_promotion_source_on_signals_raw(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=2)  # see I7337 note above — n=1 has no market level

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=_CONFIG,
            sector_map={},
            s3_client=s3,
        )
        assert out_signals["champion"] == "thinktank_coverage"
        assert out_signals["promotion_source"] == "manual_test"

    def test_rank_to_score_monotonic_within_selection(self):
        """Rating→score ordering preserved through decide_entries' score
        gates, exactly mirroring the scanner arm's rank_fraction contract
        but scoped to 'within the selection' (see module docstring: there
        is no larger scored population available for this arm)."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=6)
        cfg = dict(_CONFIG, champion_top_n_default=6)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )

        entries = out_signals["buy_candidates"]
        scores = [e["score"] for e in entries]
        # _challenger_selection_bytes generates strictly descending rating by
        # ticker index, so rank order == ticker order == score order.
        assert scores == sorted(scores, reverse=True)
        assert max(scores) <= _CONFIG["champion_score_ceiling"]
        assert min(scores) >= _CONFIG["champion_score_floor"]
        assert scores[0] == pytest.approx(_CONFIG["champion_score_ceiling"])

    def test_rank_to_score_ordering_preserved_when_producer_order_is_scrambled(self):
        """The producer sorts by rating before writing, but this module
        defensively re-sorts rather than trusting that ordering (mirrors
        the scanner arm's own defensive sort_values call) — scrambling the
        input order must not change the resulting rank-based scores."""
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=6, presorted=False)
        cfg = dict(_CONFIG, champion_top_n_default=6)

        out_signals, _ = apply_champion_selection(
            signals_raw,
            {},
            bucket="test-bucket",
            run_date="2026-07-13",
            config=cfg,
            sector_map={},
            s3_client=s3,
        )

        by_ticker = {e["ticker"]: e["score"] for e in out_signals["buy_candidates"]}
        # TKR000 has the highest rating (90) regardless of input order —
        # must land at the ceiling.
        assert by_ticker["TKR000"] == pytest.approx(_CONFIG["champion_score_ceiling"])
        assert by_ticker["TKR000"] > by_ticker["TKR005"]


# ── Regression: _read_signals wiring stays a true no-op on agentic/absent ──


class TestReadSignalsChampionRegression:
    """End-to-end regression through executor.main._read_signals (not just
    the adapter in isolation): with the champion pointer absent (S3 404) or
    explicitly "agentic", the champion adapter must be a TRUE no-op — the
    live (simulate=False) signal-read path must produce byte-identical
    signals_raw/predictions_by_ticker to a champion-disabled baseline, and
    must not touch the parquet artifact at all.

    Mirrors the idiom in tests/test_perf_simulate_mode.py
    (TestSimulateModeSkipsArcticDBFilter) — mock the S3-touching collaborators
    _read_signals calls internally so this runs hermetically, then assert on
    call counts / return values rather than real S3 round-trips.
    """

    def _minimal_signals_override(self):
        return {
            "date": "2026-04-25",
            "market_regime": "neutral",
            "sector_ratings": {},
            "enter": [],
            "exit": [],
            "reduce": [],
            "hold": [],
            "universe": [],
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER", "sector": "Technology", "score": 80},
            ],
        }

    def _patch_non_champion_s3_boundaries(self, monkeypatch):
        """Passthrough every OTHER S3-touching call _read_signals makes in
        live mode, isolating the champion pointer as the only boundary this
        test actually exercises."""

        monkeypatch.setattr(
            "executor.signal_reader.filter_buy_candidates_to_universe",
            lambda s, b: s,
        )
        monkeypatch.setattr(
            "executor.eod_reconcile._load_constituents_sector_map",
            lambda bucket: {},
        )
        monkeypatch.setattr(
            "executor.signal_reader.patch_unknown_sectors_with_constituents",
            lambda signals_raw, bucket: 0,
        )
        monkeypatch.setattr(
            "executor.signal_reader.read_predictions",
            lambda bucket: ({"AAPL": {"predicted_alpha": 0.01}}, "2026-04-24"),
        )

    def test_absent_pointer_produces_identical_signals_to_agentic(self, monkeypatch):
        """A 404 (no pointer object) and an explicit agentic pointer must
        both be true no-ops, producing IDENTICAL _read_signals output."""
        import executor.main as main_mod

        self._patch_non_champion_s3_boundaries(monkeypatch)
        config = {"signals_bucket": "test-bucket", "coverage_admission_enabled": False}

        for pointer_s3 in (_FakeS3({}), _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="agentic")})):
            monkeypatch.setattr("boto3.client", lambda *a, s3=pointer_s3, **kw: s3)
            result = main_mod._read_signals(
                config=config,
                signals_bucket="test-bucket",
                run_date="2026-04-25",
                simulate=False,
                signals_override=self._minimal_signals_override(),
                conn=None,
            )
            signals_raw, signals, run_date, predictions_by_ticker, predictions_date = result

            assert signals_raw["buy_candidates"] == [
                {"ticker": "AAPL", "signal": "ENTER", "sector": "Technology", "score": 80},
            ], "champion adapter must not alter buy_candidates on agentic/absent pointer"
            assert "champion" not in signals_raw, (
                "agentic/absent pointer must not stamp a champion field — "
                "only the scanner_predictor_direct path stamps attribution"
            )
            assert predictions_by_ticker == {"AAPL": {"predicted_alpha": 0.01}}, (
                "champion adapter must not inject/alter predictions on agentic/absent pointer"
            )

    def test_agentic_pointer_never_reads_research_free_parquet(self, monkeypatch):
        """No-op path must not even attempt the parquet round-trip —
        get_object on the parquet key would raise if called (key absent),
        proving the parquet load is gated on champion=scanner_predictor_direct."""
        import executor.main as main_mod

        self._patch_non_champion_s3_boundaries(monkeypatch)
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="agentic")})
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)

        # Should not raise — proves the parquet key (absent from `s3`) is
        # never fetched on the agentic path.
        main_mod._read_signals(
            config={"signals_bucket": "test-bucket", "coverage_admission_enabled": False},
            signals_bucket="test-bucket",
            run_date="2026-04-25",
            simulate=False,
            signals_override=self._minimal_signals_override(),
            conn=None,
        )

    def test_agentic_pointer_skips_sector_map_load(self, monkeypatch):
        """Efficiency regression: on the common agentic/pre-bootstrap path,
        _read_signals must NOT pay for _load_constituents_sector_map's S3
        list_objects_v2 + get_object round-trip just for the champion
        adapter to immediately discard it on a no-op passthrough. Only
        scanner_predictor_direct needs the sector_map (to stamp synthesized
        entries), and even then it's the same map
        patch_unknown_sectors_with_constituents fetches later — the champion
        adapter must not add a SECOND unconditional fetch on top of that."""
        import executor.main as main_mod

        sector_map_calls = []
        monkeypatch.setattr(
            "executor.signal_reader.filter_buy_candidates_to_universe",
            lambda s, b: s,
        )
        monkeypatch.setattr(
            "executor.eod_reconcile._load_constituents_sector_map",
            lambda bucket: (sector_map_calls.append(bucket) or {}),
        )
        monkeypatch.setattr(
            "executor.signal_reader.patch_unknown_sectors_with_constituents",
            lambda signals_raw, bucket: 0,
        )
        monkeypatch.setattr(
            "executor.signal_reader.read_predictions",
            lambda bucket: ({"AAPL": {"predicted_alpha": 0.01}}, "2026-04-24"),
        )
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion="agentic")})
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)

        main_mod._read_signals(
            config={"signals_bucket": "test-bucket", "coverage_admission_enabled": False},
            signals_bucket="test-bucket",
            run_date="2026-04-25",
            simulate=False,
            signals_override=self._minimal_signals_override(),
            conn=None,
        )

        assert sector_map_calls == [], (
            "agentic champion must not trigger _load_constituents_sector_map "
            f"at all — got {len(sector_map_calls)} call(s)"
        )

    def test_scanner_predictor_direct_loads_sector_map_exactly_once_for_champion(self, monkeypatch):
        """When scanner_predictor_direct IS active, the champion adapter's
        own sector-map load must fire exactly once (it's gated behind the
        pointer check, not called speculatively before the pointer is known)."""
        import executor.main as main_mod

        sector_map_calls = []
        monkeypatch.setattr(
            "executor.signal_reader.filter_buy_candidates_to_universe",
            lambda s, b: s,
        )
        monkeypatch.setattr(
            "executor.eod_reconcile._load_constituents_sector_map",
            lambda bucket: (sector_map_calls.append(bucket) or {"TKR000": "Technology"}),
        )
        monkeypatch.setattr(
            "executor.signal_reader.patch_unknown_sectors_with_constituents",
            lambda signals_raw, bucket: 0,
        )
        monkeypatch.setattr(
            "executor.signal_reader.read_predictions",
            lambda bucket: ({}, "2026-04-24"),
        )
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
                # n=2 — a single-name cohort is refused (I7337).
                RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows("2026-04-24", n=2)),
            }
        )
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)

        signals_raw, *_rest = main_mod._read_signals(
            config={"signals_bucket": "test-bucket", "coverage_admission_enabled": False},
            signals_bucket="test-bucket",
            run_date="2026-04-25",
            simulate=False,
            signals_override=self._minimal_signals_override(),
            conn=None,
        )

        # Exactly one call from the champion adapter's own gated load (the
        # patch_unknown_sectors_with_constituents call below it is mocked
        # separately above and doesn't route through this same spy).
        assert sector_map_calls == ["test-bucket"], (
            f"expected exactly one sector-map fetch from the champion adapter path, got {sector_map_calls}"
        )
        assert signals_raw["buy_candidates"][0]["sector"] == "Technology"

    def test_thinktank_coverage_loads_sector_map_exactly_once_for_champion(self, monkeypatch):
        """Same wiring guarantee as the scanner arm, for thinktank_coverage
        (config-I2518 / epic I2515): the challenger-selection artifact
        carries no sector field, so this arm also needs the champion
        adapter's own sector-map load."""
        import executor.main as main_mod

        sector_map_calls = []
        monkeypatch.setattr(
            "executor.signal_reader.filter_buy_candidates_to_universe",
            lambda s, b: s,
        )
        monkeypatch.setattr(
            "executor.eod_reconcile._load_constituents_sector_map",
            lambda bucket: (sector_map_calls.append(bucket) or {"TKR000": "Technology"}),
        )
        monkeypatch.setattr(
            "executor.signal_reader.patch_unknown_sectors_with_constituents",
            lambda signals_raw, bucket: 0,
        )
        monkeypatch.setattr(
            "executor.signal_reader.read_predictions",
            lambda bucket: ({}, "2026-04-24"),
        )
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="thinktank_coverage"),
                CHALLENGER_SELECTION_LATEST_KEY: _challenger_selection_bytes(trading_day="2026-04-24", n=1),
            }
        )
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)

        signals_raw, *_rest = main_mod._read_signals(
            config={"signals_bucket": "test-bucket", "coverage_admission_enabled": False},
            signals_bucket="test-bucket",
            run_date="2026-04-25",
            simulate=False,
            signals_override=self._minimal_signals_override(),
            conn=None,
        )

        assert sector_map_calls == ["test-bucket"], (
            f"expected exactly one sector-map fetch from the champion adapter path, got {sector_map_calls}"
        )
        assert signals_raw["buy_candidates"][0]["sector"] == "Technology"


# ── Order-book stamp (item 5): champion/promotion_source attribution ──────


class TestOrderBookChampionStamp:
    """_write_order_book_summary and its call site in
    _write_stops_and_finalize must stamp champion/promotion_source
    (additive fields) sourced from signals_raw, so trades are attributable
    to the arm that produced them. Regression coverage for a real bug caught
    in review: the first cut of the _write_stops_and_finalize call site
    referenced ``signals_raw`` without it being a parameter of that
    function, which would have raised NameError the first time a live
    (non-simulate, non-dry-run) run reached this code path."""

    def _order_book(self, run_date="2026-04-25"):
        from executor.order_book import OrderBook, _default_book

        return OrderBook(_default_book(run_date))

    def test_write_order_book_summary_stamps_champion_fields(self, monkeypatch, tmp_path):
        import executor.main as main_mod

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())

        ob = self._order_book()
        main_mod._write_order_book_summary(
            ob,
            [],
            "test-bucket",
            "2026-04-25",
            champion="scanner_predictor_direct",
            promotion_source="manual_test",
        )

        assert len(put_calls) == 1
        body = json.loads(put_calls[0]["Body"])
        assert body["champion"] == "scanner_predictor_direct"
        assert body["promotion_source"] == "manual_test"

    def test_write_order_book_summary_defaults_champion_fields_to_none(self, monkeypatch):
        """Call sites that don't pass champion/promotion_source (or an
        agentic run where signals_raw carries neither) get None, not a
        hardcoded 'agentic' label — the pointer read is the single source
        of truth for that attribution."""
        import executor.main as main_mod

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())

        ob = self._order_book()
        main_mod._write_order_book_summary(ob, [], "test-bucket", "2026-04-25")

        body = json.loads(put_calls[0]["Body"])
        assert body["champion"] is None
        assert body["promotion_source"] is None

    def test_write_stops_and_finalize_threads_signals_raw_champion_fields(self, monkeypatch):
        """End-to-end: _write_stops_and_finalize must not NameError on
        ``signals_raw`` and must forward its champion/promotion_source into
        the order-book summary write."""
        import executor.main as main_mod
        from executor.ibkr import SimulatedIBKRClient

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())
        monkeypatch.setattr("executor.order_book.OrderBook.save", lambda self: None)

        ibkr = SimulatedIBKRClient(prices={}, nav=1_000_000.0)
        ob = self._order_book()
        signals_raw = {
            "champion": "scanner_predictor_direct",
            "promotion_source": "manual_test",
        }

        main_mod._write_stops_and_finalize(
            ibkr,
            ob,
            {},
            {},
            {},
            None,
            "2026-04-25",
            blocked_entries=[],
            signals_bucket="test-bucket",
            use_optimizer=False,
            signals_raw=signals_raw,
        )

        summary_calls = [c for c in put_calls if c["Key"].endswith("summary.json")]
        assert len(summary_calls) == 1
        body = json.loads(summary_calls[0]["Body"])
        assert body["champion"] == "scanner_predictor_direct"
        assert body["promotion_source"] == "manual_test"


# ── I7216: cohort age is measured and emitted on every run ───────────────────
#
# 2026-08-13: the champion cohort sat frozen at prediction_date 2026-08-07
# because its producer (the weekly pipeline's PredictorBacktest) was failing.
# Six calendar days stale — under the 8-day hard bound — so every trading day
# drew its entry candidates from the same frozen pool and every run reported a
# clean success. Distinct names newly entered fell from ~20/month to 3. The
# only place the cohort date appeared was an INFO line on the trading box.


class TestCohortStaleness:
    def test_fresh_cohort_is_not_stale(self):
        rec = evaluate_cohort_staleness("2026-08-12", "2026-08-13", {})
        assert rec["age_days"] == 1
        assert rec["is_stale"] is False

    def test_friday_cohort_used_monday_is_still_fresh(self):
        # The one legitimate multi-day gap: a Friday cohort consumed on the
        # following Monday is 3 calendar days old and must NOT flag.
        rec = evaluate_cohort_staleness("2026-08-07", "2026-08-10", {})
        assert rec["age_days"] == 3
        assert rec["is_stale"] is False

    def test_the_20260813_frozen_cohort_flags_stale(self):
        # The exact live state that went unreported.
        rec = evaluate_cohort_staleness("2026-08-07", "2026-08-13", {})
        assert rec["age_days"] == 6
        assert rec["is_stale"] is True
        assert rec["cohort_prediction_date"] == "2026-08-07"

    def test_it_never_raises_and_never_blocks(self):
        # Deliberately not a gate: halting a trading day is itself expensive
        # (sf-pipeline-policy §1.2). This classifies; _check_freshness gates.
        rec = evaluate_cohort_staleness("2020-01-02", "2026-08-13", {})
        assert rec["is_stale"] is True
        assert rec["age_days"] > 2000

    def test_the_fresh_window_is_configurable(self):
        rec = evaluate_cohort_staleness("2026-08-07", "2026-08-13", {"champion_cohort_fresh_max_days": 10})
        assert rec["is_stale"] is False
        assert rec["fresh_max_days"] == 10

    def test_the_record_is_emitted_on_a_healthy_run_too(self):
        # An absent field is unmeasured, not fine (principles.md §2.7) — so the
        # healthy path must carry the same keys as the stale one.
        healthy = evaluate_cohort_staleness("2026-08-12", "2026-08-13", {})
        stale = evaluate_cohort_staleness("2026-08-01", "2026-08-13", {})
        assert set(healthy) == set(stale)

    def test_the_fresh_window_is_tighter_than_the_hard_fail_bound(self):
        # The whole point: between "current" and "so old we refuse to trade"
        # there was no signal. If these ever converge, that gap returns.
        assert COHORT_FRESH_MAX_DAYS < 8


# ── scanner_top20_predictor (alpha-engine-config-I8755) ──────────────────────
#
# Brian, 2026-08-27: "Im saying top 20 attractiveness from scanner, which is
# evaluated weekly, gets passed to the predictor as an arm to research's
# champion/challenger" — and "the predictor's live daily output is downstream
# of what the scanner provides, correct? The scanner should be providing the
# top 20 to predictor, weekly."
#
# So the arm ranks by the PREDICTOR'S OWN alpha over the weekly cut. It reads
# nothing `_apply_scanner_predictor_direct` reads: that arm consumes the
# research-free parquet, whose cohort is the scanner's 60 and which never
# touches the predictor's cut. Measured 2026-08-27 on live artifacts — the
# parquet covered 8 of the 20 cut members, the predictor's output covered
# 20 of 20, and the two alphas ranked their 12 overlapping names at Spearman
# 0.587.

_MEMBERSHIP_LATEST = "universe_membership/latest.json"


def _membership_bytes(
    run_date: str = "2026-07-13",
    cut_name: str = "attractiveness_top_20",
    tickers: list[str] | None = None,
) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "run_date": run_date,
            "predictor_universe_cut": cut_name,
            "cuts": {
                cut_name: {"tickers": tickers if tickers is not None else ["CUT0", "CUT1", "CUT2"]},
                "scanner_champion_60": {"tickers": [f"TKR{i:03d}" for i in range(5)]},
            },
        }
    ).encode()


def _predictions(alphas: dict[str, float], *, anchor="market_relative_21d_log",
                 stance="momentum") -> dict:
    """predictions_by_ticker as the executor already holds it — records the
    PREDICTOR wrote, already on the declared anchor."""
    return {
        t: {
            "ticker": t,
            "predicted_alpha": a,
            "alpha_anchor": anchor,
            "stance": stance,
            "prediction_confidence": 0.1,
        }
        for t, a in alphas.items()
    }


class TestScannerTop20Predictor:
    def _s3(self, membership: bytes | None = None):
        objects = {CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_top20_predictor")}
        if membership is not None:
            objects[_MEMBERSHIP_LATEST] = membership
        return _FakeS3(objects)

    def _run(self, s3, preds, *, config=None):
        return apply_champion_selection(
            {"date": "2026-07-13", "buy_candidates": [], "universe": []},
            preds,
            bucket="test-bucket",
            run_date="2026-07-13",
            config=config or _CONFIG,
            sector_map={},
            s3_client=s3,
        )

    def test_ranks_the_cut_by_the_predictors_own_alpha(self):
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1", "CUT2"]))
        preds = _predictions({"CUT0": 0.01, "CUT1": 0.05, "CUT2": 0.03})

        out, _ = self._run(s3, preds, config=dict(_CONFIG, champion_top_n_default=2))

        assert [c["ticker"] for c in out["buy_candidates"]] == ["CUT1", "CUT2"]
        assert out["champion"] == "scanner_top20_predictor"

    def test_reads_no_research_free_parquet_at_all(self):
        """The arm shares no input with `scanner_predictor_direct`. The fake S3
        has no parquet object, so a read would raise NoSuchKey."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1"]))
        assert RESEARCH_FREE_PARQUET_KEY not in s3.objects

        out, _ = self._run(s3, _predictions({"CUT0": 0.02, "CUT1": 0.01}))

        assert len(out["buy_candidates"]) == 2

    def test_predictions_are_returned_UNCHANGED(self):
        """Nothing is injected. The picks ARE the predictor's output, already on
        the declared anchor — the other arms must fabricate records because
        their picks come from outside it."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1"]))
        preds = _predictions({"CUT0": 0.02, "CUT1": 0.01})
        before = json.loads(json.dumps(preds))

        out, after = self._run(s3, preds)

        assert after == before
        assert_predictions_cover_buy_candidates(out, after)

    def test_a_name_outside_the_cut_is_never_entered(self):
        """The predictor unions HELD names into its scoring universe so exits
        can be decided. This arm proposes ENTRIES, so a held name outside the
        cut is not a candidate however high its alpha."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1"]))
        preds = _predictions({"CUT0": 0.01, "CUT1": 0.005, "HELDX": 0.99})

        out, _ = self._run(s3, preds)

        assert "HELDX" not in {c["ticker"] for c in out["buy_candidates"]}

    def test_the_predictors_stance_is_carried_not_dropped(self):
        """Stance sizes the position downstream (max_position_pct x
        stance_multiplier); dropping it to None moves every pick to the default
        cap silently."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0"]))
        preds = _predictions({"CUT0": 0.02}, stance="quality")

        out, _ = self._run(s3, preds)

        assert out["buy_candidates"][0]["stance"] == "quality"

    def test_a_mixed_anchor_batch_raises(self):
        """Ranking one anchor against another orders names by where their level
        was measured from, not by their alpha."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1"]))
        preds = _predictions({"CUT0": 0.02, "CUT1": 0.01})
        preds["CUT1"]["alpha_anchor"] = "raw_21d_log"

        with pytest.raises(AlphaAnchorError, match="mixed-anchor"):
            self._run(s3, preds)

    def test_a_partially_scored_cut_is_REPORTED_not_refused(self):
        """The arm still selects from what it has — but an arm scored on a
        short pool must not read as an arm that chose badly."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1", "CUT2", "CUT3"]))
        preds = _predictions({"CUT0": 0.02, "CUT1": 0.01})

        out, _ = self._run(s3, preds)

        block = out["champion_cohort"]
        assert block["pool_declared_size"] == 4
        assert block["pool_size"] == 2
        assert block["n_cut_members_unscored"] == 2
        assert block["pool_cut"] == "attractiveness_top_20"
        assert block["pool_source"] == "predictor_predictions"

    def test_a_wholly_unscored_cut_raises(self):
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1"]))

        with pytest.raises(ChampionPointerError, match="usable predicted_alpha"):
            self._run(s3, _predictions({"OTHER": 0.02}))

    def test_missing_membership_raises_rather_than_widening(self):
        """A silent widening would make this arm a duplicate of one it is
        measured against — a vacuous comparison presented as a real one."""
        with pytest.raises(ChampionPointerError, match="no universe membership"):
            self._run(self._s3(None), _predictions({"CUT0": 0.02}))

    def test_the_cut_name_comes_from_the_artifact_not_a_literal(self):
        s3 = self._s3(_membership_bytes(cut_name="attractiveness_top_25", tickers=["CUT9"]))

        out, _ = self._run(s3, _predictions({"CUT9": 0.02}))

        assert out["champion_cohort"]["pool_cut"] == "attractiveness_top_25"

    def test_the_best_name_scores_at_the_ceiling(self):
        """The band is ranked within the arm's OWN pool. Ranking against a
        wider cross-section would push its picks under `min_score` and make the
        arm look like it selected nothing."""
        s3 = self._s3(_membership_bytes(tickers=["CUT0", "CUT1", "CUT2"]))

        out, _ = self._run(s3, _predictions({"CUT0": 0.01, "CUT1": 0.05, "CUT2": 0.03}))

        assert out["buy_candidates"][0]["ticker"] == "CUT1"
        assert out["buy_candidates"][0]["score"] == pytest.approx(_CONFIG["champion_score_ceiling"])


class TestScannerPredictorDirectIsUntouched:
    """The incumbent stays a challenger and must be bit-identical."""

    def test_it_still_selects_from_the_whole_research_free_cohort(self):
        s3 = _FakeS3({
            CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
            RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows("2026-07-10", 5)),
        })

        out, _ = apply_champion_selection(
            {"date": "2026-07-13", "buy_candidates": [], "universe": []},
            {}, bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )

        assert {c["ticker"] for c in out["buy_candidates"]} == {f"TKR{i:03d}" for i in range(5)}
        assert out["champion_cohort"]["pool_source"] == "research_free_parquet"
        for c in out["buy_candidates"]:
            assert c["champion_arm"] == "scanner_predictor_direct"

    def test_it_never_reads_the_membership_artifact(self):
        class _Recording(_FakeS3):
            def __init__(self, objects):
                super().__init__(objects)
                self.reads: list[str] = []

            def get_object(self, Bucket, Key):  # noqa: N803
                self.reads.append(Key)
                return super().get_object(Bucket=Bucket, Key=Key)

        s3 = _Recording({
            CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
            RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows("2026-07-10", 5)),
        })

        apply_champion_selection(
            {"date": "2026-07-13", "buy_candidates": [], "universe": []},
            {}, bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )

        assert not any(k.startswith("universe_membership/") for k in s3.reads), s3.reads


# ── The generic shadow-signals arm handler (alpha-engine-config-I9299) ──────


def _shadow_key(arm: str, date: str) -> str:
    return SHADOW_SIGNALS_KEY_TEMPLATE.format(arm=arm, date=date)


def _shadow_bytes(
    arm: str,
    date: str,
    *,
    entries: list[tuple[str, float]] | None = None,
    holds: list[str] | None = None,
    producer: str | None = ...,  # type: ignore[assignment]
    declared_date: str | None = ...,  # type: ignore[assignment]
) -> bytes:
    """A conforming ``arm_shadow_signals.schema.json`` document.

    Built from the schema's declared shape, not copied from a live artifact:
    the shape under test is ``signals`` keyed by ticker with ``signal`` and a
    numeric ``score``, which is what the shared scorer reads.
    """
    entries = entries if entries is not None else [("AAA", 91.0), ("BBB", 77.0), ("CCC", 64.0)]
    signals: dict[str, dict] = {
        t: {"ticker": t, "signal": "ENTER", "score": s, "champion_arm": arm}
        for t, s in entries
    }
    for t in holds or []:
        signals[t] = {"ticker": t, "signal": "HOLD", "score": 12.0}
    doc: dict = {"schema_version": 1, "signals": signals}
    if producer is not ...:
        doc["producer"] = producer
    else:
        doc["producer"] = arm
    if declared_date is not ...:
        doc["date"] = declared_date
    else:
        doc["date"] = date
    return json.dumps(doc).encode()


_SHADOW_CONFIG = {
    "champion_freshness_max_days": 8,
    "champion_top_n_default": 2,
    "champion_score_floor": 60,
    "champion_score_ceiling": 95,
}


class TestShadowSignalsArm:
    """The two arms Brian's 2026-08-29 ruling made promotion-eligible."""

    RUN_DATE = "2026-08-31"

    def _apply(self, arm="no_agent_quant", objects=None, config=None, signals_raw=None):
        if objects is None:
            objects = {_shadow_key(arm, self.RUN_DATE): _shadow_bytes(arm, self.RUN_DATE)}
        s3 = _FakeS3(objects)
        return apply_champion_selection(
            signals_raw if signals_raw is not None else {"date": self.RUN_DATE, "buy_candidates": [], "universe": []},
            {},
            bucket="test-bucket",
            run_date=self.RUN_DATE,
            config=config or _SHADOW_CONFIG,
            sector_map={"AAA": "Tech"},
            s3_client=s3,
            pointer={"schema_version": 1, "champion": arm, "promotion_source": "gate_engine"},
        )

    @pytest.mark.parametrize("arm", ["no_agent_quant", "single_agent_quant"])
    def test_serves_both_eligible_arms(self, arm):
        signals_raw, preds = self._apply(arm=arm)
        assert signals_raw["champion"] == arm
        tickers = [c["ticker"] for c in signals_raw["buy_candidates"]]
        assert tickers == ["AAA", "BBB"]  # champion_top_n_default=2, best first
        assert all(c["champion_arm"] == arm for c in signals_raw["buy_candidates"])
        assert set(preds) == {"AAA", "BBB"}

    def test_pointer_for_a_shadow_arm_is_accepted_by_the_loader(self):
        """The regression I9299 exists to prevent: promoting onto one of these
        arms used to raise at planner start and HALT trading."""
        for arm in ("no_agent_quant", "single_agent_quant"):
            s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes(champion=arm)})
            assert load_champion_pointer("test-bucket", s3_client=s3)["champion"] == arm

    def test_hold_entries_are_ignored_and_enter_entries_ranked(self):
        objects = {
            _shadow_key("no_agent_quant", self.RUN_DATE): _shadow_bytes(
                "no_agent_quant", self.RUN_DATE,
                entries=[("LOW", 61.0), ("HIGH", 99.0), ("MID", 80.0)],
                holds=["ZZZ", "YYY"],
            )
        }
        signals_raw, _ = self._apply(objects=objects, config={**_SHADOW_CONFIG, "champion_top_n_default": 3})
        assert [c["ticker"] for c in signals_raw["buy_candidates"]] == ["HIGH", "MID", "LOW"]

    def test_score_is_remapped_onto_the_shared_band_not_carried_raw(self):
        """The arm's own composite is 0-100; the served score must be on the
        same band every other arm uses or min_score_to_enter would gate the
        arms at different effective thresholds (§4)."""
        signals_raw, _ = self._apply(
            config={**_SHADOW_CONFIG, "champion_top_n_default": 3},
        )
        scores = [c["score"] for c in signals_raw["buy_candidates"]]
        assert scores[0] == 95.0
        assert scores[-1] == 60.0
        assert scores == sorted(scores, reverse=True)
        # ...and the arm's own score is preserved for forensics.
        assert [c["arm_score"] for c in signals_raw["buy_candidates"]] == [91.0, 77.0, 64.0]

    def test_ties_break_on_ticker_so_the_served_set_is_reproducible(self):
        objects = {
            _shadow_key("no_agent_quant", self.RUN_DATE): _shadow_bytes(
                "no_agent_quant", self.RUN_DATE, entries=[("ZZZ", 70.0), ("AAA", 70.0)],
            )
        }
        signals_raw, _ = self._apply(objects=objects)
        assert [c["ticker"] for c in signals_raw["buy_candidates"]] == ["AAA", "ZZZ"]

    def test_count_matches_live_buy_candidates_when_present(self):
        signals_raw, _ = self._apply(
            signals_raw={
                "date": self.RUN_DATE,
                "buy_candidates": [{"ticker": "OLD1"}],
                "universe": [],
            },
        )
        assert len(signals_raw["buy_candidates"]) == 1

    def test_injected_predictions_assert_no_alpha_and_are_confidence_neutral(self):
        _, preds = self._apply()
        for row in preds.values():
            assert row["predicted_alpha"] is None
            assert row["predicted_direction"] is None
            assert row["prediction_confidence"] == 0.0
            assert "alpha_anchor" not in row

    def test_injected_predictions_cover_the_synthesized_candidates(self):
        signals_raw, preds = self._apply()
        assert_predictions_cover_buy_candidates(signals_raw, preds)

    def test_champion_cohort_block_is_emitted_on_the_healthy_path(self):
        signals_raw, _ = self._apply()
        cohort = signals_raw["champion_cohort"]
        assert cohort["cohort_prediction_date"] == self.RUN_DATE
        assert cohort["age_days"] == 0
        assert cohort["is_stale"] is False
        assert cohort["pool_source"] == "shadow_signals:no_agent_quant"
        assert cohort["pool_key"] == _shadow_key("no_agent_quant", self.RUN_DATE)

    def test_sector_is_stamped_from_the_sector_map(self):
        signals_raw, _ = self._apply()
        by_ticker = {c["ticker"]: c["sector"] for c in signals_raw["buy_candidates"]}
        assert by_ticker["AAA"] == "Tech"
        assert by_ticker["BBB"] == "Unknown"

    # ── failure modes ───────────────────────────────────────────────────

    def test_absent_artifact_raises_and_says_absent_not_stale(self):
        with pytest.raises(ChampionPointerError) as exc:
            self._apply(objects={})
        assert "ABSENT" in str(exc.value)

    def test_stale_cohort_raises_stale_not_absent(self):
        """A shadow that exists but is older than the serving bound is a
        DIFFERENT failure from one that was never written."""
        old = "2026-08-10"  # 21 days before RUN_DATE, inside the 30d lookback
        with pytest.raises(StaleChampionFeedError):
            self._apply(objects={_shadow_key("no_agent_quant", old): _shadow_bytes("no_agent_quant", old)})

    def test_malformed_json_raises(self):
        with pytest.raises(ChampionPointerError):
            self._apply(objects={_shadow_key("no_agent_quant", self.RUN_DATE): b"{not json"})

    def test_non_object_document_raises(self):
        with pytest.raises(ChampionPointerError):
            self._apply(objects={_shadow_key("no_agent_quant", self.RUN_DATE): b"[1,2,3]"})

    def test_missing_signals_object_raises(self):
        payload = json.dumps({"date": self.RUN_DATE, "producer": "no_agent_quant"}).encode()
        with pytest.raises(ChampionPointerError):
            self._apply(objects={_shadow_key("no_agent_quant", self.RUN_DATE): payload})

    def test_zero_enter_picks_raises_rather_than_serving_an_empty_book(self):
        objects = {
            _shadow_key("no_agent_quant", self.RUN_DATE): _shadow_bytes(
                "no_agent_quant", self.RUN_DATE, entries=[], holds=["AAA", "BBB"],
            )
        }
        with pytest.raises(ChampionPointerError) as exc:
            self._apply(objects=objects)
        assert "MISS" in str(exc.value)

    def test_non_numeric_score_on_an_enter_pick_raises(self):
        doc = {
            "date": self.RUN_DATE,
            "producer": "no_agent_quant",
            "signals": {"AAA": {"ticker": "AAA", "signal": "ENTER", "score": None}},
        }
        with pytest.raises(ChampionPointerError):
            self._apply(objects={_shadow_key("no_agent_quant", self.RUN_DATE): json.dumps(doc).encode()})

    def test_producer_mismatch_raises(self):
        objects = {
            _shadow_key("no_agent_quant", self.RUN_DATE): _shadow_bytes(
                "no_agent_quant", self.RUN_DATE, producer="single_agent_quant",
            )
        }
        with pytest.raises(ChampionPointerError) as exc:
            self._apply(objects=objects)
        assert "single_agent_quant" in str(exc.value)

    def test_cohort_date_mismatch_raises(self):
        objects = {
            _shadow_key("no_agent_quant", self.RUN_DATE): _shadow_bytes(
                "no_agent_quant", self.RUN_DATE, declared_date="2026-01-01",
            )
        }
        with pytest.raises(ChampionPointerError):
            self._apply(objects=objects)

    def test_non_404_s3_error_raises_rather_than_walking_past_it(self):
        class _DeniedS3:
            def get_object(self, Bucket, Key):  # noqa: N803
                raise ClientError(
                    error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
                    operation_name="GetObject",
                )

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                {"date": self.RUN_DATE, "buy_candidates": [], "universe": []},
                {},
                bucket="test-bucket",
                run_date=self.RUN_DATE,
                config=_SHADOW_CONFIG,
                sector_map={},
                s3_client=_DeniedS3(),
                pointer={"schema_version": 1, "champion": "no_agent_quant"},
            )

    def test_walk_back_picks_the_NEWEST_cohort_within_the_window(self):
        objects = {
            _shadow_key("no_agent_quant", "2026-08-29"): _shadow_bytes(
                "no_agent_quant", "2026-08-29", entries=[("NEW", 90.0)],
            ),
            _shadow_key("no_agent_quant", "2026-08-25"): _shadow_bytes(
                "no_agent_quant", "2026-08-25", entries=[("OLD", 90.0)],
            ),
        }
        signals_raw, _ = self._apply(objects=objects)
        assert [c["ticker"] for c in signals_raw["buy_candidates"]] == ["NEW"]
        assert signals_raw["champion_cohort"]["cohort_prediction_date"] == "2026-08-29"


class TestServingRegisterIsDerived:
    """alpha-engine-config-I9299 deliverable 4 — the allowlist IS the dispatch."""

    def test_every_valid_champion_resolves_to_exactly_one_way_of_being_served(self):
        from executor import champion as champ

        for arm in champ.VALID_CHAMPIONS:
            ways = [
                arm in champ.NOOP_CHAMPION_ARMS_SERVED,
                arm in champ._DEDICATED_ARM_HANDLERS,
                arm in champ.SHADOW_SERVED_ARMS,
            ]
            assert sum(ways) == 1, f"{arm} is served in {sum(ways)} ways"

    def test_valid_champions_is_not_a_standalone_literal(self):
        """The defect: a hand-typed tuple 70 lines from the dispatch it had to
        agree with. It went stale twice. Assert it is BUILT, not typed."""
        import pathlib
        import re

        src = pathlib.Path(champion_module_path()).read_text()
        m = re.search(r"^VALID_CHAMPIONS = (.*?)^\n", src, re.M | re.S)
        assert m, "VALID_CHAMPIONS assignment not found"
        body = m.group(1)
        assert "NOOP_CHAMPION_ARMS_SERVED" in body
        assert "_DEDICATED_ARM_HANDLERS" in body
        assert "SHADOW_SERVED_ARMS" in body
        assert '"agentic"' not in body

    def test_the_two_eligible_arms_are_servable(self):
        from executor.champion import VALID_CHAMPIONS

        assert "no_agent_quant" in VALID_CHAMPIONS
        assert "single_agent_quant" in VALID_CHAMPIONS


#: crucible-research's `producers.registry.research_slot_producers()` as of
#: 2026-09-22 (alpha-engine-config-I11393). Restated here because this repo
#: cannot import that one; alpha-engine-config-I11442 removes the restatement
#: by deriving it from `arena/research/register.json`.
RESEARCH_SLOT_ARMS = (
    "attractiveness_60",
    "attractiveness_20",
    "tech_score_20",
    "predictor_from_60",
    "thinktank_20",
)


class TestTheResearchSlotIsServable:
    """alpha-engine-config-I11438. Until this, NOT ONE arm of the live research
    slot was in `VALID_CHAMPIONS` — every name there was a RETIRED arm.

    The consequence was not a degraded run. `apply_champion_selection` raises
    `ChampionPointerError` on an unknown champion, at PLANNER START, so the
    first promotion onto a research arm would have HALTED TRADING — the exact
    failure this module's -I9299 comment describes having been designed against
    once already, one slot later.

    VERIFIED RED: with the five names removed from `SHADOW_SERVED_ARMS`, every
    test in this class fails, and `test_a_research_arm_pointer_does_not_raise`
    fails with the literal production error.
    """

    def test_every_research_slot_arm_is_servable(self):
        from executor.champion import VALID_CHAMPIONS

        missing = [a for a in RESEARCH_SLOT_ARMS if a not in VALID_CHAMPIONS]
        assert not missing, (
            f"{missing} are live arms of the research slot and cannot be "
            "served — a promotion onto one raises at planner start and halts "
            "trading (alpha-engine-config-I11438)"
        )

    def test_they_are_served_generically_not_by_new_branches(self):
        """ONE handler, not five more per-arm branches. A per-arm branch
        differing only in an S3 prefix is what produced three divergent arm
        registers (-I9299)."""
        from executor import champion as champ

        for arm in RESEARCH_SLOT_ARMS:
            assert arm in champ.SHADOW_SERVED_ARMS
            assert arm not in champ._DEDICATED_ARM_HANDLERS

    def test_they_get_a_sector_stamped(self):
        """They synthesize `buy_candidates` from an artifact carrying no sector
        of its own. Missing this is silent — `Unknown` is a legal value — and it
        switches the sector-concentration cap off for that arm's entries, which
        is how `scanner_top20_predictor` went unnoticed for weeks."""
        from executor.champion import ARMS_REQUIRING_SECTOR_MAP

        for arm in RESEARCH_SLOT_ARMS:
            assert arm in ARMS_REQUIRING_SECTOR_MAP

    def test_a_research_arm_pointer_does_not_raise(self):
        """The production failure, directly: resolve the guard the planner hits
        on its first call with a research champion."""
        from executor.champion import VALID_CHAMPIONS

        for arm in RESEARCH_SLOT_ARMS:
            # The same membership test `apply_champion_selection` performs
            # before dispatching; a miss there is ChampionPointerError.
            assert arm in VALID_CHAMPIONS, arm

    def test_sector_map_arm_set_is_derived_and_covers_every_synthesizing_arm(self):
        from executor.champion import (
            ARMS_REQUIRING_SECTOR_MAP,
            NOOP_CHAMPION_ARMS_SERVED,
            VALID_CHAMPIONS,
        )

        assert set(ARMS_REQUIRING_SECTOR_MAP) == set(VALID_CHAMPIONS) - set(NOOP_CHAMPION_ARMS_SERVED)

    def test_main_does_not_carry_its_own_arm_literal(self):
        import pathlib

        src = pathlib.Path(champion_module_path()).parent.joinpath("main.py").read_text()
        assert 'in ("scanner_predictor_direct", "thinktank_coverage")' not in src
        assert "ARMS_REQUIRING_SECTOR_MAP" in src


def champion_module_path() -> str:
    from executor import champion as champ

    return champ.__file__
