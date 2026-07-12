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

from executor.champion import (
    CHAMPION_POINTER_KEY,
    RESEARCH_FREE_PARQUET_KEY,
    ChampionPointerError,
    StaleChampionFeedError,
    apply_champion_selection,
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
            signals_raw, predictions_by_ticker,
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
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
            signals_raw, predictions_by_ticker,
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )

        assert out_signals is signals_raw
        assert out_preds is predictions_by_ticker


# ── apply_champion_selection: scanner_predictor_direct ──────────────────


class TestScannerPredictorDirect:
    def _s3(self, cohort_date="2026-07-10", n=5, champion="scanner_predictor_direct"):
        return _FakeS3({
            CHAMPION_POINTER_KEY: _pointer_bytes(champion=champion),
            RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows(cohort_date, n)),
        })

    def test_synthesized_entries_route_to_enter(self):
        signals_raw = {
            "date": "2026-07-13",
            "buy_candidates": [],  # empty → falls back to champion_top_n_default
            "universe": [{"ticker": "HOLDX", "signal": "HOLD", "sector": "Technology"}],
        }
        s3 = self._s3(cohort_date="2026-07-10", n=3)

        out_signals, out_preds = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={"TKR000": "Technology"}, s3_client=s3,
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
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
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
        s3 = self._s3(n=1)

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
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
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_empty_buy_candidates_uses_top_n_default(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=5)
        cfg = dict(_CONFIG, champion_top_n_default=3)

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=cfg, sector_map={}, s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 3

    def test_stale_cohort_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # cohort is 20 days before run_date; max allowed is 8.
        s3 = self._s3(cohort_date="2026-06-23", n=3)

        with pytest.raises(StaleChampionFeedError):
            apply_champion_selection(
                signals_raw, {},
                bucket="test-bucket", run_date="2026-07-13",
                config=_CONFIG, sector_map={}, s3_client=s3,
            )

    def test_cohort_within_freshness_window_does_not_raise(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        # exactly 8 days old — boundary, should be allowed (<=).
        s3 = self._s3(cohort_date="2026-07-05", n=2)

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )
        assert len(out_signals["buy_candidates"]) == 2

    def test_rank_to_score_monotonic_within_cohort(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=6)
        cfg = dict(_CONFIG, champion_top_n_default=6)

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=cfg, sector_map={}, s3_client=s3,
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
        s3 = _FakeS3({
            CHAMPION_POINTER_KEY: _pointer_bytes(),
            RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(bad_rows),
        })

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw, {},
                bucket="test-bucket", run_date="2026-07-13",
                config=_CONFIG, sector_map={}, s3_client=s3,
            )

    def test_missing_parquet_raises(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = _FakeS3({CHAMPION_POINTER_KEY: _pointer_bytes()})  # no parquet key

        with pytest.raises(ChampionPointerError):
            apply_champion_selection(
                signals_raw, {},
                bucket="test-bucket", run_date="2026-07-13",
                config=_CONFIG, sector_map={}, s3_client=s3,
            )

    def test_sector_map_applied_to_synthesized_entries(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=2)
        sector_map = {"TKR000": "Health Care", "TKR001": "Financials"}

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map=sector_map, s3_client=s3,
        )
        by_ticker = {e["ticker"]: e for e in out_signals["buy_candidates"]}
        assert by_ticker["TKR000"]["sector"] == "Health Care"
        assert by_ticker["TKR001"]["sector"] == "Financials"

    def test_stamps_champion_and_promotion_source_on_signals_raw(self):
        signals_raw = {"date": "2026-07-13", "buy_candidates": [], "universe": []}
        s3 = self._s3(n=1)

        out_signals, _ = apply_champion_selection(
            signals_raw, {},
            bucket="test-bucket", run_date="2026-07-13",
            config=_CONFIG, sector_map={}, s3_client=s3,
        )
        assert out_signals["champion"] == "scanner_predictor_direct"
        assert out_signals["promotion_source"] == "manual_test"


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
        import executor.main as main_mod

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
        s3 = _FakeS3({
            CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
            RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(_cohort_rows("2026-04-24", n=1)),
        })
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
            f"expected exactly one sector-map fetch from the champion adapter "
            f"path, got {sector_map_calls}"
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
            ob, [], "test-bucket", "2026-04-25",
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
            ibkr, ob, {}, {}, {}, None, "2026-04-25",
            blocked_entries=[], signals_bucket="test-bucket",
            use_optimizer=False, signals_raw=signals_raw,
        )

        summary_calls = [c for c in put_calls if c["Key"].endswith("summary.json")]
        assert len(summary_calls) == 1
        body = json.loads(summary_calls[0]["Body"])
        assert body["champion"] == "scanner_predictor_direct"
        assert body["promotion_source"] == "manual_test"
