"""Strategy identity on ``order_books/{date}/summary.json`` (alpha-engine-config-I11089).

The summary already named the ARM (``champion`` / ``promotion_source``). It did
not name the executor code or the configuration a session traded on, and the
pointer version had to be reconstructed from the S3 version history by
plan-time timestamp. These tests pin the three additive fields:

* ``executor_sha`` — the checkout's git HEAD;
* ``config_sha256`` — hash of the effective run config;
* ``champion_pointer_version`` — S3 VersionId of the pointer object read.
"""

from __future__ import annotations

import hashlib
import io
import json

from botocore.exceptions import ClientError

from executor.champion import (
    CHAMPION_POINTER_KEY,
    load_champion_pointer,
    load_champion_pointer_versioned,
)

IDENTITY_FIELDS = ("executor_sha", "config_sha256", "champion_pointer_version")


class _VersionedFakeS3:
    """get_object over {key: (bytes, version_id | None)}."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                operation_name="GetObject",
            )
        body, version_id = self.objects[Key]
        resp = {"Body": io.BytesIO(body)}
        if version_id is not None:
            resp["VersionId"] = version_id
        return resp


def _pointer(champion="scanner_predictor_direct") -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "champion": champion,
            "promoted_at": "2026-10-10T17:17:56Z",
            "promotion_source": "arena_decided",
        }
    ).encode()


class _FakeS3Put:
    def __init__(self):
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)


def _order_book(run_date="2026-10-13"):
    from executor.order_book import OrderBook, _default_book

    return OrderBook(_default_book(run_date))


# ── pointer VersionId ────────────────────────────────────────────────────


class TestPointerVersion:
    def test_versioned_read_returns_the_objects_version_id(self):
        s3 = _VersionedFakeS3({CHAMPION_POINTER_KEY: (_pointer(), "i.L5kWQVIk5rgy")})
        pointer, version_id = load_champion_pointer_versioned("b", s3_client=s3)
        assert pointer["champion"] == "scanner_predictor_direct"
        assert version_id == "i.L5kWQVIk5rgy"

    def test_404_default_has_no_version(self):
        pointer, version_id = load_champion_pointer_versioned("b", s3_client=_VersionedFakeS3())
        assert pointer["champion"] == "agentic"
        assert version_id is None

    def test_unversioned_response_yields_none(self):
        s3 = _VersionedFakeS3({CHAMPION_POINTER_KEY: (_pointer(), None)})
        _pointer_dict, version_id = load_champion_pointer_versioned("b", s3_client=s3)
        assert version_id is None

    def test_plain_loader_still_returns_the_pointer_dict_only(self):
        s3 = _VersionedFakeS3({CHAMPION_POINTER_KEY: (_pointer(), "v1")})
        pointer = load_champion_pointer("b", s3_client=s3)
        assert isinstance(pointer, dict)
        assert pointer["champion"] == "scanner_predictor_direct"
        assert "VersionId" not in pointer


# ── config hash ──────────────────────────────────────────────────────────


class TestConfigSha256:
    def test_is_independent_of_key_order(self):
        import executor.main as main_mod

        a = {"x": 1, "strategy": {"b": 2, "a": [1, 2]}}
        b = {"strategy": {"a": [1, 2], "b": 2}, "x": 1}
        assert main_mod._config_sha256(a) == main_mod._config_sha256(b)

    def test_matches_the_documented_canonical_form(self):
        import executor.main as main_mod

        cfg = {"max_position_pct": 0.05, "strategy": {"min_score_to_enter": 60}}
        canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
        assert main_mod._config_sha256(cfg) == hashlib.sha256(canonical.encode()).hexdigest()

    def test_moves_when_a_value_moves(self):
        import executor.main as main_mod

        assert main_mod._config_sha256({"k": 1}) != main_mod._config_sha256({"k": 2})

    def test_none_config_and_unserializable_config_yield_none(self):
        import executor.main as main_mod

        assert main_mod._config_sha256(None) is None
        # Mixed key types cannot be sorted — logged, not raised.
        assert main_mod._config_sha256({1: "a", "b": 2}) is None


# ── executor SHA ─────────────────────────────────────────────────────────


class TestExecutorSha:
    def test_reads_through_the_preflight_reader(self, monkeypatch):
        import executor.main as main_mod

        monkeypatch.setattr(
            "executor.preflight._read_deployed_git_sha", lambda root: "a" * 40
        )
        assert main_mod._executor_sha() == "a" * 40

    def test_unreadable_checkout_yields_none_not_a_raise(self, monkeypatch):
        import executor.main as main_mod

        def _boom(root):
            raise RuntimeError("Deploy drift: no .git directory")

        monkeypatch.setattr("executor.preflight._read_deployed_git_sha", _boom)
        assert main_mod._executor_sha() is None


# ── the summary body ─────────────────────────────────────────────────────


class TestSummaryBody:
    def test_writes_all_three_identity_fields(self, monkeypatch):
        import executor.main as main_mod

        s3 = _FakeS3Put()
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        main_mod._write_order_book_summary(
            _order_book(),
            [],
            "test-bucket",
            "2026-10-13",
            champion="scanner_predictor_direct",
            promotion_source="arena_decided",
            executor_sha="b" * 40,
            config_sha256="c" * 64,
            champion_pointer_version="i.L5kWQVIk5rgy",
        )
        body = json.loads(s3.calls[0]["Body"])
        assert body["executor_sha"] == "b" * 40
        assert body["config_sha256"] == "c" * 64
        assert body["champion_pointer_version"] == "i.L5kWQVIk5rgy"
        # Pre-existing fields are untouched.
        assert body["champion"] == "scanner_predictor_direct"
        assert body["entries_approved"] == []

    def test_identity_keys_are_present_as_null_when_unknown(self, monkeypatch):
        """Key presence distinguishes 'not recorded this run' (null) from
        'a summary written before I11089' (key absent)."""
        import executor.main as main_mod

        s3 = _FakeS3Put()
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        main_mod._write_order_book_summary(_order_book(), [], "test-bucket", "2026-10-13")
        body = json.loads(s3.calls[0]["Body"])
        for field in IDENTITY_FIELDS:
            assert field in body
            assert body[field] is None


# ── wiring: _read_signals → _write_stops_and_finalize → summary ──────────


class TestWiring:
    def test_read_signals_stamps_the_pointer_version_it_read(self, monkeypatch):
        import executor.main as main_mod

        monkeypatch.setattr(
            "executor.signal_reader.filter_buy_candidates_to_universe", lambda s, b: s
        )
        monkeypatch.setattr(
            "executor.signal_reader.patch_unknown_sectors_with_constituents",
            lambda signals_raw, bucket: 0,
        )
        monkeypatch.setattr(
            "executor.signal_reader.read_predictions",
            lambda bucket: ({"AAPL": {"predicted_alpha": 0.01}}, "2026-10-12"),
        )
        s3 = _VersionedFakeS3({CHAMPION_POINTER_KEY: (_pointer("agentic"), "ptr-v7")})
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        override = {
            "date": "2026-10-13",
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

        signals_raw, *_rest = main_mod._read_signals(
            config={"signals_bucket": "test-bucket", "coverage_admission_enabled": False},
            signals_bucket="test-bucket",
            run_date="2026-10-13",
            simulate=False,
            signals_override=override,
            conn=None,
        )

        assert signals_raw["champion_pointer_version"] == "ptr-v7"
        assert "champion_pointer_version" not in override, "caller's dict must not be mutated"

    def test_simulate_mode_does_not_read_the_pointer_or_stamp_a_version(self):
        import executor.main as main_mod

        override = {"date": "2026-10-13", "buy_candidates": [], "universe": []}
        signals_raw, *_rest = main_mod._read_signals(
            config={"signals_bucket": "test-bucket"},
            signals_bucket="test-bucket",
            run_date="2026-10-13",
            simulate=True,
            signals_override=override,
            conn=None,
        )
        assert "champion_pointer_version" not in signals_raw

    def test_finalize_threads_all_identity_fields_into_the_summary(self, monkeypatch):
        import executor.main as main_mod
        from executor.ibkr import SimulatedIBKRClient

        s3 = _FakeS3Put()
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        monkeypatch.setattr("executor.order_book.OrderBook.save", lambda self: None)
        monkeypatch.setattr(
            "executor.preflight._read_deployed_git_sha", lambda root: "d" * 40
        )
        monkeypatch.setattr(
            "executor.zero_entries_alarm.check_zero_entries_floor", lambda *a, **kw: None
        )
        config = {"max_position_pct": 0.05}

        main_mod._write_stops_and_finalize(
            SimulatedIBKRClient(prices={}, nav=1_000_000.0),
            _order_book(),
            {},
            {},
            {},
            None,
            "2026-10-13",
            blocked_entries=[],
            signals_bucket="test-bucket",
            use_optimizer=False,
            signals_raw={
                "champion": "scanner_predictor_direct",
                "promotion_source": "arena_decided",
                "champion_pointer_version": "ptr-v7",
            },
            config=config,
        )

        summary = [c for c in s3.calls if c["Key"].endswith("summary.json")]
        assert len(summary) == 1
        body = json.loads(summary[0]["Body"])
        assert body["executor_sha"] == "d" * 40
        assert body["config_sha256"] == main_mod._config_sha256(config)
        assert body["champion_pointer_version"] == "ptr-v7"
