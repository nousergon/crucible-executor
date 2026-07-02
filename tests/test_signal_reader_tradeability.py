"""Tests for read_universe_tradeability / extract_adv_usd — the executor's
reader for the scanner per-name tradeability block (config#1401).

The scanner (crucible-research#343) emits, on
``scanner/universe/{date}/universe.json``, a schema-v3 board where each
``stocks[]`` entry carries a ``tradeability`` block including ``adv_usd``. The
executor lifts ADV$ into the portfolio optimizer's participation-aware √-impact
cost term + max-%-ADV constraint and the position sizer's ADV size cap.

Fail-soft contract: tradeability is a construction refinement, NOT a gate — so
EVERY read failure mode (missing artifact, AccessDenied, no credentials,
endpoint/connection error, malformed payload) must degrade to an empty map, so
the optimizer falls back to the flat L1 turnover penalty rather than crashing.
This is the #321 CI-red regression guard: a no-AWS-creds environment raises
``NoCredentialsError`` (a ``BotoCoreError``, NOT a ``ClientError``) deep in
botocore, and that must be swallowed.
"""
from __future__ import annotations

import io
import json
import math
from unittest.mock import MagicMock, patch

from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)

from executor.signal_reader import extract_adv_usd, read_universe_tradeability


def _fake_s3(payload: dict):
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode())}
    return s3


def _s3_raising(exc):
    s3 = MagicMock()
    s3.get_object.side_effect = exc
    return s3


def _board(**over):
    stocks = over.get("stocks", [
        {"ticker": "AAPL", "tradeability": {"adv_usd": 8.0e9, "tradeability_score": 95.0}},
        {"ticker": "MSFT", "tradeability": {"adv_usd": 6.0e9, "tradeability_score": 92.0}},
        {"ticker": "THIN", "tradeability": {"adv_usd": None, "tradeability_score": None}},
    ])
    return {"schema_version": 3, "as_of": "2026-05-11", "stocks": stocks}


# ── happy path ───────────────────────────────────────────────────────────────


def test_returns_tradeability_map_on_success():
    with patch("executor.signal_reader.boto3.client", return_value=_fake_s3(_board())):
        out = read_universe_tradeability("bucket", "2026-05-11")
    assert set(out) == {"AAPL", "MSFT", "THIN"}
    assert out["AAPL"]["adv_usd"] == 8.0e9


def test_reads_dated_key():
    s3 = _fake_s3(_board())
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        read_universe_tradeability("bucket", "2026-05-11")
    assert s3.get_object.call_args.kwargs["Key"] == "scanner/universe/2026-05-11/universe.json"


# ── fail-soft matrix — every failure mode → {} ───────────────────────────────


def test_missing_artifact_nosuchkey_returns_empty():
    err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    with patch("executor.signal_reader.boto3.client", return_value=_s3_raising(err)):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_access_denied_clienterror_returns_empty():
    err = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    with patch("executor.signal_reader.boto3.client", return_value=_s3_raising(err)):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_no_credentials_returns_empty():
    """#321 regression: NoCredentialsError subclasses BotoCoreError, not
    ClientError — the broadened catch must swallow it (CI / no-creds box)."""
    with patch("executor.signal_reader.boto3.client",
               return_value=_s3_raising(NoCredentialsError())):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_endpoint_connection_error_returns_empty():
    err = EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")
    with patch("executor.signal_reader.boto3.client", return_value=_s3_raising(err)):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_malformed_json_returns_empty():
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"{not valid json")}
    with patch("executor.signal_reader.boto3.client", return_value=s3):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_no_stocks_key_returns_empty_map():
    with patch("executor.signal_reader.boto3.client",
               return_value=_fake_s3({"schema_version": 3})):
        assert read_universe_tradeability("bucket", "2026-05-11") == {}


def test_skips_entries_without_ticker_or_block():
    board = _board(stocks=[
        {"ticker": "AAPL", "tradeability": {"adv_usd": 1.0e9}},
        {"tradeability": {"adv_usd": 2.0e9}},          # no ticker
        {"ticker": "MSFT"},                             # no block
        {"ticker": "NULLB", "tradeability": None},      # block None
        "junk",                                         # not a dict
    ])
    with patch("executor.signal_reader.boto3.client", return_value=_fake_s3(board)):
        out = read_universe_tradeability("bucket", "2026-05-11")
    assert set(out) == {"AAPL"}


# ── extract_adv_usd ──────────────────────────────────────────────────────────


def test_extract_adv_usd_keeps_positive_drops_gaps():
    m = {
        "AAPL": {"adv_usd": 8.0e9},
        "MSFT": {"adv_usd": 6.0e9},
        "NULLC": {"adv_usd": None},          # coverage gap
        "ZERO": {"adv_usd": 0.0},            # ≤0 → gap
        "NEG": {"adv_usd": -5.0},            # <0 → gap
        "NAN": {"adv_usd": float("nan")},    # NaN → gap
        "BAD": {"adv_usd": "not-a-number"},  # unparseable → gap
        "NOTBLK": "x",                        # not a dict
    }
    out = extract_adv_usd(m)
    assert out == {"AAPL": 8.0e9, "MSFT": 6.0e9}
    assert all(math.isfinite(v) and v > 0 for v in out.values())


def test_extract_adv_usd_on_empty_and_none():
    assert extract_adv_usd({}) == {}
    assert extract_adv_usd(None) == {}
