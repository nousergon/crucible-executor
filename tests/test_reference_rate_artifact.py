"""Producer contract test for the reference-rate showcase artifact.

Guards the cross-repo contract Metron consumes (metron/reference_rate.json):
required keys + shape are stable, AND — the privacy invariant — no strategy-edge /
internal field ever leaks into the published object. Mirrors the
producer-side-contract pattern (test_executor_params_consumer_contract.py).
"""

import json

import pandas as pd
import pytest

from executor import reference_rate

# Fields that MUST NEVER appear anywhere in the published artifact — strategy edge,
# assumptions, or account internals. The artifact is illustrative-only.
_FORBIDDEN_KEYS = {
    "weights",
    "scoring",
    "scores",
    "score",
    "model",
    "params",
    "predictions",
    "signal",
    "conviction",
    "buying_power",
    "settled_cash",
    "realized_pnl",
    "unrealized_pnl",
    "accrued_interest",
    "accrued_dividends",
    "daily_return_usd",
    "daily_return_pct",
    "alpha_contribution_usd",
    "alpha_contribution_pct",
}


def _sample_positions():
    # Enriched EOD positions dict shape (mirrors eod_reconcile in-memory state),
    # deliberately carrying internal attribution fields that MUST be stripped.
    return {
        "AMD": {
            "shares": 192,
            "market_value": 103175.04,
            "avg_cost": 130.5,
            "sector": "Information Technology",
            "unrealized_pnl": 18000.0,
            "daily_return_usd": 1234.5,
            "alpha_contribution_pct": 0.12,
        },
        "SPY": {
            "shares": 658,
            "market_value": 491354.92,
            "avg_cost": 600.0,
            "sector": "Broad Market",
        },
        "CLOSED": {  # zero-share row — not a holding, must be dropped
            "shares": 0,
            "market_value": 0.0,
            "avg_cost": 50.0,
            "sector": "Industrials",
        },
    }


def _walk_keys(obj):
    """Yield every dict key appearing anywhere in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


def test_build_payload_required_keys_and_shape():
    nav_history = [
        {"date": "2026-06-17", "nav": 1_000_000.0, "spy_close": 745.0},
        {"date": "2026-06-18", "nav": 1_001_593.11, "spy_close": 746.74},
    ]
    payload = reference_rate.build_payload(
        positions=_sample_positions(),
        nav=1_001_593.11,
        nav_history=nav_history,
        run_date="2026-06-18",
    )

    assert payload["schema_version"] == reference_rate.SCHEMA_VERSION
    assert payload["as_of"] == "2026-06-18"
    assert payload["base_currency"] == "USD"
    assert payload["label"] == reference_rate.LABEL
    assert payload["disclaimer"]  # non-empty illustrative-only notice
    assert payload["account"] == {"net_liquidation": 1_001_593.11}

    # Positions: zero-share row dropped, only the disclosed fields present.
    tickers = {p["ticker"] for p in payload["positions"]}
    assert tickers == {"AMD", "SPY"}
    for p in payload["positions"]:
        assert set(p) == {"ticker", "shares", "avg_cost", "market_value", "sector"}

    assert payload["nav_history"] == nav_history


def test_privacy_no_forbidden_keys_leak():
    payload = reference_rate.build_payload(
        positions=_sample_positions(),
        nav=1_001_593.11,
        nav_history=[{"date": "2026-06-18", "nav": 1_001_593.11, "spy_close": 746.74}],
        run_date="2026-06-18",
    )
    leaked = set(_walk_keys(payload)) & _FORBIDDEN_KEYS
    assert not leaked, f"strategy-internal fields leaked into the artifact: {leaked}"


def test_payload_json_serializable():
    payload = reference_rate.build_payload(
        positions=_sample_positions(),
        nav=1_001_593.11,
        nav_history=[{"date": "2026-06-18", "nav": 1_001_593.11, "spy_close": 746.74}],
        run_date="2026-06-18",
    )
    # Round-trips with the same default=str publish() uses.
    assert json.loads(json.dumps(payload, default=str))["positions"]


def test_nav_history_from_eod_df_tolerates_gaps_and_truncates():
    df = pd.DataFrame(
        [
            {"date": "2026-06-16", "portfolio_nav": 999_000.0, "spy_close": 744.0},
            {"date": "2026-06-17", "portfolio_nav": None, "spy_close": 745.0},  # dropped
            {"date": "2026-06-18", "portfolio_nav": 1_001_593.11, "spy_close": None},  # spy null OK
        ]
    )
    rows = reference_rate.nav_history_from_eod_df(df)
    assert [r["date"] for r in rows] == ["2026-06-16", "2026-06-18"]
    assert rows[-1]["spy_close"] is None
    assert rows[0]["spy_close"] == 744.0


def test_nav_history_truncates_to_max_rows():
    df = pd.DataFrame(
        [{"date": f"2024-{(i % 12) + 1:02d}-01", "portfolio_nav": float(i), "spy_close": 1.0}
         for i in range(reference_rate._NAV_HISTORY_MAX_ROWS + 50)]
    )
    history = reference_rate.nav_history_from_eod_df(df)
    payload = reference_rate.build_payload({}, 0.0, history, "2024-01-01")
    assert len(payload["nav_history"]) == reference_rate._NAV_HISTORY_MAX_ROWS


def test_nav_history_empty_df():
    assert reference_rate.nav_history_from_eod_df(None) == []
    assert reference_rate.nav_history_from_eod_df(pd.DataFrame()) == []


def test_publish_writes_expected_key():
    captured = {}

    class _FakeS3:
        def put_object(self, **kw):
            captured.update(kw)

    payload = reference_rate.build_payload(
        {}, 0.0, [{"date": "2026-06-18", "nav": 1.0, "spy_close": None}], "2026-06-18"
    )
    reference_rate.publish(_FakeS3(), "alpha-engine-research", payload)
    assert captured["Key"] == reference_rate.REFERENCE_RATE_KEY
    assert captured["Bucket"] == "alpha-engine-research"
    assert json.loads(captured["Body"])["label"] == reference_rate.LABEL
