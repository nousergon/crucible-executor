"""Tests for executor/eod_report.py — the structured EOD report artifact.

The load-bearing invariant is **attribution tie-out**: the per-sleeve daily
dollar-alpha contributions (positions + cash & rotation + unattributed) must
sum to the headline dollar-alpha (prior_nav × (daily_return − spy_return)/100)
to within floating-point tolerance, for arbitrary intraday rotation. This is
the property the old emailer violated — its positions-table alpha total never
reconciled with the NAV-based headline, and its "α % of Total" column
sign-flipped on negative-alpha days.
"""

import sqlite3

import pytest

from executor.eod_report import build_eod_report, compute_alpha_attribution


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE eod_pnl (date TEXT, portfolio_nav REAL, "
        "daily_return_pct REAL, spy_return_pct REAL, daily_alpha_pct REAL)"
    )
    conn.execute(
        "CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT, shares INTEGER, "
        "price_at_order REAL, created_at TEXT)"
    )
    conn.executemany(
        "INSERT INTO eod_pnl VALUES (?,?,?,?,?)",
        [
            ("2026-06-18", 1001593, 0.10, 0.78, -0.68),
            ("2026-06-22", 991322, -1.03, -0.33, -0.70),
        ],
    )
    conn.commit()
    return conn


def _attr_residual(**kwargs) -> float:
    attr = compute_alpha_attribution(**kwargs)
    return attr["residual_usd"]


class TestAttributionTieOut:
    def test_ties_to_headline_simple(self):
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.33,
            positions={
                "ADBE": {"daily_return_usd": 1156.0},   # entered today → prior_mv 0
                "GOOG": {"daily_return_usd": -4278.0},
            },
            prior_positions={"GOOG": {"market_value": 84149.0}},  # held yesterday
            interest_usd=2.0,
            # EOD identity: unattributed = nav_change − position_pnl − interest
            # = -10271 − (1156 − 4278) − 2 = -7151
            unattributed_usd=-7151.0,
            nav_change_usd=-10271.0,
        )
        # Sum of components == dollar_alpha
        assert attr["ties_to_headline"] is True
        assert abs(attr["residual_usd"]) < 1.0
        summed = sum(c["contrib_usd"] for c in attr["components"])
        assert summed == pytest.approx(attr["dollar_alpha"], abs=1e-6)

    def test_dollar_alpha_matches_headline_formula(self):
        prior_nav, nav_change, spy = 1_001_593.0, -10271.0, -0.33
        attr = compute_alpha_attribution(
            prior_nav=prior_nav, spy_return=spy,
            positions={"X": {"daily_return_usd": -10271.0}},
            prior_positions={"X": {"market_value": 950000.0}},
            interest_usd=0.0, unattributed_usd=0.0, nav_change_usd=nav_change,
        )
        daily_return = nav_change / prior_nav * 100.0
        expected_alpha_pct = daily_return - spy
        assert attr["alpha_pct"] == pytest.approx(expected_alpha_pct, abs=1e-9)

    def test_ties_with_heavy_rotation(self):
        """Capital from positions exited today lands in cash & rotation; the
        identity must still close (unattributed absorbs the rest)."""
        residual = _attr_residual(
            prior_nav=1_000_000.0,
            spy_return=0.5,
            positions={"NEW": {"daily_return_usd": 300.0}},  # entered today
            prior_positions={
                "GEHC": {"market_value": 78000.0},  # exited today
                "WDAY": {"market_value": 66000.0},  # exited today
            },
            interest_usd=10.0,
            # EOD identity: unattributed = 2000 − 300 − 10 = 1690. The exited
            # GEHC/WDAY prior MV is NOT in today's positions, so it stays in
            # the cash & rotation residual (prior_cash_residual = prior_nav).
            unattributed_usd=1690.0,
            nav_change_usd=2000.0,
        )
        assert abs(residual) < 1.0

    def test_positive_alpha_position_stays_positive_on_negative_total(self):
        """Regression for the 'α % of Total' sign-flip: a position that beat
        SPY must show a POSITIVE contribution even when the day's total alpha
        is negative."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.33,  # SPY down
            positions={"ADBE": {"daily_return_usd": 1156.0}},  # up, entered today
            prior_positions={},
            interest_usd=0.0,
            unattributed_usd=-5000.0,  # drives the TOTAL negative
            nav_change_usd=-3844.0,
        )
        adbe = next(c for c in attr["components"] if c["label"] == "ADBE")
        assert adbe["contrib_usd"] > 0  # NOT sign-flipped by a negative total
        assert attr["dollar_alpha"] < 0  # total is still negative

    def test_none_when_first_trading_day(self):
        assert compute_alpha_attribution(
            prior_nav=None, spy_return=-0.33, positions={},
            prior_positions=None, interest_usd=0.0,
            unattributed_usd=0.0, nav_change_usd=None,
        ) is None

    def test_none_when_no_spy_reference(self):
        assert compute_alpha_attribution(
            prior_nav=1_000_000.0, spy_return=None, positions={},
            prior_positions=None, interest_usd=0.0,
            unattributed_usd=0.0, nav_change_usd=0.0,
        ) is None


class TestBuildEodReport:
    def test_payload_shape(self):
        conn = _conn()
        conn.execute(
            "INSERT INTO trades VALUES "
            "('2026-06-22','ADBE','ENTER',360,191.99,'2026-06-22T13:00:00')"
        )
        conn.commit()
        report = build_eod_report(
            run_date="2026-06-22",
            nav=991322.0,
            prior_nav=1001593.0,
            daily_return=-1.03,
            spy_return=-0.33,
            alpha=-0.70,
            positions={
                "ADBE": {
                    "shares": 360, "market_value": 70164.0,
                    "daily_return_pct": 1.68, "daily_return_usd": 1156.0,
                    "sector": "Information Technology",
                },
            },
            prior_positions={},
            conn=conn,
            account_snapshot={
                "total_cash": 35587.0, "gross_position_value": 955656.0,
                "unrealized_pnl": 289.0, "realized_pnl": -18799.0,
                "accrued_interest": 80.0,
            },
            nav_reconciliation={
                "nav_change_usd": -10271.0, "position_pnl_usd": -1156.0,
                "interest_usd": 2.0, "dividend_usd": 0.0,
                "unattributed_usd": -2404.0,
            },
            position_narratives={"ADBE": "Entered 2026-06-22 at $191.99."},
            sector_attribution={
                "Information Technology": {
                    "weight": 0.071, "contribution": 0.12, "positions": 1,
                },
            },
            roundtrip_stats={"n_roundtrips": 211, "avg_return_pct": 3.82},
            data_warnings=["NAV reconciliation gap: $-2,404 unattributed"],
            generated_at="2026-06-22T20:10:00Z",
        )
        assert report["schema_version"]
        assert report["run_date"] == "2026-06-22"
        assert report["summary"]["nav"] == 991322.0
        # Per-position alpha contribution is wired from the attribution
        adbe = report["positions"][0]
        assert adbe["ticker"] == "ADBE"
        assert adbe["alpha_contrib_bps"] is not None
        assert adbe["rationale"]
        # Trades + trailing history pulled from conn
        assert report["trades_today"][0]["ticker"] == "ADBE"
        assert len(report["trailing_history"]) == 2
        assert report["data_warnings"]

    def test_first_day_no_attribution(self):
        conn = _conn()
        report = build_eod_report(
            run_date="2026-06-22", nav=991322.0, prior_nav=None,
            daily_return=None, spy_return=None, alpha=None,
            positions={}, prior_positions=None, conn=conn,
        )
        assert report["alpha_attribution"] is None
