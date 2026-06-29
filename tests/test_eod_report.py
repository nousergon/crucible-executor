"""Tests for executor/eod_report.py — the structured EOD report artifact.

The load-bearing invariant is **attribution tie-out**: the per-sleeve daily
dollar-alpha contributions (positions + rotation + cash + pricing&timing +
unattributed) must sum to the headline dollar-alpha (prior_nav × (daily_return −
spy_return)/100) to within floating-point tolerance, for arbitrary intraday
rotation. This is the property the old emailer violated.

Schema 2.0 (2026-06-29) splits the former catch-all sleeves so each is
economically meaningful: a position is benchmarked on its *retained* prior MV,
rotated-out shares get their own sleeve, the cash sleeve is genuine idle cash
only, and the IB-mark-vs-settled-close basis difference is isolated as a named
"Pricing & timing" reconciliation sleeve so "Unattributed" shrinks to the true
residual.
"""

import sqlite3

import pytest

from executor.eod_report import (
    SCHEMA_VERSION,
    build_eod_report,
    compute_alpha_attribution,
    compute_rotation_realized,
)


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


def _kinds(attr: dict) -> set[str]:
    return {c["kind"] for c in attr["components"]}


def _by_kind(attr: dict, kind: str) -> dict | None:
    return next((c for c in attr["components"] if c["kind"] == kind), None)


class TestAttributionTieOut:
    def test_ties_to_headline_simple(self):
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.33,
            positions={
                # entered today → no prior position
                "ADBE": {"shares": 10, "daily_return_usd": 1156.0},
                # held through → benchmarked on its retained prior MV
                "GOOG": {"shares": 200, "daily_return_usd": -4278.0},
            },
            prior_positions={
                "GOOG": {"shares": 200, "closing_price": 420.745},  # mv ≈ 84149
            },
            interest_usd=2.0,
            # EOD identity: unattributed = nav_change − position_pnl − interest
            # = -10271 − (1156 − 4278) − 2 = -7151
            unattributed_usd=-7151.0,
            nav_change_usd=-10271.0,
        )
        assert attr["ties_to_headline"] is True
        assert abs(attr["residual_usd"]) < 1.0
        summed = sum(c["contrib_usd"] for c in attr["components"])
        assert summed == pytest.approx(attr["dollar_alpha"], abs=1e-6)

    def test_dollar_alpha_matches_headline_formula(self):
        prior_nav, nav_change, spy = 1_001_593.0, -10271.0, -0.33
        attr = compute_alpha_attribution(
            prior_nav=prior_nav, spy_return=spy,
            positions={"X": {"shares": 100, "daily_return_usd": -10271.0}},
            prior_positions={"X": {"shares": 100, "closing_price": 9500.0}},
            interest_usd=0.0, unattributed_usd=0.0, nav_change_usd=nav_change,
        )
        daily_return = nav_change / prior_nav * 100.0
        expected_alpha_pct = daily_return - spy
        assert attr["alpha_pct"] == pytest.approx(expected_alpha_pct, abs=1e-9)

    def test_ties_with_heavy_rotation(self):
        """Two positions fully exited today + one entered; the identity must
        still close and the exited capital lands in the rotation sleeve, not
        cash."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=0.5,
            positions={"NEW": {"shares": 50, "daily_return_usd": 300.0}},
            prior_positions={
                "GEHC": {"shares": 1000, "closing_price": 78.0},  # exited (mv 78k)
                "WDAY": {"shares": 300, "closing_price": 220.0},  # exited (mv 66k)
            },
            trades_today=[
                {"action": "EXIT", "ticker": "GEHC", "shares": 1000, "price": 79.0},
                {"action": "EXIT", "ticker": "WDAY", "shares": 300, "price": 219.0},
            ],
            # realized = (79-78)*1000 + (219-220)*300 = 1000 - 300 = 700
            # EOD identity unattributed = nav_change − pos_pnl − interest
            # = 2000 − 300 − 10 = 1690 (the 700 realized is inside it)
            interest_usd=10.0,
            unattributed_usd=1690.0,
            nav_change_usd=2000.0,
        )
        assert abs(attr["residual_usd"]) < 1.0
        assert "rotation" in _kinds(attr)
        assert attr["rotation_realized_usd"] == pytest.approx(700.0, abs=1e-6)

    def test_positive_alpha_position_stays_positive_on_negative_total(self):
        """Regression for the 'α % of Total' sign-flip: a position that beat
        SPY must show a POSITIVE contribution even when the day's total alpha
        is negative."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.33,
            positions={"ADBE": {"shares": 10, "daily_return_usd": 1156.0}},
            prior_positions={},
            interest_usd=0.0,
            unattributed_usd=-5000.0,
            nav_change_usd=-3844.0,
        )
        adbe = next(c for c in attr["components"] if c["label"] == "ADBE")
        assert adbe["contrib_usd"] > 0
        assert attr["dollar_alpha"] < 0

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


class TestRotationSleeve:
    def test_rotation_realized_not_in_cash(self):
        """A trim's sold shares form a rotation sleeve benchmarked on the sold
        prior MV; the cash sleeve reflects genuine idle cash only (idle_cash
        excludes both held and rotated capital)."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=1.0,
            positions={
                "AMD": {"shares": 60, "daily_return_usd": 600.0},   # trimmed from 100
                "GOOG": {"shares": 200, "daily_return_usd": 200.0},  # held through
            },
            prior_positions={
                "AMD": {"shares": 100, "closing_price": 500.0},   # mv 50k
                "GOOG": {"shares": 200, "closing_price": 100.0},  # mv 20k
            },
            trades_today=[
                {"action": "REDUCE", "ticker": "AMD", "shares": 40, "price": 505.0},
            ],
            interest_usd=0.0,
            # realized = (505-500)*40 = 200; unattributed identity = 1000-800-0 = 200
            unattributed_usd=200.0,
            nav_change_usd=1000.0,
        )
        assert abs(attr["residual_usd"]) < 1.0
        assert attr["rotation_realized_usd"] == pytest.approx(200.0, abs=1e-6)
        # idle cash excludes held (50k+20k) and rotated (20k) capital
        assert attr["idle_cash"] == pytest.approx(930_000.0, abs=1e-6)
        cash = _by_kind(attr, "cash")
        assert cash["contrib_usd"] == pytest.approx(-9300.0, abs=1e-6)

    def test_compute_rotation_realized_tolerates_price_at_order_key(self):
        """The reconcile path passes raw trade rows keyed by price_at_order."""
        realized = compute_rotation_realized(
            positions={"AMD": {"shares": 60}},
            prior_positions={"AMD": {"shares": 100, "closing_price": 500.0}},
            trades_today=[
                {"action": "REDUCE", "ticker": "AMD", "shares": 40,
                 "price_at_order": 505.0},
            ],
        )
        assert realized == pytest.approx(200.0, abs=1e-6)


class TestPricingTiming:
    def test_pricing_timing_isolated(self):
        """The IB-mark-vs-settled-close basis term is its own sleeve and shrinks
        Unattributed to the true residual (mirrors the live 2026-06-26 case)."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.72,
            positions={"SPY": {"shares": 600, "daily_return_usd": -4000.0}},
            prior_positions={"SPY": {"shares": 600, "closing_price": 800.0}},  # mv 480k
            interest_usd=0.0,
            # nav_change = pos_pnl + interest + unattributed = -4000 + 0 + -2890
            unattributed_usd=-2890.0,
            nav_change_usd=-6890.0,
            pricing_timing_usd=-2858.0,
            pricing_timing_available=True,
        )
        assert abs(attr["residual_usd"]) < 1.0
        recon = _by_kind(attr, "reconciliation")
        assert recon is not None
        assert recon["contrib_usd"] == pytest.approx(-2858.0, abs=1e-6)
        # true residual collapses from -2890 to -32
        assert attr["unattributed_true_usd"] == pytest.approx(-32.0, abs=1e-6)
        assert abs(attr["unattributed_true_usd"]) < 100.0

    def test_pricing_timing_unavailable_fallback(self):
        """When the term can't be reconstructed it is 0, the gap stays in
        Unattributed, and the components still tie."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=-0.72,
            positions={"SPY": {"shares": 600, "daily_return_usd": -4000.0}},
            prior_positions={"SPY": {"shares": 600, "closing_price": 800.0}},
            interest_usd=0.0,
            unattributed_usd=-2890.0,
            nav_change_usd=-6890.0,
            pricing_timing_usd=0.0,
            pricing_timing_available=False,
        )
        assert abs(attr["residual_usd"]) < 1.0
        assert "reconciliation" not in _kinds(attr)
        assert attr["unattributed_true_usd"] == pytest.approx(-2890.0, abs=1e-6)

    def test_unattributed_true_small_after_split(self):
        """rotation + pricing&timing together account for nearly all of the raw
        unattributed; what's left is small."""
        attr = compute_alpha_attribution(
            prior_nav=1_000_000.0,
            spy_return=0.10,
            positions={"AMD": {"shares": 60, "daily_return_usd": 600.0}},
            prior_positions={"AMD": {"shares": 100, "closing_price": 500.0}},
            trades_today=[
                {"action": "REDUCE", "ticker": "AMD", "shares": 40, "price": 505.0},
            ],
            interest_usd=0.0,
            unattributed_usd=1200.0,   # 200 rotation + 985 pricing + 15 true
            nav_change_usd=1800.0,
            pricing_timing_usd=985.0,
            pricing_timing_available=True,
        )
        assert abs(attr["residual_usd"]) < 1.0
        assert attr["unattributed_true_usd"] == pytest.approx(15.0, abs=1e-6)


class TestBuildEodReport:
    def test_schema_version_is_2_0(self):
        assert SCHEMA_VERSION == "2.0"

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
                "pricing_timing_usd": -2200.0, "pricing_timing_available": True,
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
        assert report["schema_version"] == "2.0"
        assert report["run_date"] == "2026-06-22"
        assert report["summary"]["nav"] == 991322.0
        adbe = report["positions"][0]
        assert adbe["ticker"] == "ADBE"
        assert adbe["alpha_contrib_bps"] is not None
        assert adbe["rationale"]
        # New nav_reconciliation fields surfaced
        nr = report["nav_reconciliation"]
        assert nr["pricing_timing_usd"] == -2200.0
        assert nr["pricing_timing_available"] is True
        assert nr["rotation_realized_usd"] is not None
        assert nr["unattributed_true_usd"] is not None
        # The reconciliation sleeve appears in the attribution
        assert "reconciliation" in {
            c["kind"] for c in report["alpha_attribution"]["components"]
        }
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
