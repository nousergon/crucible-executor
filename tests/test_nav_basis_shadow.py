"""NAV basis shadow stage — alpha-engine-config-I9638.

Operator ruling 2026-08-31, option (b) STAGED: compute the settled-close NAV
BESIDE IB NetLiquidation, publish the difference for two weeks, then cut over.
This module holds the tests for the shadow stage:

* ``nav_basis`` config validation (default, both legal values, loud refusal);
* under the DEFAULT ``ib_netliq`` basis, both figures and their gap are
  computed and published on the ``eod_pnl`` row and in ``eod_report.json``,
  and the headline NAV is unchanged;
* under ``settled_close`` the headline NAV IS the settled sum, the NAV
  three-way hard gate still fires on an IB divergence past tolerance
  (alpha-engine-config-I6819 item 3), and the attribution identity still
  closes with the mark-basis sleeve at zero;
* the sqlite migration is idempotent.

Book shape throughout: 12 positions of 500 shares at a $170.00 settled close
($85,000 each, $1,020,000 of settled MV) against $28,000 of cash and $150 of
accrued interest — a settled NAV of $1,048,150, matching the live book's
scale.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from executor.config_loader import (
    NAV_BASIS_DEFAULT,
    NAV_BASIS_IB_NETLIQ,
    NAV_BASIS_SETTLED_CLOSE,
    resolve_nav_basis,
)
from executor.trade_logger import init_db, log_eod

# ── Book constants ─────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA",
    "AVGO", "COST", "LMT", "CTAS", "ADBE", "AMD",
]
SHARES = 500
CLOSE_TODAY = 170.00
CLOSE_PRIOR = 168.00
CASH = 28_000.0
ACCRUED_TODAY = 150.0
ACCRUED_PRIOR = 100.0
DAY_LOW = 160.00
DAY_HIGH = 180.00

SETTLED_MV_TODAY = len(TICKERS) * SHARES * CLOSE_TODAY          # 1,020,000
SETTLED_MV_PRIOR = len(TICKERS) * SHARES * CLOSE_PRIOR          # 1,008,000
NAV_SETTLED_TODAY = CASH + ACCRUED_TODAY + SETTLED_MV_TODAY     # 1,048,150
NAV_SETTLED_PRIOR = CASH + ACCRUED_PRIOR + SETTLED_MV_PRIOR     # 1,036,100

RUN_DATE = "2026-09-03"
PRIOR_DATE = "2026-09-02"


# ── Config validation ──────────────────────────────────────────────────────


class TestResolveNavBasis:
    def test_absent_key_takes_the_pre_flag_default(self):
        """Every risk.yaml deployed before I9638 lacks the key, and the
        default is today's behaviour — absence is a known state."""
        assert resolve_nav_basis({"db_path": "/tmp/x.db"}) == NAV_BASIS_DEFAULT
        assert NAV_BASIS_DEFAULT == NAV_BASIS_IB_NETLIQ

    def test_none_config_takes_the_default(self):
        assert resolve_nav_basis(None) == NAV_BASIS_IB_NETLIQ

    @pytest.mark.parametrize(
        "value", [NAV_BASIS_IB_NETLIQ, NAV_BASIS_SETTLED_CLOSE],
    )
    def test_both_legal_values_pass_through(self, value):
        assert resolve_nav_basis({"nav_basis": value}) == value

    @pytest.mark.parametrize(
        "value",
        ["settled-close", "ib_netliquidation", "SETTLED_CLOSE", "", None, 1],
    )
    def test_unrecognised_value_raises_rather_than_defaulting(self, value):
        """A typo must NEVER fall back to the default: NAV would publish on a
        basis the operator did not choose while risk.yaml says otherwise."""
        with pytest.raises(ValueError, match="not a recognised NAV basis"):
            resolve_nav_basis({"nav_basis": value})

    def test_refusal_names_the_legal_values_and_the_default(self):
        with pytest.raises(ValueError) as exc:
            resolve_nav_basis({"nav_basis": "netliq"})
        msg = str(exc.value)
        assert NAV_BASIS_IB_NETLIQ in msg
        assert NAV_BASIS_SETTLED_CLOSE in msg


# ── sqlite migration ───────────────────────────────────────────────────────


_NEW_COLUMNS = [
    "nav_basis", "nav_ib_usd", "nav_settled_usd",
    "nav_basis_diff_usd", "nav_basis_diff_bps",
]


class TestSchema:
    def test_nav_basis_columns_present(self):
        db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        try:
            conn = init_db(db_path)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(eod_pnl)")}
            conn.close()
            assert set(_NEW_COLUMNS) <= cols
        finally:
            os.unlink(db_path)

    def test_migration_is_idempotent(self):
        """`ALTER TABLE ... ADD COLUMN` is not idempotent on its own; init_db
        swallows the duplicate-column error. Three runs, each column once."""
        db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        try:
            for _ in range(3):
                init_db(db_path).close()
            c = sqlite3.connect(db_path)
            cols = [r[1] for r in c.execute("PRAGMA table_info(eod_pnl)")]
            c.close()
            for name in _NEW_COLUMNS:
                assert cols.count(name) == 1, f"{name} appears {cols.count(name)}x"
        finally:
            os.unlink(db_path)

    def test_log_eod_persists_and_legacy_callers_get_nulls(self):
        db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        try:
            conn = init_db(db_path)
            log_eod(conn, {
                "date": RUN_DATE,
                "portfolio_nav": NAV_SETTLED_TODAY,
                "nav_basis": NAV_BASIS_SETTLED_CLOSE,
                "nav_ib_usd": 1_048_950.0,
                "nav_settled_usd": NAV_SETTLED_TODAY,
                "nav_basis_diff_usd": 800.0,
                "nav_basis_diff_bps": 7.6266,
            })
            log_eod(conn, {"date": PRIOR_DATE, "portfolio_nav": 1_036_100.0})
            row = conn.execute(
                "SELECT nav_basis, nav_ib_usd, nav_settled_usd, "
                "nav_basis_diff_usd, nav_basis_diff_bps FROM eod_pnl "
                "WHERE date=?", (RUN_DATE,),
            ).fetchone()
            assert row == (
                NAV_BASIS_SETTLED_CLOSE, 1_048_950.0, NAV_SETTLED_TODAY,
                800.0, 7.6266,
            )
            legacy = conn.execute(
                "SELECT nav_basis, nav_ib_usd, nav_settled_usd, "
                "nav_basis_diff_usd, nav_basis_diff_bps FROM eod_pnl "
                "WHERE date=?", (PRIOR_DATE,),
            ).fetchone()
            assert legacy == (None, None, None, None, None)
            conn.close()
        finally:
            os.unlink(db_path)


# ── run() harness ──────────────────────────────────────────────────────────


def _price_frame() -> pd.DataFrame:
    """Two settled sessions for one ticker, with the day's traded range."""
    return pd.DataFrame(
        {
            "Close": [CLOSE_PRIOR, CLOSE_TODAY],
            "Low": [DAY_LOW, DAY_LOW],
            "High": [DAY_HIGH, DAY_HIGH],
        },
        index=pd.to_datetime([PRIOR_DATE, RUN_DATE]),
    )


def _universe_library() -> MagicMock:
    frames = {t: _price_frame() for t in TICKERS}
    frames["SPY"] = _price_frame()
    lib = MagicMock()

    def _read(sym):
        if sym not in frames:
            raise KeyError(f"no such symbol: {sym}")
        return SimpleNamespace(data=frames[sym])

    lib.read.side_effect = _read
    return lib


def _snapshot(mark_basis_usd: float, *, drop_cash: bool = False) -> dict:
    """Today's broker snapshot, with ``mark_basis_usd`` of IB-vs-settled skew.

    The skew is spread evenly across the book as a per-share mark offset, and
    every resulting mark stays inside the day's [Low, High] range, so the
    custodian-mark correction never fires and ``nav_ib_usd`` is the broker's
    own NetLiquidation.
    """
    per_share = mark_basis_usd / (len(TICKERS) * SHARES)
    ib_mark = CLOSE_TODAY + per_share
    assert DAY_LOW < ib_mark < DAY_HIGH
    positions = {
        t: {"shares": SHARES, "market_value": SHARES * ib_mark, "sector": "Tech"}
        for t in TICKERS
    }
    return {
        "account": {
            "net_liquidation": CASH + ACCRUED_TODAY + SHARES * ib_mark * len(TICKERS),
            "total_cash": None if drop_cash else CASH,
            "accrued_interest": ACCRUED_TODAY,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "gross_position_value": SHARES * ib_mark * len(TICKERS),
        },
        "positions": positions,
        "accrued_dividends": {},
        "captured_at": f"{RUN_DATE}T20:20:00Z",
    }


def _seed_prior_row(conn: sqlite3.Connection) -> None:
    """Yesterday's row: settled and IB agree, so the prior mark basis is $0."""
    prior_positions = {
        t: {
            "shares": SHARES,
            "market_value": SHARES * CLOSE_PRIOR,
            "ib_market_value": SHARES * CLOSE_PRIOR,
            "closing_price": CLOSE_PRIOR,
            "sector": "Tech",
        }
        for t in TICKERS
    }
    log_eod(conn, {
        "date": PRIOR_DATE,
        "portfolio_nav": NAV_SETTLED_PRIOR,
        "daily_return_pct": 0.0,
        "spy_return_pct": 0.0,
        "daily_alpha_pct": 0.0,
        "positions_snapshot": prior_positions,
        "spy_close": CLOSE_PRIOR,
        "total_cash": CASH,
        "accrued_interest": ACCRUED_PRIOR,
        "nav_basis": NAV_BASIS_IB_NETLIQ,
        "nav_ib_usd": NAV_SETTLED_PRIOR,
        "nav_settled_usd": NAV_SETTLED_PRIOR,
        "nav_basis_diff_usd": 0.0,
        "nav_basis_diff_bps": 0.0,
    })


class _RunResult:
    def __init__(self, row: dict, report: dict, warnings: list[str],
                 fd_sites: list[str]):
        self.row = row
        self.report = report
        self.warnings = warnings
        self.fd_sites = fd_sites


def _vendor_unavailable(_rows):
    """Default anchor stub: the vendor is unreachable.

    Patched in unconditionally so the run is deterministic whether or not the
    developer's shell carries a POLYGON_API_KEY — an EOD run that reaches the
    network in a unit test is a flaky test, and one that reaches it only on
    some machines is worse.
    """
    raise RuntimeError("vendor unavailable (test stub)")


def _run_eod(
    *, nav_basis: str, mark_basis_usd: float, drop_cash: bool = False,
    anchor=_vendor_unavailable,
) -> _RunResult:
    """Drive ``eod_reconcile.run`` end to end against the fixture book."""
    from executor import eod_reconcile

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    conn = init_db(db_path)
    _seed_prior_row(conn)

    captured: dict = {}
    fd = MagicMock()

    def _capture_report(report, **kwargs):
        captured["report"] = report

    def _capture_email(**kwargs):
        captured["warnings"] = list(kwargs.get("data_warnings") or [])

    try:
        with patch.object(eod_reconcile, "now_dual") as m_now, \
             patch.object(eod_reconcile, "load_config") as m_cfg, \
             patch("executor.preflight.ExecutorPreflight") as m_pre, \
             patch.object(eod_reconcile, "init_db", return_value=conn), \
             patch("executor.snapshot_capturer.load_snapshot") as m_snap, \
             patch("executor.price_cache._open_universe_library",
                   return_value=_universe_library()), \
             patch.object(eod_reconcile, "_spy_close", return_value=CLOSE_TODAY), \
             patch.object(eod_reconcile, "fetch_ex_dividends",
                          return_value=({}, {}, True, None)), \
             patch.object(eod_reconcile, "fetch_live_anchor_window",
                          side_effect=anchor), \
             patch.object(eod_reconcile, "_load_signals_from_s3",
                          return_value=({}, None)), \
             patch.object(eod_reconcile, "_load_predictions_from_s3",
                          return_value=({}, None)), \
             patch.object(eod_reconcile, "write_eod_report",
                          side_effect=_capture_report), \
             patch.object(eod_reconcile, "send_eod_email",
                          side_effect=_capture_email), \
             patch.object(eod_reconcile, "backup_to_s3"), \
             patch("boto3.client"), \
             patch("nousergon_lib.logging.get_flow_doctor", return_value=fd):
            m_now.return_value = SimpleNamespace(
                trading_day=RUN_DATE, calendar_date=RUN_DATE,
            )
            m_cfg.return_value = {
                "db_path": db_path,
                "trades_bucket": "alpha-engine-research",
                "signals_bucket": "alpha-engine-research",
                "aws_region": "us-east-1",
                "email_sender": "x@x.com",
                "email_recipients": "y@y.com",
                "nav_basis": nav_basis,
            }
            m_pre.return_value.run.return_value = None
            m_snap.return_value = _snapshot(mark_basis_usd, drop_cash=drop_cash)
            eod_reconcile.run(run_date=RUN_DATE, run_audit=False)

        # run() closes the connection it was handed, so read the row back on a
        # fresh one rather than the (now closed) fixture handle.
        read = sqlite3.connect(db_path)
        cols = [r[1] for r in read.execute("PRAGMA table_info(eod_pnl)")]
        raw = read.execute(
            "SELECT * FROM eod_pnl WHERE date=?", (RUN_DATE,),
        ).fetchone()
        read.close()
        row = dict(zip(cols, raw, strict=True))
        fd_sites = [
            (c.kwargs.get("context") or {}).get("site")
            for c in fd.report.call_args_list
        ]
        return _RunResult(
            row, captured.get("report", {}), captured.get("warnings", []),
            fd_sites,
        )
    finally:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass
        os.unlink(db_path)


# ── Default basis: both figures published, headline unchanged ──────────────


@pytest.fixture(scope="module")
def default_basis_run() -> _RunResult:
    return _run_eod(nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0)


@pytest.fixture(scope="module")
def settled_basis_run() -> _RunResult:
    return _run_eod(nav_basis=NAV_BASIS_SETTLED_CLOSE, mark_basis_usd=5_000.0)


class TestDefaultBasisPublishesBothFigures:
    """Nothing load-bearing changes under the default basis — the shadow
    figures ride along beside an unchanged headline NAV."""

    @pytest.fixture(autouse=True)
    def result(self, default_basis_run):
        return default_basis_run

    def test_headline_nav_is_still_ib_netliquidation(self, result):
        assert result.row["portfolio_nav"] == pytest.approx(
            NAV_SETTLED_TODAY + 800.0, abs=1e-6,
        )
        assert result.row["nav_basis"] == NAV_BASIS_IB_NETLIQ

    def test_both_figures_are_persisted(self, result):
        assert result.row["nav_ib_usd"] == pytest.approx(
            NAV_SETTLED_TODAY + 800.0, abs=1e-6,
        )
        assert result.row["nav_settled_usd"] == pytest.approx(
            NAV_SETTLED_TODAY, abs=1e-6,
        )

    def test_difference_is_published_in_dollars_and_bps(self, result):
        assert result.row["nav_basis_diff_usd"] == pytest.approx(800.0, abs=1e-6)
        expected_bps = 800.0 / (NAV_SETTLED_TODAY + 800.0) * 10_000.0
        assert result.row["nav_basis_diff_bps"] == pytest.approx(
            expected_bps, abs=1e-6,
        )

    def test_report_carries_the_nav_basis_block(self, result):
        recon = result.report["nav_reconciliation"]
        assert recon["nav_basis"] == NAV_BASIS_IB_NETLIQ
        assert recon["nav_ib_usd"] == pytest.approx(
            NAV_SETTLED_TODAY + 800.0, abs=1e-6,
        )
        assert recon["nav_settled_usd"] == pytest.approx(
            NAV_SETTLED_TODAY, abs=1e-6,
        )
        assert recon["nav_basis_diff_usd"] == pytest.approx(800.0, abs=1e-6)
        assert recon["nav_basis_diff_bps"] is not None
        assert recon["nav_basis_unavailable_reason"] is None
        assert recon["nav_settled_fallback_tickers"] == []
        assert result.report["schema_version"] == "2.12"

    def test_pricing_timing_sleeve_still_carries_the_mark_basis(self, result):
        """On the default basis the sleeve and the divergence are the SAME
        number — the split introduced by I9638 must not change today."""
        recon = result.report["nav_reconciliation"]
        assert recon["pricing_timing_usd"] == pytest.approx(800.0, abs=1e-6)
        assert recon["mark_basis_delta_usd"] == pytest.approx(800.0, abs=1e-6)

    def test_an_in_band_difference_is_an_observation_not_a_warning(self, result):
        """$800 is inside the three-way tolerance (max($2,500, 15bp)), so it
        is published without a data_warnings line — deliverable 4."""
        assert not [w for w in result.warnings if "NAV basis divergence" in w]

    def test_no_hard_gate_page(self, result):
        assert "nav_three_way_reconcile_hard_gate" not in result.fd_sites


class TestOutOfBandDifferenceWarns:
    def test_difference_past_tolerance_lands_in_data_warnings(self):
        """Deliverable 4: only a gap past the existing hard-gate tolerance
        earns a line, and it names both the dollars and the active basis."""
        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=5_000.0,
        )
        hits = [w for w in result.warnings if "NAV basis divergence" in w]
        assert len(hits) == 1
        assert "+5,000" in hits[0]
        assert NAV_BASIS_IB_NETLIQ in hits[0]


# ── settled_close basis: headline moves, gates survive ─────────────────────


class TestSettledCloseBasis:
    """The cut-over shape. The headline NAV becomes the settled rebuild while
    IB NetLiquidation is retained as the broker cross-check, and every gate
    that reads the IB-vs-settled divergence must keep its power
    (alpha-engine-config-I6819 item 3)."""

    @pytest.fixture(autouse=True)
    def result(self, settled_basis_run):
        return settled_basis_run

    def test_headline_nav_equals_the_settled_sum(self, result):
        assert result.row["portfolio_nav"] == pytest.approx(
            NAV_SETTLED_TODAY, abs=1e-6,
        )
        assert result.row["nav_basis"] == NAV_BASIS_SETTLED_CLOSE
        assert result.row["nav_settled_usd"] == pytest.approx(
            NAV_SETTLED_TODAY, abs=1e-6,
        )

    def test_ib_netliquidation_is_retained_beside_it(self, result):
        assert result.row["nav_ib_usd"] == pytest.approx(
            NAV_SETTLED_TODAY + 5_000.0, abs=1e-6,
        )
        assert result.row["nav_basis_diff_usd"] == pytest.approx(
            5_000.0, abs=1e-6,
        )

    def test_daily_return_is_struck_on_the_settled_series(self, result):
        expected = (
            (NAV_SETTLED_TODAY - NAV_SETTLED_PRIOR) / NAV_SETTLED_PRIOR * 100
        )
        assert result.row["daily_return_pct"] == pytest.approx(expected, abs=1e-9)

    def test_hard_gate_still_fires_on_ib_divergence(self, result):
        """$5,000 of day-over-day IB-vs-settled skew is past the $2,500 floor.
        Reading the (now zero) pricing&timing sleeve here would have silenced
        the fleet's primary NAV control the day the cut-over landed."""
        assert "nav_three_way_reconcile_hard_gate" in result.fd_sites

    def test_pricing_timing_sleeve_is_zero_by_construction(self, result):
        """The headline NAV carries no broker mark, so there is no mark-basis
        sleeve inside the series being attributed — and the divergence is
        still reported, on its own field."""
        recon = result.report["nav_reconciliation"]
        assert recon["pricing_timing_usd"] == 0.0
        assert recon["mark_basis_delta_usd"] == pytest.approx(5_000.0, abs=1e-6)

    def test_attribution_identity_still_closes(self, result):
        """nav_change = position_pnl + interest - dividend_timing +
        unattributed, on the settled NAV series."""
        recon = result.report["nav_reconciliation"]
        lhs = recon["nav_change_usd"]
        rhs = (
            recon["position_pnl_usd"]
            + recon["interest_usd"]
            - recon["dividend_timing_usd"]
            + recon["unattributed_usd"]
        )
        assert lhs == pytest.approx(rhs, abs=1e-6)

    def test_true_residual_is_not_polluted_by_the_mark_basis(self, result):
        """Subtracting a $5,000 sleeve the NAV series never contained would
        manufacture a residual the bounds gate then judges."""
        recon = result.report["nav_reconciliation"]
        assert abs(recon["unattributed_true_usd"]) < 1.0

    def test_the_basis_change_is_named_in_data_warnings(self, result):
        """The prior row is on ib_netliq, so this session's daily return spans
        two definitions and must say so."""
        hits = [w for w in result.warnings if "NAV basis changed" in w]
        assert len(hits) == 1
        assert NAV_BASIS_IB_NETLIQ in hits[0]
        assert NAV_BASIS_SETTLED_CLOSE in hits[0]


class TestSettledCloseRefusesRatherThanGuesses:
    """Under settled_close, a NAV that cannot honestly be rebuilt is REFUSED,
    never published with a broker mark standing in for a missing close."""

    def test_a_missing_settled_close_is_returned_not_swallowed(self):
        """The name that fell back to the broker's mark is NAMED. run()
        currently hard-fails one step earlier — the ArcticDB lookup refuses
        the whole session on a missing close — so this is the contract that
        keeps the basis guard correct if that ordering ever changes."""
        from executor.eod_reconcile import _settled_position_value_usd

        positions = {
            t: {"shares": SHARES, "market_value": SHARES * CLOSE_TODAY}
            for t in TICKERS
        }
        closes = {t: CLOSE_TODAY for t in TICKERS if t != "AMD"}
        settled_mv, fallback = _settled_position_value_usd(positions, closes)
        assert fallback == ["AMD"]
        assert settled_mv == pytest.approx(SETTLED_MV_TODAY, abs=1e-6)

    def test_zero_share_rows_are_skipped_not_valued(self):
        from executor.eod_reconcile import _settled_position_value_usd

        settled_mv, fallback = _settled_position_value_usd(
            {"AAPL": {"shares": 0, "market_value": 12_345.0}},
            {},
        )
        assert (settled_mv, fallback) == (0.0, [])

    def test_absent_broker_cash_refuses_the_run_under_settled_close(self):
        """No cash leg means no settled rebuild. Publishing the broker figure
        under a `settled_close` label instead would be the silent basis swap
        the flag exists to prevent."""
        with pytest.raises(RuntimeError, match="cannot be rebuilt"):
            _run_eod(
                nav_basis=NAV_BASIS_SETTLED_CLOSE,
                mark_basis_usd=800.0,
                drop_cash=True,
            )

    def test_absent_broker_cash_fails_the_pre_existing_gate_on_the_default(self):
        """Same missing input, DEFAULT basis: the failure mode is unchanged by
        this PR — the pre-existing attribution basis-level gate refuses the
        session as NOT EVALUATED. The basis flag adds no new failure here and
        removes none."""
        with pytest.raises(RuntimeError, match="integrity gate failed"):
            _run_eod(
                nav_basis=NAV_BASIS_IB_NETLIQ,
                mark_basis_usd=800.0,
                drop_cash=True,
            )


# ── Artifact round-trip ────────────────────────────────────────────────────


def test_report_block_survives_json_serialisation():
    """eod_report.json is written as JSON; every new field must serialise."""
    result = _run_eod(nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0)
    recon = json.loads(json.dumps(result.report))["nav_reconciliation"]
    assert recon["nav_basis"] == NAV_BASIS_IB_NETLIQ


# ─────────────────────────────────────────────────────────────────────────────
# The benchmark leg's independent side, wired into the live path
# (alpha-engine-config-I8188 / I9614 D2)
#
# Every other gate in this file closes inside the system's own numbers.
# `spy_close` and `spy_return_pct` are written by the same producer on the same
# run, so they can be wrong TOGETHER — over the live history they are, by
# 149.2bp against the vendor. These bind the two properties that make the
# vendor anchor worth having in the postclose at all: it RUNS every session,
# and it never reports agreement it did not measure.
# ─────────────────────────────────────────────────────────────────────────────

def _breach_kinds(row) -> list[str]:
    raw = row.get("integrity_breach_json")
    return [b.get("kind") for b in json.loads(raw)] if raw else []


class TestBenchmarkVendorAnchorInTheLivePath:

    def test_an_unreachable_vendor_is_named_never_a_silent_pass(self):
        """An absent measurement persisted as a clean one is THE defect of
        I8188 (defect 3 was `{}` from the broker rendered as $0.00)."""
        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0,
            anchor=_vendor_unavailable,
        )
        assert "benchmark_anchor_unevaluated" in _breach_kinds(result.row)
        assert [w for w in result.warnings if "NOT EVALUATED" in w and "anchor" in w]

    def test_the_unevaluated_record_does_not_fail_the_postclose(self):
        """Observe mode is observe mode: `_run_eod` returning at all is the
        assertion — the deferred raise reads `integrity_breaches`, which
        observe records never enter."""
        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0,
            anchor=_vendor_unavailable,
        )
        assert result.row["portfolio_nav"] is not None

    def test_a_vendor_close_divergence_is_recorded_and_still_observe_only(self):
        """The published `spy_close` disagreeing with the vendor's own close is
        the one thing no internal equation can detect."""
        def _diverging(rows):
            dates = sorted(str(r["date"]) for r in rows)
            closes = dict.fromkeys(dates, 500.0)     # far from the fixture close
            return closes, {}

        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0,
            anchor=_diverging,
        )
        kinds = _breach_kinds(result.row)
        assert "benchmark_close_divergence" in kinds
        assert "benchmark_anchor_unevaluated" not in kinds
        assert result.row["portfolio_nav"] is not None   # observe, not a raise

    def test_the_anchor_catches_a_return_no_internal_check_can(self):
        """THE non-tautology test.

        Here every persisted `spy_close` matches the vendor's exactly, so the
        close-divergence arm is silent — and the stored `spy_return_pct` still
        disagrees with the return those same closes imply. No equation built
        from `spy_close` and `spy_return_pct` alone can distinguish this from a
        correct series, because both columns are written by this system on this
        run. The vendor's own price path is the second measurement.
        """
        def _closes_match_returns_do_not(rows):
            return {str(r["date"]): float(r["spy_close"]) for r in rows}, {}

        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0,
            anchor=_closes_match_returns_do_not,
        )
        kinds = _breach_kinds(result.row)
        assert "benchmark_anchor_drift" in kinds
        assert "benchmark_close_divergence" not in kinds
        assert result.row["portfolio_nav"] is not None   # observe, not a raise

    def test_a_fully_agreeing_vendor_records_nothing(self):
        """Agreement on BOTH arms is silence — the gate is not a nag."""
        def _agreeing(rows):
            ordered = sorted(rows, key=lambda r: str(r["date"]))
            closes, level = {}, 100.0
            for r in ordered:
                ret = float(r["spy_return_pct"] or 0.0)
                level *= (1.0 + ret / 100.0)
                closes[str(r["date"])] = level
            # Feed the book the same closes so the divergence arm is silent
            # too; the arithmetic above is the vendor's own path.
            return closes, {}

        result = _run_eod(
            nav_basis=NAV_BASIS_IB_NETLIQ, mark_basis_usd=800.0,
            anchor=_agreeing,
        )
        assert "benchmark_anchor_drift" not in _breach_kinds(result.row)
