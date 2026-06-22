"""Tests for executor/eod_emailer.py — the slim EOD link email.

The EOD email is a headline + DATA WARNINGS + console deep-link (config#856).
The full positions / attribution / rationale / sector / roundtrip / trailing
report lives on the console EOD Report page (rendered from the
consolidated/{date}/eod_report.json artifact). These tests cover the slim
email surface only; attribution correctness is in test_eod_report.py.
"""

from executor.eod_emailer import (
    EOD_REPORT_SLUG,
    _dollar,
    _pct,
    _plain_pct,
    build_eod_email,
    eod_report_url,
)


class TestFormatters:
    def test_pct_positive(self):
        result = _pct(1.5)
        assert "+1.50%" in result
        assert "pos" in result

    def test_pct_negative(self):
        result = _pct(-2.3)
        assert "-2.30%" in result
        assert "neg" in result

    def test_pct_none(self):
        assert _pct(None) == "—"

    def test_dollar_negative(self):
        result = _dollar(-800.0)
        assert "800" in result
        assert "neg" in result

    def test_plain_pct_none(self):
        assert _plain_pct(None) == "—"


class TestEodReportUrl:
    def test_default_base(self):
        url = eod_report_url("2026-06-22")
        assert url == f"https://console.nousergon.ai/{EOD_REPORT_SLUG}?date=2026-06-22"

    def test_custom_base_strips_trailing_slash(self):
        url = eod_report_url("2026-06-22", "https://console.example.com/")
        assert url == f"https://console.example.com/{EOD_REPORT_SLUG}?date=2026-06-22"


class TestBuildEodEmail:
    def test_subject_carries_nav_and_alpha(self):
        subject, html, plain = build_eod_email(
            run_date="2026-04-08",
            nav=100500.0,
            daily_return=0.5,
            spy_return=0.2,
            alpha=0.3,
        )
        assert "2026-04-08" in subject
        assert "100,500" in subject
        assert "+0.30%" in subject

    def test_body_links_to_console_report(self):
        _, html, plain = build_eod_email(
            run_date="2026-04-08",
            nav=100500.0,
            daily_return=0.5,
            spy_return=0.2,
            alpha=0.3,
        )
        expected = eod_report_url("2026-04-08")
        assert expected in html
        assert expected in plain

    def test_summary_numbers_present(self):
        _, html, plain = build_eod_email(
            run_date="2026-04-08",
            nav=100500.0,
            daily_return=0.5,
            spy_return=0.2,
            alpha=0.3,
            account_snapshot={
                "total_cash": 35587.0,
                "gross_position_value": 64913.0,
                "unrealized_pnl": 289.0,
                "realized_pnl": -18799.0,
            },
        )
        assert "35,587" in html
        assert "Daily Alpha" in html

    def test_data_warnings_still_push_in_email(self):
        """A red flag must reach the inbox, not hide behind the click."""
        _, html, plain = build_eod_email(
            run_date="2026-04-08",
            nav=100500.0,
            daily_return=0.5,
            spy_return=0.2,
            alpha=0.3,
            data_warnings=["NAV reconciliation gap: $-2,404 unattributed"],
        )
        assert "reconciliation gap" in html
        assert "reconciliation gap" in plain

    def test_no_inline_positions_table(self):
        """The slim email must NOT inline the per-position table or the
        sign-flipping 'α % of Total' column — that's the bug we removed."""
        _, html, _ = build_eod_email(
            run_date="2026-04-08",
            nav=100500.0,
            daily_return=0.5,
            spy_return=0.2,
            alpha=0.3,
        )
        assert "% of Total" not in html
        assert "Open Positions" not in html

    def test_none_returns(self):
        subject, _, _ = build_eod_email(
            run_date="2026-04-08",
            nav=100000.0,
            daily_return=None,
            spy_return=None,
            alpha=None,
        )
        assert "None" not in subject
