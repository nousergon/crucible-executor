"""
EOD performance email — sent after market close on trading days.

As of 2026-06-22 this is a **slim link email** (config#856 pipeline-reporting
revamp, "pull-for-state console page + push-on-transition emails"): the body
carries the at-a-glance Daily Summary (NAV, daily return vs SPY, alpha, cash/
P&L) plus any DATA WARNINGS, then links to the full computed report on the
private console. The full positions / attribution / rationale / sector /
roundtrip / trailing tables now live ONLY on the console EOD Report page
(alpha-engine-dashboard ``views/19_EOD_Report.py``), which renders the
structured ``consolidated/{date}/eod_report.json`` artifact.

Why the change: the old inline email recomputed a per-position "α % of Total"
column that divided each position's alpha by the *signed* grand-total alpha
(sign-flipping every row on negative-alpha days) and carried a positions-table
alpha total that never reconciled with the NAV-based headline. The console page
renders a single, correct, prior-NAV-basis attribution that ties to the
headline exactly. Keeping the report in one place (the artifact) removes the
two-implementations-that-disagree failure mode.

Gmail SMTP is used when GMAIL_APP_PASSWORD is set in the environment (avoids the
SPF/DKIM drop that occurs when SES sends on behalf of a @gmail.com sender).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_CONSOLE_BASE_URL = "https://console.nousergon.ai"
# Pinned slug — must match the EOD Report page's ``url_path`` in
# alpha-engine-dashboard app.py. A drift here breaks the email deep-link
# (config#856 Phase 2.5). Guarded by the dashboard's slug-drift test.
EOD_REPORT_SLUG = "eod-report"

_HTML = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body {{ font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6;
          color: #222; max-width: 700px; margin: 0 auto; padding: 20px; }}
  h2   {{ font-size: 15px; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .pos  {{ color: #006600; font-weight: bold; }}
  .neg  {{ color: #990000; font-weight: bold; }}
  .neu  {{ color: #555; }}
  .cta  {{ display: inline-block; margin: 14px 0; padding: 10px 18px;
           background: #0b5; color: #fff !important; text-decoration: none;
           border-radius: 4px; font-weight: bold; }}
  .foot {{ margin-top: 28px; font-size: 11px; color: #888;
           border-top: 1px solid #ccc; padding-top: 8px; }}
</style>
</head>
<body>
{body}
<div class="foot">Alpha Engine | {date}</div>
</body>
</html>
"""


def _pct(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    css = "pos" if v > 0 else ("neg" if v < 0 else "neu")
    return f'<span class="{css}">{sign}{v:.{decimals}f}%</span>'


def _dollar(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    css = "pos" if v > 0 else ("neg" if v < 0 else "neu")
    return f'<span class="{css}">{sign}${v:,.0f}</span>'


def _plain_pct(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def eod_report_url(run_date: str, console_base_url: str | None = None) -> str:
    """Deep-link to the console EOD Report page for ``run_date``."""
    base = (console_base_url or DEFAULT_CONSOLE_BASE_URL).rstrip("/")
    return f"{base}/{EOD_REPORT_SLUG}?date={run_date}"


def build_eod_email(
    run_date: str,
    nav: float,
    daily_return: float | None,
    spy_return: float | None,
    alpha: float | None,
    account_snapshot: dict | None = None,
    data_warnings: list[str] | None = None,
    console_base_url: str | None = None,
) -> tuple[str, str, str]:
    """Build the slim EOD email: subject + (html_body, plain_body).

    The subject is unchanged (NAV + alpha at a glance). The body is the Daily
    Summary, any DATA WARNINGS, and a deep-link to the full console report.
    """
    alpha_str = _plain_pct(alpha)
    nav_fmt = f"${nav:,.0f}"
    subject = f"Alpha Engine | {run_date} | NAV {nav_fmt} | α {alpha_str}"
    url = eod_report_url(run_date, console_base_url)

    acct = account_snapshot or {}
    ib_cash = acct.get("total_cash")
    ib_gross_pos = acct.get("gross_position_value")
    ib_unrealized = acct.get("unrealized_pnl")
    ib_realized = acct.get("realized_pnl")
    ib_accrued = acct.get("accrued_interest")

    html_parts = [
        "<h2>Daily Summary</h2>", "<table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>NAV</td><td>{nav_fmt}</td></tr>",
        f"<tr><td>Daily Return</td><td>{_pct(daily_return)}</td></tr>",
        f"<tr><td>SPY Return</td><td>{_pct(spy_return)}</td></tr>",
        f"<tr><td>Daily Alpha</td><td>{_pct(alpha)}</td></tr>",
    ]
    if ib_cash is not None:
        html_parts.append(f"<tr><td>Cash</td><td>${ib_cash:,.0f}</td></tr>")
    if ib_gross_pos is not None:
        html_parts.append(f"<tr><td>Positions (MV)</td><td>${ib_gross_pos:,.0f}</td></tr>")
    if ib_unrealized is not None:
        html_parts.append(f"<tr><td>Unrealized P&L</td><td>{_dollar(ib_unrealized)}</td></tr>")
    if ib_realized is not None and ib_realized != 0:
        html_parts.append(f"<tr><td>Realized P&L</td><td>{_dollar(ib_realized)}</td></tr>")
    if ib_accrued is not None and ib_accrued != 0:
        html_parts.append(f"<tr><td>Accrued Interest</td><td>{_dollar(ib_accrued)}</td></tr>")
    html_parts.append("</table>")

    plain_parts = [
        f"Alpha Engine EOD — {run_date}",
        "=" * 40,
        f"NAV:          {nav_fmt}",
        f"Daily Return: {_plain_pct(daily_return)}",
        f"SPY Return:   {_plain_pct(spy_return)}",
        f"Daily Alpha:  {_plain_pct(alpha)}",
    ]
    if ib_cash is not None:
        plain_parts.append(f"Cash:         ${ib_cash:,.0f}")
    if ib_gross_pos is not None:
        plain_parts.append(f"Positions:    ${ib_gross_pos:,.0f}")
    if ib_unrealized is not None:
        plain_parts.append(f"Unrealized:   ${ib_unrealized:+,.0f}")
    if ib_realized is not None and ib_realized != 0:
        plain_parts.append(f"Realized:     ${ib_realized:+,.0f}")
    plain_parts.append("")

    # DATA WARNINGS still push in the email — a red flag must reach the inbox,
    # not hide behind a click (per [[feedback_no_silent_fails]]).
    if data_warnings:
        html_parts.append('<div style="background:#fff3cd;border:1px solid #ffc107;padding:10px;margin:10px 0;">')
        html_parts.append('<strong>DATA WARNINGS:</strong><ul>')
        for w in data_warnings:
            html_parts.append(f'<li>{w}</li>')
        html_parts.append('</ul></div>')
        plain_parts.append("!! DATA WARNINGS !!")
        for w in data_warnings:
            plain_parts.append(f"  - {w}")
        plain_parts.append("")

    html_parts.append(
        f'<a class="cta" href="{url}">View full EOD report on the console →</a>'
    )
    html_parts.append(
        '<p style="font-size:11px;color:#888;">Positions, daily-alpha attribution, '
        'rationale, sector contribution, trades, roundtrips, and trailing history '
        'are on the console report.</p>'
    )
    plain_parts.append(f"Full EOD report: {url}")
    plain_parts.append("")

    html_body = _HTML.format(body="\n".join(html_parts), date=run_date)
    plain_body = "\n".join(plain_parts)
    return subject, html_body, plain_body


def send_eod_email(
    run_date: str,
    nav: float,
    daily_return: float | None,
    spy_return: float | None,
    alpha: float | None,
    sender: str,
    recipients: list[str],
    region: str = "us-east-1",
    account_snapshot: dict | None = None,
    data_warnings: list[str] | None = None,
    console_base_url: str | None = None,
) -> None:
    subject, html_body, plain_body = build_eod_email(
        run_date, nav, daily_return, spy_return, alpha,
        account_snapshot=account_snapshot,
        data_warnings=data_warnings,
        console_base_url=console_base_url,
    )

    # SMTP/SES dispatch via the alpha_engine_lib.email_sender chokepoint
    # (Gmail SMTP primary, SES fallback). The full report is archived as the
    # structured eod_report.json artifact (executor/eod_report.py), not as a
    # rendered email HTML — the console page is the canonical render.
    from nousergon_lib.email_sender import send_email as _send_email
    _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=region,
    )
