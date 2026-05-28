"""
EOD performance email — sent after market close on trading days.

Builds a summary of daily NAV, alpha vs SPY, open positions, trades placed,
and a trailing 10-day history, then sends via Gmail SMTP (primary) or AWS SES
(fallback).

Gmail SMTP is used when GMAIL_APP_PASSWORD is set in the environment. This
avoids the SPF/DKIM failure that occurs when SES sends on behalf of a
@gmail.com sender address (SES is not authorized by Gmail's SPF policy, so
emails are silently dropped).

Setup: set GMAIL_APP_PASSWORD in the EC2 environment:
    echo 'export GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxxx' >> ~/.bashrc && source ~/.bashrc
"""

from __future__ import annotations

import logging
import os
import sqlite3

import boto3


logger = logging.getLogger(__name__)

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


def build_eod_email(
    run_date: str,
    nav: float,
    daily_return: float | None,
    spy_return: float | None,
    alpha: float | None,
    positions: dict,
    conn: sqlite3.Connection,
    position_narratives: dict[str, str] | None = None,
    sector_attribution: dict | None = None,
    data_warnings: list[str] | None = None,
    roundtrip_stats: dict | None = None,
    account_snapshot: dict | None = None,
    nav_reconciliation: dict | None = None,
) -> tuple[str, str, str]:
    """
    Build the EOD email subject + (html_body, plain_body).

    Returns: (subject, html_body, plain_body)
    """
    alpha_str = _plain_pct(alpha)
    nav_fmt = f"${nav:,.0f}"
    subject = f"Alpha Engine | {run_date} | NAV {nav_fmt} | α {alpha_str}"

    # ── Daily summary ────────────────────────────────────────────────────────
    # Extract IB ground truth values
    acct = account_snapshot or {}
    ib_cash = acct.get("total_cash")
    ib_accrued = acct.get("accrued_interest")
    ib_gross_pos = acct.get("gross_position_value")
    ib_unrealized = acct.get("unrealized_pnl")
    ib_realized = acct.get("realized_pnl")

    html_parts = ["<h2>Daily Summary</h2>", "<table>",
                  "<tr><th>Metric</th><th>Value</th></tr>",
                  f"<tr><td>NAV</td><td>{nav_fmt}</td></tr>",
                  f"<tr><td>Daily Return</td><td>{_pct(daily_return)}</td></tr>",
                  f"<tr><td>SPY Return</td><td>{_pct(spy_return)}</td></tr>",
                  f"<tr><td>Daily Alpha</td><td>{_pct(alpha)}</td></tr>"]

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
    plain_parts.append("")

    # ── Data warnings banner ──────────────────────────────────────────────
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

    # ── Positions ────────────────────────────────────────────────────────────
    html_parts.append("<h2>Open Positions</h2>")
    plain_parts.append("OPEN POSITIONS")
    plain_parts.append("-" * 40)

    if positions:
        total_mv = sum(pos.get("market_value", 0) for pos in positions.values())
        total_day_usd = sum(pos.get("daily_return_usd", 0) for pos in positions.values())
        total_alpha_usd = sum(pos.get("alpha_contribution_usd", 0) for pos in positions.values())

        # NAV change components (computed by eod_reconcile). Cash earns interest
        # only — we do NOT stuff the NAV-vs-position residual into cash, since
        # that absorbs bookkeeping noise and inflates apparent alpha.
        recon = nav_reconciliation or {}
        cash = ib_cash if ib_cash is not None else (nav - total_mv if nav else 0)
        total_nav_change = nav * (daily_return / 100) if daily_return is not None else 0
        interest_usd = recon.get("interest_usd", 0.0) or 0.0
        dividend_usd = recon.get("dividend_usd", 0.0) or 0.0
        # Unattributed = NAV change - position P&L - interest - dividends.
        # Nonzero values usually indicate pricing/snapshot mismatches, corporate
        # actions, fees, or FX — surfaced separately so they stay visible.
        unattributed_usd = recon.get(
            "unattributed_usd",
            total_nav_change - total_day_usd - interest_usd - dividend_usd,
        )

        # Cash α = interest earned, minus the opportunity cost of not holding
        # SPY on that cash sleeve. Dividends are attributed to source positions,
        # so they flow through position α. Unattributed noise is excluded.
        cash_alpha_usd = interest_usd - (cash * (spy_return or 0) / 100) if cash else 0

        grand_day_usd = total_day_usd + interest_usd + dividend_usd + unattributed_usd
        grand_alpha_usd = total_alpha_usd + cash_alpha_usd
        grand_alpha_pct = grand_alpha_usd / nav * 100 if nav else 0

        html_parts += ["<table>",
                       "<tr><th>Ticker</th><th>Shares</th><th>Mkt Value</th>"
                       "<th>% NAV</th><th>Day Ret %</th><th>Day Ret $</th>"
                       "<th>α $</th><th>α % of Total</th></tr>"]
        for ticker, pos in sorted(positions.items()):
            mv = pos.get("market_value", 0)
            pct_nav = mv / nav * 100 if nav else 0
            daily_ret = pos.get("daily_return_pct")
            daily_usd = pos.get("daily_return_usd", 0)
            alpha_usd = pos.get("alpha_contribution_usd", 0)
            alpha_pct_of_total = (alpha_usd / grand_alpha_usd * 100) if grand_alpha_usd else 0
            html_parts.append(
                f"<tr><td>{ticker}</td><td>{pos['shares']:,}</td>"
                f"<td>${mv:,.0f}</td><td>{pct_nav:.1f}%</td>"
                f"<td>{_pct(daily_ret)}</td><td>{_dollar(daily_usd)}</td>"
                f"<td>{_dollar(alpha_usd)}</td><td>{_pct(alpha_pct_of_total)}</td></tr>"
            )
            dr_str = _plain_pct(daily_ret) if daily_ret is not None else "—"
            plain_parts.append(
                f"  {ticker:<6} {pos['shares']:>5}  ${mv:>9,.0f}  {pct_nav:>5.1f}%"
                f"  {dr_str:>7}  ${daily_usd:>+8,.0f}  ${alpha_usd:>+8,.0f}  {alpha_pct_of_total:>+6.1f}%"
            )

        # Cash row — balance + α from interest vs SPY, no fabricated daily return
        if abs(cash) > 1:
            cash_pct = cash / nav * 100 if nav else 0
            cash_alpha_pct_of_total = (cash_alpha_usd / grand_alpha_usd * 100) if grand_alpha_usd else 0
            html_parts.append(
                f'<tr style="color:#888"><td><i>Cash</i></td><td></td>'
                f'<td>${cash:,.0f}</td><td>{cash_pct:.1f}%</td>'
                f'<td>—</td><td>—</td>'
                f'<td>{_dollar(cash_alpha_usd)}</td><td>{_pct(cash_alpha_pct_of_total)}</td></tr>'
            )
            plain_parts.append(
                f"  {'Cash':<6} {'':>5}  ${cash:>9,.0f}  {cash_pct:>5.1f}%"
                f"  {'—':>7}  {'—':>9}  ${cash_alpha_usd:>+8,.0f}  {cash_alpha_pct_of_total:>+6.1f}%"
            )

        # Interest / Dividends / Unattributed rows — only show if non-trivial
        def _extra_row(label: str, usd: float, style: str = "color:#888;font-style:italic"):
            if abs(usd) < 1:
                return
            html_parts.append(
                f'<tr style="{style}"><td>{label}</td><td></td><td></td><td></td>'
                f'<td>—</td><td>{_dollar(usd)}</td><td>—</td><td>—</td></tr>'
            )
            plain_parts.append(f"  {label:<14} ${usd:>+8,.0f}")

        _extra_row("Interest", interest_usd)
        _extra_row("Dividends", dividend_usd)
        _extra_row(
            "Unattributed",
            unattributed_usd,
            style="color:#c00;font-style:italic",  # red — this should be small
        )

        # Totals row — ties to Daily Summary (NAV-based)
        html_parts.append(
            f'<tr style="font-weight:bold;border-top:2px solid #333">'
            f'<td>Total</td><td></td>'
            f'<td>${nav:,.0f}</td><td>100%</td>'
            f'<td></td><td>{_dollar(grand_day_usd)}</td>'
            f'<td>{_dollar(grand_alpha_usd)}</td>'
            f'<td><b>{_pct(grand_alpha_pct)}</b></td></tr>'
        )
        html_parts.append("</table>")
        plain_parts.append(f"  {'':->80}")
        plain_parts.append(
            f"  {'Total':<6} {'':>5}  ${nav:>9,.0f}  100.0%"
            f"  {'':>7}  ${grand_day_usd:>+8,.0f}  ${grand_alpha_usd:>+8,.0f}  {grand_alpha_pct:>+6.2f}%"
        )
    else:
        html_parts.append("<p>No open positions.</p>")
        plain_parts.append("  No open positions.")
    plain_parts.append("")

    # ── Position rationale ─────────────────────────────────────────────────
    if position_narratives and positions:
        html_parts.append("<h2>Position Rationale</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Ticker</th><th>Rationale</th></tr>")
        plain_parts.append("POSITION RATIONALE")
        plain_parts.append("-" * 40)
        for ticker in sorted(positions.keys()):
            narrative = position_narratives.get(ticker, "No rationale available.")
            html_parts.append(
                f"<tr><td><b>{ticker}</b></td><td>{narrative}</td></tr>"
            )
            plain_parts.append(f"  {ticker}: {narrative}")
        html_parts.append("</table>")
        html_parts.append(
            '<p style="font-size:11px; color:#888; margin-top:4px;">'
            'ⓘ α in <b>GBM:</b> rows is log-domain decimal at the 21d horizon '
            '(post 2026-05-09 canonical-alpha cutover). Top-of-email Daily Alpha '
            'is unchanged: arithmetic portfolio return − SPY return.'
            '</p>'
        )
        plain_parts.append(
            "  Note: α in GBM: rows is log-domain decimal at 21d horizon "
            "(post 2026-05-09 cutover). Top-of-email Daily Alpha is "
            "unchanged arithmetic portfolio − SPY."
        )
        plain_parts.append("")

    # ── Sector Attribution ─────────────────────────────────────────────────
    if sector_attribution:
        html_parts.append("<h2>Sector Attribution</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Sector</th><th>Weight</th><th>Contribution</th><th>Positions</th></tr>")
        plain_parts.append("SECTOR ATTRIBUTION")
        plain_parts.append("-" * 40)
        for sector, data in sorted(sector_attribution.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True):
            weight_pct = data["weight"] * 100
            contrib = data["contribution"]
            n_pos = data["positions"]
            html_parts.append(
                f"<tr><td>{sector}</td><td>{weight_pct:.1f}%</td>"
                f"<td>{_pct(contrib)}</td><td>{n_pos}</td></tr>"
            )
            plain_parts.append(f"  {sector:<25} {weight_pct:>5.1f}%  {_plain_pct(contrib):>8}  {n_pos} pos")
        html_parts.append("</table>")
        plain_parts.append("")

    # ── Trades today ─────────────────────────────────────────────────────────
    trades_today = conn.execute(
        "SELECT action, ticker, shares, price_at_order FROM trades WHERE date=? ORDER BY created_at",
        (run_date,),
    ).fetchall()

    html_parts.append("<h2>Trades Today</h2>")
    plain_parts.append("TRADES TODAY")
    plain_parts.append("-" * 40)

    if trades_today:
        html_parts += ["<table>",
                       "<tr><th>Action</th><th>Ticker</th><th>Shares</th><th>Price</th></tr>"]
        for action, ticker, shares, price in trades_today:
            price_str = f"${price:.2f}" if price else "—"
            html_parts.append(
                f"<tr><td>{action}</td><td>{ticker}</td>"
                f"<td>{shares:,}</td><td>{price_str}</td></tr>"
            )
            plain_parts.append(f"  {action:<6} {ticker:<6} {shares:>6} shares @ {price_str}")
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No trades today.</p>")
        plain_parts.append("  No trades today.")
    plain_parts.append("")

    # ── Roundtrip performance ────────────────────────────────────────────────
    if roundtrip_stats and roundtrip_stats.get("n_roundtrips", 0) > 0:
        rs = roundtrip_stats
        html_parts.append("<h2>Roundtrip Performance (All Time)</h2>")
        html_parts += [
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Closed Roundtrips</td><td>{rs['n_roundtrips']}</td></tr>",
            f"<tr><td>Avg Return</td><td>{_pct(rs.get('avg_return_pct'))}</td></tr>",
            f"<tr><td>Avg Alpha vs SPY</td><td>{_pct(rs.get('avg_alpha_pct'))}</td></tr>",
            f"<tr><td>Win Rate vs SPY</td><td>{rs.get('win_rate_vs_spy', 0):.0f}%</td></tr>",
            f"<tr><td>Avg Hold (days)</td><td>{rs.get('avg_hold_days', '—')}</td></tr>",
            "</table>",
        ]
        plain_parts.append("ROUNDTRIP PERFORMANCE (ALL TIME)")
        plain_parts.append("-" * 40)
        plain_parts.append(f"  Closed roundtrips: {rs['n_roundtrips']}")
        plain_parts.append(f"  Avg return:        {_plain_pct(rs.get('avg_return_pct'))}")
        plain_parts.append(f"  Avg alpha vs SPY:  {_plain_pct(rs.get('avg_alpha_pct'))}")
        plain_parts.append(f"  Win rate vs SPY:   {rs.get('win_rate_vs_spy', 0):.0f}%")
        plain_parts.append(f"  Avg hold (days):   {rs.get('avg_hold_days', '—')}")
        plain_parts.append("")

    # ── Trailing 10-day history ──────────────────────────────────────────────
    history = conn.execute(
        """SELECT date, portfolio_nav, daily_return_pct, spy_return_pct, daily_alpha_pct
           FROM eod_pnl ORDER BY date DESC LIMIT 10""",
    ).fetchall()

    html_parts.append("<h2>Trailing History</h2>")
    plain_parts.append("TRAILING HISTORY")
    plain_parts.append("-" * 40)

    if history:
        html_parts += ["<table>",
                       "<tr><th>Date</th><th>NAV</th><th>Return</th><th>SPY</th><th>Alpha</th></tr>"]
        for date_, hnav, ret, spy, alp in history:
            html_parts.append(
                f"<tr><td>{date_}</td><td>${hnav:,.0f}</td>"
                f"<td>{_pct(ret)}</td><td>{_pct(spy)}</td><td>{_pct(alp)}</td></tr>"
            )
            plain_parts.append(
                f"  {date_}  ${hnav:>10,.0f}  "
                f"ret {_plain_pct(ret):>8}  SPY {_plain_pct(spy):>8}  α {_plain_pct(alp):>8}"
            )
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No history yet.</p>")
        plain_parts.append("  No history yet.")

    html_body = _HTML.format(body="\n".join(html_parts), date=run_date)
    plain_body = "\n".join(plain_parts)

    return subject, html_body, plain_body


def send_eod_email(
    run_date: str,
    nav: float,
    daily_return: float | None,
    spy_return: float | None,
    alpha: float | None,
    positions: dict,
    conn: sqlite3.Connection,
    sender: str,
    recipients: list[str],
    region: str = "us-east-1",
    position_narratives: dict[str, str] | None = None,
    sector_attribution: dict | None = None,
    data_warnings: list[str] | None = None,
    roundtrip_stats: dict | None = None,
    trades_bucket: str = "",
    account_snapshot: dict | None = None,
    nav_reconciliation: dict | None = None,
) -> None:
    subject, html_body, plain_body = build_eod_email(
        run_date, nav, daily_return, spy_return, alpha, positions, conn,
        position_narratives=position_narratives,
        sector_attribution=sector_attribution,
        data_warnings=data_warnings,
        roundtrip_stats=roundtrip_stats,
        account_snapshot=account_snapshot,
        nav_reconciliation=nav_reconciliation,
    )

    # Archive email HTML to S3
    if trades_bucket:
        try:
            _s3 = boto3.client("s3")
            _s3.put_object(
                Bucket=trades_bucket,
                Key=f"consolidated/{run_date}/eod.html",
                Body=html_body.encode("utf-8"),
                ContentType="text/html",
            )
            logger.info("EOD email archived to S3: consolidated/%s/eod.html", run_date)
        except Exception as e:
            logger.warning("EOD email archival failed (non-fatal): %s", e)

    # SMTP/SES dispatch via the alpha_engine_lib.email_sender chokepoint
    # (L4356 — Gmail SMTP primary, SES fallback). Identical semantics to
    # the pre-consolidation inline path.
    from alpha_engine_lib.email_sender import send_email as _send_email
    _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=region,
    )
