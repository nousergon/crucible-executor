"""
EOD performance email — sent after market close on trading days.

Builds a summary of daily NAV, alpha vs SPY, open positions, trades placed,
and a trailing 10-day history, then sends via AWS SES.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import boto3
from botocore.exceptions import ClientError

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
) -> tuple[str, str, str]:
    """
    Build the EOD email subject + (html_body, plain_body).

    Returns: (subject, html_body, plain_body)
    """
    alpha_str = _plain_pct(alpha)
    nav_fmt = f"${nav:,.0f}"
    subject = f"Alpha Engine | {run_date} | NAV {nav_fmt} | α {alpha_str}"

    # ── Daily summary ────────────────────────────────────────────────────────
    html_parts = ["<h2>Daily Summary</h2>", "<table>",
                  "<tr><th>Metric</th><th>Value</th></tr>",
                  f"<tr><td>NAV</td><td>{nav_fmt}</td></tr>",
                  f"<tr><td>Daily Return</td><td>{_pct(daily_return)}</td></tr>",
                  f"<tr><td>SPY Return</td><td>{_pct(spy_return)}</td></tr>",
                  f"<tr><td>Daily Alpha</td><td>{_pct(alpha)}</td></tr>",
                  "</table>"]

    plain_parts = [
        f"Alpha Engine EOD — {run_date}",
        "=" * 40,
        f"NAV:          {nav_fmt}",
        f"Daily Return: {_plain_pct(daily_return)}",
        f"SPY Return:   {_plain_pct(spy_return)}",
        f"Daily Alpha:  {_plain_pct(alpha)}",
        "",
    ]

    # ── Positions ────────────────────────────────────────────────────────────
    html_parts.append("<h2>Open Positions</h2>")
    plain_parts.append("OPEN POSITIONS")
    plain_parts.append("-" * 40)

    if positions:
        html_parts += ["<table>",
                       "<tr><th>Ticker</th><th>Shares</th><th>Market Value</th><th>% NAV</th></tr>"]
        for ticker, pos in sorted(positions.items()):
            mv = pos.get("market_value", 0)
            pct_nav = mv / nav * 100 if nav else 0
            html_parts.append(
                f"<tr><td>{ticker}</td><td>{pos['shares']:,}</td>"
                f"<td>${mv:,.0f}</td><td>{pct_nav:.1f}%</td></tr>"
            )
            plain_parts.append(f"  {ticker:<6} {pos['shares']:>6} shares  ${mv:>10,.0f}  {pct_nav:.1f}% NAV")
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No open positions.</p>")
        plain_parts.append("  No open positions.")
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
) -> None:
    subject, html_body, plain_body = build_eod_email(
        run_date, nav, daily_return, spy_return, alpha, positions, conn
    )

    ses = boto3.client("ses", region_name=region)
    try:
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        logger.info(f"EOD email sent: '{subject}' → {recipients}")
    except ClientError as e:
        logger.error(f"SES send failed: {e.response['Error']['Message']}")
    except Exception as e:
        logger.error(f"EOD email error: {e}")
