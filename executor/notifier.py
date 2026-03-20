"""
Telegram trade notification sender.

Sends push notifications for intraday trade executions via Telegram bot API.
Fire-and-forget with a 5-second timeout — a failed notification never blocks
trade execution.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → save the bot token
  2. Add to .env: TELEGRAM_BOT_TOKEN=<token>
  3. Message the bot, then call getUpdates to get your chat_id
  4. Add to .env: TELEGRAM_CHAT_ID=<chat_id>
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_trade_alert(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str = "",
    source: str = "daemon",
) -> bool:
    """
    Send a Telegram push notification for a trade execution.

    Returns True if sent successfully, False otherwise.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping notification")
        return False

    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "REDUCE": "\U0001f7e1"}.get(action, "\u26aa")
    msg = (
        f"{emoji} *{action} {ticker}*\n"
        f"Shares: {shares} @ ${price:.2f}\n"
        f"Trigger: {trigger}\n"
        f"Source: {source}"
    )

    try:
        resp = requests.post(
            _TELEGRAM_API.format(token=token),
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=5,
        )
        if resp.status_code == 200:
            logger.info("Telegram alert sent: %s %s", action, ticker)
            return True
        else:
            logger.warning("Telegram API returned %d: %s", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)
        return False


def send_daemon_status(message: str) -> bool:
    """Send a general status message (daemon start/stop, errors)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    try:
        requests.post(
            _TELEGRAM_API.format(token=token),
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=5,
        )
        return True
    except Exception:
        return False
