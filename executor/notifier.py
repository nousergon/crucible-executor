"""
Telegram trade notification sender — daemon-side structured-message formatters.

Routes through flow-doctor ``notify_event()`` when the daemon's shared
``FlowDoctor`` instance is active (forum topic ``#trades`` via
``nousergon_lib.flow_doctor_fleet``). Falls back to the legacy
``nousergon_lib.telegram.send_message`` primitive when flow-doctor is
inactive (local dev / tests without yaml).

Migration arc: config#1741 (fleet Telegram consolidation T1).

config#1813: ``notify_event()`` returns a non-None report id on every
dispatch outcome except dedup — including ``severity_filtered`` (no
notifier opted in at this event's severity), which is exactly what a
stale/shadowed flow-doctor override yaml produces. Success logging here
checks ``FlowDoctor.last_dispatched()`` (flow-doctor>=0.8.3, via the
nousergon-lib[flow-doctor] pin) instead of trusting a non-None report id.
"""

from __future__ import annotations

import logging

from nousergon_lib.flow_doctor_fleet import trade_alert_dedup_key
from nousergon_lib.logging import get_flow_doctor
from nousergon_lib.telegram import send_message

logger = logging.getLogger(__name__)


def _format_trade_message(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str,
    source: str,
) -> str:
    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "REDUCE": "\U0001f7e1"}.get(
        action, "⚪"
    )
    return (
        f"{emoji} *{action} {ticker}*\n"
        f"Shares: {shares} @ ${price:.2f}\n"
        f"Trigger: {trigger}\n"
        f"Source: {source}"
    )


def send_trade_alert(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str = "",
    source: str = "daemon",
) -> bool:
    """Send a Telegram push notification for a trade execution."""
    msg = _format_trade_message(action, ticker, shares, price, trigger, source)
    fd = get_flow_doctor()
    if fd is not None:
        try:
            rid = fd.notify_event(
                f"{action} {ticker}",
                body=msg,
                severity="info",
                context={
                    "action": action,
                    "ticker": ticker,
                    "shares": shares,
                    "price": price,
                    "trigger": trigger,
                    "source": source,
                },
                dedup_key=trade_alert_dedup_key(action, ticker, shares, price),
            )
            # A non-None report id means the event was seen and persisted —
            # NOT that it reached a notifier. flow-doctor>=0.8.3's
            # last_dispatched()/last_dispatch_reason() expose the real
            # per-call outcome; a stale/shadowed override yaml missing the
            # trades Telegram topic returns a report id via
            # severity_filtered while sending nothing (config#1813 — this
            # is exactly the bug that logged a false "Telegram alert sent"
            # for ~all of 2026-07-06's morning trades).
            if rid is None:
                logger.warning(
                    "Telegram trade alert suppressed by flow-doctor dedup: %s %s",
                    action,
                    ticker,
                )
                return False
            if not fd.last_dispatched():
                logger.warning(
                    "Telegram trade alert NOT delivered by flow-doctor (reason=%s): %s %s",
                    fd.last_dispatch_reason(),
                    action,
                    ticker,
                )
                return False
            logger.info("Telegram alert sent via flow-doctor: %s %s", action, ticker)
            return True
        except Exception as exc:
            logger.warning(
                "flow-doctor notify_event failed for trade alert (%s %s): %s — falling back",
                action,
                ticker,
                exc,
            )

    ok = send_message(msg)
    if ok:
        logger.info("Telegram alert sent: %s %s", action, ticker)
    else:
        logger.warning("Telegram trade alert failed for %s %s", action, ticker)
    return ok


def send_daemon_status(message: str) -> bool:
    """Send a general status message (daemon start/stop, errors, IB events)."""
    fd = get_flow_doctor()
    if fd is not None:
        try:
            rid = fd.notify_event(message, severity="warning")
            if rid is None:
                return False
            return fd.last_dispatched()
        except Exception as e:
            logger.debug("flow-doctor notify_event failed, falling back to send_message: %s", e)
    return send_message(message)


def _normalize_flow_doctor_severity(severity: str) -> str:
    """Map alerts.publish severity strings to flow-doctor notify_on tiers."""
    normalized = severity.lower()
    if normalized == "warn":
        return "warning"
    return normalized


def publish_ops_alert(
    message: str,
    *,
    severity: str,
    source: str,
    dedup_key: str | None = None,
) -> None:
    """Dual-channel ops alert: SNS via alerts.publish + Telegram via flow-doctor.

    Migration arc: config#1740 T3 — retire raw ``telegram=True`` fan-out to
    General; executor flow-doctor.yaml routes by severity to forum topics.
    """
    from nousergon_lib import alerts as _alerts

    _alerts.publish(
        message=message,
        severity=severity,
        source=source,
        sns=True,
        telegram=False,
        dedup_key=dedup_key,
    )
    fd = get_flow_doctor()
    if fd is None:
        return
    try:
        subject = message.split("\n", 1)[0].replace("*", "").strip()
        if not subject:
            subject = f"Executor alert [{severity.upper()}]"
        fd.notify_event(
            subject,
            body=message,
            severity=_normalize_flow_doctor_severity(severity),
            dedup_key=dedup_key or subject,
            context={"source": source},
        )
    except Exception as exc:
        logger.warning(
            "flow-doctor notify_event failed for ops alert (%s): %s — SNS already sent",
            source,
            exc,
        )
