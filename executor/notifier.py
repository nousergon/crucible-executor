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
import os
from datetime import datetime

import pytz
from nousergon_lib.flow_doctor_fleet import trade_alert_dedup_key
from nousergon_lib.logging import get_flow_doctor
from nousergon_lib.telegram import send_message

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")


# Messages Brian acts on every day must never be dropped by flow-doctor's
# daily alert cap. On 2026-10-23 the 12:37Z "Order book written" message was
# dropped as `rate_limited` because overnight reconcile reruns had used the
# executor's 10/day budget after UTC midnight. When flow-doctor reports
# `rate_limited` for an ESSENTIAL message, we send it straight to the same
# forum topic instead. Repeats are still bounded by flow-doctor's dedup, which
# runs before the cap.
_RATE_LIMITED = "rate_limited"
# alpha-engine-config-I12702 (2026-10-26): seven executor notifications came
# back `delivery_failed` (flow-doctor's heartbeat: delivery_failed=7, fired=4,
# rate_limited=3) -- flow-doctor attempted the send and the transport failed --
# and an essential one in that state was logged at WARNING and dropped,
# because only `rate_limited` was re-sent.
# Those failed attempts then counted against the same 10/day budget (the
# store counts every telegram_alert action, sent or not), so by 13:53Z the
# cap held BUY PLTR as well, and the past-cap send went to the very topic
# that had just failed seven times, and failed the same way. An essential
# message is now re-sent on either reason, and the direct path no longer
# depends on the one topic being healthy: it tries the message's own topic,
# then the ops-health topic, then the chat with no topic, and says which
# route worked. `severity_filtered` etc. are configuration verdicts, not
# delivery failures, and are still not re-sent.
_DELIVERY_FAILED = "delivery_failed"
_RESEND_REASONS = frozenset({_RATE_LIMITED, _DELIVERY_FAILED})
_TOPIC_ENV_BY_SEVERITY = {
    "critical": "FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
    "error": "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
    "warning": "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
    "info": "FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
}
# The transport logs Telegram's own error description (never the token) on
# this logger; capturing it is what lets the failure page name its cause.
_TRANSPORT_LOGGER = "krepis.telegram"
# One ERROR per process per refusing topic; later re-routes stay WARNING so a
# broken topic is reported once a day, not once per trade.
_REROUTE_REPORTED: set[int | None] = set()


def _topic_thread_id(severity: str) -> int | None:
    raw = os.environ.get(_TOPIC_ENV_BY_SEVERITY.get(severity, ""), "")
    try:
        return int(raw) if raw.strip() else None
    except ValueError:
        return None


class _TransportErrorCapture(logging.Handler):
    """Collect the transport's WARNING+ lines for the duration of one send."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = record.getMessage()
        except Exception:  # noqa: BLE001 -- diagnostic capture only
            return
        if line not in self.lines:
            self.lines.append(line)

    def summary(self) -> str:
        return ("; ".join(self.lines) or "no transport detail logged")[:400]


def _send_past_cap(
    message: str, severity: str, label: str, reason: str = _RATE_LIMITED,
) -> bool:
    """Deliver an essential message that flow-doctor did not deliver.

    ``reason`` is flow-doctor's dispatch verdict (``rate_limited`` or
    ``delivery_failed``). Routes, in order: the message's own topic, the
    ops-health topic, then the chat with no topic.
    """
    primary = _topic_thread_id(severity)
    routes: list[int | None] = [primary]
    ops_health = _topic_thread_id("warning")
    if ops_health is not None and ops_health not in routes:
        routes.append(ops_health)
    if None not in routes:
        routes.append(None)

    capture = _TransportErrorCapture()
    transport_logger = logging.getLogger(_TRANSPORT_LOGGER)
    prior_level = transport_logger.level
    if transport_logger.getEffectiveLevel() > logging.WARNING:
        # The cause must be captured even where the process logs at ERROR.
        transport_logger.setLevel(logging.WARNING)
    transport_logger.addHandler(capture)
    try:
        for i, thread in enumerate(routes):
            text = message if i == 0 else (
                f"[re-routed: Telegram topic {primary} refused this message]\n"
                f"{message}"
            )
            if not send_message(text, message_thread_id=thread):
                continue
            if i == 0:
                logger.warning(
                    "Essential Telegram message sent directly (flow-doctor %s): %s",
                    reason, label,
                )
            elif primary not in _REROUTE_REPORTED:
                _REROUTE_REPORTED.add(primary)
                logger.error(
                    "Telegram topic %s refused an essential message; delivered via "
                    "topic %s instead (flow-doctor %s). Telegram said: %s "
                    "(alpha-engine-config-I12702)",
                    primary, thread, reason, capture.summary(),
                )
            else:
                logger.warning(
                    "Essential Telegram message re-routed from topic %s to %s: %s",
                    primary, thread, label,
                )
            return True
    finally:
        transport_logger.removeHandler(capture)
        transport_logger.setLevel(prior_level)

    logger.error(
        "Essential Telegram message NOT delivered on any route (flow-doctor %s; "
        "tried topics %s): %s. Telegram said: %s (alpha-engine-config-I12702)",
        reason, routes, label, capture.summary(),
    )
    return False


def _format_fill_time(fill_time: str | None) -> str:
    """Render a trade's execution timestamp in US/Eastern for the alert.

    Falls back to "now" (ET) when ``fill_time`` is missing or unparseable —
    e.g. a bracket-order fill that didn't populate IB's execution time.
    """
    if fill_time:
        try:
            dt = datetime.fromisoformat(str(fill_time).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            dt = dt.astimezone(_ET)
        except ValueError:
            dt = datetime.now(_ET)
    else:
        dt = datetime.now(_ET)
    return dt.strftime("%Y-%m-%d %H:%M:%S ET")


def _format_trade_message(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str,
    source: str,
    fill_time: str | None = None,
    realized_pnl: float | None = None,
    realized_return_pct: float | None = None,
    realized_alpha_pct: float | None = None,
    days_held: int | None = None,
) -> str:
    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "REDUCE": "\U0001f7e1"}.get(action, "⚪")
    lines = [
        f"{emoji} *{action} {ticker}*",
        f"Shares: {shares} @ ${price:.2f}",
        f"Time: {_format_fill_time(fill_time)}",
        f"Trigger: {trigger}",
        f"Source: {source}",
    ]
    # Realized P&L only applies to closing/reducing trades (EXIT/REDUCE/
    # COVER) — an ENTER/BUY has no prior fill to compare against, so
    # realized_pnl stays None and this block is skipped for entries.
    if realized_pnl is not None:
        pct_str = f" ({realized_return_pct:+.1f}%)" if realized_return_pct is not None else ""
        lines.append(f"Realized P&L: ${realized_pnl:+,.2f}{pct_str}")
    if realized_alpha_pct is not None:
        held_str = f" | Held: {days_held}d" if days_held is not None else ""
        lines.append(f"Alpha vs SPY: {realized_alpha_pct:+.1f}%{held_str}")
    return "\n".join(lines)


def send_trade_alert(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str = "",
    source: str = "daemon",
    fill_time: str | None = None,
    realized_pnl: float | None = None,
    realized_return_pct: float | None = None,
    realized_alpha_pct: float | None = None,
    days_held: int | None = None,
) -> bool:
    """Send a Telegram push notification for a trade execution."""
    msg = _format_trade_message(
        action,
        ticker,
        shares,
        price,
        trigger,
        source,
        fill_time=fill_time,
        realized_pnl=realized_pnl,
        realized_return_pct=realized_return_pct,
        realized_alpha_pct=realized_alpha_pct,
        days_held=days_held,
    )
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
                    "realized_pnl": realized_pnl,
                    "realized_return_pct": realized_return_pct,
                    "realized_alpha_pct": realized_alpha_pct,
                    "days_held": days_held,
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
                reason = str(fd.last_dispatch_reason())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType] -- FlowDoctor is untyped here
                if reason in _RESEND_REASONS:
                    return _send_past_cap(msg, "info", f"{action} {ticker}", reason)
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


def send_daemon_status(message: str, *, essential: bool = False) -> bool:
    """Send a general status message (daemon start/stop, errors, IB events).

    ``essential=True`` marks a message the operator acts on (the daily order
    book, a crash, a failed order): if flow-doctor's daily cap holds it back,
    it is sent directly instead of being dropped.
    """
    fd = get_flow_doctor()
    if fd is not None:
        try:
            rid = fd.notify_event(message, severity="warning")
            if rid is None:
                return False
            if fd.last_dispatched():
                return True
            reason = str(fd.last_dispatch_reason())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType] -- FlowDoctor is untyped here
            if essential and reason in _RESEND_REASONS:
                label = message.split("\n", 1)[0].replace("*", "").strip()
                return _send_past_cap(message, "warning", label, reason)
            return False
        except Exception as e:
            # (a) flow-doctor's own notify_event transport failed.
            # (c) not recorded elsewhere — deliberate carve-out
            # (alpha-engine-config-I10031): this is a transport swallow
            # with an explicit fallback path (send_message below), the
            # same carve-out class as the Telegram-transport swallows.
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
