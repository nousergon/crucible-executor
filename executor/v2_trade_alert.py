"""Telegram alerts for the v2 trader's daily session (alpha-engine-config-I13005).

Run by ``alpha-engine-v2-trade-alert.service``, which
``alpha-engine-trader-session.service`` starts when it ends, successfully or
not (``OnSuccess=`` and ``OnFailure=``).

**Why this exists.** When paper orders moved from this executor to the v2
trader (2026-11-03), the per-trade Telegram alerts stayed behind: the v2
trader is a public repository with no Telegram path, and it placed its first
13 orders on 2026-11-04 without a single alert. Three no-trade days before
that were equally silent.

**What it does.** Reads what the session WROTE, never the broker: the newest
``runs/trader.session/{day}/run.json`` and, when the run is ``ok``,
``trader/execution_shortfall/{day}.json``. It sends one trade alert per
filled order through :func:`executor.notifier.send_trade_alert` (the same
``#trades`` topic v1's fills used) and one essential session summary through
:func:`executor.notifier.send_daemon_status`. A refused book, a run that wrote
no record, or a session with no orders is said so in the summary, and so is
whether the session's order book was written (``trader_order_book_written``).

**Absent is not unreadable** (alpha-engine-config-I13284). S3 answers a GET
without ListBucket with AccessDenied whether the key is absent or merely
unreadable, so the run record decides: a shortfall file the run lists among
its ``outputs`` exists, and failing to read it is reported loudly as fills
that could not be read, never as "no orders". The process exits non-zero when
the summary was not delivered, the run record could not be read, or the
fills the run wrote could not be read, which pages through
``alpha-engine-trader-failed@``.

**Credentials.** The v2 store is read as the trader's own role, through the
profile named in the trader's env file. Everything else (the Telegram
secrets and topics in SSM) is read under this box's instance role, exactly as
v1's alerts are, so ``AWS_PROFILE`` is never put into this process's
environment.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from zoneinfo import ZoneInfo

from executor.notifier import send_daemon_status, send_trade_alert

logger = logging.getLogger(__name__)

TRADER_ENV = Path("/home/ec2-user/alpha-engine-config/crucible-trader/trader.env")
RUNS_PREFIX = "runs/trader.session/"
#: A run record older than this was not written by the session that just
#: ended: this run wrote nothing (it died before its manifest).
FRESH_WITHIN = dt.timedelta(hours=3)

_NEW_YORK = ZoneInfo("America/New_York")


Doc = dict[str, object]


def _get(doc: Mapping[str, object] | None, key: str) -> object:
    return None if doc is None else doc.get(key)


def _docs(doc: Mapping[str, object] | None, key: str) -> list[Doc]:
    """A list of JSON objects under ``key``; anything else reads as empty."""
    value = _get(doc, key)
    if not isinstance(value, list):
        return []
    return [cast(Doc, item) for item in cast(list[object], value) if isinstance(item, dict)]


def _num(doc: Mapping[str, object], key: str) -> float:
    value = doc.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


#: What S3 returns for a GET of an absent key: NoSuchKey with ListBucket on
#: the bucket, AccessDenied (403) without it. AccessDenied is also what an
#: unreadable key returns, so it reads as absent only when the run record does
#: not list the key among what it wrote (:func:`_read_shortfall`).
_ABSENT_CODES = frozenset({"NoSuchKey", "404", "AccessDenied", "403"})


def _error_code(exc: Exception) -> str:
    """The botocore ``Error.Code`` of ``exc``, or ``""`` for any other error."""
    response: object = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    error: object = cast(dict[str, object], response).get("Error")
    if not isinstance(error, dict):
        return ""
    return str(cast(dict[str, object], error).get("Code", ""))


class _Exceptions(Protocol):
    NoSuchKey: type[Exception]


class _Paginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str) -> Iterable[Doc]: ...  # noqa: N803


class S3Client(Protocol):
    """The three S3 calls this module makes (a boto3 client, or a test fake)."""

    @property
    def exceptions(self) -> _Exceptions: ...

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]: ...  # noqa: N803

    def get_paginator(self, name: str) -> _Paginator: ...


@dataclass(frozen=True)
class Store:
    """The v2 store's bucket and key prefix, read through one S3 client."""

    client: S3Client
    bucket: str
    prefix: str

    def key(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def read_json(self, relative: str) -> Doc | None:
        """The document, or ``None`` when absent (or AccessDenied); other errors raise.

        For readers that cannot tell absent from unreadable any better;
        :meth:`read` is the one that keeps the difference.
        """
        missing = self.client.exceptions.NoSuchKey
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(relative))
        except missing:
            return None
        except Exception as exc:
            # The trader role has no bucket-level ListBucket, so S3 answers a
            # GET for an absent key with 403 AccessDenied, not NoSuchKey
            # (alpha-engine-config-I13279).
            if _error_code(exc) in _ABSENT_CODES:
                return None
            raise
        body = cast(BinaryIO, response["Body"])
        loaded: object = json.loads(body.read())
        return cast(Doc, loaded) if isinstance(loaded, dict) else None

    def read(self, relative: str) -> tuple[Doc | None, str | None]:
        """``(document, None)``, ``(None, None)`` when plainly absent, or ``(None, reason)``.

        ``reason`` is set whenever the object could not be read and S3 did not
        say it is absent: ``AccessDenied`` is returned as a reason (it may
        still mean absent; the caller decides from the run record), and so is
        any other error, instead of raising.
        """
        missing = self.client.exceptions.NoSuchKey
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(relative))
            body = cast(BinaryIO, response["Body"])
            loaded: object = json.loads(body.read())
        except missing:
            return None, None
        except Exception as exc:
            code = _error_code(exc)
            if code in ("NoSuchKey", "404"):
                return None, None
            return None, code or type(exc).__name__
        if not isinstance(loaded, dict):
            return None, "not a JSON object"
        return cast(Doc, loaded), None

    def newest_run(self) -> tuple[str, dt.datetime] | None:
        """``(served day, LastModified)`` of the newest session run record."""
        newest: tuple[str, dt.datetime] | None = None
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.key(RUNS_PREFIX)):
            for obj in _docs(page, "Contents"):
                key = str(obj.get("Key", ""))
                modified = obj.get("LastModified")
                if not key.endswith("/run.json") or not isinstance(modified, dt.datetime):
                    continue
                if newest is None or modified > newest[1]:
                    newest = (key.rsplit("/", 2)[-2], modified)
        return newest


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of a systemd env file; comments and blanks skipped."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values


def open_store(env: Mapping[str, str]) -> Store:
    """The trader's store, read as the trader's role (its profile, its config file)."""
    import boto3  # pyright: ignore[reportMissingTypeStubs]
    import botocore.session  # pyright: ignore[reportMissingTypeStubs]

    uri = env["CRUCIBLE_TRADER_STORE_URI"]
    if not uri.startswith("s3://"):
        raise ValueError(f"CRUCIBLE_TRADER_STORE_URI is not an s3 URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://") :].partition("/")
    core = botocore.session.Session()
    if env.get("AWS_CONFIG_FILE"):
        core.set_config_variable("config_file", env["AWS_CONFIG_FILE"])  # pyright: ignore[reportUnknownMemberType]
    if env.get("AWS_PROFILE"):
        core.set_config_variable("profile", env["AWS_PROFILE"])  # pyright: ignore[reportUnknownMemberType]
    session = boto3.Session(botocore_session=core, region_name=env.get("AWS_REGION", "us-east-1"))
    client = cast(S3Client, session.client("s3"))  # pyright: ignore[reportUnknownMemberType]
    return Store(client, bucket, prefix.strip("/"))


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _held_reason(run: Mapping[str, object]) -> str | None:
    """Why an ``ok`` session served nothing, or ``None`` when it served.

    A held session (no predictions feed, say) ends ``ok`` with the
    ``trader_session_served`` flag at 0 and writes no execution_shortfall
    file; without this it read as "the book needed no orders".
    """
    for metric in _docs(run, "metrics"):
        if metric.get("name") == "trader_session_served" and metric.get("value") == 0:
            return str(metric.get("status_reason") or "none recorded")
    return None


#: How much of a store error the summary carries. Botocore's AccessDenied
#: text goes on to name the caller's role; that adds nothing to the page.
_REASON_CHARS = 160


def _short(reason: object) -> str:
    """``reason`` cut before any ARN, and to :data:`_REASON_CHARS`."""
    text = str(reason or "none recorded")
    cut = text.find("arn:")
    if cut > 0:
        text = text[:cut].removesuffix("User: ").rstrip(" :")
    return text if len(text) <= _REASON_CHARS else text[: _REASON_CHARS - 1] + "…"


def shortfall_key(day: str) -> str:
    return f"trader/execution_shortfall/{day}.json"


def run_lists_output(run: Mapping[str, object], relative: str) -> bool:
    """Whether the run record says the run WROTE ``relative`` (store-relative key)."""
    keys = [str(o.get("key")) for o in _docs(run, "outputs")]
    link = _get(run, "money_path_link")
    if isinstance(link, dict):
        writes: object = cast(dict[str, object], link).get("money_path_writes")
        if isinstance(writes, list):
            keys += [str(w) for w in cast(list[object], writes)]
    return relative in keys


def order_book_line(run: Mapping[str, object]) -> str | None:
    """The session's order book, from the ``trader_order_book_written`` metric."""
    for metric in _docs(run, "metrics"):
        if metric.get("name") != "trader_order_book_written":
            continue
        if metric.get("status") == "OK" and _num(metric, "value") > 0:
            return "Order book: written"
        return f"*ORDER BOOK FAILED*: {_short(metric.get('status_reason'))}"
    return None


def _with_order_book(run: Mapping[str, object], lines: list[str]) -> str:
    book = order_book_line(run)
    return "\n".join([*lines, book] if book else lines)


def summarize(
    day: str,
    run: Mapping[str, object],
    shortfall: Mapping[str, object] | None,
    *,
    unreadable: str | None = None,
) -> str:
    """The essential one-message summary of a session.

    ``unreadable`` is why the execution_shortfall file the run wrote could not
    be read; the summary then says orders were placed and their fills are
    unknown, never "no orders".

    **A failed run is "no orders" only when it filed none.** The session runs
    all day (sells at 09:35, buys on their triggers until 15:55 ET), so a run
    that fails at 13:00 may have sent its sells and some buys; the trader files
    those orders before the run fails (``trader.session``'s run lists the
    execution_shortfall file among its outputs). Such a run is summarised with
    its orders, and ``NO ORDERS PLACED`` is said only of a failed run that
    filed no order at all.
    """
    if run.get("status") != "ok":
        failure = [
            f"Run status: {run.get('status')}",
            f"Reason: {run.get('reason') or 'none recorded'}",
        ]
        if unreadable is not None:
            return _with_order_book(
                run,
                [
                    f"*v2 session {day}: RUN FAILED after ORDERS WERE PLACED — fill details "
                    f"unreadable ({_short(unreadable)})*",
                    *failure,
                    f"The run wrote {shortfall_key(day)} before it failed but this alert could not "
                    "read it; check the broker for what is working and what filled.",
                    "Auto-fix: none. Orders already at the broker stand; the session's remaining "
                    "buys were not sent. Resolution: fix the cause, then rerun today (a rerun "
                    "adopts the orders already sent and never resends them).",
                ],
            )
        orders = _docs(shortfall, "orders")
        if orders:
            filled = [o for o in orders if _num(o, "filled_quantity") > 0]
            return _with_order_book(
                run,
                [
                    f"*v2 session {day}: RUN FAILED after {len(orders)} order(s) placed, {len(filled)} filled*",
                    *failure,
                    *_order_lines(shortfall, orders),
                    "Auto-fix: none. Orders already at the broker stand; the session's remaining "
                    "buys were not sent. Resolution: fix the cause, then rerun today (a rerun "
                    "adopts the orders already sent and never resends them).",
                ],
            )
        return _with_order_book(
            run,
            [
                f"*v2 session {day}: NO ORDERS PLACED*",
                f"Run status: {run.get('status')}",
                f"Reason: {run.get('reason') or 'none recorded'}",
                "Auto-fix: none; the session timer does not catch up. "
                "Resolution: fix the cause before the next 09:35 ET session, or rerun today.",
            ],
        )
    held = _held_reason(run)
    if held is not None:
        return _with_order_book(
            run,
            [
                f"*v2 session {day}: HELD, NO ORDERS PLACED*",
                f"Reason: {held}",
                "Resolution: once the missing input is published, rerun the session today.",
            ],
        )
    if unreadable is not None:
        return _with_order_book(
            run,
            [
                f"*v2 session {day}: ORDERS PLACED — fill details unreadable ({_short(unreadable)})*",
                f"The run wrote {shortfall_key(day)} but this alert could not read it, "
                "so no per-fill alert was sent and the fills are unconfirmed here.",
                "Auto-fix: none. Resolution: give the trader role read on trader/execution_shortfall/, "
                "then start alpha-engine-v2-trade-alert.service within 3 hours of the session to resend its alerts.",
            ],
        )
    orders = _docs(shortfall, "orders")
    if not orders:
        reason = _get(shortfall, "reason") or "the book needed no orders"
        return _with_order_book(run, [f"*v2 session {day}: no orders*", str(reason)])
    filled = [o for o in orders if _num(o, "filled_quantity") > 0]
    lines = [f"*v2 session {day}: {len(filled)}/{len(orders)} orders filled*", *_order_lines(shortfall, orders)]
    return _with_order_book(run, lines)


def _order_lines(shortfall: Mapping[str, object] | None, orders: list[Doc]) -> list[str]:
    """What the filed orders did: notional filled each way, shortfall, unfilled, champion."""
    filled = [o for o in orders if _num(o, "filled_quantity") > 0]
    sells = sum(_num(o, "notional_usd") for o in filled if o.get("side") == "sell")
    buys = sum(_num(o, "notional_usd") for o in filled if o.get("side") == "buy")
    lines = [f"Sells {_money(sells)} · Buys {_money(buys)}"]
    for metric in _docs(shortfall, "metrics"):
        if metric.get("name") == "execution_shortfall_bps" and metric.get("value") is not None:
            lines.append(f"Shortfall {_num(metric, 'value'):+.1f} bps ({metric.get('status')})")
    unfilled = [str(o.get("symbol")) for o in orders if _num(o, "filled_quantity") <= 0]
    if unfilled:
        lines.append(f"UNFILLED: {', '.join(unfilled)}")
    champion = _get(shortfall, "champion")
    if champion:
        lines.append(f"Champion: {champion}")
    return lines


@dataclass(frozen=True)
class Outcome:
    """What one alert run delivered, and whether it must page."""

    delivered: int
    summary_delivered: bool
    run_unreadable: bool = False
    #: The run wrote fills this alert could not read: no per-fill alert went out.
    fills_unreadable: bool = False

    @property
    def failed(self) -> bool:
        return not self.summary_delivered or self.run_unreadable or self.fills_unreadable


def _read_shortfall(store: Store, day: str, run: Mapping[str, object]) -> tuple[Doc | None, str | None]:
    """``(shortfall, unreadable reason)`` for an ``ok`` run.

    AccessDenied reads as absent only when the run does not list the file
    among its outputs: a held session writes none, and the trader role has no
    ListBucket to say NoSuchKey (alpha-engine-config-I13279).
    """
    key = shortfall_key(day)
    written = run_lists_output(run, key)
    shortfall, reason = store.read(key)
    if reason is not None:
        if reason in _ABSENT_CODES and not written:
            return None, None
        return None, reason
    if shortfall is None and written:
        return None, "listed in the run's outputs but not found"
    return shortfall, None


def run_alerts(
    store: Store,
    *,
    now: dt.datetime,
    trade: Callable[..., bool] = send_trade_alert,
    status: Callable[..., bool] = send_daemon_status,
) -> Outcome:
    """Send the session's alerts; returns what was delivered."""
    newest = store.newest_run()
    today = now.astimezone(_NEW_YORK).date().isoformat()
    if newest is None or now - newest[1] > FRESH_WITHIN:
        ok = status(
            f"*v2 session {today}: NO RUN RECORD*\n"
            "The session ended without writing runs/trader.session/{day}/run.json. "
            "It runs all day, so orders it sent before it died may be at the broker; "
            "check the broker. The unit's own failure page carries its journal.",
            essential=True,
        )
        return Outcome(int(bool(ok)), bool(ok))
    day = newest[0]
    run, run_error = store.read(f"{RUNS_PREFIX}{day}/run.json")
    if run is None:
        reason = run_error or "listed but not found"
        ok = status(
            f"*v2 session {day}: RUN RECORD UNREADABLE ({_short(reason)})*\n"
            "Whether orders were placed is unknown here; no per-fill alert was sent.\n"
            "Auto-fix: none. Resolution: give the trader role read on runs/trader.session/, "
            "then start alpha-engine-v2-trade-alert.service within 3 hours of the session.",
            essential=True,
        )
        return Outcome(int(bool(ok)), bool(ok), run_unreadable=True)
    # A failed run is read too when it FILED its orders before failing (the
    # all-day session can fail after its sells and some buys went out).
    shortfall, unreadable = (
        _read_shortfall(store, day, run)
        if run.get("status") == "ok" or run_lists_output(run, shortfall_key(day))
        else (None, None)
    )
    sent = 0
    for order in _docs(shortfall, "orders"):
        quantity = _num(order, "filled_quantity")
        if quantity <= 0:
            continue
        price = _num(order, "fill_price") or _num(order, "decision_price")
        ok = trade(
            "BUY" if order.get("side") == "buy" else "SELL",
            str(order.get("symbol")),
            int(quantity),
            price,
            trigger=f"v2 book {day} (decision ${_num(order, 'decision_price'):,.2f})",
            source="crucible v2 trader",
            fill_time=order.get("submitted_at_utc"),
        )
        sent += int(bool(ok))
    delivered = bool(status(summarize(day, run, shortfall, unreadable=unreadable), essential=True))
    return Outcome(sent + int(delivered), delivered, fills_unreadable=unreadable is not None)


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_trade_alert``; takes no arguments."""
    if argv:
        print(f"usage: v2_trade_alert (no arguments); got {argv!r}", file=sys.stderr)
        return 2
    store = open_store(read_env_file(TRADER_ENV))
    outcome = run_alerts(store, now=now or dt.datetime.now(dt.UTC))
    logger.info("v2 trade alerts delivered: %d", outcome.delivered)
    if outcome.run_unreadable:
        logger.error("v2 trade alert: the session's run record could not be read")
    if outcome.fills_unreadable:
        logger.error("v2 trade alert: the session's fills could not be read; no per-fill alert was sent")
    if not outcome.summary_delivered:
        logger.error("v2 trade alert: the session summary was not delivered")
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-trade-alert", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
