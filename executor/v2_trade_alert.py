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
no record, or a session with no orders is said so in the summary.

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
        missing = self.client.exceptions.NoSuchKey
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(relative))
        except missing:
            return None
        body = cast(BinaryIO, response["Body"])
        loaded: object = json.loads(body.read())
        return cast(Doc, loaded) if isinstance(loaded, dict) else None

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


def summarize(day: str, run: Mapping[str, object], shortfall: Mapping[str, object] | None) -> str:
    """The essential one-message summary of a session."""
    if run.get("status") != "ok":
        return (
            f"*v2 session {day}: NO ORDERS PLACED*\n"
            f"Run status: {run.get('status')}\n"
            f"Reason: {run.get('reason') or 'none recorded'}\n"
            "Auto-fix: none; the session timer does not catch up. "
            "Resolution: fix the cause before the next 10:45 ET session, or rerun today."
        )
    orders = _docs(shortfall, "orders")
    if not orders:
        reason = _get(shortfall, "reason") or "the book needed no orders"
        return f"*v2 session {day}: no orders*\n{reason}"
    filled = [o for o in orders if _num(o, "filled_quantity") > 0]
    sells = sum(_num(o, "notional_usd") for o in filled if o.get("side") == "sell")
    buys = sum(_num(o, "notional_usd") for o in filled if o.get("side") == "buy")
    lines = [
        f"*v2 session {day}: {len(filled)}/{len(orders)} orders filled*",
        f"Sells {_money(sells)} · Buys {_money(buys)}",
    ]
    for metric in _docs(shortfall, "metrics"):
        if metric.get("name") == "execution_shortfall_bps" and metric.get("value") is not None:
            lines.append(f"Shortfall {_num(metric, 'value'):+.1f} bps ({metric.get('status')})")
    unfilled = [str(o.get("symbol")) for o in orders if _num(o, "filled_quantity") <= 0]
    if unfilled:
        lines.append(f"UNFILLED: {', '.join(unfilled)}")
    champion = _get(shortfall, "champion")
    if champion:
        lines.append(f"Champion: {champion}")
    return "\n".join(lines)


def run_alerts(
    store: Store,
    *,
    now: dt.datetime,
    trade: Callable[..., bool] = send_trade_alert,
    status: Callable[..., bool] = send_daemon_status,
) -> int:
    """Send the session's alerts; returns how many messages were delivered."""
    newest = store.newest_run()
    today = now.astimezone(_NEW_YORK).date().isoformat()
    if newest is None or now - newest[1] > FRESH_WITHIN:
        status(
            f"*v2 session {today}: NO RUN RECORD*\n"
            "The session ended without writing runs/trader.session/{day}/run.json, "
            "so no order was placed. The unit's own failure page carries its journal.",
            essential=True,
        )
        return 1
    day = newest[0]
    run = store.read_json(f"{RUNS_PREFIX}{day}/run.json") or {}
    shortfall = store.read_json(f"trader/execution_shortfall/{day}.json") if run.get("status") == "ok" else None
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
    sent += int(bool(status(summarize(day, run, shortfall), essential=True)))
    return sent


def main(argv: list[str] | None = None, *, now: dt.datetime | None = None) -> int:
    """``python -m executor.v2_trade_alert``; takes no arguments."""
    if argv:
        print(f"usage: v2_trade_alert (no arguments); got {argv!r}", file=sys.stderr)
        return 2
    store = open_store(read_env_file(TRADER_ENV))
    delivered = run_alerts(store, now=now or dt.datetime.now(dt.UTC))
    logger.info("v2 trade alerts delivered: %d", delivered)
    return 0


if __name__ == "__main__":
    from krepis.logging import guard_entrypoint, setup_logging

    from executor.config_loader import get_flow_doctor_yaml_path

    setup_logging("v2-trade-alert", flow_doctor_yaml=get_flow_doctor_yaml_path())
    with guard_entrypoint():
        code = main(sys.argv[1:])
    sys.exit(code)
