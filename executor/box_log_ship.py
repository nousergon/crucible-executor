"""Copy today's trading-box logs to S3, so a read-only session can read them (alpha-engine-config-I12024).

Run by ``alpha-engine-v2-box-log-ship.timer`` every ten minutes while the box
is up, and once by :mod:`executor.v2_day_end` just before it powers the box off.

**Why this exists.** On 2026-11-11 a false "code pull failed / watch silent"
alert and a stray evening session start had to be diagnosed by inference: the
box's journals and ``/var/log/boot-pull.log`` reach no log group, and an agent
session holds read-only AWS access, which cannot run a command on the box. The
box stops every evening, so whatever it logged is out of reach until it next
boots. This copies the day's logs off the box while it is still up.

**What it copies.** For each v2 unit (:data:`UNIT_PATTERNS`) that logged today,
``journalctl -o short-iso`` since New York midnight, and the tail of
``/var/log/boot-pull.log``; each capped at :data:`MAX_BYTES` (its newest part).
Each lands at ``<prefix>/trading/<YYYY-MM-DD New York>/<unit>.log`` and is
overwritten in place on every run, so the object is always the day so far.

**Secrets.** Every line goes through :func:`redact` before it leaves the box:
GitHub tokens, AWS access key ids, credentials in URLs and ``password=``-style
values. The units here are not meant to log any of these; this is a backstop.

**Where it writes.** The bucket is ``ALPHA_ENGINE_BOX_LOG_BUCKET`` when set,
otherwise the box's ``signals_bucket`` (``risk.yaml``, the config every other
executor module reads); the prefix is ``ALPHA_ENGINE_BOX_LOG_PREFIX`` (default
:data:`DEFAULT_PREFIX`). It writes with the box's own instance role, read from
the instance metadata service, never a profile: the trader's profile never
reaches this process even if its environment names one.

**It never pages.** A log copy is not worth a page: every failure is a warning
in this unit's journal and the process exits 0.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")
JOURNALCTL = "/usr/bin/journalctl"
BOOT_PULL_LOG = Path("/var/log/boot-pull.log")
DEFAULT_PREFIX = "ops/box-logs"
BUCKET_ENV = "ALPHA_ENGINE_BOX_LOG_BUCKET"
PREFIX_ENV = "ALPHA_ENGINE_BOX_LOG_PREFIX"
#: Each object keeps at most this many bytes, the newest ones.
MAX_BYTES = 2 * 1024 * 1024
JOURNAL_TIMEOUT_SECONDS = 60

#: The units whose journals are copied (``fnmatch`` patterns on systemd unit names).
UNIT_PATTERNS = (
    "alpha-engine-trader-session.service",
    "alpha-engine-trader-reconcile.service",
    "alpha-engine-trader-pin-smoke.service",
    "alpha-engine-trader-shadow-books.service",
    "alpha-engine-trader-failed@*.service",
    "alpha-engine-v2-*.service",
    "boot-pull.service",
    "ibgateway.service",
)

#: (pattern, replacement) pairs applied in order by :func:`redact`.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "github_pat_[REDACTED]"),
    (re.compile(r"\b(gh[pousr]_)[A-Za-z0-9]{20,}"), r"\1[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED-AWS-KEY-ID]"),
    # user:secret@host in a URL (a token in a git remote, say).
    (re.compile(r"(://[^/\s:@]+:)[^@\s/]+@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)(\bauthorization\s*:\s*(?:bearer|token|basic)\s+)\S+"), r"\1[REDACTED]"),
    # key=value / key: value, with an optional quote around the key.
    (
        re.compile(
            r"(?i)((?<![A-Za-z])(?:password|passwd|pwd|secret|secret[_-]?access[_-]?key|session[_-]?token|"
            r"access[_-]?token|api[_-]?key|token)[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]+"
        ),
        r"\1[REDACTED]",
    ),
)


def redact(text: str) -> str:
    """``text`` with every credential-shaped value replaced (module docstring)."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def tail_bytes(data: bytes, limit: int = MAX_BYTES) -> bytes:
    """The last ``limit`` bytes of ``data``, starting on a whole line when cut."""
    if len(data) <= limit:
        return data
    cut = data[-limit:]
    newline = cut.find(b"\n")
    return cut[newline + 1 :] if 0 <= newline < len(cut) - 1 else cut


class PutClient(Protocol):
    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> object: ...  # noqa: N803


Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[bytes]"]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(argv), capture_output=True, timeout=JOURNAL_TIMEOUT_SECONDS, check=False)


def day_start(now: dt.datetime) -> tuple[str, str]:
    """(New York date, its midnight as a UTC ``journalctl --since`` value)."""
    local = now.astimezone(NEW_YORK)
    midnight = dt.datetime.combine(local.date(), dt.time(0, 0), tzinfo=NEW_YORK)
    return local.date().isoformat(), midnight.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def units_to_ship(run: Runner = _run) -> list[str]:
    """The journal's unit names that match :data:`UNIT_PATTERNS`, sorted."""
    result = run([JOURNALCTL, "--no-pager", "-F", "_SYSTEMD_UNIT"])
    if result.returncode != 0:
        raise RuntimeError(f"journalctl -F exited {result.returncode}: {result.stderr.decode(errors='replace')[:200]}")
    names = {line.strip() for line in result.stdout.decode(errors="replace").splitlines()}
    return sorted(n for n in names if n and any(fnmatch.fnmatchcase(n, p) for p in UNIT_PATTERNS))


def journal(unit: str, since: str, run: Runner = _run) -> bytes:
    result = run([JOURNALCTL, "--no-pager", "-o", "short-iso", "-u", unit, "--since", since])
    if result.returncode != 0:
        raise RuntimeError(f"journalctl -u {unit} exited {result.returncode}")
    return result.stdout


def read_file_tail(path: Path, limit: int = MAX_BYTES) -> bytes:
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - limit - 4096))
        return tail_bytes(handle.read(), limit)


@dataclass(frozen=True)
class Target:
    bucket: str
    prefix: str


def resolve_target(env: Mapping[str, str], load_config: Callable[[], Mapping[str, object]]) -> Target | None:
    """Where the logs go (module docstring), or ``None`` when no bucket is known."""
    prefix = (env.get(PREFIX_ENV) or DEFAULT_PREFIX).strip("/")
    bucket = env.get(BUCKET_ENV, "").strip()
    if not bucket:
        try:
            bucket = str(load_config().get("signals_bucket") or "").strip()
        except Exception as exc:  # noqa: BLE001 - a missing config is a warning, not a page
            logger.warning("box log ship: risk.yaml unreadable (%s)", exc)
            bucket = ""
    return Target(bucket, prefix) if bucket else None


class _Frozen(Protocol):
    access_key: str
    secret_key: str
    token: str


class _Credentials(Protocol):
    def get_frozen_credentials(self) -> _Frozen: ...


def instance_role_client(region: str) -> PutClient:
    """An S3 client on the instance role's credentials, never a profile's."""
    import boto3  # pyright: ignore[reportMissingTypeStubs]
    from botocore.credentials import (  # pyright: ignore[reportMissingTypeStubs]
        InstanceMetadataFetcher,
        InstanceMetadataProvider,
    )

    provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=2, num_attempts=3))
    creds = cast("_Credentials | None", provider.load())
    if creds is None:
        raise RuntimeError("no instance role credentials from the instance metadata service")
    frozen = creds.get_frozen_credentials()
    client: object = boto3.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "s3",
        region_name=region,
        aws_access_key_id=frozen.access_key,
        aws_secret_access_key=frozen.secret_key,
        aws_session_token=frozen.token,
    )
    return cast(PutClient, client)


def _default_config() -> Mapping[str, object]:
    from executor import config_loader

    return cast(Mapping[str, object], config_loader.load_config())  # pyright: ignore[reportUnknownMemberType]


def ship(
    *,
    client: PutClient,
    target: Target,
    now: dt.datetime,
    run: Runner = _run,
    boot_pull_log: Path = BOOT_PULL_LOG,
) -> tuple[int, list[str]]:
    """Upload today's logs. Returns (objects written, warnings); never raises on one log."""
    date, since = day_start(now)
    base = f"{target.prefix}/trading/{date}" if target.prefix else f"trading/{date}"
    sources: list[tuple[str, Callable[[], bytes]]] = []
    warnings: list[str] = []
    try:
        for unit in units_to_ship(run):
            sources.append((f"{unit}.log", lambda unit=unit: journal(unit, since, run)))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"unit list: {exc}")
    if boot_pull_log.exists():
        sources.append(("boot-pull.log", lambda: read_file_tail(boot_pull_log)))

    written = 0
    for name, read in sources:
        try:
            data = tail_bytes(read())
            if not data.strip() or data.strip() == b"-- No entries --":
                continue
            body = redact(data.decode("utf-8", errors="replace")).encode("utf-8")
            client.put_object(
                Bucket=target.bucket, Key=f"{base}/{name}", Body=body, ContentType="text/plain; charset=utf-8"
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 - one log failing never stops the others
            warnings.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
    return written, warnings


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[str], PutClient] = instance_role_client,
    load_config: Callable[[], Mapping[str, object]] = _default_config,
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    run: Runner = _run,
    boot_pull_log: Path = BOOT_PULL_LOG,
) -> int:
    """``python -m executor.box_log_ship``. Always exits 0 (module docstring)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        logger.warning("box log ship: takes no arguments; ignoring %r", args)
    environ = os.environ if env is None else env
    try:
        target = resolve_target(environ, load_config)
        if target is None:
            logger.warning("box log ship: no bucket (%s unset and no signals_bucket); nothing copied", BUCKET_ENV)
            return 0
        client = client_factory(environ.get("AWS_REGION", "us-east-1"))
        written, warnings = ship(client=client, target=target, now=clock(), run=run, boot_pull_log=boot_pull_log)
    except Exception as exc:  # noqa: BLE001 - never page over a log copy
        logger.warning("box log ship: not copied: %s: %s", type(exc).__name__, str(exc)[:300])
        return 0
    for warning in warnings:
        logger.warning("box log ship: %s", warning)
    logger.info("box log ship: %d log(s) copied under %s/trading/", written, target.prefix)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    sys.exit(main())
