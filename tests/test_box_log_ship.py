"""The trading box's log copy to S3 (alpha-engine-config-I12024): executor.box_log_ship."""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from executor import box_log_ship as m

#: Wednesday 2026-11-11, 20:38 ET (EST) = 01:38 UTC the next day.
EVENING = dt.datetime(2026, 11, 12, 1, 38, tzinfo=dt.UTC)


class _Client:
    def __init__(self, fail_on: str = "") -> None:
        self.puts: dict[str, bytes] = {}
        self.buckets: set[str] = set()
        self.fail_on = fail_on

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> object:  # noqa: N803
        if self.fail_on and Key.endswith(self.fail_on):
            raise RuntimeError("AccessDenied")
        assert ContentType.startswith("text/plain")
        self.buckets.add(Bucket)
        self.puts[Key] = Body
        return {}


class _Journal:
    def __init__(self, logs: Mapping[str, bytes], *, list_fails: bool = False) -> None:
        self.logs = dict(logs)
        self.list_fails = list_fails
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        argv = list(argv)
        self.calls.append(argv)
        if "-F" in argv:
            if self.list_fails:
                return subprocess.CompletedProcess(argv, 1, b"", b"permission denied")
            names = [*self.logs, "sshd.service", "amazon-ssm-agent.service", ""]
            return subprocess.CompletedProcess(argv, 0, "\n".join(names).encode(), b"")
        unit = argv[argv.index("-u") + 1]
        return subprocess.CompletedProcess(argv, 0, self.logs[unit], b"")


# Fixture credentials are assembled at runtime, so no credential-shaped literal
# sits in the source for the repository's secret scan to flag.
_AKID = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
_ASID = "AS" + "IA" + "QRSTUVWXYZ234567"
_SECRET = "-".join(["fixture", "not", "a", "real", "secret", "value"])


# --- redaction --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("remote: https://x-access-token:ghs_" + "a" * 36 + "@github.com/o/r", "ghs_" + "a" * 36),
        ("token ghp_" + "B" * 36 + " used", "B" * 36),
        ("GH_TOKEN=github_pat_" + "c" * 40, "c" * 40),
        (f"creds {_AKID} found", _AKID),
        (f"session {_ASID}", _ASID),
        ("login password=hunter2 ok", "hunter2"),
        ('{"Password": "s3cr3t!"}', "s3cr3t!"),
        (f"aws_secret_access_key = {_SECRET}", _SECRET),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", "eyJhbGciOi.payload.sig"),
        ("git clone https://user:pa55word@example.com/repo.git", "pa55word"),
        ("IBC TWS_PASSWORD: trademe", "trademe"),
    ],
)
def test_redact_removes_the_secret(line: str, secret: str) -> None:
    out = m.redact(line)
    assert secret not in out
    assert "REDACTED" in out


@pytest.mark.parametrize(
    "line",
    [
        "2026-11-11T09:30:01-0500 box python[123]: session started, 12 orders adopted",
        "boot-pull: OK   systemd: updated alpha-engine-v2-day-end.timer",
        "HEAD is now at 21b8d3c fix(trader): refuse a session start",
    ],
)
def test_redact_leaves_ordinary_lines_alone(line: str) -> None:
    assert m.redact(line) == line


# --- helpers ----------------------------------------------------------------------


def test_tail_keeps_the_newest_bytes_from_a_whole_line() -> None:
    data = b"".join(f"line {i}\n".encode() for i in range(1000))
    out = m.tail_bytes(data, 100)
    assert len(out) <= 100 and out.endswith(b"line 999\n")
    assert out.startswith(b"line ")
    assert m.tail_bytes(b"short\n", 100) == b"short\n"


def test_read_file_tail(tmp_path: Path) -> None:
    path = tmp_path / "boot-pull.log"
    path.write_bytes(b"".join(f"row {i}\n".encode() for i in range(10000)))
    out = m.read_file_tail(path, 200)
    assert len(out) <= 200 and out.endswith(b"row 9999\n") and out.startswith(b"row ")


def test_the_day_is_new_yorks_not_utcs() -> None:
    date, since = m.day_start(EVENING)
    assert date == "2026-11-11"
    assert since == "2026-11-11 05:00:00 UTC"
    summer = dt.datetime(2026, 7, 1, 14, 0, tzinfo=dt.UTC)
    assert m.day_start(summer) == ("2026-07-01", "2026-07-01 04:00:00 UTC")


def test_resolve_target_prefers_the_env_then_the_config() -> None:
    def config() -> Mapping[str, object]:
        return {"signals_bucket": "cfg-bucket"}

    assert m.resolve_target({}, config) == m.Target("cfg-bucket", m.DEFAULT_PREFIX)
    env = {m.BUCKET_ENV: "env-bucket", m.PREFIX_ENV: "/custom/prefix/"}
    assert m.resolve_target(env, config) == m.Target("env-bucket", "custom/prefix")


def test_resolve_target_without_a_bucket_is_none() -> None:
    def broken() -> Mapping[str, object]:
        raise FileNotFoundError("risk.yaml")

    assert m.resolve_target({}, broken) is None
    assert m.resolve_target({}, dict) is None


# --- ship -------------------------------------------------------------------------

LOGS = {
    "alpha-engine-trader-session.service": b"2026-11-11T20:38:00-0500 started\n",
    "alpha-engine-v2-session-watchdog.service": b"watch ok token=ghs_" + b"z" * 36 + b"\n",
    "alpha-engine-trader-failed@alpha-engine-trader-session.service.service": b"paged\n",
    "boot-pull.service": b"pulled\n",
    "ibgateway.service": b"-- No entries --\n",
}


def test_ship_writes_each_matching_unit_and_the_boot_pull_file(tmp_path: Path) -> None:
    boot_log = tmp_path / "boot-pull.log"
    boot_log.write_text("FAIL alpha-engine pull: https://x-access-token:ghs_" + "q" * 36 + "@github.com\n")
    client = _Client()
    journal = _Journal(LOGS)
    written, warnings = m.ship(
        client=client, target=m.Target("b", "ops/box-logs"), now=EVENING, run=journal, boot_pull_log=boot_log
    )
    base = "ops/box-logs/trading/2026-11-11"
    assert warnings == []
    assert set(client.puts) == {
        f"{base}/alpha-engine-trader-session.service.log",
        f"{base}/alpha-engine-v2-session-watchdog.service.log",
        f"{base}/alpha-engine-trader-failed@alpha-engine-trader-session.service.service.log",
        f"{base}/boot-pull.service.log",
        f"{base}/boot-pull.log",
    }
    assert written == 5
    # Unrelated units are never copied; an empty journal writes nothing.
    assert not any("sshd" in k or "ibgateway" in k for k in client.puts)
    # Redacted on the way out.
    assert b"ghs_z" not in client.puts[f"{base}/alpha-engine-v2-session-watchdog.service.log"]
    assert b"ghs_q" not in client.puts[f"{base}/boot-pull.log"]
    # The journal is read from New York midnight.
    session_call = next(c for c in journal.calls if "alpha-engine-trader-session.service" in c)
    assert session_call[-2:] == ["--since", "2026-11-11 05:00:00 UTC"]
    assert "short-iso" in session_call


def test_one_failed_upload_does_not_stop_the_others(tmp_path: Path) -> None:
    client = _Client(fail_on="boot-pull.service.log")
    written, warnings = m.ship(
        client=client, target=m.Target("b", "p"), now=EVENING, run=_Journal(LOGS), boot_pull_log=tmp_path / "none"
    )
    assert written == 3
    assert len(warnings) == 1 and "AccessDenied" in warnings[0]


def test_an_unreadable_journal_still_copies_the_boot_pull_file(tmp_path: Path) -> None:
    boot_log = tmp_path / "boot-pull.log"
    boot_log.write_text("ok\n")
    client = _Client()
    written, warnings = m.ship(
        client=client, target=m.Target("b", "p"), now=EVENING, run=_Journal({}, list_fails=True), boot_pull_log=boot_log
    )
    assert written == 1 and list(client.puts) == ["p/trading/2026-11-11/boot-pull.log"]
    assert warnings and "unit list" in warnings[0]


# --- main never pages -------------------------------------------------------------


def _main(**kwargs: object) -> int:
    defaults: dict[str, object] = {
        "argv": [],
        "env": {m.BUCKET_ENV: "bkt"},
        "client_factory": lambda region: _Client(),
        "load_config": dict,
        "clock": lambda: EVENING,
        "run": _Journal(LOGS),
        "boot_pull_log": Path("/nonexistent/boot-pull.log"),
    }
    defaults.update(kwargs)
    return m.main(**defaults)  # pyright: ignore[reportArgumentType]


def test_main_uploads_and_exits_zero() -> None:
    client = _Client()
    assert _main(client_factory=lambda region: client) == 0
    assert client.buckets == {"bkt"} and len(client.puts) == 4


def test_main_exits_zero_when_credentials_are_missing(caplog: pytest.LogCaptureFixture) -> None:
    def no_role(region: str) -> m.PutClient:
        raise RuntimeError("no instance role credentials")

    assert _main(client_factory=no_role) == 0
    assert "not copied" in caplog.text


def test_main_exits_zero_without_a_bucket(caplog: pytest.LogCaptureFixture) -> None:
    assert _main(env={}) == 0
    assert "no bucket" in caplog.text


def test_main_exits_zero_on_a_failed_upload(caplog: pytest.LogCaptureFixture) -> None:
    assert _main(client_factory=lambda region: _Client(fail_on=".log")) == 0
    assert "AccessDenied" in caplog.text


def test_the_client_never_uses_a_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trader's profile in the environment must not reach the uploader."""
    import botocore.credentials

    seen: dict[str, object] = {}

    class _Provider:
        def __init__(self, iam_role_fetcher: object) -> None:
            seen["fetcher"] = iam_role_fetcher

        def load(self) -> None:
            return None

    monkeypatch.setenv("AWS_PROFILE", "crucible-v2-trader")
    monkeypatch.setattr(botocore.credentials, "InstanceMetadataProvider", _Provider)
    with pytest.raises(RuntimeError, match="instance role"):
        m.instance_role_client("us-east-1")
    assert isinstance(seen["fetcher"], botocore.credentials.InstanceMetadataFetcher)
