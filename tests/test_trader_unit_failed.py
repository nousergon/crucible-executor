"""`executor.trader_unit_failed`: the OnFailure page for a v2 trader unit.

alpha-engine-config-I12353: on 2026-10-14/15 four trader unit runs ended
``ActiveState=failed Result=exit-code`` before writing any record, and
nothing paged. These tests pin what the page says and that it is sent once,
through the executor's existing ops path, without touching the trader.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from executor import trader_unit_failed as tuf

UNIT = "alpha-engine-trader-shadow-books.service"
NOW = dt.datetime(2026, 10, 15, 15, 1, tzinfo=dt.UTC)

SHOW = (
    "ActiveState=failed\n"
    "Result=exit-code\n"
    "ExecMainStatus=1\n"
    "ExecMainExitTimestamp=Thu 2026-10-15 11:00:41 EDT\n"
)
TAIL = "\n".join(f"2026-10-15T11:00:{i:02d}-0400 box trader[1]: line {i}" for i in range(50))


class _Box:
    """A fake `run`: answers systemctl/journalctl, records every argv."""

    def __init__(self, *, journal: str = TAIL, journal_rc: int = 0, sudo_journal: str = "") -> None:
        self.calls: list[list[str]] = []
        self.journal = journal
        self.journal_rc = journal_rc
        self.sudo_journal = sudo_journal

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[0] == "systemctl":
            return subprocess.CompletedProcess(args, 0, SHOW, "")
        if args[0] == "sudo":
            return subprocess.CompletedProcess(args, 0, self.sudo_journal, "")
        return subprocess.CompletedProcess(args, self.journal_rc, self.journal, "access denied")


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def _publish(message: str, **kwargs: Any) -> None:
        sent.append({"message": message, **kwargs})

    monkeypatch.setattr(tuf, "publish_ops_alert", _publish)
    return sent


def test_a_failed_unit_pages_once_with_its_state_and_journal_tail(
    published: list[dict[str, Any]],
) -> None:
    assert tuf.main([UNIT], run=_Box(), now=NOW) == 0
    (alert,) = published
    assert alert["severity"] == "error"
    assert alert["source"] == "alpha-engine/executor/trader_unit_failed.py"
    assert alert["dedup_key"] == f"trader-unit-failed:{UNIT}:2026-10-15"
    message = alert["message"]
    assert UNIT in message
    assert "ActiveState=failed Result=exit-code ExecMainStatus=1" in message
    assert "line 0" in message and "line 49" in message
    assert "Auto-fix:" in message and "Resolution:" in message


def test_it_only_reads_the_unit_and_never_acts_on_it(published: list[dict[str, Any]]) -> None:
    box = _Box()
    tuf.main([UNIT], run=box, now=NOW)
    verbs = {(c[0], c[1]) for c in box.calls}
    assert verbs <= {("systemctl", "show"), ("journalctl", "--unit")}


def test_an_unreadable_journal_falls_back_to_sudo(published: list[dict[str, Any]]) -> None:
    box = _Box(journal="", journal_rc=1, sudo_journal="via sudo: the venv build failed")
    tuf.main([UNIT], run=box, now=NOW)
    assert box.calls[-1][:2] == ["sudo", "-n"]
    assert "via sudo: the venv build failed" in published[0]["message"]


def test_a_journal_nobody_can_read_still_pages_and_says_so(
    published: list[dict[str, Any]],
) -> None:
    tuf.main([UNIT], run=_Box(journal="", journal_rc=1), now=NOW)
    assert "journal tail unreadable" in published[0]["message"]


def test_a_long_tail_keeps_the_lines_nearest_the_failure(published: list[dict[str, Any]]) -> None:
    long_tail = "x" * 10_000 + "\nTraceback: the last line"
    tuf.main([UNIT], run=_Box(journal=long_tail), now=NOW)
    message = published[0]["message"]
    assert "Traceback: the last line" in message
    assert len(message) < 4096


def test_the_v2_trade_alert_unit_is_accepted() -> None:
    """alpha-engine-config-I13284: the alert names this notifier in OnFailure=."""
    assert tuf.UNIT_RE.fullmatch("alpha-engine-v2-trade-alert.service")
    assert not tuf.UNIT_RE.fullmatch("alpha-engine-v2-trade-alert.service; rm")


@pytest.mark.parametrize(
    "argv",
    [[], ["alpha-engine-daemon.service"], [UNIT, UNIT], ["alpha-engine-trader-x.service; rm"]],
)
def test_anything_but_one_trader_unit_is_refused_without_paging(
    argv: list[str], published: list[dict[str, Any]]
) -> None:
    box = _Box()
    assert tuf.main(argv, run=box, now=NOW) == 2
    assert published == [] and box.calls == []
