"""boot-pull serializes whole runs on its own lock (alpha-engine-config-I12615).

2026-10-24 00:31:17Z the trading box paged "boot-pull FAILED ...
alpha-engine-data (pip)". The SF started the instance at 00:30:23Z, so
boot-pull.service ran this script at boot. At 00:30:37Z RefreshExecutorDeploy
found the executor checkout drifted and ran it a second time over SSM. The git
half of each run was serialized by GIT_SYNC_LOCK and the per-repo
`pip install -r` was not. With boto3/botocore, edgartools and certifi upgrades
pending, the two pips raced on the same venv.

The structural tests pin where the lock sits and that it is not GIT_SYNC_LOCK.
The behavioural tests run the real block, cut from the script, under bash
against a real flock: it acquires when free, and it records a failure (so the
end-of-run alert pages) instead of hanging when a holder is stuck.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_START = "# ── Whole-run serialization (alpha-engine-config-I12615)"


def _src() -> str:
    return _BOOT_PULL.read_text()


def _lock_block() -> str:
    """The lock block as it appears in boot-pull.sh, through its closing `fi`."""
    src = _src()
    start = src.index(_START)
    end = src.index("\nfi\n", src.index('exec 9>>"$BOOT_PULL_RUN_LOCK"', start)) + len("\nfi\n")
    return src[start:end]


def test_run_lock_sits_after_lib_only_guard_and_before_any_repo_work() -> None:
    src = _src()
    lock_at = src.index('flock -w "$BOOT_PULL_RUN_LOCK_WAIT" 9')
    assert src.index('AE_BOOT_PULL_LIB_ONLY:-0') < lock_at, (
        "the run lock must not be taken when tests source the file as a library"
    )
    assert src.index("FAILED_REPOS=()") < lock_at, (
        "a lock failure must be able to record itself in FAILED_REPOS"
    )
    assert lock_at < src.index('for repo in "${REPOS[@]}"; do'), (
        "the per-repo git+pip loop must run under the run lock"
    )


def test_run_lock_is_not_the_git_sync_lock() -> None:
    src = _src()
    run_lock = 'BOOT_PULL_RUN_LOCK="${AE_BOOT_PULL_RUN_LOCK:-/home/ec2-user/.ae-boot-pull-run.lock}"'
    git_lock = 'GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"'
    assert run_lock in src and git_lock in src
    # A whole-run flock on GIT_SYNC_LOCK would self-deadlock against the
    # per-repo `flock -w ... "$GIT_SYNC_LOCK"` the run takes inside itself.
    assert 'exec 9>>"$BOOT_PULL_RUN_LOCK"' in _lock_block()
    assert 'exec 9>>"$GIT_SYNC_LOCK"' not in src


_needs_flock = pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) not installed")


def _run_block(tmp_path: Path, wait: str) -> tuple[str, str]:
    log = tmp_path / "boot-pull.log"
    script = (
        "set -uo pipefail\n"
        f'LOG="{log}"\n'
        'log() { echo "$*" >> "$LOG"; }\n'
        "PULL_FAILURES=0\nFAILED_REPOS=()\n"
        + _lock_block()
        + 'echo "failures=$PULL_FAILURES repos=${FAILED_REPOS[*]:-}"\n'
    )
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "AE_BOOT_PULL_RUN_LOCK": str(tmp_path / "run.lock"),
        "AE_BOOT_PULL_RUN_LOCK_WAIT": wait,
    }
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip(), log.read_text() if log.exists() else ""


@_needs_flock
def test_free_lock_is_acquired_and_records_no_failure(tmp_path: Path) -> None:
    stdout, log = _run_block(tmp_path, "5")
    assert stdout == "failures=0 repos="
    assert "run lock acquired" in log


@_needs_flock
def test_held_lock_waits_then_records_a_paging_failure(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "20"])
    try:
        # Wait until the holder really has it, so the test measures contention.
        for _ in range(50):
            probe = subprocess.run(["flock", "-n", str(lock), "true"])
            if probe.returncode != 0:
                break
            subprocess.run(["sleep", "0.1"])
        stdout, log = _run_block(tmp_path, "1")
    finally:
        holder.kill()
        holder.wait()
    assert stdout == "failures=1 repos=boot-pull run lock"
    assert "still held after 1s" in log


@_needs_flock
def test_second_run_proceeds_once_the_first_releases(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "1"])
    try:
        stdout, log = _run_block(tmp_path, "10")
    finally:
        holder.wait()
    assert stdout == "failures=0 repos="
    assert "run lock acquired" in log
