"""boot-pull retries a transient remote failure, bounded, with a credential
re-mint between attempts (alpha-engine-config-I10950).

Measured on the trading box i-018eb3307a21329bf, 2026-10-02 12:16Z: boot-pull
published "1 repo(s) could not be updated — /home/ec2-user/alpha-engine-config
(git)" after a credential-helper cache MISS minted a token GitHub refused. The
preopen's CodeFreshnessGate absorbed the same condition minutes later only
because nousergon-data-PR1776 gave its git_retry() a transient-remote class.
The dashboard box's boot-pull got the same class in crucible-dashboard-PR869.
This box's boot-pull had only the one-shot ref-CAS retry, which re-serves the
same token and fails identically.

These tests run the real sync_repo_to_main(), sourced with
AE_BOOT_PULL_LIB_ONLY=1, against a real local bare remote. A PATH-shadowing
`git` scripts each `fetch` call. Every other git call passes straight through.
This is the same idiom as test_boot_pull_post_condition.py. A regex over the
source cannot see the retry loop's control flow, and the defect lives there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}

_REFUSED = ("remote: Repository not found.\n"
            "fatal: repository 'https://github.com/nousergon/alpha-engine-config.git/' not found")
_CAS = "error: cannot lock ref 'refs/remotes/origin/main'"


def _git(*args: str, cwd: Path) -> str:
    res = subprocess.run(
        ["git", *args], cwd=cwd, env={**os.environ, **_GIT_ENV},
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def box(tmp_path: Path):
    """A checkout one commit behind a real local bare remote."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git("init", "--bare", "--initial-branch=main", ".", cwd=remote)

    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "--initial-branch=main", ".", cwd=seed)
    (seed / "f.txt").write_text("v1\n")
    _git("add", "f.txt", cwd=seed)
    _git("commit", "-m", "v1", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    checkout = tmp_path / "alpha-engine-config"
    _git("clone", str(remote), str(checkout), cwd=tmp_path)
    old_sha = _git("rev-parse", "HEAD", cwd=checkout)

    (seed / "f.txt").write_text("v2\n")
    _git("commit", "-am", "v2", cwd=seed)
    _git("push", "origin", "main", cwd=seed)
    new_sha = _git("rev-parse", "HEAD", cwd=seed)
    return {"checkout": checkout, "old_sha": old_sha, "new_sha": new_sha,
            "tmp": tmp_path}


def _shim_dir(tmp_path: Path, name: str) -> Path:
    """PATH-prepended dir, with a `flock` stand-in where flock(1) is absent
    (macOS). Same shape as test_boot_pull_post_condition.py's helper."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    if shutil.which("flock") is None:
        (d / "flock").write_text(
            "#!/bin/sh\n"
            'while [ "$1" = "-w" ]; do shift 2; done\n'
            "shift\n"
            'exec "$@"\n'
        )
        (d / "flock").chmod(0o755)
    return d


def _install_git_shim(tmp_path: Path, *, name: str, fetch_outcomes: list[str]) -> Path:
    """A PATH-shadowing `git` whose Nth `fetch` follows fetch_outcomes[N-1].
    The last entry repeats once the list is exhausted. "ok" forwards to the
    real git. Anything else is printed to stderr and the fetch exits 1.
    """
    real_git = shutil.which("git")
    assert real_git, "git must be on PATH"
    shim_dir = _shim_dir(tmp_path, name)
    counter = shim_dir / "fetch_count"
    branches = []
    for i, outcome in enumerate(fetch_outcomes, start=1):
        cond = (f'[ "$n" -eq {i} ]' if i < len(fetch_outcomes)
                else f'[ "$n" -ge {i} ]')
        if outcome == "ok":
            branches.append(f'if {cond}; then exec "{real_git}" "$@"; fi')
        else:
            msg = outcome.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            branches.append(f'if {cond}; then printf "%b\\n" "{msg}" >&2; exit 1; fi')
    body = "\n    ".join(branches)
    (shim_dir / "git").write_text(
        "#!/bin/sh\n"
        f'COUNT_FILE="{counter}"\n'
        'case " $* " in\n'
        '*" fetch "*)\n'
        '    n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)\n'
        '    n=$((n + 1))\n'
        '    echo "$n" > "$COUNT_FILE"\n'
        f"    {body}\n"
        "    ;;\n"
        "esac\n"
        f'exec "{real_git}" "$@"\n'
    )
    (shim_dir / "git").chmod(0o755)
    return shim_dir


def _fetch_count(shim: Path) -> int:
    f = shim / "fetch_count"
    return int(f.read_text().strip()) if f.exists() else 0


def _fake_cred_helper(tmp_path: Path) -> tuple[str, Path]:
    """A stand-in git-credential-nousergon-app recording each `erase` stdin."""
    calls = tmp_path / "erase_calls.txt"
    helper = tmp_path / "fake-cred-helper.sh"
    helper.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "erase" ]; then cat >> "{calls}"; echo "---" >> "{calls}"; fi\n'
        "exit 0\n"
    )
    helper.chmod(0o755)
    return str(helper), calls


def _run_lib(call: str, tmp_path: Path, *, shim: Path | None = None,
             cred_helper: str | None = None):
    log = tmp_path / "boot-pull.log"
    env = {
        **os.environ, **_GIT_ENV,
        "AE_BOOT_PULL_LIB_ONLY": "1",
        "AE_BOOT_PULL_LOG": str(log),
        "AE_GIT_SYNC_LOCK": str(tmp_path / "sync.lock"),
        "AE_GIT_SYNC_LOCK_WAIT": "10",
        "AE_BOOT_PULL_RETRY_BACKOFF_1": "0",
        "AE_BOOT_PULL_RETRY_BACKOFF_2": "0",
        "AE_CRED_HELPER": cred_helper or str(tmp_path / "no-such-helper"),
    }
    if shim is not None:
        env["PATH"] = f"{shim}{os.pathsep}{os.environ['PATH']}"
    res = subprocess.run(["bash", "-c", f'. "{_BOOT_PULL}"; {call}'],
                         env=env, capture_output=True, text=True)
    return res.returncode, (log.read_text() if log.exists() else ""), res.stderr


def _sync(box, *, shim: Path, cred_helper: str | None = None):
    return _run_lib(f'sync_repo_to_main "{box["checkout"]}"', box["tmp"],
                    shim=shim, cred_helper=cred_helper)


class TestTransientClassRetries:
    def test_refused_token_recovers_on_a_later_attempt(self, box):
        """THE 2026-10-02 SHAPE. The first attempt's fetch and its CAS retry
        are both refused. Without this class that is a FAIL. With it, the
        cached credential is erased and the next attempt succeeds."""
        shim = _install_git_shim(box["tmp"], name="shim-refused",
                                 fetch_outcomes=[_REFUSED, _REFUSED, "ok"])
        helper, calls = _fake_cred_helper(box["tmp"])

        rc, log, err = _sync(box, shim=shim, cred_helper=helper)

        assert rc == 0, f"a transient refusal must recover; log:\n{log}\n{err}"
        assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["new_sha"]
        assert _fetch_count(shim) == 3
        assert "RETRY" in log and "attempt 1/3" in log
        assert "'Repository not found'" in log
        assert "recovered on attempt 2/3" in log
        assert calls.exists(), "the cached credential must be erased before the retry"

    def test_exhausted_transient_retries_still_fail_loud(self, box):
        """A remote that never comes back still FAILS after exactly 3 attempts,
        and the stale-but-consistent checkout is still not reported as OK."""
        shim = _install_git_shim(box["tmp"], name="shim-dead",
                                 fetch_outcomes=[_REFUSED])
        rc, log, err = _sync(box, shim=shim)

        assert rc == 1, f"log:\n{log}\n{err}"
        assert _fetch_count(shim) == 6, "3 attempts x (fetch + its CAS retry), never more"
        assert "exhausted retrying transient class" in log
        assert "cannot be shown to be current" in log
        assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["old_sha"]

    def test_missing_helper_does_not_block_the_retry(self, box):
        shim = _install_git_shim(box["tmp"], name="shim-nohelper",
                                 fetch_outcomes=["fatal: RPC failed; curl 56"] * 2 + ["ok"])
        rc, log, err = _sync(box, shim=shim)  # AE_CRED_HELPER points nowhere

        assert rc == 0, f"log:\n{log}\n{err}"
        assert "SKIP credential erase" in log
        assert _fetch_count(shim) == 3


class TestOtherFailuresKeepTheirBehaviour:
    def test_unmatched_fetch_failure_is_not_retried_by_this_class(self, box):
        """A fetch failure with no transient signature fails after the existing
        CAS retry exactly as before: 2 fetches, no erase, no RETRY line."""
        shim = _install_git_shim(box["tmp"], name="shim-unmatched",
                                 fetch_outcomes=[_CAS, "fatal: bad object refs/heads/main"])
        helper, calls = _fake_cred_helper(box["tmp"])

        rc, log, err = _sync(box, shim=shim, cred_helper=helper)

        assert rc == 1, f"log:\n{log}\n{err}"
        assert _fetch_count(shim) == 2
        assert "RETRY" not in log
        assert not calls.exists()

    def test_cas_race_still_heals_inside_one_attempt(self, box):
        """The existing ref-CAS class is untouched: first fetch fails, the
        inner retry succeeds, and the outer loop never runs a second time."""
        shim = _install_git_shim(box["tmp"], name="shim-cas",
                                 fetch_outcomes=[_CAS, "ok"])
        rc, log, err = _sync(box, shim=shim)

        assert rc == 0, f"log:\n{log}\n{err}"
        assert _fetch_count(shim) == 2
        assert "RETRY" not in log
        assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["new_sha"]


class TestCredentialEraseKeysOnRemoteSlug:
    def test_erase_derives_slug_from_origin_not_dirname(self, tmp_path):
        repo = tmp_path / "alpha-engine-data"
        repo.mkdir()
        _git("init", "--initial-branch=main", ".", cwd=repo)
        _git("remote", "add", "origin",
             "https://github.com/nousergon/nousergon-data.git", cwd=repo)
        helper, calls = _fake_cred_helper(tmp_path)

        rc, log, err = _run_lib(f'_boot_pull_erase_credential "{repo}"',
                                tmp_path, cred_helper=helper)

        assert rc == 0, err
        recorded = calls.read_text()
        assert "path=nousergon/nousergon-data\n" in recorded
        assert "alpha-engine-data" not in recorded

    def test_unresolvable_remote_is_skipped_not_failed(self, tmp_path):
        repo = tmp_path / "noorigin"
        repo.mkdir()
        _git("init", "--initial-branch=main", ".", cwd=repo)
        helper, calls = _fake_cred_helper(tmp_path)

        rc, log, err = _run_lib(f'_boot_pull_erase_credential "{repo}"',
                                tmp_path, cred_helper=helper)

        assert rc == 0, err
        assert "skipping credential erase" in log
        assert not calls.exists()
