"""alpha-engine-config-I9739 deliverable 3: the trading box must stop writing
a long-lived GitHub PAT to disk and must instead depend on, and verify,
`git-credential-nousergon-app` (alpha-engine-config-I9628, nous-ergon-ops-PR962).

Root cause this closes, measured 2026-09-01: git sets `CURLOPT_NETRC` to
`CURL_NETRC_OPTIONAL` unconditionally, so libcurl answers GitHub's 401
challenge from the on-disk netrc file BEFORE git ever consults a configured
credential helper. The netrc file does not compete with the helper — it
outranks it. The dashboard box had the helper installed, configured, and
provably able to mint a token, and it was never once consulted, because its
own boot-pull.sh kept rewriting the netrc file on every boot. Every assertion
below is a regex over the script's source text (the same convention every
other guard in test_trading_box_boot_pull.py uses for this hardcoded-path
script) because there is no way to exercise the real
`/usr/local/bin/git-credential-nousergon-app` path or a real
`/home/ec2-user` HOME from a local test run without root.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_NETRC_LITERAL = "/home/ec2-user/" + ".netrc"
_HELPER_PATH = "/usr/local/bin/git-credential-nousergon-app"


def _src() -> str:
    return _BOOT_PULL.read_text()


def test_ssm_netrc_hydration_is_gone():
    """The block that pulled /alpha-engine/GITHUB_TOKEN out of SSM and wrote
    it into the netrc file must be deleted entirely, not disabled behind a
    flag — a box that still WOULD write it on some code path is a box where
    the long-lived secret can still reappear. A historical comment naming the
    old parameter for context is fine; an actual SSM read of it is not."""
    src = _src()
    assert "aws ssm get-parameter --name /alpha-engine/GITHUB_TOKEN" not in src, (
        "boot-pull.sh must no longer read /alpha-engine/GITHUB_TOKEN — that "
        "PAT hydration is the mechanism this issue removes."
    )
    assert "GH_TOKEN" not in src
    assert "NEW_NETRC" not in src


def test_netrc_is_unconditionally_removed_every_run():
    """Every run must `rm -f` the netrc file and log that it did so — a box
    with a leftover file from an earlier boot is a box where the helper is
    installed and never consulted (measured on the dashboard box, I9739)."""
    src = _src()
    assert f'NETRC="{_NETRC_LITERAL}"' in src
    assert 'rm -f "$NETRC"' in src
    # Must be unconditional at the shell level: only gated on the file
    # existing (so `rm -f` doesn't spuriously log every boot), never gated on
    # whether a fresh token could be minted, an SSM read succeeded, or any
    # other condition that could leave a stale file in place.
    removal_idx = src.index('rm -f "$NETRC"')
    guard_window = src[max(0, removal_idx - 200) : removal_idx]
    assert re.search(r'if \[ -e "\$NETRC" \]; then', guard_window), (
        "the removal must be guarded only on file existence, not on any "
        "other condition (e.g. a successful SSM read)."
    )


def test_credential_helper_is_asserted_before_the_repo_loop():
    """The helper's presence and a live --check must be verified BEFORE the
    REPOS pull loop, so a box that cannot authenticate names that as the
    cause rather than surfacing as a generic 403 deep in the git-sync retry
    logic for alpha-engine-config specifically."""
    src = _src()
    assert f'HELPER="{_HELPER_PATH}"' in src

    helper_check_idx = src.index(f'HELPER="{_HELPER_PATH}"')
    # "\nREPOS=(" and not "REPOS=(" — the latter also matches inside
    # "FAILED_REPOS=()", which this script (deliberately) declares earlier
    # than either of these.
    repos_array_idx = src.index("\nREPOS=(")
    assert helper_check_idx < repos_array_idx, (
        "the credential helper assertion must run before the REPOS pull loop "
        "is even defined, so it is unconditionally the first thing boot-pull "
        "checks after removing the netrc file."
    )


def test_credential_helper_missing_is_fail_loud():
    """A box with no helper installed at all must not pass silently — this
    is the FAIL-LOUD condition the issue requires, reusing boot-pull's
    existing PULL_FAILURES/FAILED_REPOS accumulator rather than inventing a
    second failure-reporting mechanism."""
    src = _src()
    assert '[ ! -x "$HELPER" ]' in src
    missing_branch = src[src.index('[ ! -x "$HELPER" ]') :]
    missing_branch = missing_branch[: missing_branch.index("elif")]
    assert 'log "FAIL' in missing_branch
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in missing_branch
    assert "FAILED_REPOS+=(" in missing_branch


def test_credential_helper_check_failure_is_fail_loud():
    """A box with the helper installed but unable to mint a token (e.g. the
    App installation lost read on this repo, or the instance role lost
    ssm:GetParameter) must also increment the same failure accumulator."""
    src = _src()
    call = '_boot_pull_check_credential_helper "$HELPER" alpha-engine-config'
    assert call in src
    check_branch = src[src.index(call) :]
    check_branch = check_branch[: check_branch.index("else")]
    assert 'log "FAIL' in check_branch
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in check_branch
    assert "FAILED_REPOS+=(" in check_branch


def test_pull_failures_accumulator_is_declared_before_the_credential_check():
    """PULL_FAILURES/FAILED_REPOS moved up (from below the REPOS loop) so the
    credential-helper assertion — now the first possible failure in the
    script — can use the SAME counter every later failure uses instead of a
    second, parallel mechanism."""
    src = _src()
    decl_idx = src.index("PULL_FAILURES=0")
    helper_idx = src.index(f'HELPER="{_HELPER_PATH}"')
    assert decl_idx < helper_idx, (
        "PULL_FAILURES=0 / FAILED_REPOS=() must be declared before the "
        "credential helper assertion uses them."
    )
    # Declared exactly once — a stray second declaration further down would
    # silently reset the counter after the credential check already
    # incremented it, erasing that failure before the end-of-script report.
    assert src.count("PULL_FAILURES=0") == 1


def test_no_second_failure_reporting_mechanism_was_invented():
    """The credential assertion must route through log()/PULL_FAILURES/
    FAILED_REPOS — the script's one existing failure-accumulation idiom —
    not a bespoke exit or a separate alert call of its own."""
    src = _src()
    helper_block_start = src.index(f'HELPER="{_HELPER_PATH}"')
    repos_array_idx = src.index("REPOS=(")
    helper_block = src[helper_block_start:repos_array_idx]
    assert "exit 1" not in helper_block, (
        "the credential assertion must not exit the script directly — it "
        "must fall through to the existing end-of-script PULL_FAILURES check "
        "so the rest of boot-pull (systemd sync, trades.db restore) still "
        "runs on a box that can't authenticate."
    )
    assert "krepis.alerts" not in helper_block, (
        "alerting is the end-of-script block's job, driven off "
        "PULL_FAILURES/FAILED_REPOS — the credential check must not publish "
        "its own alert."
    )


# --- the --check retry (alpha-engine-config-I12024) --------------------------
# Measured 2026-11-11: ONE transient mint failure at --check was counted as a
# pull failure and boot-pull exited 1, while the repo loop that followed
# authenticated every repo. These run the real function, sourced with
# AE_BOOT_PULL_LIB_ONLY=1, against a scripted stand-in helper.


def _scripted_helper(tmp_path: Path, outcomes: list[int]) -> Path:
    """A helper whose Nth --check exits outcomes[N-1] (last repeats)."""
    counter = tmp_path / "check_count"
    cases = "\n".join(
        f'[ "$n" {"-eq" if i < len(outcomes) else "-ge"} {i} ] && exit {rc}'
        for i, rc in enumerate(outcomes, start=1)
    )
    helper = tmp_path / "helper.sh"
    helper.write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{counter}"\n'
        f"{cases}\n"
    )
    helper.chmod(0o755)
    return helper


def _check(tmp_path: Path, outcomes: list[int]) -> tuple[int, str, int]:
    helper = _scripted_helper(tmp_path, outcomes)
    log = tmp_path / "boot-pull.log"
    env = {
        **os.environ,
        "AE_BOOT_PULL_LIB_ONLY": "1",
        "AE_BOOT_PULL_LOG": str(log),
        "AE_BOOT_PULL_RETRY_BACKOFF_1": "0",
        "AE_BOOT_PULL_RETRY_BACKOFF_2": "0",
    }
    res = subprocess.run(
        ["bash", "-c", f'. "{_BOOT_PULL}"; _boot_pull_check_credential_helper "{helper}" alpha-engine-config'],
        env=env, capture_output=True, text=True,
    )
    count = int((tmp_path / "check_count").read_text().strip())
    return res.returncode, (log.read_text() if log.exists() else ""), count


def test_check_transient_failure_then_success_is_not_a_failure(tmp_path: Path) -> None:
    rc, log, count = _check(tmp_path, [1, 0])
    assert rc == 0, log
    assert count == 2
    assert "RETRY git credential helper --check alpha-engine-config — attempt 1/3" in log
    assert "recovered on attempt 2/3" in log
    assert "FAIL" not in log


def test_check_failing_three_times_still_fails(tmp_path: Path) -> None:
    rc, log, count = _check(tmp_path, [1])
    assert rc == 1, log
    assert count == 3, "exactly 3 attempts, never more"
    assert "failed on all 3 attempts" in log


def test_check_success_first_time_does_not_retry(tmp_path: Path) -> None:
    rc, log, count = _check(tmp_path, [0])
    assert rc == 0 and count == 1
    assert "RETRY" not in log


def test_check_failure_still_counts_in_the_main_script() -> None:
    """The caller still increments PULL_FAILURES when the function returns 1."""
    src = _src()
    branch = src[src.index('elif ! _boot_pull_check_credential_helper "$HELPER" alpha-engine-config') :]
    branch = branch[: branch.index("elif", 5)]
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in branch
    assert 'FAILED_REPOS+=("git-credential-helper (check failed)")' in branch
