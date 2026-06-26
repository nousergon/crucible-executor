"""
Tests for ExecutorPreflight mode composition + deploy-drift check.

BasePreflight primitives are tested in alpha-engine-lib. These tests
verify that each executor mode composes the expected primitive calls,
rejects unknown modes, and that the executor's git-checkout
``check_deploy_drift`` override behaves like the predictor preflight
(pass on match, fail-loud on drift) while hard-failing the executor-
specific missing-``.git`` case (issue config#892).

Data-freshness checks (universe + macro/SPY) moved upstream to
alpha-engine-data's preflight 2026-05-05; the data step in every Step
Function hard-fails on staleness before the executor runs, so re-checking
here is redundant.

The GitHub-fetch helper (``_fetch_origin_main_sha``) is owned by
alpha-engine-lib and tested there; this module re-imports it via the
``executor.preflight`` namespace so ``patch.object(pf_mod,
"_fetch_origin_main_sha", ...)`` mocks the same symbol production calls.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import executor.preflight as pf_mod
from executor.preflight import ExecutorPreflight


class TestExecutorPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ExecutorPreflight(bucket="b", mode="bogus")

    def test_main_mode_composes_check_sequence(self):
        """main mode: env + S3 + deploy-drift. Data-freshness moved upstream."""
        pf = ExecutorPreflight(bucket="b", mode="main")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_deploy_drift") as drift:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        drift.assert_called_once()

    def test_daemon_mode_composes_check_sequence(self):
        """daemon mode: same as main."""
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_deploy_drift") as drift:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        drift.assert_called_once()

    def test_eod_mode_composes_check_sequence(self):
        """eod mode: env + S3 + deploy-drift."""
        pf = ExecutorPreflight(bucket="b", mode="eod")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_deploy_drift") as drift:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        drift.assert_called_once()

    def test_no_mode_calls_data_freshness_primitives(self):
        """Regression: no executor mode may call macro or universe
        freshness checks. Those moved to alpha-engine-data's preflight,
        which is the SF data step's responsibility."""
        for mode in ("main", "daemon", "eod"):
            pf = ExecutorPreflight(bucket="b", mode=mode)
            with patch.object(pf, "check_env_vars"), \
                 patch.object(pf, "check_s3_bucket"), \
                 patch.object(pf, "check_deploy_drift"), \
                 patch.object(pf, "check_arcticdb_fresh") as fresh, \
                 patch.object(pf, "check_arcticdb_universe_fresh") as universe:
                pf.run()
            fresh.assert_not_called()
            universe.assert_not_called()

    def test_check_ib_paper_account_available_on_instance(self):
        """Daemon reuses the preflight instance to validate the IB
        account ID after IBKRClient connects. The primitive is
        inherited from BasePreflight — smoke-test the chain."""
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        pf.check_ib_paper_account("DU1234567")  # paper — no raise

        with pytest.raises(RuntimeError, match="not a paper"):
            pf.check_ib_paper_account("U1234567")  # live — must raise


# ── check_deploy_drift (executor git-checkout variant) ───────────────────────

def _make_git_checkout(tmp_path: Path) -> Path:
    """Init a real throwaway git repo with one commit; return its root."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.io"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


class TestCheckDeployDrift:
    def _head_sha(self, root: Path) -> str:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    def test_matching_sha_passes(self, tmp_path):
        """Deployed HEAD == upstream HEAD → no raise (the happy path)."""
        root = _make_git_checkout(tmp_path)
        head = self._head_sha(root)
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf_mod, "_fetch_origin_main_sha", return_value=head):
            pf.check_deploy_drift(repo_root=root)  # must not raise

    def test_drift_fails_loud(self, tmp_path):
        """Deployed HEAD != upstream HEAD → hard-fail RuntimeError.

        This is the whole point: boot-pull is behind origin/main and the
        daemon must refuse to run stale code on fresh signals.
        """
        root = _make_git_checkout(tmp_path)
        upstream = "deadbeef" * 5  # 40-char SHA that is NOT the local HEAD
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf_mod, "_fetch_origin_main_sha", return_value=upstream):
            with pytest.raises(RuntimeError, match="Deploy drift"):
                pf.check_deploy_drift(repo_root=root)

    def test_github_outage_is_warn_and_continue(self, tmp_path):
        """GitHub unreachable (helper returns None) → no raise.

        Mirrors the predictor / lib posture: an outage must not block a
        trading-hours daemon. Can't prove drift → don't block.
        """
        root = _make_git_checkout(tmp_path)
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf_mod, "_fetch_origin_main_sha", return_value=None):
            pf.check_deploy_drift(repo_root=root)  # must not raise

    def test_missing_git_dir_hard_fails(self, tmp_path):
        """No .git directory → hard-fail (executor-specific, issue#892).

        Unlike the predictor's missing-stamp WARN path (a legacy image
        legitimately predates stamping), a missing .git on the executor
        box means the checkout is gone — boot-pull never ran. There is
        no legitimate first-boot case; the box is provisioned by git
        clone. The upstream fetch must not even be reached.
        """
        empty = tmp_path / "not-a-checkout"
        empty.mkdir()
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf_mod, "_fetch_origin_main_sha") as fetch:
            with pytest.raises(RuntimeError, match="no .git directory"):
                pf.check_deploy_drift(repo_root=empty)
        fetch.assert_not_called()

    def test_git_rev_parse_failure_hard_fails(self, tmp_path):
        """`.git` exists but rev-parse fails (corrupt repo) → hard-fail."""
        broken = tmp_path / "broken"
        (broken / ".git").mkdir(parents=True)  # .git present but not a real repo
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf_mod, "_fetch_origin_main_sha") as fetch:
            with pytest.raises(RuntimeError, match="rev-parse HEAD. failed"):
                pf.check_deploy_drift(repo_root=broken)
        fetch.assert_not_called()

    def test_default_repo_is_current_repo(self):
        """The default repo target is the migrated public repo, not the
        pre-migration monorepo path."""
        assert pf_mod._EXECUTOR_REPO == "nousergon/crucible-executor"
