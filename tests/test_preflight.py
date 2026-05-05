"""
Tests for ExecutorPreflight mode composition.

BasePreflight primitives are tested in alpha-engine-lib. These tests
verify that each executor mode composes the expected primitive calls
and rejects unknown modes.

Data-freshness checks (universe + macro/SPY) moved upstream to
alpha-engine-data's preflight 2026-05-05; the data step in every Step
Function hard-fails on staleness before the executor runs, so re-checking
here is redundant.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.preflight import ExecutorPreflight


class TestExecutorPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ExecutorPreflight(bucket="b", mode="bogus")

    def test_main_mode_composes_check_sequence(self):
        """main mode: env + S3. Data-freshness moved upstream."""
        pf = ExecutorPreflight(bucket="b", mode="main")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()

    def test_daemon_mode_composes_check_sequence(self):
        """daemon mode: same as main."""
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()

    def test_eod_mode_composes_check_sequence(self):
        """eod mode: env + S3 only."""
        pf = ExecutorPreflight(bucket="b", mode="eod")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()

    def test_no_mode_calls_data_freshness_primitives(self):
        """Regression: no executor mode may call macro or universe
        freshness checks. Those moved to alpha-engine-data's preflight,
        which is the SF data step's responsibility."""
        for mode in ("main", "daemon", "eod"):
            pf = ExecutorPreflight(bucket="b", mode=mode)
            with patch.object(pf, "check_env_vars"), \
                 patch.object(pf, "check_s3_bucket"), \
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
