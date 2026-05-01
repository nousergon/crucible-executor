"""
Tests for ExecutorPreflight mode composition.

BasePreflight primitives are tested in alpha-engine-lib. These tests
verify that each executor mode composes the expected primitive calls
and rejects unknown modes.
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

    def test_main_mode_composes_full_check_sequence(self):
        """main mode: env + S3 + macro/SPY freshness + universe/SPY
        freshness + per-ticker universe scan."""
        pf = ExecutorPreflight(bucket="b", mode="main")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as fresh, \
             patch.object(pf, "check_arcticdb_universe_fresh") as universe:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        # macro/SPY + universe/SPY
        assert fresh.call_count == 2
        macro_call, universe_sym_call = fresh.call_args_list
        assert macro_call.args[:2] == ("macro", "SPY")
        assert universe_sym_call.args[:2] == ("universe", "SPY")
        universe.assert_called_once()
        assert universe.call_args.args[0] == "universe"

    def test_daemon_mode_composes_full_check_sequence(self):
        """daemon mode mirrors main — same ArcticDB liveness gates."""
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as fresh, \
             patch.object(pf, "check_arcticdb_universe_fresh") as universe:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        assert fresh.call_count == 2
        universe.assert_called_once()

    def test_eod_mode_runs_macro_only(self):
        """eod reads macro/SPY for alpha + per-position closes; full
        universe scan is overkill since only the ~20 held names matter."""
        pf = ExecutorPreflight(bucket="b", mode="eod")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as fresh, \
             patch.object(pf, "check_arcticdb_universe_fresh") as universe:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        # Only macro/SPY — no universe-side checks
        fresh.assert_called_once()
        assert fresh.call_args.args[:2] == ("macro", "SPY")
        universe.assert_not_called()

    def test_check_ib_paper_account_available_on_instance(self):
        """Daemon reuses the preflight instance to validate the IB
        account ID after IBKRClient connects. The primitive is
        inherited from BasePreflight — smoke-test the chain."""
        pf = ExecutorPreflight(bucket="b", mode="daemon")
        pf.check_ib_paper_account("DU1234567")  # paper — no raise

        with pytest.raises(RuntimeError, match="not a paper"):
            pf.check_ib_paper_account("U1234567")  # live — must raise
