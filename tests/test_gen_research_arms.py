"""Tests for scripts/gen_research_arms.py (alpha-engine-config-I11442).

The research slot's servable arms are GENERATED from its arm register and
committed as ``executor/_research_arms.py``, so no S3 read sits on the
trading-start path. These tests pin the derivation, the drift guard CI runs,
and the two failure properties the issue asks for: a read failure neither
halts a run nor admits an unregistered arm.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import gen_research_arms as gen  # noqa: E402


def _registered(name: str, slot: str = "research") -> dict:
    arm_id = f"{slot}:{name}:000000000000"
    return {
        "arm_id": arm_id,
        "date": "2026-09-22",
        "kind": "registered",
        "reason": "",
        "record": {
            "arm_id": arm_id,
            "bootstrap": False,
            "control": False,
            "created_date": "2026-09-22",
            "name": name,
            "notes": "",
            "slot": slot,
            "spec_hash": "000000000000",
            "supersedes": None,
        },
    }


def _retired(name: str) -> dict:
    return {
        "arm_id": f"research:{name}:000000000000",
        "date": "2026-10-20",
        "kind": "retired",
        "reason": "cap-with-grace",
        "record": None,
    }


def _write_register(tmp_path: Path, events: list) -> str:
    path = tmp_path / "register.json"
    path.write_text(json.dumps(events))
    return str(path)


def _write_module(tmp_path: Path, arms: tuple[str, ...]) -> Path:
    path = tmp_path / "_research_arms.py"
    path.write_text(gen.render(arms))
    return path


class TestTheCommittedModule:
    def test_it_is_exactly_what_the_generator_renders(self):
        """A hand edit to the generated file — reordering, a typo'd arm, an extra
        comment — fails here: the file is the generator's output or nothing."""
        arms = gen.committed_arms()
        assert arms, "executor/_research_arms.py declares no arm"
        assert gen.OUTPUT_PATH.read_text() == gen.render(arms)

    def test_it_agrees_with_the_package_import(self):
        from executor._research_arms import RESEARCH_SLOT_ARMS

        assert RESEARCH_SLOT_ARMS == gen.committed_arms()

    def test_it_does_not_trip_the_standalone_literal_guard(self):
        """`test_valid_champions_is_not_a_standalone_literal` greps for a
        `VALID_CHAMPIONS =` assignment; the generated module must not carry one."""
        assert "VALID_CHAMPIONS" not in gen.OUTPUT_PATH.read_text()


class TestArmsFromRegister:
    def test_every_registered_research_arm_sorted(self):
        events = [_registered("b_arm"), _registered("a_arm")]
        assert gen.arms_from_register(events) == ("a_arm", "b_arm")

    def test_a_retired_arm_stays_servable(self):
        """alpha-engine-config-I11436: a retired arm may still be the pointer's
        champion. Retirement is not subtracted."""
        events = [_registered("a_arm"), _registered("b_arm"), _retired("b_arm")]
        assert gen.arms_from_register(events) == ("a_arm", "b_arm")

    def test_another_slots_arm_is_not_admitted(self):
        events = [_registered("a_arm"), _registered("cut_arm", slot="universe_cut")]
        assert gen.arms_from_register(events) == ("a_arm",)

    def test_an_empty_register_is_refused_not_read_as_serve_nothing(self):
        with pytest.raises(gen.RegisterReadError, match="empty served set"):
            gen.arms_from_register([_retired("a_arm")])

    @pytest.mark.parametrize(
        "event",
        [
            "not-a-dict",
            {"arm_id": "x"},
            {"kind": "registered", "arm_id": "x", "record": None},
            {"kind": "registered", "arm_id": "x", "record": {"slot": "research"}},
        ],
    )
    def test_a_malformed_event_is_refused(self, event):
        with pytest.raises(gen.RegisterReadError):
            gen.arms_from_register([_registered("a_arm"), event])


class TestCheck:
    def test_passes_when_every_registered_arm_is_committed(self, tmp_path):
        module = _write_module(tmp_path, ("a_arm", "b_arm"))
        register = _write_register(tmp_path, [_registered("a_arm"), _registered("b_arm")])
        assert gen.main(["--check", "--register", register, "--output", str(module)]) == gen.EXIT_OK

    def test_fails_when_the_register_carries_an_arm_the_executor_cannot_serve(self, tmp_path, capsys):
        module = _write_module(tmp_path, ("a_arm",))
        register = _write_register(tmp_path, [_registered("a_arm"), _registered("new_arm")])
        before = module.read_text()
        assert gen.main(["--check", "--register", register, "--output", str(module)]) == gen.EXIT_DRIFT
        assert "new_arm" in capsys.readouterr().err
        assert module.read_text() == before, "--check must never write"

    def test_a_committed_arm_the_register_dropped_is_a_notice_not_a_failure(self, tmp_path, capsys):
        module = _write_module(tmp_path, ("a_arm", "old_arm"))
        register = _write_register(tmp_path, [_registered("a_arm")])
        assert gen.main(["--check", "--register", register, "--output", str(module)]) == gen.EXIT_OK
        assert "old_arm" in capsys.readouterr().out


class TestRegenerate:
    def test_adds_a_newly_registered_arm(self, tmp_path):
        module = _write_module(tmp_path, ("a_arm",))
        register = _write_register(tmp_path, [_registered("a_arm"), _registered("new_arm")])
        assert gen.main(["--register", register, "--output", str(module)]) == gen.EXIT_OK
        assert gen.committed_arms(module) == ("a_arm", "new_arm")

    def test_never_drops_a_committed_arm(self, tmp_path):
        """The served set only grows on regeneration, so a regenerate can never
        be the change that halts trading on a still-championed arm."""
        module = _write_module(tmp_path, ("a_arm", "old_arm"))
        register = _write_register(tmp_path, [_registered("a_arm")])
        assert gen.main(["--register", register, "--output", str(module)]) == gen.EXIT_OK
        assert gen.committed_arms(module) == ("a_arm", "old_arm")


class TestADeliberateReadFailure:
    """The issue's closes-when: a read failure is shown NOT to halt a run and
    NOT to admit an unregistered arm."""

    @pytest.mark.parametrize("check", [True, False])
    def test_the_generator_writes_nothing_and_exits_nonzero(self, tmp_path, check):
        module = _write_module(tmp_path, ("a_arm",))
        before = module.read_text()
        argv = ["--register", str(tmp_path / "missing.json"), "--output", str(module)]
        assert gen.main(argv + (["--check"] if check else [])) == gen.EXIT_UNREADABLE
        assert module.read_text() == before

    def test_an_unparseable_register_writes_nothing(self, tmp_path):
        module = _write_module(tmp_path, ("a_arm",))
        before = module.read_text()
        bad = tmp_path / "register.json"
        bad.write_text("{not json")
        assert gen.main(["--register", str(bad), "--output", str(module)]) == gen.EXIT_UNREADABLE
        assert module.read_text() == before

    def test_an_s3_read_failure_is_a_register_read_error(self, monkeypatch):
        import boto3

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated S3 outage")

        monkeypatch.setattr(boto3, "client", _boom)
        with pytest.raises(gen.RegisterReadError, match="simulated S3 outage"):
            gen.read_register(gen.LIVE_REGISTER)

    def test_the_executor_serves_its_arms_with_every_network_read_failing(self):
        """In a fresh interpreter with S3 and HTTP both raising, `executor.champion`
        still imports with every research arm servable — the served set is read
        from nowhere at runtime — and an unregistered arm is still refused."""
        script = textwrap.dedent(
            """
            import urllib.request

            import boto3

            def _boom(*_a, **_kw):
                raise RuntimeError("simulated read failure")

            boto3.client = _boom
            boto3.resource = _boom
            urllib.request.urlopen = _boom

            import pytest

            from executor._research_arms import RESEARCH_SLOT_ARMS
            from executor.champion import (
                VALID_CHAMPIONS,
                ChampionPointerError,
                apply_champion_selection,
            )

            assert RESEARCH_SLOT_ARMS
            assert all(arm in VALID_CHAMPIONS for arm in RESEARCH_SLOT_ARMS)
            with pytest.raises(ChampionPointerError, match="not a servable arm"):
                apply_champion_selection(
                    {"buy_candidates": []},
                    {},
                    bucket="unused",
                    run_date="2026-09-25",
                    config={},
                    sector_map=None,
                    pointer={"champion": "an_arm_nobody_registered"},
                )
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("ok")
