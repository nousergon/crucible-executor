"""alpha-engine-config-I9466 — the executor's §2.3a consumer gate.

A guard that has not been VERIFIED TO FAIL is not a guard. The core of this
file removes or corrupts the attestation and asserts a BLOCK, one degenerate
input at a time — absent, unreadable, non-JSON, wrong shape, a truthy
non-"PASS" verdict, and a verdict bound to a different weekly cycle.

The second half is the other obligation, and it is the one that is easy to
skip: asserting what the gate must NEVER block. §2.3a rule 4's amendment
record exists because over-broad withholding read as conservative and was
destructive — the actions swept in were the repair actions. Here the
equivalent is an EXIT. Blocking one does not withhold a guarantee, it strips
the risk management off a book sized by exactly the parameters in doubt.
"""

from __future__ import annotations

import io
import json

import pytest
from botocore.exceptions import ClientError

from executor import correctness_verdict as cv

BUCKET = "alpha-engine-research"
_RUN_DATE = "2026-08-28"


class _FakeS3:
    """Mirrors tests/test_derisk_gate.py's fake — ``objects`` maps key ->
    (bytes | Exception). A key that is absent raises NoSuchKey, as S3 does."""

    def __init__(self, objects: dict):
        self.objects = objects

    def get_object(self, Bucket=None, Key=None):  # noqa: N803 — boto3 casing
        val = self.objects.get(Key)
        if val is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject",
            )
        if isinstance(val, Exception):
            raise val
        return {"Body": io.BytesIO(val)}


def _attestation(verdict="PASS", run_date=_RUN_DATE, **extra) -> bytes:
    return json.dumps({
        "schema": "backtest_attestation-1.0.0",
        "component": "backtester",
        "run_date": run_date,
        "trading_day": run_date,
        "status": "ok",
        "verdict": verdict,
        "n_checks": 9,
        "n_failed": 0,
        **extra,
    }).encode()


def _params(updated_at=_RUN_DATE) -> bytes:
    return json.dumps({
        "min_score": 75,
        "max_position_pct": 0.1,
        "atr_multiplier": 2.0,
        "updated_at": updated_at,
    }).encode()


def _clean() -> dict:
    return {cv.ATTESTATION_KEY: _attestation(),
            cv.EXECUTOR_PARAMS_KEY: _params()}


def _evaluate(objects, mode=cv.MODE_ENFORCE):
    return cv.evaluate_correctness_verdict(
        BUCKET, {"correctness_verdict_gate_mode": mode}, s3_client=_FakeS3(objects),
    )


# ════════════════════════════════════════════════════════════════════════════
# The gate PASSES only on the one input that earns it
# ════════════════════════════════════════════════════════════════════════════

def test_a_provenance_matched_PASS_is_the_only_thing_that_proceeds():
    """This is the live shape as of 2026-08-31: attestation verdict PASS for
    run_date 2026-08-28, executor_params updated_at 2026-08-28."""
    state = _evaluate(_clean())
    assert state.verdict == cv.PASS
    assert state.is_pass
    assert state.blocks_new_entries is False


# ════════════════════════════════════════════════════════════════════════════
# REMOVE OR BLANK THE ARTIFACT AND ASSERT A BLOCK — the guard's own proof
# ════════════════════════════════════════════════════════════════════════════

def test_an_ABSENT_attestation_blocks_and_is_UNKNOWN():
    """The defining case. `sf-pipeline-policy` §2.3a rule 2 — a missing verdict
    propagates as UNKNOWN, never as pass, and never as 'an older artifact,
    proceed'."""
    objects = _clean()
    del objects[cv.ATTESTATION_KEY]
    state = _evaluate(objects)
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True
    assert "ABSENT" in state.reason


def test_a_BLANK_attestation_body_blocks():
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: b""})
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


def test_a_NON_JSON_attestation_body_blocks():
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: b"{not json"})
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


def test_a_JSON_ARRAY_body_blocks():
    """A body that parses but is not an object — `.get` would raise, and a
    reader that catches that and returns a default is the bug."""
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: b"[]"})
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


def test_an_UNREADABLE_attestation_blocks():
    boom = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: boom})
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True
    assert "AccessDenied" in state.reason


def test_a_FAIL_verdict_blocks_and_stays_DISTINCT_from_unknown():
    state = _evaluate({**_clean(),
                       cv.ATTESTATION_KEY: _attestation(verdict="FAIL", n_failed=2)})
    assert state.verdict == cv.FAIL
    assert state.verdict != cv.UNKNOWN
    assert state.blocks_new_entries is True


# ════════════════════════════════════════════════════════════════════════════
# Only the LITERAL "PASS" grants the guarantee
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw", ["ok", "pass", "Pass", "PASSED", True, 1, None, {}])
def test_nothing_but_the_literal_PASS_is_a_pass(raw):
    """The truthiness read is the defect this module exists to make
    unwritable. `"ok"` is the backtester artifact's OWN `status` field value —
    a reader that grabbed the wrong key would sail straight through."""
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: _attestation(verdict=raw)})
    assert state.verdict == cv.UNKNOWN
    assert state.is_pass is False
    assert state.blocks_new_entries is True


def test_a_missing_verdict_KEY_blocks():
    body = json.loads(_attestation())
    del body["verdict"]
    state = _evaluate({**_clean(), cv.ATTESTATION_KEY: json.dumps(body).encode()})
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


# ════════════════════════════════════════════════════════════════════════════
# The binding is PROVENANCE, not recency
# ════════════════════════════════════════════════════════════════════════════

def test_a_verdict_from_ANOTHER_cycle_is_never_inherited():
    """A PASS for last week's numbers says nothing about the parameters in the
    file the executor just read."""
    state = _evaluate({**_clean(),
                       cv.ATTESTATION_KEY: _attestation(run_date="2026-08-21")})
    assert state.verdict == cv.STALE_BINDING
    assert state.blocks_new_entries is True
    assert "2026-08-21" in state.reason and "2026-08-28" in state.reason


def test_stale_binding_is_distinct_from_both_fail_and_unknown():
    state = _evaluate({**_clean(),
                       cv.ATTESTATION_KEY: _attestation(run_date="2026-08-21")})
    assert state.verdict not in (cv.FAIL, cv.UNKNOWN)


def test_absent_params_make_a_PASS_say_nothing():
    """The verdict PASSes, but the artifact it attests is not in play, so it
    grants nothing about the parameters actually used this session."""
    objects = _clean()
    del objects[cv.EXECUTOR_PARAMS_KEY]
    state = _evaluate(objects)
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


# ── alpha-engine-config-I12440: latest/ moved on, the params did not ─────────
# Live shape 2026-10-19/20: latest/ attests 2026-10-16 (PASS), the params are
# 2026-10-09's because the apply loop declined to promote, and
# backtest/2026-10-09/attestation.json PASSed for run_date 2026-10-09.

_PARAMS_CYCLE = "2026-10-09"
_NEWER_CYCLE = "2026-10-16"
_DATED_KEY = f"backtest/{_PARAMS_CYCLE}/attestation.json"


def _apply_loop_held_back(**dated) -> dict:
    objects = {cv.ATTESTATION_KEY: _attestation(run_date=_NEWER_CYCLE),
               cv.EXECUTOR_PARAMS_KEY: _params(updated_at=_PARAMS_CYCLE)}
    objects[_DATED_KEY] = _attestation(**{"run_date": _PARAMS_CYCLE, **dated})
    return objects


def test_params_held_back_by_the_apply_loop_bind_to_THEIR_OWN_cycle():
    """The I12440 false positive. The params' own cycle wrote a PASS for
    exactly these parameters; latest/ moving on to an unpromoted cycle does
    not make them unattested."""
    state = _evaluate(_apply_loop_held_back())
    assert state.verdict == cv.PASS
    assert state.blocks_new_entries is False
    assert state.attested_run_date == _PARAMS_CYCLE == state.params_updated_at
    assert state.context["attestation_key"] == _DATED_KEY
    # The newer, unpromoted cycle stays visible rather than vanishing.
    assert state.context["latest_attestation"]["run_date"] == _NEWER_CYCLE
    assert _NEWER_CYCLE in state.reason
    assert _DATED_KEY in cv.operator_message(state)


def test_params_rolled_back_after_a_FAIL_bind_to_the_PASSing_cycle():
    """The FAIL remedy is 'roll executor_params back to the last cycle whose
    attestation PASSed'. Reading only latest/ made that remedy itself a
    permanent STALE_BINDING."""
    objects = _apply_loop_held_back()
    objects[cv.ATTESTATION_KEY] = _attestation(
        verdict="FAIL", run_date=_NEWER_CYCLE, n_failed=2)
    state = _evaluate(objects)
    assert state.verdict == cv.PASS
    assert state.context["latest_attestation"]["verdict"] == "FAIL"


def test_held_back_params_whose_own_cycle_has_NO_attestation_stay_STALE():
    """The gate must still fire: latest/ is another cycle's PASS and the
    params' cycle wrote nothing. Never inherited."""
    objects = _apply_loop_held_back()
    del objects[_DATED_KEY]
    state = _evaluate(objects)
    assert state.verdict == cv.STALE_BINDING
    assert state.blocks_new_entries is True
    assert "ABSENT" in state.reason and _NEWER_CYCLE in state.reason


def test_held_back_params_whose_own_cycle_FAILED_are_FAIL():
    state = _evaluate(_apply_loop_held_back(verdict="FAIL", n_failed=1))
    assert state.verdict == cv.FAIL
    assert state.blocks_new_entries is True
    assert state.attested_run_date == _PARAMS_CYCLE


@pytest.mark.parametrize("raw", ["ok", "pass", True, None])
def test_held_back_params_own_cycle_needs_the_literal_PASS(raw):
    state = _evaluate(_apply_loop_held_back(verdict=raw))
    assert state.verdict == cv.UNKNOWN
    assert state.blocks_new_entries is True


def test_a_dated_object_naming_ANOTHER_run_date_is_STALE():
    """The dated key is a lookup, not a grant: its body must itself attest the
    params' cycle."""
    state = _evaluate(_apply_loop_held_back(run_date="2026-10-02"))
    assert state.verdict == cv.STALE_BINDING
    assert state.blocks_new_entries is True
    assert "2026-10-02" in state.reason


def test_a_NEWER_PASS_never_covers_older_params_via_the_dated_key():
    """Only backtest/{updated_at}/ is consulted — a PASS stored under the
    newer cycle's dated key grants nothing to the older params."""
    objects = _apply_loop_held_back()
    del objects[_DATED_KEY]
    objects[f"backtest/{_NEWER_CYCLE}/attestation.json"] = _attestation(
        run_date=_NEWER_CYCLE)
    state = _evaluate(objects)
    assert state.verdict == cv.STALE_BINDING


@pytest.mark.parametrize("updated_at", [None, "", "../latest", "2026-10-09/x", 20261009])
def test_a_params_updated_at_that_is_not_a_cycle_date_is_STALE(updated_at):
    """updated_at is templated into an S3 key; anything but a bare date is no
    cycle to bind to, and is never used to build a key."""
    objects = _apply_loop_held_back()
    objects[cv.EXECUTOR_PARAMS_KEY] = _params(updated_at=updated_at)
    state = _evaluate(objects)
    assert state.verdict == cv.STALE_BINDING
    assert state.blocks_new_entries is True


def test_held_back_params_observe_mode_reports_PASS_without_blocking():
    state = _evaluate(_apply_loop_held_back(), mode=cv.MODE_OBSERVE)
    assert state.verdict == cv.PASS and state.blocks_new_entries is False


def test_a_daily_run_between_weekly_cycles_is_NOT_a_false_positive():
    """The predicate that makes this a guard rather than an outage on a timer.
    executor_params is written weekly and the preopen runs daily; requiring the
    attestation to be stamped TODAY would fire on four days in five."""
    for _weekday in range(5):
        state = _evaluate(_clean())
        assert state.blocks_new_entries is False


# ════════════════════════════════════════════════════════════════════════════
# §7a — OBSERVE MODE NEVER BLOCKS, AND IS STILL LOUD
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("objects", [
    {},
    {cv.EXECUTOR_PARAMS_KEY: _params()},
    {cv.ATTESTATION_KEY: _attestation(verdict="FAIL"), cv.EXECUTOR_PARAMS_KEY: _params()},
    {cv.ATTESTATION_KEY: _attestation(run_date="2026-01-01"),
     cv.EXECUTOR_PARAMS_KEY: _params()},
])
def test_observe_mode_never_blocks_whatever_the_verdict(objects):
    """§7a: a guard newly added to a scheduled path may not take its halt
    branch on its first production run. The system is trading live paper."""
    state = _evaluate(objects, mode=cv.MODE_OBSERVE)
    assert state.blocks_new_entries is False
    assert state.verdict != cv.PASS


def test_observe_mode_is_the_default():
    state = cv.evaluate_correctness_verdict(BUCKET, {}, s3_client=_FakeS3({}))
    assert state.mode == cv.MODE_OBSERVE
    assert state.blocks_new_entries is False


def test_observe_mode_still_reports_the_verdict_honestly():
    """Not blocking is not the same as not knowing."""
    state = _evaluate({}, mode=cv.MODE_OBSERVE)
    assert state.verdict == cv.UNKNOWN
    assert state.is_pass is False


def test_an_unrecognised_mode_is_read_as_ENFORCE_not_observe():
    """A typo must never disarm a guard. Reading an unknown mode as `observe`
    would make `correctness_verdict_gate_mode: enfroce` a silent, permanent
    disarm that looks exactly like a deliberate staging decision."""
    state = cv.evaluate_correctness_verdict(
        BUCKET, {"correctness_verdict_gate_mode": "enfroce"},
        s3_client=_FakeS3({}),
    )
    assert state.mode == cv.MODE_ENFORCE
    assert state.blocks_new_entries is True


def test_the_promotion_criterion_is_declared_in_this_module():
    """§7a: the promotion criterion lives in the guard's own module, not in a
    plan doc the guard cannot read."""
    assert "enforce" in cv.PROMOTION_CRITERION
    assert "risk.yaml" in cv.PROMOTION_CRITERION


# ════════════════════════════════════════════════════════════════════════════
# §2.3a rule 4 — WHAT THE GATE MUST NEVER WITHHOLD
# ════════════════════════════════════════════════════════════════════════════

def test_exactly_one_action_is_withheld_and_it_is_opening_a_position():
    assert set(cv.WITHHELD_ACTIONS) == {"open_new_position"}


def test_every_exit_path_is_declared_UNGATED():
    """The load-bearing carve-out. Blocking an exit does not withhold a
    guarantee, it withholds risk management."""
    for action in ("place_exit_order", "cancel_order"):
        assert action in cv.UNGATED_ACTIONS
        assert action not in cv.WITHHELD_ACTIONS


def test_reads_and_own_ledger_writes_are_declared_UNGATED():
    """§2.3a rule 4's test: for each gated action, name the system OUTSIDE this
    consumer whose durable state it would change. A read has none, so it must
    not be gated."""
    for action in ("read_signals_predictions_prices", "write_order_book",
                   "write_decision_capture", "log_risk_event"):
        assert action in cv.UNGATED_ACTIONS


def test_both_sides_are_emitted_in_both_verdict_states():
    """§2.3a rule 4's second obligation. A withheld list that appears only when
    something stopped cannot be told apart from a producer that stopped."""
    blocked = _evaluate({}).to_log_dict()
    clean = _evaluate(_clean()).to_log_dict()
    assert blocked["withheld_actions"] == ["open_new_position"]
    assert blocked["ungated_actions"] == sorted(cv.UNGATED_ACTIONS)
    assert clean["withheld_actions"] == []
    assert clean["ungated_actions"] == sorted(cv.UNGATED_ACTIONS), (
        "the ran-regardless set must be emitted on a clean cycle too"
    )


# ════════════════════════════════════════════════════════════════════════════
# §2.3a rule 1 — the DEPENDENCY is declared, not inferred
# ════════════════════════════════════════════════════════════════════════════

def test_the_backtester_attestation_is_declared_GATING():
    row = cv.UPSTREAM_VERDICTS[cv.ATTESTATION_KEY]
    assert row["gating"] is True
    assert row["attests"] == cv.EXECUTOR_PARAMS_KEY


def test_the_director_is_declared_CONDITIONAL_with_its_predicate():
    """Not 'advisory' full stop. executor/derisk_gate.py composes a SIZING
    MULTIPLIER out of director/carryover_ledger.json, so the Director becomes a
    gating upstream the moment `derisk_on_expectancy_enabled` is set true.
    Rule 1 is about the dependency, not about what happens to be read today."""
    row = cv.UPSTREAM_VERDICTS["director/latest/action_plan.json"]
    assert row["gating"] is False
    assert row["predicate"] == "risk.yaml derisk_on_expectancy_enabled is true"

    from executor import derisk_gate
    assert derisk_gate.DERISK_ON_EXPECTANCY_ENABLED_DEFAULT is False, (
        "the predicate above is only correct while the flag defaults False — "
        "if this ever flips, the Director row becomes gating and this module "
        "must gate on it"
    )
    assert derisk_gate.CARRYOVER_LEDGER_KEY == "director/carryover_ledger.json"


# ════════════════════════════════════════════════════════════════════════════
# The operator message names the cause AND the unblocking step
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("objects,verdict", [
    ({}, cv.UNKNOWN),
    ({cv.ATTESTATION_KEY: _attestation(verdict="FAIL"),
      cv.EXECUTOR_PARAMS_KEY: _params()}, cv.FAIL),
    ({cv.ATTESTATION_KEY: _attestation(run_date="2026-08-21"),
      cv.EXECUTOR_PARAMS_KEY: _params()}, cv.STALE_BINDING),
])
def test_every_blocking_verdict_carries_an_unblocking_step(objects, verdict):
    state = _evaluate(objects)
    assert state.verdict == verdict
    msg = cv.operator_message(state)
    assert "UNBLOCK:" in msg
    assert verdict in msg


def test_the_message_says_what_is_STILL_RUNNING():
    """The operator must not read a blocked entry as an unmanaged book."""
    msg = cv.operator_message(_evaluate({}))
    assert "STILL RUNNING" in msg
    for word in ("exits", "stops", "urgent exits", "order book"):
        assert word in msg


def test_the_observe_message_says_nothing_is_blocked():
    msg = cv.operator_message(_evaluate({}, mode=cv.MODE_OBSERVE))
    assert "OBSERVE MODE" in msg
    assert "NEW ENTRIES ARE BLOCKED" not in msg


# ════════════════════════════════════════════════════════════════════════════
# The CLI the SF state runs
# ════════════════════════════════════════════════════════════════════════════

def _cli(monkeypatch, objects, mode):
    monkeypatch.setattr(cv, "boto3", type("_B", (), {
        "client": staticmethod(lambda *a, **k: _FakeS3(objects))})())
    return cv.main(["--bucket", BUCKET, "--mode", mode])


def test_cli_returns_zero_on_a_clean_pass(monkeypatch):
    assert _cli(monkeypatch, _clean(), cv.MODE_ENFORCE) == cv.EXIT_OK


def test_cli_returns_zero_in_observe_mode_however_bad_the_verdict(monkeypatch):
    assert _cli(monkeypatch, {}, cv.MODE_OBSERVE) == cv.EXIT_OK


@pytest.mark.parametrize("objects,code", [
    ({}, cv.EXIT_UNKNOWN),
    ({cv.ATTESTATION_KEY: _attestation(verdict="FAIL"),
      cv.EXECUTOR_PARAMS_KEY: _params()}, cv.EXIT_FAIL),
    ({cv.ATTESTATION_KEY: _attestation(run_date="2026-08-21"),
      cv.EXECUTOR_PARAMS_KEY: _params()}, cv.EXIT_STALE_BINDING),
])
def test_cli_exit_codes_keep_the_three_causes_distinct(monkeypatch, objects, code):
    """The SF reads these off the SSM ResponseCode so FAIL, UNKNOWN and
    STALE_BINDING reach the operator as three different terminal states."""
    assert _cli(monkeypatch, objects, cv.MODE_ENFORCE) == code


def test_cli_prints_the_machine_readable_line(monkeypatch, capsys):
    _cli(monkeypatch, _clean(), cv.MODE_OBSERVE)
    out = capsys.readouterr().out
    assert "CORRECTNESS_VERDICT_GATE verdict=PASS" in out
    assert "blocks_new_entries=false" in out


def test_cli_is_loud_in_observe_mode(monkeypatch, capsys):
    """§7a: loud while observing. A silent observe mode is indistinguishable
    from a guard that was never wired up."""
    _cli(monkeypatch, {}, cv.MODE_OBSERVE)
    out = capsys.readouterr().out
    assert "CORRECTNESS VERDICT UNKNOWN" in out
    assert "UNBLOCK:" in out
