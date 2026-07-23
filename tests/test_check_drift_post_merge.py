"""Unit tests for infrastructure/iam/check-drift.py --post-merge (config#3495).

The PR-triggered drift check is structurally circular: a PR that codifies new
IAM is compared against live AWS state that hasn't been applied yet, so it is
guaranteed to show drift until apply.sh runs. --post-merge instead applies
each drifted role via apply.sh and re-checks, only failing on residual
(real) drift. These tests mock both the AWS-calling layer (_aws_iam) and the
apply-invocation layer (_apply_role) so no real AWS/subprocess calls happen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "iam" / "check-drift.py"
)
_spec = importlib.util.spec_from_file_location("check_drift", _SCRIPT_PATH)
check_drift = importlib.util.module_from_spec(_spec)
sys.modules["check_drift"] = check_drift
_spec.loader.exec_module(check_drift)


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_role(tmp_path: Path, name: str, policy_doc: dict) -> Path:
    role_dir = tmp_path / name
    role_dir.mkdir()
    (role_dir / "inline.json").write_text(
        __import__("json").dumps(policy_doc)
    )
    return role_dir


def test_post_merge_resolves_when_apply_fixes_drift(tmp_path, monkeypatch):
    _make_role(tmp_path, "some-role", {"Version": "2012-10-17", "Statement": []})
    monkeypatch.setattr(check_drift, "SCRIPT_DIR", tmp_path)

    # First _check_role call (initial scan): drifted. Second (re-check
    # after apply): clean. list-role-policies then get-role-policy per call.
    live_states = [
        # initial scan: AWS has no such inline policy yet -> "missing in AWS"
        {"PolicyNames": []},
        # re-check after apply: AWS now has it, content matches
        {"PolicyNames": ["inline"]},
        {"PolicyDocument": {"Version": "2012-10-17", "Statement": []}},
    ]

    def fake_aws_iam(*args):
        return live_states.pop(0)

    with patch.object(check_drift, "_aws_iam", side_effect=fake_aws_iam):
        with patch.object(
            check_drift, "_apply_role", return_value=_FakeCompletedProcess(0, "applied\n")
        ) as mock_apply:
            with patch.object(sys, "argv", ["check-drift.py", "--post-merge"]):
                exit_code = check_drift.main()

    assert exit_code == 0
    mock_apply.assert_called_once_with("some-role")


def test_post_merge_fails_on_residual_drift(tmp_path, monkeypatch):
    _make_role(tmp_path, "some-role", {"Version": "2012-10-17", "Statement": []})
    monkeypatch.setattr(check_drift, "SCRIPT_DIR", tmp_path)

    # Both initial scan and re-check show the policy missing on AWS —
    # apply.sh "succeeded" but the drift didn't actually clear (real drift).
    live_states = [
        {"PolicyNames": []},
        {"PolicyNames": []},
    ]

    def fake_aws_iam(*args):
        return live_states.pop(0)

    with patch.object(check_drift, "_aws_iam", side_effect=fake_aws_iam):
        with patch.object(
            check_drift, "_apply_role", return_value=_FakeCompletedProcess(0, "applied\n")
        ):
            with patch.object(sys, "argv", ["check-drift.py", "--post-merge"]):
                exit_code = check_drift.main()

    assert exit_code == 1


def test_post_merge_fails_immediately_on_apply_error(tmp_path, monkeypatch):
    _make_role(tmp_path, "some-role", {"Version": "2012-10-17", "Statement": []})
    monkeypatch.setattr(check_drift, "SCRIPT_DIR", tmp_path)

    with patch.object(check_drift, "_aws_iam", return_value={"PolicyNames": []}):
        with patch.object(
            check_drift,
            "_apply_role",
            return_value=_FakeCompletedProcess(2, "", "aws iam put-role-policy failed\n"),
        ) as mock_apply:
            with patch.object(sys, "argv", ["check-drift.py", "--post-merge"]):
                exit_code = check_drift.main()

    assert exit_code == 1
    mock_apply.assert_called_once_with("some-role")


def test_clean_state_never_invokes_apply(tmp_path, monkeypatch):
    _make_role(tmp_path, "some-role", {"Version": "2012-10-17", "Statement": []})
    monkeypatch.setattr(check_drift, "SCRIPT_DIR", tmp_path)

    def fake_aws_iam(*args):
        if args[0] == "list-role-policies":
            return {"PolicyNames": ["inline"]}
        return {"PolicyDocument": {"Version": "2012-10-17", "Statement": []}}

    with patch.object(check_drift, "_aws_iam", side_effect=fake_aws_iam):
        with patch.object(check_drift, "_apply_role") as mock_apply:
            with patch.object(sys, "argv", ["check-drift.py", "--post-merge"]):
                exit_code = check_drift.main()

    assert exit_code == 0
    mock_apply.assert_not_called()
