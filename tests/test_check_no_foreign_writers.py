"""Unit tests for infrastructure/iam/check-no-foreign-writers.py.

Covers the config#2423 false-positive fix: a deploy script that writes ITS
OWN unrelated role must not be flagged just because a codified role's name
is mentioned elsewhere in the same file (e.g. an operator-facing comment
about a separate, correctly-codified grant). Also covers that a genuine
foreign write (the write's role argument actually resolves to a codified
role name, directly or via one level of variable indirection) is still
caught, and that a write whose role argument can't be statically resolved
fails toward caution rather than being silently dropped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "iam" / "check-no-foreign-writers.py"
)
_spec = importlib.util.spec_from_file_location("check_no_foreign_writers", _SCRIPT_PATH)
cnfw = importlib.util.module_from_spec(_spec)
sys.modules["check_no_foreign_writers"] = cnfw
_spec.loader.exec_module(cnfw)

ROLE_NAMES = {"alpha-engine-step-functions-role", "alpha-engine-dashboard-role"}


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---- the config#2423 false positive itself ----------------------------------


def test_own_role_write_with_unrelated_codified_mention_not_flagged(tmp_path):
    """ssm-liveness-poller/deploy.sh shape: writes its OWN role inline, and
    separately mentions a DIFFERENT codified role in an operator NOTE echo.
    Must NOT be flagged — the write never targets the codified role."""
    content = (
        "#!/usr/bin/env bash\n"
        'ROLE_NAME="alpha-engine-ssm-liveness-poller-role"\n'
        "run aws iam put-role-policy \\\n"
        '    --role-name "${ROLE_NAME}" \\\n'
        '    --policy-name "${POLICY_NAME}" \\\n'
        '    --policy-document "file://iam-policy.json"\n'
        'echo "        NOTE: grant lambda:InvokeFunction to"\n'
        'echo "        alpha-engine-step-functions-role via the codified IAM"\n'
    )
    path = _write(tmp_path, "deploy.sh", content)
    assert cnfw._scan_file(path, ROLE_NAMES) == []


def test_literal_own_role_not_flagged(tmp_path):
    content = (
        "#!/usr/bin/env bash\n"
        "run aws iam put-role-policy --role-name my-own-unrelated-role "
        "--policy-name p --policy-document d\n"
        "# see also alpha-engine-step-functions-role for the SF grant\n"
    )
    path = _write(tmp_path, "deploy.sh", content)
    assert cnfw._scan_file(path, ROLE_NAMES) == []


# ---- genuine violations must still be caught --------------------------------


def test_direct_literal_foreign_write_flagged(tmp_path):
    content = (
        "aws iam put-role-policy --role-name alpha-engine-step-functions-role "
        "--policy-name evil --policy-document d\n"
    )
    path = _write(tmp_path, "deploy.sh", content)
    findings = cnfw._scan_file(path, ROLE_NAMES)
    assert len(findings) == 1
    assert "alpha-engine-step-functions-role" in findings[0]


def test_indirect_foreign_write_via_variable_flagged(tmp_path):
    content = (
        "#!/usr/bin/env bash\n"
        'ROLE_NAME="alpha-engine-step-functions-role"\n'
        "run aws iam put-role-policy \\\n"
        '    --role-name "${ROLE_NAME}" \\\n'
        '    --policy-name "evil" \\\n'
        '    --policy-document "file://x.json"\n'
    )
    path = _write(tmp_path, "deploy.sh", content)
    findings = cnfw._scan_file(path, ROLE_NAMES)
    assert len(findings) == 1
    assert "alpha-engine-step-functions-role" in findings[0]


def test_boto3_and_cfn_forms_still_resolve(tmp_path):
    boto3_path = _write(
        tmp_path, "a.py",
        'client.put_role_policy(RoleName="alpha-engine-step-functions-role", '
        'PolicyName="x", PolicyDocument="{}")\n',
    )
    assert len(cnfw._scan_file(boto3_path, ROLE_NAMES)) == 1

    cfn_path = _write(
        tmp_path, "b.yaml",
        "Resources:\n"
        "  Foo:\n"
        "    Type: AWS::IAM::RolePolicy\n"
        "    Properties:\n"
        "      RoleName: alpha-engine-step-functions-role\n",
    )
    assert len(cnfw._scan_file(cfn_path, ROLE_NAMES)) == 1


# ---- unresolved references fail toward caution, not silence -----------------


def test_unresolved_reference_falls_back_conservatively(tmp_path):
    """No static assignment for the variable used in --role-name — can't be
    proven safe, so (unlike a resolved own-role write) it's still flagged,
    but distinguishably as unresolved."""
    content = (
        "#!/usr/bin/env bash\n"
        "run aws iam put-role-policy \\\n"
        '    --role-name "${SOME_UNKNOWN_ROLE}" \\\n'
        '    --policy-name "x"\n'
        'echo "context: alpha-engine-step-functions-role"\n'
    )
    path = _write(tmp_path, "deploy.sh", content)
    findings = cnfw._scan_file(path, ROLE_NAMES)
    assert len(findings) == 1
    assert "UNRESOLVED" in findings[0]
    assert "alpha-engine-step-functions-role" in findings[0]


# ---- comment lines never count ----------------------------------------------


def test_write_call_in_comment_ignored(tmp_path):
    content = (
        "# example: aws iam put-role-policy --role-name alpha-engine-step-functions-role\n"
        "echo hi\n"
    )
    path = _write(tmp_path, "deploy.sh", content)
    assert cnfw._scan_file(path, ROLE_NAMES) == []


# ---- helper unit coverage ----------------------------------------------------


def test_resolve_role_arg_literal():
    assert cnfw._resolve_role_arg("alpha-engine-step-functions-role", "") == (
        "alpha-engine-step-functions-role"
    )


def test_resolve_role_arg_variable_resolved():
    full_text = 'ROLE_NAME="alpha-engine-step-functions-role"\n'
    assert cnfw._resolve_role_arg("${ROLE_NAME}", full_text) == (
        "alpha-engine-step-functions-role"
    )


def test_resolve_role_arg_variable_unresolved():
    assert cnfw._resolve_role_arg("${UNSET_VAR}", "") is None


def test_resolve_role_arg_double_indirection_unresolved():
    full_text = "ROLE_NAME=${OTHER_VAR}\n"
    assert cnfw._resolve_role_arg("${ROLE_NAME}", full_text) is None


def test_join_continuations_merges_backslash_lines():
    lines = ["a \\", "b \\", "c", "d"]
    statements = cnfw._join_continuations(lines)
    assert statements == [(1, "a \\\nb \\\nc"), (4, "d")]
