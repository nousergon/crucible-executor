"""Regression tests for infrastructure/iam/check-no-foreign-writers.py.

config#2493: the checker false-positived on
nousergon-data/infrastructure/lambdas/ssm-liveness-poller/deploy.sh, which
only *echoes* an operator reminder mentioning the codified role
`alpha-engine-step-functions-role` — it never writes that role. The
checker's file-scope role-name match didn't distinguish an echo/print
message from an actual write-API argument.
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "iam" / "check-no-foreign-writers.py"
)
_spec = importlib.util.spec_from_file_location("check_no_foreign_writers", _MODULE_PATH)
check_no_foreign_writers = importlib.util.module_from_spec(_spec)
sys.modules["check_no_foreign_writers"] = check_no_foreign_writers
_spec.loader.exec_module(check_no_foreign_writers)

_scan_file = check_no_foreign_writers._scan_file

ROLE_NAMES = {"alpha-engine-step-functions-role"}


def test_echo_mentioning_codified_role_is_not_a_finding(tmp_path):
    """config#2493 regression: an operator-facing echo naming the codified
    role must not be flagged, since it can never influence which role the
    script's own put-role-policy call (against a *different*, non-codified
    role) actually targets."""
    deploy_sh = tmp_path / "deploy.sh"
    deploy_sh.write_text(
        "\n".join(
            [
                "ROLE_NAME=my-own-lambda-role",
                "run aws iam put-role-policy \\",
                '  --role-name "${ROLE_NAME}" \\',
                '  --policy-name my-policy',
                'echo "  NOTE: grant lambda:InvokeFunction on this function to"',
                'echo "        alpha-engine-step-functions-role via the codified IAM"',
            ]
        )
    )

    assert _scan_file(deploy_sh, ROLE_NAMES) == []


def test_real_write_to_codified_role_is_still_a_finding(tmp_path):
    """A genuine inline write naming the codified role directly must still
    be caught — the echo carve-out must not blind the checker to real
    foreign writes."""
    deploy_sh = tmp_path / "deploy.sh"
    deploy_sh.write_text(
        "\n".join(
            [
                "ROLE_NAME=alpha-engine-step-functions-role",
                "run aws iam put-role-policy \\",
                '  --role-name "${ROLE_NAME}" \\',
                '  --policy-name my-policy',
            ]
        )
    )

    findings = _scan_file(deploy_sh, ROLE_NAMES)
    assert len(findings) == 1
    assert "alpha-engine-step-functions-role" in findings[0]


def test_commented_out_write_is_not_a_finding(tmp_path):
    deploy_sh = tmp_path / "deploy.sh"
    deploy_sh.write_text(
        "\n".join(
            [
                "# aws iam put-role-policy --role-name alpha-engine-step-functions-role",
            ]
        )
    )

    assert _scan_file(deploy_sh, ROLE_NAMES) == []
