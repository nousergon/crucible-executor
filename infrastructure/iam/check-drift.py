#!/usr/bin/env python3
"""check-drift.py — Diff codified IAM against live AWS state.

Walks `infrastructure/iam/<role>/<policy>.json` and compares against live AWS
for each role, across THREE coverage axes:

  1. Inline policies   — every `<role>/<name>.json` (except the reserved
     filenames below) vs `aws iam list-role-policies` + `get-role-policy`.
  2. Trust policy      — optional `<role>/trust-policy.json` vs the role's
     `AssumeRolePolicyDocument` (`aws iam get-role`).
  3. Managed attachments — optional `<role>/managed-policies.json` (a JSON
     array of policy ARNs) vs `aws iam list-attached-role-policies`.

Axes 2 and 3 are OPT-IN per role: if the reserved file is absent, that axis
is skipped for that role (so a role that only codifies inline policies is not
flagged for an uncodified trust policy / managed attachment). Drop the file in
to start enforcing the axis. This lets coverage be adopted role-by-role.

Reserved filenames (NOT treated as inline policies):
  * trust-policy.json      — the role's assume-role (trust) policy document
  * managed-policies.json  — JSON array of attached managed-policy ARNs

Drift cases (all exit non-zero):
  * Inline:   source file with no AWS policy / AWS policy with no source file /
              document content differs.
  * Trust:    codified trust document differs from live AssumeRolePolicyDocument.
  * Managed:  ARN codified but not attached / attached but not codified.

JSON is compared after normalization (sorted keys, no trailing whitespace),
so cosmetic-only differences in indentation or key order don't trip the check.

Usage:
  ./infrastructure/iam/check-drift.py             # check every codified role
  ./infrastructure/iam/check-drift.py --role X    # check one role

Requires AWS creds with iam:ListRolePolicies + iam:GetRolePolicy + iam:GetRole
+ iam:ListAttachedRolePolicies on the target roles. Locally: any admin profile.
In CI: an OIDC role scoped to those four read-only actions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Filenames inside a role dir that are NOT inline policies — each backs a
# distinct coverage axis (trust policy / managed attachments). Excluded from
# the inline-policy scan so they aren't mistaken for inline policy documents.
TRUST_FILENAME = "trust-policy.json"
MANAGED_FILENAME = "managed-policies.json"
RESERVED_STEMS = {Path(TRUST_FILENAME).stem, Path(MANAGED_FILENAME).stem}


def _aws_iam(*args: str) -> dict | list | str:
    """Call aws iam ... and return the parsed JSON output."""
    result = subprocess.run(
        ["aws", "iam", *args, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"AWS CLI failed: aws iam {' '.join(args)}\n"
            f"stderr: {result.stderr}\n"
        )
        sys.exit(2)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _canonical_json(doc: dict) -> str:
    """Canonical JSON for byte-stable comparison: sorted keys, no extra ws."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def _check_trust(role_name: str, role_dir: Path) -> list[str]:
    """Diff codified trust policy vs live (opt-in; skipped if file absent)."""
    trust_path = role_dir / TRUST_FILENAME
    if not trust_path.exists():
        return []
    try:
        source_doc = json.loads(trust_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{role_name}/{TRUST_FILENAME}: source JSON invalid ({exc})"]

    aws_resp = _aws_iam("get-role", "--role-name", role_name)
    aws_doc = aws_resp.get("Role", {}).get("AssumeRolePolicyDocument", {})
    if _canonical_json(source_doc) != _canonical_json(aws_doc):
        return [
            f"{role_name}/{TRUST_FILENAME}: codified trust policy differs from "
            f"live AssumeRolePolicyDocument (content drift)"
        ]
    return []


def _check_managed(role_name: str, role_dir: Path) -> list[str]:
    """Diff codified managed-policy ARNs vs live (opt-in; skipped if absent)."""
    managed_path = role_dir / MANAGED_FILENAME
    if not managed_path.exists():
        return []
    try:
        source_arns = json.loads(managed_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{role_name}/{MANAGED_FILENAME}: source JSON invalid ({exc})"]
    if not isinstance(source_arns, list):
        return [
            f"{role_name}/{MANAGED_FILENAME}: must be a JSON array of policy ARNs"
        ]

    aws_resp = _aws_iam("list-attached-role-policies", "--role-name", role_name)
    aws_arns = {p["PolicyArn"] for p in aws_resp.get("AttachedPolicies", [])}
    source_set = set(source_arns)

    findings: list[str] = []
    for arn in sorted(source_set - aws_arns):
        findings.append(
            f"{role_name}: managed policy {arn} codified but not attached "
            f"(run apply.sh to attach)"
        )
    for arn in sorted(aws_arns - source_set):
        findings.append(
            f"{role_name}: managed policy {arn} attached but not codified "
            f"(add to {MANAGED_FILENAME} or detach from AWS)"
        )
    return findings


def _check_role(role_dir: Path) -> list[str]:
    """Return list of drift findings for a single role. Empty list means clean."""
    role_name = role_dir.name
    findings: list[str] = []

    # ── Set diff ────────────────────────────────────────────────────────────
    all_json = {p.stem for p in role_dir.glob("*.json")}
    if not all_json:
        return [f"{role_name}: no .json files in {role_dir} — empty role dir"]
    source_policies = all_json - RESERVED_STEMS  # inline policies only

    aws_resp = _aws_iam("list-role-policies", "--role-name", role_name)
    aws_policies = set(aws_resp.get("PolicyNames", []))

    extra_in_aws = aws_policies - source_policies
    missing_in_aws = source_policies - aws_policies

    for p in sorted(missing_in_aws):
        findings.append(
            f"{role_name}/{p}: codified in source but not on AWS role "
            f"(run apply.sh to push)"
        )
    for p in sorted(extra_in_aws):
        findings.append(
            f"{role_name}/{p}: present on AWS role but not codified "
            f"(add JSON file or delete from AWS)"
        )

    # ── Content diff for the policies present on both sides ────────────────
    for policy_name in sorted(source_policies & aws_policies):
        source_path = role_dir / f"{policy_name}.json"
        try:
            source_doc = json.loads(source_path.read_text())
        except json.JSONDecodeError as exc:
            findings.append(
                f"{role_name}/{policy_name}: source JSON invalid ({exc})"
            )
            continue

        aws_resp = _aws_iam(
            "get-role-policy",
            "--role-name", role_name,
            "--policy-name", policy_name,
        )
        aws_doc = aws_resp.get("PolicyDocument", {})

        if _canonical_json(source_doc) != _canonical_json(aws_doc):
            findings.append(
                f"{role_name}/{policy_name}: source document differs from "
                f"AWS document (content drift)"
            )

    # ── Trust policy + managed attachments (opt-in per role) ───────────────
    findings.extend(_check_trust(role_name, role_dir))
    findings.extend(_check_managed(role_name, role_dir))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", help="Check one role (default: every codified role)"
    )
    args = parser.parse_args()

    if args.role:
        role_dirs = [SCRIPT_DIR / args.role]
        if not role_dirs[0].is_dir():
            sys.stderr.write(f"ERROR: {role_dirs[0]} is not a directory\n")
            return 2
    else:
        role_dirs = sorted(p for p in SCRIPT_DIR.iterdir() if p.is_dir())

    if not role_dirs:
        print("No codified role directories found under "
              f"{SCRIPT_DIR} — nothing to check.")
        return 0

    total_findings: list[str] = []
    for role_dir in role_dirs:
        findings = _check_role(role_dir)
        total_findings.extend(findings)

    if total_findings:
        print(f"IAM drift detected ({len(total_findings)} finding(s)):")
        for f in total_findings:
            print(f"  - {f}")
        return 1

    role_names = ", ".join(d.name for d in role_dirs)
    print(f"OK: no IAM drift for {role_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
