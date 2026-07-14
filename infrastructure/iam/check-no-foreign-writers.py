#!/usr/bin/env python3
"""check-no-foreign-writers.py — Ensure codified IAM roles have exactly
one writer.

A "codified" role is any directory under `infrastructure/iam/<role>/`.
For each such role, scan a configurable set of repository checkouts for
`aws iam put-role-policy --role-name <role>` (or its boto3/yaml/json
equivalents). The codified policy + `apply.sh` is the only sanctioned
writer; any other reference is a regression risk and fails the check.

Why: the alpha-engine system has hit four IAM-clobber incidents in two
months, all rooted in a second writer racing the codified policy
(EB-SFN role 2026-04-21 + 2026-05-04 + 2026-05-06; SF role 2026-05-04
EOD + 2026-05-06 morning). PR #136 closed the EB-SFN twin; PR #151
closed one of two SF-role twins; this check catches the next one
before it merges.

Scope:
  - Files scanned: bash deploy scripts (`*.sh`) + CloudFormation YAML
    + python scripts under `infrastructure/`. Skips `apply.sh` (which
    legitimately writes the codified state) and `check-drift.py` (which
    only reads).
  - Repos scanned: passed via --repo (defaults to the parent of this
    repo's directory). One invocation can cover the alpha-engine
    sibling layout we use locally + in CI.

Usage:
  ./infrastructure/iam/check-no-foreign-writers.py
  ./infrastructure/iam/check-no-foreign-writers.py --repo ~/Development/alpha-engine-data
  ./infrastructure/iam/check-no-foreign-writers.py --repo ~/Development/alpha-engine --repo ~/Development/alpha-engine-data
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Files that are allowed to write codified roles.
ALLOWED_WRITERS = {"apply.sh"}

# File extensions to scan for writes.
SCAN_EXTENSIONS = {".sh", ".yaml", ".yml", ".py", ".tf"}

# Skip these directories anywhere in the path.
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def _codified_roles_in_repo(iam_dir: Path) -> set[str]:
    """Return the set of role names codified under one repo's iam dir.

    Handles both layouts in use across alpha-engine repos:
      - directory-per-role: alpha-engine/infrastructure/iam/<role>/<policy>.json
      - flat one-file-per-role: alpha-engine-data/infrastructure/iam/<role>.json
    """
    if not iam_dir.is_dir():
        return set()

    roles: set[str] = set()
    for entry in iam_dir.iterdir():
        if entry.is_dir():
            # directory-per-role layout
            roles.add(entry.name)
        elif entry.is_file() and entry.suffix == ".json":
            # flat one-file-per-role layout
            roles.add(entry.stem)
    return roles


def _all_codified_roles(repo_roots: list[Path]) -> dict[str, Path]:
    """Map role-name → home-repo across every scanned repo's iam dir."""
    home: dict[str, Path] = {}
    for repo in repo_roots:
        iam_dir = repo / "infrastructure" / "iam"
        for role in _codified_roles_in_repo(iam_dir):
            home[role] = repo
    return home


def _scan_file(path: Path, role_names: set[str]) -> list[str]:
    """Return list of findings for this file.

    Logic: if the file contains BOTH (a) a write-API call (put-role-policy
    or equivalent) and (b) a literal mention of a codified role name —
    in non-comment, non-display lines — flag it. File-scope rather than
    line-window: deploy scripts often declare `ROLE_NAME=...` at the top
    and call put-role-policy with a `$ROLE_NAME` reference far below.
    """
    findings: list[str] = []
    try:
        text = path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return findings

    write_patterns = [
        r"put-role-policy",
        r"attach-role-policy",
        r"delete-role-policy",
        r"put_role_policy",
        r"attach_role_policy",
        r"AWS::IAM::RolePolicy\b",
        r"aws_iam_role_policy\b",
    ]
    write_re = re.compile("|".join(write_patterns))

    # Skip pure comment lines (sh, py, yaml: `#`; tf, js: `//`).
    comment_re = re.compile(r"^\s*(#|//)")

    # Skip pure display/log statements: a role name (or the words
    # "put-role-policy" etc.) inside an `echo`/`print`/`logging.*` message
    # is operator-facing prose, not code that can influence which role an
    # API call targets — e.g. a deploy script's "NOTE: grant X to role Y
    # via apply.sh" reminder. Only actual identifier usage (a write-API
    # call's own arguments, or a variable assignment that could feed one)
    # should count as a role "mention". Without this, config#2493 showed
    # `check-no-foreign-writers.py` itself producing false positives on
    # deploy scripts that merely print the codified role's name in an
    # operator instruction.
    display_re = re.compile(
        r"^\s*(echo|printf)\b"  # bash
        r"|^\s*print\s*\("  # python
        r"|^\s*(logging|logger)\.(debug|info|warning|error|critical)\s*\("  # python logging
        r"|^\s*console\.(log|warn|error|info)\s*\("  # js/ts
    )

    write_lines: list[tuple[int, str]] = []
    code_lines: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if comment_re.match(line):
            continue
        if display_re.match(line):
            continue
        code_lines.append(line)
        if write_re.search(line):
            write_lines.append((lineno, line))

    if not write_lines:
        return findings

    # We have at least one non-comment write call. Now check if any codified
    # role name appears anywhere in the non-comment portion of the file.
    code_text = "\n".join(code_lines)
    matched_roles = [role for role in role_names if role in code_text]
    if not matched_roles:
        return findings

    # Report each (write line × matched role) pair.
    for write_lineno, write_line in write_lines:
        for role in matched_roles:
            findings.append(
                f"{path}:{write_lineno}: writes codified role '{role}'\n"
                f"    {write_line.strip()[:120]}"
            )

    return findings


def _walk(root: Path, role_names: set[str]) -> list[str]:
    """Walk `root`, scan eligible files, return findings."""
    findings: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in SCAN_EXTENSIONS:
            continue
        if path.name in ALLOWED_WRITERS:
            continue
        # Skip files inside any repo's codified IAM dir (apply.sh, JSON
        # docs, drift-check, this script). Those are sanctioned writers/readers.
        if any(p.name == "iam" and p.parent.name == "infrastructure"
               for p in path.parents):
            continue

        findings.extend(_scan_file(path, role_names))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        default=None,
        help="Repository root to scan (can be passed multiple times). "
             "Default: this repo + every sibling that exists.",
    )
    args = parser.parse_args()

    if args.repo:
        roots = [Path(p).expanduser().resolve() for p in args.repo]
    else:
        # Default: this repo + sibling alpha-engine-* repos that exist.
        this_repo = SCRIPT_DIR.parent.parent
        siblings = [
            this_repo.parent / name
            for name in (
                "alpha-engine",
                "alpha-engine-data",
                "alpha-engine-research",
                "alpha-engine-predictor",
                "alpha-engine-backtester",
                "alpha-engine-dashboard",
                "alpha-engine-lib",
                "alpha-engine-config",
            )
        ]
        roots = [p for p in siblings if p.exists()]

    role_to_home = _all_codified_roles(roots)
    if not role_to_home:
        print("No codified roles found across any scanned repo — nothing to check.")
        return 0

    role_names = set(role_to_home.keys())
    print(f"Scanning for foreign writers of: {sorted(role_names)}")
    print(f"Codified homes: {[(r, str(p.name)) for r, p in sorted(role_to_home.items())]}")
    print(f"Repos: {[str(r.name) for r in roots]}")

    all_findings: list[str] = []
    for root in roots:
        if not root.is_dir():
            print(f"  WARNING: {root} is not a directory — skipping")
            continue
        findings = _walk(root, role_names)
        all_findings.extend(findings)

    if all_findings:
        print(f"\nForeign IAM writers detected ({len(all_findings)} finding(s)):")
        for f in all_findings:
            print(f"  - {f}")
        print()
        print("Codified roles must have exactly one writer (apply.sh in their")
        print("home repo). Remove the inline write from the offending file or")
        print("decodify the role if the inline write is intentional.")
        return 1

    print("OK: no foreign writers found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
