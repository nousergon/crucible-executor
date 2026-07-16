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


# A shell/python variable REFERENCE on the write call's role argument:
# `${ROLE_NAME}`, `$ROLE_NAME`, or (already-literal) a bare identifier.
_VARREF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")

# The write call's role argument, across the syntaxes this repo's writers
# actually use: bash `--role-name X`, boto3 `RoleName=X`, CFN/JSON
# `"RoleName": X` / `RoleName: X`.
_ROLE_ARG_RE = re.compile(
    r'(?:--role-name[= ]+|RoleName["\']?\s*[:=]\s*)["\']?'
    r'(\$\{?[A-Za-z0-9_]+\}?|[A-Za-z0-9_.\-]+)'
)


def _join_continuations(code_lines: list[str]) -> list[tuple[int, str]]:
    """Join bash `\\`-continued lines into one logical statement (keeping the
    FIRST physical line's number), so a write call whose `--role-name` sits
    on the NEXT physical line (the common multi-line deploy-script style)
    is still visible to a single regex pass. Non-bash files have no trailing
    `\\` continuations, so this is a no-op for them."""
    statements: list[tuple[int, str]] = []
    i, n = 0, len(code_lines)
    while i < n:
        start = i + 1
        parts = [code_lines[i]]
        while parts[-1].rstrip().endswith("\\") and i + 1 < n:
            i += 1
            parts.append(code_lines[i])
        statements.append((start, "\n".join(parts)))
        i += 1
    return statements


def _resolve_role_arg(raw: str, full_code_text: str) -> str | None:
    """Resolve a write call's role-argument token to its literal value.

    Returns the literal role name, or None if it can't be pinned to a single
    literal (an unassigned/indirect reference, a terraform resource
    attribute, etc.) — the caller then falls back to the conservative
    whole-file heuristic for that one write, rather than silently dropping
    it (no-silent-fails: an unresolved write must never look identical to a
    cleared one)."""
    raw = raw.strip().strip("\"'")
    m = _VARREF_RE.match(raw)
    if not m:
        return raw if re.fullmatch(r"[A-Za-z0-9_\-]+", raw) else None
    var = m.group(1)
    assign_re = re.compile(
        r"^\s*" + re.escape(var) + r"""\s*=\s*['"]?([^'"\n]+?)['"]?\s*$""",
        re.MULTILINE,
    )
    values = assign_re.findall(full_code_text)
    if not values:
        return None
    resolved = values[-1].strip()
    if "$" in resolved:  # one level of indirection only
        return None
    return resolved


def _scan_file(path: Path, role_names: set[str]) -> list[str]:
    """Return list of findings for this file.

    For each write-API call, resolve ITS OWN `--role-name`/`RoleName`
    argument (following one level of variable indirection, e.g. a
    `ROLE_NAME=...` assignment near the top of a deploy script) and flag
    only if THAT resolves to a codified role name.

    Why not file-scope substring matching (the original approach): a deploy
    script legitimately writing its OWN unrelated Lambda-execution role can
    also mention a DIFFERENT codified role's name elsewhere in the file —
    e.g. an operator-facing comment/echo about a separate, correctly-codified
    grant (see alpha-engine-config#2423, the `ssm-liveness-poller/deploy.sh`
    false positive: the script's `put-role-policy` targets its own
    `alpha-engine-ssm-liveness-poller-role`, but an unrelated NOTE echo
    mentioning `alpha-engine-step-functions-role` made the old file-scope
    check flag it as a foreign writer of THAT role). File-scope matching
    can't tell those apart; resolving the write's actual target can.

    When a write's role argument can't be resolved to a single literal
    (unassigned var, terraform resource attribute, double indirection),
    fall back to the old conservative file-scope heuristic for THAT write
    only, clearly labeled as unresolved so it isn't mistaken for a confirmed
    hit — never silently dropped.
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

    code_lines = [
        "" if comment_re.match(line) else line for line in text.splitlines()
    ]
    full_code_text = "\n".join(code_lines)
    statements = _join_continuations(code_lines)
    write_statements = [(ln, stmt) for ln, stmt in statements if write_re.search(stmt)]

    if not write_statements:
        return findings

    for write_lineno, stmt in write_statements:
        display_line = next((s for s in stmt.splitlines() if s.strip()), stmt.strip())[:120]
        arg_match = _ROLE_ARG_RE.search(stmt)
        resolved = _resolve_role_arg(arg_match.group(1), full_code_text) if arg_match else None

        if resolved is not None:
            if resolved in role_names:
                findings.append(
                    f"{path}:{write_lineno}: writes codified role '{resolved}'\n"
                    f"    {display_line.strip()}"
                )
            # Resolved to a NON-codified role: definitively this write's own
            # target, not a violation — no fallback, regardless of what else
            # the file happens to mention.
            continue

        # Couldn't resolve to a single literal — fail toward caution.
        for role in role_names:
            if role in full_code_text:
                findings.append(
                    f"{path}:{write_lineno}: writes UNRESOLVED role target, near "
                    f"mention of codified role '{role}' (could not statically "
                    f"resolve this write's role argument — verify manually)\n"
                    f"    {display_line.strip()}"
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
