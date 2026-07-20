"""Ensure pyproject.toml and requirements.txt dependencies are synchronized.

The executor ships with both pyproject.toml (for `pip install -e .`) and
requirements.txt (for production/Docker deploys). A divergence class has bitten
the organization multiple times:

  - pyproject.toml lists only 5 packages: ib_insync, boto3, pyyaml, yfinance, pandas.
  - requirements.txt pins 11+ packages including arcticdb, krepis, nousergon-lib,
    exchange-calendars, cvxpy, scikit-learn, etc. — the actual runtime dependencies.
  - `pip install -e .` from pyproject.toml yields a silently broken environment
    (import errors, missing optional deps) that drifts independently until the
    moment a dev tries to run the executor or tests locally.

This test re-greps both sources on every CI run to enforce lockstep: all
requirements.txt packages (except transitive) must appear in pyproject.toml
dependencies, and pinned versions must match (or be compatible within declared
ranges).

A second, narrower divergence class survives even when presence is enforced:
requirements.txt pins some packages to an exact git tag (e.g.
`nousergon-lib @ git+https://...@v0.86.0`) for the production/Docker deploy,
while pyproject.toml only declares an open-ended floor for the same package
(e.g. `nousergon-lib>=0.86.0`). `pip install -e .` reads only pyproject.toml,
so it is free to resolve any version satisfying the floor — it is not
provably the same artifact as the exact git tag requirements.txt pins.
`test_git_pinned_deps_match_pyproject_exactly` closes that gap: for every
requirements.txt dependency pinned to an exact git tag, pyproject.toml's
specifier for that package must be an exact-equals (`==`) pin at the same
version, not merely a compatible range.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_pyproject_deps() -> dict[str, str]:
    """Extract dependencies from pyproject.toml [project] section."""
    text = (_REPO_ROOT / "pyproject.toml").read_text()
    deps = {}
    in_deps = False
    for line in text.split("\n"):
        if "dependencies = [" in line:
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("]"):
                break
            # Match lines like '    "ib_insync>=0.9.86",'
            match = re.search(r'"([^"]+)"', line)
            if match:
                dep_spec = match.group(1)
                # Extract package name and version spec
                pkg_name = re.split(r'[><=!@\[]', dep_spec)[0]
                deps[pkg_name] = dep_spec
    return deps


def _parse_requirements() -> dict[str, str]:
    """Extract top-level dependencies from requirements.txt (skip comments/comments)."""
    text = (_REPO_ROOT / "requirements.txt").read_text()
    deps = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle git-pinned deps like: nousergon-lib[...] @ git+https://...
        if "@" in line:
            pkg_part = line.split("@")[0].strip()
            pkg_name = re.split(r'[\[\]]', pkg_part)[0]
            deps[pkg_name] = line
        else:
            # Regular version specifiers like: ib_insync~=0.9.86
            pkg_name = re.split(r'[><=!~\[]', line)[0]
            if pkg_name:
                deps[pkg_name] = line
    return deps


def _parse_git_pin_version(dep_spec: str) -> str | None:
    """Extract the exact tag version from a requirements.txt git-pinned spec.

    Handles entries like:
      nousergon-lib[flow-doctor,contracts] @ git+https://github.com/nousergon/nousergon-lib@v0.86.0
    Returns "0.86.0" (the "v" tag prefix stripped), or None if the spec is not
    a git pin, or the git pin has no trailing `@<tag>` ref (e.g. tracks a
    branch name rather than a version tag) so no version can be compared.
    """
    if "@ git+" not in dep_spec:
        return None
    # The URL itself may contain '@' (git+https://host@ref is not valid, but
    # be defensive) — split only on the LAST '@', which separates the git URL
    # from its ref, matching how pip itself parses `url@ref` VCS specifiers.
    ref = dep_spec.rsplit("@", 1)[-1].strip()
    match = re.match(r"v?(\d+(?:\.\d+)*)$", ref)
    if not match:
        return None
    return match.group(1)


def _parse_pyproject_exact_version(dep_spec: str) -> str | None:
    """Extract the version from a pyproject.toml spec IFF it is an exact (`==`) pin.

    Returns None for open-ended specifiers (`>=`, `~=`, ranges, extras-only,
    etc.) since those cannot be proven to resolve to one concrete version.
    """
    match = re.search(r"==\s*(\d+(?:\.\d+)*)\s*$", dep_spec)
    if not match:
        return None
    return match.group(1)


def test_git_pinned_deps_match_pyproject_exactly():
    """Packages pinned to an exact git tag in requirements.txt must be pinned
    to that SAME exact version in pyproject.toml — not just a compatible range.

    Guards against the drift class where requirements.txt (production/Docker)
    pins `nousergon-lib @ git+...@v0.86.0` while pyproject.toml (dev/local
    `pip install -e .`) only declares `nousergon-lib>=0.86.0`. That floor is
    "range-compatible" with the git tag today, but it does not prove the two
    installs resolve to the same artifact: the floor is open-ended, so a dev
    environment can silently drift onto a different resolved version than
    what production actually deploys. Only an exact `==` pin at the identical
    version closes that gap.
    """
    pyproject_deps = _parse_pyproject_deps()
    requirements_deps = _parse_requirements()

    mismatches = []
    for pkg, req_spec in requirements_deps.items():
        git_version = _parse_git_pin_version(req_spec)
        if git_version is None:
            continue  # not a git-tag pin; nothing to cross-check here

        pyproject_spec = pyproject_deps.get(pkg)
        if pyproject_spec is None:
            # Absence is already caught by test_pyproject_has_all_direct_dependencies.
            continue

        pyproject_version = _parse_pyproject_exact_version(pyproject_spec)
        if pyproject_version != git_version:
            mismatches.append(
                f"  {pkg}: requirements.txt pins git tag v{git_version} "
                f"({req_spec!r}) but pyproject.toml specifies {pyproject_spec!r} "
                f"(must be exactly '=={git_version}')"
            )

    assert len(mismatches) == 0, (
        "pyproject.toml specifiers do not exactly match requirements.txt's "
        "git-tag pins:\n" + "\n".join(mismatches) + "\n\n"
        "`pip install -e .` reads pyproject.toml and can resolve any version "
        "satisfying an open-ended floor (e.g. '>=0.86.0'), which is not "
        "provably the same artifact as requirements.txt's exact git tag pin. "
        "Change pyproject.toml's specifier for these packages to an exact "
        "'==<version>' pin matching the git tag so dev (`pip install -e .`) "
        "and production (`pip install -r requirements.txt`) cannot silently "
        "diverge."
    )


def test_pyproject_has_all_direct_dependencies():
    """pyproject.toml must list all direct (non-transitive) deps from requirements.txt.

    This guards against the silent breakage class where `pip install -e .` skips
    optional packages (arcticdb, cvxpy, scikit-learn, etc.) that production
    deploys pull via requirements.txt.
    """
    pyproject_deps = _parse_pyproject_deps()
    requirements_deps = _parse_requirements()

    # Core packages that MUST be in pyproject (direct dependencies)
    # Listed by their presence in requirements.txt and importance to executor runtime.
    direct_packages = [
        "ib_insync",
        "boto3",
        "pyyaml",
        "yfinance",
        "pandas",
        "numpy",
        "pyarrow",
        "arcticdb",
        "exchange-calendars",
        "requests",
        "websocket-client",
        "nousergon-lib",
        "krepis",
        "cvxpy",
        "scikit-learn",
    ]

    missing = []
    for pkg in direct_packages:
        if pkg not in pyproject_deps:
            missing.append(pkg)

    assert len(missing) == 0, (
        f"pyproject.toml is missing direct dependencies that are in "
        f"requirements.txt:\n"
        + "\n".join(f"  {pkg}" for pkg in missing)
        + f"\n\nThese packages are required for executor runtime. Add them to "
        f"pyproject.toml [project] dependencies so `pip install -e .` yields "
        f"the same environment as `pip install -r requirements.txt`."
    )
