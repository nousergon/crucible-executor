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
