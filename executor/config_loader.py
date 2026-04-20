"""
Resolve and load risk.yaml from the private config repo or legacy local
fallback. The example template is NEVER a valid fallback — it ships
placeholder bucket names (``"your-research-bucket-name"``) that would
silently point downstream consumers at nonexistent S3 buckets. Hit
2026-04-20 via the backtester spot path: missing risk.yaml → this
loader fell through to the example → executor built an ArcticDB URI
against the placeholder bucket → 404 surfaced as a cryptic
``KeyNotFoundException: Not found: [C:universe]`` ~100 lines deep in
the executor-sim call chain.

Search order (example template NOT a fallback — copyable only):
  1. ~/alpha-engine-config/executor/risk.yaml  (EC2 — config repo cloned at home)
  2. {repo_root}/../alpha-engine-config/executor/risk.yaml  (local dev — sibling directory)
  3. {repo_root}/config/risk.yaml  (legacy fallback)

Path resolution is deliberately LAZY — consumers call ``get_config_path()``
or ``load_config()`` at runtime, not at import time. An import-time
``CONFIG_PATH = get_config_path()`` would hard-fail any test, CI runner,
or tooling that merely imports executor without needing to read config.
The old module-level constant was only safe because the removed .example
fallback guaranteed resolution — that's exactly the silent-fallthrough
trap this PR closes. Callers that used to import ``CONFIG_PATH`` now
import ``get_config_path`` and resolve inline.
"""

import os

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

_SEARCH_PATHS = [
    os.path.expanduser("~/alpha-engine-config/executor/risk.yaml"),
    os.path.join(_REPO_ROOT, "..", "alpha-engine-config", "executor", "risk.yaml"),
    os.path.join(_REPO_ROOT, "config", "risk.yaml"),
]


def get_config_path() -> str:
    """Return the first existing risk.yaml path.

    Raises ``FileNotFoundError`` with every candidate named if none
    exist. The example template at ``config/risk.yaml.example`` is NOT
    a candidate — copy it to ``config/risk.yaml`` and fill in real
    values for the intended environment.
    """
    for p in _SEARCH_PATHS:
        resolved = os.path.realpath(p)
        if os.path.isfile(resolved):
            return resolved
    raise FileNotFoundError(
        "executor risk.yaml not found in any of:\n  "
        + "\n  ".join(_SEARCH_PATHS)
        + "\nCopy config/risk.yaml.example → config/risk.yaml and fill in real "
          "values, or clone alpha-engine-config so the config-repo paths resolve. "
          "The .example template is intentionally NOT searched — it ships "
          "placeholder bucket names that silently break downstream ArcticDB + S3 reads."
    )


def load_config() -> dict:
    """Load and return the risk.yaml config dict. Resolves the path lazily."""
    with open(get_config_path()) as f:
        return yaml.safe_load(f)
