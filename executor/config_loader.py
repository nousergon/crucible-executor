"""
Resolve and load risk.yaml from the private config repo or local fallback.

Search order:
  1. ~/alpha-engine-config/executor/risk.yaml  (EC2 — config repo cloned at home)
  2. {repo_root}/../alpha-engine-config/executor/risk.yaml  (local dev — sibling directory)
  3. {repo_root}/config/risk.yaml  (legacy fallback)
"""

import os
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

_SEARCH_PATHS = [
    os.path.expanduser("~/alpha-engine-config/executor/risk.yaml"),
    os.path.join(_REPO_ROOT, "..", "alpha-engine-config", "executor", "risk.yaml"),
    os.path.join(_REPO_ROOT, "config", "risk.yaml"),
    os.path.join(_REPO_ROOT, "config", "risk.yaml.example"),
]


def get_config_path() -> str:
    """Return the first existing risk.yaml path."""
    for p in _SEARCH_PATHS:
        resolved = os.path.realpath(p)
        if os.path.isfile(resolved):
            return resolved
    raise FileNotFoundError(
        f"risk.yaml not found in any of: {_SEARCH_PATHS}"
    )


CONFIG_PATH = get_config_path()


def load_config() -> dict:
    """Load and return the risk.yaml config dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
