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
  1. ~/alpha-engine-config/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/risk.yaml
  2. {repo_root}/../alpha-engine-config/experiments/$EXP/executor/risk.yaml
  3. ~/alpha-engine-config/executor/risk.yaml          (legacy top-level)
  4. {repo_root}/../alpha-engine-config/executor/risk.yaml  (legacy, sibling)
  5. {repo_root}/config/risk.yaml                      (legacy repo-local)

Experiment-package resolution (config#1042, HARNESS_EXPERIMENT_CLASSIFICATION
§3): the executor's risk beliefs load from
``experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/risk.yaml`` (default
experiment ``reference``) ahead of the legacy top-level ``executor/risk.yaml``,
which is retained as a fallback through the transition. Mirrors the loader in
alpha-engine-research/config.py::_find_config and
alpha-engine-data/weekly_collector.py::load_config. The experiment id is read
from the environment at import time (consistent with the sibling loaders) — set
``ALPHA_ENGINE_EXPERIMENT_ID`` before the process starts to select a slot.

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


def _build_search_paths() -> list:
    """Build the risk.yaml search order, experiment-package first (config#1042).

    Returns the ordered candidate paths: the experiment-package copy under
    ``experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/`` in the config repo
    first, then the legacy top-level ``executor/`` config-repo path, then the
    legacy repo-local ``config/risk.yaml``. The ``.example`` template is never
    a candidate (see module docstring).
    """
    exp = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "reference")
    config_roots = [
        os.path.expanduser("~/alpha-engine-config"),
        os.path.join(_REPO_ROOT, "..", "alpha-engine-config"),
    ]
    paths = [os.path.join(r, "experiments", exp, "executor", "risk.yaml") for r in config_roots]
    paths += [os.path.join(r, "executor", "risk.yaml") for r in config_roots]
    paths.append(os.path.join(_REPO_ROOT, "config", "risk.yaml"))
    return paths


_SEARCH_PATHS = _build_search_paths()


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
