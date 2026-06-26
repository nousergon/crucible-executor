"""
Executor preflight: connectivity + safety checks run at the top of each
entrypoint before any real work starts.

Primitives live in ``nousergon_lib.preflight.BasePreflight``; this
module composes them into a mode-specific sequence. See the
alpha-engine-lib README for the rationale.

Data-freshness assertions (universe + macro/SPY) live upstream in
``alpha-engine-data``'s preflight, which runs before ``RunMorningPlanner``
+ ``RunDaemon`` + EOD steps in the weekday + EOD Step Functions. If
upstream data is stale, the data step hard-fails and the SF never reaches
the executor — re-checking here was redundant.

Modes:

- ``"main"`` — ``executor/main.py``, the morning order-book planner.
- ``"daemon"`` — ``executor/daemon.py``, the sole order executor. The IB
  paper-account guard is invoked separately by the daemon after IBKRClient
  connects.
- ``"eod"`` — ``executor/eod_reconcile.py``.

Deploy-drift
------------
The executor is deployed differently from the predictor inference
Lambda. The predictor ships a Docker image with a ``GIT_SHA`` stamp
baked in at build time, and ``BasePreflight.check_deploy_drift`` compares
that baked stamp against ``origin/main`` HEAD (see the predictor's
``PredictorPreflight``). The executor instead runs from a plain git
checkout on the ae-trading EC2 box, refreshed on boot by
``infrastructure/boot-pull.sh`` (``boot-pull.service``, weekday 6:10 AM
PT) via ``git fetch && git reset --hard origin/main``.

That difference matters: if boot-pull silently fails (network hiccup,
stale ``~/.netrc`` PAT — the 2026-06-03 incident), the checkout stays
on a *prior* commit while ``origin/main`` advances, and the daemon runs
stale code on fresh signals with nothing surfacing it. The 24-hour
worst-case lag between merge and the next boot makes this a real window.

``check_deploy_drift`` below is the executor's equivalent of the
predictor preflight: it reads the *deployed* SHA from ``git rev-parse
HEAD`` in the checkout (the executor's "baked stamp"), compares it to
``crucible-executor@main`` HEAD via the GitHub API, and hard-fails on a
mismatch — the same fail-loud posture as the predictor. It reuses the
lib's ``_fetch_origin_main_sha`` helper so a GitHub-outage / rate-limit
fix lands in one place (executor is the helper's 2nd consumer, predictor
the 1st).

Degraded-mode posture mirrors the predictor where the conditions
overlap, and adds the one case the issue (config#892) calls out as
executor-specific:

- GitHub API unreachable  → warn-and-continue (same as predictor: a
  GitHub outage must not block a trading-hours daemon).
- No ``.git`` directory   → **hard-fail**. Unlike the predictor's
  missing-stamp warn path (a legacy image legitimately predates
  stamping), a missing ``.git`` on the executor box means the checkout
  itself is gone — boot-pull never ran or something is very wrong. There
  is no legitimate first-boot case here: the box is provisioned by
  ``git clone``, so ``.git`` is present from the very first boot.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from nousergon_lib.preflight import BasePreflight, _fetch_origin_main_sha

log = logging.getLogger(__name__)

# crucible-executor is PUBLIC, so the GitHub branch-HEAD lookup needs no
# auth (matches the predictor's _PREDICTOR_REPO usage). Re-pointed from
# the pre-migration monorepo path cipher813/alpha-engine@main referenced
# in the original ROADMAP item.
_EXECUTOR_REPO = "nousergon/crucible-executor"

# The deployed checkout root — two levels up from this file
# (executor/preflight.py → repo root). Mirrors config_loader._REPO_ROOT.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class ExecutorPreflight(BasePreflight):
    """Preflight checks for the three executor entrypoints."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("main", "daemon", "eod"):
            raise ValueError(f"ExecutorPreflight: unknown mode {mode!r}")
        self.mode = mode

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
        self.check_deploy_drift()

    # ── Deploy-drift (executor git-checkout variant) ─────────────────────

    def check_deploy_drift(
        self,
        repo: str = _EXECUTOR_REPO,
        branch: str = "main",
        *,
        repo_root: Path | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Hard-fail if the deployed checkout's HEAD lags ``repo@branch`` HEAD.

        The executor's "deployed SHA" is ``git rev-parse HEAD`` in the
        checkout that ``boot-pull.sh`` maintains — there is no baked
        ``GIT_SHA.txt`` stamp the way the predictor Lambda image has, so
        this overrides ``BasePreflight.check_deploy_drift`` (which reads
        that stamp) with the git-checkout equivalent. Everything else —
        the upstream fetch helper and the fail-loud-on-mismatch posture —
        mirrors the predictor.

        Posture:
        - deployed == upstream            → pass (logged ✓)
        - deployed != upstream            → **hard-fail** (RuntimeError):
          boot-pull is behind ``origin/main``; daemon must not run stale
          code on fresh signals.
        - GitHub API unreachable          → warn-and-continue (same as the
          predictor / lib: a GitHub outage must not block trading hours).
        - no ``.git`` dir / git error     → **hard-fail**: the checkout is
          gone or unreadable; boot-pull never ran or the box is broken.
          (Issue config#892: "missing .git dir — hard-fail, something is
          very wrong." No legitimate first-boot case: the box is
          provisioned by ``git clone``.)

        Args:
            repo: ``"owner/name"`` to compare against. Default
                ``"nousergon/crucible-executor"``.
            branch: Branch HEAD to compare against. Default ``"main"``.
            repo_root: Checkout root to read HEAD from. Defaults to the
                executor repo root inferred from this file's location.
            timeout: GitHub API timeout in seconds.
        """
        root = repo_root or _REPO_ROOT
        deployed = _read_deployed_git_sha(root)

        upstream = _fetch_origin_main_sha(repo, branch=branch, timeout=timeout)
        if upstream is None:
            # _fetch_origin_main_sha already logged the reason. Warn-and-
            # continue: a GitHub outage must not block a trading-hours
            # daemon (same posture as the predictor / lib).
            return

        if deployed != upstream:
            raise RuntimeError(
                f"Deploy drift: executor checkout at {root} is on "
                f"{deployed[:12]} but {repo}@{branch} is now at "
                f"{upstream[:12]}. boot-pull.service did not advance the "
                f"checkout to the latest commit (network hiccup or stale "
                f"~/.netrc PAT — see infrastructure/boot-pull.sh). Re-run "
                f"boot-pull (or `git fetch && git reset --hard "
                f"origin/{branch}`) before resuming. Refusing to proceed — "
                f"running stale code on new signals is how 2026-04-20 "
                f"happened."
            )

        log.info(
            "Deploy-drift: executor checkout at %s matches %s@%s ✓",
            deployed[:12], repo, branch,
        )


def _read_deployed_git_sha(repo_root: Path) -> str:
    """Return the deployed SHA: ``git rev-parse HEAD`` in ``repo_root``.

    Hard-fails (RuntimeError) if the directory is not a git checkout or
    git cannot read HEAD. Unlike the predictor's baked-stamp reader —
    where a missing stamp is a legitimate legacy-image warn path — a
    missing/broken ``.git`` on the executor box means the checkout is
    gone or corrupt, which is never a legitimate state (the box is
    provisioned by ``git clone``; see issue config#892).
    """
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise RuntimeError(
            f"Deploy drift: no .git directory at {repo_root} — the executor "
            f"checkout is missing or corrupt. boot-pull.service never cloned "
            f"the repo (or the volume is wrong). Cannot verify deployed SHA; "
            f"refusing to proceed."
        )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Deploy drift: `git rev-parse HEAD` failed in {repo_root} "
            f"({exc.__class__.__name__}: {exc}) — checkout is unreadable. "
            f"Refusing to proceed."
        ) from exc

    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError(
            f"Deploy drift: `git rev-parse HEAD` returned empty in "
            f"{repo_root} — checkout has no commits. Refusing to proceed."
        )
    return sha
