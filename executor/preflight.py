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
import os
import subprocess
import time
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

# config#1955: the pipeline pins the executor SHA it synced the box to at its
# freshness gate (T0) into this file. The daemon is started by systemd
# (`systemctl restart alpha-engine-daemon.service`), whose process env does NOT
# inherit the RunDaemon SSM shell's exports — so a file is the only channel that
# reaches BOTH the direct-python RunMorningPlanner and the systemd daemon.
_PINNED_SHA_FILE = Path("/home/ec2-user/.frozen_executor_sha")
# The gate rewrites the pin every pipeline morning; a file older than this is
# treated as stale (e.g. an off-pipeline daemon auto-restart a day later) and
# ignored, so the check live-fetches origin/main rather than pinning yesterday.
_PINNED_SHA_MAX_AGE_S = 18 * 3600


def _resolve_pinned_executor_sha() -> str | None:
    """Resolve the pipeline-pinned executor SHA, or None to live-fetch.

    Precedence (config#1955):
      1. ``EXPECTED_EXECUTOR_SHA`` env var — explicit, e.g. exported by the
         RunMorningPlanner SSM step (direct-python, same process tree).
      2. The freshness-gate-written pin file (``_PINNED_SHA_FILE``, override
         via ``EXPECTED_EXECUTOR_SHA_FILE``) — the ONLY channel that reaches the
         systemd-restarted daemon. Honored only when RECENT (mtime within
         ``_PINNED_SHA_MAX_AGE_S``); a stale file falls through to live-fetch.
      3. None — caller compares against a live ``origin/main`` fetch (today's
         manual / off-pipeline behavior).
    """
    env = (os.environ.get("EXPECTED_EXECUTOR_SHA") or "").strip()
    if env:
        return env
    path = Path(os.environ.get("EXPECTED_EXECUTOR_SHA_FILE") or _PINNED_SHA_FILE)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None  # no pin file → live-fetch fallback
    if age > _PINNED_SHA_MAX_AGE_S:
        log.info(
            "Deploy-drift: ignoring stale executor pin file %s (age %.0fs > "
            "%ds) — live-fetching origin/main instead.",
            path, age, _PINNED_SHA_MAX_AGE_S,
        )
        return None
    return (path.read_text().strip() or None)


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
        expected_sha: str | None = None,
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
            expected_sha: When set (or ``EXPECTED_EXECUTOR_SHA`` is in the
                environment), validate the deployed checkout against THIS
                pinned SHA — the commit the pipeline synced the box to at its
                freshness gate (T0) — instead of a live-refetched
                ``origin/main``. Explicit arg wins over the env var; both
                absent → today's live-fetch behavior.
        """
        root = repo_root or _REPO_ROOT
        deployed = _read_deployed_git_sha(root)

        # config#1955: prefer the pipeline-pinned SHA over a live origin/main
        # fetch. ``check_deploy_drift`` historically compared the box's HEAD
        # against a LIVE-fetched ``origin/main`` — a moving target during the
        # ~48-min pipeline. ne-groomer[bot] merges benign docs/config commits
        # throughout the trading day, so any commit landing between the
        # freshness gate (T0) and this preflight (~T0+48min) retroactively
        # failed an already-validated run (2026-07-08 preopen FailExecution: a
        # docs-only CONTRIBUTING.md merge tripped it — no orders placed).
        # Pinning to the SHA the run synced the box to is a STRONGER fail-loud
        # invariant (it catches a genuine mid-run de-sync of the box) and is
        # immune to a moving upstream. Fail-loud is re-pointed, never loosened.
        if expected_sha is None:
            expected_sha = _resolve_pinned_executor_sha()

        if expected_sha is not None:
            if deployed != expected_sha:
                raise RuntimeError(
                    f"Deploy drift: executor checkout at {root} is on "
                    f"{deployed[:12]} but this run pinned "
                    f"EXPECTED_EXECUTOR_SHA={expected_sha[:12]} at its freshness "
                    f"gate. The box de-synced from the SHA the pipeline synced "
                    f"it to mid-run — refusing to proceed. Running code that is "
                    f"not what this run validated is the 2026-04-20 stale-code "
                    f"class. (This is NOT tripped by a benign mid-pipeline "
                    f"merge to origin/main — that no longer moves the target.)"
                )
            log.info(
                "Deploy-drift: executor checkout at %s matches pipeline-pinned "
                "EXPECTED_EXECUTOR_SHA %s ✓",
                deployed[:12], expected_sha[:12],
            )
            return

        # No pinned SHA (manual / off-pipeline invocation): preserve today's
        # behavior — compare against live ``origin/main`` HEAD, with
        # GitHub-unreachable → warn-and-continue.
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
        # The executor runs as root via SSM, but the checkout is owned by
        # ec2-user — git's dubious-ownership guard (CVE-2022-24765) then
        # blocks ``rev-parse`` with exit 128, and root's $HOME is unset
        # under SSM so no global ``safe.directory`` exception is even
        # readable. Declare the path we already control as trusted inline,
        # making the deploy-drift read immune to who runs it on any box
        # (host-independent — does not rely on /etc/gitconfig state).
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
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
