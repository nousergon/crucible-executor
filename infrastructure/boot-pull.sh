#!/bin/bash
# boot-pull.sh — Pull latest code for all Alpha Engine repos on EC2 boot.
#
# Runs as a systemd oneshot service (boot-pull.service) before any cron jobs.
# Safe to run manually too.
#
# Install:
#   sudo bash infrastructure/install-boot-pull.sh

set -uo pipefail

# AE_BOOT_PULL_LOG exists so tests can source this file and exercise
# sync_repo_to_main() without needing write access to /var/log.
LOG="${AE_BOOT_PULL_LOG:-/var/log/boot-pull.log}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# ── Shared git-sync serialization (config#1944) ────────────────────────────
# boot-pull.service and the weekday Step Function's CodeFreshnessGate +
# ChronicGapSelfHeal (nousergon-data infrastructure/step_function_daily.json)
# all run `git fetch / checkout -f main / reset --hard / pull` on the SAME
# ec2-user checkouts on THIS trading box. They are independent git writers and
# raced on `.git/index.lock`: 2026-07-08 ne-preopen-trading FailExecution —
# CodeFreshnessGate's checkout/reset died with "Another git process seems to be
# running" (exit 128) because boot-pull's `git reset --hard` still held
# alpha-engine-data/.git/index.lock.
#
# A shared advisory flock is window-free (a kernel mutex — whoever acquires
# first runs, the other blocks) and auto-releases on process death, unlike a
# bare .git/index.lock which can strand a stale lock and deadlock. Every
# trading-box git-sync section acquires this SAME lock inode so the writers
# serialize instead of racing.
#
# Lock lives in ec2-user's HOME, NOT /var/lock: /var/lock -> /run/lock is
# root:root 0755, so this script (runs as ec2-user) cannot create a lock file
# there, and the SF gate runs its git as `sudo -u ec2-user`. Every actor
# flocks this path AS ec2-user, so opening it for the lock always succeeds
# regardless of which actor created the inode first. The nousergon-data gate
# MUST use this identical path — pinned by a guard test in each repo.
#
# FAIL-LOUD: a `flock -w` timeout is a genuinely stuck git writer, not a
# swallowable condition — flock returns non-zero, the per-repo `if` below takes
# its else branch, PULL_FAILURES increments, and the script exits 1 (surfaced
# via flow-doctor + the FAIL log lines). Never swallow a lock timeout.
GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"
GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"

# ── Sync one checkout to origin/main, judged on the post-condition ──────────
# Returns 0 when the checkout provably sits on the remote's current main, 1
# otherwise. Extracted as a function (config-I4978) for two reasons: the main
# loop reads as one decision instead of four nested branches, and the decision
# becomes directly testable — every prior guard on this logic was a regex over
# this file's source text, which is exactly why the control-flow defect below
# shipped and then survived a review.
#
# THE DEFECT THIS CLOSES. `git fetch` exits non-zero when ANY ref fails to
# update, and the previous shape chained `git fetch origin ... && git checkout
# -f main && git reset --hard origin/main`. So a ref-update failure skipped the
# self-heal AND raised "the executor may be running stale code" — on every
# weekday boot for at least 7 days (07-22 through 07-28 confirmed in
# /var/log/boot-pull.log).
#
# Scoping the fetch to `origin main` narrows the blast radius but does NOT close
# it: the ref observed failing its compare-and-swap on 2026-07-28 was
# `refs/remotes/origin/main` ITSELF —
#
#   error: cannot lock ref 'refs/remotes/origin/main': is at 01a279f...
#          but expected a3b5971...
#    ! a3b5971..01a279f  main -> origin/main  (unable to update local ref)
#
# — while that same fetch had already fast-forwarded origin/main to 01a279f
# (`git reflog show refs/remotes/origin/main` confirms). The error was benign
# and the exit code still suppressed the self-heal. A concurrent git writer on
# this box wins that race whenever it interleaves, so the exit code cannot be
# the gate no matter how narrow the refspec is.
# ── Transient-remote retry class (alpha-engine-config-I10950) ───────────────
# The one-shot retry inside sync_repo_to_main() below answers a ref
# compare-and-swap race. It does NOT answer the other failure this box has
# hit: a credential-helper cache MISS that mints a token GitHub then refuses
# ("Repository not found") for about a minute, while the same fetch succeeds
# seconds later. The immediate retry re-serves the same bad token, so it fails
# identically, and boot-pull reports the repo FAILED.
#
# Measured on THIS box, i-018eb3307a21329bf: 2026-09-17 12:15Z, and again
# 2026-10-02 12:16Z ("boot-pull FAILED on the TRADING box: 1 repo(s) could not
# be updated — /home/ec2-user/alpha-engine-config (git)"). The preopen's
# CodeFreshnessGate absorbed the 10-02 occurrence only because
# nousergon-data-PR1776 gave the gate's git_retry() this class. The dashboard
# box's boot-pull got the same class in crucible-dashboard-PR869. This ports
# it to the third and last git writer on the path.
#
# This class is deliberately narrow. Every attempt re-runs the WHOLE sync below
# (fetch with its CAS retry, then checkout/reset). An attempt is retried only
# when the fetch failed twice (inner rc=10) AND the output matches a named
# transient-remote signature. The total is bounded at 3 attempts, with a
# 5s-then-10s backoff and a credential-cache erase before each retry, so the
# retry re-mints a token instead of being served the same one. Any other
# failure keeps exactly its previous behaviour:
#   - a flock timeout (rc=1)
#   - a checkout/reset failure (rc=2)
#   - a cd failure (rc=3)
#   - an unmatched fetch error
# The backoff sleeps OUTSIDE the shared git-sync flock, so a retry never makes
# the CodeFreshnessGate wait on a sleeping lock holder.
#
# The signature list is the same list the gate (nousergon-data
# step_function_daily.json git_retry) and the dashboard's boot-pull use. If
# you change one, change all three.
_BOOT_PULL_TRANSIENT_SIGNATURES=(
    'Repository not found'
    'Authentication failed'
    'could not read Username'
    'The requested URL returned error: 5'
    'Could not resolve host'
    'Failed to connect'
    'Connection timed out'
    'RPC failed'
    'early EOF'
    'remote end hung up'
    'unable to access'
)

# _boot_pull_transient_signature <sync output> — echoes the first matching
# signature and returns 0. Returns 1 if nothing matched, which means the
# failure is not believed to clear on its own and must not be retried.
_boot_pull_transient_signature() {
    local text="$1" sig
    for sig in "${_BOOT_PULL_TRANSIENT_SIGNATURES[@]}"; do
        if grep -qF -- "$sig" <<<"$text"; then
            printf '%s' "$sig"
            return 0
        fi
    done
    return 1
}

# _boot_pull_erase_credential <checkout-dir> — invalidates that repo's cached
# GitHub App token so the retry re-mints instead of re-serving it. The cache
# key is the slug from the checkout's OWN origin remote, never the directory
# name: /home/ec2-user/alpha-engine-data is nousergon/nousergon-data. A
# missing helper or an unresolvable remote is logged and SKIPPED, never
# failed. This step exists to help a retry succeed and must never turn a
# retryable failure into a hard one. boot-pull.service runs as ec2-user,
# whose per-uid cache this is, so no sudo is needed.
_boot_pull_erase_credential() {
    local repo="$1" helper url slug
    helper="${AE_CRED_HELPER:-/usr/local/bin/git-credential-nousergon-app}"
    if [ ! -x "$helper" ]; then
        log "SKIP credential erase for $repo — $helper missing or not executable"
        return 0
    fi
    url=$(git -C "$repo" remote get-url origin 2>/dev/null || echo "")
    slug=$(printf '%s' "$url" | sed -E 's#^(https://github\.com/|git@github\.com:)##; s#\.git/?$##')
    if [ -z "$url" ] || [ -z "$slug" ]; then
        log "WARN could not resolve repo slug for $repo from its origin remote — skipping credential erase"
        return 0
    fi
    if printf 'protocol=https\nhost=github.com\npath=%s\n\n' "$slug" \
            | "$helper" erase >> "$LOG" 2>&1; then
        log "OK   erased cached credential for $slug ($repo)"
    else
        log "WARN credential erase for $slug ($repo) failed — retry proceeds anyway"
    fi
    return 0
}

sync_repo_to_main() {
    local repo="$1"
    local rc=0
    local attempt=1 max_attempts=3 out sig="" backoff

    # The fetch's status is captured SEPARATELY from the checkout/reset, and the
    # two are chained with `;` rather than `&&`, for a reason the group's single
    # exit code cannot express. Two distinct failures need two distinct answers:
    #
    #   fetch failed  -> origin/main may be STALE, so no comparison between two
    #                    LOCAL refs can prove the checkout is current. Needs an
    #                    authoritative remote read.
    #   reset failed  -> the tree is not where we put it. Post-condition catches
    #                    it directly.
    #
    # Collapsing them (the previous `&&` chain, and a plain `;` too) produces the
    # false OK that matters most: a fetch that cannot reach the remote, followed
    # by a reset onto the stale origin/main, exits 0 with HEAD == origin/main
    # satisfied by two equally-old refs. That is precisely the stale-code case
    # this whole alert exists to catch.
    #
    # A failed fetch is RETRIED ONCE, scoped to main, before it is believed.
    # This is what separates the two causes, and it does so by construction:
    #
    #   ref-CAS race        — transient. The retry finds the ref already at the
    #                         value it wanted, so it succeeds.
    #   unreachable remote  — auth/network. The retry fails the same way.
    #
    # Measured on the trading box 2026-07-30: `refs/remotes/origin/main` itself
    # was among the refs failing CAS ("is at 876b73b7 but expected 7aa716e6"),
    # i.e. something had already advanced origin/main between this fetch's ref
    # negotiation and its ref update. That is precisely the transient a retry
    # absorbs and an exit code cannot.
    #
    # A remote read (`git ls-remote`) was the obvious alternative and is WORSE
    # here: `alpha-engine-config` takes merges continuously, so comparing HEAD
    # against a freshly-read remote tip fails whenever main advances in the
    # second between the reset and the read — reporting a correctly-synced box
    # as stale. The retry asks the question that actually matters ("can we reach
    # the remote and get main?") without racing anything.
    #
    # Exit code contract for the inner script:
    #   0  fetch (possibly on retry) + checkout + reset all clean
    #   10 fetch failed TWICE — remote genuinely unreachable
    #   2  checkout/reset failed
    #   3  cd failed
    # flock's own `-w` timeout surfaces as 1, which is why 10 is used for the
    # fetch rather than 1 — the two must stay distinguishable.
    #
    # The whole sync is wrapped in the bounded transient-remote retry
    # (alpha-engine-config-I10950, see _BOOT_PULL_TRANSIENT_SIGNATURES above).
    # Only rc=10 whose output names a transient signature goes round again.
    while :; do
        rc=0
        out=$(flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" bash -c '
        cd "$1" || exit 3
        fetch_rc=0
        if ! git fetch origin main --prune; then
            echo "boot-pull: fetch reported an error; retrying main once" >&2
            git fetch origin main || fetch_rc=10
        fi
        git checkout -f main && git reset --hard origin/main || exit 2
        exit $fetch_rc
    ' _ "$repo" 2>&1) || rc=$?
        [ -n "$out" ] && printf '%s\n' "$out" >> "$LOG"

        [ "$rc" -eq 10 ] || break
        if ! sig=$(_boot_pull_transient_signature "$out"); then
            sig=""
            break
        fi
        if [ "$attempt" -ge "$max_attempts" ]; then
            log "FAIL $repo — attempt $attempt/$max_attempts exhausted retrying transient class '$sig'"
            break
        fi
        # AE_BOOT_PULL_RETRY_BACKOFF_{1,2} exist so tests can drive this to 0s;
        # production never sets them.
        backoff="${AE_BOOT_PULL_RETRY_BACKOFF_1:-5}"
        [ "$attempt" -gt 1 ] && backoff="${AE_BOOT_PULL_RETRY_BACKOFF_2:-10}"
        log "RETRY $repo — attempt $attempt/$max_attempts failed matching transient class '$sig'; erasing cached credential and retrying in ${backoff}s"
        _boot_pull_erase_credential "$repo"
        sleep "$backoff"
        attempt=$((attempt + 1))
    done
    if [ "$rc" -eq 0 ] && [ "$attempt" -gt 1 ]; then
        log "OK   $repo — recovered on attempt $attempt/$max_attempts after transient '$sig'"
    fi

    # A flock timeout keeps its established fail-loud semantics (config#1944): a
    # stuck git writer on this box is worth an alert on its own, independent of
    # whether the code happens to be current.
    if [ "$rc" -eq 1 ]; then
        log "FAIL $repo — git-sync flock timed out after ${GIT_SYNC_LOCK_WAIT}s on $GIT_SYNC_LOCK (a git writer is stuck)"
        return 1
    fi

    local head_branch head_sha origin_main_sha
    head_branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    head_sha=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo "")
    origin_main_sha=$(git -C "$repo" rev-parse origin/main 2>/dev/null || echo "")

    # Judge on the POST-CONDITION, evaluated UNCONDITIONALLY. The invariant that
    # matters is "this checkout is on main at the remote's current main"; the
    # sync's exit code is only ever a proxy for it, and it is a proxy that is
    # wrong in both directions.
    if [ "$head_branch" != "main" ] || [ -z "$head_sha" ] \
            || [ "$head_sha" != "$origin_main_sha" ]; then
        log "FAIL $repo — post-condition failed: branch=$head_branch HEAD=$head_sha origin/main=$origin_main_sha (sync rc=$rc)"
        return 1
    fi

    # HEAD == origin/main is necessary but NOT sufficient. If the fetch never
    # reached the remote, origin/main is stale too and the comparison above is
    # satisfied by two equally-old LOCAL refs — a false OK in exactly the
    # stale-code case this alert exists for. rc != 0 here means the fetch failed
    # twice, so nothing local can be trusted as current.
    if [ "$rc" -ne 0 ]; then
        log "FAIL $repo — sync rc=$rc (fetch failed twice / checkout unusable); HEAD=$head_sha cannot be shown to be current"
        return 1
    fi

    log "OK   $repo — $(git -C "$repo" log --oneline -1)"
    return 0
}

# Tests source this file for sync_repo_to_main() and must not execute the boot
# sequence (SSM reads, sudo, systemctl). Everything above this line is pure
# definition; everything below has side effects.
if [ "${AE_BOOT_PULL_LIB_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

log "=== boot-pull started ==="

# Accumulate pull/pip/credential failures so we can surface them LOUD at the
# end (per ~/Development/CLAUDE.md item 3 — fail-loud). Declared here (moved
# up from below the REPOS loop, alpha-engine-config-I9739) because the
# credential-helper assertion below is now the FIRST possible failure source
# in this script, and it must be able to increment the same counter/array
# every later failure uses rather than inventing a second mechanism.
PULL_FAILURES=0
FAILED_REPOS=()

# ── Refuse the long-lived PAT; assert the credential helper instead ────────
# alpha-engine-config-I9739 deliverable 3. This block used to hydrate the
# netrc file on every boot from /alpha-engine/GITHUB_TOKEN (a fine-grained
# PAT) so libcurl could authenticate git's HTTPS calls to GitHub — see this
# repo's git history for that block. It is gone, not merely disabled behind a
# flag, for the reason measured 2026-09-01 (alpha-engine-config-I9739):
#
# git sets CURLOPT_NETRC to CURL_NETRC_OPTIONAL unconditionally, so libcurl
# answers GitHub's 401 challenge from the netrc file BEFORE git ever consults
# a configured credential helper. The netrc file does not compete with
# `/usr/local/bin/git-credential-nousergon-app` (alpha-engine-config-I9628,
# nous-ergon-ops-PR962) — it outranks it categorically. A box that still has
# one from any earlier boot is a box where the helper is installed, provably
# able to mint, and NEVER consulted — proven inert on the dashboard box for
# every merge since the 2026-08-31 token rotation. `--check` passing is not
# evidence the helper is actually used; only removing the netrc file is.
#
# So every boot removes it unconditionally (not just when refreshing it),
# rather than leaving it in place until the next SSM read happens to fail.
NETRC="/home/ec2-user/.netrc"
if [ -e "$NETRC" ]; then
    rm -f "$NETRC"
    log "OK   removed the netrc file — the credential helper (git-credential-nousergon-app) is the only authentication path now"
fi

# The helper mints a per-use, short-lived GitHub App installation token from
# this box's own instance role (alpha-engine-executor-role already reads
# /alpha-engine/groom/github_app_* via alpha-engine-ssm-read — no IAM change,
# no operator step). A box with neither the helper nor the netrc file cannot
# authenticate at all, and that must not pass silently: the private-repo pull
# below (alpha-engine-config) would otherwise fail with an opaque 403 deep in
# the git-sync retry logic instead of a named, attributable cause here.
# `--check` mints and discards a real token and prints only its LENGTH, never
# the token itself, so this is safe to log in full.
HELPER="/usr/local/bin/git-credential-nousergon-app"
if [ ! -x "$HELPER" ]; then
    log "FAIL git credential helper missing/not executable at $HELPER — this box cannot authenticate to GitHub for private-repo pulls (netrc file removed above, no fallback by design)"
    PULL_FAILURES=$((PULL_FAILURES + 1))
    FAILED_REPOS+=("git-credential-helper (missing)")
elif ! "$HELPER" --check alpha-engine-config >> "$LOG" 2>&1; then
    log "FAIL git credential helper present but --check alpha-engine-config failed — this box cannot authenticate to GitHub for private-repo pulls"
    PULL_FAILURES=$((PULL_FAILURES + 1))
    FAILED_REPOS+=("git-credential-helper (check failed)")
elif ! sudo -u ec2-user -H git ls-remote --quiet \
        https://github.com/nousergon/alpha-engine-config.git HEAD >> "$LOG" 2>&1; then
    # OBSERVE-ONLY (sf-pipeline-policy.md §7a / SFP-7a-new-guard-observes-first).
    # Deliberately does NOT increment PULL_FAILURES yet — see the promotion
    # criterion below.
    log "OBSERVE git credential helper mints a token, but git itself cannot reach nousergon/alpha-engine-config — something is answering GitHub ahead of the helper (a credential dotfile, a url.insteadOf rewrite in .gitconfig, or an embedded credential in a remote URL). This probe is in observe mode and is NOT failing the run; the repo loop below is still the enforcing path."
else
    log "OK   git credential helper verified — mints (--check) AND git ls-remote reaches alpha-engine-config"
fi

# ── Promotion criterion for the two observe-mode probes in this block ─────
# sf-pipeline-policy.md §7a requires a newly added check whose verdict can
# halt a scheduled-pipeline stage to observe first, to carry its promotion
# criterion in its own module, and to be loud while observing. Both probes
# added by alpha-engine-config-I9829 are therefore OBSERVE-only on merge.
#
#   Promote both to FAIL (increment PULL_FAILURES, append to FAILED_REPOS)
#   when BOTH hold:
#     1. Ten consecutive trading-day boots have logged the probe's OK line
#        with no OBSERVE line, proving it does not false-positive on a
#        healthy box (the ls-remote probe reaches the network early in boot,
#        which is exactly where a transient failure would live).
#     2. alpha-engine-config-I9835 is CLOSED — the .gitconfig token literal
#        is gone from every box carrying the helper. Promoting the gitconfig
#        probe before then would fail boot-pull on every boot for a condition
#        already tracked and already known, which is a self-inflicted red
#        rather than a detection.
#
# Re-exam: 2026-09-16. If neither has promoted by then, the probes are
# either wrong or unowned; decide which rather than leaving them observing.

# ── The helper minting is not evidence the helper is USED ─────────────────
# alpha-engine-config-I9829. On 2026-09-02 `--check` passed on every boot
# while every real fetch of alpha-engine-config returned 403. The helper was
# installed, configured and provably able to mint a token whose installation
# covers exactly that repo — and GIT_TRACE showed git never invoked it once.
# git sets CURLOPT_NETRC unconditionally, so libcurl answered from the
# credential dotfile first; that credential is VALID but unpermitted, so
# GitHub replied 403 rather than 401, and with no 401 challenge git had no
# reason to consult a helper at all. A minting probe and the consumer path
# can therefore disagree indefinitely, and the minting probe is the one that
# looks healthy. Hence the ls-remote above: it exercises the path the repo
# loop actually uses, against the one private repo, before any repo is pulled.
#
# The same reasoning extends past the dotfile. A token embedded in a
# `url.<...>.insteadOf` rewrite in .gitconfig, or in a remote URL, outranks
# the helper the same way and survives the dotfile removal above untouched —
# measured on this box 2026-09-02 (alpha-engine-config-I9835), where a live
# fine-grained PAT sits in ~/.gitconfig for a personal-account repo. The
# rung-5 property this box is supposed to hold is "no long-lived GitHub
# secret on disk", and checking one of the several places such a secret can
# live reports a box clean while it is not.
# Both files, not one. The 2026-09-02 sweep found the same token literal in
# ec2-user's gitconfig AND in root's. boot-pull runs as root and reaches git
# through sudo in several places, so root's copy is equally load-bearing and
# equally invisible. A probe reading one of the two reports a box clean while
# it is not - the same shape as the --check defect above, one file down.
for GITCONFIG in /home/ec2-user/.gitconfig /root/.gitconfig; do
    [ -f "$GITCONFIG" ] || continue
    grep -qE '(gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}' "$GITCONFIG" 2>/dev/null || continue
    # Never echo the match. The finding is the file and the line number; the
    # value is a live credential and this log is read by agents.
    _gc_lines=$(grep -nE '(gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}' "$GITCONFIG" 2>/dev/null | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
    # OBSERVE-ONLY on merge - see the promotion criterion above.
    log "OBSERVE a long-lived GitHub token literal is present in $GITCONFIG (line(s) ${_gc_lines}) — it outranks the credential helper for any URL it matches and survives the dotfile removal above. Tracked as alpha-engine-config-I9835; this probe is in observe mode and is NOT failing the run."
done

# Weekday/EOD SF only SSM-invokes executor + data on this box; dashboard and
# backtester live on ae-dashboard / Saturday spots (see config#1767).
REPOS=(
    /home/ec2-user/alpha-engine-config
    /home/ec2-user/alpha-engine
    /home/ec2-user/alpha-engine-data
)

for repo in "${REPOS[@]}"; do
    if [ ! -d "$repo/.git" ]; then
        log "SKIP $repo (not cloned)"
        continue
    fi

    log "Pulling $repo ..."
    cd "$repo"

    # Ownership reclaim: a script inside the checkout run via sudo (e.g. a
    # timer-install step) can leave root-owned files/dirs in the ec2-user
    # checkout. `git reset --hard` then fails with "unable to unlink ...
    # Permission denied" and the box silently runs stale code until the
    # executor's deploy-drift preflight refuses — 40 min into the pipeline
    # (2026-07-06 incident, config#1811: infrastructure/ops/ was left
    # root-owned by upstream-gate-dryrun-validation's sudo install step,
    # failing every boot-pull that day). Detect-then-chown keeps the common
    # clean-boot path to a single find(1) scan.
    if [ -n "$(find "$repo" -not -user ec2-user -print -quit 2>/dev/null)" ]; then
        log "WARN $repo — foreign-owned files found in checkout; reclaiming (chown -R ec2-user)"
        sudo chown -R ec2-user:ec2-user "$repo" >> "$LOG" 2>&1 \
            || log "WARN $repo — chown reclaim failed; git reset may fail below"
    fi

    PREV_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" != "main" ]; then
        log "NOTE $repo — on branch '$CURRENT_BRANCH', will reset to origin/main (policy: boot always tracks main)"
    fi

    # Fix stale remote URL (post-org-transfer: cipher813/alpha-engine-config →
    # nousergon/alpha-engine-config). Only the config repo was cloned pre-transfer
    # and carries the old path; sibling repos point at nousergon/ directly.
    if [ "$repo" = "/home/ec2-user/alpha-engine-config" ]; then
        _CURRENT_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
        _EXPECTED_URL="https://github.com/nousergon/alpha-engine-config.git"
        if [ -n "$_CURRENT_URL" ] && [ "$_CURRENT_URL" != "$_EXPECTED_URL" ]; then
            log "FIX $repo — remote.origin.url corrected: $_CURRENT_URL -> $_EXPECTED_URL"
            git remote set-url origin "$_EXPECTED_URL" >> "$LOG" 2>&1 \
                || log "WARN $repo — remote set-url failed"
        fi
        unset _CURRENT_URL _EXPECTED_URL
    fi

    # The sync + its post-condition live in sync_repo_to_main() (defined at the
    # top of this file, where the full rationale is). It serializes the
    # index-mutating git ops behind the shared flock (config#1944) so this cannot
    # race the weekday CodeFreshnessGate / ChronicGapSelfHeal on .git/index.lock,
    # and it judges success on the post-condition rather than on git's exit code.
    # The ownership reclaim above runs OUTSIDE the lock deliberately (it is not a
    # git-index op; a test pins that the reclaim still precedes the sync).
    # The deploy gate that used to live here now runs AFTER the pip install
    # below — see the block at the end of this loop body for why the ordering
    # is the fix and not an incidental tidy-up.
    if sync_repo_to_main "$repo"; then
        SYNC_OK=1
        NEW_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")
    else
        SYNC_OK=0
        NEW_SHA="$PREV_SHA"
        PULL_FAILURES=$((PULL_FAILURES + 1))
        FAILED_REPOS+=("$repo (git)")
    fi

    # Update pip deps if venv + requirements.txt exist.
    # flow-doctor is now pulled in transitively via
    # nousergon-lib[flow-doctor]; the previous bundled editable
    # install (pip install -e /home/ec2-user/flow-doctor) has been
    # removed so boot doesn't silently overwrite the lib-provided copy
    # with a local dev branch.
    #
    # alpha-engine-data-config#1768 Phase 1: trading no longer runs the heavy
    # weekday/EOD data phases at all (those moved to ephemeral EC2 spot boxes
    # in config#1767 Phase 2, nousergon-data#643) — the ONLY thing this box's
    # alpha-engine-data checkout still needs a working venv for is the
    # metron-intraday collector (systemd timer, see the sync_systemd_units_from
    # call below), plus a manual/emergency SSM fallback of morning_enrich /
    # daily_append if ever needed. nousergon-data's own
    # infrastructure/data-trading-requirements.txt is the traced minimal
    # subset for exactly that scope (excludes arcticdb's RAG/backfill/
    # weekly-feature-engineer-only siblings: voyageai, edgartools, embit,
    # jsonschema, beautifulsoup4, feedparser — see that file's header for the
    # full trace) — full requirements.txt (~1.5GB, RAG + backfill + weekly
    # feature_engineer deps this box never runs) stays reserved for repos
    # that actually need it. REQUIREMENTS_FILE keyed on $repo rather than a
    # generic lookup table since alpha-engine-data is (for now) the only repo
    # in $REPOS with a non-default requirements filename; extend this
    # if/elif (not a full map) only if a second repo needs the same
    # treatment.
    REQUIREMENTS_FILE="requirements.txt"
    if [ "$repo" = "/home/ec2-user/alpha-engine-data" ] && [ -f "infrastructure/data-trading-requirements.txt" ]; then
        REQUIREMENTS_FILE="infrastructure/data-trading-requirements.txt"
    fi
    if [ -f ".venv/bin/pip" ] && [ -f "$REQUIREMENTS_FILE" ]; then
        if .venv/bin/pip install --quiet -r "$REQUIREMENTS_FILE" >> "$LOG" 2>&1; then
            log "OK   $repo — deps updated (from $REQUIREMENTS_FILE)"
        else
            log "FAIL $repo — pip install failed (from $REQUIREMENTS_FILE)"
            PULL_FAILURES=$((PULL_FAILURES + 1))
            FAILED_REPOS+=("$repo (pip)")
        fi
    fi

    # ── Deploy gate: import smoke test ────────────────────────────────────
    # Catches syntax errors and transitive ImportErrors; no IB Gateway
    # connection needed. Imports pull the full transitive module graph, so a
    # broken dependency in any executor module surfaces here pre-planner
    # rather than at runtime.
    #
    # It runs AFTER the pip install above, and that ordering IS the fix
    # (alpha-engine-config-I8682). It used to sit inside the sync branch, so it
    # tested the NEW commit's code against the PREVIOUS commit's installed
    # dependency set. Any commit that bumps a pin and uses the new API failed
    # this gate by construction, was rolled back, and then had the rolled-back
    # commit's requirements.txt reinstalled over it — a stable loop with no
    # exit, re-run identically at every boot. Measured 2026-08-25/26 on the
    # trading box: #493 (krepis 0.54.0 -> 0.59.33, plus `from
    # krepis.trading_calendar import is_market_hours` in executor/market_hours.py)
    # rolled back on every boot with `ImportError: cannot import name
    # 'is_market_hours'`, and the 2026-08-26 preopen died at
    # CodeFreshnessGate's CODE-STALE-AFTER-HEAL because that gate heals forward
    # to origin/main while this rollback pushes back — the two are opposed
    # writers and the outcome was decided by which finished last.
    #
    # A rollback is a FAILURE and is counted as one. It previously neither
    # incremented PULL_FAILURES nor alerted, which made the one path that
    # knowingly leaves the trading box on stale code the only path in this
    # script with no signal at all: on 2026-08-25 the preopen went green and
    # the session traded #492 while origin/main was #493, silently.
    if [ "$SYNC_OK" -eq 1 ] && [ "$repo" = "/home/ec2-user/alpha-engine" ] \
            && [ "$PREV_SHA" != "$NEW_SHA" ]; then
        if [ -f ".venv/bin/python" ] && [ -f "executor/main.py" ]; then
            log "GATE $repo — running import smoke test (post-deps)..."
            if .venv/bin/python -c "import executor.main, executor.daemon, executor.eod_reconcile" >> "$LOG" 2>&1; then
                log "OK   $repo — import smoke test passed"
            else
                log "FAIL $repo — import smoke test failed, rolling back to $PREV_SHA"
                # Under the SAME advisory lock every other git writer on this
                # box takes (config#1944), so a rollback cannot land between the
                # weekday CodeFreshnessGate's heal and its re-check.
                flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" \
                    git -C "$repo" reset --hard "$PREV_SHA" >> "$LOG" 2>&1 \
                    || log "WARN $repo — rollback reset failed; tree left at $NEW_SHA"
                # Deps were just installed from $NEW_SHA's manifest. Restore the
                # set the reverted code was actually tested against, or the box
                # is left on a code/dependency combination nothing has ever run.
                if [ -f ".venv/bin/pip" ] && [ -f "$REQUIREMENTS_FILE" ]; then
                    .venv/bin/pip install --quiet -r "$REQUIREMENTS_FILE" >> "$LOG" 2>&1 \
                        || log "WARN $repo — post-rollback pip install failed; venv may not match $PREV_SHA"
                fi
                log "ROLLBACK $repo — reverted to $(git log --oneline -1)"
                PULL_FAILURES=$((PULL_FAILURES + 1))
                FAILED_REPOS+=("$repo (deploy-gate rollback)")
            fi
        fi
    fi
done

# ── Restore trades.db from S3 if missing or empty ──────────────────────────
# The trading instance is stopped/started daily by EventBridge. If a new
# instance is launched (or EBS is recreated), trades.db will be missing and
# init_db() creates an empty one — losing all trade history and EOD data.
# This restores trades_latest.db from S3 as a safety net.
RISK_YAML="/home/ec2-user/alpha-engine/config/risk.yaml"
if [ -f "$RISK_YAML" ]; then
    # Parse db_path and trades_bucket from risk.yaml. `\s` is a GNU-sed-only
    # shorthand — the deploy target (Amazon Linux EC2) and CI (ubuntu-latest)
    # both use GNU sed so this masked the bug there, but BSD sed (macOS, every
    # local dev machine) treats `\s` as a literal "s", leaving a stray leading
    # space/trailing quote in the parsed value. `[[:space:]]` is the POSIX
    # class both sed implementations support identically.
    DB_PATH=$(grep -E '^\s*db_path:' "$RISK_YAML" | head -1 | sed 's/.*db_path:[[:space:]]*["]*\([^"]*\)["]*[[:space:]]*/\1/' | tr -d "'\"")
    TRADES_BUCKET=$(grep -E '^\s*trades_bucket:' "$RISK_YAML" | head -1 | sed 's/.*trades_bucket:[[:space:]]*["]*\([^"]*\)["]*[[:space:]]*/\1/' | tr -d "'\"")

    if [ -n "$DB_PATH" ] && [ -n "$TRADES_BUCKET" ]; then
        S3_KEY="trades/trades_latest.db"
        RESTORE_NEEDED=false

        if [ ! -f "$DB_PATH" ]; then
            log "trades.db missing — restoring from s3://${TRADES_BUCKET}/${S3_KEY}"
            RESTORE_NEEDED=true
        else
            DB_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
            if [ "$DB_SIZE" -le 20480 ]; then
                log "trades.db empty or minimal (${DB_SIZE}B) — restoring from S3"
                RESTORE_NEEDED=true
            else
                # Check for staleness: compare local max(eod_pnl.date) against the
                # expected most-recent trading day, not just presence of any row.
                #
                # config#2356: the presence-only check ("does eod_pnl have ANY
                # row") cannot catch a snapshot-restored stale-but-large DB — a
                # box restored from a 3-week-old EBS snapshot has plenty of
                # non-null eod_pnl rows (just old ones) and would incorrectly
                # report "has recent data". Compute the expected floor date via
                # nousergon_lib.trading_calendar (same helper executor/main.py's
                # ArcticDB freshness gate and executor/eod_reconcile.py use) and
                # require LOCAL_MAX_DATE to be within 1 trading day of the last
                # closed session — that tolerance absorbs the case where today's
                # EOD ingestion simply hasn't run yet, without masking a
                # genuinely stale restore (2+ trading days behind triggers
                # restore).
                #
                # AE_VENV_PY must point at the alpha-engine venv set up earlier
                # in THIS script (the .venv/bin/pip install -r requirements.txt
                # step above, which runs before this block) so nousergon_lib is
                # importable — mirrors the FD_VENV precedent below in this same
                # file. Falls back to bare python3 (best-effort: sqlite parsing
                # still works without nousergon_lib, only the recency half of
                # the check is skipped) if the venv isn't present yet.
                AE_VENV_PY="/home/ec2-user/alpha-engine/.venv/bin/python"
                if [ ! -x "$AE_VENV_PY" ]; then
                    AE_VENV_PY="python3"
                fi

                LOCAL_MAX_DATE=$("$AE_VENV_PY" - "$DB_PATH" <<'PYEOF' 2>/dev/null || echo ""
import sqlite3
import sys

db_path = sys.argv[1]
try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT MAX(date) FROM eod_pnl")
    result = c.fetchone()
    conn.close()
    print(result[0] if result and result[0] else "")
except Exception:
    print("", file=sys.stderr)
    sys.exit(1)
PYEOF
)

                if [ -z "$LOCAL_MAX_DATE" ]; then
                    log "trades.db has no eod_pnl data — restoring from S3"
                    RESTORE_NEEDED=true
                else
                    # Floor date: last_closed_trading_day() minus 1 trading day
                    # of tolerance. LOCAL_MAX_DATE older than this floor means
                    # the DB is missing at least 2 trading days of EOD data —
                    # the "stale-but-large" snapshot-restore scenario.
                    STALENESS_FLOOR=$("$AE_VENV_PY" <<'PYEOF' 2>/dev/null || echo ""
try:
    from nousergon_lib.trading_calendar import last_closed_trading_day, subtract_trading_days
    print(subtract_trading_days(last_closed_trading_day(), 1).isoformat())
except Exception:
    print("")
PYEOF
)
                    if [ -n "$STALENESS_FLOOR" ] && [ "$LOCAL_MAX_DATE" \< "$STALENESS_FLOOR" ]; then
                        log "trades.db eod_pnl is stale (max_date=$LOCAL_MAX_DATE < floor=$STALENESS_FLOOR) — restoring from S3"
                        RESTORE_NEEDED=true
                    elif [ -z "$STALENESS_FLOOR" ]; then
                        log "WARN could not compute staleness floor (nousergon_lib.trading_calendar unavailable) — falling back to presence-only check (max_date=$LOCAL_MAX_DATE)"
                    else
                        log "trades.db has recent data (max_date=$LOCAL_MAX_DATE, floor=$STALENESS_FLOOR) — no restore needed"
                    fi
                fi
            fi
        fi

        if [ "$RESTORE_NEEDED" = "true" ]; then
            if aws s3 cp "s3://${TRADES_BUCKET}/${S3_KEY}" "$DB_PATH" >> "$LOG" 2>&1; then
                NEW_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
                log "OK   trades.db restored (${NEW_SIZE}B)"
            else
                log "WARN trades.db restore failed — executor will start with empty db"
                exit 1
            fi
        fi
    else
        log "WARN could not parse db_path/trades_bucket from risk.yaml"
    fi
else
    log "SKIP trades.db restore (no risk.yaml)"
fi

# ── Self-heal boot-pull-launcher.sh at /usr/local/sbin ──────────────────────
# alpha-engine-config-I9444: boot-pull.sh already self-heals its OWN unit
# file — sync_systemd_units_from() below re-copies boot-pull.service into
# /etc/systemd/system on every boot — but had no equivalent self-heal for the
# launcher BINARY at /usr/local/sbin/boot-pull-launcher.sh. That path is a
# two-artifact install (unit file + out-of-tree exec target, config-I8734)
# where only one artifact self-healed; install-boot-pull.sh provisions the
# launcher once, by hand, over SSM, and nothing re-verified it afterward.
#
# Measured 2026-08-31 on the trading box (i-018eb3307a21329bf): the launcher
# was never provisioned there when PR #495 shipped, and boot-pull.service
# FAILed at every boot from 2026-08-28 through 2026-08-31 with
# `status=203/EXEC: Failed to locate executable
# /usr/local/sbin/boot-pull-launcher.sh` — invisible except two levels
# downstream, via a systemd-unit-drift alert on unrelated units this box's
# normal boot-pull cycle happens to also update. Remediated live via the
# existing install-boot-pull.sh; this closes the underlying gap so a future
# occurrence self-heals on the box's own next boot instead of waiting on a
# human to notice and re-run the installer by hand.
#
# Verify+repair on EVERY run (not just first-install) using a byte-for-byte
# comparison rather than a bare existence check, so a stale launcher left
# over from an older commit is also repaired, not only a missing one.
# `install -m 0755 -o root -g root` exactly mirrors install-boot-pull.sh's
# own mode/owner, so a box that has never run install-boot-pull.sh still
# converges here using the same sudo surface this script already relies on
# for the systemd unit copies below. Best-effort by design (matches the
# ~/.netrc refresh pattern above): a missing $LAUNCHER_SRC means the
# alpha-engine checkout itself is incomplete, which the repo pull loop above
# already surfaces loudly — this block only WARNs and skips rather than
# duplicating that failure.
LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"
LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"
if [ -f "$LAUNCHER_SRC" ]; then
    if [ ! -f "$LAUNCHER_DST" ] || ! cmp -s "$LAUNCHER_SRC" "$LAUNCHER_DST"; then
        if sudo install -m 0755 -o root -g root "$LAUNCHER_SRC" "$LAUNCHER_DST" >> "$LOG" 2>&1; then
            log "OK   boot-pull-launcher: repaired $LAUNCHER_DST (src=$LAUNCHER_SRC)"
        else
            log "FAIL boot-pull-launcher: repair of $LAUNCHER_DST failed"
            PULL_FAILURES=$((PULL_FAILURES + 1))
            FAILED_REPOS+=("boot-pull-launcher (install)")
        fi
    else
        log "OK   boot-pull-launcher: $LAUNCHER_DST matches repo"
    fi
else
    log "WARN boot-pull-launcher: source $LAUNCHER_SRC not found — skipping self-heal (repo pull failure above already surfaces this loudly)"
fi

# Sync systemd service files from repo (if alpha-engine has them).
#
# Installs NEW units as well as updating existing ones. Prior versions of
# this block only touched units already in /etc/systemd/system/, which
# silently skipped newly-added repo units (e.g. alpha-engine-daily-data.timer
# added in PR #62 sat on disk for days before this was noticed). After a
# new .timer is copied and daemon-reload'd, it's enabled + started so the
# next boot picks it up and the current boot activates it immediately.
#
# sync_systemd_units_from() (config#2352): factored out so this same
# install/update/orphan/enable-reconcile logic can run a SECOND pass against
# nousergon-data's infrastructure/systemd/ below — that repo's
# metron-intraday.{service,timer} were version-tracked but boot-pull never
# looked at them, so a merged unit edit silently never took effect until an
# operator remembered to re-run install-metron-intraday.sh over SSM
# (2026-07-13 operator ruling: "queue on merge, apply on next boot" — THIS
# boot-pull pass is that queue's drain point for the trading box, which is
# off most of the day and can't reliably receive a merge-time SSM push).
# $1: source dir holding *.service/*.timer. $2: space-separated EXCLUDE list
# of exact basenames to skip entirely (install AND enable-reconcile) — for
# a source dir that ships MULTIPLE repos' unit families where one family no
# longer belongs on THIS box (config#1768 Phase 1: metron-intraday moved off
# trading onto ae-dashboard, but nousergon-data's infrastructure/systemd/
# still ships its unit files for ae-dashboard's OWN sync pass to pick up —
# the install loop below globs *.service/*.timer unconditionally, so
# without this exclude it would keep re-installing + re-enabling
# metron-intraday on trading every boot regardless of the orphan-prefix
# args, since the orphan pass only fires when a unit is ABSENT from
# $SYSTEMD_SRC, not merely unwanted on this box). Pass "" for no excludes.
# $3+: one or more glob prefixes for orphan reconciliation (keeps each
# repo's orphan sweep scoped to units IT owns — alpha-engine's sweep must
# never disable/remove a nousergon-data unit or vice versa; multiple
# prefixes let one source dir cover unit families that don't share a common
# prefix, e.g. nousergon-data ships both "metron-intraday.*" and
# "systemd-unit-drift-check.*").
sync_systemd_units_from() {
    local SYSTEMD_SRC="$1"
    shift
    local EXCLUDE_BASENAMES="$1"
    shift
    local ORPHAN_GLOB_PREFIXES=("$@")
    [ -d "$SYSTEMD_SRC" ] || return 0

    _sync_is_excluded() {
        local candidate="$1"
        local excl
        for excl in $EXCLUDE_BASENAMES; do
            [ "$candidate" = "$excl" ] && return 0
        done
        return 1
    }

    local CHANGED=false
    for unit in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        local name target
        name=$(basename "$unit")
        if _sync_is_excluded "$name"; then
            continue
        fi
        target="/etc/systemd/system/$name"
        if [ ! -f "$target" ]; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: installed $name (new, src=$SYSTEMD_SRC)"
            CHANGED=true
        elif ! diff -q "$unit" "$target" >/dev/null 2>&1; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: updated $name (src=$SYSTEMD_SRC)"
            CHANGED=true
        fi
    done
    # Orphan reconciliation: any installed unit matching $ORPHAN_GLOB_PREFIX
    # without a corresponding source in $SYSTEMD_SRC was removed from that
    # repo and must be retired here. Disable + remove. Safety: only matches
    # the given prefix, never touches unrelated system units (including the
    # OTHER repo's tracked units — each call site passes its own prefix).
    #
    # 2026-04-28: closes the asymmetry where adding a new timer was
    # self-healing (install/update/enable handled) but retiring one was
    # not — the systemd file lingered on disk and continued firing even
    # after deletion from the repo. After this pass, removing a unit
    # file from `infrastructure/systemd/` is the canonical retirement
    # path: next boot disables + deletes it.
    for prefix in "${ORPHAN_GLOB_PREFIXES[@]}"; do
        for installed in /etc/systemd/system/${prefix}*.service /etc/systemd/system/${prefix}*.timer; do
            [ -f "$installed" ] || continue  # glob may match nothing
            local oname
            oname=$(basename "$installed")
            if [ ! -f "$SYSTEMD_SRC/$oname" ]; then
                sudo systemctl disable --now "$oname" >> "$LOG" 2>&1 || true
                sudo rm -f "$installed"
                log "OK   systemd: orphan removed $oname (no longer in $SYSTEMD_SRC)"
                CHANGED=true
            fi
        done
    done

    if $CHANGED; then
        sudo systemctl daemon-reload
        log "OK   systemd: daemon-reload"
    fi

    # Reconcile timer enable state. Every timer shipped in the repo
    # MUST be enabled. `systemctl enable` is idempotent on already-
    # enabled timers, and recreates the `timers.target.wants/` symlink
    # on any timer that was disabled or whose symlink was lost. Fixes
    # three failure modes the prior new-install-only enable path could
    # not recover from:
    #
    # 1. Manual `systemctl disable` (intentional debugging, accidental).
    # 2. EBS volume state where unit files exist on disk but
    #    `timers.target.wants/` is empty.
    # 3. New timers added to `infrastructure/systemd/` after the
    #    one-shot `setup-trading-ec2.sh` run already completed — they
    #    get `cp`'d by boot-pull but the previous "enable only on first
    #    install" branch missed them because the target file already
    #    existed.
    #
    # 2026-04-21 SNDK EOD incident: `alpha-engine-eod.timer` was
    # disabled for an unknown reason. boot-pull's previous logic only
    # enabled timers inside the `[ ! -f "$target" ]` branch, so the
    # disabled state persisted for two boots. EOD emails silently
    # stopped firing until manual SSM intervention re-enabled the
    # timer. This reconciliation pass makes every boot self-healing
    # for timer-enable drift.
    for unit in "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        local tname
        tname=$(basename "$unit")
        if _sync_is_excluded "$tname"; then
            continue
        fi
        if sudo systemctl enable --now "$tname" >> "$LOG" 2>&1; then
            log "OK   systemd: enable reconciled $tname"
        else
            log "WARN systemd: enable reconcile failed: $tname"
        fi
    done
    unset -f _sync_is_excluded
}

sync_systemd_units_from "/home/ec2-user/alpha-engine/infrastructure/systemd" "" "alpha-engine-"

# nousergon-data's systemd units (metron-intraday.{service,timer} +
# config#2352's own systemd-unit-drift-check.{service,timer}) — see
# sync_systemd_units_from's comment above for why this second pass exists.
# Two orphan prefixes since nousergon-data's unit names don't share a common
# "alpha-engine-*"-style prefix: glob on the two exact basenames this repo
# ships instead of a wildcard prefix, so a future unrelated unit hand-placed
# on this box for debugging is never swept as an "orphan" of this repo.
#
# config#1768 Phase 1 (2026-07-21): metron-intraday MOVED off trading onto
# ae-dashboard (config#1768 workstream 2) — ae-dashboard now runs it
# (intraday price alerts Lambda already covers the duplicate work this was
# doing here; see that issue). nousergon-data's infrastructure/systemd/ dir
# is shared by BOTH boxes' sync passes (ae-dashboard's own boot-pull now
# also points at this same source dir for its metron-intraday pull), so
# metron-intraday.{service,timer} must stay in the exclude list here
# PERMANENTLY, not just as a one-time cleanup — without it, the very next
# nousergon-data merge touching either unit file would re-install +
# re-enable it right back onto trading via the reconcile loop above
# (orphan-removal alone can't catch this: the unit is NOT absent from
# $SYSTEMD_SRC, it's merely unwanted on THIS box).
#
# daily-news.{service,timer} are excluded for the same reason (alpha-engine-
# config-I11036). nousergon-data ships them for the DASHBOARD box
# (deploy-daily-news-units.yml installs them there), but this pass installed
# and `enable --now`-ed them here too. With `Persistent=true` on a 04:00 PT
# calendar and a box that is stopped overnight, the timer fires a catch-up run
# shortly after every weekday boot. Measured in the CloudTrail archive
# 2026-09-29..10-02: the dashboard box wrote data/news_{aggregates,articles,
# digest}_daily/ at ~11:21Z, then this box rewrote the same keys, latest.json
# included, at ~12:36Z as alpha-engine-executor-role. That is a second
# producer of a collection artifact, running on the trading box.
sync_systemd_units_from "/home/ec2-user/alpha-engine-data/infrastructure/systemd" "metron-intraday.service metron-intraday.timer daily-news.service daily-news.timer" "systemd-unit-drift-check"

# One-time (idempotent) self-heal for boxes that already had metron-intraday
# installed+enabled from before the exclude above existed: the exclude only
# stops FUTURE install/enable/re-enable, it does not retroactively touch a
# unit already present in /etc/systemd/system/, since sync_systemd_units_from
# only acts on units it manages (installs, updates, or orphan-removes ones
# absent from $SYSTEMD_SRC — metron-intraday is neither, now that it's
# excluded rather than removed from the source dir). `|| true` on every step:
# this must never fail boot-pull itself, and each systemctl call is already
# a no-op if the unit doesn't exist / isn't loaded. Safe to leave in
# permanently (config#1768's own closes-when checks `systemctl is-active
# metron-intraday` is inactive/masked on ae-trading — this is that
# self-healing path for this environment, which cannot verify it live).
#
# 2026-08-13 (alpha-engine-config-I6960): this used to DISABLE and stop there,
# so the unit FILES stayed on disk — at the version trading had in July 2026,
# while nousergon-data's copy moved on to the dashboard box's shape
# (`.venv-intraday`, MemoryHigh/Max). check-systemd-unit-drift then reported
# `metron-intraday.service: installed (3c6a241e) != repo (082a1ab8)` every
# single day, correctly and forever: a decommissioned unit cannot converge on a
# source it is excluded from syncing with. Disabling is half a retirement.
# Removing the file is the other half, and it is what makes the drift finding
# go away by being TRUE rather than by being silenced.
#
# The old condition was also always-true — `systemctl list-unit-files <name>`
# exits 0 with "0 unit files listed" when nothing matches — so the NOTE printed
# on every boot of every box regardless. Test the files.
#
# daily-news.{timer,service} take the same path (alpha-engine-config-I11036):
# excluding them above stops future installs, and this removes the copies the
# sync pass already installed.
_dashboard_only_leftovers=()
for _u in metron-intraday.timer metron-intraday.service daily-news.timer daily-news.service; do
    [ -f "/etc/systemd/system/$_u" ] && _dashboard_only_leftovers+=("$_u")
done
if [ ${#_dashboard_only_leftovers[@]} -gt 0 ]; then
    log "NOTE dashboard-box unit(s) left on trading: ${_dashboard_only_leftovers[*]} — retiring (config#1768 metron-intraday, alpha-engine-config-I11036 daily-news: both run on ae-dashboard)"
    sudo systemctl disable --now "${_dashboard_only_leftovers[@]}" 2>> "$LOG" || true
    for _u in "${_dashboard_only_leftovers[@]}"; do
        sudo rm -f "/etc/systemd/system/$_u" && log "OK   systemd: retired $_u (runs on ae-dashboard, not trading)"
    done
    sudo systemctl daemon-reload
fi
unset _dashboard_only_leftovers _u

# Config files are now in the alpha-engine-config private repo (pulled above).
# Each module's config loader searches ~/alpha-engine-config/ first.

# ── Report failures to flow-doctor if any occurred ──────────────────────────
# Mirrors the dashboard box's boot-pull.sh (alpha-engine-dashboard/
# infrastructure/) — closes ROADMAP L4490, the asymmetry where a private-repo
# fetch failure on THIS box was WARN-only-in-the-log (invisible) while the
# dashboard box raised a flow-doctor badge + email. flow-doctor.yaml here is
# flow_name=executor → reports to nousergon/crucible-executor + emails the owner.
# Fire-and-forget (`|| true`): if flow-doctor itself is broken, the FAIL lines
# in /var/log/boot-pull.log are the fallback signal, and we still exit 1.
if [ "$PULL_FAILURES" -gt 0 ]; then
    log "=== boot-pull completed with $PULL_FAILURES failure(s): ${FAILED_REPOS[*]} ==="

    # This used to construct flow-doctor by hand in a heredoc. It had been
    # BROKEN for an unknown length of time and nobody knew, because the thing
    # that was broken was the failure reporter itself
    # (alpha-engine-config-I4509). Two independent faults, either one fatal:
    #
    #   1. `flow_doctor.init()` does not exist. flow-doctor 0.8.7 exports
    #      FlowDoctor/FlowDoctorBuilder and no `init`; the call raised
    #      AttributeError every time, leaving one stderr line in a log nobody
    #      reads.
    #   2. The env hydration was incomplete anyway — flow-doctor.yaml
    #      references more ${VAR}s than the heredoc hydrated, so even with the
    #      API call fixed, construction fails with ConfigError.
    #
    # This matters more here than on the dashboard box: THIS is the trading
    # box, boot-pull is what guarantees it runs current code on a trading day,
    # and it runs once at boot on an instance that only exists during the
    # trading window — so a failure has a one-day blast radius and then the
    # evidence disappears with the instance. It was observed failing for 3+
    # days straight with zero signal.
    #
    # `krepis.alerts` is the canonical alert CLI (config#1649) and resolves its
    # own secrets, so there is no hydration list here to drift.
    ALERT_PY="/home/ec2-user/alpha-engine/.venv/bin/python"
    if [ -x "$ALERT_PY" ]; then
        # The key carries the UTC date. alpha-engine-config-I9829: without it,
        # a failure that recurs at the box's once-a-day boot cadence collides
        # with its own previous publish inside the 1440-minute window and is
        # suppressed. Measured — boot-pull failed on every boot from 08-31
        # 22:14 onward, and the alert published on 09-02 only:
        #
        #   08-31 22:14  dedup_skipped=True
        #   08-31 22:29  dedup_skipped=True
        #   09-01 12:16  dedup_skipped=True   <- the day preopen lost a session
        #   09-02 12:16  sns.ok=True          <- first and only publish
        #
        # A 24-hour window against a 24-hour cadence is a coin flip on clock
        # drift, and it lands on "suppress" exactly when the failure is
        # persistent rather than transient — the case the operator most needs
        # to see. Dating the key states the intent the window was standing in
        # for: at most one alert per distinct failure per day, however many
        # times the box boots, and never a day of silence because yesterday
        # already reported it.
        _dkey="boot-pull-trading-$(date -u +%Y-%m-%d)-$(printf '%s' "${FAILED_REPOS[*]}" | tr ' /' '__' | cut -c1-72)"
        "$ALERT_PY" -m krepis.alerts publish \
            --message "boot-pull FAILED on the TRADING box: ${PULL_FAILURES} repo(s) could not be updated — ${FAILED_REPOS[*]}. The executor may be running stale code for today's session. See /var/log/boot-pull.log." \
            --severity error \
            --source boot-pull \
            --dedup-key "$_dkey" \
            --dedup-window-min 1440 \
            || log "ALERT PUBLISH FAILED — boot-pull failure is UNREPORTED"
    else
        log "ALERT PUBLISH SKIPPED — $ALERT_PY missing; boot-pull failure is UNREPORTED"
    fi

    log "=== boot-pull complete (with failures) ==="
    exit 1
fi

log "=== boot-pull complete ==="
