"""Page when a trading-path merge lands inside the 09:00-16:30 ET window.

alpha-engine-config-I12640, option 2 (the post-merge detector). The rule
(change-control-policy § Merge permission; Brian 2026-10-16) is that a
trading-path merge never lands between 09:00 and 16:30 America/New_York on
an NYSE trading day (the regular session plus 30 minutes each side;
restated on the exchange clock by alpha-engine-config-I12769). Until this existed it was enforced only by the session doing
the merge, and nothing flagged crucible-executor#621 merging at 20:12Z on
Fri 2026-10-23.

This DETECTS; it does not prevent. Prevention at merge time is option 1
(a required check under a merge queue), a repo-admin change tracked on the
same issue.

The window is read on the exchange clock, never as a fixed UTC range: it
is 13:00-20:30Z under EDT and 14:00-21:30Z under EST. The fixed 13:30-20:30Z
it replaced was NOT wider than the session in both regimes — under EST it
missed 20:30-21:00Z, market time (alpha-engine-config-I12769).
Whether a day counts is krepis.trading_calendar.is_trading_day, the same
holiday table the executor trades on.

Usage (CI)::

    python scripts/trading_window_merge_check.py \\
        --repo nousergon/crucible-executor --repo nousergon/crucible-trader \\
        --lookback-minutes 70 [--publish]

Without ``--publish`` it only prints what it would page (the default, so a
local run sends nothing). Exit 0 when nothing is in-window or every page
was delivered; 1 when a page could not be delivered or the commit listing
could not be read, so the job goes red rather than going quiet.

With ``--publish`` the Telegram channel is part of the contract, checked on
EVERY run rather than only on the day a merge lands in the window. On
2026-10-26 crucible-executor#627 merged at 15:04Z, this detector saw it
(run 40345778242), and the page reached SNS only: Telegram logged
``TELEGRAM_BOT_TOKEN=MISSING TELEGRAM_CHAT_ID=MISSING``, ``publish()`` still
counted SNS as delivered, and the job finished green. So a run exits 1 when
either Telegram value does not resolve (``krepis.secrets.get_secret``: SSM
``/alpha-engine/<NAME>`` first, then the environment — the same resolution
``krepis.telegram`` sends with), and a page counts as delivered only when
its Telegram leg is ``ok``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

WINDOW_TZ = ZoneInfo("America/New_York")
WINDOW_START_ET = time(9, 0)
WINDOW_END_ET = time(16, 30)

ALERT_SOURCE = "crucible-executor/scripts/trading_window_merge_check.py"
ISSUE = "alpha-engine-config-I12640"


@dataclass(frozen=True)
class Merge:
    repo: str
    sha: str
    landed_at: datetime  # committer date: the merge time for a squash or merge commit
    title: str


#: The two values krepis.telegram sends with. Names only — a value is never
#: printed, returned or logged by this module.
TELEGRAM_VALUE_NAMES = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def missing_telegram_values(*, get_secret=None) -> list[str]:
    """Names of the Telegram values that do not resolve, in TELEGRAM_VALUE_NAMES order.

    Resolved through ``krepis.secrets.get_secret`` (SSM first, then the
    environment), which is exactly what ``krepis.telegram`` does when it
    sends, so an empty list here means the page has a channel.
    """
    if get_secret is None:
        from krepis.secrets import get_secret
    return [name for name in TELEGRAM_VALUE_NAMES if not get_secret(name, required=False)]


def page_reached_telegram(result) -> bool:
    """True iff ``publish()``'s result shows the page on Telegram.

    A dedup skip means an earlier run already paged this sha, so it counts.
    SNS alone does not: that is the 2026-10-26 green-but-dropped run.
    """
    if getattr(result, "dedup_skipped", False):
        return True
    telegram = getattr(result, "telegram", None)
    return bool(getattr(telegram, "ok", False))


def in_trading_window(ts: datetime, *, is_trading_day=None) -> bool:
    """True iff ``ts`` (tz-aware) is in [09:00, 16:30) America/New_York on a trading day."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware")
    if is_trading_day is None:
        from krepis.trading_calendar import is_trading_day
    et = ts.astimezone(WINDOW_TZ)
    if not (WINDOW_START_ET <= et.time() < WINDOW_END_ET):
        return False
    return bool(is_trading_day(et.date()))


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _gh_api(path: str, params: dict[str, str]) -> list:
    """GET ``path`` through the ``gh`` CLI, which reads its own token
    (``GH_TOKEN`` in the workflow) — this module never handles one."""
    cmd = ["gh", "api", "-X", "GET", path]
    for key, value in params.items():
        cmd += ["-f", f"{key}={value}"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)  # noqa: S603
    return json.loads(out.stdout)


def list_merges(repo: str, since: datetime, *, api=_gh_api) -> list[Merge]:
    """Commits on ``main`` of ``repo`` committed at or after ``since``."""
    rows = api(
        f"repos/{repo}/commits",
        {"sha": "main", "per_page": "100", "since": since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
    )
    merges = []
    for row in rows:
        commit = row["commit"]
        merges.append(
            Merge(
                repo=repo,
                sha=row["sha"],
                landed_at=_parse_ts(commit["committer"]["date"]),
                title=(commit.get("message") or "").splitlines()[0][:120],
            )
        )
    return merges


def in_window_merges(merges: list[Merge], *, is_trading_day=None) -> list[Merge]:
    return [m for m in merges if in_trading_window(m.landed_at, is_trading_day=is_trading_day)]


def alert_text(m: Merge) -> tuple[str, str, str]:
    """(message, auto_fix, resolution) for one in-window merge."""
    message = (
        f"Trading-path merge inside the 09:00-16:30 ET window: {m.repo}@{m.sha[:9]} landed "
        f"{m.landed_at.astimezone(UTC):%Y-%m-%d %H:%MZ} ({m.title}). The run rule forbids "
        "trading-path merges in this window on NYSE trading days. Check whether the trading box "
        "can pick it up before the close (deploy-on-merge or a mid-day boot-pull) and decide "
        "whether to revert."
    )
    auto_fix = "none — a merge cannot be undone automatically; a human decides whether to revert"
    resolution = (
        f"{ISSUE} — merge-time refusal (required trading-window check + merge queue) is option 1 "
        "on that issue; this page is the detector until it lands"
    )
    return message, auto_fix, resolution


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", action="append", required=True, help="owner/name; repeatable")
    ap.add_argument("--lookback-minutes", type=int, default=70)
    ap.add_argument("--publish", action="store_true", help="send the page (default: print only)")
    args = ap.parse_args(argv)

    now = datetime.now(UTC)
    since = now - timedelta(minutes=args.lookback_minutes)

    # Checked before any listing and on every --publish run, so a missing
    # channel turns the hourly run red on a quiet day too, not only on the
    # day an in-window merge needs the page.
    channel_missing = missing_telegram_values() if args.publish else []
    if channel_missing:
        print(
            # A fixed literal: nothing derived from the secret lookup is
            # printed, not even which name missed (CodeQL py/clear-text-logging).
            "::error::Telegram channel missing for the trading-window page: TELEGRAM_BOT_TOKEN "
            "and/or TELEGRAM_CHAT_ID did not resolve (SSM /alpha-engine/<NAME>, then the environment). "
            "Auto-fix: none — the workflow reads both values from SSM under the read-only role "
            "github-actions-trading-window-page; check that role exists and its two reader steps "
            f"succeeded. Resolution: {ISSUE}"
        )

    found: list[Merge] = []
    unreadable = []
    for repo in args.repo:
        try:
            found.extend(in_window_merges(list_merges(repo, since)))
        except Exception as exc:  # noqa: BLE001 — reported and turns the job red
            unreadable.append(f"{repo}: {type(exc).__name__}: {exc}")

    for line in unreadable:
        print(f"::error::could not list merges for {line}")
    if not found:
        print(f"no trading-path merge inside the window since {since:%Y-%m-%dT%H:%MZ} ({', '.join(args.repo)})")
        return 1 if (unreadable or channel_missing) else 0

    failed = 0
    for m in found:
        message, auto_fix, resolution = alert_text(m)
        print(f"::warning::{message}")
        if not args.publish:
            continue
        from krepis import alerts

        try:
            result = alerts.publish(
                message,
                severity="error",
                # A literal, not ALERT_SOURCE: the fleet alert-class guard
                # registers publish sources by literal (playbooks.yaml row
                # executor_trading_window_merge). The test ties the two.
                source="crucible-executor/scripts/trading_window_merge_check.py",
                dedup_key=f"trading-window-merge-{m.repo}-{m.sha}",
                dedup_window_min=None,
                auto_fix=auto_fix,
                resolution=resolution,
            )
        except Exception as exc:  # noqa: BLE001 — every channel failed; the job must go red
            print(f"::error::page for {m.repo}@{m.sha[:9]} was not delivered: {exc}")
            failed += 1
            continue
        if not page_reached_telegram(result):
            # The channel's detail string is not printed: a transport error's
            # repr can carry the Bot API URL, which embeds the token.
            print(f"::error::page for {m.repo}@{m.sha[:9]} did not reach Telegram; SNS alone is not a page")
            failed += 1
    return 1 if (failed or unreadable or channel_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
