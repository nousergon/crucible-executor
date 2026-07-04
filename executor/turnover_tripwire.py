"""Turnover tripwire — ROADMAP L4515 (fast standalone live-system ops fix).

``turnover_one_way`` has been computed in ``portfolio_optimizer`` since the
governor shipped, but never ALARMED. Three extreme-decision incidents (idle
cash after a hard-risk exit 2026-05-29 #229; a ~90%-DOWN-skew prediction day
2026-06-01 #230; an optimizer target silently dropped on a missing price
2026-06-04 #234/#235) were each caught by a point fix after the fact. This is
the GENERAL surface: a band check on the existing executed-turnover metric
that pages when the book churns abnormally, whatever the upstream cause.

Two independent bands, checked daily in the morning planner (via
``run_shadow_optimizer``) and persisted into the shadow artifact:

- **daily** — executed ``turnover_one_way`` above the governor cap × a
  multiple. The governor is supposed to make this impossible; a breach means
  the cap was bypassed/disabled and pages at ERROR.
- **rolling** — the sum of executed turnover over the last N sessions (read
  from the prior ``predictor/optimizer_shadow/{date}.json`` artifacts) above a
  band. Catches churn-by-a-thousand-cuts: every day under the cap but the
  week's cumulative rebalance abnormal — the actual signature of the three
  incidents above. Pages at WARN.

Posture (per [[feedback_no_silent_fails]]): the tripwire itself RAISES on a
breach via ``alerts.publish`` (SNS + Telegram, deduped per run_date). It is
secondary observability hung off the planner's primary path, so an internal
failure must never block order planning — but the failure is RECORDED, not
swallowed: WARN log + a status/sentinel block written into the daily shadow
artifact (``turnover_tripwire`` key), so a dead tripwire is itself visible.
"""
from __future__ import annotations

import json
import logging
import math
import re

import boto3

logger = logging.getLogger(__name__)

_SHADOW_PREFIX = "predictor/optimizer_shadow/"
_DATED_KEY_RE = re.compile(r"optimizer_shadow/(\d{4}-\d{2}-\d{2})\.json$")

# Absolute daily band when the governor is OFF (max_daily_turnover: None) —
# with no cap to multiply, a full-book one-way move above this is the same
# "should have been operator-reviewed" event the governor would have capped.
_DAILY_BAND_GOVERNOR_OFF = 0.25


def check_turnover_tripwire(
    diagnostics: dict,
    optimizer_cfg: dict,
    signals_bucket: str,
    run_date: str,
    s3_client=None,
) -> dict:
    """Run both bands and alert on breach. Returns the block persisted into
    the shadow artifact. Never raises (see module docstring posture)."""
    try:
        if not optimizer_cfg.get("turnover_tripwire_enabled", True):
            return {"status": "disabled"}
        today = (diagnostics or {}).get("turnover_one_way")
        if today is None or not math.isfinite(float(today)):
            # Upstream contract violation — the optimizer always writes this
            # field on a solved run. Surface it, don't quietly skip (the
            # silent-skip is how the 6/04 dropped-target class hid).
            logger.warning(
                "turnover tripwire: diagnostics carry no finite "
                "turnover_one_way (run_date=%s) — tripwire DID NOT RUN",
                run_date,
            )
            return {"status": "no_turnover_metric"}
        today = float(today)

        cap = optimizer_cfg.get("max_daily_turnover")
        multiple = float(optimizer_cfg.get("turnover_tripwire_daily_multiple", 1.25))
        daily_band = float(cap) * multiple if cap else _DAILY_BAND_GOVERNOR_OFF
        rolling_days = int(optimizer_cfg.get("turnover_tripwire_rolling_days", 5))
        rolling_band = float(
            optimizer_cfg.get("turnover_tripwire_rolling_sum_band", 0.60)
        )

        prior = _read_prior_turnovers(
            signals_bucket, run_date, rolling_days - 1, s3_client
        )
        window = [today] + prior
        rolling_sum = float(sum(window))

        daily_breach = today > daily_band
        rolling_breach = rolling_sum > rolling_band
        out = {
            "status": "ok",
            "turnover_one_way": round(today, 6),
            "daily_band": round(daily_band, 6),
            "daily_breach": daily_breach,
            "rolling_days": rolling_days,
            "n_days_used": len(window),
            "rolling_sum": round(rolling_sum, 6),
            "rolling_band": round(rolling_band, 6),
            "rolling_breach": rolling_breach,
        }
        if daily_breach:
            _publish(
                out,
                severity="ERROR",
                dedup_key=f"turnover_tripwire_daily_{run_date}",
                message=(
                    f"[executor] TURNOVER TRIPWIRE (daily): executed one-way "
                    f"turnover {today:.1%} exceeds the {daily_band:.1%} band "
                    f"(governor cap {cap if cap is None else format(cap, '.0%')}, "
                    f"run_date={run_date}). The governor should make this "
                    f"impossible — investigate before the next session."
                ),
            )
            logger.warning(
                "TURNOVER TRIPWIRE daily breach: %.1f%% > %.1f%% (run_date=%s)",
                today * 100, daily_band * 100, run_date,
            )
        if rolling_breach:
            _publish(
                out,
                severity="WARN",
                dedup_key=f"turnover_tripwire_rolling_{run_date}",
                message=(
                    f"[executor] TURNOVER TRIPWIRE (rolling): one-way turnover "
                    f"summed {rolling_sum:.1%} over the last {len(window)} "
                    f"session(s) — above the {rolling_band:.0%}/{rolling_days}d "
                    f"band (run_date={run_date}). The book is churning "
                    f"abnormally even though each day is under the cap; review "
                    f"the optimizer shadow logs for the driver."
                ),
            )
            logger.warning(
                "TURNOVER TRIPWIRE rolling breach: sum %.1f%% over %d sessions "
                "> %.1f%% (run_date=%s)",
                rolling_sum * 100, len(window), rolling_band * 100, run_date,
            )
        if not (daily_breach or rolling_breach):
            logger.info(
                "turnover tripwire OK: today=%.1f%% (band %.1f%%), "
                "rolling %.1f%%/%dd (band %.1f%%)",
                today * 100, daily_band * 100, rolling_sum * 100,
                len(window), rolling_band * 100,
            )
        return out
    except Exception as e:  # noqa: BLE001 — secondary observability: must not
        # block the planner; failure recorded in the shadow artifact + WARN.
        logger.warning("turnover tripwire failed (non-blocking): %s", e, exc_info=True)
        return {"status": "error", "error": repr(e)}


def _read_prior_turnovers(
    bucket: str, run_date: str, n: int, s3_client=None,
) -> list[float]:
    """Executed ``turnover_one_way`` from the most recent ``n`` dated shadow
    artifacts strictly before ``run_date``. Artifacts that are missing the
    metric (failed/sentinel days) are skipped with a log line — a short window
    still alerts when its partial sum already breaches (sum is monotonic)."""
    if n <= 0:
        return []
    s3 = s3_client or boto3.client("s3")
    dates: list[str] = []
    token = None
    # Hard page cap: the prefix accrues ~1 dated key per session (~250/yr) at
    # 1000 keys/page, so >10 pages is structurally impossible — the cap is a
    # guard against a pathological/non-conforming client looping forever on a
    # truthy IsTruncated (exactly how a MagicMock behaves in tests).
    for _page in range(10):
        kwargs = {"Bucket": bucket, "Prefix": _SHADOW_PREFIX}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            m = _DATED_KEY_RE.search(obj.get("Key", ""))
            if m and m.group(1) < run_date:
                dates.append(m.group(1))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    else:
        logger.warning(
            "turnover tripwire: shadow-prefix listing exceeded the 10-page "
            "cap — rolling window computed from the first %d keys only",
            len(dates),
        )
    out: list[float] = []
    for d in sorted(dates, reverse=True)[:n]:
        try:
            body = s3.get_object(Bucket=bucket, Key=f"{_SHADOW_PREFIX}{d}.json")
            log_d = json.loads(body["Body"].read())
            v = (log_d.get("diagnostics") or {}).get("turnover_one_way")
            if v is not None and math.isfinite(float(v)):
                out.append(float(v))
            else:
                logger.info(
                    "turnover tripwire: %s shadow log has no turnover "
                    "(sentinel/failed day) — excluded from rolling window", d,
                )
        except Exception as e:  # noqa: BLE001 — one unreadable day must not
            # kill the window; the exclusion is logged and n_days_used shows it.
            logger.warning(
                "turnover tripwire: could not read shadow log for %s: %s", d, e,
            )
    return out


def _publish(out: dict, *, severity: str, dedup_key: str, message: str) -> None:
    """Best-effort alert publish — mirrors the large-move flag posture: the
    band verdict is already recorded (shadow artifact + WARN log), so a
    publish failure must never block the planner; it is recorded in the
    artifact's ``publish_error`` field."""
    try:
        from executor.notifier import publish_ops_alert

        publish_ops_alert(
            message=message,
            severity=severity,
            source="alpha-engine/executor/turnover_tripwire.py",
            dedup_key=dedup_key,
        )
    except Exception as e:  # noqa: BLE001 — secondary observability
        logger.warning("turnover tripwire alert publish failed (non-fatal): %s", e)
        out["publish_error"] = repr(e)
