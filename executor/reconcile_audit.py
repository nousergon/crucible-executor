"""reconcile_audit — T+1 self-heal: re-derive EOD market values from settled data.

The same-day EOD reconcile reads SPY (and held-position) closes from ArcticDB
at ~4:20pm ET, which can still be a PRE-SETTLEMENT value. ArcticDB later
self-heals to the official settled close, but the already-written
``eod_pnl``/``eod_report.json`` froze the provisional number — so a value a
human uses for financial decisions stays silently wrong (config#1276: 2026-06-25
SPY stored 733.50 vs settled 734.30, corrupting daily alpha on 06-25 AND 06-26).

This pass closes that gap institutionally. For a trailing window of trading
days it re-reconciles (``eod_reconcile.run``, re-pricing from settled ArcticDB
and re-emitting the artifact) any day whose stored ``spy_close`` diverged from
the current settled close OR whose stored ``spy_return`` no longer matches the
value recomputed from settled closes (the cascade case). Each correction writes
an audit record and pages flow-doctor.

It does NOT blanket-synthesize a NAV for a missing row. Ledger-replay backfill
(``backfill_eod_pnl``) reconstructs positions from the full trades ledger, which
can drift from the broker's actual book and fabricate a wrong NAV (config#1276
follow-up: an auto-backfilled 2026-06-24 produced 20 positions / +51% NAV vs the
real 7). Gaps are FLAGGED for manual, position-verified backfill UNLESS they pass
the narrow, strictly-gated verified-zero-fill carry-forward + reprice auto-backfill
(``auto_backfill_gap``, config#1454) — see the gap-handling branch below for the
three-part gate. Any gap that fails that gate still falls through to the manual
flag; this pass never auto-synthesizes a NAV for a day that actually traded.

Design notes:
  * The check is CHEAP in the common case — a handful of ArcticDB reads; a
    re-reconcile only fires when a date actually diverged. A clean window is a
    no-op beyond the reads.
  * Corrections run OLDEST→NEWEST so a corrected close propagates into the next
    day's spy_return denominator within a single pass (the cascade detector
    then re-reconciles that next day off the now-corrected prior).
  * Re-reconciles run with ``send_email=False`` (never resend an old day's
    email) and ``run_audit=False`` (never recurse into this pass).
  * Both legs of the recomputed spy_return come from settled ArcticDB; the
    tolerance only filters float noise, not real moves.

Usage:
    python -m executor.reconcile_audit                       # trailing 5 trading days
    python -m executor.reconcile_audit --trailing 7
    python -m executor.reconcile_audit --start 2026-06-22 --end 2026-06-26
    python -m executor.reconcile_audit --dry-run             # report, change nothing
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone

import boto3

from executor.config_loader import load_config
from executor.eod_reconcile import _spy_close, run as eod_run
from executor.trade_logger import init_db
from nousergon_lib.dates import now_dual
from nousergon_lib.logging import get_flow_doctor
from nousergon_lib.trading_calendar import previous_trading_day

logger = logging.getLogger(__name__)

# A close that came from the SAME ArcticDB source should match its stored copy
# exactly unless ArcticDB was corrected since. 1 bp filters float/round noise
# while still catching the config#1276 class (06-22 was 1.5 bp, 06-25 was 11 bp).
DEFAULT_TOLERANCE_BPS = 1.0
DEFAULT_TRAILING_DAYS = 5
AUDIT_KEY_TEMPLATE = "trades/eod_corrections/{run_date}.json"

# A correction at/above this magnitude is a statistical outlier vs. the normal
# settlement-lag band (06-22 was 1.5bp, 06-25's genuine incident was 11bp) and
# is worth paging a human. Below it, the correction is routine provisional-vs-
# settled drift — still written to the S3 audit trail + dashboard, but logged
# at severity="info" so it doesn't page Telegram (config#2145: a 1.46bp
# correction on 2026-07-09 paged identically to a real incident, training the
# operator to ignore this producer's warnings). A SECOND (or later) correction
# within the same audit run is treated as page-worthy regardless of its own
# magnitude — one date drifting is routine noise, multiple dates drifting
# together in one pass signals a systemic feed issue, not settlement lag.
PAGE_THRESHOLD_BPS = 5.0


def _window_dates(
    *,
    start: str | None,
    end: str | None,
    trailing_days: int,
) -> list[str]:
    """The ascending list of NYSE trading days to audit.

    Explicit ``start``/``end`` → every trading day in [start, end]. Otherwise →
    the ``trailing_days`` trading days ending at ``end`` (default: today's
    trading_day), walking back over the trading calendar (skips weekends/holidays).
    """
    end_date = date.fromisoformat(end) if end else date.fromisoformat(now_dual().trading_day)
    if start:
        days: list[date] = []
        cur = end_date
        start_date = date.fromisoformat(start)
        # Walk back from end to start over trading days (inclusive both ends).
        while cur >= start_date:
            days.append(cur)
            cur = previous_trading_day(cur)
        return [d.isoformat() for d in sorted(days)]
    days = []
    cur = end_date
    for _ in range(trailing_days):
        days.append(cur)
        cur = previous_trading_day(cur)
    return [d.isoformat() for d in sorted(days)]


def _settled_close(run_date: str, config: dict) -> float | None:
    """Settled ArcticDB SPY close for ``run_date``, or None if not yet available
    (no row in ArcticDB — gap unhealed or close not yet landed; nothing to
    reconcile against this pass)."""
    try:
        return _spy_close(run_date, config)
    except Exception as e:  # noqa: BLE001 — absence is "can't check yet", not fatal
        logger.info("[reconcile_audit] no settled ArcticDB SPY close for %s yet (%s) — skipping.",
                    run_date, e.__class__.__name__)
        return None


def _write_audit_record(
    *,
    trades_bucket: str,
    run_date: str,
    record: dict,
    region: str,
) -> str | None:
    if not trades_bucket:
        return None
    key = AUDIT_KEY_TEMPLATE.format(run_date=run_date)
    try:
        boto3.client("s3", region_name=region).put_object(
            Bucket=trades_bucket,
            Key=key,
            Body=json.dumps(record, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("[reconcile_audit] wrote correction record s3://%s/%s", trades_bucket, key)
        return key
    except Exception as e:  # noqa: BLE001 — audit-trail write is best-effort observability
        logger.warning("[reconcile_audit] correction-record write failed (non-fatal): %s", e)
        return None


def audit_window(
    *,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    start: str | None = None,
    end: str | None = None,
    exclude_dates: set[str] | frozenset[str] = frozenset(),
    tolerance_bps: float = DEFAULT_TOLERANCE_BPS,
    page_threshold_bps: float = PAGE_THRESHOLD_BPS,
    dry_run: bool = False,
    send_email: bool = False,
    config: dict | None = None,
) -> dict:
    """Re-reconcile any windowed trading day whose stored SPY close has diverged
    from the settled ArcticDB close (or is missing). Returns a summary dict."""
    config = config or load_config()
    db_path = config["db_path"]
    trades_bucket = config["trades_bucket"]
    region = config.get("aws_region", "us-east-1")
    conn = init_db(db_path)

    dates = [d for d in _window_dates(start=start, end=end, trailing_days=trailing_days)
             if d not in exclude_dates]
    try:
        fd = get_flow_doctor()
    except Exception:  # noqa: BLE001 — flow-doctor optional / not configured
        fd = None

    corrected: list[dict] = []
    skipped: list[dict] = []
    gaps: list[dict] = []
    checked = 0

    for d in dates:  # oldest → newest so corrections propagate forward in one pass
        settled = _settled_close(d, config)
        if settled is None:
            skipped.append({"date": d, "reason": "no_settled_close"})
            continue
        checked += 1

        row = conn.execute(
            "SELECT spy_close, spy_return_pct, daily_alpha_pct FROM eod_pnl WHERE date = ?",
            (d,),
        ).fetchone()

        # ── Missing row (a skipped session). reconcile_audit does NOT
        # blanket-synthesize a NAV here — ledger-replay backfill
        # (backfill_eod_pnl) reconstructs positions from the full trades
        # ledger, which can drift from the broker's actual book and fabricate
        # a wrong NAV (config#1276 follow-up: an auto-backfilled 06-24
        # produced 20 positions / +51% NAV vs the real 7).
        #
        # config#1454 carves out ONE narrow, verified-honest exception: a day
        # on which ZERO fills executed is deterministic to reconstruct (carry
        # the prior day's book forward unchanged, reprice at authoritative
        # ArcticDB closes). auto_backfill_gap.attempt_auto_backfill applies
        # this ONLY when its strict 3-part gate passes (zero fills confirmed
        # / prior snapshot exists / authoritative closes complete for every
        # held ticker) — any gate failure (including a day that actually
        # traded) still falls through to the MANUAL-backfill flag below.
        if row is None:
            try:
                from executor.auto_backfill_gap import attempt_auto_backfill
                auto_result = attempt_auto_backfill(
                    conn, gap_date=d, trades_bucket=trades_bucket, region=region,
                )
            except Exception as e:  # noqa: BLE001 — auto-backfill is best-effort;
                # any failure (gate check, S3, or the downstream reconcile) must
                # fall through to the manual-flag path, never leave a half state.
                logger.error(
                    "[reconcile_audit] auto-backfill attempt FAILED for %s: %s — "
                    "falling back to MANUAL flag.", d, e,
                )
                auto_result = {"backfilled": False, "reason": str(e), "gate": {}}

            if auto_result["backfilled"]:
                logger.info(
                    "[reconcile_audit] %s auto-backfilled (verified zero-fill "
                    "carry-forward + reprice, config#1454) — gap cleared.", d,
                )
                if fd:
                    fd.report(
                        RuntimeError(
                            f"eod_pnl gap at {d} AUTO-backfilled (verified zero-fill "
                            f"carry-forward + reprice, config#1454). Prior trading day: "
                            f"{auto_result['gate'].get('prior_date')}."),
                        severity="warning",
                        context={"site": "reconcile_audit_gap_auto_backfilled", "run_date": d})
                continue

            logger.warning(
                "[reconcile_audit] %s has NO eod_pnl row (skipped session) — %s "
                "Flagging gap for MANUAL backfill; not auto-synthesizing a NAV.",
                d, auto_result["reason"])
            gaps.append({
                "date": d, "settled_spy_close": settled,
                "auto_backfill_reason": auto_result["reason"],
            })
            if fd:
                fd.report(
                    RuntimeError(
                        f"eod_pnl gap at {d}: no row (skipped session). Auto-backfill "
                        f"not eligible ({auto_result['reason']}). Manually run "
                        f"`backfill_eod_pnl --date {d}` after verifying held positions."),
                    severity="warning",
                    context={"site": "reconcile_audit_gap", "run_date": d})
            continue

        stored_close, stored_spy_return = row[0], row[1]

        # Detector: re-reconcile if the day's OWN close diverged from settled, OR
        # if its stored spy_return no longer matches the value recomputed from
        # settled closes. The second clause catches the CASCADE — a prior day's
        # close was corrected, so THIS day's spy_return denominator is stale even
        # though its own close is fine (the 06-26-after-06-25 case). Without it,
        # an own-close-only detector would never self-heal a cascaded return.
        own_close_div_bps = (
            float("inf") if stored_close is None
            else abs(settled / float(stored_close) - 1.0) * 1e4)

        prior_row = conn.execute(
            "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1", (d,)
        ).fetchone()
        settled_prior = _settled_close(prior_row[0], config) if prior_row else None
        expected_spy_return = (
            (settled / settled_prior - 1.0) * 100.0 if settled_prior else None)
        return_div_bps = (
            None if (expected_spy_return is None or stored_spy_return is None)
            else abs(expected_spy_return - float(stored_spy_return)) * 100.0)  # 1% = 100bp

        own_stale = own_close_div_bps > tolerance_bps
        return_stale = (
            (stored_spy_return is None and expected_spy_return is not None)
            or (return_div_bps is not None and return_div_bps > tolerance_bps))
        if not own_stale and not return_stale:
            continue  # clean — own close and recomputed spy_return both match

        reason = "stale_close" if own_stale else "stale_return"
        before = {"spy_close": stored_close, "spy_return_pct": stored_spy_return,
                  "daily_alpha_pct": row[2]}
        divergence_bps = own_close_div_bps if own_stale else return_div_bps

        logger.warning(
            "[reconcile_audit] %s needs correction (%s): stored_close=%s settled=%.2f "
            "own_div=%s return_div=%s%s", d, reason, stored_close, settled,
            (f"{own_close_div_bps:.2f}bp" if own_close_div_bps != float("inf") else "inf"),
            (f"{return_div_bps:.2f}bp" if return_div_bps is not None else "n/a"),
            " [dry-run]" if dry_run else "",
        )

        if dry_run:
            corrected.append({"date": d, "reason": reason, "divergence_bps": divergence_bps,
                              "before": before, "settled_spy_close": settled, "applied": False})
            continue

        # ── Apply the canonical correction (re-price + re-emit from settled data).
        try:
            eod_run(d, send_email=send_email, run_audit=False)
        except Exception as e:  # noqa: BLE001 — per-date isolation: one bad day must not abort the sweep
            logger.error("[reconcile_audit] correction FAILED for %s: %s", d, e)
            if fd:
                fd.report(e, severity="error", context={"site": "reconcile_audit_apply", "run_date": d})
            skipped.append({"date": d, "reason": f"apply_failed: {e.__class__.__name__}"})
            continue

        after_row = conn.execute(
            "SELECT spy_close, spy_return_pct, daily_alpha_pct FROM eod_pnl WHERE date = ?",
            (d,),
        ).fetchone()
        after = ({"spy_close": after_row[0], "spy_return_pct": after_row[1],
                  "daily_alpha_pct": after_row[2]} if after_row else None)

        # Page-worthiness: an outlier-magnitude correction, OR a correction
        # that isn't the first in this run (multiple dates drifting together
        # is systemic, not routine settlement lag) — see PAGE_THRESHOLD_BPS.
        # Every flow-doctor severity level maps to SOME Telegram notifier in
        # flow-doctor.yaml (critical→#critical, error/warning→#ops-health,
        # info→#trades) — there is no "silent" severity to pick. So routine,
        # in-band corrections skip the fd.report() call entirely rather than
        # trying to pick a severity that happens not to page; they still get
        # the full S3 audit trail (below) and a local INFO log line.
        is_recurrence = len(corrected) >= 1
        is_outlier = divergence_bps is None or divergence_bps == float("inf") or divergence_bps >= page_threshold_bps
        page_worthy = is_outlier or is_recurrence

        record = {
            "date": d,
            "reason": reason,
            "tolerance_bps": tolerance_bps,
            "page_threshold_bps": page_threshold_bps,
            "divergence_bps": divergence_bps,
            "paged": page_worthy,
            "settled_spy_close": settled,
            "before": before,
            "after": after,
            "source": "arcticdb_macro",
            "corrected_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_audit_record(trades_bucket=trades_bucket, run_date=d, record=record, region=region)
        correction_message = (
            f"EOD value for {d} corrected post-settlement ({reason}): "
            f"SPY close {(before or {}).get('spy_close')} → {after.get('spy_close') if after else settled}"
        )
        if page_worthy:
            if fd:
                fd.report(
                    RuntimeError(correction_message),
                    severity="warning",
                    context={"site": "reconcile_audit_corrected", "run_date": d, "reason": reason,
                             "divergence_bps": divergence_bps, "page_threshold_bps": page_threshold_bps},
                )
        else:
            logger.info("[reconcile_audit] %s (in-band, %.2fbp < %.2fbp threshold — audit trail "
                        "only, not paged)", correction_message, divergence_bps, page_threshold_bps)
        corrected.append({"date": d, "reason": reason, "divergence_bps": divergence_bps,
                          "before": before, "after": after, "applied": True, "paged": page_worthy})

    conn.close()
    summary = {
        "checked": checked,
        "corrected": corrected,
        "skipped": skipped,
        "gaps": gaps,
        "tolerance_bps": tolerance_bps,
        "dry_run": dry_run,
        "window": dates,
    }
    logger.info("[reconcile_audit] done: checked=%d corrected=%d gaps=%d skipped=%d dry_run=%s",
                checked, len(corrected), len(gaps), len(skipped), dry_run)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Re-reconcile EOD days whose stored SPY close diverged from settled ArcticDB (config#1276).",
    )
    parser.add_argument("--trailing", type=int, default=DEFAULT_TRAILING_DAYS,
                        help=f"Trailing trading days to audit (default {DEFAULT_TRAILING_DAYS}).")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD window start (overrides --trailing).")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD window end (default: today's trading_day).")
    parser.add_argument("--tolerance-bps", type=float, default=DEFAULT_TOLERANCE_BPS,
                        help=f"Divergence tolerance in bps (default {DEFAULT_TOLERANCE_BPS}).")
    parser.add_argument("--page-threshold-bps", type=float, default=PAGE_THRESHOLD_BPS,
                        help=f"Divergence at/above which a correction pages (severity=warning) "
                             f"instead of just logging (severity=info) (default {PAGE_THRESHOLD_BPS}).")
    parser.add_argument("--dry-run", action="store_true", help="Report divergences; change nothing.")
    parser.add_argument("--email", action="store_true",
                        help="Resend EOD email for corrected days (default: suppressed).")
    args = parser.parse_args()
    result = audit_window(
        trailing_days=args.trailing,
        start=args.start,
        end=args.end,
        tolerance_bps=args.tolerance_bps,
        page_threshold_bps=args.page_threshold_bps,
        dry_run=args.dry_run,
        send_email=args.email,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
