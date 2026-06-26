"""reconcile_audit — T+1 self-heal: re-derive EOD market values from settled data.

The same-day EOD reconcile reads SPY (and held-position) closes from ArcticDB
at ~4:20pm ET, which can still be a PRE-SETTLEMENT value. ArcticDB later
self-heals to the official settled close, but the already-written
``eod_pnl``/``eod_report.json`` froze the provisional number — so a value a
human uses for financial decisions stays silently wrong (config#1276: 2026-06-25
SPY stored 733.50 vs settled 734.30, corrupting daily alpha on 06-25 AND 06-26).

This pass closes that gap institutionally. For a trailing window of trading
days it compares each stored ``eod_pnl.spy_close`` against the CURRENT settled
ArcticDB close. Any date that diverged beyond tolerance — or is missing
entirely (a skipped session) — is corrected by re-running the canonical path
(``eod_reconcile.run`` for an existing row, ``backfill_eod_pnl.backfill`` for a
missing row), which re-prices everything from settled ArcticDB and re-emits the
report artifact. Each correction writes an audit record and pages flow-doctor.

Design notes:
  * The check is CHEAP in the common case — a handful of ArcticDB reads; a
    re-reconcile only fires when a date actually diverged. A clean window is a
    no-op beyond the reads.
  * Corrections run OLDEST→NEWEST so a corrected close propagates into the next
    day's spy_return denominator (and a backfilled gap row becomes the next
    day's prior-NAV baseline) within a single pass.
  * Re-reconciles run with ``send_email=False`` (never resend an old day's
    email) and ``run_audit=False`` (never recurse into this pass).
  * Comparison is stored-frozen vs current-ArcticDB — the SAME source the
    original write read. So any non-trivial divergence means ArcticDB CHANGED
    since the freeze (a settlement correction), which is exactly what we want
    to re-pick-up; the tolerance only filters float noise, not real moves.

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

        # ── Case A: missing row (a skipped session) → ledger-synthesis backfill.
        if row is None:
            before = None
            divergence_bps = None
            reason = "missing_row"
        else:
            stored_close = row[0]
            if stored_close is None:
                divergence_bps = float("inf")
            else:
                divergence_bps = abs(settled / float(stored_close) - 1.0) * 1e4
            if divergence_bps <= tolerance_bps:
                continue  # clean — stored close matches settled within tolerance
            before = {"spy_close": stored_close, "spy_return_pct": row[1], "daily_alpha_pct": row[2]}
            reason = "stale_close"

        logger.warning(
            "[reconcile_audit] %s needs correction (%s): stored=%s settled=%.2f div=%s bps%s",
            d, reason, (before or {}).get("spy_close") if before else "—", settled,
            (f"{divergence_bps:.2f}" if divergence_bps not in (None, float("inf")) else str(divergence_bps)),
            " [dry-run]" if dry_run else "",
        )

        if dry_run:
            corrected.append({"date": d, "reason": reason, "divergence_bps": divergence_bps,
                              "before": before, "settled_spy_close": settled, "applied": False})
            continue

        # ── Apply the canonical correction.
        try:
            if reason == "missing_row":
                from executor.backfill_eod_pnl import backfill
                backfill(d)  # synthesizes the snapshot + runs eod_run (no email, no audit)
            else:
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

        record = {
            "date": d,
            "reason": reason,
            "tolerance_bps": tolerance_bps,
            "divergence_bps": divergence_bps,
            "settled_spy_close": settled,
            "before": before,
            "after": after,
            "source": "arcticdb_macro",
            "corrected_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_audit_record(trades_bucket=trades_bucket, run_date=d, record=record, region=region)
        if fd:
            fd.report(
                RuntimeError(
                    f"EOD value for {d} corrected post-settlement ({reason}): "
                    f"SPY close {(before or {}).get('spy_close')} → {after.get('spy_close') if after else settled}"
                ),
                severity="warning",
                context={"site": "reconcile_audit_corrected", "run_date": d, "reason": reason},
            )
        corrected.append({"date": d, "reason": reason, "divergence_bps": divergence_bps,
                          "before": before, "after": after, "applied": True})

    conn.close()
    summary = {
        "checked": checked,
        "corrected": corrected,
        "skipped": skipped,
        "tolerance_bps": tolerance_bps,
        "dry_run": dry_run,
        "window": dates,
    }
    logger.info("[reconcile_audit] done: checked=%d corrected=%d skipped=%d dry_run=%s",
                checked, len(corrected), len(skipped), dry_run)
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
    parser.add_argument("--dry-run", action="store_true", help="Report divergences; change nothing.")
    parser.add_argument("--email", action="store_true",
                        help="Resend EOD email for corrected days (default: suppressed).")
    args = parser.parse_args()
    result = audit_window(
        trailing_days=args.trailing,
        start=args.start,
        end=args.end,
        tolerance_bps=args.tolerance_bps,
        dry_run=args.dry_run,
        send_email=args.email,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
