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
the current settled close, whose stored ``spy_return`` no longer matches the
value recomputed from settled closes (the cascade case), OR whose stored
per-ticker ``closing_price`` for ANY held position diverged from that
ticker's own current settled close (config#6349: the SPY-only checks caught
nothing for a non-SPY name's provisional-then-corrected price, which then
silently fed every later day's NAV three-way pricing&timing diff via
``prior_positions``), OR whose immediately-prior session was itself corrected
earlier in this same pass (alpha-engine-config-I10288: the position-level
analogue of the spy_return cascade — a corrected day changes the NEXT day's
``prior_positions`` basis, and that day's own stored closes are all fine, so no
other leg ever fires for it).

Each correction is CLASSIFIED as a *revision* (a bounded post-settlement vendor
change to a print that was correct when we stored it) or a *corruption* (a value
the settled source plausibly never held), is VERIFIED to have converged against
settled data before it is called done, writes an audit record naming its own
downstream blast radius, and — for a corruption or a repeat — pages flow-doctor.
A correction that does NOT converge raises ``ReconciliationUnconvergedError``.

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
from datetime import UTC, date, datetime

import boto3
import pandas as pd
from nousergon_lib.dates import now_dual
from nousergon_lib.logging import get_flow_doctor
from nousergon_lib.trading_calendar import previous_trading_day

from executor.accepted_gaps import load_accepted_gaps
from executor.config_loader import load_config
from executor.eod_reconcile import _log_paged, _spy_close
from executor.eod_reconcile import run as eod_run
from executor.trade_logger import init_db

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

# ── Revision vs corruption ───────────────────────────────────────────────────
#
# A divergence between a stored EOD value and the current settled value has two
# materially different causes, and until alpha-engine-config-I10288 this pass
# could not say which it was seeing:
#
#   REVISION   — the upstream vendor moved an already-settled print by a bounded
#                amount after we stored it. Our write was CORRECT at write time;
#                the source changed underneath it. This is a normal, expected
#                property of end-of-day equity data and it is not a defect in
#                anything we own. It is still recorded, still corrected, and
#                still makes every downstream artifact derived from the old
#                value stale — see DOWNSTREAM_ON_CORRECTION.
#   CORRUPTION — a stored value the settled source plausibly never held: absent
#                or zero (unbounded divergence), or a divergence too large to be
#                settlement lag. This is the config#1276 incident class.
#
# The band is the same number the paging decision already used (PAGE_THRESHOLD_
# BPS), so this names an existing behaviour rather than changing it: corruption
# pages, a lone revision does not. What changes is that the classification is
# now an explicit, machine-readable field on the record, in the flow-doctor
# context and in the first line of the alert text — so an operator reading the
# page can tell a healed vendor revision from a data-integrity incident without
# opening the S3 record.
CLASSIFICATION_REVISION = "revision"
CLASSIFICATION_CORRUPTION = "corruption"

# Which downstream artifacts consumed the pre-correction value, and whether the
# re-reconcile this pass performs re-derives them. Recorded verbatim on every
# correction record so the audit trail STATES ITS OWN BLAST RADIUS instead of
# leaving every reader to re-derive it from six repos (alpha-engine-config-
# I10288). Every entry below was traced against the code, not assumed.
DOWNSTREAM_ON_CORRECTION: dict[str, str] = {
    # eod_reconcile writes the row via trade_logger.log_eod's INSERT OR REPLACE,
    # so EVERY column for that date is rewritten, not only the SPY fields.
    "eod_pnl row for the corrected date (all columns)":
        "rederived — trade_logger.log_eod INSERT OR REPLACE",
    # closing_price is re-read from settled ArcticDB for every held name.
    "eod_pnl.positions_snapshot closing_price, every held ticker":
        "rederived — eod_reconcile re-prices each held name from settled ArcticDB",
    "trades/eod_pnl.csv, trades/trades_full.csv, trades/shadow_book.csv":
        "rederived — the whole table is re-exported on every run",
    "consolidated/{corrected_date}/eod_report.json":
        "rederived — re-emitted by the same run",
    "the EOD email already sent for that date":
        "stale — never resent by design; the operator saw the old number",
    # The reason this pass now carries a prior_corrected leg: without it these
    # frozen per-date artifacts never heal.
    "consolidated/{later_date}/eod_report.json — daily_return_pct, "
    "pricing_timing_usd, mark_basis_usd, trailing_history for sessions AFTER the "
    "corrected date (computed from prior_positions)":
        "rederived ONLY for the next session inside this audit window, via the "
        "prior_corrected cascade leg; any later session outside the window stays stale",
    "evaluator/{date}/report_card.json and evaluator/{date}/attribution.json "
    "(frozen weekly records, including input_closure_usd)":
        "PERMANENTLY STALE — written once per weekly cycle, no invalidation hook exists "
        "(see alpha-engine-config-I10271)",
    "evaluator/latest/report_card.json and evaluator/latest/attribution.json":
        "rederived at the next scheduled weekly run — these read trades/eod_pnl.csv live",
    "report_card trend_4w / trend_13w":
        "never fully heals — computed by diffing against the frozen past report cards above",
    "crucible-dashboard pages reading trades/eod_pnl.csv or trades_full.csv":
        "rederived — pure readers of the live CSV",
}

# leg name → the historical ``reason`` code, kept so the S3 record schema and
# every existing consumer of ``reason`` keep working unchanged.
_LEG_REASON = {
    "spy_close": "stale_close",
    "spy_return_pct": "stale_return",
    "position_close": "stale_position_close",
    "prior_corrected": "stale_prior_basis",
}


class ReconciliationUnconvergedError(RuntimeError):
    """A re-reconcile ran and the stored value STILL diverges from settled.

    Categorically different from a revision or a corruption that this pass
    HEALED: it means the self-heal did not heal. ``eod_pnl`` is knowably wrong,
    no later pass corrects it on its own (this pass would simply re-detect and
    re-fail every day), and the number is one a human uses for financial
    decisions. Fail loud — raised after the whole window is processed so every
    date is still checked and recorded first.
    """

    def __init__(self, message: str, *, summary: dict | None = None) -> None:
        super().__init__(message)
        self.summary = summary or {}


def _mag(divergence_bps: float | None) -> float:
    """Divergence magnitude for comparison/ordering. ``None`` means "we could
    not compute it" (e.g. a stored value of None), which is UNBOUNDED, not
    zero — treating it as zero would rank the worst case as the mildest."""
    return float("inf") if divergence_bps is None else float(divergence_bps)


def _fmt_bps(divergence_bps: float | None) -> str:
    m = _mag(divergence_bps)
    return "inf" if m == float("inf") else f"{m:.2f}"


def _classify(divergence_bps: float | None, *, page_threshold_bps: float) -> str:
    """``CLASSIFICATION_REVISION`` or ``CLASSIFICATION_CORRUPTION`` — see the
    band definition above. Unbounded divergence is always corruption."""
    return (CLASSIFICATION_CORRUPTION
            if _mag(divergence_bps) >= page_threshold_bps
            else CLASSIFICATION_REVISION)


def _describe_legs(legs: list[dict]) -> str:
    """Human-readable, per-leg description of what actually diverged.

    The alert text used to be hard-coded to SPY regardless of which value was
    stale, so a held-position divergence paged as "SPY close X → Y" quoting a
    SUB-TOLERANCE incidental SPY re-price while the divergent ticker appeared
    nowhere (alpha-engine-config-I10288). This names every stale leg, and only
    the stale legs.
    """
    return "; ".join(
        f"{leg['leg']} {leg['stored']} → {leg['settled']} ({_fmt_bps(leg['divergence_bps'])}bp)"
        for leg in legs
    )


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


def _settled_close_for_ticker(ticker: str, run_date: str, config: dict) -> float | None:
    """Settled ArcticDB close for any held ``ticker`` on ``run_date``, or None
    if not yet available.

    Generalizes ``_settled_close`` (SPY-only) to every ticker a day's stored
    ``positions_snapshot`` actually held (config#6349). The original SPY-only
    detector meant a non-SPY name's provisional-then-corrected price (e.g. an
    AMD or COIN close revised after this pass's window had moved on) stayed
    frozen in ``prior_positions`` indefinitely and silently fed every later
    day's pricing&timing diff. Mirrors ``eod_reconcile.run()``'s own
    macro/universe routing so this checks the same source of truth the daily
    reconcile does.
    """
    from executor.price_cache import _MACRO_SYMBOLS, _open_macro_library, _open_universe_library
    bucket = config.get("trades_bucket", "alpha-engine-research")
    try:
        lib = _open_macro_library(bucket) if ticker in _MACRO_SYMBOLS else _open_universe_library(bucket)
        df = lib.read(ticker).data
    except Exception as e:  # noqa: BLE001 — absence is "can't check yet", not fatal
        logger.info("[reconcile_audit] no settled ArcticDB close for %s on %s yet (%s) — skipping.",
                    ticker, run_date, e.__class__.__name__)
        return None
    if df.empty or "Close" not in df.columns:
        return None
    target = pd.Timestamp(run_date).normalize()
    idx = df.index.normalize() if hasattr(df.index, "normalize") else df.index
    match = df[idx == target]
    if match.empty:
        return None
    return float(match["Close"].iloc[-1])


def _detect_stale_legs(
    conn,
    run_date: str,
    config: dict,
    *,
    tolerance_bps: float,
    settled: float,
) -> tuple[tuple | None, list[dict]]:
    """Every stored EOD value for ``run_date`` that diverges from settled data.

    ONE detector, called TWICE: once to decide whether a day needs correcting,
    and again after the re-reconcile to verify the correction actually took.
    Sharing the function is the point — a verification written as a second,
    parallel comparison could pass while the real detector still fails.

    Returns ``(row, legs)``. ``row`` is the ``eod_pnl`` row (or None when the
    day has no row at all — the gap case, handled by the caller). ``legs`` is
    every divergent value, ordered worst-first, each
    ``{"leg", "divergence_bps", "stored", "settled"}``.

    Three leg families, all subject to the same post-settlement revision
    behaviour — this is why the check is a list rather than a single SPY
    number (alpha-engine-config-I10288):
      * ``spy_close``      — the day's own SPY close vs settled.
      * ``spy_return_pct`` — the stored return vs one recomputed from settled
        closes. Catches the CASCADE: a prior day's close was corrected, so this
        day's denominator is stale even though its own close is fine.
      * ``position_close:<TICKER>`` — every ticker the day actually held, vs
        that ticker's own settled close (config#6349). A stale ``closing_price``
        feeds ``mark_basis`` on every later day via ``prior_positions``.
    """
    row = conn.execute(
        "SELECT spy_close, spy_return_pct, daily_alpha_pct, positions_snapshot "
        "FROM eod_pnl WHERE date = ?",
        (run_date,),
    ).fetchone()
    if row is None:
        return None, []

    stored_close, stored_spy_return = row[0], row[1]
    legs: list[dict] = []

    own_close_div_bps = (
        float("inf") if stored_close is None
        else abs(settled / float(stored_close) - 1.0) * 1e4)
    if own_close_div_bps > tolerance_bps:
        legs.append({"leg": "spy_close", "divergence_bps": own_close_div_bps,
                     "stored": stored_close, "settled": settled})

    prior_row = conn.execute(
        "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1", (run_date,)
    ).fetchone()
    settled_prior = _settled_close(prior_row[0], config) if prior_row else None
    expected_spy_return = (
        (settled / settled_prior - 1.0) * 100.0 if settled_prior else None)
    return_div_bps = (
        None if (expected_spy_return is None or stored_spy_return is None)
        else abs(expected_spy_return - float(stored_spy_return)) * 100.0)  # 1% = 100bp
    return_stale = (
        (stored_spy_return is None and expected_spy_return is not None)
        or (return_div_bps is not None and return_div_bps > tolerance_bps))
    if return_stale:
        legs.append({"leg": "spy_return_pct", "divergence_bps": return_div_bps,
                     "stored": stored_spy_return, "settled": expected_spy_return})

    snap_raw = row[3]
    if snap_raw:
        try:
            snap_positions = json.loads(snap_raw)
        except (json.JSONDecodeError, TypeError):
            snap_positions = {}
        for ticker, pos in snap_positions.items():
            stored_cp = pos.get("closing_price")
            if stored_cp is None:
                continue
            settled_cp = _settled_close_for_ticker(ticker, run_date, config)
            if settled_cp is None:
                continue
            cp_div_bps = abs(settled_cp / float(stored_cp) - 1.0) * 1e4 if stored_cp else float("inf")
            if cp_div_bps > tolerance_bps:
                legs.append({"leg": f"position_close:{ticker}", "divergence_bps": cp_div_bps,
                             "stored": stored_cp, "settled": settled_cp})

    legs.sort(key=lambda leg: -_mag(leg["divergence_bps"]))
    return row, legs


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

    # Load the accepted-gaps registry (alpha-engine-config#5570). This is a
    # best-effort S3 read: a missing or inaccessible registry is treated as
    # "no gaps accepted yet" and the gap handler degrades gracefully (every
    # unaccepted gap still goes through to the MANUAL-backfill flag as before).
    accepted_gaps = (
        load_accepted_gaps(trades_bucket, region)
        if trades_bucket else {}
    )

    corrected: list[dict] = []
    corrected_dates: set[str] = set()
    skipped: list[dict] = []
    gaps: list[dict] = []
    unconverged: list[dict] = []
    checked = 0

    for d in dates:  # oldest → newest so corrections propagate forward in one pass
        settled = _settled_close(d, config)
        if settled is None:
            skipped.append({"date": d, "reason": "no_settled_close"})
            continue
        checked += 1

        row, legs = _detect_stale_legs(conn, d, config, tolerance_bps=tolerance_bps, settled=settled)

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
                    # Plain string, not RuntimeError() — this is a SUCCESS
                    # notice; wrapping it in an exception rendered a completed
                    # auto-backfill as a crash (alpha-engine-config-I10288).
                    fd.report(
                        f"eod_pnl gap at {d} AUTO-backfilled (verified zero-fill "
                        f"carry-forward + reprice, config#1454). Prior trading day: "
                        f"{auto_result['gate'].get('prior_date')}.",
                        severity="warning",
                        context={"site": "reconcile_audit_gap_auto_backfilled", "run_date": d})
                continue

            # ── Accepted gap (alpha-engine-config#5570): a date whose snapshot
            # never existed and is accepted as permanently unrecoverable by
            # operator ruling. Log at INFO, still list in gaps (never suppressed
            # from the record), but do NOT warn or page — the accepted-gap
            # registry is the operator's explicit declaration that no action
            # is required.
            if d in accepted_gaps:
                entry = accepted_gaps[d]
                logger.info(
                    "[reconcile_audit] %s has NO eod_pnl row but is registered as "
                    "an ACCEPTED gap (ruling: %s, reason: %s) — logged, not paged.",
                    d, entry.get("ruling", "unknown"), entry.get("reason", "unspecified"))
                gaps.append({
                    "date": d, "settled_spy_close": settled,
                    "accepted": True,
                    "ruling": entry.get("ruling"),
                    "accepted_reason": entry.get("reason"),
                })
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
                # Plain string, not RuntimeError() — a flagged gap awaiting an
                # operator is a finding, not a crashed run
                # (alpha-engine-config-I10288).
                fd.report(
                    f"eod_pnl gap at {d}: no row (skipped session). Auto-backfill "
                    f"not eligible ({auto_result['reason']}). Manually run "
                    f"`backfill_eod_pnl --date {d}` after verifying held positions.",
                    severity="warning",
                    context={"site": "reconcile_audit_gap", "run_date": d})
            continue

        # ── Position-level cascade. The stored-return check above catches the
        # SPY cascade (a corrected prior close makes THIS day's spy_return
        # denominator stale). The position-level analogue had no clause at all:
        # `eod_reconcile.run` computes this day's `pricing_timing_usd` /
        # `mark_basis_usd` / per-name `daily_return_pct` from the PRIOR day's
        # `positions_snapshot.closing_price` (eod_reconcile.py `prior_positions`),
        # so correcting day d leaves d+1's already-frozen
        # `consolidated/{d+1}/eod_report.json` derived from the OLD basis —
        # permanently, because d+1's own stored closes are all fine and no leg
        # ever fires for it (alpha-engine-config-I10288).
        #
        # This is a derived-input change, not a divergence in anything stored
        # for THIS day, so it carries 0.0 bps: it must trigger a re-reconcile
        # without being mistaken for a corruption or dominating `reason`.
        # Computed here rather than inside `_detect_stale_legs` so it is NOT
        # re-asserted by the post-correction verification (the prior stays
        # "corrected in this pass" for the whole run).
        prior_eod = conn.execute(
            "SELECT date FROM eod_pnl WHERE date < ? ORDER BY date DESC LIMIT 1", (d,)
        ).fetchone()
        if prior_eod and prior_eod[0] in corrected_dates:
            legs.append({"leg": f"prior_corrected:{prior_eod[0]}", "divergence_bps": 0.0,
                         "stored": f"basis from {prior_eod[0]} pre-correction",
                         "settled": f"basis from {prior_eod[0]} post-correction"})
            legs.sort(key=lambda leg: -_mag(leg["divergence_bps"]))

        if not legs:
            continue  # clean — own close, recomputed spy_return, and every held ticker's close all match

        stored_close, stored_spy_return = row[0], row[1]
        stale_tickers = {
            leg["leg"].split(":", 1)[1]: leg["divergence_bps"]
            for leg in legs if leg["leg"].startswith("position_close:")
        }

        # ``divergence_bps`` is the MAXIMUM over every stale leg, and ``reason``
        # names the leg that produced it. It used to be whichever leg was
        # checked first (spy_close, then spy_return, then positions), so a 1.2bp
        # SPY drift alongside a 40bp held-ticker corruption reported
        # ``stale_close``/1.2bp — and since the paging decision reads
        # ``divergence_bps``, the 40bp corruption did not page
        # (alpha-engine-config-I10288). ``reasons`` lists every triggered leg
        # class so nothing is dropped from the record either.
        divergence_bps = max(_mag(leg["divergence_bps"]) for leg in legs)
        reason = _LEG_REASON[legs[0]["leg"].split(":", 1)[0]]
        reasons = list(dict.fromkeys(_LEG_REASON[leg["leg"].split(":", 1)[0]] for leg in legs))
        classification = _classify(divergence_bps, page_threshold_bps=page_threshold_bps)
        before = {"spy_close": stored_close, "spy_return_pct": stored_spy_return,
                  "daily_alpha_pct": row[2], "stale_tickers": stale_tickers or None}

        logger.warning(
            "[reconcile_audit] %s needs correction [%s] (%s; %d stale leg(s)): %s — max divergence "
            "%sbp vs tolerance %.2fbp%s",
            d, classification, "+".join(reasons), len(legs), _describe_legs(legs),
            _fmt_bps(divergence_bps), tolerance_bps, " [dry-run]" if dry_run else "",
        )

        if dry_run:
            corrected.append({"date": d, "reason": reason, "reasons": reasons,
                              "classification": classification, "legs": legs,
                              "divergence_bps": divergence_bps,
                              "before": before, "settled_spy_close": settled, "applied": False})
            corrected_dates.add(d)
            continue

        # ── Apply the canonical correction (re-price + re-emit from settled data).
        try:
            eod_run(d, send_email=send_email, run_audit=False)
        except Exception as e:  # noqa: BLE001 — per-date isolation: one bad day must not abort the sweep
            _log_paged("[reconcile_audit] correction FAILED for %s: %s", d, e)
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

        # ── Verify the correction actually took. The re-reconcile's `after`
        # values used to be recorded and never checked, so a re-price that did
        # NOT converge was written to the audit trail as a completed correction
        # (alpha-engine-config-I10288). Re-run the SAME detector against the
        # now-rewritten row: any leg still divergent means the self-heal did not
        # heal, eod_pnl is knowably wrong, and no later pass fixes it — this
        # pass would simply re-detect and re-fail it every day. Collected here
        # and RAISED after the window completes, so per-date isolation still
        # records every finding first.
        _, residual_legs = _detect_stale_legs(
            conn, d, config, tolerance_bps=tolerance_bps, settled=settled)

        # Page-worthiness: a CORRUPTION-classified correction (the same
        # at/above-PAGE_THRESHOLD_BPS band as before), OR a correction that
        # isn't the first in this run (multiple dates drifting together is
        # systemic, not routine settlement lag) — see PAGE_THRESHOLD_BPS.
        # Every flow-doctor severity level maps to SOME Telegram notifier in
        # flow-doctor.yaml (critical→#critical, error/warning→#ops-health,
        # info→#trades) — there is no "silent" severity to pick. So routine,
        # in-band revisions skip the fd.report() call entirely rather than
        # trying to pick a severity that happens not to page; they still get
        # the full S3 audit trail (below) and a local INFO log line.
        #
        # NOT a swallow of the failure mode: (a) what is not paged is a bounded
        # vendor revision that this pass has already corrected AND verified as
        # converged — the unconverged case below is unconditionally paged at
        # severity=critical and raises; (b) the primary deliverable (a correct
        # eod_pnl row) is delivered before the paging decision is even made;
        # (c) the recording surface is the S3 correction record at
        # trades/eod_corrections/{date}.json plus the INFO log line, both of
        # which carry the full leg detail and the classification.
        is_recurrence = len(corrected) >= 1
        page_worthy = classification == CLASSIFICATION_CORRUPTION or is_recurrence
        page_reason = (
            f"classification={CLASSIFICATION_CORRUPTION} "
            f"({_fmt_bps(divergence_bps)}bp >= {page_threshold_bps:.2f}bp threshold)"
            if classification == CLASSIFICATION_CORRUPTION
            else "second-or-later correction in this pass (systemic, not settlement lag)"
            if is_recurrence else None
        )

        record = {
            "date": d,
            "reason": reason,
            "reasons": reasons,
            "classification": classification,
            "legs": legs,
            "residual_legs": residual_legs,
            "converged": not residual_legs,
            "tolerance_bps": tolerance_bps,
            "page_threshold_bps": page_threshold_bps,
            "divergence_bps": divergence_bps,
            "paged": page_worthy,
            "page_reason": page_reason,
            "settled_spy_close": settled,
            "before": before,
            "after": after,
            "downstream": DOWNSTREAM_ON_CORRECTION,
            "source": "arcticdb_macro",
            "corrected_at": datetime.now(UTC).isoformat(),
        }
        _write_audit_record(trades_bucket=trades_bucket, run_date=d, record=record, region=region)
        correction_message = (
            f"EOD {d} corrected post-settlement [{classification}] "
            f"({'+'.join(reasons)}; {len(legs)} stale leg(s)): {_describe_legs(legs)}. "
            f"max divergence {_fmt_bps(divergence_bps)}bp "
            f"(tolerance {tolerance_bps:.2f}bp, page threshold {page_threshold_bps:.2f}bp)"
            + (f". PAGED: {page_reason}" if page_worthy else "")
        )
        if page_worthy:
            if fd:
                # A plain string, not an exception: this is a COMPLETED,
                # verified self-heal, and wrapping it in RuntimeError() made a
                # successful correction indistinguishable on the alert surface
                # from an aborted run (alpha-engine-config-I10288 — the
                # 2026-09-04 page was read as a crash). FlowDoctor.report()
                # accepts a string and never raises.
                fd.report(
                    correction_message,
                    severity="warning",
                    context={"site": "reconcile_audit_corrected", "run_date": d, "reason": reason,
                             "reasons": reasons, "classification": classification,
                             "legs": legs, "converged": not residual_legs,
                             "divergence_bps": divergence_bps, "page_threshold_bps": page_threshold_bps},
                )
        else:
            logger.info("[reconcile_audit] %s — in-band, audit trail only, not paged", correction_message)

        if residual_legs:
            residual_message = (
                f"EOD {d} re-reconcile did NOT converge: {len(residual_legs)} leg(s) still diverge "
                f"from settled after correction: {_describe_legs(residual_legs)}. "
                f"eod_pnl for {d} is knowably wrong and no later pass corrects it."
            )
            _log_paged("[reconcile_audit] %s", residual_message)
            if fd:
                fd.report(
                    residual_message,
                    severity="critical",
                    context={"site": "reconcile_audit_unconverged", "run_date": d,
                             "residual_legs": residual_legs, "tolerance_bps": tolerance_bps},
                )
            unconverged.append({"date": d, "residual_legs": residual_legs})

        corrected.append({"date": d, "reason": reason, "reasons": reasons,
                          "classification": classification, "legs": legs,
                          "residual_legs": residual_legs, "converged": not residual_legs,
                          "divergence_bps": divergence_bps,
                          "before": before, "after": after, "applied": True, "paged": page_worthy})
        corrected_dates.add(d)

    conn.close()
    summary = {
        "checked": checked,
        "corrected": corrected,
        "skipped": skipped,
        "gaps": gaps,
        "unconverged": unconverged,
        "revisions": [c["date"] for c in corrected
                      if c.get("classification") == CLASSIFICATION_REVISION],
        "corruptions": [c["date"] for c in corrected
                        if c.get("classification") == CLASSIFICATION_CORRUPTION],
        "tolerance_bps": tolerance_bps,
        "page_threshold_bps": page_threshold_bps,
        "dry_run": dry_run,
        "window": dates,
    }
    logger.info("[reconcile_audit] done: checked=%d corrected=%d (revisions=%d corruptions=%d) "
                "gaps=%d skipped=%d unconverged=%d dry_run=%s",
                checked, len(corrected), len(summary["revisions"]), len(summary["corruptions"]),
                len(gaps), len(skipped), len(unconverged), dry_run)

    # Fail loud, after every date has been checked and recorded. A correction
    # that did not converge is the one outcome here that is NOT self-healing:
    # the pass ran, wrote, verified, and the value is still wrong.
    if unconverged:
        raise ReconciliationUnconvergedError(
            "reconcile_audit could not converge "
            f"{len(unconverged)} date(s) against settled data: "
            + "; ".join(f"{u['date']} ({_describe_legs(u['residual_legs'])})" for u in unconverged),
            summary=summary,
        )
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
