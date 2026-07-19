"""Expectancy-gated de-risk sizing enforcement (alpha-engine-config#1259 /
config#2071 / config-I2820).

Standing operator de-risk stance (``halt-or-derisk-live-deployment``,
migrated from ROADMAP.md via config#978): when the signal chain's
forward-looking expectancy metrics breach their configured red-lines, the
executor holds a de-risked sizing posture — position sizes are capped to
``derisk_sizing_multiplier`` (config-default 0.50 = 50% of nominal) and the
MVO optimizer's ``risk_aversion`` is floored at
``configured_risk_aversion / derisk_sizing_multiplier`` — until the metrics
clear their thresholds.

Config contract (three new ``executor/risk.yaml`` top-level blocks, added by
alpha-engine-config-PR2071 — this module tolerates their absence, since this
PR necessarily lands before PR2071 merges the schema; see module-level
``DERISK_ON_EXPECTANCY_ENABLED`` default):

    derisk_on_expectancy_enabled: true
    derisk_sizing_multiplier: 0.50
    derisk_expectancy_thresholds:
      alpha_vs_spy: -0.05              # must be >= this
      information_ratio_ci_lower: -7   # must be >= this
      sharpe_ratio: 0.5                # must be >= this

Signal-chain source: ``director/carryover_ledger.json`` (S3, updated weekly
Saturday SF) — read here the same way ``executor/champion.py`` reads
``config/producer_champion.json``: plain ``boto3`` GET off ``signals_bucket``,
no ArcticDB, no new shared-lib S3-reader dependency (none exists in this repo
to reuse — champion.py's inline boto3-GET-plus-fail-loud-parse convention is
the established pattern and is followed verbatim here).

Fail-loud posture (per the operator de-risk stance — a de-risk decision must
never silently fall through to full sizing): when
``derisk_on_expectancy_enabled`` is true, ANY of the following raises
``DeriskGateConfigError`` rather than defaulting to gate-inactive/full-sizing:
  * ``derisk_sizing_multiplier`` missing, non-numeric, or outside (0, 1].
  * ``derisk_expectancy_thresholds`` missing or not a dict.
  * the ledger's ``halt-or-derisk-live-deployment`` entry is missing any of
    the three required metrics, or a metric value is non-numeric.
  * the carryover ledger S3 object is unreadable or malformed JSON.

When the flag is false (default False — ships dormant until config#2071
merges and an operator flips it on), this module is a complete no-op: no S3
read, gate always inactive, multiplier always 1.0 — bit-identical to
pre-this-PR behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

CARRYOVER_LEDGER_KEY = "director/carryover_ledger.json"
STANCE_ENTRY_NAME = "halt-or-derisk-live-deployment"

# Default OFF: config#2071 (the counterpart PR that introduces these risk.yaml
# blocks) has not merged as of this PR — a risk.yaml without the new blocks
# must behave exactly as it did before this module existed. Once config#2071
# merges and sets ``derisk_on_expectancy_enabled: true`` in the live config,
# the gate becomes live.
DERISK_ON_EXPECTANCY_ENABLED_DEFAULT = False

REQUIRED_THRESHOLD_KEYS = ("alpha_vs_spy", "information_ratio_ci_lower", "sharpe_ratio")


class DeriskGateConfigError(RuntimeError):
    """Raised when ``derisk_on_expectancy_enabled`` is true but the gate's
    config blocks or its S3-sourced expectancy metrics are missing/malformed.

    Deliberately never raised when the flag is false/absent — an operator
    who hasn't opted into the gate yet must not be blocked by it. Once
    opted in, every ambiguous condition raises rather than silently
    resolving to full (un-derisked) sizing — a de-risk decision must be as
    reliable as the rules it's gating.
    """


@dataclass(frozen=True)
class DeriskGateState:
    """Result of evaluating the expectancy-gated de-risk stance for one
    planning cycle.

    ``active``: True iff the gate fired (any metric breached its red-line).
    ``sizing_multiplier``: 1.0 when inactive/disabled; ``derisk_sizing_multiplier``
        (e.g. 0.50) when active — the caller multiplies position-level sizing
        by this value, composed with the other sizing adjustments the same
        way ``dd_multiplier`` (drawdown) already is.
    ``risk_aversion_floor``: ``None`` when inactive/disabled; otherwise
        ``configured_risk_aversion / derisk_sizing_multiplier`` — the caller
        clamps the MVO ``risk_aversion`` cfg to be AT LEAST this floor (a
        higher risk_aversion is a MORE conservative/lower-vol MVO solve, so
        "cap risk_aversion at the floor" means "never solve less
        conservatively than the floor").
    ``triggering_metrics``: names of metrics that breached their threshold
        (empty when inactive).
    ``metrics``: the raw metric values read from the ledger (empty when
        disabled — no S3 read performed).
    ``thresholds``: the configured red-line thresholds (empty when disabled).
    ``reason``: human-readable summary for logs / decision artifacts.
    """

    active: bool
    sizing_multiplier: float
    risk_aversion_floor: float | None
    triggering_metrics: tuple[str, ...]
    metrics: dict
    thresholds: dict
    reason: str
    enabled: bool = False
    context: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        """Flatten to a JSON-serializable dict for the structured risk-event
        sink (``trade_logger.log_risk_event``'s ``context`` kwarg) and the S3
        decision-artifact writer."""
        return {
            "enabled": self.enabled,
            "active": self.active,
            "sizing_multiplier": self.sizing_multiplier,
            "risk_aversion_floor": self.risk_aversion_floor,
            "triggering_metrics": list(self.triggering_metrics),
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "reason": self.reason,
            **self.context,
        }


_INACTIVE_DISABLED = DeriskGateState(
    active=False,
    sizing_multiplier=1.0,
    risk_aversion_floor=None,
    triggering_metrics=(),
    metrics={},
    thresholds={},
    reason="derisk_on_expectancy_enabled is false — gate disabled",
    enabled=False,
)


def _load_carryover_ledger(bucket: str, s3_client=None) -> dict:
    """Read ``s3://{bucket}/director/carryover_ledger.json``.

    No pre-bootstrap default (unlike ``champion.py``'s pointer read) — the
    de-risk stance is only evaluated when the operator has explicitly
    enabled it via ``derisk_on_expectancy_enabled: true``, so a missing
    ledger at that point is itself an ambiguous/unsafe condition, not a
    legitimate "nothing has been promoted yet" state. Every failure mode
    (S3 error, malformed JSON, non-dict payload) raises
    ``DeriskGateConfigError``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=CARRYOVER_LEDGER_KEY)
    except ClientError as e:
        raise DeriskGateConfigError(
            f"derisk_on_expectancy_enabled=true but carryover ledger "
            f"s3://{bucket}/{CARRYOVER_LEDGER_KEY} is unreadable: {e}. "
            "Refusing to silently fall back to full sizing."
        ) from e

    try:
        raw = obj["Body"].read()
        ledger = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
        raise DeriskGateConfigError(
            f"Carryover ledger s3://{bucket}/{CARRYOVER_LEDGER_KEY} is "
            f"malformed: {e}"
        ) from e

    if not isinstance(ledger, dict):
        raise DeriskGateConfigError(
            f"Carryover ledger s3://{bucket}/{CARRYOVER_LEDGER_KEY} did not "
            f"parse to a JSON object (got {type(ledger).__name__})"
        )

    return ledger


def _extract_stance_metrics(ledger: dict, bucket: str) -> dict:
    """Find the ``halt-or-derisk-live-deployment`` entry in the ledger and
    extract the three required expectancy metrics.

    The ledger's top-level shape is ``{"stances": [{"name": ..., ...}, ...]}``
    OR a flat ``{stance_name: {...}}`` mapping — tolerate either since
    PR2071's body does not pin an exact ledger schema beyond "find the
    halt-or-derisk-live-deployment entry". Raises ``DeriskGateConfigError``
    on a missing entry or missing/non-numeric metric.
    """
    entry = None
    if "stances" in ledger and isinstance(ledger["stances"], list):
        for candidate in ledger["stances"]:
            if isinstance(candidate, dict) and candidate.get("name") == STANCE_ENTRY_NAME:
                entry = candidate
                break
    elif STANCE_ENTRY_NAME in ledger and isinstance(ledger[STANCE_ENTRY_NAME], dict):
        entry = ledger[STANCE_ENTRY_NAME]

    if entry is None:
        raise DeriskGateConfigError(
            f"Carryover ledger s3://{bucket}/{CARRYOVER_LEDGER_KEY} has no "
            f"{STANCE_ENTRY_NAME!r} entry — cannot evaluate the de-risk gate."
        )

    metrics_src = entry.get("metrics", entry)
    metrics: dict = {}
    for key in REQUIRED_THRESHOLD_KEYS:
        if key not in metrics_src:
            raise DeriskGateConfigError(
                f"{STANCE_ENTRY_NAME!r} entry in s3://{bucket}/"
                f"{CARRYOVER_LEDGER_KEY} is missing required metric {key!r}."
            )
        try:
            metrics[key] = float(metrics_src[key])
        except (TypeError, ValueError) as e:
            raise DeriskGateConfigError(
                f"{STANCE_ENTRY_NAME!r} entry metric {key!r}="
                f"{metrics_src[key]!r} is non-numeric: {e}"
            ) from e

    return metrics


def _validate_thresholds(thresholds: object) -> dict:
    if not isinstance(thresholds, dict):
        raise DeriskGateConfigError(
            "derisk_expectancy_thresholds must be a mapping of metric name "
            f"-> red-line float; got {type(thresholds).__name__}"
        )
    out: dict = {}
    for key in REQUIRED_THRESHOLD_KEYS:
        if key not in thresholds:
            raise DeriskGateConfigError(
                f"derisk_expectancy_thresholds is missing required key {key!r} "
                f"(expected all of {REQUIRED_THRESHOLD_KEYS})"
            )
        try:
            out[key] = float(thresholds[key])
        except (TypeError, ValueError) as e:
            raise DeriskGateConfigError(
                f"derisk_expectancy_thresholds[{key!r}]={thresholds[key]!r} "
                f"is non-numeric: {e}"
            ) from e
    return out


def _validate_multiplier(multiplier: object) -> float:
    try:
        m = float(multiplier)
    except (TypeError, ValueError) as e:
        raise DeriskGateConfigError(
            f"derisk_sizing_multiplier={multiplier!r} is non-numeric"
        ) from e
    if not (0.0 < m <= 1.0):
        raise DeriskGateConfigError(
            f"derisk_sizing_multiplier={m} out of range — must be in (0.0, 1.0] "
            "(a de-risk gate must not INCREASE sizing)"
        )
    return m


def evaluate_derisk_gate(
    config: dict,
    bucket: str | None = None,
    s3_client=None,
    ledger: dict | None = None,
) -> DeriskGateState:
    """Evaluate the expectancy-gated de-risk stance for this planning cycle.

    Args:
        config: the loaded ``risk.yaml`` dict (top-level — same object
            ``risk_guard``/``position_sizer`` already receive).
        bucket: the S3 signals bucket (``config["signals_bucket"]`` at the
            call site) to read ``director/carryover_ledger.json`` from.
            Required (and only read) when the gate is enabled.
        s3_client: optional injected boto3 S3 client (test seam, mirrors
            ``champion.py``'s ``s3_client`` kwarg).
        ledger: optional pre-fetched ledger dict — skips the S3 round-trip
            when the caller already has it in hand (test seam / future
            caching). When omitted (the common case), fetched here.

    Returns a ``DeriskGateState``. When
    ``config.get("derisk_on_expectancy_enabled", False)`` is falsy, returns
    the shared ``_INACTIVE_DISABLED`` sentinel immediately — zero S3 I/O,
    bit-identical to pre-this-PR behavior for every deployment that hasn't
    opted in yet.

    Raises ``DeriskGateConfigError`` when enabled and any config block or
    ledger metric is missing/malformed — see module docstring's fail-loud
    contract. Never silently degrades to full sizing while enabled.
    """
    enabled = bool(config.get("derisk_on_expectancy_enabled", DERISK_ON_EXPECTANCY_ENABLED_DEFAULT))
    if not enabled:
        return _INACTIVE_DISABLED

    if "derisk_sizing_multiplier" not in config:
        raise DeriskGateConfigError(
            "derisk_on_expectancy_enabled=true but derisk_sizing_multiplier "
            "is not configured — refusing to silently default to full sizing."
        )
    multiplier = _validate_multiplier(config["derisk_sizing_multiplier"])

    if "derisk_expectancy_thresholds" not in config:
        raise DeriskGateConfigError(
            "derisk_on_expectancy_enabled=true but derisk_expectancy_thresholds "
            "is not configured — refusing to silently default to full sizing."
        )
    thresholds = _validate_thresholds(config["derisk_expectancy_thresholds"])

    if ledger is None:
        if not bucket:
            raise DeriskGateConfigError(
                "derisk_on_expectancy_enabled=true but no bucket was supplied "
                "to read director/carryover_ledger.json from."
            )
        ledger = _load_carryover_ledger(bucket, s3_client=s3_client)

    metrics = _extract_stance_metrics(ledger, bucket or "<injected-ledger>")

    triggering = tuple(
        key for key in REQUIRED_THRESHOLD_KEYS if metrics[key] < thresholds[key]
    )
    active = len(triggering) > 0

    configured_risk_aversion = float(
        (config.get("portfolio_optimizer") or {}).get("risk_aversion", 5.0)
    )
    risk_aversion_floor = (configured_risk_aversion / multiplier) if active else None
    sizing_multiplier = multiplier if active else 1.0

    if active:
        reason = (
            f"DE-RISK ACTIVE: {', '.join(triggering)} breached red-line "
            f"({', '.join(f'{k}={metrics[k]:.4f}<{thresholds[k]:.4f}' for k in triggering)}) "
            f"— sizing capped to {sizing_multiplier:.0%} of nominal, MVO "
            f"risk_aversion floored at {risk_aversion_floor:.2f}"
        )
        logger.warning("[derisk_gate] %s", reason)
    else:
        reason = (
            f"de-risk inactive — all metrics clear red-lines "
            f"({', '.join(f'{k}={metrics[k]:.4f}>={thresholds[k]:.4f}' for k in REQUIRED_THRESHOLD_KEYS)})"
        )
        logger.info("[derisk_gate] %s", reason)

    return DeriskGateState(
        active=active,
        sizing_multiplier=sizing_multiplier,
        risk_aversion_floor=risk_aversion_floor,
        triggering_metrics=triggering,
        metrics=metrics,
        thresholds=thresholds,
        reason=reason,
        enabled=True,
    )


def apply_risk_aversion_floor(risk_aversion: float, gate: DeriskGateState) -> float:
    """Clamp ``risk_aversion`` to the gate's floor when active.

    Higher risk_aversion = more conservative MVO solve, so "cap at the
    floor" means ``max(risk_aversion, floor)`` — the de-risk gate can only
    push the solver MORE conservative, never override a tuner/config value
    that's already more conservative than the floor.
    """
    if gate.risk_aversion_floor is None:
        return risk_aversion
    return max(float(risk_aversion), float(gate.risk_aversion_floor))
