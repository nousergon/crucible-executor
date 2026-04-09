"""
Structured logging configuration for the executor.

JSON mode activates when ALPHA_ENGINE_JSON_LOGS=1 (set on EC2 via systemd env).
Text mode (default) preserves the current human-readable format for local dev.

Flow Doctor integration: attaches an error-monitoring handler that captures
ERROR+ log records, deduplicates, diagnoses via LLM, and creates GitHub issues.
Enabled when FLOW_DOCTOR_ENABLED=1 (default on EC2).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc"] = self.formatException(record.exc_info)
        # Merge extra context if provided via logger.info("msg", extra={"ctx": {...}})
        if hasattr(record, "ctx"):
            log_entry["ctx"] = record.ctx
        return json.dumps(log_entry, default=str)


def _attach_flow_doctor(name: str) -> None:
    """Attach flow-doctor handler to root logger (ERROR+ only)."""
    try:
        import flow_doctor

        fd = flow_doctor.init(
            flow_name=f"alpha-engine-executor-{name}",
            repo="cipher813/alpha-engine",
            owner="@cipher813",
            store={"type": "sqlite", "path": "flow_doctor.db"},
            diagnosis={"enabled": True, "model": "claude-haiku-4-5-20251001"},
            notify=[{"type": "github", "repo": "cipher813/alpha-engine"}],
            rate_limits={
                "max_diagnosed_per_day": 5,
                "max_issues_per_day": 3,
                "dedup_cooldown_minutes": 120,
            },
        )
        handler = flow_doctor.FlowDoctorHandler(fd, level=logging.ERROR)
        logging.getLogger().addHandler(handler)
    except Exception:
        # flow-doctor is non-critical — never block executor startup
        logging.getLogger(__name__).debug("flow-doctor not available, skipping")


def setup_logging(name: str = "executor") -> None:
    """
    Configure root logger.

    JSON mode: ALPHA_ENGINE_JSON_LOGS=1 (for EC2 / production)
    Text mode: default (for local dev / dry-run)
    Flow Doctor: FLOW_DOCTOR_ENABLED=1 (for EC2 / production)
    """
    json_mode = os.environ.get("ALPHA_ENGINE_JSON_LOGS", "0") == "1"

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if os.environ.get("FLOW_DOCTOR_ENABLED", "0") == "1":
        _attach_flow_doctor(name)
