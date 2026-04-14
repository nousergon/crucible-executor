"""
Structured logging configuration for the executor.

JSON mode activates when ALPHA_ENGINE_JSON_LOGS=1 (set on EC2 via systemd env).
Text mode (default) preserves the current human-readable format for local dev.

Flow Doctor integration: owns the single shared FlowDoctor instance for the
entire executor process. All call sites (main.py, daemon.py, eod_reconcile.py)
should call ``get_flow_doctor()`` instead of calling ``flow_doctor.init()``
themselves — running four independent FlowDoctor instances with separate
SQLite stores, rate limiters, and dedup states is a footgun.

Enabled when FLOW_DOCTOR_ENABLED=1 (default on EC2).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import flow_doctor

_FLOW_DOCTOR_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "flow-doctor.yaml"
)

# IB Gateway error codes that are benign for a delayed-data paper-trading
# executor. IB emits these at ERROR level, but the daemon continues to
# receive delayed ticks via the delayedLast/delayedClose fallbacks in
# price_monitor.py. Suppress to prevent alert spam when Brian opens the
# IB iOS app during market hours (competing live session preempts the
# live feed; delayed keeps flowing).
_FLOW_DOCTOR_EXCLUDE_PATTERNS = [
    r"Error 10197",  # No market data during competing live session
]

# Singleton — populated once by setup_logging() and retrieved by call sites
# via get_flow_doctor(). None until setup_logging() runs with FLOW_DOCTOR_ENABLED=1.
_fd_instance: Optional[flow_doctor.FlowDoctor] = None


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


def get_flow_doctor() -> Optional[flow_doctor.FlowDoctor]:
    """Return the shared flow-doctor instance, or None if not initialized.

    Call sites use this to access flow-doctor without creating duplicate
    instances. Returns None if setup_logging() was never called with
    FLOW_DOCTOR_ENABLED=1, or if flow-doctor init failed.
    """
    return _fd_instance


def _attach_flow_doctor(name: str) -> None:
    """Initialize the shared flow-doctor instance and attach a log handler."""
    global _fd_instance
    _fd_instance = flow_doctor.init(config_path=_FLOW_DOCTOR_YAML_PATH)
    handler = flow_doctor.FlowDoctorHandler(
        _fd_instance,
        level=logging.ERROR,
        exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    )
    logging.getLogger().addHandler(handler)


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
