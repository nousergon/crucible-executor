"""Regression: no module in this repo reads a secret via ``os.environ.get``
or ``os.getenv``.

After the 2026-05-12 ``.env`` → SSM migration (PR 6 of the arc), every
secret-bearing call site routes through ``nousergon_lib.secrets.get_secret()``.
This test re-greps the codebase on every CI run so a future commit can't
silently re-introduce a secret read via ``os.environ``.

Non-secret env vars are allowed for now — they migrate to alpha-engine-config
YAML in PR 8 of the arc.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PINNED_SECRETS = frozenset(
    [
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "VOYAGE_API_KEY",
        "POLYGON_API_KEY",
        "FMP_API_KEY",
        "FINNHUB_API_KEY",
        "FRED_API_KEY",
        "GMAIL_APP_PASSWORD",
        "GITHUB_TOKEN",
        "RAG_DATABASE_URL",
        "EDGAR_IDENTITY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        # Added by #890 (.env full deprecation): the daemon's email-config
        # secrets and flow-doctor's GitHub token. The daemon systemd unit no
        # longer has an EnvironmentFile, so these MUST come from get_secret().
        "EMAIL_SENDER",
        "EMAIL_RECIPIENTS",
        "FLOW_DOCTOR_GITHUB_TOKEN",
    ]
)

# Catches both ``os.environ.get("X")`` / ``os.getenv("X")`` and the subscript
# form ``os.environ["X"]`` so a re-introduced secret read is caught either way.
_ENV_READ_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
    r'|os\.environ\[\s*["\']([A-Z_][A-Z0-9_]*)["\']'
)


def _iter_python_files():
    for path in _REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "trading-env", "build", "tests", "node_modules", "package"}:
            continue
        yield path


def test_no_secret_environ_reads():
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ENV_READ_RE.finditer(line):
                name = match.group(1) or match.group(2)
                if name in _PINNED_SECRETS:
                    violations.append((path.relative_to(_REPO_ROOT), lineno, name))
    assert not violations, (
        "Found os.environ.get / os.getenv reads of pinned secrets — use "
        "`from nousergon_lib.secrets import get_secret` instead:\n"
        + "\n".join(f"  {p}:{ln}  {name}" for p, ln, name in violations)
    )


def test_daemon_unit_has_no_env_file():
    """Regression for #890: the intraday daemon systemd unit must NOT carry an
    ``EnvironmentFile=`` directive. The daemon obtains every secret at runtime
    via ``get_secret()`` (SSM-backed); a re-introduced ``EnvironmentFile`` would
    revive the deprecated ``.env`` plumbing this issue removed. Mirrors
    alpha-engine-morning.service, which already runs with no EnvironmentFile.
    """
    unit = _REPO_ROOT / "infrastructure" / "systemd" / "alpha-engine-daemon.service"
    assert unit.exists(), f"daemon unit not found at {unit}"
    offending = [
        line
        for line in unit.read_text().splitlines()
        if line.strip().startswith("EnvironmentFile")
    ]
    assert not offending, (
        "alpha-engine-daemon.service must not declare EnvironmentFile after "
        "#890 (.env deprecated → SSM). Offending line(s):\n"
        + "\n".join(f"  {ln}" for ln in offending)
    )
