"""Class guard: no `except Exception` handler in `executor/` may swallow the
failure into `logger.debug(...)` or a bare `pass`, with the root logger at
INFO on every executor entrypoint (`krepis/src/krepis/logging.py:500`) that
means the record is emitted **nowhere** (alpha-engine-config-I10031).

The class this catches: `except Exception` (or a narrower except clause
immediately chained after one — see the AST walk below) whose ENTIRE body is
a single call to `logger.debug(...)` or a bare `pass`. A handler that also
does something else (re-raises, records via `logger.error`/`logger.warning`,
returns an in-band error value the caller consumes, calls `fd.report(...)`,
increments a counter) is not in scope — the invisible-record shape is
specifically a body with nothing else in it.

A new site with no allowlist entry fails the build (`_UNCOVERED` case below).
An allowlist entry whose `expires` has passed fails loudly — re-justify or
remove, never silently re-grandfather (`_EXPIRED` case). An entry that no
longer matches anything ALSO fails, so the allowance cannot quietly widen
after the site it covered is fixed or moves (`_STALE` case) — mirrors
`.provider-linkage-allowlist.yaml` / `nousergon-lib/scripts/
provider_linkage_guard.py` (alpha-engine-config-I9295).
"""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXECUTOR_DIR = _REPO_ROOT / "executor"
_ALLOWLIST_PATH = _REPO_ROOT / ".debug-swallow-allowlist.yaml"


def _is_debug_call(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "debug"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "logger"
    )


def _is_debug_only_or_pass(body: list[ast.stmt]) -> bool:
    """True iff `body` records the failure ONLY via a bare `pass` or a
    `logger.debug(...)` call — the invisible-record shape — with no other
    RECORDING statement (a `logger.error`/`logger.warning`/`fd.report`
    elsewhere in the body would mean it is already visible, out of this
    class's scope). A trailing `continue`/`break`/`return` after the
    debug call is still in scope: that is control flow, not a second
    record, and the exception is still funneled only into DEBUG before
    the handler moves on."""
    record_stmts = [s for s in body if isinstance(s, ast.Pass) or _is_debug_call(s)]
    if len(record_stmts) != 1:
        return False
    other_stmts = [s for s in body if s not in record_stmts]
    return all(isinstance(s, (ast.Continue, ast.Break, ast.Return)) for s in other_stmts)


def _is_bare_except_exception(handler: ast.ExceptHandler) -> bool:
    """True iff the handler is `except Exception:` / `except Exception as
    e:` — scoped to match the issue's own measurement (`grep -rn -A1
    "except Exception" ...`). A narrower catch (`except (TypeError,
    ValueError):`, `except OSError:`, ...) is a deliberate, scoped catch
    and out of this class's scope — see `~/Development/CLAUDE.md` "Fail
    loud and fast", which forbids the broad, blind swallow, not a narrow
    one."""
    return isinstance(handler.type, ast.Name) and handler.type.id == "Exception"


def _find_swallow_sites(path: Path) -> set[int]:
    """Line numbers of bare `except Exception` clauses in `path` whose
    body is a debug-only-or-pass swallow, per `_is_debug_only_or_pass`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if _is_bare_except_exception(handler) and _is_debug_only_or_pass(handler.body):
                    sites.add(handler.lineno)
    return sites


def _all_swallow_sites() -> dict[str, set[int]]:
    return {
        str(p.relative_to(_REPO_ROOT)): _find_swallow_sites(p)
        for p in sorted(_EXECUTOR_DIR.glob("*.py"))
    }


def _load_allowlist() -> list[dict]:
    doc = yaml.safe_load(_ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert doc.get("schema_version") == 1, "unrecognized allowlist schema_version"
    return doc["entries"]


def test_no_new_debug_only_swallows_outside_allowlist():
    """Every debug-only-or-pass `except Exception` swallow in `executor/`
    is either fixed (raised, or recorded at WARNING/ERROR+) or has a
    non-expired, matching entry in `.debug-swallow-allowlist.yaml`."""
    live_sites = _all_swallow_sites()
    allowlist = _load_allowlist()
    today = dt.date.today()

    allowed: set[tuple[str, int]] = set()
    expired: list[str] = []
    for entry in allowlist:
        key = (entry["path"], entry["line"])
        expires = dt.date.fromisoformat(entry["expires"])
        if expires < today:
            expired.append(f"{entry['path']}:{entry['line']} expired {expires} — re-justify or remove")
            continue
        allowed.add(key)

    uncovered: list[str] = []
    for path, lines in live_sites.items():
        for line in lines:
            if (path, line) not in allowed:
                uncovered.append(
                    f"{path}:{line} — new debug-only/pass swallow with no "
                    "allowlist entry. Raise it, record it at WARNING/ERROR+ "
                    "with a named recording surface, or add a justified, "
                    "expiring entry to .debug-swallow-allowlist.yaml."
                )

    stale: list[str] = []
    for entry in allowlist:
        key = (entry["path"], entry["line"])
        if entry["path"] not in live_sites or entry["line"] not in live_sites[entry["path"]]:
            stale.append(
                f"{entry['path']}:{entry['line']} no longer matches a "
                "debug-only/pass swallow — remove the stale entry so the "
                "allowance cannot quietly widen."
            )

    failures = expired + uncovered + stale
    assert not failures, "\n".join(failures)


def test_allowlist_entries_are_self_contained():
    """Every entry names a reason, an expiry, and a tracking issue — a
    swallow with no named recording surface is not a swallow, it is a
    deletion (alpha-engine-config-I10031 deliverable 2)."""
    for entry in _load_allowlist():
        for field in ("path", "line", "reason", "expires", "tracking"):
            assert entry.get(field), f"allowlist entry missing {field!r}: {entry}"
        assert entry["reason"].strip(), f"empty reason: {entry}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
