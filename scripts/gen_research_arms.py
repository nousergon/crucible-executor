"""Generate ``executor/_research_arms.py`` from the research slot's arm register.

alpha-engine-config-I11442. ``executor.champion.SHADOW_SERVED_ARMS`` used to
carry the research slot's arm names as a hand-typed tuple — the FOURTH copy of
that set, and the same shape that had already gone stale twice in that module
(``ARMS_REQUIRING_SECTOR_MAP``, ``VALID_CHAMPIONS``). A stale copy here is not a
degraded run: a pointer naming an arm the executor cannot serve raises
``ChampionPointerError`` at planner start and HALTS TRADING
(alpha-engine-config-I11438).

The set is derived at BUILD time, not at runtime, because the executor must not
depend on an S3 read to decide whether it may start a trading day — a transient
read failure would either halt a healthy run or fail open. So the derivation
lives here, and drift is caught where it is cheap: ``--check`` runs in CI
(``.github/workflows/research-arms-drift.yml``) on every PR, every push to main
and daily, and fails when the register carries an arm this repo cannot serve.

Sources (``--register``), all read-only:

* default — crucible-research's committed genesis register
  (``scoring/arena/research_register.json``), which that repo's own suite
  asserts equals its live producer registry. Public, so CI needs no credential.
* ``s3://alpha-engine-research/arena/research/register.json`` — the live,
  append-only register the arena cycle maintains. Needs AWS read credentials.
* any local file path.

Semantics:

* The generated set is the UNION of what is committed and what the register
  carries. It never shrinks on regeneration: a RETIRED arm must stay servable
  while it is still the pointer's champion (alpha-engine-config-I11436), and
  servable is not selected — the arena only promotes active arms.
* A read failure, a malformed register, or a register with no research-slot
  arm exits non-zero and writes NOTHING. It can neither empty the served set
  nor admit an arm the register does not name.

Usage::

    python scripts/gen_research_arms.py            # regenerate, then commit
    python scripts/gen_research_arms.py --check    # CI drift guard
    python scripts/gen_research_arms.py --register s3://alpha-engine-research/arena/research/register.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "executor" / "_research_arms.py"

RESEARCH_SLOT = "research"
DEFAULT_REGISTER = (
    "https://raw.githubusercontent.com/nousergon/crucible-research/main/scoring/arena/research_register.json"
)
LIVE_REGISTER = "s3://alpha-engine-research/arena/research/register.json"
CONSTANT = "RESEARCH_SLOT_ARMS"

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_UNREADABLE = 2


class RegisterReadError(RuntimeError):
    """The register could not be read or does not have the register's shape."""


def read_register(source: str) -> list[dict[str, Any]]:
    """The register's event list, from a file path, an https URL or an s3:// URI."""
    try:
        if source.startswith("https://"):
            with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310 — scheme pinned to https above
                body = resp.read()
        elif source.startswith("s3://"):
            import boto3

            bucket, _, key = source[len("s3://") :].partition("/")
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        else:
            body = Path(source).read_bytes()
        payload = json.loads(body)
    except Exception as exc:  # noqa: BLE001 — every read failure is the same outcome: write nothing
        raise RegisterReadError(f"could not read the arm register from {source}: {exc}") from exc
    if not isinstance(payload, list):
        raise RegisterReadError(f"the arm register at {source} is a {type(payload).__name__}, not an event list")
    return payload


def arms_from_register(events: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Every arm name the research slot has EVER registered, sorted.

    Retirement events are deliberately not subtracted: see the module docstring.
    A register naming no research-slot arm is refused rather than read as
    "serve nothing" — that is a read of the wrong thing, not a slot state.
    """
    names: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping) or "kind" not in event:
            raise RegisterReadError(f"malformed register event: {event!r}")
        if event["kind"] != "registered":
            continue
        record = event.get("record")
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str) or not record["name"]:
            raise RegisterReadError(f"registered event carries no arm name: {event!r}")
        if record.get("slot") != RESEARCH_SLOT:
            continue
        names.add(record["name"])
    if not names:
        raise RegisterReadError(
            f"the register names no {RESEARCH_SLOT!r}-slot arm; refusing to derive an empty served set"
        )
    return tuple(sorted(names))


def committed_arms(path: Path = OUTPUT_PATH) -> tuple[str, ...]:
    """The arms the committed module declares, read without importing ``executor``."""
    if not path.exists():
        return ()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == CONSTANT and node.value is not None:
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{path} does not assign {CONSTANT}")


def render(arms: Iterable[str]) -> str:
    """The generated module's exact text. Deterministic: no timestamp, no source."""
    lines = [
        '"""GENERATED by ``scripts/gen_research_arms.py`` — do not edit by hand.',
        "",
        "The research slot's arms, derived from its arm register",
        "(``arena/research/register.json``, alpha-engine-config-I11442), for",
        "``executor.champion.SHADOW_SERVED_ARMS``. Committed rather than read at",
        "runtime so that no S3 read sits on the trading-start path;",
        "``.github/workflows/research-arms-drift.yml`` fails when the register",
        "carries an arm this tuple does not.",
        "",
        "Regenerate with ``python scripts/gen_research_arms.py`` and commit.",
        '"""',
        "",
        f"{CONSTANT}: tuple[str, ...] = (",
        *(f'    "{arm}",' for arm in arms),
        ")",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--register", default=DEFAULT_REGISTER, help="file path, https URL or s3:// URI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if the register carries an arm the committed module does not",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        registered = arms_from_register(read_register(args.register))
    except RegisterReadError as exc:
        print(f"::error::{exc}. Nothing was written.", file=sys.stderr)
        return EXIT_UNREADABLE

    committed = committed_arms(args.output)
    missing = sorted(set(registered) - set(committed))
    extra = sorted(set(committed) - set(registered))

    if args.check:
        if extra:
            print(
                f"::notice::{extra} are served but not in {args.register} — kept servable on purpose "
                "(a retired arm may still be the pointer's champion, alpha-engine-config-I11436)."
            )
        if missing:
            print(
                f"::error::the research slot registers {missing}, which the executor cannot serve: a "
                "promotion onto one raises at planner start and halts trading (alpha-engine-config-I11438). "
                "Run `python scripts/gen_research_arms.py` and commit executor/_research_arms.py.",
                file=sys.stderr,
            )
            return EXIT_DRIFT
        print(f"ok: all {len(registered)} registered research-slot arms are servable")
        return EXIT_OK

    text = render(sorted(set(committed) | set(registered)))
    if args.output.exists() and args.output.read_text() == text:
        print(f"{args.output} is up to date")
        return EXIT_OK
    args.output.write_text(text)
    print(f"wrote {args.output} (added {missing})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
