r"""config#2356: boot-pull.sh's trades.db S3-restore staleness guard.

Background — this is a follow-up fixing a REGRESSION introduced by PR#352
(35dace6, merged 2026-07-11), which set out to make the "does trades.db have
recent eod_pnl data" check genuinely recency-based instead of presence-only,
but shipped with every `$LOCAL_MAX_DATE` / `$RESTORE_NEEDED` reference
backslash-escaped inside double quotes:

    if [ -z "BACKSLASH-DOLLAR-LOCAL_MAX_DATE" ]; then   # always false: an
    ...                                                  # escaped "\$FOO"
    if [ "BACKSLASH-DOLLAR-RESTORE_NEEDED" = "true" ];  # inside double
    then                                                 # quotes is a
                                                          # literal, non-
                                                          # expanding string

Bash never expands an escaped `\$` inside double quotes, so both branches
compared a non-empty literal string against test conditions that could never
be satisfied. Net effect: the ENTIRE S3 restore-on-boot safety net was dead
code in production — worse than the original size-only bug PR#352 was meant
to fix (the original at least restored on a tiny/missing file).

These tests exercise the ACTUAL block from infrastructure/boot-pull.sh via a
real bash subprocess against fixture sqlite DBs (not a source-text regex —
a regex would not have caught the escaping bug, since `"\$LOCAL_MAX_DATE"`
and `"$LOCAL_MAX_DATE"` both "contain the variable name"). A dedicated regex
guard is included too, so this exact regression class can never silently
reoccur even if the subprocess harness is ever weakened.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_START_MARKER = "# ── Restore trades.db from S3 if missing or empty"
_END_MARKER = "# Sync systemd service files from repo"


def _extract_restore_block() -> str:
    """Pull the trades.db-restore block out of the live script by marker,
    not by hardcoded line numbers, so this test tracks the real file as it
    evolves."""
    src = _BOOT_PULL.read_text()
    assert _START_MARKER in src, "trades.db restore block start marker missing"
    assert _END_MARKER in src, "trades.db restore block end marker missing"
    start = src.index(_START_MARKER)
    end = src.index(_END_MARKER, start)
    block = src[start:end]
    assert "RESTORE_NEEDED" in block
    return block


def _with_test_risk_yaml(block: str, risk_yaml_path: Path) -> str:
    """Substitute the hardcoded RISK_YAML path for a fixture path.

    The block assigns RISK_YAML="/home/ec2-user/alpha-engine/config/risk.yaml"
    as its very first statement, so setting RISK_YAML before sourcing the
    block (rather than substituting inside it) would just get clobbered —
    this exercises the real grep/sed parsing against a fixture file instead
    of bypassing it.
    """
    real_path = "/home/ec2-user/alpha-engine/config/risk.yaml"
    assert f'RISK_YAML="{real_path}"' in block, (
        "expected boot-pull.sh's hardcoded RISK_YAML path — path may have "
        "changed, update this test"
    )
    return block.replace(f'RISK_YAML="{real_path}"', f'RISK_YAML="{risk_yaml_path}"')


def _make_db(path: Path, dates: list[str] | None) -> None:
    """Create a fixture trades.db. `dates=None` means no eod_pnl table rows
    (table exists but empty); an empty list is treated the same. Pads the
    file past the 20480-byte "minimal" threshold with filler rows in an
    unrelated table so only the eod_pnl-recency logic is under test."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE eod_pnl (date TEXT, pnl REAL)")
    for d in dates or []:
        c.execute("INSERT INTO eod_pnl VALUES (?, ?)", (d, 1.0))
    # Filler to push the file comfortably past the 20480B minimal-size gate,
    # which is a separate branch from the recency check under test here.
    c.execute("CREATE TABLE filler (blob TEXT)")
    for _i in range(2000):
        c.execute("INSERT INTO filler VALUES (?)", ("x" * 100,))
    conn.commit()
    conn.close()
    assert path.stat().st_size > 20480, "fixture DB did not clear the minimal-size gate"


def _run_restore_block(tmp_path: Path, db_path: Path, ae_venv_python: str) -> tuple[str, bool]:
    """Run the extracted restore block in a real bash subprocess.

    Stubs `log()` to append to a plain file (avoids needing /var/log
    writability) and stubs `aws` on PATH to a no-op script that records
    whether `s3 cp` was invoked, so we can assert on the actual
    restore/no-restore DECISION the block makes, not just its log text.
    """
    log_file = tmp_path / "boot-pull.log"
    aws_calls = tmp_path / "aws_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    # The block parses DB_PATH/TRADES_BUCKET out of $RISK_YAML itself (via
    # grep/sed), so drive it through a real fixture risk.yaml rather than
    # pre-setting those vars directly — that keeps this test exercising the
    # actual parsing path too, not just the recency logic downstream of it.
    risk_yaml = tmp_path / "risk.yaml"
    risk_yaml.write_text(f'db_path: "{db_path}"\ntrades_bucket: "fake-bucket"\n')
    block = _with_test_risk_yaml(_extract_restore_block(), risk_yaml)

    # Fake `aws` — records invocation, "succeeds" so downstream `[ -f
    # "$DB_PATH" ]` / size checks still see a file, without needing real S3.
    fake_aws = bin_dir / "aws"
    fake_aws.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" >> "{aws_calls}"
        exit 0
        """))
    fake_aws.chmod(0o755)

    script = textwrap.dedent(f"""\
        set -uo pipefail
        LOG="{log_file}"
        log() {{ echo "$*" >> "$LOG"; }}
        export PATH="{bin_dir}:$PATH"
        {block}
        """)

    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(tmp_path),
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "AE_TEST_VENV_PY": ae_venv_python},
        capture_output=True,
        text=True,
        timeout=30,
    )
    log_text = log_file.read_text() if log_file.exists() else ""
    restore_invoked = aws_calls.exists() and "s3 cp" in aws_calls.read_text()
    return proc.stdout + proc.stderr + log_text, restore_invoked


@pytest.fixture()
def ae_venv_python() -> str:
    """The interpreter running pytest right now — has nousergon_lib
    importable in this test environment, standing in for the alpha-engine
    venv the real script points at on the trading box."""
    return sys.executable


@pytest.fixture()
def patched_block(monkeypatch, ae_venv_python):
    """The live block hardcodes AE_VENV_PY="/home/ec2-user/alpha-engine/.venv/bin/python".
    That path won't exist in CI, so for the subprocess test we substitute
    the current interpreter — this only swaps the venv path, not any of the
    actual decision logic being tested."""

    original_extract = _extract_restore_block

    def _patched() -> str:
        block = original_extract()
        real_path = "/home/ec2-user/alpha-engine/.venv/bin/python"
        assert real_path in block, (
            "expected boot-pull.sh to reference the alpha-engine venv python "
            "directly (FD_VENV precedent) — path may have changed, update this test"
        )
        return block.replace(real_path, "${AE_TEST_VENV_PY}")

    monkeypatch.setattr(sys.modules[__name__], "_extract_restore_block", _patched)
    yield


def test_fresh_db_does_not_restore(tmp_path, ae_venv_python, patched_block):
    """A local DB whose eod_pnl max(date) is today/yesterday must NOT
    trigger a restore."""
    from nousergon_lib.trading_calendar import last_closed_trading_day

    db_path = tmp_path / "trades.db"
    _make_db(db_path, [str(last_closed_trading_day())])

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert not restored, f"fresh DB incorrectly triggered restore. Output:\n{output}"
    assert "no restore needed" in output, output


def test_stale_but_large_db_triggers_restore(tmp_path, ae_venv_python, patched_block):
    """The exact config#2356 scenario: a snapshot-restored DB with plenty
    of (old) eod_pnl rows — large enough to clear the size gate, but weeks
    stale. Must trigger a restore."""
    db_path = tmp_path / "trades.db"
    _make_db(db_path, ["2026-06-01", "2026-05-29", "2026-05-28"])

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert restored, f"stale-but-large DB failed to trigger restore. Output:\n{output}"
    assert "stale" in output, output


def test_one_trading_day_lag_is_tolerated(tmp_path, ae_venv_python, patched_block):
    """Yesterday's close is present but today's EOD ingestion simply hasn't
    run yet — must NOT false-positive as stale."""
    from nousergon_lib.trading_calendar import last_closed_trading_day, subtract_trading_days

    db_path = tmp_path / "trades.db"
    one_day_back = subtract_trading_days(last_closed_trading_day(), 1)
    _make_db(db_path, [str(one_day_back)])

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert not restored, f"1-trading-day lag incorrectly triggered restore. Output:\n{output}"


def test_two_trading_days_behind_triggers_restore(tmp_path, ae_venv_python, patched_block):
    """2+ trading days behind expected is genuinely stale, not just
    EOD-ingestion lag — must restore."""
    from nousergon_lib.trading_calendar import last_closed_trading_day, subtract_trading_days

    db_path = tmp_path / "trades.db"
    two_days_back = subtract_trading_days(last_closed_trading_day(), 2)
    _make_db(db_path, [str(two_days_back)])

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert restored, f"2-trading-day-stale DB failed to trigger restore. Output:\n{output}"


def test_no_eod_pnl_rows_triggers_restore(tmp_path, ae_venv_python, patched_block):
    """Empty eod_pnl table (no rows at all) must still restore — this is
    the original presence-only branch, kept as an ADDITIONAL check
    alongside (not replaced by) the recency check."""
    db_path = tmp_path / "trades.db"
    _make_db(db_path, [])

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert restored, f"empty eod_pnl DB failed to trigger restore. Output:\n{output}"
    assert "no eod_pnl data" in output, output


def test_missing_db_file_triggers_restore(tmp_path, ae_venv_python, patched_block):
    """Missing trades.db entirely must restore (pre-existing branch,
    unaffected by this fix — pinned so a future edit can't regress it)."""
    db_path = tmp_path / "does_not_exist.db"

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert restored, f"missing DB file failed to trigger restore. Output:\n{output}"
    assert "trades.db missing" in output, output


def test_minimal_size_db_triggers_restore(tmp_path, ae_venv_python, patched_block):
    """A DB at/under the 20480B minimal-size threshold must restore
    regardless of eod_pnl contents (pre-existing branch, unaffected by
    this fix — pinned)."""
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE eod_pnl (date TEXT, pnl REAL)")
    conn.commit()
    conn.close()
    assert db_path.stat().st_size <= 20480

    output, restored = _run_restore_block(tmp_path, db_path, ae_venv_python)
    assert restored, f"minimal-size DB failed to trigger restore. Output:\n{output}"


def test_s3_restore_failure_exits_nonzero(tmp_path, ae_venv_python, patched_block):
    """If the S3 restore itself fails, the script must fail loud (exit 1),
    not continue booting the executor against an empty/stale db. This half
    was already correct pre-fix and must stay correct."""
    log_file = tmp_path / "boot-pull.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    db_path = tmp_path / "does_not_exist.db"  # forces RESTORE_NEEDED=true
    risk_yaml = tmp_path / "risk.yaml"
    risk_yaml.write_text(f'db_path: "{db_path}"\ntrades_bucket: "fake-bucket"\n')
    block = _with_test_risk_yaml(_extract_restore_block(), risk_yaml)

    # This time the fake `aws` FAILS, simulating an S3 outage / bad creds.
    fake_aws = bin_dir / "aws"
    fake_aws.write_text("#!/bin/bash\nexit 1\n")
    fake_aws.chmod(0o755)

    script = textwrap.dedent(f"""\
        set -uo pipefail
        LOG="{log_file}"
        log() {{ echo "$*" >> "$LOG"; }}
        export PATH="{bin_dir}:$PATH"
        {block}
        echo "SHOULD_NOT_REACH_HERE"
        """)

    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(tmp_path),
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1, f"expected exit 1 on S3 restore failure, got {proc.returncode}"
    assert "SHOULD_NOT_REACH_HERE" not in proc.stdout


def test_no_literal_escaped_dollar_regression_guard():
    """Regex guard for the exact PR#352 regression class: a literal
    backslash-escaped `\\$LOCAL_MAX_DATE` / `\\$RESTORE_NEEDED` inside
    double quotes is NEVER valid shell in this file — bash does not expand
    it, so any test/comparison using it is permanently dead code. This
    can't catch every way to break the logic (a regex can't verify
    behavior), but it pins this specific, already-shipped-once regression
    so it can't silently reoccur even if the subprocess tests above are
    ever weakened or skipped in a constrained CI sandbox."""
    src = _BOOT_PULL.read_text()
    assert not re.search(r'\\\$LOCAL_MAX_DATE', src), (
        'Found a literal `\\$LOCAL_MAX_DATE` (backslash-escaped dollar) in '
        'boot-pull.sh — this is the PR#352 regression (config#2356): inside '
        'double quotes, "\\$FOO" is a non-expanding literal string, so any '
        '[ -z "\\$LOCAL_MAX_DATE" ] test is permanently false. Use an '
        'unescaped "$LOCAL_MAX_DATE".'
    )
    assert not re.search(r'\\\$RESTORE_NEEDED', src), (
        'Found a literal `\\$RESTORE_NEEDED` (backslash-escaped dollar) in '
        'boot-pull.sh — this is the PR#352 regression (config#2356): '
        '[ "\\$RESTORE_NEEDED" = "true" ] can never be true, so the entire '
        'S3 restore-on-boot safety net silently becomes dead code. Use an '
        'unescaped "$RESTORE_NEEDED".'
    )


def test_restore_block_still_checks_missing_and_minimal_size_branches():
    """Source-text sanity pin: the pre-existing missing-file and
    minimal-size branches must still be present verbatim (this fix only
    adds a recency check, it must not remove the others)."""
    block = _extract_restore_block()
    assert '[ ! -f "$DB_PATH" ]' in block
    assert '"$DB_SIZE" -le 20480' in block


def test_restore_block_uses_trading_calendar_for_recency():
    """The fix must use nousergon_lib.trading_calendar (the repo's existing
    convention — see executor/main.py L1128, executor/eod_reconcile.py,
    executor/reconcile_audit.py) rather than a naive date-diff, so it
    naturally accounts for weekends/holidays instead of e.g. flagging every
    Monday morning as stale."""
    block = _extract_restore_block()
    assert "nousergon_lib.trading_calendar" in block
    assert "last_closed_trading_day" in block


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_restore_block_extractable_and_nonempty():
    """Sanity check the marker-based extraction itself isn't silently
    returning an empty/garbage slice (would make every test above a
    false-negative no-op)."""
    block = _extract_restore_block()
    assert "RESTORE_NEEDED=false" in block
    assert "aws s3 cp" in block
    assert len(block.splitlines()) > 20
