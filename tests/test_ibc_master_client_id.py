"""`ibc-master-client-id.sh`: the gateway's Master API client ID for the v2
reconcile (alpha-engine-config-I12376, Brian's option (b)).

The v2 reconcile reads the shared paper account's fills on the gateway's Master
API client ID, so it sees the v1 executor's executions too. These tests pin
that the gateway side of the pairing is set from the trader's env file, edits
one key and nothing else, refuses an id that would collide, and can never stop
the gateway v1 trades through.
"""

from __future__ import annotations

import configparser
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "infrastructure" / "ibc-master-client-id.sh"
GATEWAY_UNIT = ROOT / "infrastructure" / "systemd" / "ibgateway.service"

INI = """\
IbLoginId=someone
IbPassword=not-a-real-secret
TradingMode=paper
ReadOnlyApi=no
"""


def _run(tmp_path: Path, env_text: str, ini_text: str | None = INI) -> tuple[subprocess.CompletedProcess, Path]:
    env_file = tmp_path / "trader.env"
    env_file.write_text(env_text)
    ini = tmp_path / "config.ini"
    if ini_text is not None:
        ini.write_text(ini_text)
        ini.chmod(0o600)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "CRUCIBLE_TRADER_ENV_FILE": str(env_file), "IBC_INI": str(ini)},
    )
    return result, ini


ENV = "CRUCIBLE_TRADER_IB_CLIENT_ID=41\nCRUCIBLE_TRADER_IB_MASTER_CLIENT_ID=42\n"


def test_sets_the_key_from_the_trader_env_and_keeps_everything_else(tmp_path: Path) -> None:
    result, ini = _run(tmp_path, ENV)
    assert result.returncode == 0, result.stderr
    text = ini.read_text()
    assert text.startswith(INI)
    assert text.rstrip().endswith("OverrideTwsMasterClientID=42")
    assert stat.S_IMODE(ini.stat().st_mode) == 0o600
    # The file holds the IB login: it is never echoed.
    assert "not-a-real-secret" not in result.stdout + result.stderr


def test_replaces_an_existing_value_in_place(tmp_path: Path) -> None:
    result, ini = _run(tmp_path, ENV, INI + "OverrideTwsMasterClientID=\nReadOnlyLogin=no\n")
    assert result.returncode == 0, result.stderr
    lines = ini.read_text().splitlines()
    assert lines.count("OverrideTwsMasterClientID=42") == 1
    assert lines[-1] == "ReadOnlyLogin=no"


def test_is_idempotent(tmp_path: Path) -> None:
    _, ini = _run(tmp_path, ENV)
    once = ini.read_text()
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "CRUCIBLE_TRADER_ENV_FILE": str(tmp_path / "trader.env"), "IBC_INI": str(ini)},
    )
    assert result.returncode == 0 and "already 42" in result.stdout
    assert ini.read_text() == once


@pytest.mark.parametrize(
    ("env_text", "reason"),
    [
        ("CRUCIBLE_TRADER_IB_CLIENT_ID=41\n", "not set"),
        ("CRUCIBLE_TRADER_IB_CLIENT_ID=41\nCRUCIBLE_TRADER_IB_MASTER_CLIENT_ID=abc\n", "integer"),
        ("CRUCIBLE_TRADER_IB_CLIENT_ID=41\nCRUCIBLE_TRADER_IB_MASTER_CLIENT_ID=2\n", "v1 executor"),
        ("CRUCIBLE_TRADER_IB_CLIENT_ID=41\nCRUCIBLE_TRADER_IB_MASTER_CLIENT_ID=41\n", "order-capable"),
    ],
)
def test_refuses_and_leaves_the_gateway_config_untouched(tmp_path: Path, env_text: str, reason: str) -> None:
    result, ini = _run(tmp_path, env_text)
    assert result.returncode != 0
    assert reason in result.stderr
    assert ini.read_text() == INI


def test_a_missing_ibc_config_fails_without_creating_one(tmp_path: Path) -> None:
    result, ini = _run(tmp_path, ENV, ini_text=None)
    assert result.returncode != 0 and "missing" in result.stderr
    assert not ini.exists()


def test_the_gateway_unit_runs_it_before_start_and_never_blocks_on_it() -> None:
    """v1 trades through this gateway: the `-` prefix means a failure here is
    logged and the gateway starts anyway."""
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    parser.read(GATEWAY_UNIT)
    pre = parser["Service"]["ExecStartPre"]
    assert pre.startswith("-")
    assert pre.endswith("/home/ec2-user/alpha-engine/infrastructure/ibc-master-client-id.sh")
    # boot-pull (which pulls the trader env file) runs first.
    assert "boot-pull.service" in parser["Unit"]["After"].split()


def test_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
