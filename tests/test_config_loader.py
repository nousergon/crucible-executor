"""Tests for executor.config_loader — risk.yaml resolution with no silent fallback."""

import importlib
import os

import pytest
import yaml

from executor import config_loader


def test_search_paths_experiment_package_first(monkeypatch):
    """Experiment-package risk.yaml resolves ahead of the legacy top-level path (config#1042)."""
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "myexp")
    paths = config_loader._build_search_paths()
    # First candidate must be the experiment-package copy for the selected slot.
    assert paths[0].endswith(os.path.join("experiments", "myexp", "executor", "risk.yaml"))
    # The legacy top-level config-repo path must still be present, AFTER all package paths.
    legacy_idx = next(i for i, p in enumerate(paths) if p.endswith(os.path.join("alpha-engine-config", "executor", "risk.yaml")))
    pkg_idx = next(i for i, p in enumerate(paths) if "myexp" in p)
    assert pkg_idx < legacy_idx
    # The repo-local legacy fallback remains last.
    assert paths[-1].endswith(os.path.join("config", "risk.yaml"))


def test_search_paths_default_experiment_is_reference(monkeypatch):
    """With no ALPHA_ENGINE_EXPERIMENT_ID set, the default experiment is `reference`."""
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)
    paths = config_loader._build_search_paths()
    assert paths[0].endswith(os.path.join("experiments", "reference", "executor", "risk.yaml"))


def test_get_config_path_returns_first_match(tmp_path, monkeypatch):
    candidate = tmp_path / "risk.yaml"
    candidate.write_text("foo: bar\n")
    monkeypatch.setattr(config_loader, "_SEARCH_PATHS", [str(candidate)])

    resolved = config_loader.get_config_path()
    assert os.path.realpath(str(candidate)) == resolved


def test_get_config_path_picks_first_existing(tmp_path, monkeypatch):
    missing = tmp_path / "nope.yaml"
    real = tmp_path / "real.yaml"
    real.write_text("foo: bar\n")
    monkeypatch.setattr(config_loader, "_SEARCH_PATHS", [str(missing), str(real)])

    resolved = config_loader.get_config_path()
    assert resolved == os.path.realpath(str(real))


def test_get_config_path_raises_when_none_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(config_loader, "_SEARCH_PATHS", [
        str(tmp_path / "a.yaml"),
        str(tmp_path / "b.yaml"),
    ])
    with pytest.raises(FileNotFoundError) as exc:
        config_loader.get_config_path()

    msg = str(exc.value)
    assert "risk.yaml not found" in msg
    assert "a.yaml" in msg
    assert "b.yaml" in msg
    # Example template MUST NOT be searched silently
    assert ".example template is intentionally NOT searched" in msg


def test_load_config_returns_parsed_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "risk.yaml"
    cfg.write_text(yaml.safe_dump({"signals_bucket": "real-bucket", "max_position_pct": 0.05}))
    monkeypatch.setattr(config_loader, "_SEARCH_PATHS", [str(cfg)])

    loaded = config_loader.load_config()
    assert loaded["signals_bucket"] == "real-bucket"
    assert loaded["max_position_pct"] == 0.05


def test_load_config_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config_loader, "_SEARCH_PATHS", [str(tmp_path / "nope.yaml")])
    with pytest.raises(FileNotFoundError):
        config_loader.load_config()


# ── flow-doctor.yaml resolution (config#1042) ───────────────────────────────


def test_flow_doctor_path_prefers_experiment_package(tmp_path, monkeypatch):
    """flow-doctor.yaml resolves from the experiment package ahead of the
    legacy top-level config-repo copy and the repo-root fallback (config#1042)."""
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "myexp")
    # Lay out a fake config repo as a sibling of the executor repo root, with
    # BOTH a package copy and a legacy top-level copy of flow-doctor.yaml.
    repo_root = tmp_path / "crucible-executor"
    (repo_root).mkdir()
    (repo_root / "flow-doctor.yaml").write_text("flow_name: repo-root\n")  # repo-root fallback
    cfg = tmp_path / "alpha-engine-config"
    pkg = cfg / "experiments" / "myexp" / "executor"
    pkg.mkdir(parents=True)
    (pkg / "flow-doctor.yaml").write_text("flow_name: package\n")
    legacy = cfg / "executor"
    legacy.mkdir(parents=True)
    (legacy / "flow-doctor.yaml").write_text("flow_name: legacy\n")
    monkeypatch.setattr(config_loader, "_REPO_ROOT", str(repo_root))
    # Keep HOME from resolving a real ~/alpha-engine-config during the test.
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))

    resolved = config_loader.get_flow_doctor_yaml_path()
    assert resolved == os.path.realpath(str(pkg / "flow-doctor.yaml"))


def test_flow_doctor_path_falls_back_to_repo_root(tmp_path, monkeypatch):
    """With no config-repo copy present, flow-doctor.yaml resolves to the
    in-repo repo-root copy (preserves pre-config#1042 behavior)."""
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)
    repo_root = tmp_path / "crucible-executor"
    repo_root.mkdir()
    (repo_root / "flow-doctor.yaml").write_text("flow_name: repo-root\n")
    monkeypatch.setattr(config_loader, "_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))

    resolved = config_loader.get_flow_doctor_yaml_path()
    assert resolved == os.path.realpath(str(repo_root / "flow-doctor.yaml"))


def test_flow_doctor_path_never_raises_when_absent(tmp_path, monkeypatch):
    """Even with no copy anywhere, resolution degrades to the repo-root path
    string rather than raising — setup_logging runs at import time and must
    never be blocked by a missing flow-doctor.yaml."""
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)
    repo_root = tmp_path / "crucible-executor"
    repo_root.mkdir()  # no flow-doctor.yaml written anywhere
    monkeypatch.setattr(config_loader, "_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))

    resolved = config_loader.get_flow_doctor_yaml_path()
    assert resolved == os.path.join(str(repo_root), "flow-doctor.yaml")
