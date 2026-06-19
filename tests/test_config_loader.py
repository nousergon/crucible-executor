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
