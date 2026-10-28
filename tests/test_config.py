from pathlib import Path

import pytest

from pokedex.config import DATA_ROOT_ENV, PROJECT_ROOT, data_path


def test_data_path_defaults_to_project_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    cfg = {"data_root": "data", "raw_dir": "raw"}
    assert data_path(cfg, "raw_dir") == PROJECT_ROOT / "data" / "raw"


def test_data_path_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = {"data_root": "data", "raw_dir": "raw"}
    assert data_path(cfg, "raw_dir") == tmp_path / "raw"
