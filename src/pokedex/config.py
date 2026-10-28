"""Leitura das configs YAML e caminhos do projeto."""

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_ROOT_ENV = "POKEDEX_DATA_ROOT"


def load_config(name: str) -> dict[str, Any]:
    """Carrega `configs/<name>.yaml` como dicionário."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do projeto."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def data_root(cfg: dict[str, Any]) -> Path:
    """Pasta base dos dados: env POKEDEX_DATA_ROOT ou `data_root` da config."""
    return project_path(os.environ.get(DATA_ROOT_ENV) or cfg["data_root"])


def data_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve `cfg[key]` relativo à pasta base dos dados."""
    return data_root(cfg) / cfg[key]
