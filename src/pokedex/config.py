"""Leitura das configs YAML e caminhos do projeto."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Carrega `configs/<name>.yaml` como dicionário."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do projeto."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
