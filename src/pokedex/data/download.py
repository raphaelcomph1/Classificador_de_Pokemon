"""Baixa o dataset do Kaggle para data/raw/.

Uso:
    python -m pokedex.data.download
    python -m pokedex.data.download --force
"""

import argparse
from pathlib import Path

import kagglehub

from pokedex.config import load_config, project_path


def download_dataset(handle: str, output_dir: Path, force: bool = False) -> Path:
    """Baixa e extrai o dataset `handle` em `output_dir`. Retorna o caminho final."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(handle, force_download=force, output_dir=str(output_dir))
    return Path(path)


def main() -> None:
    cfg = load_config("data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="baixar de novo mesmo se já existir")
    args = parser.parse_args()

    out = download_dataset(cfg["dataset_handle"], project_path(cfg["raw_dir"]), args.force)
    print(f"Dataset disponível em: {out}")


if __name__ == "__main__":
    main()
