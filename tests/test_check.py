from pathlib import Path

import numpy as np
from PIL import Image

from pokedex.data.check import find_dataset_root, find_duplicate_groups, inspect_image


def _save_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_find_dataset_root_skips_wrapper_folders(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "PokemonData"
    _save_image(root / "Pikachu" / "a.png", 0)
    _save_image(root / "Bulbasaur" / "b.png", 1)
    assert find_dataset_root(tmp_path / "raw") == root


def test_inspect_image_flags_corrupted_file(tmp_path: Path) -> None:
    good = tmp_path / "Pikachu" / "good.png"
    _save_image(good, 0)
    broken = tmp_path / "Pikachu" / "broken.jpg"
    broken.write_bytes(good.read_bytes()[:100])

    assert inspect_image(good, "Pikachu", tmp_path).ok
    record = inspect_image(broken, "Pikachu", tmp_path)
    assert not record.ok
    assert record.error


def test_inspect_image_rejects_unknown_extension(tmp_path: Path) -> None:
    svg = tmp_path / "Pikachu" / "x.svg"
    svg.parent.mkdir(parents=True)
    svg.write_text("<svg/>")
    assert inspect_image(svg, "Pikachu", tmp_path).error == "extensão não suportada"


def test_find_duplicate_groups_respects_threshold() -> None:
    hashes = [
        "0000000000000000",
        "0000000000000003",  # 2 bits de diferença do primeiro
        "ffffffffffffffff",  # muito diferente
        "000000000000000f",  # 4 bits do 1º, 2 do 2º -> mesmo grupo (transitivo)
    ]
    groups = find_duplicate_groups(hashes, max_distance=2)
    assert groups[0] == groups[1] == groups[3]
    assert groups[2] != groups[0]
    assert len(set(find_duplicate_groups(hashes, max_distance=0))) == 4
