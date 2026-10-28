from pathlib import Path

import numpy as np
from PIL import Image

from pokedex.data.check import (
    ImageRecord,
    apply_exclusions,
    find_dataset_root,
    find_duplicate_groups,
    has_wrong_extension,
    inspect_image,
)


def _save_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _saved(path: Path) -> Path:
    _save_image(path, 1)
    return path


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


def test_inspect_image_rejects_svg(tmp_path: Path) -> None:
    svg = tmp_path / "Pikachu" / "x.svg"
    svg.parent.mkdir(parents=True)
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    record = inspect_image(svg, "Pikachu", tmp_path)
    assert not record.ok
    assert record.error


def test_inspect_image_recovers_jpeg_with_wrong_extension(tmp_path: Path) -> None:
    jpeg = tmp_path / "Muk" / "x.jpg"
    _save_image(jpeg, 0)
    wrong = jpeg.rename(jpeg.with_suffix(".asp"))
    record = inspect_image(wrong, "Muk", tmp_path)
    assert record.ok
    assert record.format == "JPEG"
    assert has_wrong_extension(record)
    assert not has_wrong_extension(
        inspect_image(_saved(tmp_path / "Muk" / "y.jpeg"), "Muk", tmp_path)
    )


def test_apply_exclusions_marks_reasons() -> None:
    records = [
        ImageRecord("a", "Pidgeot", ".jpg", ok=True, dup_group=0),
        ImageRecord("b", "Pidgeotto", ".jpg", ok=True, dup_group=0),
        ImageRecord("c", "Alolan Sandslash", ".jpg", ok=True, dup_group=1),
        ImageRecord("d", "Pikachu", ".jpg", ok=True, dup_group=2),
        ImageRecord("e", "Pikachu", ".jpg", ok=True, dup_group=2),
        ImageRecord("f", "Pikachu", ".svg", ok=False),
    ]
    cfg = {"excluded_classes": ["Alolan Sandslash"], "drop_cross_class_duplicates": True}
    apply_exclusions(records, cfg)
    reasons = [r.exclude_reason for r in records]
    assert reasons == [
        "duplicata entre classes",
        "duplicata entre classes",
        "classe excluída",
        "",  # duplicata na MESMA classe continua (vai junta para o mesmo split)
        "",
        "arquivo inválido",
    ]


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
