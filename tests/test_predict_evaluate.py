from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pokedex.evaluate import compute_metrics
from pokedex.models.build import SmallCNN
from pokedex.models.checkpoint import save_checkpoint
from pokedex.predict import Predictor

CLASSES = ["Bulbasaur", "Charmander", "Pikachu", "Squirtle"]


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    save_checkpoint(path, SmallCNN(len(CLASSES)), "small_cnn", CLASSES, 64, {})
    return path


def test_predictor_returns_sorted_top_k(checkpoint: Path, tmp_path: Path) -> None:
    image_path = tmp_path / "x.png"
    Image.new("RGB", (100, 80), (255, 200, 0)).save(image_path)
    result = Predictor(checkpoint).predict(image_path, k=3)

    assert len(result) == 3
    assert all(name in CLASSES for name, _ in result)
    probs = [p for _, p in result]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predictor_k_larger_than_classes(checkpoint: Path) -> None:
    result = Predictor(checkpoint).predict(Image.new("RGB", (64, 64)), k=10)
    assert len(result) == len(CLASSES)
    assert sum(p for _, p in result) == pytest.approx(1.0, abs=1e-5)


def test_compute_metrics_known_values() -> None:
    # 4 exemplos: 3 certos no top-1; o errado (Pikachu) tem a classe certa em 2º lugar
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.5, 0.3, 0.1],  # Pikachu previsto como Charmander
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    labels = np.array([0, 1, 2, 3])
    m = compute_metrics(probs, labels, CLASSES)
    assert m["accuracy"] == 0.75
    assert m["top3_accuracy"] == 1.0
    assert m["f1_per_class"]["Pikachu"] == 0.0
    assert m["most_confused"] == [{"true": "Pikachu", "predicted": "Charmander", "count": 1}]
