"""Detector real (YOLO-World com vocabulário embutido). Pulado se os pesos não existirem."""

import pytest
from PIL import Image, ImageDraw

from pokedex.config import load_config, project_path

DET = load_config("app")["scanner"]["detector"]
WEIGHTS = project_path(DET["weights"])
pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="rode: python -m pokedex.detector")


def test_locator_returns_at_most_max_candidates_boxes() -> None:
    from pokedex.detector import PokemonLocator

    locator = PokemonLocator(WEIGHTS, conf=0.02, imgsz=640, max_candidates=2)
    image = Image.new("RGB", (640, 480), (120, 160, 200))
    ImageDraw.Draw(image).ellipse(
        (220, 140, 420, 340), fill=(250, 210, 40), outline=(0, 0, 0), width=4
    )
    boxes = locator.candidates(image)
    assert len(boxes) <= 2
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480


def test_detector_has_embedded_vocabulary() -> None:
    from ultralytics import YOLO

    assert list(YOLO(str(WEIGHTS)).names.values()) == DET["prompts"]
