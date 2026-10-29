import numpy as np
import torch
from PIL import Image

from pokedex.scanner import (
    GRAY,
    GREEN,
    YELLOW,
    LiveScanner,
    ProbabilitySmoother,
    center_square,
    status_color,
)

CLASSES = ["Bulbasaur", "Charmander", "Pikachu"]


def _top_k(probs: torch.Tensor, k: int) -> list[tuple[str, float]]:
    top = torch.topk(probs, k=min(k, len(CLASSES)))
    return [(CLASSES[i], float(p)) for p, i in zip(top.values, top.indices, strict=True)]


class FakeModel:
    """Devolve probabilidades fixas e guarda o tamanho do recorte recebido."""

    def __init__(self, probs: list[float]) -> None:
        self.probs, self.calls, self.crop_sizes = torch.tensor(probs), 0, []

    def __call__(self, image: Image.Image) -> torch.Tensor:
        self.calls += 1
        self.crop_sizes.append(image.size)
        return self.probs


def _scanner(model: FakeModel, **kwargs) -> LiveScanner:
    return LiveScanner(model, _top_k, str, **kwargs)


def _frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 30, dtype=np.uint8)


def test_center_square_is_centered_and_square() -> None:
    assert center_square(640, 480, 0.5) == (200, 120, 440, 360)
    x0, y0, x1, y1 = center_square(1920, 1080, 0.6)
    assert x1 - x0 == y1 - y0 == 648
    assert (x0 + x1) // 2 == 960 and (y0 + y1) // 2 == 540


def test_smoother_averages_last_frames() -> None:
    s = ProbabilitySmoother(window=2)
    s.add(torch.tensor([1.0, 0.0]))
    s.add(torch.tensor([0.0, 1.0]))
    assert torch.allclose(s.add(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0]))  # o 1º saiu


def test_status_color() -> None:
    assert status_color(0.9, 0.6, 0.4) == GREEN
    assert status_color(0.5, 0.6, 0.4) == YELLOW
    assert status_color(0.1, 0.6, 0.4) == GRAY


def test_process_classifies_only_the_square_and_draws_it() -> None:
    model = FakeModel([0.05, 0.05, 0.9])
    scanner = _scanner(model, square_fraction=0.5)
    out = scanner.process(_frame(), now=1.0)

    assert out.shape == (480, 640, 3)
    assert model.crop_sizes == [(240, 240)]  # recorte = mira, não o quadro inteiro
    assert scanner.result.label == "Pikachu"
    x0, y0, _, y1 = center_square(640, 480, 0.5)
    assert tuple(out[(y0 + y1) // 2, x0]) == GREEN  # borda esquerda da mira, verde


def test_low_confidence_shows_searching_in_gray() -> None:
    scanner = _scanner(FakeModel([0.34, 0.33, 0.33]), square_fraction=0.5)
    out = scanner.process(_frame(), now=1.0)
    assert scanner.result.label is None
    x0, y0, _, y1 = center_square(640, 480, 0.5)
    assert tuple(out[(y0 + y1) // 2, x0]) == GRAY


def test_process_respects_min_interval() -> None:
    model = FakeModel([0.1, 0.1, 0.8])
    scanner = _scanner(model, min_interval_s=0.1)
    for now in (1.00, 1.03, 1.06, 1.09, 1.12):
        scanner.process(_frame(), now=now)
    assert model.calls == 2  # em 1.00 e 1.12; os quadros do meio reaproveitam o resultado
