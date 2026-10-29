"""Scanner ao vivo: mira fixa no centro do vídeo + classificação do que está dentro dela.

O classificador não sabe ONDE o Pokémon está (isso exige um detector, ex.: YOLO). Por isso a
mira é fixa: o usuário enquadra o Pokémon nela e o modelo analisa só esse recorte.

Fluxo por quadro de vídeo:
    recorta a mira -> (no máx. a cada `min_interval_s`) prevê -> suaviza com a média dos
    últimos quadros -> desenha a mira colorida e o nome por cima do quadro.
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

GREEN = (46, 204, 64)
YELLOW = (255, 200, 0)
GRAY = (170, 170, 170)


@dataclass
class ScanResult:
    """Palpite atual do scanner (já suavizado)."""

    label: str | None
    prob: float
    top: list[tuple[str, float]]


def center_square(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    """Quadrado centralizado com lado = `fraction` do menor lado do quadro: (x0, y0, x1, y1)."""
    side = int(min(width, height) * fraction)
    x0, y0 = (width - side) // 2, (height - side) // 2
    return x0, y0, x0 + side, y0 + side


class ProbabilitySmoother:
    """Média das probabilidades dos últimos N quadros (evita o nome "piscar")."""

    def __init__(self, window: int) -> None:
        self.history: deque[torch.Tensor] = deque(maxlen=window)

    def add(self, probs: torch.Tensor) -> torch.Tensor:
        self.history.append(probs)
        return torch.stack(list(self.history)).mean(dim=0)

    def clear(self) -> None:
        self.history.clear()


def status_color(prob: float, confident: float, uncertain: float) -> tuple[int, int, int]:
    """Verde = confiante, amarelo = em dúvida, cinza = procurando."""
    if prob >= confident:
        return GREEN
    if prob >= uncertain:
        return YELLOW
    return GRAY


def draw_overlay(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
) -> Image.Image:
    """Desenha a mira e um rótulo com fundo acima dela (modifica e devolve `frame`)."""
    draw = ImageDraw.Draw(frame)
    thickness = max(3, frame.width // 200)
    draw.rectangle(box, outline=color, width=thickness)

    font = ImageFont.load_default(size=max(18, frame.width // 28))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    pad = text_h // 3
    x0 = box[0]
    y0 = max(0, box[1] - text_h - 3 * pad)  # acima da mira; se não couber, no topo
    draw.rectangle((x0, y0, x0 + text_w + 2 * pad, y0 + text_h + 2 * pad), fill=color)
    draw.text((x0 + pad - left, y0 + pad - top), text, fill=(0, 0, 0), font=font)
    return frame


class LiveScanner:
    """Processa quadros de vídeo. Seguro para usar da thread de vídeo e da interface ao mesmo tempo."""

    def __init__(
        self,
        predict_probs: Callable[[Image.Image], torch.Tensor],
        to_top_k: Callable[[torch.Tensor, int], list[tuple[str, float]]],
        display_name: Callable[[str], str],
        *,
        square_fraction: float = 0.6,
        smoothing_frames: int = 5,
        min_interval_s: float = 0.1,
        confident: float = 0.6,
        uncertain: float = 0.4,
    ) -> None:
        self.predict_probs, self.to_top_k, self.display_name = predict_probs, to_top_k, display_name
        self.square_fraction = square_fraction
        self.min_interval_s = min_interval_s
        self.confident, self.uncertain = confident, uncertain
        self.smoother = ProbabilitySmoother(smoothing_frames)
        self._lock = threading.Lock()
        self._last_run = 0.0
        self._result = ScanResult(None, 0.0, [])

    @property
    def result(self) -> ScanResult:
        with self._lock:
            return self._result

    def process(self, frame: np.ndarray, now: float | None = None) -> np.ndarray:
        """Recebe um quadro RGB (altura, largura, 3) e devolve o quadro com a mira desenhada."""
        now = time.monotonic() if now is None else now
        image = Image.fromarray(frame)
        box = center_square(image.width, image.height, self.square_fraction)

        # Prever em todo quadro sobrecarregaria a CPU; entre previsões, reaproveita o último resultado.
        if now - self._last_run >= self.min_interval_s:
            self._last_run = now
            smoothed = self.smoother.add(self.predict_probs(image.crop(box)))
            top = self.to_top_k(smoothed, 3)
            label, prob = top[0]
            with self._lock:
                self._result = ScanResult(label if prob >= self.uncertain else None, prob, top)

        result = self.result
        color = status_color(result.prob, self.confident, self.uncertain)
        text = (
            f"{self.display_name(result.label)} {result.prob:.0%}"
            if result.label
            else "Procurando..."
        )
        return np.asarray(draw_overlay(image, box, text, color))
