"""Scanner ao vivo: encontra o Pokémon no vídeo, desenha um quadrado e escreve quem é.

Duas formas de mira:
- **fixa**: um quadrado parado no centro; o usuário enquadra o Pokémon nele.
- **dinâmica**: um localizador (YOLO-World, ver `detector.py`) propõe regiões; o nosso
  classificador analisa cada recorte e só aceita a região se tiver confiança suficiente.
  O detector LOCALIZA e o classificador IDENTIFICA (e confirma que é mesmo um Pokémon).

A análise roda numa thread separada: o vídeo continua fluido e o desenho usa o último
resultado pronto. Entre análises, o quadrado se move suavemente (média com a posição anterior).
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

Box = tuple[float, float, float, float]


@dataclass
class ScanResult:
    """Palpite atual do scanner (já suavizado)."""

    label: str | None
    prob: float
    top: list[tuple[str, float]]
    box: Box | None = None


def center_square(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    """Quadrado centralizado com lado = `fraction` do menor lado do quadro: (x0, y0, x1, y1)."""
    side = int(min(width, height) * fraction)
    x0, y0 = (width - side) // 2, (height - side) // 2
    return x0, y0, x0 + side, y0 + side


def iou(a: Box, b: Box) -> float:
    """Interseção sobre união de duas caixas (0 = não se tocam, 1 = idênticas)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def smooth_box(previous: Box | None, new: Box, weight: float) -> Box:
    """Mistura a posição nova com a anterior (`weight` = peso da anterior).

    Se as caixas mal se sobrepõem, é outro objeto: pula direto para a nova.
    """
    if previous is None or iou(previous, new) < 0.1:
        return new
    return tuple(weight * p + (1 - weight) * n for p, n in zip(previous, new, strict=True))


def expand_box(box: Box, width: int, height: int, margin: float) -> Box:
    """Aumenta a caixa em `margin` (fração) de cada lado, sem sair da imagem."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return max(0.0, x0 - dx), max(0.0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


class ProbabilitySmoother:
    """Média das probabilidades das últimas N análises (evita o nome "piscar")."""

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


def _font(frame: Image.Image) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=max(18, frame.width // 28))


def draw_label(
    frame: Image.Image, xy: tuple[int, int], text: str, color: tuple[int, int, int]
) -> None:
    """Texto preto sobre uma faixa colorida, com o canto inferior esquerdo em `xy`."""
    draw = ImageDraw.Draw(frame)
    font = _font(frame)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    pad = text_h // 3
    x0 = min(max(0, xy[0]), max(0, frame.width - text_w - 2 * pad))
    y0 = max(0, xy[1] - text_h - 2 * pad)  # acima do ponto; se não couber, no topo
    draw.rectangle((x0, y0, x0 + text_w + 2 * pad, y0 + text_h + 2 * pad), fill=color)
    draw.text((x0 + pad - left, y0 + pad - top), text, fill=(0, 0, 0), font=font)


def draw_overlay(
    frame: Image.Image, box: Box, text: str, color: tuple[int, int, int]
) -> Image.Image:
    """Desenha o quadrado e o rótulo logo acima dele (modifica e devolve `frame`)."""
    box_int = tuple(int(round(v)) for v in box)
    ImageDraw.Draw(frame).rectangle(box_int, outline=color, width=max(3, frame.width // 200))
    draw_label(frame, (box_int[0], box_int[1] - 6), text, color)
    return frame


class LiveScanner:
    """Processa quadros de vídeo. Seguro para usar da thread de vídeo e da interface ao mesmo tempo.

    `locate=None` usa a mira fixa; com um localizador, a mira é dinâmica.
    `background=True` roda a análise numa thread (no app); False roda na hora (testes).
    """

    def __init__(
        self,
        predict_probs: Callable[[Image.Image], torch.Tensor],
        to_top_k: Callable[[torch.Tensor, int], list[tuple[str, float]]],
        display_name: Callable[[str], str],
        *,
        locate: Callable[[Image.Image], list[Box]] | None = None,
        square_fraction: float = 0.6,
        smoothing_frames: int = 5,
        min_interval_s: float = 0.1,
        confident: float = 0.6,
        uncertain: float = 0.4,
        crop_margin: float = 0.1,
        box_smoothing: float = 0.5,
        hold_s: float = 0.6,
        background: bool = False,
    ) -> None:
        self.predict_probs, self.to_top_k, self.display_name = predict_probs, to_top_k, display_name
        self.locate = locate
        self.square_fraction = square_fraction
        self.min_interval_s = min_interval_s
        self.confident, self.uncertain = confident, uncertain
        self.crop_margin, self.box_smoothing, self.hold_s = crop_margin, box_smoothing, hold_s
        self.background = background
        self.smoother = ProbabilitySmoother(smoothing_frames)
        self._lock = threading.Lock()
        self._busy = False
        self._last_run = float("-inf")
        self._last_seen = float("-inf")
        self._result = ScanResult(None, 0.0, [])

    @property
    def dynamic(self) -> bool:
        return self.locate is not None

    @property
    def result(self) -> ScanResult:
        with self._lock:
            return self._result

    def reset(self) -> None:
        """Esquece o que viu (ex.: ao parar o vídeo)."""
        with self._lock:
            self.smoother.clear()
            self._result = ScanResult(None, 0.0, [])
            self._last_seen = float("-inf")

    # --- análise (pode rodar em outra thread) ---------------------------------------------

    def _best_candidate(self, image: Image.Image) -> tuple[Box, torch.Tensor] | None:
        """Classifica cada região proposta e devolve a que o classificador mais confia."""
        if self.dynamic:
            regions = self.locate(image)
        else:
            regions = [center_square(image.width, image.height, self.square_fraction)]
        # A margem extra compensa caixas do detector "justas demais"; a mira fixa é recortada
        # exatamente como aparece na tela.
        margin = self.crop_margin if self.dynamic else 0.0
        best = None
        for region in regions:
            crop_box = expand_box(region, image.width, image.height, margin)
            probs = self.predict_probs(image.crop(tuple(int(v) for v in crop_box)))
            if best is None or probs.max() > best[1].max():
                best = (region, probs)
        return best

    def _analyze(self, image: Image.Image, now: float) -> None:
        try:
            best = self._best_candidate(image)
            with self._lock:
                previous = self._result
                if best is None or (self.dynamic and float(best[1].max()) < self.uncertain):
                    # Nada convincente: no modo dinâmico, segura o último quadrado por `hold_s`.
                    if not self.dynamic or now - self._last_seen > self.hold_s:
                        self.smoother.clear()
                        self._result = ScanResult(None, 0.0, [])
                    return
                region, probs = best
                if previous.box is not None and iou(previous.box, region) < 0.1:
                    self.smoother.clear()  # outro objeto: não misturar com o anterior
                top = self.to_top_k(self.smoother.add(probs), 3)
                label, prob = top[0]
                box = (
                    smooth_box(previous.box, region, self.box_smoothing) if self.dynamic else region
                )
                self._last_seen = now
                self._result = ScanResult(label if prob >= self.uncertain else None, prob, top, box)
        finally:
            self._busy = False

    # --- desenho (thread do vídeo) ---------------------------------------------------------

    def process(self, frame: np.ndarray, now: float | None = None) -> np.ndarray:
        """Recebe um quadro RGB (altura, largura, 3) e devolve o quadro com o desenho por cima."""
        now = time.monotonic() if now is None else now
        image = Image.fromarray(frame)

        if not self._busy and now - self._last_run >= self.min_interval_s:
            self._busy, self._last_run = True, now
            if self.background:
                threading.Thread(
                    target=self._analyze, args=(image.copy(), now), daemon=True
                ).start()
            else:
                self._analyze(image, now)

        result = self.result
        color = status_color(result.prob, self.confident, self.uncertain)
        text = (
            f"{self.display_name(result.label)} {result.prob:.0%}"
            if result.label
            else "Procurando..."
        )
        if self.dynamic:
            if result.label and result.box is not None:
                draw_overlay(image, result.box, text, color)
            else:
                draw_label(image, (10, 10 + image.width // 28 + 20), text, GRAY)
        else:
            draw_overlay(
                image, center_square(image.width, image.height, self.square_fraction), text, color
            )
        return np.asarray(image)
