"""Localizador de "candidatos a Pokémon" com YOLO-World (detecção por descrição em texto).

O YOLO-World encontra objetos a partir de palavras, sem treino. Ele só LOCALIZA: quem diz
qual Pokémon é (e se é mesmo um Pokémon) é o nosso classificador, no recorte de cada caixa.

Preparar o detector (uma vez; baixa ~25 MB + o codificador de texto CLIP, ~340 MB):
    python -m pokedex.detector

Isso salva `models/yolo/pokedex-world-s.pt` com as descrições já embutidas, então o app
carrega o detector sem precisar do CLIP.
"""

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from pokedex.config import load_config, project_path

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) em pixels


def export_detector(base_weights: Path, prompts: list[str], out_path: Path) -> Path:
    """Gera o detector com o vocabulário fixo (as `prompts`) embutido."""
    from ultralytics import YOLOWorld  # import aqui: só quem exporta precisa do CLIP

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLOWorld(str(base_weights))  # baixa os pesos base se não existirem
    model.set_classes(prompts)
    model.save(str(out_path))
    return out_path


class PokemonLocator:
    """Propõe regiões que podem conter um Pokémon (as mais confiantes primeiro)."""

    def __init__(self, weights: Path, conf: float, imgsz: int, max_candidates: int) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.max_candidates = conf, imgsz, max_candidates
        # O preditor do ultralytics não é seguro para chamadas simultâneas (ex.: duas abas).
        self._lock = threading.Lock()

    def candidates(self, image: Image.Image) -> list[Box]:
        with self._lock:
            result = self.model.predict(
                np.asarray(image), imgsz=self.imgsz, conf=self.conf, verbose=False
            )[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # já vêm ordenadas por confiança
        return [tuple(float(v) for v in b) for b in boxes[: self.max_candidates]]


def main() -> None:
    det = load_config("app")["scanner"]["detector"]
    out = export_detector(
        project_path(det["base_weights"]), det["prompts"], project_path(det["weights"])
    )
    print(f"Detector salvo em {out}")


if __name__ == "__main__":
    main()
