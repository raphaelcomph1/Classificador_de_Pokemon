"""Previsão top-k para uma imagem.

Uso:
    python -m pokedex.predict caminho/da/imagem.jpg --checkpoint models/mobilenet_v3_large/best.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from pokedex.config import project_path
from pokedex.data.datasets import load_rgb
from pokedex.data.transforms import eval_transforms
from pokedex.models.checkpoint import load_model
from pokedex.utils import get_device


class Predictor:
    """Carrega o checkpoint uma vez e prevê várias imagens (útil para a API/app)."""

    def __init__(self, checkpoint: Path, device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.model, ckpt = load_model(checkpoint, self.device)
        self.classes: list[str] = ckpt["classes"]
        self.transform = eval_transforms(ckpt["image_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path, k: int = 3) -> list[tuple[str, float]]:
        """Top-k (classe, probabilidade), da mais provável para a menos."""
        if not isinstance(image, Image.Image):
            image = load_rgb(image)
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)[0]
        top = torch.topk(probs, k=min(k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    predictor = Predictor(project_path(args.checkpoint))
    for name, prob in predictor.predict(args.image, args.k):
        print(f"{name:20s} {prob:6.1%}")


if __name__ == "__main__":
    main()
