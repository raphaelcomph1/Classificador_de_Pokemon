"""Salvar e carregar pesos treinados com os metadados necessários para inferência."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from pokedex.models.build import build_model


def save_checkpoint(
    path: Path, model: nn.Module, arch: str, classes: list[str], image_size: int, extra: dict
) -> None:
    """Salva pesos + tudo que a inferência precisa (arquitetura, classes, tamanho)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "classes": classes,
            "image_size": image_size,
            **extra,
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstrói o modelo a partir do checkpoint, já em modo de avaliação."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
