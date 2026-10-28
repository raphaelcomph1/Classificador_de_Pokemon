"""Utilidades de reprodutibilidade e hardware."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa as seeds de `random`, `numpy` e `torch` (CPU e GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """`cuda` se houver GPU, senão `cpu`."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
