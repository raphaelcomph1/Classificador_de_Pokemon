"""Datasets e DataLoaders a partir de processed/{train,val,test}/<classe>/."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from pokedex.data.transforms import eval_transforms, train_transforms

SPLITS = ("train", "val", "test")


def load_rgb(path: str | Path) -> Image.Image:
    """Abre a imagem em RGB. Transparência vira fundo branco (e não preto)."""
    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return img.convert("RGB")


def build_datasets(processed_dir: Path, image_size: int) -> dict[str, ImageFolder]:
    """Um `ImageFolder` por split; augmentation só no treino."""
    datasets = {
        split: ImageFolder(
            processed_dir / split,
            transform=train_transforms(image_size)
            if split == "train"
            else eval_transforms(image_size),
            loader=load_rgb,
        )
        for split in SPLITS
    }
    if len({tuple(ds.classes) for ds in datasets.values()}) != 1:
        raise ValueError("Os splits têm classes diferentes; rode o split de novo.")
    return datasets


def build_loaders(
    datasets: dict[str, ImageFolder],
    batch_size: int,
    num_workers: int,
    max_batches: int | None = None,
) -> dict[str, DataLoader]:
    """DataLoaders; com `max_batches`, cada split é cortado (smoke test)."""
    loaders = {}
    for split, ds in datasets.items():
        if max_batches is not None:
            # amostras espaçadas, para pegar várias classes (o ImageFolder é ordenado por classe)
            n = min(len(ds), max_batches * batch_size)
            ds = Subset(ds, range(0, len(ds), len(ds) // n)[:n])
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # acelera a cópia CPU -> GPU
        )
    return loaders
