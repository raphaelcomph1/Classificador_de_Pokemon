"""Avaliação de um checkpoint: acurácia, top-3, F1 por classe, matriz de confusão.

Uso:
    python -m pokedex.evaluate models/mobilenet_v3_large/best.pt            # split de teste
    python -m pokedex.evaluate models/resnet18/best.pt --split val

Saída na pasta do checkpoint: metrics_<split>.json e confusion_<split>.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva figuras sem abrir janela

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pokedex.config import data_path, load_config, project_path  # noqa: E402
from pokedex.data.datasets import build_datasets, build_loaders  # noqa: E402
from pokedex.models.checkpoint import load_model  # noqa: E402
from pokedex.utils import get_device  # noqa: E402


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades (N, C) e rótulos verdadeiros (N,) para todo o loader."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        all_probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def compute_metrics(
    probs: np.ndarray, labels: np.ndarray, classes: list[str], n_confused: int = 15
) -> dict:
    """Métricas globais, F1 por classe e os pares mais confundidos."""
    preds = probs.argmax(axis=1)
    all_ids = np.arange(len(classes))
    f1_per_class = f1_score(labels, preds, labels=all_ids, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=all_ids)

    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1], cm.shape))[0]
    confused = [
        {"true": classes[t], "predicted": classes[p], "count": int(cm[t, p])}
        for t, p in pairs[:n_confused]
        if off_diagonal[t, p] > 0  # só erros (a diagonal = acertos foi zerada)
    ]
    per_class = {c: round(float(f), 4) for c, f in zip(classes, f1_per_class, strict=True)}
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "top3_accuracy": float(top_k_accuracy_score(labels, probs, k=3, labels=all_ids)),
        "macro_f1": float(f1_per_class.mean()),
        "worst_classes": sorted(per_class.items(), key=lambda kv: kv[1])[:10],
        "most_confused": confused,
        "f1_per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    """Heatmap normalizado por linha (cada linha soma 1 = recall da classe)."""
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, len(classes) * 0.18)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(norm, cmap="viridis", vmin=0, vmax=1, square=True, cbar_kws={"shrink": 0.5}, ax=ax)
    ax.set_xticks(np.arange(len(classes)) + 0.5, classes, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(len(classes)) + 0.5, classes, rotation=0, fontsize=5)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    ckpt_path = project_path(args.checkpoint)
    model, ckpt = load_model(ckpt_path, device)
    datasets = build_datasets(data_path(load_config("data"), "processed_dir"), ckpt["image_size"])
    if datasets[args.split].classes != ckpt["classes"]:
        raise ValueError("Classes do checkpoint diferentes das do dataset.")
    loader = build_loaders({args.split: datasets[args.split]}, 64, args.num_workers)[args.split]

    probs, labels = predict_probs(model, loader, device)
    metrics = {
        "arch": ckpt["arch"],
        "split": args.split,
        **compute_metrics(probs, labels, ckpt["classes"]),
    }

    out = ckpt_path.parent
    (out / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(
        np.array(metrics["confusion_matrix"]), ckpt["classes"], out / f"confusion_{args.split}.png"
    )

    print(f"{ckpt['arch']} [{args.split}] n={metrics['n']}")
    print(
        f"  acurácia {metrics['accuracy']:.4f} · top-3 {metrics['top3_accuracy']:.4f} · F1 macro {metrics['macro_f1']:.4f}"
    )
    print("  mais confundidos:")
    for c in metrics["most_confused"][:10]:
        print(f"    {c['true']} -> {c['predicted']}: {c['count']}")


if __name__ == "__main__":
    main()
