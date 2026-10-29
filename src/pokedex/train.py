"""Treino em duas etapas: (1) só a cabeça, base congelada; (2) fine-tuning com lr menor.

Uso:
    python -m pokedex.train --arch mobilenet_v3_large
    python -m pokedex.train --arch resnet18 --smoke-test     # teste rápido na CPU
    python -m pokedex.train --arch small_cnn                 # baseline do zero (etapa única)

"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pokedex.config import data_path, load_config, project_path
from pokedex.data.datasets import build_datasets, build_loaders
from pokedex.models.build import (
    ARCHS,
    build_model,
    count_trainable,
    freeze_backbone,
    unfreeze_last_blocks,
)
from pokedex.models.checkpoint import save_checkpoint
from pokedex.utils import get_device, set_seed


class BestCheckpoint:
    """Guarda o melhor modelo visto até agora, medido pela acurácia de VALIDAÇÃO."""

    def __init__(self, path: Path, arch: str, classes: list[str], image_size: int) -> None:
        self.path, self.arch, self.classes, self.image_size = path, arch, classes, image_size
        self.best_acc = -1.0

    def update(self, model: nn.Module, val_acc: float, info: dict) -> bool:
        """Salva se `val_acc` for o melhor até agora. Devolve True se salvou."""
        if val_acc <= self.best_acc:
            return False
        self.best_acc = val_acc
        save_checkpoint(
            self.path, model, self.arch, self.classes, self.image_size, {"val_acc": val_acc, **info}
        )
        return True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Uma passada pelo treino. Devolve (loss média, acurácia)."""
    # train(): liga o Dropout e faz o BatchNorm usar/atualizar as estatísticas do batch.
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # o PyTorch SOMA gradientes; sem zerar, misturaria batches
        logits = model(images)  # 1. forward: palpites
        loss = criterion(logits, labels)  # 2. quanto errou
        loss.backward()  # 3. gradiente de cada peso: "para que lado ajustar"
        optimizer.step()  # 4. ajusta os pesos um pouquinho

        # loss.item() é a MÉDIA do batch; vezes o tamanho = soma. O último batch pode
        # ser menor, então somar e dividir pelo total no fim dá a média correta.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()  # sem gradientes: menos memória e mais rápido (não vamos treinar aqui)
def evaluate_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Avalia sem treinar. Devolve (loss média, acurácia)."""
    # eval(): desliga o Dropout e faz o BatchNorm usar as estatísticas aprendidas
    # (resultado determinístico e o modelo não muda).
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def run_stage(
    stage: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    best: BestCheckpoint,
) -> list[dict]:
    """Treina uma etapa por `epochs` épocas. Devolve o histórico (uma linha por época)."""
    # Só os parâmetros treináveis vão para o otimizador: os congelados não têm gradiente
    # e o AdamW gastaria memória guardando estado para eles à toa.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Cosseno: o lr começa em `lr` e desce suavemente até ~0 no fim da etapa.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        # Validação, NUNCA o teste: o teste só é usado uma vez, no fim. Se escolhêssemos o
        # melhor modelo olhando o teste, a nota do teste deixaria de ser imparcial.
        val_loss, val_acc = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        saved = best.update(model, val_acc, {"stage": stage, "epoch": epoch})
        print(
            f"[{stage}] época {epoch}/{epochs} · treino loss {train_loss:.3f} acc {train_acc:.3f}"
            f" · val loss {val_loss:.3f} acc {val_acc:.3f} · {time.time() - start:.0f}s"
            + (" · salvo" if saved else "")
        )
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
    return history


def plan_stages(cfg: dict, arch: str) -> list[tuple[str, int, float]]:
    """Lista de (nome, épocas, lr). O baseline treina tudo de uma vez."""
    if arch == "small_cnn":
        return [("full", cfg["head"]["epochs"] + cfg["finetune"]["epochs"], cfg["head"]["lr"])]
    return [
        ("head", cfg["head"]["epochs"], cfg["head"]["lr"]),
        ("finetune", cfg["finetune"]["epochs"], cfg["finetune"]["lr"]),
    ]


def prepare_stage(model: nn.Module, arch: str, stage: str, cfg: dict) -> None:
    """Congela/descongela as camadas certas para a etapa."""
    if stage == "head":
        freeze_backbone(model, arch)
    elif stage == "finetune":
        unfreeze_last_blocks(model, arch, cfg["finetune"]["unfreeze_last"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, help="sobrescreve model.arch da config")
    parser.add_argument("--smoke-test", action="store_true", help="poucos batches, 1 época")
    parser.add_argument("--num-workers", type=int, help="sobrescreve num_workers da config")
    parser.add_argument("--output-dir", help="sobrescreve output_dir (ex.: pasta no Google Drive)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("train")
    arch = args.arch or cfg["model"]["arch"]
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    max_batches = None
    if args.smoke_test:
        smoke = cfg["smoke_test"]
        cfg["batch_size"], max_batches = smoke["batch_size"], smoke["max_batches"]
        cfg["head"]["epochs"] = cfg["finetune"]["epochs"] = smoke["epochs"]

    set_seed(cfg["seed"])
    device = get_device()
    run_name = f"smoke_{arch}" if args.smoke_test else arch
    out_dir = project_path(args.output_dir or cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arch={arch} device={device} seed={cfg['seed']} saída={out_dir}")

    datasets = build_datasets(data_path(load_config("data"), "processed_dir"), cfg["image_size"])
    loaders = build_loaders(datasets, cfg["batch_size"], cfg["num_workers"], max_batches)
    classes = datasets["train"].classes
    print(
        {split: len(loader.dataset) for split, loader in loaders.items()}, f"{len(classes)} classes"
    )

    model = build_model(arch, len(classes), cfg["model"]["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    best = BestCheckpoint(out_dir / "best.pt", arch, classes, cfg["image_size"])

    history: list[dict] = []
    start = time.time()
    for stage, epochs, lr in plan_stages(cfg, arch):
        prepare_stage(model, arch, stage, cfg)
        trainable, total = count_trainable(model)
        print(f"\n== etapa {stage}: {epochs} épocas, lr={lr}, treináveis {trainable:,}/{total:,}")
        history += run_stage(
            stage, model, loaders, criterion, device, epochs, lr,
            cfg["optimizer"]["weight_decay"], best,
        )  # fmt: skip

    summary = {
        "arch": arch,
        "seed": cfg["seed"],
        "best_val_acc": best.best_acc,
        "minutes": round((time.time() - start) / 60, 1),
        "config": cfg,
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMelhor val_acc: {best.best_acc:.4f} -> {best.path}")


if __name__ == "__main__":  # obrigatório no Windows com num_workers > 0
    main()
