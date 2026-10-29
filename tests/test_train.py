import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pokedex.models.build import SmallCNN
from pokedex.train import BestCheckpoint, evaluate_epoch, run_stage, train_one_epoch

DEVICE = torch.device("cpu")


def _loader(n: int = 16, num_classes: int = 4) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    images = torch.randn(n, 3, 32, 32, generator=g)
    labels = torch.arange(n) % num_classes
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def _params(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_train_one_epoch_updates_weights() -> None:
    torch.manual_seed(0)
    model = SmallCNN(4)
    before = _params(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, acc = train_one_epoch(model, _loader(), nn.CrossEntropyLoss(), optimizer, DEVICE)
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, _params(model), strict=True))


def test_evaluate_epoch_does_not_change_model() -> None:
    torch.manual_seed(0)
    model = SmallCNN(4)
    model_before = copy.deepcopy(model.state_dict())  # inclui as estatísticas do BatchNorm
    loss, acc = evaluate_epoch(model, _loader(), nn.CrossEntropyLoss(), DEVICE)
    assert loss > 0 and 0.0 <= acc <= 1.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, model_before[key]), f"{key} mudou durante a avaliação"


def test_can_overfit_tiny_dataset() -> None:
    """Teste clássico de sanidade: um modelo correto decora 16 exemplos rapidamente.

    Se este teste falhar, o loop tem bug (ex.: esqueceu zero_grad/step, labels trocados).
    """
    torch.manual_seed(0)
    model = SmallCNN(4)
    loader = _loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(40):
        train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, DEVICE)
    _, acc = evaluate_epoch(model, loader, nn.CrossEntropyLoss(), DEVICE)
    assert acc == 1.0


def test_run_stage_returns_history_and_saves_best(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = SmallCNN(4)
    loaders = {"train": _loader(), "val": _loader()}
    best = BestCheckpoint(tmp_path / "best.pt", "small_cnn", list("abcd"), 32)
    history = run_stage("full", model, loaders, nn.CrossEntropyLoss(), DEVICE, 2, 1e-3, 0.0, best)
    assert [h["epoch"] for h in history] == [1, 2] or [h["epoch"] for h in history] == [0, 1]
    assert {"stage", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"} <= history[0].keys()
    assert (tmp_path / "best.pt").exists()
