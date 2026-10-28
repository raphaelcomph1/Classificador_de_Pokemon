import pytest
import torch
from conftest import todo

from pokedex.models.build import (
    PRETRAINED_ARCHS,
    SmallCNN,
    build_model,
    count_trainable,
    freeze_backbone,
    head_parameters,
    load_backbone,
    unfreeze_last_blocks,
)

NUM_CLASSES = 150
x = torch.randn(2, 3, 64, 64)  # imagens pequenas: o teste fica rápido


def test_small_cnn_output_shape() -> None:
    assert SmallCNN(NUM_CLASSES)(x).shape == (2, NUM_CLASSES)


def test_build_model_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError):
        build_model("vgg_qualquer", NUM_CLASSES, pretrained=False)


@todo
@pytest.mark.parametrize("arch", PRETRAINED_ARCHS)
def test_replace_head_outputs_num_classes(arch: str) -> None:
    model = build_model(arch, NUM_CLASSES, pretrained=False).eval()
    assert model(x).shape == (2, NUM_CLASSES)


@todo
@pytest.mark.parametrize("arch", PRETRAINED_ARCHS)
def test_replace_head_keeps_backbone(arch: str) -> None:
    """Só a cabeça muda: o resto do modelo tem os mesmos parâmetros do original."""
    original = load_backbone(arch, pretrained=False)
    model = build_model(arch, NUM_CLASSES, pretrained=False)
    n_head = sum(p.numel() for p in head_parameters(model, arch))
    n_head_original = sum(p.numel() for p in head_parameters(original, arch))
    assert count_trainable(model)[1] - n_head == count_trainable(original)[1] - n_head_original


@todo
@pytest.mark.parametrize("arch", PRETRAINED_ARCHS)
def test_freeze_backbone_leaves_only_head_trainable(arch: str) -> None:
    model = build_model(arch, NUM_CLASSES, pretrained=False)
    freeze_backbone(model, arch)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in head_parameters(model, arch)}


@todo
@pytest.mark.parametrize("arch", PRETRAINED_ARCHS)
def test_unfreeze_last_blocks_adds_more_params(arch: str) -> None:
    model = build_model(arch, NUM_CLASSES, pretrained=False)
    freeze_backbone(model, arch)
    head_only, _ = count_trainable(model)
    unfreeze_last_blocks(model, arch, 1)
    one_block, _ = count_trainable(model)
    unfreeze_last_blocks(model, arch, 2)
    two_blocks, total = count_trainable(model)
    assert head_only < one_block < two_blocks < total  # nunca descongela tudo
    assert all(p.requires_grad for p in head_parameters(model, arch))
