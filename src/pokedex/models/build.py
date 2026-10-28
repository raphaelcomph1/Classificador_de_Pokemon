"""Construção dos modelos: baseline do zero e transfer learning (torchvision).

Partes marcadas com `# TODO` são suas (CLAUDE.md, Fase 2). Os testes em
`tests/test_model.py` começam como "xfail" e passam a valer quando você implementar.
"""

import torch
from torch import nn
from torchvision import models

PRETRAINED_ARCHS = ("mobilenet_v3_large", "resnet18", "resnet50")
ARCHS = ("small_cnn", *PRETRAINED_ARCHS)


class SmallCNN(nn.Module):
    """Baseline treinado do zero: 4 blocos conv -> média global -> linear.

    Serve de régua: se o transfer learning não ganhar com folga dele, algo está errado.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]
        for c_in, c_out in zip(channels, channels[1:], strict=False):
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(channels[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def load_backbone(arch: str, pretrained: bool) -> nn.Module:
    """Modelo do torchvision com a cabeça original (1000 classes do ImageNet)."""
    return models.get_model(arch, weights="DEFAULT" if pretrained else None)


def replace_head(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """Troca a última camada (1000 classes do ImageNet) por uma com `num_classes` saídas."""
    # TODO: troque a camada final do modelo.
    #
    # Passo 1 — descubra onde ela está. Rode no terminal:
    #     python -c "from pokedex.models.build import load_backbone; print(load_backbone('resnet18', False))"
    #   e repita com 'mobilenet_v3_large'. Olhe as ÚLTIMAS linhas: qual atributo guarda a
    #   camada `Linear(in_features=..., out_features=1000)`? Os nomes são diferentes nas duas famílias.
    #
    # Passo 2 — a camada nova precisa receber o MESMO número de entradas que a antiga
    #   (`in_features`) e devolver `num_classes` saídas. Leia `in_features` da camada antiga
    #   em vez de escrever o número fixo; assim o código vale para resnet18 E resnet50.
    #
    # Passo 3 — substitua com `nn.Linear(...)`. Dica: em um `nn.Sequential` dá para
    #   atribuir por índice, ex.: `seq[-1] = ...`.
    #
    # Use `arch.startswith("resnet")` / `arch.startswith("mobilenet")` para separar os casos.
    # Pergunta para pensar: por que a camada nova começa com pesos aleatórios e isso não é problema?
    raise NotImplementedError("TODO: replace_head")


def head_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    """Parâmetros da cabeça (a parte que você trocou em `replace_head`)."""
    # TODO: devolva `list(<módulo da cabeça>.parameters())`.
    #   Para o mobilenet, considere a cabeça como o `classifier` inteiro (não só a última Linear):
    #   as camadas antes dela no classifier também foram treinadas para o ImageNet.
    raise NotImplementedError("TODO: head_parameters")


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Etapa 1: congela tudo, menos a cabeça."""
    # TODO:
    #   1. percorra `model.parameters()` e desligue o gradiente (`param.requires_grad = False`);
    #   2. religue só os de `head_parameters(model, arch)`.
    # Pergunta para pensar: o que muda no `backward()` para um parâmetro com requires_grad=False?
    raise NotImplementedError("TODO: freeze_backbone")


def unfreeze_last_blocks(model: nn.Module, arch: str, n_blocks: int) -> None:
    """Etapa 2 (fine-tuning): descongela os `n_blocks` últimos blocos da base."""
    # TODO: liste os blocos da base em ordem e ligue `requires_grad = True` nos `n_blocks` últimos.
    #   - mobilenet_v3: os blocos estão em `model.features` (é um Sequential: dá para fatiar,
    #     `model.features[-n_blocks:]`).
    #   - resnet: os blocos "grandes" são `layer1`, `layer2`, `layer3`, `layer4`.
    #     Monte a lista [model.layer1, ..., model.layer4] e pegue os últimos `n_blocks`.
    # Pergunta para pensar: por que descongelar os ÚLTIMOS blocos e não os primeiros?
    raise NotImplementedError("TODO: unfreeze_last_blocks")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Monta o modelo pronto para `num_classes` classes."""
    if arch not in ARCHS:
        raise ValueError(f"Arquitetura desconhecida: {arch}. Opções: {ARCHS}")
    if arch == "small_cnn":
        return SmallCNN(num_classes)
    return replace_head(load_backbone(arch, pretrained), arch, num_classes)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(parâmetros treináveis, parâmetros totais)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())
