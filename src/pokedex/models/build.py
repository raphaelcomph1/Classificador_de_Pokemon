"""Construção dos modelos: baseline do zero e transfer learning (torchvision)."""

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
    # A camada nova mantém as ENTRADAS da antiga (o que a base entrega: 512 na resnet18,
    # 2048 na resnet50, 1280 na mobilenet) e muda só as SAÍDAS (uma nota por Pokémon).
    # Ler `in_features` em vez de escrever o número faz o código valer para todas as arquiteturas.
    if arch.startswith("resnet"):
        # Na ResNet a cabeça é um atributo único: `(fc): Linear(512 -> 1000)`.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("mobilenet"):
        # Na MobileNet a cabeça é um Sequential; a camada de 1000 classes é a última da lista.
        # Mesma linha do exercício `rede2[-1] = nn.Linear(rede2[-1].in_features, 150)`.
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Sem regra de cabeça para {arch}")
    # A camada nova nasce com pesos aleatórios: tudo bem, ela é pequena e aprende rápido.
    return model


def _head(model: nn.Module, arch: str) -> nn.Module:
    """Módulo da cabeça: `fc` na ResNet, `classifier` inteiro na MobileNet."""
    # Na MobileNet o classifier todo (Linear 960->1280 + Hardswish + Dropout + Linear nova)
    # foi treinado para o ImageNet, então tratamos ele inteiro como cabeça.
    return model.fc if arch.startswith("resnet") else model.classifier


def head_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    """Parâmetros da cabeça (pesos e bias das camadas da cabeça)."""
    return list(_head(model, arch).parameters())


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Etapa 1: congela tudo, menos a cabeça."""
    # requires_grad=False -> o backward() não calcula gradiente para o parâmetro e o
    # otimizador não o altera. Bônus: economiza memória e tempo.
    for param in model.parameters():
        param.requires_grad = False
    for param in head_parameters(model, arch):
        param.requires_grad = True


def _backbone_blocks(model: nn.Module, arch: str) -> list[nn.Module]:
    """Blocos da base, do primeiro (bordas/cores) ao último (partes de objetos)."""
    if arch.startswith("resnet"):
        # Cada layerN tem vários blocos residuais (2 na resnet18, 3–6 na resnet50). Contamos
        # os blocos residuais, e não os layers: "3 blocos" fica com tamanho parecido nas duas
        # famílias (3 de 4 layers seria quase a rede inteira).
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        return [block for layer in layers for block in layer]
    return list(model.features)  # MobileNet: 17 blocos em sequência


def unfreeze_last_blocks(model: nn.Module, arch: str, n_blocks: int) -> None:
    """Etapa 2 (fine-tuning): descongela os `n_blocks` últimos blocos da base."""
    # Só os ÚLTIMOS: eles detectam coisas específicas (partes de objetos) e podem se
    # especializar em Pokémon. Os primeiros (bordas, cores) já servem para qualquer imagem.
    for block in _backbone_blocks(model, arch)[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


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
