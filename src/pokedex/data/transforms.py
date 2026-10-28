"""Transforms de imagem: augmentation só no treino; avaliação só resize + normalização."""

from torchvision import transforms

# Média e desvio padrão por canal (RGB) das imagens do ImageNet. Os modelos pré-treinados
# viram as imagens normalizadas assim, então as nossas precisam da mesma escala.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size: int) -> transforms.Compose:
    """Augmentation leve: recorte/zoom, espelho, rotação pequena, brilho/contraste.

    Sem mudança de matiz (hue): a cor é uma pista forte para distinguir Pokémon
    (ex.: Sandslash amarelo vs. Alolan Sandslash azul).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int) -> transforms.Compose:
    """Determinístico: mesma imagem -> mesmo tensor."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
