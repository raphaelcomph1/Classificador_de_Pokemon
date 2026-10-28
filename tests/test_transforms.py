import torch
from PIL import Image

from pokedex.data.datasets import load_rgb
from pokedex.data.transforms import eval_transforms, train_transforms


def _image() -> Image.Image:
    return Image.new("RGB", (300, 200), (200, 50, 50))


def test_transforms_output_shape() -> None:
    for transform in (train_transforms(224), eval_transforms(224)):
        assert transform(_image()).shape == (3, 224, 224)


def test_eval_transforms_are_deterministic() -> None:
    t = eval_transforms(224)
    assert torch.equal(t(_image()), t(_image()))


def test_train_transforms_are_random() -> None:
    torch.manual_seed(0)
    img = Image.effect_noise((300, 200), 50).convert("RGB")
    t = train_transforms(224)
    assert not torch.equal(t(img), t(img))


def test_eval_transforms_normalize_with_imagenet_stats() -> None:
    # um pixel com exatamente a média do ImageNet vira ~0 depois da normalização
    mean_pixel = Image.new("RGB", (10, 10), (124, 116, 104))
    assert eval_transforms(8)(mean_pixel).abs().max() < 0.02


def test_load_rgb_uses_white_background_for_transparency(tmp_path) -> None:
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(path)  # totalmente transparente
    img = load_rgb(path)
    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 255, 255)
