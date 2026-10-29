import pytest
from PIL import Image

from pokedex.camera import CAMERA_WIDTHS, capture_preset, resize_to_width


@pytest.mark.parametrize(
    ("width", "preset"),
    [(320, "480p"), (640, "480p"), (800, "720p"), (960, "720p"), (1280, "1080p"), (1920, "1080p")],
)
def test_capture_preset_reaches_requested_width(width: int, preset: str) -> None:
    assert capture_preset(width) == preset


def test_widths_cover_320_to_1920() -> None:
    assert CAMERA_WIDTHS[0] == 320 and CAMERA_WIDTHS[-1] == 1920
    assert CAMERA_WIDTHS == sorted(CAMERA_WIDTHS)


def test_resize_to_width_keeps_aspect_ratio() -> None:
    image = Image.new("RGB", (1920, 1080))
    assert resize_to_width(image, 640).size == (640, 360)
    assert resize_to_width(image, 320).size == (320, 180)


def test_resize_to_width_never_upscales() -> None:
    image = Image.new("RGB", (1280, 720))
    assert resize_to_width(image, 1920) is image
