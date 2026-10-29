"""Resolução da câmera: escolher a captura e reduzir para a largura pedida.

O `st.camera_input` só aceita 3 predefinições (480p, 720p, 1080p). Para oferecer qualquer
largura, pedimos a menor predefinição que alcança a largura e depois reduzimos a foto.
"""

from PIL import Image

CAMERA_WIDTHS = [320, 480, 640, 800, 1024, 1280, 1600, 1920]

# Altura de cada predefinição. A largura depende da câmera: assumimos 4:3 (o pior caso;
# câmeras 16:9 entregam ainda mais largura) para garantir que a captura alcance a largura pedida.
_PRESET_HEIGHTS = {"480p": 480, "720p": 720, "1080p": 1080}


def capture_preset(width: int) -> str:
    """Menor predefinição da câmera cuja largura (em 4:3) alcança `width`."""
    for preset, height in _PRESET_HEIGHTS.items():
        if height * 4 / 3 >= width:
            return preset
    return "1080p"  # o máximo disponível


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    """Reduz para `width` mantendo a proporção. Nunca aumenta (ampliar não cria detalhe)."""
    if image.width <= width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)
