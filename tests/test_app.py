"""Testes da interface com o AppTest do Streamlit (roda o app sem navegador).

O teste com upload usa o modelo treinado e a PokéAPI de mentira, para não depender
de internet. Sem `models/resnet18/best.pt` (ex.: clone novo do repositório), ele é pulado.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pokedex.config import load_config, project_path
from pokedex.pokeapi import PokemonInfo

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
HAS_MODEL = project_path(load_config("app")["checkpoint"]).exists()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 120), (250, 210, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_pokemon(self, label: str) -> PokemonInfo:
    return PokemonInfo(
        label=label,
        name=label,
        number=25,
        types=["electric"],
        stats={
            "hp": 35,
            "attack": 55,
            "defense": 40,
            "special-attack": 50,
            "special-defense": 50,
            "speed": 90,
        },
        height_m=0.4,
        weight_kg=6.0,
        sprite_url=None,
        multipliers={"ground": 2.0, "electric": 0.5, "flying": 0.5, "steel": 0.5},
    )


@pytest.mark.skipif(not HAS_MODEL, reason="modelo treinado não encontrado")
def test_app_starts_without_errors() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Pokédex AI" in at.title[0].value
    assert at.info[0].value.startswith("Envie uma imagem")


@pytest.mark.skipif(not HAS_MODEL, reason="modelo treinado não encontrado")
def test_app_shows_top3_and_card_after_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokedex.pokeapi.PokeAPIClient.pokemon", _fake_pokemon)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.file_uploader[0].upload("pikachu.png", _png_bytes(), "image/png").run()

    assert not at.exception
    assert len(at.progress) == 3 + 6  # 3 palpites + 6 stats
    assert len(at.radio(key="chosen").options) == 3  # "Ver na Pokédex": os 3 palpites
    assert any("#025" in md.value for md in at.markdown)
    assert any("Fraco contra" in md.value for md in at.markdown)


@pytest.mark.skipif(not HAS_MODEL, reason="modelo treinado não encontrado")
def test_camera_mode_has_resolution_selector() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("📷 Câmera").run()
    assert not at.exception
    slider = at.select_slider[0]
    assert slider.options[0] == "320 px" and slider.options[-1] == "1920 px"
    assert slider.value == 1280
    assert len(at.camera_input) == 1
    assert len(at.file_uploader) == 0  # só o campo do modo escolhido aparece


@pytest.mark.skipif(not HAS_MODEL, reason="modelo treinado não encontrado")
def test_paste_mode_renders() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("📋 Colar").run()
    assert not at.exception
    assert len(at.file_uploader) == 0 and len(at.camera_input) == 0
    assert at.info[0].value.startswith("Envie uma imagem")


class _FakeWebRtcContext:
    class state:  # noqa: N801
        playing = False


@pytest.mark.skipif(not HAS_MODEL, reason="modelo treinado não encontrado")
def test_scanner_mode_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    # O componente de vídeo precisa do servidor real do Streamlit; o AppTest usa um simulado.
    monkeypatch.setattr("streamlit_webrtc.webrtc_streamer", lambda **kwargs: _FakeWebRtcContext())
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="mode").set_value("🎯 Scanner ao vivo").run()
    assert not at.exception
    assert at.select_slider(key="scanner_width").value == 1280
    assert len(at.file_uploader) == 0 and len(at.camera_input) == 0
