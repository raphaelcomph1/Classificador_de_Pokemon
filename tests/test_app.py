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
    assert len(at.radio[0].options) == 3
    assert any("#025" in md.value for md in at.markdown)
    assert any("Fraco contra" in md.value for md in at.markdown)
