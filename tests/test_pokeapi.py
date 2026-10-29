from pathlib import Path
from typing import Any

import pytest

from pokedex.pokeapi import (
    PokeAPIClient,
    api_name,
    display_name,
    species_number,
    type_multipliers,
)


def _rel(double=(), half=(), none=()) -> dict[str, Any]:
    as_list = lambda names: [{"name": n} for n in names]  # noqa: E731
    return {
        "double_damage_from": as_list(double),
        "half_damage_from": as_list(half),
        "no_damage_from": as_list(none),
    }


FIRE = _rel(
    double=["water", "ground", "rock"], half=["fire", "grass", "bug", "steel", "fairy", "ice"]
)
FLYING = _rel(
    double=["electric", "ice", "rock"], half=["grass", "fighting", "bug"], none=["ground"]
)

POKEMON = {
    "id": 6,
    "species": {"url": "https://pokeapi.co/api/v2/pokemon-species/6/"},
    "types": [{"slot": 2, "type": {"name": "flying"}}, {"slot": 1, "type": {"name": "fire"}}],
    "stats": [
        {"base_stat": 78, "stat": {"name": "hp"}},
        {"base_stat": 100, "stat": {"name": "speed"}},
    ],
    "height": 17,
    "weight": 905,
    "sprites": {
        "front_default": "small.png",
        "other": {"official-artwork": {"front_default": "art.png"}},
    },
}


class FakeResponse:
    def __init__(self, data: dict) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.data


class FakeSession:
    """Simula a PokéAPI e conta quantas requisições foram feitas."""

    def __init__(self, routes: dict[str, dict]) -> None:
        self.routes, self.calls = routes, []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(self.routes[url.split("/api/v2/")[1]])


@pytest.fixture
def session() -> FakeSession:
    return FakeSession(
        {
            "pokemon/charizard": POKEMON,
            "type/fire": {"damage_relations": FIRE},
            "type/flying": {"damage_relations": FLYING},
        }
    )


def test_names() -> None:
    assert api_name("Pikachu") == "pikachu"
    assert api_name("MrMime") == "mr-mime"
    assert api_name("Alolan Sandslash") == "sandslash-alola"
    assert display_name("Farfetchd") == "Farfetch'd"
    assert display_name("Pikachu") == "Pikachu"


def test_species_number_uses_species_not_form_id() -> None:
    assert species_number("https://pokeapi.co/api/v2/pokemon-species/28/") == 28


def test_type_multipliers_combine_dual_types() -> None:
    m = type_multipliers([FIRE, FLYING])
    assert m["rock"] == 4.0  # fraco nos dois tipos
    assert m["water"] == 2.0
    assert m["ground"] == 0.0  # Voador é imune a Terra, mesmo Fogo sendo fraco
    assert m["grass"] == 0.25  # resiste nos dois
    assert "ice" not in m  # 0,5 (Fogo) x 2 (Voador) = 1x -> dano normal, não aparece


def test_pokemon_info(session: FakeSession, tmp_path: Path) -> None:
    client = PokeAPIClient("https://pokeapi.co/api/v2", tmp_path, session=session)
    info = client.pokemon("Charizard")
    assert info.number == 6
    assert info.types == ["fire", "flying"]  # ordenado pelo slot
    assert info.stats == {"hp": 78, "speed": 100}
    assert (info.height_m, info.weight_kg) == (1.7, 90.5)
    assert info.sprite_url == "art.png"
    assert list(info.weaknesses)[0] == "rock" and info.weaknesses["rock"] == 4.0
    assert info.immunities == ["ground"]
    assert info.resistances["grass"] == 0.25


def test_client_uses_disk_cache(session: FakeSession, tmp_path: Path) -> None:
    PokeAPIClient("https://pokeapi.co/api/v2", tmp_path, session=session).pokemon("Charizard")
    assert len(session.calls) == 3  # pokemon + 2 tipos

    # cliente novo (ex.: app reiniciado), mesma pasta de cache: nenhuma requisição nova
    PokeAPIClient("https://pokeapi.co/api/v2", tmp_path, session=session).pokemon("Charizard")
    assert len(session.calls) == 3
