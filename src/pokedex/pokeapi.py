"""Cliente da PokéAPI (https://pokeapi.co) com cache em disco.

Cada resposta é salva como JSON. Na segunda consulta do mesmo Pokémon não há requisição
de rede: a interface fica mais rápida e respeita o limite de uso da API pública.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Nome da classe no dataset -> nome na PokéAPI (o padrão é só minúsculas).
API_NAMES = {"MrMime": "mr-mime", "Alolan Sandslash": "sandslash-alola"}
# Nome da classe -> nome para exibir.
DISPLAY_NAMES = {"MrMime": "Mr. Mime", "Farfetchd": "Farfetch'd"}

STAT_NAMES = {
    "hp": "HP",
    "attack": "Ataque",
    "defense": "Defesa",
    "special-attack": "Ataque Esp.",
    "special-defense": "Defesa Esp.",
    "speed": "Velocidade",
}

TYPE_NAMES = {
    "normal": "Normal", "fire": "Fogo", "water": "Água", "electric": "Elétrico",
    "grass": "Planta", "ice": "Gelo", "fighting": "Lutador", "poison": "Venenoso",
    "ground": "Terra", "flying": "Voador", "psychic": "Psíquico", "bug": "Inseto",
    "rock": "Pedra", "ghost": "Fantasma", "dragon": "Dragão", "dark": "Sombrio",
    "steel": "Aço", "fairy": "Fada",
}  # fmt: skip

TYPE_COLORS = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0", "electric": "#F7D02C",
    "grass": "#7AC74C", "ice": "#96D9D6", "fighting": "#C22E28", "poison": "#A33EA1",
    "ground": "#E2BF65", "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC", "dark": "#705746",
    "steel": "#B7B7CE", "fairy": "#D685AD",
}  # fmt: skip


def api_name(label: str) -> str:
    """Nome usado na URL da PokéAPI para uma classe do modelo."""
    return API_NAMES.get(label, label.lower())


def display_name(label: str) -> str:
    """Nome bonito para a interface."""
    return DISPLAY_NAMES.get(label, label)


@dataclass
class PokemonInfo:
    """Dados de um Pokémon prontos para exibir."""

    label: str
    name: str
    number: int
    types: list[str]
    stats: dict[str, int]
    height_m: float
    weight_kg: float
    sprite_url: str | None
    multipliers: dict[str, float] = field(default_factory=dict)

    @property
    def weaknesses(self) -> dict[str, float]:
        """Tipos que causam dano aumentado (2x ou 4x), do maior para o menor."""
        return dict(
            sorted(((t, m) for t, m in self.multipliers.items() if m > 1), key=lambda x: -x[1])
        )

    @property
    def resistances(self) -> dict[str, float]:
        """Tipos que causam dano reduzido (0,5x ou 0,25x)."""
        return dict(
            sorted(((t, m) for t, m in self.multipliers.items() if 0 < m < 1), key=lambda x: x[1])
        )

    @property
    def immunities(self) -> list[str]:
        """Tipos que não causam dano (0x)."""
        return [t for t, m in self.multipliers.items() if m == 0]


def type_multipliers(damage_relations: list[dict[str, Any]]) -> dict[str, float]:
    """Multiplicador de dano RECEBIDO por tipo de ataque, combinando os tipos do Pokémon.

    Para dois tipos, os multiplicadores se multiplicam: ex. Charizard (Fogo/Voador) contra
    Pedra = 2 (fraco no Fogo) x 2 (fraco no Voador) = 4x. Só devolve tipos diferentes de 1x.
    """
    result: dict[str, float] = {}
    factors = {"double_damage_from": 2.0, "half_damage_from": 0.5, "no_damage_from": 0.0}
    for relations in damage_relations:
        for key, factor in factors.items():
            for attacking in relations[key]:
                name = attacking["name"]
                result[name] = result.get(name, 1.0) * factor
    return {t: m for t, m in result.items() if m != 1.0}


def species_number(species_url: str) -> int:
    """Número da Pokédex nacional a partir da URL da espécie (.../pokemon-species/28/)."""
    match = re.search(r"/pokemon-species/(\d+)/?$", species_url)
    if not match:
        raise ValueError(f"URL de espécie inesperada: {species_url}")
    return int(match.group(1))


class PokeAPIClient:
    """Busca dados na PokéAPI, com cache de cada resposta em `cache_dir`."""

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        timeout_s: float = 10,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def _cache_path(self, path: str) -> Path:
        return self.cache_dir / (path.strip("/").replace("/", "__") + ".json")

    def get(self, path: str) -> dict[str, Any]:
        """GET `base_url/path`, usando o cache em disco quando existir."""
        cache_file = self._cache_path(path)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        response = self.session.get(f"{self.base_url}/{path.strip('/')}", timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def pokemon(self, label: str) -> PokemonInfo:
        """Tipos, stats, tamanho, sprite e fraquezas de uma classe do modelo."""
        data = self.get(f"pokemon/{api_name(label)}")
        types = [t["type"]["name"] for t in sorted(data["types"], key=lambda t: t["slot"])]
        relations = [self.get(f"type/{t}")["damage_relations"] for t in types]
        artwork = data["sprites"].get("other", {}).get("official-artwork", {}).get("front_default")
        return PokemonInfo(
            label=label,
            name=display_name(label),
            number=species_number(data["species"]["url"]),
            types=types,
            stats={s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
            height_m=data["height"] / 10,  # a API usa decímetros
            weight_kg=data["weight"] / 10,  # e hectogramas
            sprite_url=artwork or data["sprites"].get("front_default"),
            multipliers=type_multipliers(relations),
        )
