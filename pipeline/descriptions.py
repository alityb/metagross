from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path: sys.path.insert(0, _p)

from pathlib import Path
from typing import Any

from metagross.model.state import build_vocabulary, normalize_name


URLS = {
    "pokedex": "https://play.pokemonshowdown.com/data/pokedex.json",
    "moves": "https://play.pokemonshowdown.com/data/moves.json",
    "items": "https://play.pokemonshowdown.com/data/items.json",
    "abilities": "https://play.pokemonshowdown.com/data/abilities.json",
}


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "AlphaPokemonDescriptionBuilder/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        with urllib.request.urlopen(request, timeout=60, context=ssl._create_unverified_context()) as response:
            return json.loads(response.read().decode("utf-8"))


def _species_description(name: str, pokedex: dict[str, Any]) -> str:
    entry = pokedex.get(name, {})
    display = entry.get("name") or name
    types = entry.get("types") or ["Unknown"]
    stats = entry.get("baseStats") or {}
    type_text = "/".join(types)
    return (
        f"{display} is a {type_text} Pokemon with base stats: "
        f"HP {stats.get('hp', '?')}, Atk {stats.get('atk', '?')}, Def {stats.get('def', '?')}, "
        f"SpA {stats.get('spa', '?')}, SpD {stats.get('spd', '?')}, Spe {stats.get('spe', '?')}."
    )


def _move_description(name: str, moves: dict[str, Any]) -> str:
    entry = moves.get(name, {})
    display = entry.get("name") or name
    effect = entry.get("shortDesc") or entry.get("desc") or "No additional effect text available."
    return f"{display}: {entry.get('type', 'Unknown')} {entry.get('category', 'Unknown')} move with {entry.get('basePower', 0)} base power. {effect}"


def _item_description(name: str, items: dict[str, Any]) -> str:
    entry = items.get(name, {})
    display = entry.get("name") or name
    effect = entry.get("shortDesc") or entry.get("desc") or "No item effect text available."
    return f"{display}: {effect}"


def _ability_description(name: str, abilities: dict[str, Any]) -> str:
    entry = abilities.get(name, {})
    display = entry.get("name") or name
    effect = entry.get("shortDesc") or entry.get("desc") or "No ability effect text available."
    return f"{display}: {effect}"


def build_descriptions(pool: str | Path) -> dict[str, dict[str, str]]:
    vocab = build_vocabulary(pool)
    pokedex = fetch_json(URLS["pokedex"])
    moves = fetch_json(URLS["moves"])
    items = fetch_json(URLS["items"])
    abilities = fetch_json(URLS["abilities"])
    return {
        "species": {name: _species_description(name, pokedex) for name in vocab.species if name != "<UNKNOWN>"},
        "moves": {name: _move_description(name, moves) for name in vocab.moves if name != "<UNKNOWN>"},
        "items": {name: _item_description(name, items) for name in vocab.items if name != "<UNKNOWN>"},
        "abilities": {name: _ability_description(name, abilities) for name in vocab.abilities if name != "<UNKNOWN>"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build entity description corpus for e5 embeddings")
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--output", default="data/descriptions.json")
    args = parser.parse_args()
    descriptions = build_descriptions(args.pool)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(descriptions, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({group: len(values) for group, values in descriptions.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
