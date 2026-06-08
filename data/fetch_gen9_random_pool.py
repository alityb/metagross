from __future__ import annotations

import argparse
import json
import math
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SETS_URL = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/random-battles/gen9/sets.json"
POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "AlphaPokemonPoolBuilder/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        with urllib.request.urlopen(request, timeout=60, context=ssl._create_unverified_context()) as response:
            return json.loads(response.read().decode("utf-8"))


def display_name(species_id: str, pokedex: dict[str, Any]) -> str:
    entry = pokedex.get(species_id) or {}
    return str(entry.get("name") or species_id)


def speed_stat(species_id: str, level: int, pokedex: dict[str, Any]) -> int:
    entry = pokedex.get(species_id) or {}
    base_speed = int((entry.get("baseStats") or {}).get("spe", 80))
    ev = 84
    iv = 31
    return math.floor(((2 * base_speed + iv + math.floor(ev / 4)) * level) / 100) + 5


def infer_item(role: str, movepool: list[str], ability: str) -> str:
    role_l = role.lower()
    moves = {normalize(move) for move in movepool}
    if "av pivot" in role_l:
        return "Assault Vest"
    if "trick" in moves or "switcheroo" in moves:
        return "Choice Scarf"
    if "bellydrum" in moves:
        return "Sitrus Berry"
    if "eviolite" in role_l:
        return "Eviolite"
    if "setup sweeper" in role_l or "wallbreaker" in role_l or "fast attacker" in role_l:
        return "Life Orb"
    if "bulky" in role_l or "support" in role_l:
        return "Leftovers"
    if ability.lower().replace(" ", "") in {"guts", "quickfeet"}:
        return "Flame Orb"
    return "Leftovers"


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def choose_four_moves(movepool: list[str]) -> list[str]:
    moves = [str(move) for move in movepool[:4]]
    while len(moves) < 4:
        moves.append("Struggle")
    return moves


def convert_pool(sets_data: dict[str, Any], pokedex: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, list[dict[str, Any]]] = {}
    for species_id, species_data in sorted(sets_data.items()):
        level = int(species_data.get("level", 80))
        species_name = display_name(species_id, pokedex)
        entries: list[dict[str, Any]] = []
        for set_index, random_set in enumerate(species_data.get("sets") or []):
            movepool = [str(move) for move in random_set.get("movepool") or []]
            abilities = [str(ability) for ability in random_set.get("abilities") or [""]]
            ability = abilities[0] if abilities else ""
            role = str(random_set.get("role") or "")
            entries.append(
                {
                    "moves": choose_four_moves(movepool),
                    "item": infer_item(role, movepool, ability),
                    "ability": ability,
                    "stats": {"spe": speed_stat(species_id, level, pokedex)},
                    "role": role,
                    "level": level,
                    "source_set_index": set_index,
                }
            )
        if entries:
            pool[species_name] = entries
    return pool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch real Smogon Gen9 random battle pool")
    parser.add_argument("--output", default="data/gen9_random_pool.json")
    parser.add_argument("--sets-url", default=SETS_URL)
    parser.add_argument("--pokedex-url", default=POKEDEX_URL)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sets_data = fetch_json(args.sets_url)
    pokedex = fetch_json(args.pokedex_url)
    pool = convert_pool(sets_data, pokedex)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2, sort_keys=True))
    print(json.dumps({"species": len(pool), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
