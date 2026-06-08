from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any

from model.state import normalize_name


RANDBATS_URL = "https://pkmn.github.io/randbats/data/full/{gen}randombattle.json"


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "metagross/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return json.loads(r.read().decode("utf-8"))


def fetch_randbats_sets(gen: int) -> dict[str, Any]:
    """Fetch from pkmn.github.io/randbats — same source Foul Play uses."""
    url = RANDBATS_URL.format(gen=gen)
    try:
        data = _fetch_json(url)
        print(f"  Gen{gen}: {len(data)} species from pkmn.github.io/randbats")
        return data
    except Exception as exc:
        print(f"  Gen{gen}: randbats fetch failed ({exc}), using local fallback")
        return {}


def parse_randbats_entry(entry: dict[str, Any], level: int) -> list[dict[str, Any]]:
    """Parse pkmn.github.io/randbats format into our set list format.
    
    Input format:
    {
      "level": 84,
      "sets": [
        {"role": "Fast Attacker", "movepool": [...], "abilities": [...], "teraTypes": [...], "count": 5},
        ...
      ]
    }
    """
    sets_raw = entry.get("sets") or []
    result = []
    for s in sets_raw:
        if not isinstance(s, dict):
            continue
        abilities = s.get("abilities") or []
        ability = str(abilities[0]) if abilities else ""
        tera_types = s.get("teraTypes") or []
        moves = [str(m) for m in (s.get("movepool") or s.get("moves") or [])]
        count = int(s.get("count", 1) or 1)
        result.append({
            "moves": moves,
            "item": s.get("item", ""),
            "ability": ability,
            "role": s.get("role", ""),
            "level": int(entry.get("level", level)),
            "stats": s.get("stats", {}),
            "tera_type": str(tera_types[0]) if tera_types else "",
            "count": count,  # frequency weight — used for weighted sampling
        })
    return result


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ts_object_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("export ") or stripped.startswith("}"):
            continue
        if ": {" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip().strip('"\'')
        if key and key.replace("_", "").isalnum():
            keys.append(normalize_name(key))
    return keys


def normalize_moves(raw_set: dict[str, Any]) -> list[str]:
    moves = raw_set.get("moves") or raw_set.get("movepool") or []
    if isinstance(moves, dict):
        moves = list(moves)
    return [str(move) for move in moves]


def normalize_set(raw_set: dict[str, Any]) -> dict[str, Any]:
    abilities = raw_set.get("abilities") or raw_set.get("ability") or []
    if isinstance(abilities, str):
        ability = abilities
    elif isinstance(abilities, dict):
        ability = str(next(iter(abilities), ""))
    else:
        ability = str(abilities[0]) if abilities else ""
    item = raw_set.get("item") or raw_set.get("items") or ""
    if isinstance(item, list):
        item = str(item[0]) if item else ""
    elif isinstance(item, dict):
        item = str(next(iter(item), ""))
    return {
        "moves": normalize_moves(raw_set),
        "item": str(item),
        "ability": ability,
        "role": raw_set.get("role", ""),
        "level": raw_set.get("level"),
        "stats": raw_set.get("stats", {}),
    }


def read_gen_sets(gen: int, random_sets_dir: Path, ps_data_dir: Path | None, use_randbats: bool = True) -> dict[str, Any]:
    """Read gen-N random battle sets. Priority: pkmn.github.io/randbats > local file > PS server."""
    if use_randbats:
        randbats = fetch_randbats_sets(gen)
        if randbats:
            return randbats
    candidates = [random_sets_dir / f"gen{gen}.json"]
    if ps_data_dir is not None:
        candidates.append(ps_data_dir / "random-battles" / f"gen{gen}" / "sets.json")
    for path in candidates:
        if path.exists():
            return load_json(path)
    return {}


def build_pool(foul_play_dir: Path, random_sets_dir: Path, ps_data_dir: Path | None = None) -> dict[str, Any]:
    pokedex_path = Path("data/pokedex.json")
    moves_path = Path("data/moves.json")
    pokedex = load_json(pokedex_path if pokedex_path.exists() else foul_play_dir / "data" / "pokedex.json")
    moves = load_json(moves_path if moves_path.exists() else foul_play_dir / "data" / "moves.json")
    species: dict[str, dict[str, Any]] = {}
    items: set[str] = set()
    abilities: set[str] = set()
    for gen in range(1, 10):
        gen_sets = read_gen_sets(gen, random_sets_dir, ps_data_dir, use_randbats=True)
        print(f"Gen{gen}: {len(gen_sets)} species")
        for raw_name, raw_entry in gen_sets.items():
            species_id = normalize_name(raw_name)
            entry = species.setdefault(species_id, {"pokedex": pokedex.get(species_id, {}), "sets": {}})
            if not entry["pokedex"] and species_id in pokedex:
                entry["pokedex"] = pokedex[species_id]
            # Handle both pkmn.github.io/randbats format and local PS sets format
            if isinstance(raw_entry, dict) and "sets" in raw_entry:
                # randbats format: {level: N, sets: [{role, movepool, abilities, ...}]}
                normalized_sets = parse_randbats_entry(raw_entry, level=100)
            else:
                raw_sets = raw_entry.get("sets") if isinstance(raw_entry, dict) else None
                if raw_sets is None:
                    raw_sets = raw_entry if isinstance(raw_entry, list) else []
                normalized_sets = []
                for raw_set in raw_sets:
                    if not isinstance(raw_set, dict):
                        continue
                    normalized = normalize_set(raw_set)
                    normalized_sets.append(normalized)
            for ns in normalized_sets:
                if ns.get("item"):
                    items.add(normalize_name(ns["item"]))
                if ns.get("ability"):
                    abilities.add(normalize_name(ns["ability"]))
            if normalized_sets:
                entry["sets"][f"gen{gen}"] = normalized_sets
    # Include all Pokédex species so cross-generation forms are represented even
    # when a random-battle set file omits them.
    for raw_name, entry in pokedex.items():
        species.setdefault(normalize_name(raw_name), {"pokedex": entry, "sets": {}})
    # Fallback item/ability coverage when set files do not include generated items.
    fallback_items = [
        "leftovers", "choicescarf", "choiceband", "choicespecs", "lifeorb", "assaultvest", "heavydutyboots",
        "sitrusberry", "focussash", "eviolite", "rockyhelmet", "lightclay", "blacksludge", "expertbelt",
    ]
    items.update(fallback_items)
    if ps_data_dir is not None:
        items.update(ts_object_keys(ps_data_dir / "items.ts"))
        abilities.update(ts_object_keys(ps_data_dir / "abilities.ts"))
    for entry in pokedex.values():
        for ability in (entry.get("abilities") or {}).values() if isinstance(entry, dict) else []:
            abilities.add(normalize_name(ability))
    return {
        "species": species,
        "moves": {normalize_name(name): value for name, value in moves.items()},
        "items": sorted(item for item in items if item),
        "abilities": sorted(ability for ability in abilities if ability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Foul Play/PS all-generation random pools")
    parser.add_argument("--foul-play-dir", default="../foul-play")
    parser.add_argument("--random-sets-dir", default="data/random_sets")
    parser.add_argument("--ps-data-dir", default="../ps-server-gen9/data")
    parser.add_argument("--output", default="data/all_gen_pool.json")
    args = parser.parse_args()
    pool = build_pool(Path(args.foul_play_dir), Path(args.random_sets_dir), Path(args.ps_data_dir) if args.ps_data_dir else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"species": len(pool["species"]), "moves": len(pool["moves"]), "items": len(pool["items"]), "abilities": len(pool["abilities"])}, indent=2))


if __name__ == "__main__":
    main()
