from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path: sys.path.insert(0, _p)

from pathlib import Path

import torch


def species_description(name: str, entry: dict) -> str:
    types = "/".join(entry.get("types", ["Normal"]))
    stats = entry.get("baseStats", {})
    return (
        f"passage: {name} is a {types} Pokemon. Types: {types} {types}. "
        f"HP {stats.get('hp', 0)} Atk {stats.get('atk', 0)} Def {stats.get('def', 0)} "
        f"SpA {stats.get('spa', 0)} SpD {stats.get('spd', 0)} Spe {stats.get('spe', 0)}."
    )


def move_description(name: str, entry: dict) -> str:
    bp = entry.get("basePower", 0)
    typ = entry.get("type", "Normal")
    cat = entry.get("category", "Physical")
    desc = entry.get("shortDesc") or entry.get("desc") or ""
    return f"passage: {name} is a {typ} {cat} move with {bp} base power. {typ} {typ}. {desc}"


def encode_dict(model, desc_dict: dict[str, str], batch_size: int) -> dict[str, torch.Tensor]:
    names = list(desc_dict.keys())
    vecs = model.encode(list(desc_dict.values()), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return {name: torch.as_tensor(vecs[index], dtype=torch.float32) for index, name in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen e5-small-v2 entity embeddings from all-gen pool data")
    parser.add_argument("--pool", default="data/all_gen_pool.json")
    parser.add_argument("--pokedex", default="data/pokedex.json")
    parser.add_argument("--moves", default="data/moves.json")
    parser.add_argument("--output", default="data/entity_embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    pokedex = json.loads(Path(args.pokedex).read_text(encoding="utf-8"))
    moves = json.loads(Path(args.moves).read_text(encoding="utf-8"))
    species_descs = {
        name: species_description(name, (pool["species"].get(name, {}) or {}).get("pokedex") or pokedex.get(name, {}))
        for name in pool["species"].keys()
    }
    move_descs = {name: move_description(name, pool["moves"].get(name, {}) or moves.get(name, {})) for name in pool["moves"].keys()}
    item_descs = {name: f"passage: {name} is a Pokemon held item." for name in pool.get("items", [])}
    ability_descs = {name: f"passage: {name} is a Pokemon ability." for name in pool.get("abilities", [])}
    model = SentenceTransformer(args.model)
    embeddings = {
        "species": encode_dict(model, species_descs, args.batch_size),
        "moves": encode_dict(model, move_descs, args.batch_size),
        "items": encode_dict(model, item_descs, args.batch_size),
        "abilities": encode_dict(model, ability_descs, args.batch_size),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output)
    print(json.dumps({"output": str(output), **{key: len(value) for key, value in embeddings.items()}}, indent=2))


if __name__ == "__main__":
    main()
