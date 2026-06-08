from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .state import Vocabulary, build_vocabulary, normalize_name


def text_descriptions(pool: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, str]]:
    descriptions = {"species": {}, "moves": {}, "items": {}, "abilities": {}}
    for species, sets in pool.items():
        descriptions["species"][normalize_name(species)] = f"Pokemon species {species} in Gen 9 random battles."
        for candidate in sets or []:
            for move in candidate.get("moves", []) or []:
                descriptions["moves"][normalize_name(move)] = f"Pokemon move {move}."
            item = candidate.get("item")
            ability = candidate.get("ability")
            if item:
                descriptions["items"][normalize_name(item)] = f"Pokemon held item {item}."
            if ability:
                descriptions["abilities"][normalize_name(ability)] = f"Pokemon ability {ability}."
    return descriptions


def build_e5_embeddings(
    pool_path: str | Path = "data/gen9_random_pool.json",
    output_path: str | Path = "data/e5_embeddings.npz",
    model_name: str = "intfloat/e5-small-v2",
) -> Path:
    pool = json.loads(Path(pool_path).read_text())
    vocab = build_vocabulary(pool)
    descriptions = text_descriptions(pool)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise RuntimeError("sentence-transformers is required to build e5 embeddings") from exc
    model = SentenceTransformer(model_name)
    arrays: dict[str, np.ndarray] = {}
    for group, mapping in (
        ("species", vocab.species),
        ("moves", vocab.moves),
        ("items", vocab.items),
        ("abilities", vocab.abilities),
    ):
        texts = [descriptions[group].get(name, f"Unknown Pokemon {group} token.") for name, _idx in sorted(mapping.items(), key=lambda item: item[1])]
        arrays[group] = model.encode(texts, normalize_embeddings=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return output


def load_embedding_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}
