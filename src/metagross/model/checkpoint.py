from __future__ import annotations

from pathlib import Path
from dataclasses import fields
from typing import Any

import torch

from .network import PokeNet, PokeNetConfig
from .state import Vocabulary, build_vocabulary


def save_checkpoint(path: str | Path, model: PokeNet, optimizer: torch.optim.Optimizer | None = None, **metadata: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": model.config.__dict__,
        "metadata": metadata,
    }
    payload.update(metadata)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> PokeNet:
    payload = torch.load(path, map_location=map_location)
    allowed = {field.name for field in fields(PokeNetConfig)}
    config = PokeNetConfig(**{key: value for key, value in payload["config"].items() if key in allowed})
    model = PokeNet(config=config)
    model.load_state_dict(payload["model_state"])
    return model


def load_matching_weights(model: PokeNet, checkpoint_path: str | Path | dict[str, Any], strict_shapes: bool = True) -> dict[str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu") if not isinstance(checkpoint_path, dict) else checkpoint_path
    source = payload.get("model_state") or payload.get("state_dict") or payload
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key not in target:
            skipped += 1
            continue
        if strict_shapes and tuple(value.shape) != tuple(target[normalized_key].shape):
            skipped += 1
            continue
        matched[normalized_key] = value
    target.update(matched)
    model.load_state_dict(target)
    return {"matched": len(matched), "skipped": skipped}


def new_model_from_pool(pool: str | Path = "data/all_gen_pool.json") -> tuple[PokeNet, Vocabulary]:
    if not Path(pool).exists() and str(pool).endswith("all_gen_pool.json"):
        pool = "data/gen9_random_pool.json"
    vocab = build_vocabulary(pool)
    return PokeNet(vocab=vocab), vocab
