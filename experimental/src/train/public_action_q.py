"""Leak-free action-conditioned features and a tiny public root-Q model."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


TYPES = (
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE",
    "FLYING", "GHOST", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
    "PSYCHIC", "ROCK", "STEEL", "WATER",
)
CATEGORIES = ("physical", "special", "status")
HASH_DIM = 32
FEATURE_COUNT = 176
MODEL_SCHEMA = "metagross_public_action_q_v1"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _one_hot(value: str, vocabulary: Sequence[str]) -> list[float]:
    normalized = str(value).upper()
    return [float(normalized == item.upper()) for item in vocabulary]


def _multi_hot(values: Sequence[str], vocabulary: Sequence[str]) -> list[float]:
    normalized = {str(value).upper() for value in values}
    return [float(item.upper() in normalized) for item in vocabulary]


def _signed_hash(token: str, dimensions: int = HASH_DIM) -> list[float]:
    output = [0.0] * dimensions
    for salt in range(4):
        digest = hashlib.sha256(f"{salt}\0{token}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        output[index] += sign * 0.5
    return output


def _active(side: Any) -> Any:
    try:
        index = int(side.active_index)
        return side.pokemon[index]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("state does not expose a valid active Pokemon") from exc


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator)
        return float(numerator) / denominator if denominator > 0 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _fraction(value: Any, *, scale: float) -> float:
    try:
        result = float(value) / scale
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _switch_target(state: Any, action: str) -> Any:
    query = _norm(action.removeprefix("switch "))
    active_index = int(state.side_one.active_index)
    candidates = []
    for index, pokemon in enumerate(state.side_one.pokemon):
        if index == active_index or float(pokemon.hp) <= 0:
            continue
        identity = _norm(str(pokemon.id))
        if query == identity or query.startswith(identity) or identity.startswith(query):
            candidates.append(pokemon)
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely resolve switch action {action!r}")
    return candidates[0]


def load_move_database(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise ValueError("move database must be an object")
    return rows


def action_features(
    state: Any,
    action: str,
    *,
    poke_engine: Any,
    move_database: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Return features that are invariant to sampled private opponent reserves."""
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a nonempty string")
    own = _active(state.side_one)
    opponent = _active(state.side_two)
    canonical = action.lower()
    is_switch = canonical.startswith("switch ")
    is_tera = canonical.endswith("-tera")
    is_move = not is_switch
    values = list(map(float, poke_engine.compute_public_value_features(state)))
    values += _multi_hot(own.types, TYPES)
    values += _multi_hot(opponent.types, TYPES)
    values += [float(is_move and not is_tera), float(is_switch), float(is_tera)]

    move_type = [0.0] * len(TYPES)
    category = [0.0] * len(CATEGORIES)
    move_scalars = [0.0] * 8
    switch_types = [0.0] * len(TYPES)
    switch_scalars = [0.0] * 8
    if is_move:
        move_id = canonical.removesuffix("-tera")
        move = move_database.get(move_id)
        if not isinstance(move, dict):
            raise ValueError(f"move database lacks {move_id!r}")
        move_type = _one_hot(str(move.get("type", "")), TYPES)
        category = _one_hot(str(move.get("category", "")), CATEGORIES)
        accuracy = move.get("accuracy", 100)
        accuracy = 100 if accuracy is True else accuracy
        drain = move.get("drain", [0, 1])
        recoil = move.get("recoil", [0, 1])
        heal = move.get("heal", [0, 1])
        current_pp = next(
            (candidate.pp for candidate in own.moves if str(candidate.id).lower() == move_id),
            0,
        )
        move_scalars = [
            _fraction(move.get("basePower", 0), scale=200.0),
            _fraction(accuracy, scale=100.0),
            _fraction(move.get("priority", 0), scale=7.0),
            _ratio(drain[0], drain[1]) if isinstance(drain, list) and len(drain) == 2 else 0.0,
            _ratio(recoil[0], recoil[1]) if isinstance(recoil, list) and len(recoil) == 2 else 0.0,
            _ratio(heal[0], heal[1]) if isinstance(heal, list) and len(heal) == 2 else 0.0,
            float(bool(move.get("flags", {}).get("contact"))),
            _ratio(current_pp, move.get("pp", 1) * 1.6),
        ]
    else:
        target = _switch_target(state, canonical)
        switch_types = _multi_hot(target.types, TYPES)
        switch_scalars = [
            _ratio(target.hp, target.maxhp),
            float(str(target.status).upper() != "NONE"),
            _fraction(target.attack, scale=500.0),
            _fraction(target.defense, scale=500.0),
            _fraction(target.special_attack, scale=500.0),
            _fraction(target.special_defense, scale=500.0),
            _fraction(target.speed, scale=500.0),
            _fraction(target.level, scale=100.0),
        ]
    values += move_type + category + move_scalars + switch_types + switch_scalars
    values += _signed_hash(canonical)
    values += _signed_hash(f"{str(own.id).lower()}|{str(opponent.id).lower()}|{canonical}")
    if len(values) != FEATURE_COUNT or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid action feature vector: {len(values)}")
    return np.asarray(values, dtype=np.float32)


class PublicActionQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_COUNT, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def save_model(path: Path, model: PublicActionQ, mean: torch.Tensor, std: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "metadata": metadata,
    }, path)


def load_model(path: Path) -> tuple[PublicActionQ, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("invalid public action-Q model schema")
    model = PublicActionQ()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    mean, std = payload["mean"].float(), payload["std"].float()
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,) or torch.any(std <= 0):
        raise ValueError("invalid action-Q normalization")
    return model, mean, std, dict(payload.get("metadata", {}))


@torch.no_grad()
def predict_priors(model: PublicActionQ, mean: torch.Tensor, std: torch.Tensor, features: np.ndarray, actions: Sequence[str], temperature: float = 0.05) -> list[tuple[str, float]]:
    if len(features) != len(actions) or not actions or temperature <= 0:
        raise ValueError("invalid action-Q prior request")
    tensor = (torch.as_tensor(features, dtype=torch.float32) - mean) / std
    advantages = model(tensor)
    probabilities = torch.softmax(advantages / temperature, dim=0)
    return [(action, float(probability)) for action, probability in zip(actions, probabilities, strict=True)]
