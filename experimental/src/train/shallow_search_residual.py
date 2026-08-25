"""Selective residual controller over deployment-budget search statistics."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


MODEL_SCHEMA = "metagross-shallow-search-residual/v1"
FEATURE_NAMES = (
    "visit_mass", "mean_value", "value_std", "visit_std", "world_support", "world_top_vote",
    "visit_delta", "value_delta", "top_vote_delta", "visit_rank",
    "root_visit_entropy", "root_disagreement", "root_js", "root_top_mass", "root_top_margin",
    "root_action_count", "reveal_species", "reveal_moves", "reveal_items", "reveal_abilities",
    "history_fraction", "r1_entropy", "r1_top_gap", "r1_top_probability",
)


def battle_split(battle_id: str, seed: int = 20260817) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}\0{battle_id}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 60 else "calibration" if bucket < 80 else "test"


def is_ambiguous(root: dict[str, float]) -> bool:
    return (
        root["aggregate_top_visit_mass"] <= 0.65
        or root["aggregate_top_two_margin"] <= 0.20
        or root["weighted_top_action_disagreement"] >= 0.25
        or root["weighted_js_divergence"] >= 0.10
    )


def action_features(
    row: dict[str, Any],
    action: str,
    *,
    reveal: list[float],
    history: int,
    selection: dict[str, float],
) -> list[float]:
    stats = row["action_statistics"]
    current = stats[action]
    baseline = stats[row["selected_action"]]
    ordered = sorted(stats, key=lambda candidate: (stats[candidate]["visit_mass"], candidate), reverse=True)
    rank = ordered.index(action) / max(1, len(ordered) - 1)
    root = row["root_statistics"]
    values = [
        current["visit_mass"], current["mean_value"], current["value_std"], current["visit_std"],
        current["world_support"], current["world_top_vote"],
        current["visit_mass"] - baseline["visit_mass"], current["mean_value"] - baseline["mean_value"],
        current["world_top_vote"] - baseline["world_top_vote"], rank,
        root["visit_entropy"], root["weighted_top_action_disagreement"], root["weighted_js_divergence"],
        root["aggregate_top_visit_mass"], root["aggregate_top_two_margin"], root["action_count"] / 13.0,
        *reveal, min(1.0, history / 128.0), selection["policy_entropy"], selection["policy_top_gap"],
        selection["policy_top_probability"],
    ]
    if len(values) != len(FEATURE_NAMES) or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("invalid shallow-search residual feature vector")
    return [float(value) for value in values]


def ensemble_predict(models: list[Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.stack([model.predict(features) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0)


def choose_action(
    row: dict[str, Any],
    predictions: dict[str, float],
    *,
    conformal_penalty: float,
    margin: float = 0.01,
    minimum_world_support: float = 0.75,
) -> tuple[str, str, float]:
    baseline = row["selected_action"]
    if not is_ambiguous(row["root_statistics"]):
        return baseline, "not_ambiguous", float("-inf")
    eligible = [
        action for action in row["action_statistics"]
        if action != baseline and row["action_statistics"][action]["world_support"] >= minimum_world_support
    ]
    if not eligible:
        return baseline, "no_supported_alternative", float("-inf")
    selected = max(eligible, key=lambda action: (predictions[action] - conformal_penalty, action))
    lower_bound = predictions[selected] - conformal_penalty
    if lower_bound <= margin:
        return baseline, "lcb_below_margin", lower_bound
    return selected, "override", lower_bound
