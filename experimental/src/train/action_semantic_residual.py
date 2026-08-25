"""Leak-free action-semantic features and development-panel guards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from train.resource_shadow import FEATURE_NAMES as RESOURCE_NAMES
from train.shallow_search_residual import FEATURE_NAMES as SEARCH_FEATURE_NAMES
from train.shallow_search_residual import action_features


SCHEMA = "metagross-action-semantic-residual/v1"
FREEZE_SCHEMA = "metagross-development-panel-freeze/v1"
SUMMARY_STATS = ("mean", "std", "lower_tail")
TRANSITION_SIGNALS = (
    "own_team_hp_delta",
    "own_alive_delta",
    "own_active_hp_delta",
    "own_bench_hp_delta",
    "own_switch_depth_delta",
    "opponent_active_hp_deficit_delta",
    "opponent_fainted_delta",
    "boost_advantage_delta",
    "screen_advantage_delta",
    "hazard_advantage_delta",
    "substitute_advantage_delta",
    "damage_tempo_delta",
    "speed_advantage_delta",
    "turn_order_delta",
    "switch_entry_cost",
    "preservation_value",
)
ABSOLUTE_SEMANTIC_FEATURE_NAMES = (
    "action_is_attack",
    "action_is_setup",
    "action_is_switch",
    "action_is_tera",
    *(f"{signal}_{stat}" for signal in TRANSITION_SIGNALS for stat in SUMMARY_STATS),
    "live_visit_lower_tail",
    "live_value_lower_tail",
    "live_world_value_range",
    "live_world_top_vote",
)
SEMANTIC_FEATURE_NAMES = (
    "action_is_attack", "action_is_setup", "action_is_switch", "action_is_tera",
    "baseline_is_attack", "baseline_is_setup", "baseline_is_switch", "baseline_is_tera",
    "action_type_changed",
    *(f"relative_{name}" for name in ABSOLUTE_SEMANTIC_FEATURE_NAMES[4:]),
)
ENRICHED_FEATURE_NAMES = (*SEARCH_FEATURE_NAMES, *SEMANTIC_FEATURE_NAMES)

_RESOURCE_INDEX = {name: index for index, name in enumerate(RESOURCE_NAMES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_freeze(
    rows: Sequence[dict[str, Any]], panel_path: Path, *, expected_count: int | None = None
) -> dict[str, Any]:
    root_ids = sorted({str(row["root_id"]) for row in rows})
    battle_ids = sorted({str(row["battle_id"]) for row in rows})
    if not root_ids or len(root_ids) != len(battle_ids):
        raise ValueError(f"development freeze requires one nonempty root per battle, got {len(root_ids)}/{len(battle_ids)}")
    if expected_count is not None and len(root_ids) != expected_count:
        raise ValueError(f"expected exactly {expected_count} development roots/battles, got {len(root_ids)}/{len(battle_ids)}")
    return {
        "schema": FREEZE_SCHEMA,
        "purpose": "development_only_never_final_confirmation",
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "root_count": len(root_ids),
        "battle_count": len(battle_ids),
        "root_ids": root_ids,
        "battle_ids": battle_ids,
        "confirmation_policy": "reject_any_root_or_battle_overlap",
    }


def assert_confirmation_disjoint(
    confirmation_rows: Iterable[dict[str, Any]], freeze: dict[str, Any]
) -> None:
    denied_roots = set(freeze["root_ids"])
    denied_battles = set(freeze["battle_ids"])
    root_overlap = sorted({str(row["root_id"]) for row in confirmation_rows} & denied_roots)
    battle_overlap = sorted({str(row["battle_id"]) for row in confirmation_rows} & denied_battles)
    if root_overlap or battle_overlap:
        raise ValueError(
            f"final confirmation overlaps development freeze: roots={len(root_overlap)}, battles={len(battle_overlap)}"
        )


def aggregate_search_features(
    search_rows: Sequence[dict[str, Any]], root: dict[str, Any], action: str
) -> list[float]:
    vectors = []
    reveal = [float(value) for value in root["source_context"]["public_reveal_fractions"]]
    history = int(root["source_context"].get("decision_idx", 0))
    selection = root["source_context"]["r1_selection"]
    for row in search_rows:
        if action not in row["action_statistics"]:
            raise ValueError(f"action {action!r} absent from live search statistics")
        vectors.append(action_features(row, action, reveal=reveal, history=history, selection=selection))
    values = np.asarray(vectors, dtype=float).mean(axis=0)
    return values.tolist()


def transition_signals(before_resource: Sequence[float], after_resource: Sequence[float], before_value: Sequence[float], after_value: Sequence[float], *, is_switch: bool) -> list[float]:
    delta = np.asarray(after_resource, dtype=float) - np.asarray(before_resource, dtype=float)
    vdelta = np.asarray(after_value, dtype=float) - np.asarray(before_value, dtype=float)
    own_hp_loss = max(0.0, -float(delta[_RESOURCE_INDEX["own_team_hp"]]))
    values = [
        delta[_RESOURCE_INDEX["own_team_hp"]],
        delta[_RESOURCE_INDEX["own_alive"]],
        delta[_RESOURCE_INDEX["own_active_hp"]],
        delta[_RESOURCE_INDEX["own_bench_hp"]],
        delta[_RESOURCE_INDEX["own_switch_depth"]],
        delta[_RESOURCE_INDEX["opponent_active_hp_deficit"]],
        delta[_RESOURCE_INDEX["opponent_fainted"]],
        delta[_RESOURCE_INDEX["boost_advantage"]],
        delta[_RESOURCE_INDEX["screen_advantage"]],
        delta[_RESOURCE_INDEX["hazard_advantage"]],
        delta[_RESOURCE_INDEX["substitute_advantage"]],
        vdelta[11],
        vdelta[12],
        vdelta[13],
        own_hp_loss if is_switch else 0.0,
        float(after_resource[_RESOURCE_INDEX["own_switch_depth"]]) - own_hp_loss if is_switch else 0.0,
    ]
    if len(values) != len(TRANSITION_SIGNALS) or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("invalid transition signals")
    return [float(value) for value in values]


def summarize_semantics(
    action: str,
    transition_rows: Sequence[Sequence[float]],
    world_policies: Sequence[float],
    world_values: Sequence[float],
    *,
    is_attack: bool | None = None,
) -> list[float]:
    matrix = np.asarray(transition_rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(TRANSITION_SIGNALS):
        raise ValueError("invalid semantic transition matrix")
    summaries: list[float] = []
    for column in matrix.T:
        summaries.extend((float(column.mean()), float(column.std()), float(np.quantile(column, 0.10))))
    policy = np.asarray(world_policies, dtype=float)
    values = np.asarray(world_values, dtype=float)
    is_switch = action.startswith("switch ")
    is_tera = action.endswith("-tera")
    damage = matrix[:, TRANSITION_SIGNALS.index("opponent_active_hp_deficit_delta")]
    attack = ((not is_switch) and bool(np.quantile(damage, 0.5) > 1e-6)) if is_attack is None else is_attack
    setup = (not is_switch) and not attack
    result = [
        float(attack), float(setup), float(is_switch), float(is_tera), *summaries,
        float(np.quantile(policy, 0.10)),
        float(np.quantile(values, 0.10)),
        float(values.max() - values.min()),
        float(np.mean(policy == np.max(policy))),
    ]
    if len(result) != len(ABSOLUTE_SEMANTIC_FEATURE_NAMES) or any(not math.isfinite(value) for value in result):
        raise ValueError("invalid semantic feature vector")
    return result


def residualize_semantics(candidate: Sequence[float], baseline: Sequence[float]) -> list[float]:
    if len(candidate) != len(ABSOLUTE_SEMANTIC_FEATURE_NAMES) or len(baseline) != len(candidate):
        raise ValueError("invalid absolute semantic vectors")
    candidate_type = [float(value) for value in candidate[:4]]
    baseline_type = [float(value) for value in baseline[:4]]
    result = [
        *candidate_type,
        *baseline_type,
        float(candidate_type != baseline_type),
        *(float(left) - float(right) for left, right in zip(candidate[4:], baseline[4:], strict=True)),
    ]
    if len(result) != len(SEMANTIC_FEATURE_NAMES) or any(not math.isfinite(value) for value in result):
        raise ValueError("invalid residual semantic vector")
    return result


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
