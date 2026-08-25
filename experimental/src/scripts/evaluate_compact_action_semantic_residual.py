#!/usr/bin/env python3
"""Nested grouped OOF gate for a 10-20 feature outcome residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_action_semantic_residual import choose_threshold, decisions, metrics, read_jsonl
from train.action_semantic_residual import ENRICHED_FEATURE_NAMES, json_dump, sha256


COMPACT_CANDIDATES = (
    "visit_delta", "value_delta", "top_vote_delta", "value_std", "world_support",
    "root_disagreement", "root_js", "root_top_margin", "r1_entropy", "r1_top_gap",
    "action_is_switch", "baseline_is_switch", "action_type_changed",
    "relative_own_team_hp_delta_mean", "relative_opponent_active_hp_deficit_delta_mean",
    "relative_boost_advantage_delta_mean", "relative_hazard_advantage_delta_mean",
    "relative_damage_tempo_delta_mean", "relative_switch_entry_cost_mean",
    "relative_preservation_value_mean",
)


def stable_key(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")


def grouped_folds(rows: list[dict[str, Any]], fold_count: int, seed: int) -> dict[str, int]:
    positive = sorted({row["battle_id"] for row in rows if row["durable_correction"]}, key=lambda value: stable_key(value, seed))
    groups = sorted({row["battle_id"] for row in rows})
    assignment: dict[str, int] = {}
    loads = [0] * fold_count
    for index, group in enumerate(positive):
        fold = index % fold_count
        assignment[group] = fold; loads[fold] += 1
    for group in sorted(set(groups) - set(positive), key=lambda value: stable_key(value, seed)):
        fold = min(range(fold_count), key=lambda value: (loads[value], value))
        assignment[group] = fold; loads[fold] += 1
    return assignment


def selected_indices(train: list[dict[str, Any]], feature_key: str, k: int | None) -> list[int]:
    if feature_key == "baseline_features":
        return list(range(len(train[0][feature_key])))
    names = list(ENRICHED_FEATURE_NAMES)
    pool = [names.index(name) for name in COMPACT_CANDIDATES]
    if k is None:
        return pool
    x = np.asarray([[row[feature_key][index] for index in pool] for row in train], dtype=float)
    y = np.asarray([row["outcome_advantage"] for row in train], dtype=float)
    scores = []
    for column, index in enumerate(pool):
        values = x[:, column]
        score = 0.0 if values.std() < 1e-12 or y.std() < 1e-12 else abs(float(np.corrcoef(values, y)[0, 1]))
        scores.append((score, -index, index))
    return [item[2] for item in sorted(scores, reverse=True)[:k]]


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], feature_key: str, c: float, k: int | None) -> tuple[np.ndarray, list[int]]:
    indices = selected_indices(train, feature_key, k)
    x_train = np.asarray([[row[feature_key][index] for index in indices] for row in train], dtype=float)
    y_train = np.asarray([row["outcome_advantage"] for row in train], dtype=float)
    mean = x_train.mean(axis=0); scale = x_train.std(axis=0); scale[scale < 1e-12] = 1.0
    x = np.column_stack((np.ones(len(x_train)), (x_train - mean) / scale))
    regularizer = np.eye(x.shape[1], dtype=float) / c; regularizer[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + regularizer + np.eye(x.shape[1]) * 1e-9, x.T @ y_train)
    test_x = np.asarray([[row[feature_key][index] for index in indices] for row in test], dtype=float)
    return np.column_stack((np.ones(len(test)), (test_x - mean) / scale)) @ beta, indices


def inner_oof(rows: list[dict[str, Any]], feature_key: str, c: float, k: int | None, seed: int) -> np.ndarray:
    assignment = grouped_folds(rows, 5, seed)
    scores = np.zeros(len(rows), dtype=float)
    for fold in range(5):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        indices = [index for index, row in enumerate(rows) if assignment[row["battle_id"]] == fold]
        prediction, _ = fit_predict(train, [rows[index] for index in indices], feature_key, c, k)
        scores[indices] = prediction
    return scores


def nested_oof(rows: list[dict[str, Any]], assignment: dict[str, int], feature_key: str, seed: int) -> dict[str, Any]:
    all_selected = []; folds = []
    for fold in range(10):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        test = [row for row in rows if assignment[row["battle_id"]] == fold]
        choices = []
        ks = (None,) if feature_key == "baseline_features" else (10, 15, 20)
        for c in (0.1, 1.0, 10.0):
            for k in ks:
                scores = inner_oof(train, feature_key, c, k, seed + fold * 101)
                threshold, result = choose_threshold(train, scores)
                choices.append((result["persistent_corrections_identified"], result["summed_development_advantage"], -result["overrides"], -(k or 999), -c, c, k, threshold, result))
        *_, c, k, threshold, inner_metrics = max(choices)
        prediction, selected = fit_predict(train, test, feature_key, c, k)
        chosen = decisions(test, prediction, threshold)
        all_selected.extend(chosen)
        folds.append({
            "fold": fold, "c": c, "feature_count": len(selected),
            "selected_features": [ENRICHED_FEATURE_NAMES[index] if feature_key != "baseline_features" else index for index in selected],
            "threshold": threshold, "inner_oof_metrics": inner_metrics,
            "test_metrics": metrics(chosen), "selected": chosen,
        })
    return {"metrics": metrics(all_selected), "selected": all_selected, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    durable = sum(row["durable_correction"] for row in rows)
    if durable < 40:
        report = {
            "schema": "metagross-compact-action-semantic-oof-gate/v1", "dataset_sha256": sha256(args.dataset),
            "claim_status": "stopped_before_fit_insufficient_durable_corrections",
            "durable_corrections": durable, "minimum_required": 40, "passed": False,
        }
        json_dump(args.output, report); print(json.dumps(report, indent=2)); return
    assignment = grouped_folds(rows, 10, args.seed)
    baseline = nested_oof(rows, assignment, "baseline_features", args.seed)
    compact = nested_oof(rows, assignment, "enriched_features", args.seed)
    bm, cm = baseline["metrics"], compact["metrics"]
    minimum_recovered = math.ceil(durable * 0.30)
    strictly_beats = cm["harmful_overrides"] <= bm["harmful_overrides"] and (
        cm["persistent_corrections_identified"] > bm["persistent_corrections_identified"]
        or (cm["persistent_corrections_identified"] == bm["persistent_corrections_identified"] and cm["summed_development_advantage"] > bm["summed_development_advantage"] + 1e-12)
    )
    passed = cm["persistent_corrections_identified"] >= minimum_recovered and cm["harmful_overrides"] == 0 and strictly_beats
    report = {
        "schema": "metagross-compact-action-semantic-oof-gate/v1", "dataset_sha256": sha256(args.dataset),
        "claim_status": "development_only_not_confirmation", "durable_corrections": durable,
        "minimum_recovered": minimum_recovered, "candidate_features": list(COMPACT_CANDIDATES),
        "feature_selection": "training_fold_only_absolute_correlation_k_in_10_15_20",
        "baseline": baseline, "compact": compact, "strictly_beats_baseline": strictly_beats, "passed": passed,
    }
    json_dump(args.output, report)
    print(json.dumps({"durable": durable, "baseline": bm, "compact": cm, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
