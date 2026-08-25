#!/usr/bin/env python3
"""Nested battle-grouped OOF gate for an abstaining action residual."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from train.action_semantic_residual import json_dump, sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")


def make_outer_folds(rows: list[dict[str, Any]], seed: int) -> dict[str, int]:
    groups = sorted({row["battle_id"] for row in rows})
    positive = sorted({row["battle_id"] for row in rows if row["persistent_correction"]}, key=lambda value: stable_key(value, seed))
    if len(positive) != 7:
        raise ValueError(f"frozen gate requires seven persistent-correction battles, got {len(positive)}")
    assignment = {battle: fold for fold, battle in enumerate(positive)}
    loads = [1] * 7
    for battle in sorted(set(groups) - set(positive), key=lambda value: stable_key(value, seed)):
        fold = min(range(7), key=lambda index: (loads[index], index))
        assignment[battle] = fold
        loads[fold] += 1
    return assignment


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], feature_key: str, c: float) -> np.ndarray:
    x_train = np.asarray([row[feature_key] for row in train], dtype=float)
    y_train = np.asarray([row["outcome_advantage"] for row in train], dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = np.column_stack((np.ones(len(x_train)), (x_train - mean) / scale))
    regularizer = np.eye(x.shape[1], dtype=float) / c
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + regularizer + np.eye(x.shape[1]) * 1e-9, x.T @ y_train)
    x_test = np.asarray([row[feature_key] for row in test], dtype=float)
    return np.column_stack((np.ones(len(test)), (x_test - mean) / scale)) @ beta


def decisions(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row["battle_id"], []).append((row, float(score)))
    selected = []
    for candidates in grouped.values():
        row, score = max(candidates, key=lambda item: (item[1], item[0]["action"]))
        if score >= threshold:
            selected.append({
                "root_id": row["root_id"], "battle_id": row["battle_id"], "action": row["action"],
                "score": score, "persistent_correction": row["persistent_correction"],
                "harmful": row["harmful"], "outcome_advantage": row["outcome_advantage"],
            })
    return selected


def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overrides": len(selected),
        "persistent_corrections_identified": sum(row["persistent_correction"] for row in selected),
        "harmful_overrides": sum(row["harmful"] for row in selected),
        "summed_development_advantage": sum(row["outcome_advantage"] for row in selected),
    }


def choose_threshold(rows: list[dict[str, Any]], scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(score) for score in scores}, reverse=True)
    candidates = [float(np.nextafter(max(candidates), np.inf)), *candidates]
    viable = []
    for threshold in candidates:
        selected = decisions(rows, scores, threshold)
        result = metrics(selected)
        if result["harmful_overrides"] == 0:
            viable.append((result["persistent_corrections_identified"], result["summed_development_advantage"], threshold, result))
    _, _, threshold, result = max(viable, key=lambda item: (item[0], item[1], item[2]))
    return threshold, result


def inner_oof(rows: list[dict[str, Any]], feature_key: str, c: float) -> np.ndarray:
    groups = sorted({row["battle_id"] for row in rows})
    scores = np.zeros(len(rows), dtype=float)
    for group in groups:
        train = [row for row in rows if row["battle_id"] != group]
        indices = [index for index, row in enumerate(rows) if row["battle_id"] == group]
        test = [rows[index] for index in indices]
        scores[indices] = fit_predict(train, test, feature_key, c)
    return scores


def nested_oof(rows: list[dict[str, Any]], assignment: dict[str, int], feature_key: str) -> dict[str, Any]:
    all_selected: list[dict[str, Any]] = []
    folds = []
    for fold in range(7):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        test = [row for row in rows if assignment[row["battle_id"]] == fold]
        choices = []
        for c in (0.01, 0.1, 1.0, 10.0):
            inner_scores = inner_oof(train, feature_key, c)
            threshold, inner_metrics = choose_threshold(train, inner_scores)
            choices.append((inner_metrics["persistent_corrections_identified"], inner_metrics["summed_development_advantage"], -inner_metrics["overrides"], -c, c, threshold, inner_metrics))
        *_, c, threshold, inner_metrics = max(choices)
        test_scores = fit_predict(train, test, feature_key, c)
        selected = decisions(test, test_scores, threshold)
        all_selected.extend(selected)
        folds.append({
            "fold": fold, "train_battles": len({row["battle_id"] for row in train}),
            "test_battles": len({row["battle_id"] for row in test}), "c": c,
            "threshold": threshold, "inner_oof_metrics": inner_metrics,
            "test_metrics": metrics(selected), "selected": selected,
        })
    return {"metrics": metrics(all_selected), "selected": all_selected, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    assignment = make_outer_folds(rows, args.seed)
    baseline = nested_oof(rows, assignment, "baseline_features")
    enriched = nested_oof(rows, assignment, "enriched_features")
    bm, em = baseline["metrics"], enriched["metrics"]
    strictly_beats = (
        em["harmful_overrides"] <= bm["harmful_overrides"]
        and (
            em["persistent_corrections_identified"] > bm["persistent_corrections_identified"]
            or (
                em["persistent_corrections_identified"] == bm["persistent_corrections_identified"]
                and em["summed_development_advantage"] > bm["summed_development_advantage"] + 1e-12
            )
        )
    )
    passed = em["persistent_corrections_identified"] >= 3 and em["harmful_overrides"] == 0 and strictly_beats
    report = {
        "schema": "metagross-action-semantic-oof-gate/v1",
        "claim_status": "development_only_not_final_confirmation",
        "dataset_sha256": sha256(args.dataset), "seed": args.seed,
        "grouping": "nested_battle_grouped_oof_seven_outer_folds_one_persistent_root_each",
        "model": "standardized_l2_ridge_outcome_advantage_residual",
        "thresholding": "inner_oof_maximize_persistent_corrections_subject_to_zero_harmful_overrides",
        "baseline": baseline, "enriched": enriched,
        "frozen_pass_rule": {
            "minimum_persistent_corrections": 3, "maximum_harmful_overrides": 0,
            "must_strictly_beat_24_feature_baseline": True,
        },
        "strictly_beats_baseline": strictly_beats, "passed": passed,
    }
    json_dump(args.output, report)
    print(json.dumps({"baseline": bm, "enriched": em, "strictly_beats_baseline": strictly_beats, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
