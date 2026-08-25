#!/usr/bin/env python3
"""Nested battle-grouped zero-harm gate for candidate-switch features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_action_semantic_residual import metrics, read_jsonl
from scripts.evaluate_compact_action_semantic_residual import grouped_folds
from scripts.evaluate_specialist_action_residual import BoostedStumps
from train.action_semantic_residual import json_dump, sha256
from train.candidate_switch_residual import CANDIDATE_FEATURE_NAMES
from train.shallow_search_residual import FEATURE_NAMES as SEARCH_FEATURE_NAMES


SCHEMA = "metagross-candidate-switch-oof/v1"
SEARCH_POOL = (
    "visit_mass", "mean_value", "value_std", "visit_std", "world_support",
    "world_top_vote", "visit_delta", "value_delta", "top_vote_delta",
    "visit_rank", "root_disagreement", "root_js", "root_top_margin",
    "r1_entropy", "r1_top_gap", "r1_top_probability",
)
FEATURE_NAMES = (*SEARCH_FEATURE_NAMES, *CANDIDATE_FEATURE_NAMES)
FEATURE_POOL = (*SEARCH_POOL, *CANDIDATE_FEATURE_NAMES)
FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in FEATURE_POOL)


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]],
                rounds: int, learning_rate: float) -> np.ndarray:
    x = np.asarray([[row["enriched_features"][i] for i in FEATURE_INDICES] for row in train], dtype=float)
    y = np.asarray([row["outcome_advantage"] for row in train], dtype=float)
    test_x = np.asarray([[row["enriched_features"][i] for i in FEATURE_INDICES] for row in test], dtype=float)
    return BoostedStumps(rounds, learning_rate).fit(x, y).predict(test_x)


def selected(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(str(row["battle_id"]), []).append((row, float(score)))
    output = []
    for candidates in grouped.values():
        row, score = max(candidates, key=lambda item: (item[1], item[0]["action"]))
        if score >= threshold and score > 0.0:
            output.append({
                "root_id": row["root_id"], "battle_id": row["battle_id"],
                "action": row["action"], "score": score,
                "persistent_correction": row["durable_correction"],
                "harmful": row["harmful"], "outcome_advantage": row["outcome_advantage"],
            })
    return output


def choose_threshold(rows: list[dict[str, Any]], scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    positives = sorted({float(value) for value in scores if value > 0.0}, reverse=True)
    if not positives:
        return float("inf"), metrics([])
    viable = []
    for threshold in [float(np.nextafter(max(positives), np.inf)), *positives]:
        result = metrics(selected(rows, scores, threshold))
        if result["harmful_overrides"] == 0:
            viable.append((result["persistent_corrections_identified"],
                           result["summed_development_advantage"], -result["overrides"],
                           threshold, result))
    *_, threshold, result = max(viable)
    return threshold, result


def inner_oof(rows: list[dict[str, Any]], rounds: int, learning_rate: float, seed: int) -> np.ndarray:
    assignment = grouped_folds(rows, 5, seed)
    scores = np.zeros(len(rows), dtype=float)
    for fold in range(5):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        indices = [index for index, row in enumerate(rows) if assignment[row["battle_id"]] == fold]
        scores[indices] = fit_predict(train, [rows[index] for index in indices], rounds, learning_rate)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    if (len(rows), len({row["battle_id"] for row in rows}),
            sum(bool(row["durable_correction"]) for row in rows),
            sum(bool(row["harmful"]) for row in rows)) != (383, 200, 56, 155):
        raise ValueError("frozen corpus contract changed")
    outer = grouped_folds(rows, 10, args.seed)
    predictions = []
    folds = []
    for fold in range(10):
        train = [row for row in rows if outer[row["battle_id"]] != fold]
        test = [row for row in rows if outer[row["battle_id"]] == fold]
        choices = []
        for rounds in (8, 16, 32):
            for rate in (0.05, 0.10):
                scores = inner_oof(train, rounds, rate, args.seed + fold * 101)
                threshold, result = choose_threshold(train, scores)
                choices.append((result["persistent_corrections_identified"],
                                result["summed_development_advantage"], -result["overrides"],
                                -rounds, -rate, rounds, rate, threshold, result))
        *_, rounds, rate, threshold, inner_result = max(choices)
        test_scores = fit_predict(train, test, rounds, rate)
        fold_predictions = []
        for row, score in zip(test, test_scores, strict=True):
            prediction = {
                "root_id": row["root_id"], "battle_id": row["battle_id"],
                "action": row["action"], "score": float(score),
                "threshold": float(threshold),
                "eligible": bool(score >= threshold and score > 0.0),
                "persistent_correction": row["durable_correction"],
                "harmful": row["harmful"], "outcome_advantage": row["outcome_advantage"],
            }
            predictions.append(prediction)
            fold_predictions.append(prediction)
        chosen = [row for row in fold_predictions if row["eligible"]]
        chosen = [max(group, key=lambda item: (item["score"], item["action"]))
                  for battle in sorted({row["battle_id"] for row in chosen})
                  for group in [[row for row in chosen if row["battle_id"] == battle]]]
        folds.append({
            "fold": fold, "rounds": rounds, "learning_rate": rate,
            "threshold": threshold, "inner_oof_metrics": inner_result,
            "test_metrics": metrics(chosen), "selected": chosen,
        })
    chosen = [row for row in predictions if row["eligible"]]
    chosen = [max(group, key=lambda item: (item["score"], item["action"]))
              for battle in sorted({row["battle_id"] for row in chosen})
              for group in [[row for row in chosen if row["battle_id"] == battle]]]
    result = metrics(chosen)
    prior = json.loads(args.prior_report.read_text())
    prior_metrics = prior["candidate"]["metrics"]
    strictly_beats = (
        result["harmful_overrides"] <= prior_metrics["harmful_overrides"]
        and (result["persistent_corrections_identified"] > prior_metrics["persistent_corrections_identified"]
             or (result["persistent_corrections_identified"] == prior_metrics["persistent_corrections_identified"]
                 and result["summed_development_advantage"]
                 > prior_metrics["summed_development_advantage"] + 1e-12))
    )
    admitted = (result["harmful_overrides"] == 0
                and result["persistent_corrections_identified"] >= 17
                and strictly_beats)
    report = {
        "schema": SCHEMA, "claim_status": "development_only_not_confirmation",
        "dataset_sha256": sha256(args.dataset), "prior_report_sha256": sha256(args.prior_report),
        "seed": args.seed, "model": "candidate_relative_squared_error_boosted_stumps",
        "grouping": "ten_outer_folds_with_five_fold_inner_battle_grouped_oof",
        "thresholding": "inner_oof_maximize_durable_corrections_subject_to_zero_harm",
        "features": list(FEATURE_POOL), "feature_count": len(FEATURE_POOL),
        "durable_corrections": 56, "minimum_recovered": 17,
        "metrics": result, "selected": chosen, "predictions": predictions, "folds": folds,
        "prior_switch_metrics": prior_metrics, "strictly_beats_prior_switch": strictly_beats,
        "admitted": admitted,
    }
    json_dump(args.output, report)
    print(json.dumps({
        "metrics": result, "minimum_recovered": 17,
        "strictly_beats_prior_switch": strictly_beats, "admitted": admitted,
    }, indent=2))


if __name__ == "__main__":
    main()
