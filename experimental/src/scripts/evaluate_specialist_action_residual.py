#!/usr/bin/env python3
"""Nested grouped OOF gate for label-blind action-family specialists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_action_semantic_residual import metrics, read_jsonl
from scripts.evaluate_compact_action_semantic_residual import grouped_folds
from train.action_semantic_residual import (
    ENRICHED_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
    json_dump,
    sha256,
)


SCHEMA = "metagross-specialist-action-residual-oof/v1"
FAMILIES = ("switch_option", "status_tempo", "direct_attack")

COMMON = (
    "visit_delta", "value_delta", "top_vote_delta", "value_std", "visit_std",
    "world_support", "world_top_vote", "root_disagreement", "root_js",
    "root_top_margin", "r1_entropy", "r1_top_gap", "relative_live_visit_lower_tail",
    "relative_live_value_lower_tail", "relative_live_world_value_range",
    "relative_live_world_top_vote",
)
FEATURE_POOLS = {
    "switch_option": (*COMMON, "action_is_switch", "baseline_is_switch",
        "action_type_changed", "relative_own_team_hp_delta_mean",
        "relative_own_team_hp_delta_lower_tail", "relative_own_alive_delta_mean",
        "relative_own_bench_hp_delta_mean", "relative_own_switch_depth_delta_mean",
        "relative_speed_advantage_delta_mean", "relative_turn_order_delta_mean",
        "relative_switch_entry_cost_mean", "relative_switch_entry_cost_lower_tail",
        "relative_preservation_value_mean", "relative_preservation_value_lower_tail"),
    "status_tempo": (*COMMON, "action_is_setup", "baseline_is_setup",
        "action_type_changed", "relative_own_team_hp_delta_mean",
        "relative_opponent_active_hp_deficit_delta_mean",
        "relative_boost_advantage_delta_mean", "relative_boost_advantage_delta_lower_tail",
        "relative_screen_advantage_delta_mean", "relative_hazard_advantage_delta_mean",
        "relative_hazard_advantage_delta_lower_tail", "relative_substitute_advantage_delta_mean",
        "relative_damage_tempo_delta_mean", "relative_speed_advantage_delta_mean",
        "relative_turn_order_delta_mean", "relative_preservation_value_mean"),
    "direct_attack": (*COMMON, "action_is_attack", "action_is_tera",
        "baseline_is_attack", "baseline_is_tera", "action_type_changed",
        "relative_own_team_hp_delta_mean", "relative_own_team_hp_delta_lower_tail",
        "relative_opponent_active_hp_deficit_delta_mean",
        "relative_opponent_active_hp_deficit_delta_lower_tail",
        "relative_opponent_fainted_delta_mean", "relative_damage_tempo_delta_mean",
        "relative_damage_tempo_delta_lower_tail", "relative_speed_advantage_delta_mean",
        "relative_turn_order_delta_mean"),
}


def action_family(row: dict[str, Any]) -> str:
    index = {name: offset for offset, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    values = row["semantic_features"]
    active = lambda name: float(values[index[name]]) > 0.5
    if active("action_is_switch") or active("baseline_is_switch"):
        return "switch_option"
    if active("action_is_setup") or active("baseline_is_setup"):
        return "status_tempo"
    if active("action_is_attack") and active("baseline_is_attack"):
        return "direct_attack"
    transition_names = (
        "relative_hazard_advantage_delta_mean", "relative_boost_advantage_delta_mean",
        "relative_speed_advantage_delta_mean", "relative_damage_tempo_delta_mean",
        "relative_switch_entry_cost_mean", "relative_preservation_value_mean",
    )
    if any(abs(float(values[index[name]])) > 1e-9 for name in transition_names):
        return "status_tempo"
    return "other"


def action_subfamily(row: dict[str, Any]) -> str:
    index = {name: offset for offset, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    values = row["semantic_features"]
    active = lambda name: float(values[index[name]]) > 0.5
    family = action_family(row)
    if family == "switch_option":
        if active("action_is_switch") and active("baseline_is_switch"):
            return "switch_to_switch"
        if active("action_is_switch"):
            return "attack_or_status_to_switch"
        return "switch_to_attack_or_status"
    if family == "status_tempo":
        if active("action_is_setup") and active("baseline_is_setup"):
            return "status_to_status"
        if active("action_is_setup"):
            return "attack_to_status"
        if active("baseline_is_setup"):
            return "status_to_attack"
        return "transition_only_tempo"
    if family == "direct_attack":
        if active("action_is_tera") and not active("baseline_is_tera"):
            return "tera_attack"
        if active("baseline_is_tera") and not active("action_is_tera"):
            return "preserve_tera_attack"
        return "direct_move_choice"
    return "other"


class BoostedStumps:
    def __init__(self, rounds: int, learning_rate: float):
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.initial = 0.0
        self.stumps: list[tuple[int, float, float, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedStumps":
        self.initial = float(y.mean())
        prediction = np.full(len(y), self.initial, dtype=float)
        self.stumps = []
        for _ in range(self.rounds):
            residual = y - prediction
            best: tuple[float, int, float, float, float] | None = None
            for column in range(x.shape[1]):
                values = x[:, column]
                thresholds = np.unique(np.quantile(values, np.linspace(0.10, 0.90, 9)))
                for threshold in thresholds:
                    left = values <= threshold
                    if left.sum() < 5 or (~left).sum() < 5:
                        continue
                    left_value = float(residual[left].mean())
                    right_value = float(residual[~left].mean())
                    error = float(np.square(residual - np.where(left, left_value, right_value)).sum())
                    candidate = (error, column, float(threshold), left_value, right_value)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, column, threshold, left_value, right_value = best
            self.stumps.append((column, threshold, left_value, right_value))
            prediction += self.learning_rate * np.where(x[:, column] <= threshold, left_value, right_value)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = np.full(len(x), self.initial, dtype=float)
        for column, threshold, left_value, right_value in self.stumps:
            result += self.learning_rate * np.where(x[:, column] <= threshold, left_value, right_value)
        return result


def feature_indices(family: str) -> list[int]:
    names = list(ENRICHED_FEATURE_NAMES)
    return [names.index(name) for name in FEATURE_POOLS[family]]


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], family: str,
                rounds: int, learning_rate: float) -> np.ndarray:
    indices = feature_indices(family)
    x = np.asarray([[row["enriched_features"][i] for i in indices] for row in train], dtype=float)
    y = np.asarray([row["outcome_advantage"] for row in train], dtype=float)
    test_x = np.asarray([[row["enriched_features"][i] for i in indices] for row in test], dtype=float)
    return BoostedStumps(rounds, learning_rate).fit(x, y).predict(test_x)


def selected(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float,
             family: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row["battle_id"], []).append((row, float(score)))
    result = []
    for candidates in grouped.values():
        row, score = max(candidates, key=lambda item: (item[1], item[0]["action"]))
        if score >= threshold and score > 0.0:
            result.append({
                "root_id": row["root_id"], "battle_id": row["battle_id"],
                "action": row["action"], "family": family or action_family(row),
                "score": score, "persistent_correction": row["durable_correction"],
                "harmful": row["harmful"], "outcome_advantage": row["outcome_advantage"],
            })
    return result


def choose_threshold(rows: list[dict[str, Any]], scores: np.ndarray, family: str) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(value) for value in scores if value > 0.0}, reverse=True)
    if not candidates:
        return float("inf"), metrics([])
    thresholds = [float(np.nextafter(max(candidates), np.inf)), *candidates]
    viable = []
    for threshold in thresholds:
        chosen = selected(rows, scores, threshold, family)
        result = metrics(chosen)
        if result["harmful_overrides"] == 0:
            viable.append((result["persistent_corrections_identified"],
                           result["summed_development_advantage"], -result["overrides"],
                           threshold, result))
    *_, threshold, result = max(viable)
    return threshold, result


def inner_oof(rows: list[dict[str, Any]], family: str, rounds: int,
              learning_rate: float, seed: int) -> np.ndarray:
    assignment = grouped_folds(rows, 5, seed)
    scores = np.zeros(len(rows), dtype=float)
    for fold in range(5):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        indices = [i for i, row in enumerate(rows) if assignment[row["battle_id"]] == fold]
        test = [rows[i] for i in indices]
        scores[indices] = fit_predict(train, test, family, rounds, learning_rate)
    return scores


def nested_family_oof(rows: list[dict[str, Any]], family: str,
                      outer: dict[str, int], seed: int) -> dict[str, Any]:
    predictions = []
    folds = []
    for fold in range(10):
        train = [row for row in rows if outer[row["battle_id"]] != fold]
        test = [row for row in rows if outer[row["battle_id"]] == fold]
        choices = []
        for rounds in (8, 16, 32):
            for rate in (0.05, 0.10):
                scores = inner_oof(train, family, rounds, rate, seed + fold * 101)
                threshold, result = choose_threshold(train, scores, family)
                choices.append((result["persistent_corrections_identified"],
                                result["summed_development_advantage"], -result["overrides"],
                                -rounds, -rate, rounds, rate, threshold, result))
        *_, rounds, rate, threshold, inner_result = max(choices)
        test_scores = fit_predict(train, test, family, rounds, rate)
        fold_predictions = []
        for row, score in zip(test, test_scores, strict=True):
            prediction = {
                "root_id": row["root_id"], "battle_id": row["battle_id"],
                "action": row["action"], "family": family, "score": float(score),
                "threshold": float(threshold), "eligible": bool(score >= threshold and score > 0.0),
                "persistent_correction": row["durable_correction"], "harmful": row["harmful"],
                "outcome_advantage": row["outcome_advantage"],
            }
            predictions.append(prediction); fold_predictions.append(prediction)
        chosen = [row for row in fold_predictions if row["eligible"]]
        # At most one alternative in a specialist can override each root.
        chosen = [max(group, key=lambda item: (item["score"], item["action"]))
                  for battle in sorted({row["battle_id"] for row in chosen})
                  for group in [[row for row in chosen if row["battle_id"] == battle]]]
        folds.append({"fold": fold, "rounds": rounds, "learning_rate": rate,
                      "threshold": threshold, "inner_oof_metrics": inner_result,
                      "test_metrics": metrics(chosen), "selected": chosen})
    chosen = [row for row in predictions if row["eligible"]]
    chosen = [max(group, key=lambda item: (item["score"], item["action"]))
              for battle in sorted({row["battle_id"] for row in chosen})
              for group in [[row for row in chosen if row["battle_id"] == battle]]]
    durable = sum(row["durable_correction"] for row in rows)
    result = metrics(chosen)
    minimum = max(3, math.ceil(0.25 * durable))
    admitted = result["harmful_overrides"] == 0 and result["persistent_corrections_identified"] >= minimum
    return {"rows": len(rows), "battles": len({row['battle_id'] for row in rows}),
            "durable_corrections": durable, "minimum_recovered": minimum,
            "features": list(FEATURE_POOLS[family]), "metrics": result,
            "admitted": admitted, "predictions": predictions, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--compact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taxonomy-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    for row in rows:
        row["family"] = action_family(row)
        row["subfamily"] = action_subfamily(row)
    taxonomy = {}
    for family in (*FAMILIES, "other"):
        subset = [row for row in rows if row["family"] == family]
        taxonomy[family] = {
            "rows": len(subset), "battles": len({row["battle_id"] for row in subset}),
            "durable_corrections": sum(row["durable_correction"] for row in subset),
            "harmful_alternatives": sum(row["harmful"] for row in subset),
            "subfamilies": {
                name: {
                    "rows": len(members),
                    "durable_corrections": sum(row["durable_correction"] for row in members),
                    "harmful_alternatives": sum(row["harmful"] for row in members),
                }
                for name in sorted({row["subfamily"] for row in subset})
                for members in [[row for row in subset if row["subfamily"] == name]]
            },
            "durable_examples": [
                {"root_id": row["root_id"], "baseline_action": row["baseline_action"],
                 "corrected_action": row["action"], "subfamily": row["subfamily"],
                 "outcome_advantage": row["outcome_advantage"]}
                for row in subset if row["durable_correction"]
            ],
        }
    json_dump(args.taxonomy_output, {"schema": "metagross-action-family-taxonomy/v1",
        "dataset_sha256": sha256(args.dataset), "assignment": "label_blind_protocol_precedence",
        "families": taxonomy})

    outer = grouped_folds(rows, 10, args.seed)
    specialists = {}
    for family in FAMILIES:
        specialists[family] = nested_family_oof(
            [row for row in rows if row["family"] == family], family, outer, args.seed)

    eligible = [prediction for family in FAMILIES if specialists[family]["admitted"]
                for prediction in specialists[family]["predictions"] if prediction["eligible"]]
    combined = [max(group, key=lambda item: (item["score"], item["action"]))
                for battle in sorted({row["battle_id"] for row in eligible})
                for group in [[row for row in eligible if row["battle_id"] == battle]]]
    combined_metrics = metrics(combined)
    compact = json.loads(args.compact_report.read_text())["compact"]["metrics"]
    strictly_beats = (combined_metrics["harmful_overrides"] <= compact["harmful_overrides"] and
        (combined_metrics["persistent_corrections_identified"] > compact["persistent_corrections_identified"] or
         (combined_metrics["persistent_corrections_identified"] == compact["persistent_corrections_identified"] and
          combined_metrics["summed_development_advantage"] > compact["summed_development_advantage"] + 1e-12)))
    passed = (combined_metrics["harmful_overrides"] == 0 and
              combined_metrics["persistent_corrections_identified"] >= 17 and strictly_beats)
    report = {"schema": SCHEMA, "claim_status": "development_only_not_confirmation",
        "dataset_sha256": sha256(args.dataset), "taxonomy_sha256": sha256(args.taxonomy_output),
        "seed": args.seed, "model": "family_specific_squared_error_boosted_stumps",
        "grouping": "ten_outer_folds_with_five_fold_inner_battle_grouped_oof",
        "thresholding": "inner_oof_positive_score_maximize_durable_corrections_subject_to_zero_harm",
        "specialists": specialists, "admitted_families": [f for f in FAMILIES if specialists[f]["admitted"]],
        "combined": {"metrics": combined_metrics, "selected": combined},
        "compact_baseline_metrics": compact, "minimum_global_recovered": 17,
        "strictly_beats_compact": strictly_beats, "passed": passed}
    json_dump(args.output, report)
    print(json.dumps({"taxonomy": taxonomy,
        "specialists": {f: {k: specialists[f][k] for k in ("durable_corrections", "minimum_recovered", "metrics", "admitted")} for f in FAMILIES},
        "admitted_families": report["admitted_families"], "combined": combined_metrics,
        "strictly_beats_compact": strictly_beats, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
