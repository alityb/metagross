#!/usr/bin/env python3
"""Train and honestly admit a selective 500ms-search residual controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import tempfile
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.run_public_mcts_leaf_gate import _load_oracle, _load_panel
from scripts.collect_shallow_search_statistics import SCHEMA as SEARCH_SCHEMA
from train.shallow_search_residual import (
    FEATURE_NAMES,
    MODEL_SCHEMA,
    action_features,
    battle_split,
    choose_action,
    ensemble_predict,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != SEARCH_SCHEMA for row in rows):
        raise ValueError("invalid shallow-search statistics")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="ascii") as handle:
        temporary = Path(handle.name)
        for row in rows:
            clean = {
                key: (value if not isinstance(value, float) or math.isfinite(value) else None)
                for key, value in row.items()
            }
            handle.write(json.dumps(clean, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    temporary.replace(path)


def bootstrap(values: list[float], seed: int, repeats: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    estimates = [math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats)]
    estimates.sort()
    return [estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats) - 1]]


def train_ensemble(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int, members: int = 16):
    models = []
    rng = np.random.default_rng(seed)
    for member in range(members):
        indices = rng.integers(0, len(x), len(x))
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=seed + member,
        )
        model.fit(x[indices], y[indices], sample_weight=weights[indices])
        models.append(model)
    return models


def build_examples(panel, oracle, search_rows):
    roots = {root["root_id"]: root for root in panel}
    units = []
    features = []
    targets = []
    weights = []
    example_keys = []
    missing_teacher_actions = 0
    for row in search_rows:
        root = roots.get(row["root_id"])
        reference = oracle.get(row["pair_id"])
        if root is None or reference is None or row["battle_id"] != root["battle_id"]:
            raise ValueError("shallow row does not align with panel/oracle")
        support = set(reference["action_values"])
        searchable_support = support.intersection(row["action_statistics"])
        missing_teacher_actions += len(support - searchable_support)
        baseline = row["selected_action"]
        if baseline not in support:
            raise ValueError("shallow selected action absent from teacher")
        unit_index = len(units)
        units.append({
            "row": row,
            "root": root,
            "reference": reference,
            "split": battle_split(root["battle_id"]),
            "example_indices": {},
        })
        for action in sorted(searchable_support):
            index = len(features)
            units[unit_index]["example_indices"][action] = index
            features.append(action_features(
                row,
                action,
                reveal=[float(value) for value in root["public_reveal_fractions"]],
                history=int(root["decision_idx"]) + 1,
                selection=root["selection"],
            ))
            targets.append(float(reference["action_values"][action]) - float(reference["action_values"][baseline]))
            weights.append(1.0 / len(searchable_support))
            example_keys.append((root["battle_id"], row["pair_id"], action))
    return units, np.asarray(features), np.asarray(targets), np.asarray(weights), example_keys, missing_teacher_actions


def evaluate(units, models, x, conformal_penalty, split):
    selected_units = [unit for unit in units if unit["split"] == split]
    all_indices = [index for unit in selected_units for index in unit["example_indices"].values()]
    means, deviations = ensemble_predict(models, x[all_indices])
    predictions = dict(zip(all_indices, means))
    uncertainty = dict(zip(all_indices, deviations))
    rows = []
    for unit in selected_units:
        search = unit["row"]
        reference = unit["reference"]
        by_action = {action: float(predictions[index]) for action, index in unit["example_indices"].items()}
        selected, reason, lower_bound = choose_action(
            search, by_action, conformal_penalty=conformal_penalty
        )
        baseline = search["selected_action"]
        best = reference["oracle_action"]
        values = reference["action_values"]
        rows.append({
            "battle_id": unit["root"]["battle_id"],
            "pair_id": search["pair_id"],
            "baseline_action": baseline,
            "candidate_action": selected,
            "oracle_action": best,
            "baseline_regret": float(reference["oracle_best_value"]) - float(values[baseline]),
            "candidate_regret": float(reference["oracle_best_value"]) - float(values[selected]),
            "reason": reason,
            "lower_bound": lower_bound,
            "prediction_std": float(uncertainty[unit["example_indices"][selected]]) if selected in unit["example_indices"] else 0.0,
        })
    battle_deltas = []
    for battle_id in sorted({row["battle_id"] for row in rows}):
        current = [row for row in rows if row["battle_id"] == battle_id]
        battle_deltas.append(math.fsum(row["baseline_regret"] - row["candidate_regret"] for row in current) / len(current))
    overrides = [row for row in rows if row["candidate_action"] != row["baseline_action"]]
    return {
        "battles": len(set(row["battle_id"] for row in rows)),
        "units": len(rows),
        "baseline_mean_regret": math.fsum(row["baseline_regret"] for row in rows) / len(rows),
        "candidate_mean_regret": math.fsum(row["candidate_regret"] for row in rows) / len(rows),
        "mean_regret_improvement": math.fsum(battle_deltas) / len(battle_deltas),
        "improvement_ci95": bootstrap(battle_deltas, 20260817 + len(rows)),
        "baseline_top1": sum(row["baseline_action"] == row["oracle_action"] for row in rows) / len(rows),
        "candidate_top1": sum(row["candidate_action"] == row["oracle_action"] for row in rows) / len(rows),
        "baseline_catastrophes": sum(row["baseline_regret"] >= 0.10 for row in rows),
        "candidate_catastrophes": sum(row["candidate_regret"] >= 0.10 for row in rows),
        "overrides": len(overrides),
        "beneficial_overrides": sum(row["candidate_regret"] < row["baseline_regret"] for row in overrides),
        "harmful_overrides": sum(row["candidate_regret"] > row["baseline_regret"] for row in overrides),
        "neutral_overrides": sum(row["candidate_regret"] == row["baseline_regret"] for row in overrides),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    search_rows = read_rows(args.shallow)
    if {row["pair_id"] for row in search_rows} != set(oracle):
        raise ValueError("shallow statistics do not cover exactly the teacher pairs")
    units, x, y, weights, example_keys, missing_teacher_actions = build_examples(panel, oracle, search_rows)
    counts = {split: len({unit["root"]["battle_id"] for unit in units if unit["split"] == split}) for split in ("train", "calibration", "test")}
    if min(counts.values()) < 200:
        raise ValueError(f"split too small: {counts}")
    train_units = [unit for unit in units if unit["split"] == "train"]
    train_indices = np.asarray([index for unit in train_units for index in unit["example_indices"].values()])
    models = train_ensemble(x[train_indices], y[train_indices], weights[train_indices], args.seed)

    calibration_units = [unit for unit in units if unit["split"] == "calibration"]
    calibration_indices = np.asarray([index for unit in calibration_units for index in unit["example_indices"].values()])
    calibration_mean, _ = ensemble_predict(models, x[calibration_indices])
    overprediction = calibration_mean - y[calibration_indices]
    conformal_penalty = float(np.quantile(overprediction, 0.90, method="higher"))
    calibration = evaluate(units, models, x, conformal_penalty, "calibration")
    test = evaluate(units, models, x, conformal_penalty, "test")
    admitted = (
        test["overrides"] >= 10
        and test["improvement_ci95"][0] > 0.0
        and test["candidate_mean_regret"] < test["baseline_mean_regret"]
        and test["candidate_catastrophes"] <= test["baseline_catastrophes"]
        and test["beneficial_overrides"] > test["harmful_overrides"]
    )
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_names": FEATURE_NAMES,
        "models": models,
        "conformal_penalty": conformal_penalty,
        "margin": 0.01,
        "minimum_world_support": 0.75,
        "ambiguity": {"top_mass": 0.65, "top_margin": 0.20, "disagreement": 0.25, "js": 0.10},
        "provenance": {"panel_sha256": panel_hash, "oracle_sha256": oracle_hash, "shallow_sha256": sha256(args.shallow)},
        "admitted": admitted,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.model.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(artifact, handle, protocol=5)
    temporary.replace(args.model)
    decision_rows = [
        {"split": split, **row}
        for split, result in (("calibration", calibration), ("test", test))
        for row in result.pop("rows")
    ]
    write_jsonl(args.decisions, decision_rows)
    report = {
        "schema": "metagross-shallow-search-residual-training/v1",
        "panel_sha256": panel_hash,
        "oracle_sha256": oracle_hash,
        "shallow_sha256": sha256(args.shallow),
        "model_sha256": sha256(args.model),
        "decisions_sha256": sha256(args.decisions),
        "feature_names": FEATURE_NAMES,
        "examples": len(x),
        "teacher_actions_without_shallow_statistics": missing_teacher_actions,
        "split_battles": counts,
        "conformal_penalty": conformal_penalty,
        "calibration": calibration,
        "test": test,
        "admitted": admitted,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("examples", "split_battles", "conformal_penalty", "calibration", "test", "admitted", "model_sha256")}, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
