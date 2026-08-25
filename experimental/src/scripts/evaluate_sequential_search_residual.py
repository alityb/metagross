#!/usr/bin/env python3
"""Nested grouped OOF gate for an R1-history-aware search residual."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.evaluate_action_semantic_residual import metrics, read_jsonl
from scripts.evaluate_compact_action_semantic_residual import COMPACT_CANDIDATES, grouped_folds
from train.action_semantic_residual import ENRICHED_FEATURE_NAMES, json_dump, sha256


SCHEMA = "metagross-sequential-search-residual-oof/v1"


class TinyInteractionHead(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "metagross-outcome-r1-history-embeddings/v1":
        raise ValueError("invalid R1 history embedding schema")
    root_ids = payload.get("root_ids")
    values = payload.get("embeddings")
    if not isinstance(root_ids, list) or not isinstance(values, torch.Tensor) or values.shape != (len(root_ids), 900):
        raise ValueError("invalid R1 history embedding payload")
    result = {}
    for root_id, tensor in zip(root_ids, values.float(), strict=True):
        vector = tensor.numpy().astype(np.float64)
        scale = vector.std()
        result[str(root_id)] = (vector - vector.mean()) / (scale if scale >= 1e-12 else 1.0)
    return result, dict(payload.get("provenance", {}))


def projection(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + width * 1009)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(900), size=(900, width))
    return matrix


def raw_features(rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray], width: int, seed: int) -> np.ndarray:
    names = list(ENRICHED_FEATURE_NAMES)
    compact = [names.index(name) for name in COMPACT_CANDIDATES]
    matrix = projection(width, seed)
    result = []
    for row in rows:
        embedding = embeddings.get(row["root_id"])
        if embedding is None:
            raise ValueError(f"missing history embedding for root {row['root_id']}")
        result.append([row["enriched_features"][index] for index in compact] + (embedding @ matrix).tolist())
    return np.asarray(result, dtype=np.float32)


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], embeddings: dict[str, np.ndarray],
                width: int, hidden: int, seed: int, projection_seed: int) -> np.ndarray:
    x_train = raw_features(train, embeddings, width, projection_seed)
    x_test = raw_features(test, embeddings, width, projection_seed)
    mean = x_train.mean(axis=0); scale = x_train.std(axis=0); scale[scale < 1e-6] = 1.0
    x_train = torch.from_numpy((x_train - mean) / scale)
    x_test = torch.from_numpy((x_test - mean) / scale)
    target = torch.tensor([row["outcome_advantage"] for row in train], dtype=torch.float32)
    torch.manual_seed(seed)
    model = TinyInteractionHead(x_train.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(120):
        prediction = model(x_train)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x_test).numpy().astype(float)


def selected(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row["battle_id"], []).append((row, float(score)))
    result = []
    for candidates in grouped.values():
        row, score = max(candidates, key=lambda item: (item[1], item[0]["action"]))
        if score >= threshold and score > 0.0:
            result.append({"root_id": row["root_id"], "battle_id": row["battle_id"],
                "action": row["action"], "score": score,
                "persistent_correction": row["durable_correction"], "harmful": row["harmful"],
                "outcome_advantage": row["outcome_advantage"]})
    return result


def choose_threshold(rows: list[dict[str, Any]], scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(value) for value in scores if value > 0.0}, reverse=True)
    if not candidates:
        return float("inf"), metrics([])
    candidates = [float(np.nextafter(max(candidates), np.inf)), *candidates]
    viable = []
    for threshold in candidates:
        result = metrics(selected(rows, scores, threshold))
        if result["harmful_overrides"] == 0:
            viable.append((result["persistent_corrections_identified"],
                           result["summed_development_advantage"], -result["overrides"],
                           threshold, result))
    *_, threshold, result = max(viable)
    return threshold, result


def inner_oof(rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray], width: int,
              hidden: int, seed: int, projection_seed: int) -> np.ndarray:
    assignment = grouped_folds(rows, 5, seed)
    scores = np.zeros(len(rows), dtype=float)
    for fold in range(5):
        train = [row for row in rows if assignment[row["battle_id"]] != fold]
        indices = [index for index, row in enumerate(rows) if assignment[row["battle_id"]] == fold]
        test = [rows[index] for index in indices]
        scores[indices] = fit_predict(train, test, embeddings, width, hidden,
                                      seed + fold * 7919, projection_seed)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--compact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--projection-seed", type=int, default=2026090201)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = read_jsonl(args.dataset)
    embeddings, provenance = load_embeddings(args.embeddings)
    if {row["root_id"] for row in rows} != set(embeddings):
        raise ValueError("dataset and embedding root identities differ")
    outer = grouped_folds(rows, 10, args.seed)
    predictions = []
    folds = []
    for fold in range(10):
        train = [row for row in rows if outer[row["battle_id"]] != fold]
        test = [row for row in rows if outer[row["battle_id"]] == fold]
        choices = []
        for width in (8, 16, 32):
            for hidden in (4, 8):
                scores = inner_oof(train, embeddings, width, hidden,
                                   args.seed + fold * 101, args.projection_seed)
                threshold, result = choose_threshold(train, scores)
                choices.append((result["persistent_corrections_identified"],
                                result["summed_development_advantage"], -result["overrides"],
                                -width, -hidden, width, hidden, threshold, result))
        *_, width, hidden, threshold, inner_result = max(choices)
        scores = fit_predict(train, test, embeddings, width, hidden,
                             args.seed + fold * 104729, args.projection_seed)
        fold_selected = selected(test, scores, threshold)
        predictions.extend({"root_id": row["root_id"], "battle_id": row["battle_id"],
            "action": row["action"], "score": float(score), "threshold": float(threshold),
            "eligible": bool(score >= threshold and score > 0.0),
            "persistent_correction": row["durable_correction"], "harmful": row["harmful"],
            "outcome_advantage": row["outcome_advantage"]}
            for row, score in zip(test, scores, strict=True))
        folds.append({"fold": fold, "projection_width": width, "hidden_width": hidden,
            "parameter_count": (20 + width) * hidden + hidden + hidden + 1,
            "threshold": threshold, "inner_oof_metrics": inner_result,
            "test_metrics": metrics(fold_selected), "selected": fold_selected})
    chosen = [row for row in predictions if row["eligible"]]
    chosen = [max(group, key=lambda item: (item["score"], item["action"]))
              for battle in sorted({row["battle_id"] for row in chosen})
              for group in [[row for row in chosen if row["battle_id"] == battle]]]
    result = metrics(chosen)
    compact = json.loads(args.compact_report.read_text())["compact"]["metrics"]
    strictly_beats = (result["harmful_overrides"] <= compact["harmful_overrides"] and
        (result["persistent_corrections_identified"] > compact["persistent_corrections_identified"] or
         (result["persistent_corrections_identified"] == compact["persistent_corrections_identified"] and
          result["summed_development_advantage"] > compact["summed_development_advantage"] + 1e-12)))
    passed = result["harmful_overrides"] == 0 and result["persistent_corrections_identified"] >= 17 and strictly_beats
    report = {"schema": SCHEMA, "claim_status": "development_only_not_confirmation",
        "dataset_sha256": sha256(args.dataset), "embeddings_sha256": sha256(args.embeddings),
        "embedding_provenance": provenance, "seed": args.seed,
        "projection": {"kind": "label_independent_gaussian_after_per_root_layernorm",
                       "seed": args.projection_seed, "widths": [8, 16, 32]},
        "model": {"kind": "one_hidden_layer_tanh_mlp", "hidden_widths": [4, 8],
                  "epochs": 120, "learning_rate": 0.01, "weight_decay": 0.01,
                  "maximum_parameters": 433},
        "grouping": "ten_outer_folds_with_five_fold_inner_battle_grouped_oof",
        "thresholding": "inner_oof_positive_score_maximize_durable_corrections_subject_to_zero_harm",
        "metrics": result, "selected": chosen, "predictions": predictions, "folds": folds,
        "compact_baseline_metrics": compact, "minimum_recovered": 17,
        "strictly_beats_compact": strictly_beats, "passed": passed}
    json_dump(args.output, report)
    print(json.dumps({"metrics": result, "compact": compact,
        "strictly_beats_compact": strictly_beats, "passed": passed,
        "fold_choices": [{k: fold[k] for k in ("fold", "projection_width", "hidden_width", "threshold", "test_metrics")} for fold in folds]}, indent=2))


if __name__ == "__main__":
    main()
