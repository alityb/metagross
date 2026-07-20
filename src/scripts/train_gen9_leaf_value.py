#!/usr/bin/env python3
"""Train a Rust-compatible Gen9 leaf-value model from Foul Play decision logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_examples(paths: list[Path]):
    import poke_engine

    decisions: list[dict] = []
    labels: dict[tuple[str, str], int] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(row.get("battle_tag")), str(row.get("username")))
            if row.get("record_type") == "battle_result" and row.get("label") in (0, 1):
                labels[key] = int(row["label"])
            elif row.get("record_type") == "decision" and isinstance(row.get("state"), str):
                decisions.append(row)

    features, targets, groups = [], [], []
    skipped = defaultdict(int)
    for row in decisions:
        key = (str(row.get("battle_tag")), str(row.get("username")))
        if key not in labels:
            skipped["missing_terminal_label"] += 1
            continue
        try:
            state = poke_engine.State.from_string(row["state"])
            feat = list(poke_engine.compute_value_features(state))
        except Exception:
            skipped["feature_extract_failed"] += 1
            continue
        if len(feat) != 14 or not np.isfinite(feat).all():
            skipped["invalid_features"] += 1
            continue
        features.append(feat)
        targets.append(labels[key])
        groups.append(key)
    if not features:
        raise ValueError("no labeled, feature-valid decision rows")
    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64), groups, dict(skipped)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def train(x, y, epochs, lr, l2):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        error = sigmoid(bias + x @ weights) - y
        bias -= lr * float(error.mean())
        weights -= lr * ((x.T @ error) / len(x) + l2 * weights)
    return bias, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    x, y, groups, skipped = load_examples(args.decision_log)
    unique_groups = sorted(set(groups))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    heldout_groups = set(unique_groups[:max(1, len(unique_groups) // 5)])
    heldout = np.asarray([group in heldout_groups for group in groups])
    bias, weights = train(x[~heldout], y[~heldout], args.epochs, args.lr, args.l2)
    probs = sigmoid(bias + x[heldout] @ weights)
    metrics = {
        "feature_count": int(x.shape[1]), "examples": int(len(x)),
        "train_examples": int((~heldout).sum()), "heldout_examples": int(heldout.sum()),
        "battles": len(unique_groups), "heldout_battles": len(heldout_groups),
        "heldout_brier": float(np.mean((probs - y[heldout]) ** 2)),
        "heldout_accuracy": float(np.mean((probs >= 0.5) == y[heldout])),
        "skipped": skipped,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(
        "metagross_value_net_v1\n"
        "# Gen9 14-feature decision-log value model\n"
        f"bias {bias:.9g}\n"
        "weights " + " ".join(f"{value:.9g}" for value in weights) + "\n"
    )
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
