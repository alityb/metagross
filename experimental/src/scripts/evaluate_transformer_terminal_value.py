#!/usr/bin/env python3
"""Held-out calibration audit for the frozen-r1 terminal value head."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from inference.transformer_value_oracle import TransformerValueOracle
from train.transformer_terminal_value import batch_indices, build_value_batch, load_terminal_value_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_agent(base_root: Path, run_name: str, checkpoint: int):
    import gin
    from amago.nets.transformer import VanillaAttention
    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except ValueError:
        pass
    import metamon.rl.pretrained as pretrained
    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(base_root),
        model_name=run_name,
        default_checkpoint=checkpoint,
    )
    return model.initialize_agent(checkpoint=checkpoint, log=False).policy


class Accumulator:
    def __init__(self):
        self.weight = self.brier = self.log_loss = self.correct = 0.0
        self.bins = [[0.0, 0.0, 0.0] for _ in range(10)]

    def add(self, probability: float, label: float, weight: float) -> None:
        probability = min(max(probability, 1e-7), 1 - 1e-7)
        self.weight += weight
        self.brier += weight * (probability - label) ** 2
        self.log_loss += -weight * (label * math.log(probability) + (1-label) * math.log(1-probability))
        self.correct += weight * ((probability >= 0.5) == bool(label))
        index = min(9, int(probability * 10))
        self.bins[index][0] += weight
        self.bins[index][1] += weight * probability
        self.bins[index][2] += weight * label

    def report(self) -> dict[str, float]:
        if self.weight <= 0:
            return {"weight": 0.0}
        ece = sum(
            abs(probability_sum / weight - label_sum / weight) * weight / self.weight
            for weight, probability_sum, label_sum in self.bins if weight
        )
        return {
            "weight": self.weight,
            "brier": self.brier / self.weight,
            "log_loss": self.log_loss / self.weight,
            "accuracy": self.correct / self.weight,
            "ece_10": ece,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-sha256", required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=20260813)
    args = parser.parse_args()

    dataset = load_terminal_value_dataset([args.dataset], split_seed=args.split_seed)
    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    oracle = TransformerValueOracle(
        agent,
        args.head,
        expected_base_sha256=args.base_sha256,
        expected_head_sha256=args.head_sha256,
    )
    device = oracle.device
    groups: defaultdict[str, Accumulator] = defaultdict(Accumulator)
    test_indices = dataset.indices("test")
    for indices in batch_indices(test_indices, args.batch_size):
        obs, rl2, time_idxs, labels, weights = build_value_batch(
            dataset, indices, device, max_seq_len=args.max_seq_len
        )
        with torch.no_grad():
            embeddings = oracle.agent.get_state_embedding(
                obs=obs, rl2s=rl2, time_idxs=time_idxs, hidden_state=None
            )[0]
            probabilities = oracle.head(embeddings).sigmoid().cpu()
        labels, weights = labels.cpu(), weights.cpu()
        for batch_index, raw_index in enumerate(indices.tolist()):
            row = dataset.trajectories[raw_index]
            size = min(row.length, args.max_seq_len)
            start = row.length - size
            for step in range(size):
                probability = float(probabilities[batch_index, step])
                label, weight = float(labels[batch_index, step]), float(weights[batch_index, step])
                progress = (start + step) / max(1, row.length - 1)
                progress_bin = "early_0_25" if progress < .25 else "mid_25_75" if progress < .75 else "late_75_100"
                for group in ("overall", f"source/{row.source_kind}", f"progress/{progress_bin}"):
                    groups[group].add(probability, label, weight)
    report = {
        "schema": "metagross-transformer-terminal-value-audit/v1",
        "dataset_sha256": sha256(args.dataset),
        "head_sha256": args.head_sha256,
        "base_checkpoint_sha256": args.base_sha256,
        "test_povs": len(test_indices),
        "test_battles": dataset.split_battles["test"],
        "groups": {key: groups[key].report() for key in sorted(groups)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
