#!/usr/bin/env python3
"""Train a Rust-compatible Gen9 14-feature MLP leaf evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_gen9_leaf_value import load_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x, y, groups, skipped = load_examples(args.decision_log)
    unique = sorted(set(groups))
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(unique), generator=generator).tolist()
    heldout_groups = {unique[i] for i in perm[:max(1, len(unique) // 5)]}
    heldout = torch.tensor([group in heldout_groups for group in groups], dtype=torch.bool)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # Features are already bounded differentials, but standardizing on train
    # data prevents raw stat-total terms dominating the MLP.
    mean, std = x[~heldout].mean(0), x[~heldout].std(0).clamp_min(1e-4)
    x = (x - mean) / std

    net = torch.nn.Sequential(torch.nn.Linear(14, 64), torch.nn.Tanh(), torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    train_idx = torch.where(~heldout)[0]
    for _ in range(args.epochs):
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for batch in order.split(args.batch_size):
            logits = net(x[batch]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(net(x[heldout]).squeeze(-1))
        labels = y[heldout]
    metrics = {"examples": int(len(x)), "battles": len(unique), "heldout_brier": float(((probs - labels) ** 2).mean()), "heldout_accuracy": float(((probs >= .5) == labels.bool()).float().mean()), "skipped": skipped}

    # Rust's MLP receives the unnormalized 14 features. Fold normalization into
    # the first layer so inference exactly matches training.
    first, second, third = net[0], net[2], net[4]
    w1 = (first.weight.detach() / std.unsqueeze(0)).T.contiguous()
    b1 = first.bias.detach() - (first.weight.detach() * mean.unsqueeze(0) / std.unsqueeze(0)).sum(1)
    w2 = second.weight.detach().T.contiguous()
    w3 = third.weight.detach().squeeze(0)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(
        "metagross_value_mlp_v1\n# Gen9 learned leaf evaluator\n"
        "dims 14 64 32 1\n"
        + "w1 " + " ".join(f"{v:.9g}" for v in w1.flatten().tolist()) + "\n"
        + "b1 " + " ".join(f"{v:.9g}" for v in b1.tolist()) + "\n"
        + "w2 " + " ".join(f"{v:.9g}" for v in w2.flatten().tolist()) + "\n"
        + "b2 " + " ".join(f"{v:.9g}" for v in second.bias.detach().tolist()) + "\n"
        + "w3 " + " ".join(f"{v:.9g}" for v in w3.tolist()) + "\n"
        + f"b3 {third.bias.detach().item():.9g}\n"
    )
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
