#!/usr/bin/env python3
"""Cache frozen R1 embeddings and train the preregistered 13-action Q head."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.train_transformer_terminal_value import initialize_agent
from train.transformer_action_q import (
    MODEL_SCHEMA,
    TransformerActionQHead,
    action_q_loss,
    action_q_metrics,
    build_stateless_batch,
    frozen_current_embeddings,
    load_dataset,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_embeddings(agent, dataset, batch_size: int) -> torch.Tensor:
    device = next(agent.parameters()).device
    if device.type != "cpu":
        raise RuntimeError(f"local transformer-Q protocol requires CPU, found {device}")
    cached = []
    for start in range(0, len(dataset.battle_ids), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(dataset.battle_ids)))
        obs, rl2, time_idxs = build_stateless_batch(dataset, indices, device)
        cached.append(frozen_current_embeddings(agent, obs, rl2, time_idxs).cpu())
    return torch.cat(cached)


@torch.no_grad()
def split_metrics(head, embeddings, dataset, split: str):
    indices = dataset.indices(split)
    prediction = head(embeddings[indices])
    return action_q_metrics(
        prediction,
        dataset.teacher_q[indices],
        dataset.teacher_support[indices],
        dataset.historical_selected_index[indices],
    )


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = args.base_root / args.base_run / "ckpts/policy_weights" / f"policy_epoch_{args.base_checkpoint}.pt"
    actual_base_hash = sha256(checkpoint)
    if actual_base_hash != args.base_sha256:
        raise ValueError(f"base checkpoint hash mismatch: {actual_base_hash}")
    dataset = load_dataset(args.dataset, args.split_seed)
    split_counts = {split: len(dataset.indices(split)) for split in ("train", "validation", "test")}
    if min(split_counts.values()) < 50:
        raise ValueError(f"transformer-Q split is too small: {split_counts}")
    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    embeddings = cache_embeddings(agent, dataset, args.embedding_batch_size)
    if embeddings.shape != (len(dataset.battle_ids), 900):
        raise RuntimeError(f"unexpected frozen R1 embedding shape: {tuple(embeddings.shape)}")
    embedding_payload = {
        "schema": "metagross-transformer-action-q-embeddings/v1",
        "embeddings": embeddings.to(torch.bfloat16),
        "dataset_sha256": sha256(args.dataset),
        "base_checkpoint_sha256": actual_base_hash,
    }
    atomic_save(embedding_payload, args.embeddings_out)

    head = TransformerActionQHead(900, 256)
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if parameter_count != 235_797:
        raise RuntimeError(f"transformer action-Q architecture changed: {parameter_count}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices = dataset.indices("train")
    best_state = None
    best_epoch = 0
    best_regret = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for batch in order.split(args.batch_size):
            prediction = head(embeddings[batch])
            loss = action_q_loss(
                prediction,
                dataset.teacher_q[batch],
                dataset.teacher_support[batch],
                temperature=args.teacher_temperature,
                listwise_weight=args.listwise_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        head.eval()
        validation = split_metrics(head, embeddings, dataset, "validation")
        history.append({"epoch": epoch, "validation": validation})
        current = float(validation["mean_oracle_regret"])
        if current < best_regret - 1e-5:
            best_regret = current
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("transformer-Q training selected no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    final = {
        split: split_metrics(head, embeddings, dataset, split)
        for split in ("train", "validation", "test")
    }
    validation = final["validation"]
    admitted = (
        validation["top1_agreement"] >= 0.50
        and validation["mean_oracle_regret"] <= 0.04
        and validation["top1_agreement"] >= validation["historical_top1_agreement"]
        and validation["mean_oracle_regret"] <= validation["historical_mean_oracle_regret"]
    )
    payload = {
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
        "architecture": {"embedding_dim": 900, "hidden_dim": 256, "actions": 13, "parameters": parameter_count},
        "deployment": {"stateless_two_step": True, "prior_temperature": 0.05, "c_puct": 2.0},
        "provenance": {
            "dataset_sha256": sha256(args.dataset),
            "embeddings_sha256": sha256(args.embeddings_out),
            "base_checkpoint_sha256": actual_base_hash,
            "base_run": args.base_run,
            "base_checkpoint": args.base_checkpoint,
        },
        "selection": {"best_epoch": best_epoch, "split_seed": args.split_seed, "metrics": final, "admitted": admitted},
    }
    atomic_save(payload, args.model_out)
    report = {
        "schema": "metagross-transformer-action-q-training/v1",
        "records": len(dataset.battle_ids),
        "split_counts": split_counts,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "admitted": admitted,
        "metrics": final,
        "history": history,
        "dataset_sha256": sha256(args.dataset),
        "embeddings_sha256": sha256(args.embeddings_out),
        "model_sha256": sha256(args.model_out),
        "base_checkpoint_sha256": actual_base_hash,
    }
    args.metrics_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, default=Path("srcs/models"))
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--embeddings-out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--listwise-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--split-seed", type=int, default=20260815)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({key: report[key] for key in ("records", "split_counts", "parameter_count", "best_epoch", "admitted", "metrics", "model_sha256")}, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
