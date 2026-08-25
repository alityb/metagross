#!/usr/bin/env python3
"""Train a linear counterfactual-Q residual over full causal R1 histories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.train_transformer_terminal_value import initialize_agent
from train.causal_transformer_action_q import (
    MODEL_SCHEMA,
    CausalResidualQHead,
    corrected_logits,
    load_dataset,
    ranking_metrics,
    residual_q_loss,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_embeddings(agent, dataset, batch_size: int):
    device = next(agent.parameters()).device
    if device.type != "cpu":
        raise RuntimeError(f"causal action-Q protocol requires local CPU, found {device}")
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_length[len(record["time_indices"])].append(index)
    embeddings = torch.empty((len(dataset.records), 900), dtype=torch.float32)
    causal_probs = torch.empty((len(dataset.records), 13), dtype=torch.float32)
    with torch.no_grad():
        for length in sorted(by_length):
            for indices in torch.tensor(by_length[length]).split(batch_size):
                selected = [dataset.records[int(index)] for index in indices]
                obs = {
                    "text_tokens": torch.tensor([row["text_tokens"] for row in selected], dtype=torch.int32, device=device),
                    "numbers": torch.nan_to_num(torch.tensor([row["numbers"] for row in selected], dtype=torch.float32, device=device)),
                    "illegal_actions": torch.tensor([row["illegal_actions"] for row in selected], dtype=torch.bool, device=device),
                }
                rl2 = torch.tensor([row["rl2"] for row in selected], dtype=torch.float32, device=device)
                times = torch.tensor([[[value] for value in row["time_indices"]] for row in selected], dtype=torch.int64, device=device)
                encoded, _ = agent.get_state_embedding(obs=obs, rl2s=rl2, time_idxs=times, hidden_state=None)
                distribution = agent.actor(
                    encoded,
                    straight_from_obs={key: obs[key][:, : encoded.shape[1]] for key in agent.pass_obs_keys_to_actor},
                )
                probs = distribution.probs[:, -1, -1, :]
                illegal = obs["illegal_actions"][:, -1]
                probs = probs * (~illegal)
                probs = probs / probs.sum(dim=1, keepdim=True)
                embeddings[indices] = encoded[:, -1].cpu()
                causal_probs[indices] = probs.cpu()
    return embeddings, causal_probs


def tensors(dataset):
    support = torch.tensor([row["teacher_support"] for row in dataset.records], dtype=torch.bool)
    q = torch.tensor([row["teacher_q"] for row in dataset.records], dtype=torch.float32)
    historical = torch.tensor([row["historical_selected_index"] for row in dataset.records], dtype=torch.int64)
    return support, q, historical


@torch.no_grad()
def metrics(head, embeddings, base_probs, support, q, historical, indices):
    correction = head(embeddings[indices])
    candidate = torch.softmax(corrected_logits(base_probs[indices], correction, support[indices]), dim=1)
    return {
        "candidate": ranking_metrics(candidate, q[indices], support[indices], historical[indices]),
        "causal_r1": ranking_metrics(base_probs[indices], q[indices], support[indices], historical[indices]),
    }


def bootstrap_improvement(candidate_regrets: list[float], baseline_regrets: list[float], seed: int, repeats: int = 10000):
    values = np.asarray(baseline_regrets) - np.asarray(candidate_regrets)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for index in range(repeats):
        sample = rng.integers(0, len(values), len(values))
        means[index] = values[sample].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint = args.base_root / args.base_run / "ckpts/policy_weights" / f"policy_epoch_{args.base_checkpoint}.pt"
    actual_hash = sha256(checkpoint)
    if actual_hash != args.base_sha256:
        raise ValueError(f"base checkpoint hash mismatch: {actual_hash}")
    dataset = load_dataset(args.dataset, args.split_seed)
    split_counts = {split: len(dataset.indices(split)) for split in ("train", "validation", "test")}
    if min(split_counts.values()) < 100:
        raise ValueError(f"causal action-Q split is too small: {split_counts}")
    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    agent.eval()
    embeddings, base_probs = cache_embeddings(agent, dataset, args.embedding_batch_size)
    support, q, historical = tensors(dataset)
    atomic_save({
        "schema": "metagross-causal-transformer-action-q-embeddings/v1",
        "embeddings": embeddings.to(torch.bfloat16),
        "causal_r1_probs": base_probs,
        "dataset_sha256": sha256(args.dataset),
        "base_checkpoint_sha256": actual_hash,
    }, args.embeddings_out)

    head = CausalResidualQHead()
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if parameter_count != 11_713:
        raise RuntimeError(f"causal residual architecture changed: {parameter_count}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            correction = head(embeddings[batch])
            loss = residual_q_loss(
                correction, base_probs[batch], q[batch], support[batch],
                teacher_temperature=args.teacher_temperature,
                anchor_weight=args.anchor_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        head.eval()
        validation = metrics(head, embeddings, base_probs, support, q, historical, dataset.indices("validation"))
        history.append({"epoch": epoch, "validation": validation})
        current = float(validation["candidate"]["mean_oracle_regret"])
        if current < best_regret - 1e-5:
            best_regret, best_epoch, best_state, stale = current, epoch, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("causal action-Q selected no checkpoint")
    head.load_state_dict(best_state); head.eval()
    final = {}
    for offset, split in enumerate(("train", "validation", "test")):
        split_result = metrics(head, embeddings, base_probs, support, q, historical, dataset.indices(split))
        split_result["improvement_ci95"] = bootstrap_improvement(
            split_result["candidate"]["regrets"], split_result["causal_r1"]["regrets"], args.seed + offset
        )
        final[split] = split_result
    test = final["test"]
    admitted = (
        test["improvement_ci95"][0] > 0
        and test["candidate"]["top1_agreement"] >= test["causal_r1"]["top1_agreement"]
        and test["candidate"]["mean_oracle_regret"] < test["causal_r1"]["mean_oracle_regret"]
    )
    payload = {
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
        "architecture": {"embedding_dim": 900, "actions": 13, "parameters": parameter_count, "kind": "linear_log_prior_residual"},
        "deployment": {"teacher_temperature": args.teacher_temperature, "causal_history_required": True},
        "provenance": {
            "dataset_sha256": sha256(args.dataset),
            "embeddings_sha256": sha256(args.embeddings_out),
            "base_checkpoint_sha256": actual_hash,
            "base_run": args.base_run,
            "base_checkpoint": args.base_checkpoint,
        },
        "selection": {"best_epoch": best_epoch, "split_seed": args.split_seed, "metrics": final, "admitted": admitted},
    }
    atomic_save(payload, args.model_out)
    report = {
        "schema": "metagross-causal-transformer-action-q-training/v1",
        "records": len(dataset.records),
        "split_counts": split_counts,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "admitted": admitted,
        "metrics": final,
        "history": history,
        "dataset_sha256": sha256(args.dataset),
        "embeddings_sha256": sha256(args.embeddings_out),
        "model_sha256": sha256(args.model_out),
        "base_checkpoint_sha256": actual_hash,
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
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--split-seed", type=int, default=20260816)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({key: report[key] for key in ("records", "split_counts", "parameter_count", "best_epoch", "admitted", "metrics", "model_sha256")}, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
