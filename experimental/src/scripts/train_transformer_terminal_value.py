#!/usr/bin/env python3
"""Fit a calibrated terminal-value head above the frozen accepted r1 encoder."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from train.transformer_terminal_value import (
    SOURCE_KINDS,
    TransformerValueHead,
    batch_indices,
    battle_balanced_bce,
    build_value_batch,
    frozen_r1_embeddings,
    load_terminal_value_dataset,
)


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


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def cache_split(
    agent,
    dataset,
    split: str,
    batch_size: int,
    max_seq_len: int,
    *,
    seed: int,
    source_mix: dict[str, float] | None = None,
):
    """Run the 142.8M encoder once; later epochs train only the small head."""
    device = next(agent.parameters()).device
    cached = []
    generator = torch.Generator().manual_seed(seed)
    for indices in batch_indices(
        dataset.indices(split),
        batch_size,
        generator=generator,
        shuffle=split == "train",
    ):
        obs, rl2, time_idxs, labels, weights = build_value_batch(
            dataset,
            indices,
            device,
            max_seq_len=max_seq_len,
            source_mix=source_mix,
        )
        embeddings = frozen_r1_embeddings(agent, obs, rl2, time_idxs)
        cached.append(
            (
                embeddings.to(device="cpu", dtype=torch.bfloat16),
                labels.cpu(),
                weights.cpu(),
            )
        )
    return cached


def cached_metrics(head, cached, device) -> dict[str, float]:
    head.eval()
    totals = {"weight": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    with torch.no_grad():
        for embeddings, labels, weights in cached:
            labels, weights = labels.to(device), weights.to(device)
            logits = head(embeddings.to(device=device, dtype=torch.float32))
            probabilities = logits.sigmoid()
            valid_weight = float(weights.sum())
            eps = torch.finfo(probabilities.dtype).eps
            totals["weight"] += valid_weight
            totals["brier"] += float(((probabilities - labels).square() * weights).sum())
            totals["log_loss"] += float(
                (-(labels * probabilities.clamp_min(eps).log() + (1-labels) * (1-probabilities).clamp_min(eps).log()) * weights).sum()
            )
            totals["accuracy"] += float((((probabilities >= 0.5) == labels.bool()).float() * weights).sum())
    return {key: totals[key] / totals["weight"] for key in ("brier", "log_loss", "accuracy")}


def weighted_base_rate(cached) -> float:
    numerator = denominator = 0.0
    for _, labels, weights in cached:
        numerator += float((labels * weights).sum())
        denominator += float(weights.sum())
    return numerator / denominator


def constant_metrics(cached, probability: float) -> dict[str, float]:
    logit = math.log(probability / (1.0 - probability))
    totals = {"weight": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    for _, labels, weights in cached:
        totals["weight"] += float(weights.sum())
        totals["brier"] += float(((probability - labels).square() * weights).sum())
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full_like(labels, logit), labels, reduction="none"
        )
        totals["log_loss"] += float((losses * weights).sum())
        totals["accuracy"] += float((((probability >= 0.5) == labels.bool()).float() * weights).sum())
    return {key: totals[key] / totals["weight"] for key in ("brier", "log_loss", "accuracy")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, default=Path("srcs/models"))
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--split-seed", type=int, default=20260813)
    args = parser.parse_args()
    if min(args.batch_size, args.max_seq_len, args.hidden_dim, args.epochs, args.patience) < 1:
        parser.error("positive dimensions, epochs, and patience are required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = args.base_root / args.base_run / "ckpts/policy_weights" / f"policy_epoch_{args.base_checkpoint}.pt"
    actual_sha = sha256(checkpoint)
    if actual_sha != args.base_sha256.lower():
        raise ValueError(f"base checkpoint hash mismatch: {actual_sha}")
    dataset_sha = sha256(args.dataset)
    dataset = load_terminal_value_dataset([args.dataset], split_seed=args.split_seed)
    print(json.dumps({"split_povs": dataset.split_counts, "split_battles": dataset.split_battles, "source_povs": dataset.source_counts, "source_battles": dataset.source_battles}, sort_keys=True), flush=True)

    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    device = next(agent.parameters()).device
    training_mix = {"human": 0.40, "selfplay": 0.40, "league": 0.20}
    cached = {
        split: cache_split(
            agent,
            dataset,
            split,
            args.batch_size,
            args.max_seq_len,
            seed=args.seed + (0 if split == "train" else 1 if split == "validation" else 2),
            source_mix=training_mix if split == "train" else None,
        )
        for split in ("train", "validation", "test")
    }
    embedding_dim = int(cached["train"][0][0].shape[-1])
    head = TransformerValueHead(embedding_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    base_rate = weighted_base_rate(cached["train"])
    baseline = {split: constant_metrics(cached[split], base_rate) for split in ("validation", "test")}
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None; best_brier = float("inf"); best_epoch = 0; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(cached["train"]), generator=generator).tolist()
        for index in order:
            embeddings, labels, weights = cached["train"][index]
            optimizer.zero_grad(set_to_none=True)
            logits = head(embeddings.to(device=device, dtype=torch.float32))
            loss = battle_balanced_bce(logits, labels.to(device), weights.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        validation = cached_metrics(head, cached["validation"], device)
        history.append({"epoch": epoch, "validation": validation})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if validation["brier"] < best_brier - 1e-5:
            best_brier = validation["brier"]; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no value-head checkpoint selected")
    head.load_state_dict(best_state)
    final = {split: cached_metrics(head, cached[split], device) for split in ("validation", "test")}
    payload = {
        "schema": "metagross-transformer-terminal-value/v1",
        "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
        "architecture": {"embedding_dim": embedding_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "parameters": sum(p.numel() for p in head.parameters())},
        "provenance": {"dataset_sha256": dataset_sha, "base_checkpoint_sha256": actual_sha, "base_run": args.base_run, "base_checkpoint": args.base_checkpoint},
        "selection": {"best_epoch": best_epoch, "training_source_mix": training_mix, "baseline": baseline, "final": final, "history": history},
    }
    atomic_torch_save(payload, args.output)
    print(json.dumps({"output": str(args.output), "best_epoch": best_epoch, "baseline": baseline, "final": final, "parameters": payload["architecture"]["parameters"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
