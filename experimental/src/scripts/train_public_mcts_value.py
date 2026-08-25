#!/usr/bin/env python3
"""Train a CPU-fast, public-information MCTS leaf value from terminal outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


FEATURE_CONTRACT = "metagross-public-information-value-features/v1"
FEATURE_COUNT = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(path: Path, row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(path.resolve()), str(row.get("battle_tag")), str(row.get("username")))


def _split(group: tuple[str, str, str], seed: int) -> str:
    material = json.dumps([seed, *group], separators=(",", ":")).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "test"


def _excluded_battle_tags(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                identity = row.get("identity")
                if isinstance(identity, dict) and identity.get("battle_tag"):
                    excluded.add(str(identity["battle_tag"]))
    return excluded


def load_examples(
    paths: list[Path], *, exclude_capture: list[Path], seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str, str]], dict[str, Any]]:
    import poke_engine

    excluded = _excluded_battle_tags(exclude_capture)
    labels: dict[tuple[str, str, str], int] = {}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("record_type") == "battle_result" and row.get("label") in (0, 1):
                    labels[_group(path, row)] = int(row["label"])

    features: list[list[float]] = []
    targets: list[float] = []
    groups: list[tuple[str, str, str]] = []
    skipped = Counter()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped["invalid_json"] += 1
                    continue
                if row.get("record_type") != "decision" or not isinstance(row.get("state"), str):
                    continue
                group = _group(path, row)
                if group[1] in excluded:
                    skipped["heldout_panel_battle"] += 1
                    continue
                if group not in labels:
                    skipped["missing_terminal_label"] += 1
                    continue
                try:
                    state = poke_engine.State.from_string(row["state"])
                    values = list(poke_engine.compute_public_value_features(state))
                except Exception:
                    skipped["feature_extract_failed"] += 1
                    continue
                if len(values) != FEATURE_COUNT or any(not math.isfinite(value) for value in values):
                    skipped["invalid_features"] += 1
                    continue
                features.append(values)
                targets.append(float(labels[group]))
                groups.append(group)
    if not features:
        raise ValueError("no labeled public-information examples")
    split_counts = Counter(_split(group, seed) for group in set(groups))
    if any(split_counts[name] == 0 for name in ("train", "validation", "test")):
        raise ValueError("battle-disjoint split produced an empty partition")
    provenance = {
        "feature_contract": FEATURE_CONTRACT,
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "excluded_capture_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in exclude_capture
        ],
        "excluded_battle_tags": len(excluded),
        "skipped": dict(skipped),
    }
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        groups,
        provenance,
    )


def _partition(groups: list[tuple[str, str, str]], seed: int, name: str) -> torch.Tensor:
    return torch.tensor([_split(group, seed) == name for group in groups], dtype=torch.bool)


def _battle_weights(groups: list[tuple[str, str, str]]) -> torch.Tensor:
    counts = Counter(groups)
    return torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.float32)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    return float((values * weights).sum() / weights.sum())


def _metrics(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    brier = _weighted_mean((probabilities - labels).square(), weights)
    accuracy = _weighted_mean(((probabilities >= 0.5) == labels.bool()).float(), weights)
    base_rate = _weighted_mean(labels, weights)
    reference_brier = _weighted_mean((labels - base_rate).square(), weights)
    ece = 0.0
    total = float(weights.sum())
    for lower in torch.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        member = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if member.any():
            bin_weights = weights[member]
            confidence = _weighted_mean(probabilities[member], bin_weights)
            outcome = _weighted_mean(labels[member], bin_weights)
            ece += float(bin_weights.sum()) / total * abs(confidence - outcome)
    return {
        "brier": brier,
        "accuracy": accuracy,
        "ece_10": ece,
        "base_rate": base_rate,
        "constant_reference_brier": reference_brier,
    }


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> float:
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_temperature.exp(), labels, reduction="none"
        )
        weighted = (loss * weights).sum() / weights.sum()
        weighted.backward()
        return weighted

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.1, 10.0))


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    x, y, groups, provenance = load_examples(
        args.decision_log, exclude_capture=args.exclude_capture, seed=args.seed
    )
    masks = {
        name: _partition(groups, args.seed, name)
        for name in ("train", "validation", "test")
    }
    weights = _battle_weights(groups)
    train_x = x[masks["train"]]
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-4)
    normalized = (x - mean) / std

    model = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, args.hidden1),
        torch.nn.Tanh(),
        torch.nn.Linear(args.hidden1, args.hidden2),
        torch.nn.Tanh(),
        torch.nn.Linear(args.hidden2, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_indices = torch.where(masks["train"])[0]
    generator = torch.Generator().manual_seed(args.seed)
    best_brier = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        model.train()
        for batch in order.split(args.batch_size):
            logits = model(normalized[batch]).squeeze(-1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[batch], reduction="none"
            )
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mask = masks["validation"]
            validation = _metrics(
                model(normalized[mask]).squeeze(-1), y[mask], weights[mask]
            )
        if validation["brier"] < best_brier - 1e-6:
            best_brier = validation["brier"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_mask = masks["validation"]
        validation_logits = model(normalized[validation_mask]).squeeze(-1)
    temperature = _fit_temperature(
        validation_logits, y[validation_mask], weights[validation_mask]
    )

    results: dict[str, Any] = {}
    unique_groups = set(groups)
    for name, mask in masks.items():
        with torch.no_grad():
            logits = model(normalized[mask]).squeeze(-1) / temperature
        results[name] = {
            "examples": int(mask.sum()),
            "battles": sum(_split(group, args.seed) == name for group in unique_groups),
            **_metrics(logits, y[mask], weights[mask]),
        }

    first, second, third = model[0], model[2], model[4]
    w1 = (first.weight.detach() / std.unsqueeze(0)).T.contiguous()
    b1 = first.bias.detach() - (
        first.weight.detach() * mean.unsqueeze(0) / std.unsqueeze(0)
    ).sum(1)
    w2 = second.weight.detach().T.contiguous()
    w3 = third.weight.detach().squeeze(0) / temperature
    b3 = third.bias.detach().item() / temperature
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(
        "metagross_public_value_mlp_v1\n"
        f"# {FEATURE_CONTRACT}; terminal win/loss labels; battle-disjoint split\n"
        f"dims {FEATURE_COUNT} {args.hidden1} {args.hidden2} 1\n"
        + "w1 " + " ".join(f"{value:.9g}" for value in w1.flatten().tolist()) + "\n"
        + "b1 " + " ".join(f"{value:.9g}" for value in b1.tolist()) + "\n"
        + "w2 " + " ".join(f"{value:.9g}" for value in w2.flatten().tolist()) + "\n"
        + "b2 " + " ".join(f"{value:.9g}" for value in second.bias.detach().tolist()) + "\n"
        + "w3 " + " ".join(f"{value:.9g}" for value in w3.tolist()) + "\n"
        + f"b3 {b3:.9g}\n",
        encoding="ascii",
    )
    report = {
        "schema": "metagross-public-mcts-value-training/v1",
        "feature_contract": FEATURE_CONTRACT,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "temperature": temperature,
        "split_method": "sha256_battle_group_70_15_15",
        "battle_weighted_loss": True,
        "metrics": results,
        "provenance": provenance,
        "model_sha256": _sha256(args.model_out),
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(payload.encode("ascii")).hexdigest()
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--exclude-capture", type=Path, action="append", default=[])
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
