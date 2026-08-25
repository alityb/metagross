#!/usr/bin/env python3
"""Probe eventual outcomes from exact first-decision frozen-r1 observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ProbeError(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    key: str
    worker: str
    shard: str
    battle_tag: str
    username: str
    label: int
    text_tokens: list[int]
    numbers: list[float]
    illegal_actions: list[bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_first_decisions(targets_root: Path) -> list[Example]:
    paths = sorted(targets_root.glob("w*/shard_*.jsonl"))
    if not paths:
        raise ProbeError(f"no target shards under {targets_root}")
    examples: dict[str, Example] = {}
    labels: dict[str, int] = {}
    shapes: tuple[int, int] | None = None
    for path in paths:
        worker, shard = path.parent.name, path.stem
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ProbeError(f"{path}:{line_number}: blank row")
            row = json.loads(line)
            if row.get("schema") != 3:
                raise ProbeError(f"{path}:{line_number}: expected schema 3")
            tag, username, decision, label = (
                row.get("battle_tag"), row.get("username"), row.get("decision_idx"), row.get("label")
            )
            if not isinstance(tag, str) or not isinstance(username, str) or label not in {0, 1}:
                raise ProbeError(f"{path}:{line_number}: invalid identity or label")
            key = f"{worker}|{shard}|{tag}|{username}"
            if key in labels and labels[key] != label:
                raise ProbeError(f"{path}:{line_number}: conflicting battle label")
            labels[key] = label
            if decision != 0:
                continue
            text, numbers, illegal = row.get("text_tokens"), row.get("numbers"), row.get("illegal_actions")
            if (
                not isinstance(text, list) or not all(isinstance(value, int) for value in text)
                or not isinstance(numbers, list) or not all(isinstance(value, (int, float)) for value in numbers)
                or not isinstance(illegal, list) or len(illegal) != 13
            ):
                raise ProbeError(f"{path}:{line_number}: invalid observation")
            if any(not math.isfinite(float(value)) for value in numbers):
                raise ProbeError(f"{path}:{line_number}: nonfinite numeric observation")
            current_shape = (len(text), len(numbers))
            if shapes is None:
                shapes = current_shape
            elif current_shape != shapes:
                raise ProbeError(f"{path}:{line_number}: inconsistent observation shape")
            if key in examples:
                raise ProbeError(f"{path}:{line_number}: duplicate first decision")
            examples[key] = Example(
                key, worker, shard, tag, username, label, text,
                [float(value) for value in numbers], [bool(value) for value in illegal],
            )
    missing = sorted(set(labels) - set(examples))
    if missing:
        raise ProbeError(f"{len(missing)} battles lack decision_idx 0")
    return [examples[key] for key in sorted(examples)]


def split_for_key(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10000
    return "train" if bucket < 6000 else "development" if bucket < 8000 else "test"


def reliability(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (probabilities >= lower) & ((probabilities < upper) if index < 9 else (probabilities <= upper))
        count = int(mask.sum())
        mean_probability = float(probabilities[mask].mean()) if count else None
        win_rate = float(labels[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(mean_probability - win_rate)
        rows.append({"lower": lower, "upper": upper, "count": count, "mean_probability": mean_probability, "win_rate": win_rate})
    return ece, rows


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1 - 1e-9)
    labels = np.asarray(labels, dtype=np.float64)
    ece, reliability_rows = reliability(probabilities, labels)
    order = np.argsort(probabilities)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    auc = (
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        if positives and negatives else None
    )
    return {
        "rows": len(labels),
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(np.mean(-(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels.astype(bool))),
        "auroc": float(auc) if auc is not None else None,
        "ece": float(ece),
        "reliability": reliability_rows,
    }


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train = torch.tensor((train_x - mean) / std, dtype=torch.float32)
    dev = torch.tensor((dev_x - mean) / std, dtype=torch.float32)
    test = torch.tensor((test_x - mean) / std, dtype=torch.float32)
    targets = torch.tensor(train_y, dtype=torch.float32)
    best = None
    trials = []
    for weight_decay in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        layer = torch.nn.Linear(train.shape[1], 1)
        optimizer = torch.optim.LBFGS(layer.parameters(), lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = layer(train).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss = loss + weight_decay * layer.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            dev_probabilities = torch.sigmoid(layer(dev).squeeze(1)).numpy()
            dev_brier = float(np.mean((dev_probabilities - dev_y) ** 2))
        trials.append({"weight_decay": weight_decay, "development_brier": dev_brier})
        if best is None or dev_brier < best[0]:
            best = (dev_brier, weight_decay, layer.state_dict())
    assert best is not None
    chosen = torch.nn.Linear(train.shape[1], 1)
    chosen.load_state_dict(best[2])
    with torch.no_grad():
        probabilities = torch.sigmoid(chosen(test).squeeze(1)).numpy()
    return probabilities, {"selected_weight_decay": best[1], "development_trials": trials}


def extract_embeddings(agent: Any, examples: list[Example], device: Any, batch_size: int) -> np.ndarray:
    import torch

    embeddings = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        text_now = np.asarray([example.text_tokens for example in batch], dtype=np.int32)
        numbers_now = np.asarray([example.numbers for example in batch], dtype=np.float32)
        illegal_now = np.asarray([example.illegal_actions for example in batch], dtype=bool)
        obs = {
            "text_tokens": torch.tensor(np.stack([np.zeros_like(text_now), text_now], axis=1), device=device),
            "numbers": torch.tensor(np.stack([np.zeros_like(numbers_now), numbers_now], axis=1), device=device),
            "illegal_actions": torch.tensor(np.stack([np.ones_like(illegal_now), illegal_now], axis=1), device=device),
        }
        rl2s = torch.zeros((len(batch), 2, 14), device=device)
        time_idxs = torch.arange(2, device=device).long().view(1, 2, 1).expand(len(batch), 2, 1)
        with torch.no_grad():
            embedding, _ = agent.get_state_embedding(obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None)
        embeddings.append(embedding[:, -1, :].detach().cpu().numpy())
    return np.concatenate(embeddings)


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-name", default="randbats_exit_r1")
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bootstrap_repeats <= 0:
        raise ProbeError("batch size and bootstrap repeats must be positive")
    checkpoint = args.checkpoint_root / args.run_name / "ckpts/policy_weights" / f"policy_epoch_{args.checkpoint}.pt"
    if _sha256(checkpoint) != args.checkpoint_sha256.lower():
        raise ProbeError("checkpoint SHA-256 does not match")

    examples = load_first_decisions(args.targets_root)
    splits = {name: [example for example in examples if split_for_key(example.key) == name] for name in ("train", "development", "test")}
    if any(not split for split in splits.values()):
        raise ProbeError("one or more deterministic splits are empty")
    split_rows = [{"key": example.key, "split": split_for_key(example.key), "label": example.label} for example in examples]
    _atomic_json(args.split_manifest, split_rows)

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(args.checkpoint_root),
        model_name=args.run_name,
        default_checkpoint=args.checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device

    ordered = splits["train"] + splits["development"] + splits["test"]
    embeddings = extract_embeddings(agent, ordered, device, args.batch_size)
    numbers = np.asarray([example.numbers for example in ordered], dtype=np.float32)
    labels = np.asarray([example.label for example in ordered], dtype=np.float32)
    train_end = len(splits["train"])
    dev_end = train_end + len(splits["development"])
    slices = (slice(0, train_end), slice(train_end, dev_end), slice(dev_end, None))
    train_slice, dev_slice, test_slice = slices

    constant_probability = float(labels[train_slice].mean())
    constant = np.full(len(labels[test_slice]), constant_probability)
    numbers_probs, numbers_fit = fit_linear_probe(
        numbers[train_slice], labels[train_slice], numbers[dev_slice], labels[dev_slice], numbers[test_slice], seed=args.seed
    )
    embedding_probs, embedding_fit = fit_linear_probe(
        embeddings[train_slice], labels[train_slice], embeddings[dev_slice], labels[dev_slice], embeddings[test_slice], seed=args.seed
    )
    test_labels = labels[test_slice]
    metrics = {
        "constant": binary_metrics(constant, test_labels),
        "numbers_linear": binary_metrics(numbers_probs, test_labels),
        "frozen_r1_embedding_linear": binary_metrics(embedding_probs, test_labels),
    }
    rng = np.random.default_rng(args.seed)
    numbers_deltas, embedding_deltas = [], []
    for _ in range(args.bootstrap_repeats):
        indices = rng.integers(0, len(test_labels), len(test_labels))
        baseline_loss = np.mean((constant[indices] - test_labels[indices]) ** 2)
        numbers_deltas.append(np.mean((numbers_probs[indices] - test_labels[indices]) ** 2) - baseline_loss)
        embedding_deltas.append(np.mean((embedding_probs[indices] - test_labels[indices]) ** 2) - baseline_loss)
    report = {
        "schema_version": 1,
        "claim_status": "observational_policy_conditional_first_decision_probe",
        "targets_root": str(args.targets_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "counts": {name: len(split) for name, split in splits.items()},
        "metrics": metrics,
        "fit": {"numbers_linear": numbers_fit, "frozen_r1_embedding_linear": embedding_fit},
        "paired_brier_delta_vs_constant_ci95": {
            "numbers_linear": percentile_interval(np.asarray(numbers_deltas)),
            "frozen_r1_embedding_linear": percentile_interval(np.asarray(embedding_deltas)),
        },
        "limitations": [
            "Outcome prediction is observational and conditional on accepted-r1 self-play behavior.",
            "This is not an action value or a counterfactual continuation estimate.",
            "Only the first retained decision per physical battle is evaluated.",
        ],
    }
    _atomic_json(args.output_json, report)
    print(json.dumps({"counts": report["counts"], "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
