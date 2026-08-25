#!/usr/bin/env python3
"""Fit non-negative resource shadow prices from terminal battle outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from train.resource_shadow import (  # noqa: E402
    FEATURE_COUNT,
    FEATURE_NAMES,
    RESOURCE_FEATURE_COUNT,
    SCHEMA,
    extract_resource_features,
)
from belief.public_reveal_mask import (  # noqa: E402
    from_replay_facts,
    information_fractions,
    replay_reveal_snapshots,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(path: Path, row: dict[str, Any]) -> tuple[str, str, str]:
    return str(path.resolve()), str(row.get("battle_tag")), str(row.get("username"))


def _split(group: tuple[str, str, str], seed: int) -> str:
    material = json.dumps([seed, *group], separators=(",", ":")).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "test"


def _replay_index(path: Path) -> dict[str, Path]:
    replay_dir = path.parent / "replays"
    if not replay_dir.is_dir():
        return {}
    index: dict[str, Path] = {}
    for replay_path in sorted(replay_dir.glob("*.json")):
        try:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        battle_id = payload.get("id")
        if isinstance(battle_id, str) and battle_id and battle_id not in index:
            index[battle_id] = replay_path
    return index


def _public_snapshots(
    replay_path: Path,
    observer_name: str,
) -> dict[int, Any]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    return replay_reveal_snapshots(payload.get("log", ""), observer_name)


def _load(
    paths: list[Path],
    seed: int,
    *,
    public_reveals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str, str]], dict[str, Any]]:
    import poke_engine

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
    skipped: Counter[str] = Counter()
    replay_indices = {path: _replay_index(path) for path in paths} if public_reveals else {}
    snapshot_cache: dict[tuple[Path, str, str], dict[int, Any]] = {}
    decision_turn_counts: Counter[tuple[tuple[str, str, str], int]] = Counter()
    public_fraction_sums = [0.0] * 4
    public_nonzero_rows = 0
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
                if group not in labels:
                    skipped["missing_terminal_label"] += 1
                    continue
                try:
                    state = poke_engine.State.from_string(row["state"])
                    turn = int(row.get("turn", 0))
                    if public_reveals:
                        battle_tag = str(row.get("battle_tag"))
                        observer_name = str(row.get("username"))
                        replay_path = replay_indices[path].get(battle_tag)
                        if replay_path is None:
                            skipped["missing_replay"] += 1
                            continue
                        cache_key = (path, battle_tag, observer_name)
                        if cache_key not in snapshot_cache:
                            snapshot_cache[cache_key] = _public_snapshots(
                                replay_path, observer_name
                            )
                        facts = snapshot_cache[cache_key].get(turn)
                        if facts is None:
                            skipped["missing_turn_snapshot"] += 1
                            continue
                        bits = from_replay_facts(state, facts)
                        state = state.with_side_one_public_reveals(bits)
                        fractions = information_fractions(bits)
                        for index, value in enumerate(fractions):
                            public_fraction_sums[index] += value
                        public_nonzero_rows += int(bits != 0)
                        decision_turn_counts[(group, turn)] += 1
                    vector = extract_resource_features(
                        state,
                        turn=turn,
                        include_public_information=public_reveals,
                    )
                except (TypeError, ValueError, OverflowError):
                    skipped["feature_extract_failed"] += 1
                    continue
                features.append(vector)
                targets.append(float(labels[group]))
                groups.append(group)
    if not features:
        raise ValueError("no labeled resource-shadow examples")
    split_counts = Counter(_split(group, seed) for group in set(groups))
    if any(split_counts[name] == 0 for name in ("train", "validation", "test")):
        raise ValueError("battle-disjoint resource split is empty")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        groups,
        {
            "sources": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
            "skipped": dict(skipped),
            "public_reveals": {
                "enabled": public_reveals,
                "alignment": "conservative_start_of_turn",
                "matched_replays": len(snapshot_cache),
                "rows_with_nonzero_mask": public_nonzero_rows,
                "mean_fractions": [
                    value / len(features) for value in public_fraction_sums
                ] if public_reveals else [0.0] * 4,
                "same_battle_turn_extra_decisions": sum(
                    max(0, count - 1) for count in decision_turn_counts.values()
                ),
            },
        },
    )


class ResourceShadowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.resource_raw = torch.nn.Parameter(torch.full((RESOURCE_FEATURE_COUNT,), -2.0))
        self.context = torch.nn.Parameter(torch.zeros(FEATURE_COUNT - RESOURCE_FEATURE_COUNT))

    def coefficients(self) -> torch.Tensor:
        return torch.cat((torch.nn.functional.softplus(self.resource_raw), self.context))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bias + features @ self.coefficients()


def _weights(groups: list[tuple[str, str, str]]) -> torch.Tensor:
    counts = Counter(groups)
    return torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.float32)


def _metrics(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    total = weights.sum()
    mean = lambda values: float((values * weights).sum() / total)
    base_rate = mean(labels)
    return {
        "accuracy": mean(((probabilities >= 0.5) == labels.bool()).float()),
        "brier": mean((probabilities - labels).square()),
        "base_rate": base_rate,
        "constant_reference_brier": mean((labels - base_rate).square()),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    x, y, groups, provenance = _load(
        args.decision_log,
        args.seed,
        public_reveals=args.public_reveals,
    )
    masks = {
        name: torch.tensor([_split(group, args.seed) == name for group in groups])
        for name in ("train", "validation", "test")
    }
    weights = _weights(groups)
    model = ResourceShadowModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_indices = torch.where(masks["train"])[0]
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None
    best_brier = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for batch in order.split(args.batch_size):
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x[batch]), y[batch], reduction="none"
            )
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mask = masks["validation"]
            brier = _metrics(model(x[mask]), y[mask], weights[mask])["brier"]
        if brier < best_brier - 1e-7:
            best_brier = brier
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("resource-shadow training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    unique_groups = set(groups)
    metrics = {}
    with torch.no_grad():
        for name, mask in masks.items():
            metrics[name] = {
                "examples": int(mask.sum()),
                "battles": sum(_split(group, args.seed) == name for group in unique_groups),
                **_metrics(model(x[mask]), y[mask], weights[mask]),
            }
        coefficients = model.coefficients().tolist()
    if any(value < 0 for value in coefficients[:RESOURCE_FEATURE_COUNT]):
        raise RuntimeError("trained resource price violated non-negative constraint")
    artifact = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "resource_feature_count": RESOURCE_FEATURE_COUNT,
        "coefficients": coefficients,
        "bias": float(model.bias.detach()),
        "training": {
            "seed": args.seed,
            "split": "sha256_battle_group_70_15_15",
            "battle_weighted_loss": True,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "provenance": provenance,
        },
        "claim_limit": "associational shadow prices; action utility requires an independent root/outcome gate",
        "public_reveals_enabled": args.public_reveals,
    }
    if args.engine_model_out is not None:
        args.engine_model_out.parent.mkdir(parents=True, exist_ok=True)
        args.engine_model_out.write_text(
            (
                "metagross_resource_shadow_v2\n"
                if args.public_reveals
                else "metagross_resource_shadow_v1\n"
            )
            +
            f"dims {FEATURE_COUNT}\n"
            f"bias {float(model.bias.detach()):.9g}\n"
            + "weights "
            + " ".join(f"{value:.9g}" for value in coefficients)
            + "\n",
            encoding="ascii",
        )
        artifact["engine_model"] = {
            "path": str(args.engine_model_out),
            "sha256": _sha256(args.engine_model_out),
            "leaf_semantics": "root-centered resource-logit delta blended with hand evaluation",
        }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    artifact["artifact_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-model-out", type=Path)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--public-reveals",
        action="store_true",
        help="join each decision to its replay and train on causal reveal masks",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), sort_keys=True))


if __name__ == "__main__":
    main()
