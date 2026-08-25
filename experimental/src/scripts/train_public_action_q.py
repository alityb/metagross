#!/usr/bin/env python3
"""Train a tiny public action-Q model from frozen deep-search root values."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = EXPERIMENTAL_ROOT.parent
sys.path[:0] = [str(EXPERIMENTAL_ROOT / "src")]

from scripts.run_public_mcts_leaf_gate import (  # noqa: E402
    ORACLE_SCHEMA,
    PANEL_SCHEMA,
    _sha256,
)
from train.public_action_q import (  # noqa: E402
    PublicActionQ,
    action_features,
    load_move_database,
    save_model,
)


@dataclass
class RootExample:
    battle_id: str
    root_id: str
    actions: list[str]
    features: np.ndarray
    values: np.ndarray

    @property
    def advantages(self) -> np.ndarray:
        return self.values - self.values.mean()


def _split(battle_id: str, seed: int) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}\0{battle_id}".encode()).digest()[:8], "big"
    ) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def load_examples(panel_path: Path, oracle_path: Path, move_database_path: Path) -> tuple[list[RootExample], dict[str, Any]]:
    import poke_engine

    panel_hash = _sha256(panel_path)
    panel = _read_jsonl(panel_path)
    oracle = _read_jsonl(oracle_path)
    if any(row.get("schema") != PANEL_SCHEMA for row in panel):
        raise ValueError("invalid action-Q training panel")
    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in oracle:
        if row.get("schema") != ORACLE_SCHEMA or row.get("panel_sha256") != panel_hash:
            raise ValueError("action-Q oracle does not match training panel")
        by_root.setdefault(str(row["root_id"]), []).append(row)
    move_database = load_move_database(move_database_path)
    examples = []
    invariance_checks = 0
    for root in panel:
        schedule_rows = sorted(by_root.get(root["root_id"], []), key=lambda row: row["pair_id"])
        if len(schedule_rows) != len(root["schedules"]):
            raise ValueError("training root is missing oracle schedules")
        action_sets = [set(row["action_values"]) for row in schedule_rows]
        if not action_sets or any(actions != action_sets[0] for actions in action_sets[1:]):
            raise ValueError("oracle legal-action support changes across schedules")
        actions = sorted(action_sets[0])
        values = np.asarray(
            [np.mean([float(row["action_values"][action]) for row in schedule_rows]) for action in actions],
            dtype=np.float32,
        )
        if not actions or not np.isfinite(values).all():
            raise ValueError("training root has invalid action values")
        reference = None
        for schedule in root["schedules"]:
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                current = np.stack([
                    action_features(
                        state,
                        action,
                        poke_engine=poke_engine,
                        move_database=move_database,
                    )
                    for action in actions
                ])
                if reference is None:
                    reference = current
                elif not np.array_equal(reference, current):
                    raise ValueError("action-Q features changed under hidden determinization")
                invariance_checks += len(actions)
        examples.append(RootExample(root["battle_id"], root["root_id"], actions, reference, values))
    if len(examples) != len(panel) or len({row.battle_id for row in examples}) != len(examples):
        raise ValueError("action-Q examples are not one root per source battle")
    return examples, {
        "panel_sha256": panel_hash,
        "oracle_sha256": _sha256(oracle_path),
        "move_database_sha256": _sha256(move_database_path),
        "determinization_invariance_checks": invariance_checks,
    }


@torch.no_grad()
def metrics(model: PublicActionQ, rows: list[RootExample], mean: torch.Tensor, std: torch.Tensor) -> dict[str, float | int]:
    top1 = 0
    regret = 0.0
    mae = 0.0
    pairs_correct = pairs_total = 0
    actions = 0
    for row in rows:
        features = (torch.from_numpy(row.features) - mean) / std
        prediction = model(features).cpu().numpy()
        chosen = int(np.argmax(prediction))
        best = int(np.argmax(row.values))
        top1 += chosen == best
        regret += float(row.values[best] - row.values[chosen])
        mae += float(np.abs(prediction - row.advantages).sum())
        actions += len(row.actions)
        for left in range(len(row.actions)):
            for right in range(left + 1, len(row.actions)):
                truth = float(row.values[left] - row.values[right])
                estimate = float(prediction[left] - prediction[right])
                if truth != 0:
                    pairs_total += 1
                    pairs_correct += (truth > 0) == (estimate > 0)
    return {
        "battles": len(rows),
        "actions": actions,
        "top1_agreement": top1 / len(rows),
        "mean_oracle_regret": regret / len(rows),
        "advantage_mae": mae / actions,
        "pairwise_order_accuracy": pairs_correct / pairs_total if pairs_total else 0.0,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples, provenance = load_examples(args.panel, args.oracle, args.move_database)
    splits = {
        name: [row for row in examples if _split(row.battle_id, args.split_seed) == name]
        for name in ("train", "validation", "test")
    }
    if any(len(rows) < 25 for rows in splits.values()):
        raise ValueError(f"action-Q battle split is too small: { {k: len(v) for k, v in splits.items()} }")
    train_features = torch.from_numpy(np.concatenate([row.features for row in splits["train"]]))
    mean = train_features.mean(0)
    std = train_features.std(0).clamp_min(1e-4)
    model = PublicActionQ()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 13_441:
        raise RuntimeError(f"action-Q architecture changed: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None
    best_epoch = 0
    best_regret = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(splits["train"]), generator=generator).tolist()
        for start in range(0, len(order), args.batch_roots):
            batch = [splits["train"][index] for index in order[start : start + args.batch_roots]]
            losses = []
            for row in batch:
                features = (torch.from_numpy(row.features) - mean) / std
                target = torch.from_numpy(row.advantages)
                prediction = model(features)
                mse = torch.nn.functional.mse_loss(prediction, target)
                teacher = torch.softmax(target / args.teacher_temperature, dim=0)
                listwise = -(teacher * torch.log_softmax(prediction / args.teacher_temperature, dim=0)).sum()
                losses.append(mse + args.listwise_weight * listwise)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        validation = metrics(model, splits["validation"], mean, std)
        history.append({"epoch": epoch, "validation": validation})
        current = float(validation["mean_oracle_regret"])
        if current < best_regret - 1e-5:
            best_regret = current
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("action-Q training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_metrics = {name: metrics(model, rows, mean, std) for name, rows in splits.items()}
    gate_eligible = (
        final_metrics["validation"]["top1_agreement"] >= 0.35
        and final_metrics["validation"]["mean_oracle_regret"] < 0.03
    )
    source_files = [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in args.source_log]
    panel_sources = {str(row["source_file_sha256"]) for row in _read_jsonl(args.panel)}
    if panel_sources != {row["sha256"] for row in source_files}:
        raise ValueError("declared source logs do not exactly match panel provenance")
    metadata = {
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "split_seed": args.split_seed,
        "teacher_temperature": args.teacher_temperature,
        "prior_temperature": 0.05,
        "c_puct": 2.0,
        "provenance": provenance,
    }
    save_model(args.model_out, model, mean, std, metadata)
    report = {
        "schema": "metagross-public-action-q-training/v1",
        "examples": len(examples),
        "forced_single_action_examples": sum(len(row.actions) == 1 for row in examples),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "gate_eligible": gate_eligible,
        "metrics": final_metrics,
        "history": history,
        "model_sha256": _sha256(args.model_out),
        "provenance": provenance | {"source_files": source_files},
        "configuration": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "batch_roots": args.batch_roots,
            "learning_rate": args.learning_rate,
            "teacher_temperature": args.teacher_temperature,
            "listwise_weight": args.listwise_weight,
        },
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.metrics_out.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.metrics_out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, action="append", required=True)
    parser.add_argument("--move-database", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-roots", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--listwise-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--split-seed", type=int, default=20260814)
    args = parser.parse_args()
    if min(args.epochs, args.batch_roots, args.patience) < 1 or min(args.learning_rate, args.teacher_temperature) <= 0:
        parser.error("training arguments must be positive")
    report = train(args)
    print(json.dumps({key: report[key] for key in ("examples", "parameter_count", "best_epoch", "gate_eligible", "metrics", "model_sha256")}, sort_keys=True))
    raise SystemExit(0 if report["gate_eligible"] else 2)


if __name__ == "__main__":
    main()
