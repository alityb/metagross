#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


NUMERIC_FIELDS = ["active_hp", "opponent_hp", "player_alive", "opponent_alive", "turn_index"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def game_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["run_id"]), int(row["game_index"])


def token_features(row: dict[str, Any]) -> list[str]:
    tokens = [
        f"bucket={row.get('bucket')}",
        f"active={row.get('active')}",
        f"opponent={row.get('opponent_active')}",
        f"active_status={row.get('active_status')}",
        f"opponent_status={row.get('opponent_status')}",
        f"player_alive={row.get('player_alive')}",
        f"opponent_alive={row.get('opponent_alive')}",
        f"forced_switch={row.get('forced_switch')}",
        f"has_sleep={row.get('has_sleep_move')}",
        f"has_para={row.get('has_para_move')}",
        f"has_boom={row.get('has_boom_move')}",
        f"has_recovery={row.get('has_recovery_move')}",
        f"active_hp_bin={hp_bin(row.get('active_hp'))}",
        f"opponent_hp_bin={hp_bin(row.get('opponent_hp'))}",
    ]
    tokens.extend(f"active_move={move}" for move in row.get("active_moves", []))
    tokens.extend(f"opponent_move={move}" for move in row.get("opponent_revealed_moves", []))
    return tokens


def hp_bin(value: Any) -> str:
    try:
        hp = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if hp <= 0:
        return "0"
    if hp <= 0.25:
        return "1-25"
    if hp <= 0.5:
        return "26-50"
    if hp <= 0.75:
        return "51-75"
    return "76-100"


def split_by_game(rows: list[dict[str, Any]], test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    keys = sorted({game_key(row) for row in rows})
    rng.shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    test_keys = set(keys[:n_test])
    train_idx, test_idx = [], []
    for idx, row in enumerate(rows):
        if game_key(row) in test_keys:
            test_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, test_idx


def build_vocab(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, int]:
    counts = Counter()
    for idx in indices:
        counts.update(token_features(rows[idx]))
    return {token: i for i, (token, _count) in enumerate(counts.most_common())}


def build_classes(rows: list[dict[str, Any]], target_field: str) -> list[str]:
    return sorted({str(row[target_field]) for row in rows})


def featurize(rows: list[dict[str, Any]], indices: list[int], vocab: dict[str, int]) -> torch.Tensor:
    width = len(vocab) + len(NUMERIC_FIELDS)
    x = torch.zeros((len(indices), width), dtype=torch.float32)
    for row_pos, idx in enumerate(indices):
        row = rows[idx]
        for token in token_features(row):
            col = vocab.get(token)
            if col is not None:
                x[row_pos, col] = 1.0
        offset = len(vocab)
        for field_pos, field in enumerate(NUMERIC_FIELDS):
            value = row.get(field)
            try:
                x[row_pos, offset + field_pos] = float(value)
            except (TypeError, ValueError):
                x[row_pos, offset + field_pos] = 0.0
    # Keep turn scale tame.
    turn_col = len(vocab) + NUMERIC_FIELDS.index("turn_index")
    x[:, turn_col] /= 200.0
    return x


def labels(rows: list[dict[str, Any]], indices: list[int], classes: list[str], target_field: str) -> torch.Tensor:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    return torch.tensor([class_to_idx[str(rows[idx][target_field])] for idx in indices], dtype=torch.long)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return 0.0
    return (logits.argmax(dim=1) == y).float().mean().item()


def per_class_accuracy(logits: torch.Tensor, y: torch.Tensor, classes: list[str]) -> dict[str, dict[str, float]]:
    pred = logits.argmax(dim=1)
    out = {}
    for idx, label in enumerate(classes):
        mask = y == idx
        support = int(mask.sum().item())
        correct = int(((pred == idx) & mask).sum().item())
        out[label] = {"support": support, "accuracy": correct / support if support else 0.0}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Tauros action-kind distillation model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--target-field", choices=["action_kind", "action"], default="action_kind")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = []
    for row in load_rows(args.data):
        target = str(row.get(args.target_field))
        if target in {"unknown", "noop"} or target.startswith("unknown:"):
            continue
        rows.append(row)
    train_idx, test_idx = split_by_game(rows, args.test_fraction, args.seed)
    vocab = build_vocab(rows, train_idx)
    classes = build_classes(rows, args.target_field)
    x_train = featurize(rows, train_idx, vocab)
    y_train = labels(rows, train_idx, classes, args.target_field)
    x_test = featurize(rows, test_idx, vocab)
    y_test = labels(rows, test_idx, classes, args.target_field)

    if args.hidden_size > 0:
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, len(classes)),
        )
    else:
        model = torch.nn.Linear(x_train.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_logits = model(x_train)
        test_logits = model(x_test)
        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)

    majority = Counter(y_train.tolist()).most_common(1)[0][0]
    majority_test_acc = (y_test == majority).float().mean().item()
    class_counts = Counter(classes[int(y)] for y in y_train.tolist())
    metrics = {
        "data": str(args.data),
        "target_field": args.target_field,
        "examples": len(rows),
        "train_examples": len(train_idx),
        "test_examples": len(test_idx),
        "classes": classes,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "majority_class": classes[majority],
        "majority_test_accuracy": majority_test_acc,
        "train_class_counts": dict(class_counts),
        "test_per_class": per_class_accuracy(test_logits, y_test, classes),
        "vocab_size": len(vocab),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    if args.hidden_size > 0:
        first = model[0]
        second = model[2]
        model_payload = {
            "model_type": "mlp_relu",
            "classes": classes,
            "vocab": vocab,
            "numeric_fields": NUMERIC_FIELDS,
            "hidden_size": args.hidden_size,
            "w1": first.weight.detach().tolist(),
            "b1": first.bias.detach().tolist(),
            "w2": second.weight.detach().tolist(),
            "b2": second.bias.detach().tolist(),
            "metrics": metrics,
        }
    else:
        model_payload = {
            "model_type": "linear",
            "classes": classes,
            "vocab": vocab,
            "numeric_fields": NUMERIC_FIELDS,
            "weight": model.weight.detach().tolist(),
            "bias": model.bias.detach().tolist(),
            "metrics": metrics,
        }
    args.model_out.write_text(json.dumps(model_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
