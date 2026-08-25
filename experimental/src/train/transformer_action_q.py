"""Frozen-R1 embedding dataset and action-Q head."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


DATASET_SCHEMA = "metagross-transformer-action-q-dataset/v1"
MODEL_SCHEMA = "metagross-transformer-action-q/v1"
NUM_ACTIONS = 13
SPLITS = ("train", "validation", "test")


def battle_split(battle_id: str, seed: int = 20260815) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}\0{battle_id}".encode()).digest()[:8], "big"
    ) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


@dataclass
class TransformerActionQDataset:
    battle_ids: list[str]
    root_ids: list[str]
    text_tokens: torch.Tensor
    numbers: torch.Tensor
    illegal_actions: torch.Tensor
    teacher_support: torch.Tensor
    teacher_q: torch.Tensor
    historical_selected_index: torch.Tensor
    split_ids: torch.Tensor
    provenance: dict[str, Any]

    def indices(self, split: str) -> torch.Tensor:
        return torch.nonzero(
            self.split_ids == SPLITS.index(split), as_tuple=False
        ).squeeze(-1)


def load_dataset(path: Path, split_seed: int = 20260815) -> TransformerActionQDataset:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid transformer action-Q dataset schema")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) < 950:
        raise ValueError("transformer action-Q dataset is too small")
    text_width = len(rows[0]["text_tokens"])
    number_width = len(rows[0]["numbers"])
    battle_ids, root_ids = [], []
    text, numbers, illegal, support, q, historical, splits = [], [], [], [], [], [], []
    for row in rows:
        if (
            len(row["text_tokens"]) != text_width
            or len(row["numbers"]) != number_width
            or len(row["illegal_actions"]) != NUM_ACTIONS
            or len(row["teacher_support"]) != NUM_ACTIONS
            or len(row["teacher_q"]) != NUM_ACTIONS
            or not any(row["teacher_support"])
            or any(flag and illegal_flag for flag, illegal_flag in zip(row["teacher_support"], row["illegal_actions"], strict=True))
        ):
            raise ValueError("invalid transformer action-Q record")
        index = row["historical_selected_index"]
        if index is not None and (not isinstance(index, int) or not row["teacher_support"][index]):
            raise ValueError("invalid historical action index")
        battle_ids.append(str(row["battle_id"]))
        root_ids.append(str(row["root_id"]))
        text.append(row["text_tokens"])
        numbers.append(row["numbers"])
        illegal.append(row["illegal_actions"])
        support.append(row["teacher_support"])
        q.append(row["teacher_q"])
        historical.append(-1 if index is None else index)
        splits.append(SPLITS.index(battle_split(battle_ids[-1], split_seed)))
    if len(set(battle_ids)) != len(rows) or len(set(root_ids)) != len(rows):
        raise ValueError("transformer action-Q identities are not unique")
    return TransformerActionQDataset(
        battle_ids=battle_ids,
        root_ids=root_ids,
        text_tokens=torch.tensor(text, dtype=torch.int32),
        numbers=torch.tensor(numbers, dtype=torch.float32),
        illegal_actions=torch.tensor(illegal, dtype=torch.bool),
        teacher_support=torch.tensor(support, dtype=torch.bool),
        teacher_q=torch.tensor(q, dtype=torch.float32),
        historical_selected_index=torch.tensor(historical, dtype=torch.int64),
        split_ids=torch.tensor(splits, dtype=torch.uint8),
        provenance=dict(payload.get("provenance", {})),
    )


def build_stateless_batch(dataset: TransformerActionQDataset, indices: torch.Tensor, device: torch.device):
    text = dataset.text_tokens[indices].to(device)
    numbers = dataset.numbers[indices].to(device)
    illegal = dataset.illegal_actions[indices].to(device)
    obs = {
        "text_tokens": torch.stack((torch.zeros_like(text), text), dim=1),
        "numbers": torch.stack((torch.zeros_like(numbers), numbers), dim=1),
        "illegal_actions": torch.stack((torch.ones_like(illegal), illegal), dim=1),
    }
    rl2 = torch.zeros((len(indices), 2, NUM_ACTIONS + 1), device=device)
    time_idxs = torch.arange(2, device=device).view(1, 2, 1).expand(len(indices), 2, 1)
    return obs, rl2, time_idxs


class TransformerActionQHead(nn.Module):
    def __init__(self, embedding_dim: int = 900, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("action-Q embeddings must be [batch, width]")
        return self.net(embeddings)


def frozen_current_embeddings(agent, obs, rl2, time_idxs) -> torch.Tensor:
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        embeddings, _ = agent.get_state_embedding(
            obs=obs, rl2s=rl2, time_idxs=time_idxs, hidden_state=None
        )
    return embeddings[:, -1].detach()


def centered_targets(q: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    count = support.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (q * support).sum(dim=1, keepdim=True) / count
    return (q - mean) * support


def action_q_loss(prediction: torch.Tensor, q: torch.Tensor, support: torch.Tensor, *, temperature: float = 0.05, listwise_weight: float = 0.1) -> torch.Tensor:
    if prediction.shape != q.shape or prediction.shape != support.shape:
        raise ValueError("action-Q tensors must have identical [batch, 13] shapes")
    target = centered_targets(q, support)
    mse = ((prediction - target).square() * support).sum() / support.sum()
    teacher = torch.softmax((target / temperature).masked_fill(~support, float("-inf")), dim=1)
    student_log = torch.log_softmax((prediction / temperature).masked_fill(~support, float("-inf")), dim=1)
    listwise = -(teacher * student_log).masked_fill(~support, 0.0).sum(dim=1).mean()
    return mse + listwise_weight * listwise


@torch.no_grad()
def action_q_metrics(prediction: torch.Tensor, q: torch.Tensor, support: torch.Tensor, historical: torch.Tensor | None = None) -> dict[str, float | int]:
    masked_prediction = prediction.masked_fill(~support, float("-inf"))
    masked_q = q.masked_fill(~support, float("-inf"))
    selected = masked_prediction.argmax(dim=1)
    best = masked_q.argmax(dim=1)
    rows = torch.arange(len(q))
    report: dict[str, float | int] = {
        "battles": len(q),
        "actions": int(support.sum()),
        "top1_agreement": float((selected == best).float().mean()),
        "mean_oracle_regret": float((q[rows, best] - q[rows, selected]).mean()),
    }
    pairs_correct = pairs_total = 0
    for row in range(len(q)):
        indices = torch.nonzero(support[row], as_tuple=False).squeeze(-1)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                truth = float(q[row, left] - q[row, right])
                estimate = float(prediction[row, left] - prediction[row, right])
                if truth != 0:
                    pairs_total += 1
                    pairs_correct += (truth > 0) == (estimate > 0)
    report["pairwise_order_accuracy"] = pairs_correct / pairs_total if pairs_total else 0.0
    if historical is not None:
        available = historical >= 0
        available_rows = rows[available]
        chosen = historical[available]
        report["historical_battles"] = int(available.sum())
        report["historical_top1_agreement"] = float((chosen == best[available]).float().mean())
        report["historical_mean_oracle_regret"] = float(
            (q[available_rows, best[available]] - q[available_rows, chosen]).mean()
        )
    return report
