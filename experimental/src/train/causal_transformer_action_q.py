"""Low-capacity residual action-Q guidance over full causal R1 histories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


DATASET_SCHEMA = "metagross-causal-transformer-action-q-dataset/v1"
MODEL_SCHEMA = "metagross-causal-transformer-action-q/v1"
NUM_ACTIONS = 13
SPLITS = ("train", "validation", "test")


def battle_split(battle_id: str, seed: int = 20260816) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}\0{battle_id}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


@dataclass
class CausalActionQDataset:
    records: list[dict[str, Any]]
    split_ids: torch.Tensor
    provenance: dict[str, Any]

    def indices(self, split: str) -> torch.Tensor:
        return torch.nonzero(self.split_ids == SPLITS.index(split), as_tuple=False).squeeze(-1)


def load_dataset(path: Path, split_seed: int = 20260816) -> CausalActionQDataset:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid causal transformer action-Q dataset schema")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 1000:
        raise ValueError("causal transformer action-Q dataset is too small")
    seen = set()
    splits = []
    for record in records:
        battle_id = record.get("battle_id")
        length = len(record.get("text_tokens", []))
        if not isinstance(battle_id, str) or battle_id in seen or not 1 <= length <= 128:
            raise ValueError("invalid causal action-Q record identity or length")
        seen.add(battle_id)
        if not (
            len(record.get("numbers", [])) == length
            and len(record.get("illegal_actions", [])) == length
            and len(record.get("rl2", [])) == length
            and record.get("time_indices") == list(range(length))
            and len(record.get("teacher_support", [])) == NUM_ACTIONS
            and len(record.get("teacher_q", [])) == NUM_ACTIONS
            and len(record.get("r1_probs", [])) == NUM_ACTIONS
        ):
            raise ValueError("invalid causal action-Q sequence")
        splits.append(SPLITS.index(battle_split(battle_id, split_seed)))
    return CausalActionQDataset(records, torch.tensor(splits, dtype=torch.uint8), dict(payload.get("provenance", {})))


class CausalResidualQHead(nn.Module):
    """A linear correction to the accepted causal-R1 action logits."""

    def __init__(self, embedding_dim: int = 900, actions: int = NUM_ACTIONS):
        super().__init__()
        self.normalization = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.correction = nn.Linear(embedding_dim, actions)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("causal action-Q embeddings must be [batch, width]")
        return self.correction(self.normalization(embeddings))


def corrected_logits(base_probs: torch.Tensor, correction: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if base_probs.shape != correction.shape or support.shape != correction.shape:
        raise ValueError("causal action-Q tensors must share [batch, 13] shape")
    logits = base_probs.clamp_min(1e-8).log() + correction
    return logits.masked_fill(~support, float("-inf"))


def residual_q_loss(
    correction: torch.Tensor,
    base_probs: torch.Tensor,
    teacher_q: torch.Tensor,
    support: torch.Tensor,
    *,
    teacher_temperature: float = 0.05,
    anchor_weight: float = 0.01,
) -> torch.Tensor:
    count = support.sum(dim=1, keepdim=True).clamp_min(1)
    centered = teacher_q - (teacher_q * support).sum(dim=1, keepdim=True) / count
    teacher = torch.softmax((centered / teacher_temperature).masked_fill(~support, float("-inf")), dim=1)
    student_log = torch.log_softmax(corrected_logits(base_probs, correction, support), dim=1)
    listwise = -(teacher * student_log).masked_fill(~support, 0.0).sum(dim=1).mean()
    anchor = (correction.square() * support).sum() / support.sum()
    return listwise + anchor_weight * anchor


@torch.no_grad()
def ranking_metrics(
    probabilities: torch.Tensor,
    teacher_q: torch.Tensor,
    support: torch.Tensor,
    historical: torch.Tensor | None = None,
) -> dict[str, float | int | list[float]]:
    masked_probs = probabilities.masked_fill(~support, float("-inf"))
    masked_q = teacher_q.masked_fill(~support, float("-inf"))
    selected = masked_probs.argmax(dim=1)
    best = masked_q.argmax(dim=1)
    row = torch.arange(len(teacher_q))
    regrets = teacher_q[row, best] - teacher_q[row, selected]
    result: dict[str, float | int | list[float]] = {
        "battles": len(teacher_q),
        "top1_agreement": float((selected == best).float().mean()),
        "mean_oracle_regret": float(regrets.mean()),
        "regrets": [float(value) for value in regrets],
    }
    if historical is not None:
        historical_regret = teacher_q[row, best] - teacher_q[row, historical]
        result["historical_top1_agreement"] = float((historical == best).float().mean())
        result["historical_mean_oracle_regret"] = float(historical_regret.mean())
    return result

