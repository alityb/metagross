"""Terminal-outcome value learning on the accepted r1 transformer representation.

This is deliberately separate from search-policy distillation.  Targets are only
observed terminal wins/losses, splits are battle-disjoint, and the accepted r1
encoder is frozen while a small calibrated value head is fitted on its sequence
embeddings.
"""
from __future__ import annotations

import hashlib
import contextlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch
from torch import nn


SCHEMA = 1
NUM_ACTIONS = 13
RL2_WIDTH = NUM_ACTIONS + 1
SPLITS = ("train", "validation", "test")
SOURCE_KINDS = ("human", "selfplay", "league")
SourceKind = Literal["human", "selfplay", "league"]


class TerminalValueDatasetError(ValueError):
    """A trajectory cannot enter the causal terminal-outcome experiment."""


def battle_split(battle_tag: str, seed: int = 20260813) -> str:
    digest = hashlib.sha256(f"{seed}\0{battle_tag}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


@dataclass(frozen=True)
class TerminalTrajectory:
    battle_tag: str
    pov: str
    source_kind: SourceKind
    outcome: float
    text_tokens: torch.Tensor
    numbers: torch.Tensor
    illegal_actions: torch.Tensor
    rl2: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.text_tokens.shape[0])


@dataclass
class TerminalValueDataset:
    trajectories: tuple[TerminalTrajectory, ...]
    split_ids: torch.Tensor
    split_counts: dict[str, int]
    split_battles: dict[str, int]
    source_counts: dict[str, int]
    source_battles: dict[str, int]
    battle_multiplicity: dict[str, int]

    def indices(self, split: str) -> torch.Tensor:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        return torch.nonzero(
            self.split_ids == SPLITS.index(split), as_tuple=False
        ).squeeze(-1)


def _fail(path: Path, line_number: int, message: str) -> TerminalValueDatasetError:
    return TerminalValueDatasetError(f"{path}:{line_number}: {message}")


def load_terminal_value_dataset(
    paths: Sequence[Path], *, split_seed: int = 20260813
) -> TerminalValueDataset:
    """Load exact transformer trajectories without mixing battles across splits."""
    trajectories: list[TerminalTrajectory] = []
    split_ids: list[int] = []
    seen: set[tuple[str, str, str]] = set()
    split_counts = {name: 0 for name in SPLITS}
    split_battle_sets = {name: set() for name in SPLITS}
    source_counts: Counter[str] = Counter()
    source_battle_sets = {name: set() for name in SOURCE_KINDS}
    battle_sources: dict[str, str] = {}
    battle_multiplicity: Counter[str] = Counter()
    expected_shapes: tuple[int, int] | None = None

    for path in map(Path, paths):
        if path.name.endswith(".lz4"):
            import lz4.frame

            opened = lz4.frame.open(path, mode="rt", encoding="utf-8")
        else:
            opened = path.open(encoding="utf-8")
        with contextlib.closing(opened) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise _fail(path, line_number, "invalid JSON") from None
                if row.get("schema") != SCHEMA:
                    raise _fail(path, line_number, "unsupported schema")
                battle_tag, pov, source = (
                    row.get("battle_tag"),
                    row.get("pov"),
                    row.get("source_kind"),
                )
                if not isinstance(battle_tag, str) or not battle_tag:
                    raise _fail(path, line_number, "invalid battle_tag")
                if not isinstance(pov, str) or not pov:
                    raise _fail(path, line_number, "invalid pov")
                if source not in SOURCE_KINDS:
                    raise _fail(path, line_number, "invalid source_kind")
                key = (battle_tag, pov, source)
                if key in seen:
                    raise _fail(path, line_number, "duplicate trajectory identity")
                seen.add(key)
                previous_source = battle_sources.setdefault(battle_tag, source)
                if previous_source != source:
                    raise _fail(path, line_number, "battle appears in multiple source kinds")
                outcome = row.get("outcome")
                if isinstance(outcome, bool) or outcome not in (0, 1):
                    raise _fail(path, line_number, "outcome must be terminal 0 or 1")

                text, numbers = row.get("text_tokens"), row.get("numbers")
                illegal, rl2 = row.get("illegal_actions"), row.get("rl2")
                if not all(isinstance(value, list) and value for value in (text, numbers, illegal, rl2)):
                    raise _fail(path, line_number, "empty or invalid sequence")
                length = len(text)
                if len(numbers) != length or len(illegal) != length or len(rl2) != length:
                    raise _fail(path, line_number, "unaligned sequence lengths")
                text_width = len(text[0]) if isinstance(text[0], list) else 0
                number_width = len(numbers[0]) if isinstance(numbers[0], list) else 0
                if not text_width or not number_width:
                    raise _fail(path, line_number, "invalid observation rank")
                if expected_shapes is None:
                    expected_shapes = (text_width, number_width)
                if expected_shapes != (text_width, number_width):
                    raise _fail(path, line_number, "observation shape changed")
                if any(
                    not isinstance(row_values, list) or len(row_values) != text_width
                    or not all(isinstance(value, int) and not isinstance(value, bool) for value in row_values)
                    for row_values in text
                ):
                    raise _fail(path, line_number, "invalid text_tokens")
                try:
                    numeric_rows = [[float(value) for value in values] for values in numbers]
                    rl2_rows = [[float(value) for value in values] for values in rl2]
                except (TypeError, ValueError, OverflowError):
                    raise _fail(path, line_number, "non-numeric observation") from None
                if any(len(values) != number_width for values in numeric_rows) or not all(
                    math.isfinite(value) for values in numeric_rows for value in values
                ):
                    raise _fail(path, line_number, "invalid numbers")
                if any(
                    not isinstance(values, list)
                    or len(values) != NUM_ACTIONS
                    or not all(isinstance(value, bool) for value in values)
                    for values in illegal
                ):
                    raise _fail(path, line_number, "invalid illegal_actions")
                if any(len(values) != RL2_WIDTH for values in rl2_rows) or not all(
                    math.isfinite(value) for values in rl2_rows for value in values
                ):
                    raise _fail(path, line_number, "invalid rl2")
                if any(rl2_rows[0]):
                    raise _fail(path, line_number, "rl2 must begin at the causal zero boundary")

                split = battle_split(battle_tag, split_seed)
                trajectories.append(
                    TerminalTrajectory(
                        battle_tag=battle_tag,
                        pov=pov,
                        source_kind=source,
                        outcome=float(outcome),
                        text_tokens=torch.tensor(text, dtype=torch.int32),
                        numbers=torch.tensor(numeric_rows, dtype=torch.float32),
                        illegal_actions=torch.tensor(illegal, dtype=torch.bool),
                        rl2=torch.tensor(rl2_rows, dtype=torch.float32),
                    )
                )
                split_ids.append(SPLITS.index(split))
                split_counts[split] += 1
                split_battle_sets[split].add(battle_tag)
                source_counts[source] += 1
                source_battle_sets[source].add(battle_tag)
                battle_multiplicity[battle_tag] += 1

    if not trajectories:
        raise TerminalValueDatasetError("no admitted terminal trajectories")
    return TerminalValueDataset(
        trajectories=tuple(trajectories),
        split_ids=torch.tensor(split_ids, dtype=torch.uint8),
        split_counts=split_counts,
        split_battles={key: len(value) for key, value in split_battle_sets.items()},
        source_counts={key: source_counts[key] for key in SOURCE_KINDS},
        source_battles={key: len(source_battle_sets[key]) for key in SOURCE_KINDS},
        battle_multiplicity=dict(battle_multiplicity),
    )


def batch_indices(
    indices: torch.Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    shuffle: bool = False,
) -> Iterator[torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if shuffle:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def build_value_batch(
    dataset: TerminalValueDataset,
    indices: torch.Tensor,
    device: torch.device,
    *,
    max_seq_len: int = 128,
    source_mix: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad causal histories and weight every battle equally.

    Histories longer than r1's context window keep their most recent states,
    exactly as deployment does.
    """
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if source_mix is not None and (
        set(source_mix) != set(SOURCE_KINDS)
        or not all(math.isfinite(value) and value >= 0.0 for value in source_mix.values())
        or not math.isclose(sum(source_mix.values()), 1.0, abs_tol=1e-6)
        or any(source_mix[source] > 0 and dataset.source_battles[source] == 0 for source in SOURCE_KINDS)
    ):
        raise ValueError("source_mix must be a feasible distribution over all source kinds")
    rows = [dataset.trajectories[int(index)] for index in indices]
    if not rows:
        raise ValueError("cannot build an empty value batch")
    batch_size, length = len(rows), min(max_seq_len, max(row.length for row in rows))
    text_width, number_width = rows[0].text_tokens.shape[1], rows[0].numbers.shape[1]
    obs = {
        "text_tokens": torch.zeros((batch_size, length, text_width), dtype=torch.int32, device=device),
        "numbers": torch.zeros((batch_size, length, number_width), dtype=torch.float32, device=device),
        "illegal_actions": torch.ones((batch_size, length, NUM_ACTIONS), dtype=torch.bool, device=device),
    }
    rl2 = torch.zeros((batch_size, length, RL2_WIDTH), dtype=torch.float32, device=device)
    time_idxs = torch.arange(length, device=device).view(1, length, 1).expand(batch_size, length, 1)
    labels = torch.zeros((batch_size, length), dtype=torch.float32, device=device)
    weights = torch.zeros_like(labels)
    for batch_index, row in enumerate(rows):
        size = min(row.length, max_seq_len)
        start = row.length - size
        obs["text_tokens"][batch_index, :size] = row.text_tokens[start:].to(device)
        obs["numbers"][batch_index, :size] = row.numbers[start:].to(device)
        obs["illegal_actions"][batch_index, :size] = row.illegal_actions[start:].to(device)
        rl2[batch_index, :size] = row.rl2[start:].to(device)
        labels[batch_index, :size] = row.outcome
        battle_weight = 1.0
        if source_mix is not None:
            battle_weight = source_mix[row.source_kind] / dataset.source_battles[row.source_kind]
        weights[batch_index, :size] = battle_weight / (
            size * dataset.battle_multiplicity[row.battle_tag]
        )
    return obs, rl2, time_idxs, labels, weights


class TransformerValueHead(nn.Module):
    """Small win-probability head trained above frozen r1 embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid value-head dimensions")
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("transformer embeddings must be [batch, time, width]")
        return self.net(embeddings).squeeze(-1)


def frozen_r1_embeddings(agent, obs, rl2, time_idxs) -> torch.Tensor:
    """Use r1 as a fixed representation, preventing accidental policy drift."""
    agent.eval()
    if not getattr(agent, "_metagross_value_frozen", False):
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
        agent._metagross_value_frozen = True
    with torch.no_grad():
        embeddings, _ = agent.get_state_embedding(
            obs=obs, rl2s=rl2, time_idxs=time_idxs, hidden_state=None
        )
    return embeddings.detach()


def battle_balanced_bce(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError("value tensors must have identical [batch, time] shapes")
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid value weights")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none"
    )
    return (losses * weights).sum() / weights.sum()


def value_metrics(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> dict[str, float]:
    probabilities = logits.float().sigmoid()
    normalized = weights.float() / weights.sum()
    eps = torch.finfo(probabilities.dtype).eps
    brier = ((probabilities - labels.float()).square() * normalized).sum()
    log_loss = (
        -(
            labels.float() * probabilities.clamp_min(eps).log()
            + (1.0 - labels.float()) * (1.0 - probabilities).clamp_min(eps).log()
        )
        * normalized
    ).sum()
    accuracy = (
        ((probabilities >= 0.5) == labels.bool()).float() * normalized
    ).sum()
    return {
        "brier": float(brier),
        "log_loss": float(log_loss),
        "accuracy": float(accuracy),
    }
