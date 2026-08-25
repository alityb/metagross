"""Dedicated, deployment-aligned policy distillation for schema-v3 roots.

This module deliberately does not call the offline actor-critic trainer.  It
optimizes only the policy path consumed by the live prior server and only the
final (inference) gamma head.  The exported state-dict schema is unchanged, so
the candidate is a one-checkpoint substitution in the accepted r1 stack.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch


NUM_ACTIONS = 13
SCHEMA = 3
HYBRID_ACTION_WEIGHT = 0.25
Arm = Literal["parent", "action", "visits", "hybrid"]
SPLITS = ("train", "validation", "test")


class SearchPolicyDatasetError(ValueError):
    """A root row cannot be admitted to the causal policy experiment."""


def battle_split(battle_tag: str, seed: int = 20260812) -> str:
    """Stable 80/10/10 split by battle, never by decision row."""
    digest = hashlib.sha256(f"{seed}\0{battle_tag}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _distribution(
    row: dict, field: str, illegal: list[bool], path: Path, line_number: int
) -> list[float]:
    raw = row.get(field)
    if not isinstance(raw, list) or len(raw) != NUM_ACTIONS:
        raise SearchPolicyDatasetError(
            f"{path}:{line_number}: {field} must have {NUM_ACTIONS} entries"
        )
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise SearchPolicyDatasetError(
            f"{path}:{line_number}: {field} is non-numeric"
        ) from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise SearchPolicyDatasetError(
            f"{path}:{line_number}: {field} has invalid mass"
        )
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise SearchPolicyDatasetError(
            f"{path}:{line_number}: {field} mass is {total}"
        )
    if any(value > 0.0 and flag for value, flag in zip(values, illegal)):
        raise SearchPolicyDatasetError(
            f"{path}:{line_number}: {field} puts mass on an illegal action"
        )
    return [value / total for value in values]


@dataclass
class SearchPolicyDataset:
    text_tokens: torch.Tensor
    numbers: torch.Tensor
    illegal_actions: torch.Tensor
    parent_targets: torch.Tensor
    action_targets: torch.Tensor
    visit_targets: torch.Tensor
    split_ids: torch.Tensor
    split_counts: dict[str, int]
    battle_counts: dict[str, int]

    @property
    def count(self) -> int:
        return int(self.text_tokens.shape[0])

    def indices(self, split: str) -> torch.Tensor:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        return torch.nonzero(
            self.split_ids == SPLITS.index(split), as_tuple=False
        ).squeeze(-1)

    def targets(self, arm: Arm) -> torch.Tensor:
        if arm == "hybrid":
            return (
                HYBRID_ACTION_WEIGHT * self.action_targets
                + (1.0 - HYBRID_ACTION_WEIGHT) * self.visit_targets
            )
        return {
            "parent": self.parent_targets,
            "action": self.action_targets,
            "visits": self.visit_targets,
        }[arm]


def load_search_policy_dataset(
    path: Path, *, split_seed: int = 20260812
) -> SearchPolicyDataset:
    """Load exact schema-v3 observations and all three causal policy targets."""
    path = Path(path)
    text_rows: list[list[int]] = []
    number_rows: list[list[float]] = []
    illegal_rows: list[list[bool]] = []
    parent_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    visit_rows: list[list[float]] = []
    split_rows: list[int] = []
    split_counts = {name: 0 for name in SPLITS}
    split_battles = {name: set() for name in SPLITS}
    seen: set[tuple[str, str, int]] = set()
    text_len: int | None = None
    numbers_len: int | None = None

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if row.get("schema") != SCHEMA:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: unsupported schema {row.get('schema')!r}"
                )
            battle_tag = row.get("battle_tag")
            username = row.get("username")
            decision_idx = row.get("decision_idx")
            if not isinstance(battle_tag, str) or not battle_tag:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid battle_tag"
                )
            if not isinstance(username, str) or not username:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid username"
                )
            if (
                isinstance(decision_idx, bool)
                or not isinstance(decision_idx, int)
                or decision_idx < 0
            ):
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid decision_idx"
                )
            key = (battle_tag, username, decision_idx)
            if key in seen:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: duplicate decision key {key!r}"
                )
            seen.add(key)

            text = row.get("text_tokens")
            numbers = row.get("numbers")
            illegal = row.get("illegal_actions")
            if not isinstance(text, list) or not text or not all(
                isinstance(token, int) and not isinstance(token, bool) for token in text
            ):
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid text_tokens"
                )
            if not isinstance(numbers, list) or not numbers:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid numbers"
                )
            try:
                number_values = [float(value) for value in numbers]
            except (TypeError, ValueError) as exc:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: non-numeric numbers"
                ) from exc
            if not all(math.isfinite(value) for value in number_values):
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: non-finite numbers"
                )
            if (
                not isinstance(illegal, list)
                or len(illegal) != NUM_ACTIONS
                or not all(isinstance(flag, bool) for flag in illegal)
                or all(illegal)
            ):
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid illegal_actions"
                )
            if text_len is None:
                text_len, numbers_len = len(text), len(numbers)
            if len(text) != text_len or len(numbers) != numbers_len:
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: inconsistent observation shape"
                )

            parent = _distribution(
                row, "policy_probs", illegal, path, line_number
            )
            visits = _distribution(
                row, "visit_target_13", illegal, path, line_number
            )
            selected = row.get("selected_action_index")
            if (
                isinstance(selected, bool)
                or not isinstance(selected, int)
                or not 0 <= selected < NUM_ACTIONS
                or illegal[selected]
                or visits[selected] <= 0.0
            ):
                raise SearchPolicyDatasetError(
                    f"{path}:{line_number}: invalid selected_action_index"
                )
            action = [0.0] * NUM_ACTIONS
            action[selected] = 1.0
            split = battle_split(battle_tag, split_seed)

            text_rows.append(text)
            number_rows.append(number_values)
            illegal_rows.append(illegal)
            parent_rows.append(parent)
            action_rows.append(action)
            visit_rows.append(visits)
            split_rows.append(SPLITS.index(split))
            split_counts[split] += 1
            split_battles[split].add(battle_tag)

    if not text_rows:
        raise SearchPolicyDatasetError(f"{path}: no admitted records")
    if any(count == 0 for count in split_counts.values()):
        raise SearchPolicyDatasetError(
            f"{path}: empty battle-level split: {split_counts}"
        )
    return SearchPolicyDataset(
        text_tokens=torch.tensor(text_rows, dtype=torch.int32),
        numbers=torch.tensor(number_rows, dtype=torch.float32),
        illegal_actions=torch.tensor(illegal_rows, dtype=torch.bool),
        parent_targets=torch.tensor(parent_rows, dtype=torch.float32),
        action_targets=torch.tensor(action_rows, dtype=torch.float32),
        visit_targets=torch.tensor(visit_rows, dtype=torch.float32),
        split_ids=torch.tensor(split_rows, dtype=torch.uint8),
        split_counts=split_counts,
        battle_counts={name: len(split_battles[name]) for name in SPLITS},
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
        order = torch.randperm(len(indices), generator=generator)
        indices = indices[order]
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def build_stateless_batch(
    dataset: SearchPolicyDataset,
    indices: torch.Tensor,
    arm: Arm,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact blank-step/current-step request used by deployment."""
    text = dataset.text_tokens[indices].to(device)
    numbers = dataset.numbers[indices].to(device)
    illegal = dataset.illegal_actions[indices].to(device)
    targets = dataset.targets(arm)[indices].to(device)
    obs = {
        "text_tokens": torch.stack((torch.zeros_like(text), text), dim=1),
        "numbers": torch.stack((torch.zeros_like(numbers), numbers), dim=1),
        "illegal_actions": torch.stack((torch.ones_like(illegal), illegal), dim=1),
    }
    rl2s = torch.zeros((len(indices), 2, NUM_ACTIONS + 1), device=device)
    time_idxs = (
        torch.arange(2, device=device)
        .long()
        .view(1, 2, 1)
        .expand(len(indices), 2, 1)
    )
    return obs, rl2s, time_idxs, targets


def deployment_policy_probs(agent, obs, rl2s, time_idxs) -> torch.Tensor:
    """Return only the exact policy head read by the live prior server."""
    embedding, _ = agent.get_state_embedding(
        obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
    )
    distributions = agent.actor(
        embedding,
        straight_from_obs={
            key: obs[key][:, : embedding.shape[1]]
            for key in agent.pass_obs_keys_to_actor
        },
    )
    return distributions.probs[:, -1, -1, :]


def policy_cross_entropy(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if probs.shape != targets.shape or probs.shape[-1] != NUM_ACTIONS:
        raise ValueError("policy probabilities and targets must be [B, 13]")
    eps = torch.finfo(probs.dtype).eps
    return -(targets * probs.clamp_min(eps).log()).sum(dim=-1).mean()


def policy_metrics(probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    eps = torch.finfo(probs.dtype).eps
    target_eps = torch.finfo(targets.dtype).eps
    ce = -(targets * probs.clamp_min(eps).log()).sum(dim=-1)
    entropy = -(targets * targets.clamp_min(target_eps).log()).sum(dim=-1)
    return {
        "cross_entropy_nats": float(ce.mean()),
        "kl_target_student_nats": float((ce - entropy).mean()),
        "top1_agreement": float(
            (probs.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean()
        ),
    }
