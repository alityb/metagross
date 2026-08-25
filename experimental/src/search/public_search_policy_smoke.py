"""Deterministic CPU-only policy plumbing smoke for PublicSearchStateV1."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from search.public_search_state_v1 import canonical_bytes, validate_finite_policy


ENCODING_DIM = 512
HIDDEN_DIM = 1536
BOTTLENECK_DIM = 256
SEED = 2026081502


def encode_states(states: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Hash canonical public bytes into a fixed, bounded stateless input."""
    result = np.zeros((len(states), ENCODING_DIM), dtype=np.float32)
    for row_index, state in enumerate(states):
        payload = canonical_bytes(state)
        if not payload:
            raise ValueError("empty public state")
        current = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
        previous = np.empty_like(current)
        previous[0] = 0
        previous[1:] = current[:-1]
        indexes = (previous * 257 + current) % ENCODING_DIM
        result[row_index] = np.bincount(indexes, minlength=ENCODING_DIM)
        result[row_index] /= max(1.0, float(result[row_index].sum()))
    return result


class PublicSearchPolicySmoke(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ENCODING_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
            nn.Tanh(),
            nn.Linear(BOTTLENECK_DIM, 13),
        )

    def forward(self, inputs: torch.Tensor, illegal: torch.Tensor) -> torch.Tensor:
        logits = self.layers(inputs)
        logits = logits.masked_fill(illegal, -1.0e9)
        return torch.softmax(logits, dim=-1)


def make_policy() -> PublicSearchPolicySmoke:
    previous = torch.random.get_rng_state()
    try:
        torch.manual_seed(SEED)
        model = PublicSearchPolicySmoke()
    finally:
        torch.random.set_rng_state(previous)
    model.eval()
    return model


def infer(
    model: nn.Module, states: Sequence[Mapping[str, object]]
) -> np.ndarray:
    encoded = torch.from_numpy(encode_states(states))
    illegal = torch.tensor(
        [state["action_table"]["illegal_actions"] for state in states],
        dtype=torch.bool,
    )
    if torch.any(torch.all(illegal, dim=1)):
        raise ValueError("automatic/terminal state has no ordinary policy query")
    with torch.no_grad():
        probabilities = model(encoded, illegal).cpu().numpy()
    for row, state in zip(probabilities, states, strict=True):
        validate_finite_policy(row.tolist(), state["action_table"]["illegal_actions"])
    return probabilities


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
