from __future__ import annotations

from typing import Iterable

import torch

from model.network import PokeNet
from model.state import EncodedState, stack_encoded


@torch.no_grad()
def batched_policy_value(model: PokeNet, states: Iterable[EncodedState]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = stack_encoded(list(states))
    return model.policy_value(batch)
