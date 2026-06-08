from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from agent.player import decode_action_to_order
from model.network import PokeNet
from model.state import EncodedState, Vocabulary, encode_state

try:
    from poke_env.player import Player
except Exception:  # pragma: no cover - poke-env is optional for local imports
    Player = object  # type: ignore[assignment,misc]


@dataclass
class RolloutStep:
    state: EncodedState
    action: int
    log_prob: float
    value: float
    outcome: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


class RolloutPlayer(Player):  # type: ignore[misc,valid-type]
    """Poke-env player that samples from PokeNet and records PPO trajectories."""

    def __init__(self, model: PokeNet, vocab: Vocabulary, device: torch.device, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.vocab = vocab
        self.device = device
        self.buffer: list[RolloutStep] = []

    def choose_move(self, battle: Any) -> Any:
        state = encode_state(battle, vocab=self.vocab)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state)
            dist = torch.distributions.Categorical(logits=logits[0])
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        action = int(action_idx.detach().cpu())
        self.buffer.append(
            RolloutStep(
                state=state,
                action=action,
                log_prob=float(log_prob.detach().cpu()),
                value=float(value[0].detach().cpu()),
            )
        )
        return decode_action_to_order(self, battle, action)

    def collect_episode(self, outcome: float) -> list[RolloutStep]:
        steps = list(self.buffer)
        for step in steps:
            step.outcome = float(outcome)
        self.buffer.clear()
        return steps


def compute_gae(steps: list[RolloutStep], gamma: float = 0.9999, lam: float = 0.754) -> list[RolloutStep]:
    if not steps:
        return steps
    gae = 0.0
    next_value = 0.0
    for index in range(len(steps) - 1, -1, -1):
        reward = steps[index].outcome if index == len(steps) - 1 else 0.0
        delta = reward + gamma * next_value - steps[index].value
        gae = delta + gamma * lam * gae
        steps[index].advantage = gae
        steps[index].return_ = gae + steps[index].value
        next_value = steps[index].value
    advantages = torch.tensor([step.advantage for step in steps], dtype=torch.float32)
    std = float(advantages.std(unbiased=False))
    mean = float(advantages.mean())
    for step in steps:
        step.advantage = 0.0 if std <= 1e-8 or not math.isfinite(std) else (step.advantage - mean) / (std + 1e-8)
    return steps
