from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ActionStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MCTSNode:
    priors: list[float]
    to_play: int = 1
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    stats: dict[int, ActionStats] = field(init=False)

    def __post_init__(self) -> None:
        self.stats = {idx: ActionStats(float(prior)) for idx, prior in enumerate(self.priors)}

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def select_action(self, c_puct: float = 1.25) -> int:
        total = max(1, self.visits)
        best_score = -float("inf")
        best_action = 0
        for action, stats in self.stats.items():
            if stats.prior <= 0:
                continue
            exploration = c_puct * stats.prior * math.sqrt(total) / (1 + stats.visits)
            score = stats.q + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def update(self, action: int, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        stats = self.stats[action]
        stats.visits += 1
        stats.value_sum += value

    def visit_distribution(self) -> list[float]:
        visits = [self.stats[idx].visits for idx in range(len(self.priors))]
        total = sum(visits)
        if total <= 0:
            return list(self.priors)
        return [visit / total for visit in visits]
