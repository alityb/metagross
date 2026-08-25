"""Contracts and statistics for matched outcome-grounded continuations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Sequence


PANEL_SCHEMA = "metagross-outcome-grounded-panel/v1"
RESULT_SCHEMA = "metagross-outcome-grounded-continuations/v1"


def stable_u64(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_uniform(*parts: Any) -> float:
    return (stable_u64(*parts) >> 11) / 2**53


def weighted_choice(actions: Sequence[tuple[str, float]], uniform: float) -> str:
    if not 0.0 <= uniform < 1.0 or not actions:
        raise ValueError("invalid weighted-choice input")
    total = math.fsum(float(weight) for _, weight in actions)
    if total <= 0 or any(not math.isfinite(float(weight)) or weight < 0 for _, weight in actions):
        raise ValueError("weighted choices require finite nonnegative mass")
    target = uniform * total
    cumulative = 0.0
    for action, weight in actions:
        cumulative += float(weight)
        if target < cumulative:
            return action
    return actions[-1][0]


def bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 10_000) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    rng = random.Random(seed)
    estimates = [math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats)]
    estimates.sort()
    return [estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats) - 1]]
