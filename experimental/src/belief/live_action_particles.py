"""Fail-closed helpers for live action-conditioned particle sampling.

The caller supplies candidate-conditioned likelihoods.  Public-only priors are
not evidence and must never be passed to this module as likelihoods.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence, TypeVar


T = TypeVar("T")


def bounded_candidates(candidates: Sequence[T], limit: int = 32) -> list[T]:
    """Keep the caller's ranked candidate order while bounding endpoint work."""
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    return list(candidates[:limit])


def validated_weights(candidate_ids: Sequence[str], likelihoods: Mapping[str, object] | None) -> list[float] | None:
    """Return usable non-uniform weights, or None for an exact uniform fallback.

    A missing candidate, malformed value, or all-zero result rejects the whole
    update rather than partially applying an uncertain response.
    """
    if likelihoods is None or not candidate_ids:
        return None
    weights: list[float] = []
    for candidate_id in candidate_ids:
        value = likelihoods.get(candidate_id)
        if isinstance(value, bool):
            return None
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(weight) or weight < 0.0:
            return None
        weights.append(weight)
    return weights if sum(weights) > 0.0 else None


def cumulative_tempered_weights(
    action_weights: Sequence[Sequence[float]], temperature: float = 0.5
) -> list[float] | None:
    """Combine independent action evidence without numerical underflow.

    Zero likelihood permanently eliminates a particle. ``temperature < 1``
    tempers policy-model overconfidence; malformed/all-collapsed histories
    fail closed to the caller's uniform fallback.
    """
    if not math.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must be finite and in (0, 1]")
    if not action_weights:
        return None
    width = len(action_weights[0])
    if width == 0 or any(len(weights) != width for weights in action_weights):
        return None
    log_weights = [0.0] * width
    alive = [True] * width
    for weights in action_weights:
        for index, value in enumerate(weights):
            if not math.isfinite(value) or value < 0.0:
                return None
            if value == 0.0:
                alive[index] = False
            elif alive[index]:
                log_weights[index] += math.log(value)
    finite = [temperature * value for value, keep in zip(log_weights, alive) if keep]
    if not finite:
        return None
    offset = max(finite)
    combined = [
        math.exp(temperature * value - offset) if keep else 0.0
        for value, keep in zip(log_weights, alive)
    ]
    return combined if sum(combined) > 0.0 else None


class ActionEvidenceCache:
    """Cache only validated candidate-conditioned action evidence by signature."""

    def __init__(self) -> None:
        self._weights: dict[str, list[float] | None] = {}

    def get(self, signature: str) -> list[float] | None | object:
        return self._weights.get(signature, _MISSING)

    def put(self, signature: str, candidate_ids: Sequence[str], likelihoods: Mapping[str, object] | None) -> list[float] | None:
        weights = validated_weights(candidate_ids, likelihoods)
        self._weights[signature] = weights
        return weights


_MISSING = object()


def request_if_enabled(enabled: bool, request: Callable[[], Mapping[str, object] | None]) -> Mapping[str, object] | None:
    """Avoid all endpoint activity unless the experimental feature is enabled."""
    return request() if enabled else None
