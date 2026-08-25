"""Authorization-gated MAPLE-style multi-state vertical-slice core.

This is deliberately not the archived shared-root RM+ policy extractor.  It
uses deterministic entropic mirror descent: one player policy is updated from
the belief-weighted value of *all* particles every round, while a separate
two-sided opponent policy is updated inside each particle.  Production payoff
cells are forced-root continuation searches; the pure matrix core here makes
the game-theoretic and reproducibility invariants independently testable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from srcs.metagross.known_team_decision_v2 import canonical_json, write_json


SCHEMA = "metagross-maple-vertical-slice/v2"


@dataclass(frozen=True)
class ParticleGame:
    identity: str
    weight: float
    # player action -> opponent action -> player payoff
    payoffs: Mapping[str, Mapping[str, float]]


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def mixture_weights(
    current: Sequence[float], history: Sequence[float], alpha: float
) -> list[float]:
    if len(current) != len(history) or not current:
        raise ValueError("current and history weights must be nonempty and aligned")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    values = [
        (1.0 - alpha) * float(left) + alpha * float(right)
        for left, right in zip(current, history, strict=True)
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("mixture weights must be finite and nonnegative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("mixture weights have zero mass")
    return [value / total for value in values]


def allocate_exact_compute(total_iterations: int, cell_count: int) -> list[int]:
    """Allocate an exact treatment-blind budget over a frozen cell ordering."""
    if total_iterations < cell_count or cell_count <= 0:
        raise ValueError("total iterations must cover every positive cell")
    quotient, remainder = divmod(total_iterations, cell_count)
    allocation = [quotient + int(index < remainder) for index in range(cell_count)]
    assert sum(allocation) == total_iterations
    return allocation


def _validated_particles(
    particles: Sequence[ParticleGame], allowed_actions: set[str] | None
) -> tuple[list[ParticleGame], list[str]]:
    if not particles:
        raise ValueError("at least one particle is required")
    grouped: dict[str, tuple[float, Mapping[str, Mapping[str, float]]]] = {}
    for particle in particles:
        if not particle.identity:
            raise ValueError("particle identity must not be empty")
        if not math.isfinite(particle.weight) or particle.weight < 0.0:
            raise ValueError("particle weights must be finite and nonnegative")
        actions = set(particle.payoffs)
        if not actions:
            raise ValueError("player action support must not be empty")
        for row in particle.payoffs.values():
            if not row or any(not math.isfinite(float(value)) for value in row.values()):
                raise ValueError("payoff rows must be nonempty and finite")
        previous = grouped.get(particle.identity)
        if previous is not None and canonical_json(previous[1]) != canonical_json(particle.payoffs):
            raise ValueError("duplicate particle identity has different payoffs")
        grouped[particle.identity] = (
            particle.weight + (previous[0] if previous else 0.0),
            particle.payoffs,
        )
    ordered = [
        ParticleGame(identity, weight, payoffs)
        for identity, (weight, payoffs) in sorted(grouped.items())
        if weight > 0.0
    ]
    if not ordered:
        raise ValueError("at least one particle must have positive weight")
    shared = set.intersection(*(set(particle.payoffs) for particle in ordered))
    if allowed_actions is not None:
        shared &= set(allowed_actions)
    if not shared:
        raise ValueError("shared request-authorized action support is empty")
    actions = sorted(shared)
    total = math.fsum(particle.weight for particle in ordered)
    ordered = [
        ParticleGame(particle.identity, particle.weight / total, particle.payoffs)
        for particle in ordered
    ]
    return ordered, actions


def solve_multistate_game(
    particles: Sequence[ParticleGame],
    *,
    rounds: int = 20_000,
    learning_rate: float | None = None,
    allowed_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Solve the shared-player/two-sided particle game deterministically."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    particles, actions = _validated_particles(particles, allowed_actions)
    opponents = [sorted(next(iter(particle.payoffs.values())).keys()) for particle in particles]
    for particle, opponent_actions in zip(particles, opponents, strict=True):
        expected = set(opponent_actions)
        if any(set(particle.payoffs[action]) != expected for action in actions):
            raise ValueError("each particle must have a rectangular payoff matrix")
    if learning_rate is None:
        largest_support = max([len(actions), *(len(rows) for rows in opponents)])
        learning_rate = math.sqrt(2.0 * math.log(max(2, largest_support)) / rounds)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")

    player_scores = [0.0] * len(actions)
    opponent_scores = [[0.0] * len(rows) for rows in opponents]
    player_average = [0.0] * len(actions)
    opponent_averages = [[0.0] * len(rows) for rows in opponents]
    player_values = [0.0] * len(actions)

    for _round in range(rounds):
        player_policy = _softmax([learning_rate * value for value in player_scores])
        opponent_policies = [
            _softmax([-learning_rate * value for value in scores])
            for scores in opponent_scores
        ]
        for index, probability in enumerate(player_policy):
            player_average[index] += probability
        for average, policy in zip(opponent_averages, opponent_policies, strict=True):
            for index, probability in enumerate(policy):
                average[index] += probability

        for action_index, action in enumerate(actions):
            player_values[action_index] = math.fsum(
                particle.weight
                * math.fsum(
                    probability * float(particle.payoffs[action][opponent])
                    for opponent, probability in zip(
                        opponent_actions, opponent_policy, strict=True
                    )
                )
                for particle, opponent_actions, opponent_policy in zip(
                    particles, opponents, opponent_policies, strict=True
                )
            )
            player_scores[action_index] += player_values[action_index]

        for particle_index, (particle, opponent_actions) in enumerate(
            zip(particles, opponents, strict=True)
        ):
            for opponent_index, opponent in enumerate(opponent_actions):
                value = math.fsum(
                    probability * float(particle.payoffs[action][opponent])
                    for action, probability in zip(actions, player_policy, strict=True)
                )
                opponent_scores[particle_index][opponent_index] += value

    policy = {
        action: player_average[index] / rounds for index, action in enumerate(actions)
    }
    opponent_policies = {
        particle.identity: {
            opponent: opponent_averages[particle_index][opponent_index] / rounds
            for opponent_index, opponent in enumerate(opponents[particle_index])
        }
        for particle_index, particle in enumerate(particles)
    }
    selected = sorted(policy, key=lambda action: (-policy[action], action))[0]
    cell_count = sum(len(actions) * len(rows) for rows in opponents)
    return {
        "schema": SCHEMA,
        "algorithm": "full-information-entropic-mirror-descent",
        "policy_extraction": "average_shared_player_policy",
        "rounds": rounds,
        "learning_rate": learning_rate,
        "particle_count": len(particles),
        "shared_actions": actions,
        "policy": policy,
        "selected_action": selected,
        "opponent_policies": opponent_policies,
        "payoff_cells": cell_count,
        "belief_weighted_inside_every_round": True,
        "independent_world_votes": False,
    }


def require_phase3_authorization(phase2: Mapping[str, Any]) -> None:
    summary = phase2.get("summary") or {}
    if (
        not summary.get("phase3_authorized")
        or summary.get("decision") != "authorize_phase3_maple_vertical_slice"
    ):
        raise RuntimeError("frozen Phase 2 result does not authorize Phase 3")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    phase2 = json.loads(args.phase2.read_text(encoding="utf-8"))
    require_phase3_authorization(phase2)
    write_json(
        args.output,
        {
            "schema": SCHEMA,
            "status": "authorized_vertical_slice_core_ready",
            "phase2_decision": phase2["summary"]["decision"],
        },
    )


if __name__ == "__main__":
    main()
