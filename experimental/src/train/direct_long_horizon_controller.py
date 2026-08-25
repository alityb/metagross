"""Frozen fail-closed gate for matched long-horizon root continuations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from train.outcome_grounded import RESULT_SCHEMA, bootstrap_ci, stable_u64
from train.shallow_search_residual import is_ambiguous


@dataclass(frozen=True)
class DirectControllerConfig:
    schedules: tuple[int, int] = (0, 1)
    worlds_per_schedule: int = 8
    stage_one_rollouts: int = 4
    full_rollouts: int = 16
    minimum_terminal_coverage: float = 0.95
    minimum_paired_coverage: float = 0.90
    minimum_schedule_advantage: float = 0.02
    minimum_cluster_lcb: float = 0.01
    bootstrap_repeats: int = 10_000
    seed: int = 20260815


FROZEN_CONFIG = DirectControllerConfig()


def _mean(rows: Sequence[Mapping[str, Any]], action: str, field: str) -> float:
    return math.fsum(
        float(row["action_statistics"][action][field]) for row in rows
    ) / len(rows)


def shortlist_top_two(
    schedule_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Return the deployment baseline and one challenger without labels."""
    rows = sorted(schedule_rows, key=lambda row: int(row["schedule_id"]))
    if (
        len(rows) != 2
        or {int(row["schedule_id"]) for row in rows} != {0, 1}
        or len({str(row["root_id"]) for row in rows}) != 1
    ):
        raise ValueError("shortlist requires exactly two schedules for one root")
    if not all(is_ambiguous(dict(row["root_statistics"])) for row in rows):
        raise ValueError("shortlist requires ambiguity in both schedules")
    support = set.intersection(
        *(set(row["action_statistics"]) for row in rows)
    )
    if len(support) < 2:
        raise ValueError("shortlist has fewer than two shared actions")
    ranked = sorted(
        support,
        key=lambda action: (
            _mean(rows, action, "visit_mass"),
            _mean(rows, action, "mean_value"),
            action,
        ),
        reverse=True,
    )
    return str(ranked[0]), str(ranked[1])


def _validated_rows(
    rows: Sequence[Mapping[str, Any]], config: DirectControllerConfig
) -> tuple[list[Mapping[str, Any]], str, str, int]:
    ordered = sorted(rows, key=lambda row: int(row["schedule_id"]))
    if (
        len(ordered) != 2
        or {int(row.get("schedule_id", -1)) for row in ordered}
        != set(config.schedules)
        or len({str(row.get("root_id")) for row in ordered}) != 1
        or any(row.get("schema") != RESULT_SCHEMA for row in ordered)
    ):
        raise ValueError("invalid direct-controller schedule bundle")
    baseline = str(ordered[0]["baseline_action"])
    candidates = list(ordered[0]["candidate_actions"])
    if (
        len(candidates) != 2
        or candidates[0] != baseline
        or any(
            row["baseline_action"] != baseline
            or list(row["candidate_actions"]) != candidates
            for row in ordered
        )
    ):
        raise ValueError("bundle is not the frozen top-two action comparison")
    alternative = str(candidates[1])
    rollout_sets = []
    expected_worlds = set(range(config.worlds_per_schedule))
    for row in ordered:
        for action in candidates:
            samples = row["action_outcomes"].get(action)
            if not isinstance(samples, list) or not samples:
                raise ValueError("action is missing continuation outcomes")
            keys = [
                (int(sample["world_index"]), int(sample["rollout"]))
                for sample in samples
            ]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate continuation sample key")
            if {world for world, _ in keys} != expected_worlds:
                raise ValueError("continuation bundle has incomplete world support")
            rollout_set = {rollout for _, rollout in keys}
            if set(keys) != {
                (world, rollout)
                for world in expected_worlds
                for rollout in rollout_set
            }:
                raise ValueError("candidate actions do not share one matched rollout tape")
            rollout_sets.append(rollout_set)
    if len({tuple(sorted(values)) for values in rollout_sets}) != 1:
        raise ValueError("candidate actions do not share one matched rollout tape")
    rollouts = sorted(rollout_sets[0])
    if rollouts not in [
        list(range(config.stage_one_rollouts)),
        list(range(config.full_rollouts)),
    ]:
        raise ValueError("bundle is neither the frozen four- nor sixteen-rollout stage")
    return ordered, baseline, alternative, len(rollouts)


def decide(
    rows: Sequence[Mapping[str, Any]],
    config: DirectControllerConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    """Apply the preregistered matched-outcome rule to one root."""
    ordered, baseline, alternative, rollout_count = _validated_rows(rows, config)
    actions = (baseline, alternative)
    total = terminal = paired = 0
    schedule_deltas: list[float | None] = []
    cluster_deltas: list[float] = []
    half_deltas: list[list[float | None]] = []

    for row in ordered:
        by_action: dict[str, dict[tuple[int, int], float]] = {}
        for action in actions:
            values: dict[tuple[int, int], float] = {}
            samples = row["action_outcomes"][action]
            total += len(samples)
            for sample in samples:
                outcome = sample.get("outcome")
                if outcome is None:
                    continue
                value = float(outcome)
                if not math.isfinite(value) or value not in {0.0, 1.0}:
                    raise ValueError("continuation outcome is not an engine terminal value")
                terminal += 1
                values[(int(sample["world_index"]), int(sample["rollout"]))] = value
            by_action[action] = values
        keys = sorted(set(by_action[baseline]).intersection(by_action[alternative]))
        paired += len(keys)
        deltas = [
            by_action[alternative][key] - by_action[baseline][key] for key in keys
        ]
        schedule_deltas.append(
            math.fsum(deltas) / len(deltas) if deltas else None
        )
        for world in range(config.worlds_per_schedule):
            values = [delta for key, delta in zip(keys, deltas) if key[0] == world]
            if values:
                cluster_deltas.append(math.fsum(values) / len(values))
        if rollout_count == config.full_rollouts:
            halves = []
            for lower, upper in ((0, 8), (8, 16)):
                values = [
                    delta
                    for key, delta in zip(keys, deltas)
                    if lower <= key[1] < upper
                ]
                halves.append(math.fsum(values) / len(values) if values else None)
            half_deltas.append(halves)

    terminal_coverage = terminal / total
    paired_denominator = len(ordered) * config.worlds_per_schedule * rollout_count
    paired_coverage = paired / paired_denominator
    interval = (
        bootstrap_ci(
            cluster_deltas,
            stable_u64(config.seed, ordered[0]["root_id"], alternative) % (2**32),
            repeats=config.bootstrap_repeats,
        )
        if cluster_deltas
        else [None, None]
    )
    checks = {
        "terminal_coverage": terminal_coverage >= config.minimum_terminal_coverage,
        "paired_coverage": paired_coverage >= config.minimum_paired_coverage,
        "schedule_agreement": all(
            value is not None and value > config.minimum_schedule_advantage
            for value in schedule_deltas
        ),
        "cluster_lcb": interval[0] is not None
        and interval[0] > config.minimum_cluster_lcb,
        "half_agreement": rollout_count < config.full_rollouts
        or all(
            value is not None and value > 0.0
            for schedule in half_deltas
            for value in schedule
        ),
    }
    passed = all(checks.values())
    if passed:
        selected, decision = alternative, "override"
    elif rollout_count == config.stage_one_rollouts:
        selected, decision = baseline, "extend_to_16"
    else:
        selected, decision = baseline, "abstain"
    return {
        "selected_action": selected,
        "baseline_action": baseline,
        "alternative_action": alternative,
        "decision": decision,
        "rollouts": rollout_count,
        "terminal_coverage": terminal_coverage,
        "paired_coverage": paired_coverage,
        "schedule_advantages": schedule_deltas,
        "cluster_bootstrap_ci95": interval,
        "half_advantages": half_deltas,
        "checks": checks,
    }
