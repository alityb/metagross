#!/usr/bin/env python3
"""Strict descriptive stability analysis for frozen teacher root bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from eval.experiment_manifest import validate_manifest  # noqa: E402
from scripts.collect_independent_action_values import validate_action_value_record  # noqa: E402
from scripts.evaluate_teacher_root_bundles import validate_root_evaluation  # noqa: E402
from scripts.teacher_root_bundle import RootBundleError, validate_root_bundle  # noqa: E402


SCHEMA_VERSION = 1
MASS_TOLERANCE = 1e-9
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
METRICS = (
    "jensen_shannon_nats",
    "total_variation",
    "top1_fractional_agreement",
    "top_set_overlap",
    "spearman_rank_correlation",
    "left_entropy_nats",
    "right_entropy_nats",
    "left_top_mass",
    "right_top_mass",
    "left_top_margin",
    "right_top_margin",
)


class AnalysisError(ValueError):
    """A fail-closed root-bundle or output-contract error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _number(value: Any, description: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise AnalysisError(f"{description} must be finite and nonnegative")
    return result


def _policy(entries: Any, description: str) -> dict[str, float]:
    if not isinstance(entries, list) or not entries:
        raise AnalysisError(f"{description} must be a nonempty list")
    policy: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AnalysisError(f"{description} entries must be objects")
        action = entry.get("action")
        if not isinstance(action, str) or not action or action in policy:
            raise AnalysisError(f"{description} contains an invalid or duplicate action")
        policy[action] = _number(entry.get("probability"), f"{description} probability")
    total = math.fsum(policy.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise AnalysisError(f"{description} mass is {total!r}, expected 1")
    return {action: mass / total for action, mass in policy.items()}


def _pairs(entries: Any, description: str) -> dict[str, float]:
    if not isinstance(entries, list) or not entries:
        raise AnalysisError(f"{description} must be a nonempty list")
    result: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise AnalysisError(f"{description} entries must be [action, mass] pairs")
        action, raw_mass = entry
        if not isinstance(action, str) or not action or action in result:
            raise AnalysisError(f"{description} contains an invalid or duplicate action")
        result[action] = _number(raw_mass, f"{description} mass")
    total = math.fsum(result.values())
    if total <= 0.0:
        raise AnalysisError(f"{description} has no positive mass")
    return {action: mass / total for action, mass in result.items()}


def _visit_policy(result: Any, description: str) -> dict[str, float]:
    if not isinstance(result, dict):
        raise AnalysisError(f"{description} must be an object")
    total = result.get("total_visits")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise AnalysisError(f"{description}.total_visits must be a positive integer")
    entries = result.get("side_one")
    if not isinstance(entries, list) or not entries:
        raise AnalysisError(f"{description}.side_one must be nonempty")
    visits: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AnalysisError(f"{description}.side_one entries must be objects")
        action = entry.get("action")
        count = entry.get("visits")
        if not isinstance(action, str) or not action or action in visits:
            raise AnalysisError(f"{description}.side_one has an invalid or duplicate action")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise AnalysisError(f"{description}.side_one visits must be nonnegative integers")
        _number(entry.get("total_score"), f"{description}.side_one total_score", nonnegative=False)
        visits[action] = count
    if sum(visits.values()) != total:
        raise AnalysisError(f"{description}.side_one visits do not sum to total_visits")
    side_two = result.get("side_two")
    if not isinstance(side_two, list) or not side_two:
        raise AnalysisError(f"{description}.side_two must be nonempty")
    side_two_visits = 0
    seen_side_two: set[str] = set()
    for entry in side_two:
        if not isinstance(entry, dict):
            raise AnalysisError(f"{description}.side_two entries must be objects")
        action = entry.get("action")
        count = entry.get("visits")
        if not isinstance(action, str) or not action or action in seen_side_two:
            raise AnalysisError(f"{description}.side_two has an invalid or duplicate action")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise AnalysisError(f"{description}.side_two visits must be nonnegative integers")
        _number(entry.get("total_score"), f"{description}.side_two total_score", nonnegative=False)
        seen_side_two.add(action)
        side_two_visits += count
    if side_two_visits != total:
        raise AnalysisError(f"{description}.side_two visits do not sum to total_visits")
    return {action: count / total for action, count in visits.items()}


def _entropy(policy: Mapping[str, float]) -> float:
    return -math.fsum(mass * math.log(mass) for mass in policy.values() if mass > 0.0)


def _top_set(policy: Mapping[str, float]) -> set[str]:
    top = max(policy.values())
    return {action for action, mass in policy.items() if mass == top}


def _top_margin(policy: Mapping[str, float]) -> float:
    ranked = sorted(policy.values(), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else 1.0


def _ranks(policy: Mapping[str, float], actions: list[str]) -> list[float]:
    ordered = sorted(((policy[action], action) for action in actions), reverse=True)
    ranks: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for _, action in ordered[position:end]:
            ranks[action] = average_rank
        position = end
    return [ranks[action] for action in actions]


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if left == right else 0.0
    return max(-1.0, min(1.0, numerator / (left_scale * right_scale)))


def distribution_metrics(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    if set(left) != set(right) or not left:
        raise AnalysisError("compared policies must have the same nonempty action set")
    actions = sorted(left)
    left_top = _top_set(left)
    right_top = _top_set(right)
    intersection = left_top & right_top
    midpoint = {action: (left[action] + right[action]) / 2.0 for action in actions}
    js = 0.5 * math.fsum(
        left[action] * math.log(left[action] / midpoint[action])
        for action in actions
        if left[action] > 0.0
    ) + 0.5 * math.fsum(
        right[action] * math.log(right[action] / midpoint[action])
        for action in actions
        if right[action] > 0.0
    )
    return {
        "jensen_shannon_nats": max(0.0, js),
        "total_variation": 0.5 * math.fsum(abs(left[action] - right[action]) for action in actions),
        "top1_fractional_agreement": len(intersection) / (len(left_top) * len(right_top)),
        "top_set_overlap": float(bool(intersection)),
        "spearman_rank_correlation": _correlation(_ranks(left, actions), _ranks(right, actions)),
        "left_entropy_nats": _entropy(left),
        "right_entropy_nats": _entropy(right),
        "left_top_mass": max(left.values()),
        "right_top_mass": max(right.values()),
        "left_top_margin": _top_margin(left),
        "right_top_margin": _top_margin(right),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot summarize an empty metric")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean_policy(policies: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not policies:
        raise AnalysisError("cannot average no policies")
    actions = set(policies[0])
    if any(set(policy) != actions for policy in policies[1:]):
        raise AnalysisError("repeat policies disagree on their action set")
    return {
        action: math.fsum(policy[action] for policy in policies) / len(policies)
        for action in sorted(actions)
    }


def _mean_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise AnalysisError("cannot average no metric rows")
    return {name: math.fsum(row[name] for row in rows) / len(rows) for name in METRICS}


def _weighted_percentile(values: Sequence[float], weights: Sequence[float], probability: float) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = math.fsum(weight for _, weight in ordered)
    threshold = probability * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _summary(
    rows: Sequence[Mapping[str, float]],
    *,
    roots: int,
    pairs: int | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"status": "not_evaluable", "reason": "no matched observations", "n_roots": roots}
    effective_weights = list(weights) if weights is not None else [1.0] * len(rows)
    if len(effective_weights) != len(rows) or any(
        not math.isfinite(weight) or weight <= 0 for weight in effective_weights
    ):
        raise AnalysisError("summary weights must be finite, positive, and aligned")
    weight_total = math.fsum(effective_weights)
    result: dict[str, Any] = {
        "status": "descriptive_only",
        "n_roots": roots,
        "weight_sum": weight_total,
        "metrics": {},
    }
    if pairs is not None:
        result["n_repeat_pairs"] = pairs
    for name in METRICS:
        values = [row[name] for row in rows]
        result["metrics"][name] = {
            "mean": math.fsum(value * weight for value, weight in zip(values, effective_weights)) / weight_total,
            "median": _weighted_percentile(values, effective_weights, 0.5),
            "p90": _weighted_percentile(values, effective_weights, 0.9),
        }
    return result


def _expected_treatments(configuration: Mapping[str, Any]) -> tuple[set[str], dict[str, int]]:
    iterations = configuration.get("iterations")
    repeats = configuration.get("repeats")
    multiplier = configuration.get("deep_multiplier")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (iterations, repeats, multiplier)):
        raise AnalysisError("bundle iterations, repeats, and deep multiplier must be positive integers")
    expected = {"U-B", "S-B"}
    budgets = {"U-B": iterations, "S-B": iterations}
    if multiplier > 1:
        deep = f"S-{multiplier}B"
        expected.add(deep)
        budgets[deep] = iterations * multiplier
    return expected, budgets


def _validate_root(bundle: Mapping[str, Any], expected_manifest: str) -> dict[str, Any]:
    try:
        validate_root_bundle(bundle)
    except RootBundleError as exc:
        raise AnalysisError(str(exc)) from exc
    identity = bundle.get("identity")
    configuration = bundle.get("configuration")
    if not isinstance(identity, dict) or not isinstance(configuration, dict):
        raise AnalysisError("bundle identity and configuration must be objects")
    battle_tag = identity.get("battle_tag")
    username = identity.get("username")
    decision_idx = identity.get("decision_idx")
    if not isinstance(battle_tag, str) or not battle_tag or not isinstance(username, str) or not username:
        raise AnalysisError("bundle has invalid battle or player identity")
    if isinstance(decision_idx, bool) or not isinstance(decision_idx, int) or decision_idx < 0:
        raise AnalysisError("bundle has invalid decision index")
    if configuration.get("input_manifest_sha256") != expected_manifest:
        raise AnalysisError("bundle does not link to the supplied frozen input manifest")
    if configuration.get("execution") != "offline":
        raise AnalysisError("stability analysis requires offline treatment bundles")
    if configuration.get("threads") != 1:
        raise AnalysisError("stability analysis requires single-thread treatment searches")
    if configuration.get("primary_side_two_treatment") != "equal_legal_priors":
        raise AnalysisError("primary side-two treatment must be equal legal priors")
    base_seed = configuration.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed < 2**64:
        raise AnalysisError("bundle has an invalid base seed")
    c_puct = _number(configuration.get("c_puct"), "bundle c_puct")
    if c_puct <= 0.0:
        raise AnalysisError("bundle c_puct must be positive")
    source_hash = configuration.get("source_bundle_sha256")
    source_manifest = configuration.get("source_input_manifest_sha256")
    if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
        raise AnalysisError("bundle has invalid source-bundle linkage")
    if not isinstance(source_manifest, str) or not SHA256_RE.fullmatch(source_manifest):
        raise AnalysisError("bundle has invalid source-manifest linkage")
    expected_treatments, budgets = _expected_treatments(configuration)
    if set(bundle.get("aggregate_treatments", {})) != expected_treatments:
        raise AnalysisError("bundle treatment set does not match its configuration")

    repeat_count = int(configuration["repeats"])
    aggregate_policies: dict[str, list[dict[str, float]]] = {}
    for treatment in sorted(expected_treatments):
        aggregate_entries = bundle["aggregate_treatments"][treatment]
        if not isinstance(aggregate_entries, list) or len(aggregate_entries) != repeat_count:
            raise AnalysisError(f"{treatment} aggregate repeats are incomplete")
        policies = []
        for repeat, entry in enumerate(aggregate_entries):
            if not isinstance(entry, dict) or entry.get("repeat") != repeat:
                raise AnalysisError(f"{treatment} aggregate repeat indices are invalid")
            policies.append(_policy(entry.get("side_one_policy"), f"{treatment} aggregate policy"))
        aggregate_policies[treatment] = policies

    recomputed: dict[str, list[dict[str, float]]] = {
        treatment: [dict() for _ in range(repeat_count)] for treatment in expected_treatments
    }
    direct_mass: dict[str, float] = {}
    weight_sum = 0.0
    action_set: set[str] | None = None
    seen_seeds: set[tuple[str, int, int]] = set()
    for expected_index, world in enumerate(bundle["worlds"]):
        if not isinstance(world, dict) or world.get("world_index") != expected_index:
            raise AnalysisError("world indices are invalid")
        weight = _number(world.get("sample_weight"), "world sample weight")
        capture = world.get("capture")
        if not isinstance(capture, dict) or capture.get("world_index") != expected_index:
            raise AnalysisError("world capture index is invalid")
        state = capture.get("sampled_state")
        state_hash = capture.get("state_sha256")
        if not isinstance(state, str) or not state or hashlib.sha256(state.encode()).hexdigest() != state_hash:
            raise AnalysisError("sampled private state is missing or has the wrong hash")
        effective = _pairs(capture.get("effective_player_priors"), "effective player priors")
        current_actions = set(effective)
        if action_set is None:
            action_set = current_actions
        elif current_actions != action_set:
            raise AnalysisError("worlds disagree on the legal player action set")
        equal_one = _pairs(capture.get("equal_side_one_priors"), "equal side-one priors")
        equal_two = _pairs(capture.get("equal_side_two_priors"), "equal side-two priors")
        if set(equal_one) != current_actions or max(equal_one.values()) - min(equal_one.values()) > MASS_TOLERANCE:
            raise AnalysisError("equal side-one priors are not equal on the legal action set")
        if max(equal_two.values()) - min(equal_two.values()) > MASS_TOLERANCE:
            raise AnalysisError("equal side-two priors are not equal")
        treatments = capture.get("treatments")
        if not isinstance(treatments, dict) or set(treatments) != expected_treatments:
            raise AnalysisError("world treatment set is incomplete")
        for action, mass in effective.items():
            direct_mass[action] = direct_mass.get(action, 0.0) + weight * mass
        weight_sum += weight
        for treatment in expected_treatments:
            entries = treatments[treatment]
            if not isinstance(entries, list) or len(entries) != repeat_count:
                raise AnalysisError(f"{treatment} world repeats are incomplete")
            for repeat, entry in enumerate(entries):
                if not isinstance(entry, dict) or entry.get("repeat") != repeat:
                    raise AnalysisError(f"{treatment} world repeat indices are invalid")
                if entry.get("iterations") != budgets[treatment]:
                    raise AnalysisError(f"{treatment} iteration budget is inconsistent")
                seed = entry.get("seed")
                if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
                    raise AnalysisError(f"{treatment} has an invalid seed")
                seed_key = (treatment, repeat, seed)
                if seed_key in seen_seeds:
                    raise AnalysisError("tree seed was reused across sampled worlds")
                seen_seeds.add(seed_key)
                policy = _visit_policy(entry.get("result"), f"{treatment} world result")
                if set(policy) != current_actions:
                    raise AnalysisError(f"{treatment} result action set is inconsistent")
                for action, mass in policy.items():
                    target = recomputed[treatment][repeat]
                    target[action] = target.get(action, 0.0) + weight * mass
    declared_weight = _number(bundle.get("world_weight_sum"), "world weight sum")
    if weight_sum <= 0.0 or not math.isclose(weight_sum, declared_weight, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise AnalysisError("world weights do not match the declared sum")
    direct_policy = {action: mass / weight_sum for action, mass in direct_mass.items()}
    for treatment in expected_treatments:
        for repeat in range(repeat_count):
            recomputed_policy = {
                action: mass / weight_sum for action, mass in recomputed[treatment][repeat].items()
            }
            stored = aggregate_policies[treatment][repeat]
            if set(recomputed_policy) != set(stored) or any(
                not math.isclose(recomputed_policy[action], stored[action], rel_tol=0.0, abs_tol=MASS_TOLERANCE)
                for action in stored
            ):
                raise AnalysisError(f"{treatment} aggregate policy does not match its worlds")
    return {
        "battle_tag": battle_tag,
        "root_key": (battle_tag, username, decision_idx),
        "world_count": len(bundle["worlds"]),
        "repeat_count": repeat_count,
        "treatments": aggregate_policies,
        "direct_policy": direct_policy,
        "schedules": [
            {
                "schedule_id": 0,
                "treatments": aggregate_policies,
                "direct_policy": direct_policy,
            }
        ],
        "sampling": None,
        "configuration": configuration,
    }


def _validate_scheduled_root(
    evaluation: Mapping[str, Any], expected_manifest: str
) -> dict[str, Any]:
    try:
        validate_root_evaluation(evaluation)
    except RootBundleError as exc:
        raise AnalysisError(str(exc)) from exc
    identity = evaluation.get("identity")
    configuration = evaluation.get("configuration")
    if not isinstance(identity, dict) or not isinstance(configuration, dict):
        raise AnalysisError("scheduled evaluation identity and configuration are invalid")
    if configuration.get("input_manifest_sha256") != expected_manifest:
        raise AnalysisError("scheduled evaluation does not link to the supplied manifest")
    battle_tag = identity.get("battle_tag")
    username = identity.get("username")
    decision_idx = identity.get("decision_idx")
    if not isinstance(battle_tag, str) or not battle_tag or not isinstance(username, str) or not username:
        raise AnalysisError("scheduled evaluation has invalid root identity")
    if isinstance(decision_idx, bool) or not isinstance(decision_idx, int) or decision_idx < 0:
        raise AnalysisError("scheduled evaluation has invalid decision index")
    expected_treatments, budgets = _expected_treatments(configuration)
    repeat_count = int(configuration["repeats"])
    validated_schedules = []
    total_worlds = 0
    for schedule_id, schedule in enumerate(evaluation["schedules"]):
        aggregates = schedule.get("aggregate_treatments")
        if not isinstance(aggregates, dict) or set(aggregates) != expected_treatments:
            raise AnalysisError("schedule treatment aggregates are incomplete")
        policies = {}
        for treatment in sorted(expected_treatments):
            entries = aggregates[treatment]
            if not isinstance(entries, list) or len(entries) != repeat_count:
                raise AnalysisError("schedule treatment repeats are incomplete")
            policies[treatment] = [
                _policy(entry.get("side_one_policy"), f"{treatment} schedule policy")
                for repeat, entry in enumerate(entries)
                if entry.get("repeat") == repeat
            ]
            if len(policies[treatment]) != repeat_count:
                raise AnalysisError("schedule repeat indices are invalid")
        worlds = schedule.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise AnalysisError("scheduled evaluation has no worlds")
        direct_mass: dict[str, float] = {}
        weight_sum = 0.0
        for world_index, world in enumerate(worlds):
            if world.get("world_index") != world_index:
                raise AnalysisError("scheduled world indices are invalid")
            state = world.get("sampled_state")
            if not isinstance(state, str) or hashlib.sha256(state.encode()).hexdigest() != world.get("state_sha256"):
                raise AnalysisError("scheduled world state hash is invalid")
            weight = _number(world.get("sample_weight"), "scheduled world weight")
            effective = _pairs(world.get("effective_player_priors"), "effective player priors")
            for action, mass in effective.items():
                direct_mass[action] = direct_mass.get(action, 0.0) + weight * mass
            weight_sum += weight
            treatments = world.get("treatments")
            if not isinstance(treatments, dict) or set(treatments) != expected_treatments:
                raise AnalysisError("scheduled world treatments are incomplete")
            for treatment in expected_treatments:
                entries = treatments[treatment]
                if not isinstance(entries, list) or len(entries) != repeat_count:
                    raise AnalysisError("scheduled world repeats are incomplete")
                for repeat, entry in enumerate(entries):
                    if entry.get("repeat") != repeat or entry.get("iterations") != budgets[treatment]:
                        raise AnalysisError("scheduled world repeat budget is invalid")
                    _visit_policy(entry.get("result"), f"{treatment} scheduled world result")
        if weight_sum <= 0:
            raise AnalysisError("scheduled world weights have no positive mass")
        validated_schedules.append(
            {
                "schedule_id": schedule_id,
                "treatments": policies,
                "direct_policy": {
                    action: mass / weight_sum for action, mass in direct_mass.items()
                },
            }
        )
        total_worlds += len(worlds)
    root_treatments = {
        treatment: [
            _mean_policy(
                [schedule["treatments"][treatment][repeat] for schedule in validated_schedules]
            )
            for repeat in range(repeat_count)
        ]
        for treatment in sorted(expected_treatments)
    }
    sampling = evaluation.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise AnalysisError("scheduled evaluation sampling metadata is invalid")
        post_weight = _number(
            sampling.get("poststratification_weight"),
            "poststratification weight",
        )
        inclusion = _number(
            sampling.get("inclusion_probability"), "inclusion probability"
        )
        if post_weight <= 0 or not 0 < inclusion <= 1:
            raise AnalysisError("scheduled evaluation sampling weights are invalid")
    return {
        "battle_tag": battle_tag,
        "root_key": (battle_tag, username, decision_idx),
        "world_count": total_worlds,
        "repeat_count": repeat_count,
        "treatments": root_treatments,
        "direct_policy": _mean_policy(
            [schedule["direct_policy"] for schedule in validated_schedules]
        ),
        "schedules": validated_schedules,
        "sampling": sampling,
        "source_capture_sha256": configuration.get("source_capture_sha256"),
        "configuration": configuration,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                AnalysisError(f"non-finite manifest constant {constant}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read frozen manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError("frozen manifest must be an object")
    try:
        validate_manifest(value)
    except ValueError as exc:
        raise AnalysisError(f"invalid frozen manifest: {exc}") from exc
    if value.get("manifest_type") != "experiment_input":
        raise AnalysisError("analysis requires an experiment_input manifest")
    return value


def _load_independent_values(path: Path) -> tuple[dict[tuple[str, str, int], dict[str, Any]], str]:
    payload = path.read_bytes()
    records = {}
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
            validate_action_value_record(record)
        except Exception as exc:
            raise AnalysisError(f"{path}:{line_number}: invalid independent values: {exc}") from exc
        identity = record.get("root_identity")
        if not isinstance(identity, dict):
            raise AnalysisError("independent values are missing root identity")
        key = (identity.get("battle_tag"), identity.get("username"), identity.get("decision_idx"))
        if key in records:
            raise AnalysisError("duplicate independent-value root identity")
        records[key] = record
    if not records:
        raise AnalysisError("independent-value input contains no records")
    return records, hashlib.sha256(payload).hexdigest()


def analyze(
    input_path: Path,
    manifest_path: Path,
    independent_values_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    manifest_hash = str(manifest["manifest_sha256"])
    input_hash = hashlib.sha256()
    roots = []
    seen: set[tuple[str, str, int]] = set()
    try:
        source = input_path.open("rb")
    except OSError as exc:
        raise AnalysisError(f"cannot read root bundles {input_path}: {exc}") from exc
    with source:
        for line_number, raw_line in enumerate(source, start=1):
            input_hash.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                bundle = json.loads(
                    raw_line,
                    parse_constant=lambda constant: (_ for _ in ()).throw(
                        AnalysisError(f"non-finite JSON constant {constant}")
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, AnalysisError) as exc:
                raise AnalysisError(f"{input_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(bundle, dict):
                raise AnalysisError(f"{input_path}:{line_number}: record must be an object")
            try:
                root = (
                    _validate_scheduled_root(bundle, manifest_hash)
                    if bundle.get("record_type") == "teacher_root_evaluation"
                    else _validate_root(bundle, manifest_hash)
                )
            except AnalysisError as exc:
                raise AnalysisError(f"{input_path}:{line_number}: {exc}") from exc
            if root["root_key"] in seen:
                raise AnalysisError(f"{input_path}:{line_number}: duplicate root identity")
            seen.add(root["root_key"])
            roots.append(root)
    if not roots:
        raise AnalysisError("root-bundle input contains no records")

    reference = roots[0]["configuration"]
    stable_fields = (
        "iterations", "repeats", "deep_multiplier", "base_seed", "c_puct",
        "input_manifest_sha256", "source_input_manifest_sha256",
        "primary_side_two_treatment", "threads", "execution",
    )
    if any(any(root["configuration"].get(field) != reference.get(field) for field in stable_fields) for root in roots[1:]):
        raise AnalysisError("input mixes incompatible treatment configurations")

    treatments = sorted(roots[0]["treatments"])
    root_weights = [
        float(root["sampling"]["poststratification_weight"])
        if root["sampling"] is not None
        else 1.0
        for root in roots
    ]
    repeat_rows: dict[str, list[dict[str, float]]] = {treatment: [] for treatment in treatments}
    repeat_weights: dict[str, list[float]] = {treatment: [] for treatment in treatments}
    repeat_pairs = {treatment: 0 for treatment in treatments}
    for root_index, root in enumerate(roots):
        for treatment in treatments:
            pairs = []
            for schedule in root["schedules"]:
                pairs.extend(
                    distribution_metrics(left, right)
                    for left, right in combinations(schedule["treatments"][treatment], 2)
                )
            if pairs:
                repeat_rows[treatment].append(_mean_rows(pairs))
                repeat_weights[treatment].append(root_weights[root_index])
                repeat_pairs[treatment] += len(pairs)

    schedule_rows: dict[str, list[dict[str, float]]] = {treatment: [] for treatment in treatments}
    schedule_weights: dict[str, list[float]] = {treatment: [] for treatment in treatments}
    schedule_pairs = {treatment: 0 for treatment in treatments}
    for root_index, root in enumerate(roots):
        for treatment in treatments:
            schedule_policies = [
                _mean_policy(schedule["treatments"][treatment])
                for schedule in root["schedules"]
            ]
            pairs = [
                distribution_metrics(left, right)
                for left, right in combinations(schedule_policies, 2)
            ]
            if pairs:
                schedule_rows[treatment].append(_mean_rows(pairs))
                schedule_weights[treatment].append(root_weights[root_index])
                schedule_pairs[treatment] += len(pairs)

    mean_policies = {
        (index, treatment): _mean_policy(root["treatments"][treatment])
        for index, root in enumerate(roots)
        for treatment in treatments
    }
    independent_value_report: dict[str, Any]
    independent_values_sha256 = None
    if independent_values_path is not None:
        value_records, independent_values_sha256 = _load_independent_values(
            independent_values_path
        )
        root_value_rows = []
        for index, root in enumerate(roots):
            record = value_records.get(root["root_key"])
            if record is None:
                raise AnalysisError("independent values are missing a matched root")
            if root.get("source_capture_sha256") and (
                record["source_linkage"].get("capture_sha256")
                != root["source_capture_sha256"]
            ):
                raise AnalysisError("independent values do not match the evaluated capture")
            q = {entry["action"]: float(entry["q"]) for entry in record["actions"]}
            policies = {"P": root["direct_policy"]}
            policies.update(
                {
                    treatment: mean_policies[index, treatment]
                    for treatment in treatments
                }
            )
            if any(set(policy) != set(q) for policy in policies.values()):
                raise AnalysisError("independent values do not cover every policy action")
            values = {
                treatment: math.fsum(mass * q[action] for action, mass in policy.items())
                for treatment, policy in policies.items()
            }
            best_q = max(q.values())
            root_value_rows.append(
                {
                    "values": values,
                    "regrets": {name: best_q - value for name, value in values.items()},
                }
            )
        weight_total = math.fsum(root_weights)
        treatment_values = {
            name: math.fsum(row["values"][name] * weight for row, weight in zip(root_value_rows, root_weights)) / weight_total
            for name in ["P", *treatments]
        }
        treatment_regrets = {
            name: math.fsum(row["regrets"][name] * weight for row, weight in zip(root_value_rows, root_weights)) / weight_total
            for name in ["P", *treatments]
        }
        contrasts = {
            f"{left}_minus_{right}": treatment_values[left] - treatment_values[right]
            for left, right in (("S-B", "P"), ("U-B", "P"))
        }
        for treatment in treatments:
            if treatment.startswith("S-") and treatment not in {"S-B"}:
                contrasts[f"{treatment}_minus_S-B"] = treatment_values[treatment] - treatment_values["S-B"]
        independent_value_report = {
            "status": "development_uniform_legal_continuation",
            "estimand": "one-decision value under uniform-legal opponent and continuation policies",
            "r1_continuation_value": False,
            "n_roots": len(roots),
            "weighted_treatment_values": treatment_values,
            "weighted_treatment_regrets": treatment_regrets,
            "weighted_contrasts": contrasts,
        }
    else:
        independent_value_report = {
            "status": "not_available",
            "reason": "no independent action-value artifact; MCTS total_score is not used as evidence",
        }
    comparisons: dict[str, list[dict[str, float]]] = {}
    for left, right in combinations(["P", *treatments], 2):
        name = f"{left}_vs_{right}"
        comparisons[name] = []
        for index, root in enumerate(roots):
            left_policy = root["direct_policy"] if left == "P" else mean_policies[index, left]
            right_policy = root["direct_policy"] if right == "P" else mean_policies[index, right]
            comparisons[name].append(distribution_metrics(left_policy, right_policy))

    repeat_stability = {
        treatment: _summary(
            repeat_rows[treatment],
            roots=len(repeat_rows[treatment]),
            pairs=repeat_pairs[treatment],
            weights=repeat_weights[treatment],
        )
        for treatment in treatments
    }
    observed_screens = {}
    for treatment, summary in repeat_stability.items():
        if summary["status"] != "descriptive_only":
            observed_screens[treatment] = {"status": "not_evaluable", "reason": summary["reason"]}
            continue
        metrics = summary["metrics"]
        observed_screens[treatment] = {
            "status": "observed_only_no_confidence_bound",
            "median_js_le_0_05": metrics["jensen_shannon_nats"]["median"] <= 0.05,
            "p90_js_le_0_15": metrics["jensen_shannon_nats"]["p90"] <= 0.15,
            "top_set_overlap_ge_0_80": metrics["top_set_overlap"]["mean"] >= 0.80,
        }

    report: dict[str, Any] = {
        "analysis_schema_version": SCHEMA_VERSION,
        "analysis_mode": "descriptive_conditional_stability",
        "claim_status": "descriptive_only",
        "inputs": {
            "root_bundle_filename": input_path.name,
            "root_bundle_sha256": input_hash.hexdigest(),
            "frozen_manifest_filename": manifest_path.name,
            "frozen_manifest_sha256": manifest_hash,
            "independent_values_sha256": independent_values_sha256,
        },
        "privacy": {
            "sampled_private_states_read": True,
            "sampled_private_states_emitted": False,
            "root_or_player_identifiers_emitted": False,
        },
        "counts": {
            "source_battles": len({root["battle_tag"] for root in roots}),
            "roots": len(roots),
            "sampled_worlds": sum(root["world_count"] for root in roots),
            "determinization_schedules": sum(len(root["schedules"]) for root in roots),
            "repeats_per_treatment": int(reference["repeats"]),
            "repeat_pairs": repeat_pairs,
            "schedule_pairs": schedule_pairs,
            "poststratification_weight_sum": math.fsum(root_weights),
            "root_effective_sample_size": (
                math.fsum(root_weights) ** 2
                / math.fsum(weight * weight for weight in root_weights)
            ),
        },
        "treatment_configuration": {
            field: reference[field]
            for field in stable_fields
            if field not in {"input_manifest_sha256", "source_input_manifest_sha256"}
        },
        "metric_definitions": {
            "jensen_shannon_nats": "symmetric Jensen-Shannon divergence using natural logarithms",
            "total_variation": "half the L1 distance between action distributions",
            "top1_fractional_agreement": "uniform tie-breaking agreement: intersection/(left ties * right ties)",
            "top_set_overlap": "one when top-action sets intersect, else zero",
            "spearman_rank_correlation": "Pearson correlation of average tie ranks; constant equal ranks map to one",
            "repeat_reduction": "all repeat-pair metrics are averaged within root before root-level summaries",
            "budget_policy": "each budget policy is the arithmetic mean of its repeat policies on the same root",
            "direct_policy_P": "world-weighted effective recorded player prior",
        },
        "repeat_stability": repeat_stability,
        "schedule_stability": {
            treatment: _summary(
                schedule_rows[treatment],
                roots=len(schedule_rows[treatment]),
                pairs=schedule_pairs[treatment],
                weights=schedule_weights[treatment],
            )
            for treatment in treatments
        },
        "matched_distribution_comparisons": {
            name: _summary(rows, roots=len(roots), weights=root_weights)
            for name, rows in sorted(comparisons.items())
        },
        "variance_components": {
            "tree_randomness_conditional_on_frozen_worlds": "descriptively_estimable",
            "determinization_schedule_variance": (
                "descriptively_estimable"
                if all(len(root["schedules"]) >= 2 for root in roots)
                else "not_estimable: fewer than two schedules for at least one root"
            ),
            "continuation_rollout_uncertainty": "not_available: no independent-value artifact",
        },
        "independent_value": independent_value_report,
        "gates": {
            "observed_distribution_screens": observed_screens,
            "formal_stability_gate": {
                "status": "not_evaluable",
                "reason": "requires frozen final roots and simultaneous battle-clustered one-sided confidence bounds",
            },
            "value_regret_gate": {
                "status": "not_evaluable",
                "reason": (
                    "development uniform-continuation values lack formal rollout depth and clustered bounds"
                    if independent_values_path is not None
                    else "requires independent action values for every legal action"
                ),
            },
            "teacher_qualification": {
                "status": "not_evaluable",
                "reason": "requires independent root values, full-game strength, and transfer evidence",
            },
        },
        "limitations": [
            "Development or smoke roots do not support final-root inference.",
            "Multiple worlds alone do not estimate schedule variance; at least two explicit schedules are required.",
            "Distributional stability does not establish action quality or value equivalence.",
            "Formal simultaneous source-battle-clustered bounds require a frozen adequately powered panel.",
            "Uniform-legal continuation values are a development estimand and are not r1-continuation values.",
        ],
    }
    report["analysis_sha256"] = hashlib.sha256(_canonical_json(report).encode("ascii")).hexdigest()
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    independent_values_available = (
        report["independent_value"]["status"] != "not_available"
    )
    lines = [
        "# Teacher Root Stability Analysis",
        "",
        "**Claim status: descriptive only.** This report does not qualify a teacher or establish action quality.",
        "",
        f"- Roots: {report['counts']['roots']}",
        f"- Source battles: {report['counts']['source_battles']}",
        f"- Sampled worlds: {report['counts']['sampled_worlds']}",
        f"- Root-bundle SHA-256: `{report['inputs']['root_bundle_sha256']}`",
        f"- Frozen manifest SHA-256: `{report['inputs']['frozen_manifest_sha256']}`",
        "",
        "## Repeat Stability",
        "",
        "| Treatment | Root summaries | Repeat pairs | Median JS | P90 JS | Top-set overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for treatment, summary in report["repeat_stability"].items():
        if summary["status"] != "descriptive_only":
            lines.append(f"| {treatment} | 0 | 0 | n/a | n/a | n/a |")
            continue
        metrics = summary["metrics"]
        lines.append(
            f"| {treatment} | {summary['n_roots']} | {summary['n_repeat_pairs']} | "
            f"{metrics['jensen_shannon_nats']['median']:.6f} | "
            f"{metrics['jensen_shannon_nats']['p90']:.6f} | "
            f"{metrics['top_set_overlap']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Status",
            "",
            "- Formal stability gate: not evaluable without clustered one-sided confidence bounds.",
            (
                "- Value-regret gate: not evaluable from the development uniform-continuation estimand."
                if independent_values_available
                else "- Value-regret gate: not evaluable without independent action values."
            ),
            "- Teacher qualification: not evaluable without full-game and transfer evidence.",
            "",
            "MCTS `total_score` values were validated structurally but were not used as independent evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_private_atomic(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise AnalysisError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise AnalysisError(f"output already exists: {path}") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_outputs(
    report: Mapping[str, Any], output_json: Path, output_markdown: Path, *, force: bool
) -> None:
    resolved = [output_json.resolve(), output_markdown.resolve()]
    if resolved[0] == resolved[1]:
        raise AnalysisError("JSON and Markdown outputs must differ")
    if not force:
        for path in (output_json, output_markdown):
            if os.path.lexists(path):
                raise AnalysisError(f"output already exists: {path}")
    json_payload = (_canonical_json(dict(report)) + "\n").encode("ascii")
    markdown_payload = render_markdown(report).encode("ascii")
    _write_private_atomic(output_json, json_payload, force=force)
    try:
        _write_private_atomic(output_markdown, markdown_payload, force=force)
    except Exception:
        if not force:
            output_json.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--independent-values", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = {args.input.resolve(), args.input_manifest.resolve()}
    if args.independent_values is not None:
        inputs.add(args.independent_values.resolve())
    outputs = {args.output_json.resolve(), args.output_markdown.resolve()}
    if inputs & outputs:
        raise AnalysisError("an output path collides with an input path")
    report = analyze(args.input, args.input_manifest, args.independent_values)
    write_outputs(report, args.output_json, args.output_markdown, force=args.force)
    print(json.dumps({"analysis_sha256": report["analysis_sha256"], "roots": report["counts"]["roots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
