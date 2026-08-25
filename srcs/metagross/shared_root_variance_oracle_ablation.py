#!/usr/bin/env python3
"""Evaluate a multi-seed averaged shared-root payoff oracle offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    shared_root_result_payload,
)
from srcs.metagross.shared_root_paired_verifier_experiment import (
    _independent_search_action,
    _teacher_mass,
)


SEEDS = (8675309, 8675310, 8675311)
CONTINUATION_ITERATIONS = 32
RM_ITERATIONS = 10_000
PRIOR_STRENGTH = 1.0
SEVERE_DELTA = -0.20
OPPONENT_UNIFORM_MIX = 0.25
OPPONENT_OBJECTIVES = (
    "s_b_smoothed",
    "s_b_raw",
    "s_4b_smoothed_reference",
    "captured_prior_smoothed",
    "uniform",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _identity_key(identity: dict) -> str:
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)


def _normalized_or_uniform(regrets: np.ndarray) -> np.ndarray:
    total = float(regrets.sum())
    if total > 0 and math.isfinite(total):
        return regrets / total
    return np.full(len(regrets), 1.0 / len(regrets), dtype=np.float64)


def solve_rm_plus(
    weights: list[float],
    payoffs: list[list[list[float]]],
    iterations: int,
    player_prior: list[float] | None,
    opponent_priors: list[list[float] | None],
    prior_strength: float,
) -> dict[str, object]:
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in payoffs]
    weight_vector = np.asarray(weights, dtype=np.float64)
    action_count = matrices[0].shape[0]
    player_regrets = (
        np.asarray(player_prior, dtype=np.float64) * prior_strength
        if player_prior is not None
        else np.zeros(action_count, dtype=np.float64)
    )
    opponent_regrets = [
        (
            np.asarray(prior, dtype=np.float64) * prior_strength
            if prior is not None
            else np.zeros(matrix.shape[1], dtype=np.float64)
        )
        for matrix, prior in zip(matrices, opponent_priors, strict=True)
    ]
    player_sum = np.zeros(action_count, dtype=np.float64)
    opponent_sums = [np.zeros_like(regrets) for regrets in opponent_regrets]
    for _ in range(iterations):
        player = _normalized_or_uniform(player_regrets)
        opponents = [_normalized_or_uniform(regrets) for regrets in opponent_regrets]
        player_sum += player
        for total, policy in zip(opponent_sums, opponents, strict=True):
            total += policy
        own_values = np.zeros(action_count, dtype=np.float64)
        for weight, matrix, opponent in zip(
            weight_vector, matrices, opponents, strict=True
        ):
            own_values += weight * (matrix @ opponent)
        expected = float(player @ own_values)
        player_regrets = np.maximum(player_regrets + own_values - expected, 0.0)
        for index, (matrix, opponent) in enumerate(
            zip(matrices, opponents, strict=True)
        ):
            if weight_vector[index] == 0.0:
                continue
            opponent_values = player @ matrix
            local_expected = float(opponent @ opponent_values)
            opponent_regrets[index] = np.maximum(
                opponent_regrets[index] + local_expected - opponent_values, 0.0
            )

    def average(total: np.ndarray) -> np.ndarray:
        result = total / iterations
        result /= result.sum()
        result[np.abs(result) < 1e-9] = 0.0
        result /= result.sum()
        return result

    player_policy = average(player_sum)
    opponent_policies = [average(total) for total in opponent_sums]
    counterfactual = np.zeros(action_count, dtype=np.float64)
    for weight, matrix, opponent in zip(
        weight_vector, matrices, opponent_policies, strict=True
    ):
        counterfactual += weight * (matrix @ opponent)
    return {
        "player_policy": player_policy.tolist(),
        "opponent_policies": [policy.tolist() for policy in opponent_policies],
        "counterfactual_values": counterfactual.tolist(),
    }


def _runtime_identity() -> dict:
    native = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    root = Path(__file__).resolve().parent
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "engine_contract": ENGINE_CONTRACT,
        "engine_source_sha256": ENGINE_SOURCE_SHA256,
        "engine_native_sha256": _sha256(native),
        "dependencies": {
            "mcts_contract.py": _sha256(root / "mcts_contract.py"),
            "run_foul_play.py": _sha256(root / "run_foul_play.py"),
            "shared_root_paired_verifier_experiment.py": _sha256(
                root / "shared_root_paired_verifier_experiment.py"
            ),
        },
    }


def _validate_protocol(
    protocol_path: Path, evaluation_path: Path, enrichment_path: Path
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "evaluation_sha256": _sha256(evaluation_path),
            "prior_enrichment_sha256": _sha256(enrichment_path),
        }
        or protocol.get("runtime") != _runtime_identity()
        or protocol.get("configuration")
        != {
            "seeds": list(SEEDS),
            "continuation_iterations_per_seed": CONTINUATION_ITERATIONS,
            "rm_iterations": RM_ITERATIONS,
            "prior_strength": PRIOR_STRENGTH,
            "payoff_aggregation": "cellwise_arithmetic_mean_in_seed_order",
            "baseline": "independent_s_b_search_argmax",
            "severe_regression_delta": SEVERE_DELTA,
            "opponent_objectives": list(OPPONENT_OBJECTIVES),
            "opponent_uniform_mix": OPPONENT_UNIFORM_MIX,
        }
    ):
        raise ValueError("variance-oracle ablation differs from its frozen protocol")


def _capture_key(particle: dict) -> str:
    return _canonical_sha256(
        {
            "state": particle["state"],
            "opponent_action_support": particle["opponent_action_support"],
            "normalized_opponent_prior": particle["normalized_opponent_prior"],
            "source_particles": particle["source_particles"],
        }
    )


def _policy_argmax(
    actions: list[str], probabilities: list[float], counterfactual_values: list[float]
) -> str:
    return sorted(
        range(len(actions)),
        key=lambda index: (
            -probabilities[index],
            -counterfactual_values[index],
            actions[index],
        ),
    )[0]


def _snapshot_opponent_policy(snapshot: dict, support: list[str]) -> list[float]:
    visits = {
        str(row["action"]).lower(): float(row["visits"])
        for row in snapshot["side_two"]
    }
    values = [visits.get(action, 0.0) for action in support]
    total = math.fsum(values)
    if total <= 0:
        return [1.0 / len(support) for _action in support]
    return [value / total for value in values]


def _canonical_opponent_policy(
    worlds: list[dict], particle: dict, treatment: str, support: list[str]
) -> list[float]:
    weighted = [0.0 for _action in support]
    weight_total = math.fsum(
        float(source["input_weight"]) for source in particle["source_particles"]
    )
    for source in particle["source_particles"]:
        world = worlds[source["input_index"]]
        policies = [
            _snapshot_opponent_policy(repeat["result"], support)
            for repeat in world["treatments"][treatment]
        ]
        policy = [
            math.fsum(values) / len(values) for values in zip(*policies, strict=True)
        ]
        source_weight = float(source["input_weight"]) / weight_total
        for index, value in enumerate(policy):
            weighted[index] += source_weight * value
    return weighted


def _smoothed(policy: list[float], uniform_mix: float) -> list[float]:
    uniform = 1.0 / len(policy)
    return [
        (1.0 - uniform_mix) * probability + uniform_mix * uniform
        for probability in policy
    ]


def _expected_action(
    actions: list[str],
    weights: list[float],
    matrices: list[list[list[float]]],
    policies: list[list[float]],
) -> tuple[str, list[float]]:
    scores = [0.0 for _action in actions]
    for weight, matrix, policy in zip(weights, matrices, policies, strict=True):
        for action_index, row in enumerate(matrix):
            scores[action_index] += weight * math.fsum(
                payoff * probability
                for payoff, probability in zip(row, policy, strict=True)
            )
    selected = sorted(
        range(len(actions)), key=lambda index: (-scores[index], actions[index])
    )[0]
    return actions[selected], scores


def analyze(
    evaluation_path: Path, enrichment_path: Path, protocol_path: Path
) -> dict[str, object]:
    _validate_protocol(protocol_path, evaluation_path, enrichment_path)
    import poke_engine

    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))
    enriched_roots = {_identity_key(root["identity"]): root for root in enrichment["roots"]}
    roots = []
    native_replay_checks = 0
    native_argmax_matches = 0
    for evaluation in evaluations:
        enriched = enriched_roots[_identity_key(evaluation["identity"])]
        raw_opponent_prior = [
            tuple(row) for row in enriched["source_captured_raw_opponent_priors"]
        ]
        schedules = []
        for schedule in evaluation["schedules"]:
            worlds = schedule["worlds"]
            source_weights = [float(world["sample_weight"]) for world in worlds]
            weight_total = math.fsum(source_weights)
            weights = [weight / weight_total for weight in source_weights]
            player_prior = worlds[0]["effective_player_priors"] or None
            captures = []
            native_results = []
            solve_latencies = []
            for seed in SEEDS:
                started = time.perf_counter_ns()
                native = poke_engine.shared_information_set_root_search(
                    states=[
                        poke_engine.State.from_string(world["sampled_state"])
                        for world in worlds
                    ],
                    particle_weights=weights,
                    iterations=RM_ITERATIONS,
                    continuation_iterations=CONTINUATION_ITERATIONS,
                    seed=seed,
                    prior_strength=PRIOR_STRENGTH,
                    s1_prior=player_prior,
                    s2_priors=[raw_opponent_prior for _world in worlds],
                )
                solve_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                result = shared_root_result_payload(native)
                native_results.append(result)
                captures.append(result["replay_capture"])
            own_support = captures[0]["own_action_support"]
            player_prior_vector = captures[0]["normalized_player_prior"]
            capture_maps = [
                {_capture_key(particle): particle for particle in capture["canonical_particles"]}
                for capture in captures
            ]
            if any(
                capture["own_action_support"] != own_support
                or capture["normalized_player_prior"] != player_prior_vector
                or capture_map.keys() != capture_maps[0].keys()
                for capture, capture_map in zip(captures, capture_maps, strict=True)
            ):
                raise ValueError("multi-seed native captures do not share a canonical game")
            keys = list(capture_maps[0])
            averaged_matrices = []
            opponent_priors = []
            canonical_weights = []
            canonical_particles = []
            for key in keys:
                particles = [capture_map[key] for capture_map in capture_maps]
                matrices = [np.asarray(particle["payoff_matrix"]) for particle in particles]
                averaged_matrices.append(np.mean(matrices, axis=0).tolist())
                opponent_priors.append(particles[0]["normalized_opponent_prior"])
                canonical_weights.append(float(particles[0]["normalized_weight"]))
                canonical_particles.append(particles[0])
            replay_started = time.perf_counter_ns()
            solved = solve_rm_plus(
                canonical_weights,
                averaged_matrices,
                RM_ITERATIONS,
                player_prior_vector,
                opponent_priors,
                PRIOR_STRENGTH,
            )
            replay_ms = (time.perf_counter_ns() - replay_started) / 1_000_000
            for result, capture_map in zip(native_results, capture_maps, strict=True):
                replay = solve_rm_plus(
                    canonical_weights,
                    [capture_map[key]["payoff_matrix"] for key in keys],
                    RM_ITERATIONS,
                    player_prior_vector,
                    [capture_map[key]["normalized_opponent_prior"] for key in keys],
                    PRIOR_STRENGTH,
                )
                replay_action = own_support[
                    _policy_argmax(
                        own_support,
                        replay["player_policy"],
                        replay["counterfactual_values"],
                    )
                ]
                native_action = sorted(
                    result["policy"],
                    key=lambda row: (
                        -row["probability"],
                        -row["counterfactual_value"],
                        row["action"],
                    ),
                )[0]["action"]
                native_replay_checks += 1
                native_argmax_matches += replay_action == native_action
            candidate_index = _policy_argmax(
                own_support,
                solved["player_policy"],
                solved["counterfactual_values"],
            )
            candidate = own_support[candidate_index]
            baseline = _independent_search_action(schedule)
            baseline_mass = _teacher_mass(schedule, baseline)
            candidate_mass = _teacher_mass(schedule, candidate)
            objective_policies = {name: [] for name in OPPONENT_OBJECTIVES}
            for particle in canonical_particles:
                support = particle["opponent_action_support"]
                uniform = [1.0 / len(support) for _action in support]
                s_b = _canonical_opponent_policy(worlds, particle, "S-B", support)
                s_4b = _canonical_opponent_policy(worlds, particle, "S-4B", support)
                captured = particle["normalized_opponent_prior"] or uniform
                objective_policies["s_b_smoothed"].append(
                    _smoothed(s_b, OPPONENT_UNIFORM_MIX)
                )
                objective_policies["s_b_raw"].append(s_b)
                objective_policies["s_4b_smoothed_reference"].append(
                    _smoothed(s_4b, OPPONENT_UNIFORM_MIX)
                )
                objective_policies["captured_prior_smoothed"].append(
                    _smoothed(captured, OPPONENT_UNIFORM_MIX)
                )
                objective_policies["uniform"].append(uniform)
            objectives = {}
            for name, policies in objective_policies.items():
                objective_action, scores = _expected_action(
                    own_support, canonical_weights, averaged_matrices, policies
                )
                objective_mass = _teacher_mass(schedule, objective_action)
                objectives[name] = {
                    "action": objective_action,
                    "scores": dict(zip(own_support, scores, strict=True)),
                    "candidate_differs": objective_action != baseline,
                    "teacher_mass": objective_mass,
                    "teacher_mass_delta": objective_mass - baseline_mass,
                }
            schedules.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "particles": len(worlds),
                    "canonical_particles": len(keys),
                    "baseline_action": baseline,
                    "candidate_action": candidate,
                    "candidate_differs": candidate != baseline,
                    "baseline_teacher_mass": baseline_mass,
                    "candidate_teacher_mass": candidate_mass,
                    "teacher_mass_delta": candidate_mass - baseline_mass,
                    "opponent_objectives": objectives,
                    "native_capture_sha256s": [
                        _canonical_sha256(capture) for capture in captures
                    ],
                    "averaged_payoff_sha256": _canonical_sha256(averaged_matrices),
                    "native_solve_ms": solve_latencies,
                    "matrix_replay_ms": replay_ms,
                }
            )
        root_delta = math.fsum(row["teacher_mass_delta"] for row in schedules) / len(
            schedules
        )
        roots.append(
            {
                "identity": evaluation["identity"],
                "poststratification_weight": float(
                    evaluation["sampling"]["poststratification_weight"]
                ),
                "teacher_mass_delta": root_delta,
                "severe_regression": root_delta <= SEVERE_DELTA,
                "opponent_objectives": {
                    name: {
                        "teacher_mass_delta": math.fsum(
                            schedule["opponent_objectives"][name][
                                "teacher_mass_delta"
                            ]
                            for schedule in schedules
                        )
                        / len(schedules)
                    }
                    for name in OPPONENT_OBJECTIVES
                },
                "schedules": schedules,
            }
        )
    total_weight = math.fsum(root["poststratification_weight"] for root in roots)
    poststratified_delta = math.fsum(
        root["teacher_mass_delta"] * root["poststratification_weight"] for root in roots
    ) / total_weight
    severe = sum(root["severe_regression"] for root in roots)
    objective_metrics = {}
    for name in OPPONENT_OBJECTIVES:
        deltas = [
            root["opponent_objectives"][name]["teacher_mass_delta"] for root in roots
        ]
        poststratified = math.fsum(
            delta * root["poststratification_weight"]
            for delta, root in zip(deltas, roots, strict=True)
        ) / total_weight
        severe_count = sum(delta <= SEVERE_DELTA for delta in deltas)
        objective_metrics[name] = {
            "poststratified_teacher_mass_delta": poststratified,
            "mean_teacher_mass_delta": math.fsum(deltas) / len(deltas),
            "minimum_root_delta": min(deltas),
            "maximum_root_delta": max(deltas),
            "severe_regressions": severe_count,
            "candidate_differences": sum(
                schedule["opponent_objectives"][name]["candidate_differs"]
                for root in roots
                for schedule in root["schedules"]
            ),
            "gate_passed": poststratified > 0 and severe_count == 0,
        }
    return {
        "schema_version": 1,
        "mode": "shared_root_multi_seed_variance_oracle_ablation",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "runtime": _runtime_identity(),
        "inputs": {
            "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
            "prior_enrichment": {
                "path": str(enrichment_path),
                "sha256": _sha256(enrichment_path),
            },
        },
        "configuration": {
            "seeds": list(SEEDS),
            "continuation_iterations_per_seed": CONTINUATION_ITERATIONS,
            "rm_iterations": RM_ITERATIONS,
            "prior_strength": PRIOR_STRENGTH,
            "payoff_aggregation": "cellwise_arithmetic_mean_in_seed_order",
            "baseline": "independent_s_b_search_argmax",
            "severe_regression_delta": SEVERE_DELTA,
            "opponent_objectives": list(OPPONENT_OBJECTIVES),
            "opponent_uniform_mix": OPPONENT_UNIFORM_MIX,
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
            "native_replay_checks": native_replay_checks,
            "native_argmax_matches": native_argmax_matches,
            "candidate_differences": sum(
                schedule["candidate_differs"]
                for root in roots
                for schedule in root["schedules"]
            ),
            "severe_regressions": severe,
        },
        "metrics": {
            "poststratified_teacher_mass_delta": poststratified_delta,
            "mean_teacher_mass_delta": math.fsum(root["teacher_mass_delta"] for root in roots)
            / len(roots),
            "minimum_root_delta": min(root["teacher_mass_delta"] for root in roots),
            "maximum_root_delta": max(root["teacher_mass_delta"] for root in roots),
        },
        "opponent_objective_metrics": objective_metrics,
        "production_candidate_gate": {
            "objective": "s_b_smoothed",
            "passed": objective_metrics["s_b_smoothed"]["gate_passed"],
            "conditions": {
                "positive_poststratified_teacher_mass_delta": objective_metrics[
                    "s_b_smoothed"
                ]["poststratified_teacher_mass_delta"]
                > 0,
                "zero_severe_root_regressions": objective_metrics["s_b_smoothed"][
                    "severe_regressions"
                ]
                == 0,
            },
            "production_implementation_authorized": False,
            "new_games_authorized": False,
        },
        "gate": {
            "passed": native_argmax_matches == native_replay_checks
            and poststratified_delta > 0
            and severe == 0,
            "conditions": {
                "all_native_argmaxes_reproduced": native_argmax_matches
                == native_replay_checks,
                "positive_poststratified_teacher_mass_delta": poststratified_delta > 0,
                "zero_severe_root_regressions": severe == 0,
            },
            "production_contract_change_authorized": False,
            "new_games_authorized": False,
        },
        "roots": roots,
        "limitations": [
            "This is a development-panel ablation, not held-out game evidence.",
            "The S-4B teacher is a stronger-search proxy, not game-outcome ground truth.",
            "Cellwise averaging uses three deterministic oracle seeds and does not estimate all payoff uncertainty.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prior-enrichment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.evaluation.expanduser().resolve(),
        args.prior_enrichment.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "metrics": report["metrics"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
