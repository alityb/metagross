#!/usr/bin/env python3
"""Compare opponent and selection treatments on captured shared-root games."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

from srcs.metagross import run_foul_play, shared_root_replay
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    shared_root_result_payload,
)
from srcs.metagross.shared_root_capture import validate_search_row
from srcs.metagross.shared_root_prior_enrichment import load_and_validate


ROBUST_LAMBDAS = (0.10, 0.25, 0.50)
SEVERE_REGRESSION_DELTA = -0.20
SERIALIZATION_REPEATS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values, default=None),
        "max": max(values, default=None),
    }


def _weighted_mean(rows: list[dict], field: str) -> float | None:
    available = [row for row in rows if row[field] is not None]
    if not available:
        return None
    denominator = math.fsum(row["poststratification_weight"] for row in available)
    return math.fsum(
        row[field] * row["poststratification_weight"] for row in available
    ) / denominator


def _select(scores: dict[str, float]) -> str:
    return sorted(scores, key=lambda action: (-scores[action], action))[0]


def _policy_argmax(policy: list[dict]) -> str:
    return sorted(
        policy,
        key=lambda row: (
            -row["probability"],
            -row["counterfactual_value"],
            row["action"],
        ),
    )[0]["action"]


def _teacher_policy(repeat: dict) -> dict[str, float]:
    policy = {
        str(row["action"]).lower(): float(row["probability"])
        for row in repeat["side_one_policy"]
    }
    total = math.fsum(policy.values())
    if total <= 0:
        raise ValueError("teacher policy has no probability mass")
    return {action: probability / total for action, probability in policy.items()}


def _teacher_argmax(policy: dict[str, float]) -> str:
    return sorted(policy, key=lambda action: (-policy[action], action))[0]


def _strategy_scores(result: dict) -> dict[str, dict[str, float] | None]:
    capture = result["replay_capture"]
    actions = capture["own_action_support"]
    counterfactual = {
        row["action"]: row["counterfactual_value"] for row in result["policy"]
    }
    prior_expected = {action: 0.0 for action in actions}
    worst_case = {action: 0.0 for action in actions}
    priors_available = True
    for particle in capture["canonical_particles"]:
        weight = particle["normalized_weight"]
        opponent_prior = particle["normalized_opponent_prior"]
        if opponent_prior is None:
            priors_available = False
        for action_index, action in enumerate(actions):
            row = particle["payoff_matrix"][action_index]
            worst_case[action] += weight * min(row)
            if opponent_prior is not None:
                prior_expected[action] += weight * math.fsum(
                    payoff * probability
                    for payoff, probability in zip(row, opponent_prior, strict=True)
                )
    scores: dict[str, dict[str, float] | None] = {
        "counterfactual_argmax": counterfactual,
        "opponent_prior_expected": prior_expected if priors_available else None,
        "worst_case_endpoint": worst_case,
    }
    for contamination in ROBUST_LAMBDAS:
        scores[f"bounded_robust_{contamination:.2f}"] = (
            {
                action: (1 - contamination) * prior_expected[action]
                + contamination * worst_case[action]
                for action in actions
            }
            if priors_available
            else None
        )
    return scores


def _strategy_metrics(
    result: dict, schedule: dict
) -> tuple[dict[str, dict], list[dict]]:
    policy = result["policy"]
    mixed = {row["action"]: row["probability"] for row in policy}
    selections = {
        "rm_policy_argmax": _policy_argmax(policy),
    }
    score_sets = _strategy_scores(result)
    for name, scores in score_sets.items():
        selections[name] = _select(scores) if scores is not None else None
    repeats = schedule.get("aggregate_treatments", {}).get("S-4B")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError("schedule has no S-4B teacher repeats")
    repeat_rows = []
    for repeat in repeats:
        teacher = _teacher_policy(repeat)
        teacher_argmax = _teacher_argmax(teacher)
        row = {
            "repeat": repeat["repeat"],
            "teacher_argmax": teacher_argmax,
            "teacher_top_mass": teacher[teacher_argmax],
            "rm_mixed_alignment": math.fsum(
                probability * teacher.get(action, 0.0)
                for action, probability in mixed.items()
            ),
            "strategies": {},
        }
        for name, action in selections.items():
            row["strategies"][name] = {
                "selected_action": action,
                "teacher_mass": teacher.get(action, 0.0) if action is not None else None,
                "teacher_argmax_agreement": action == teacher_argmax if action is not None else None,
            }
        repeat_rows.append(row)
    aggregates = {
        "rm_mixed_alignment": _summary(
            [row["rm_mixed_alignment"] for row in repeat_rows]
        ),
        "strategies": {},
    }
    for name in selections:
        masses = [
            row["strategies"][name]["teacher_mass"]
            for row in repeat_rows
            if row["strategies"][name]["teacher_mass"] is not None
        ]
        agreements = [
            row["strategies"][name]["teacher_argmax_agreement"]
            for row in repeat_rows
            if row["strategies"][name]["teacher_argmax_agreement"] is not None
        ]
        aggregates["strategies"][name] = {
            "selected_action": selections[name],
            "teacher_mass": _summary(masses),
            "teacher_argmax_agreements": sum(agreements),
            "teacher_argmax_comparisons": len(agreements),
        }
    return aggregates, repeat_rows


def _measure_json(value: object) -> dict[str, float | int]:
    encoded_latencies = []
    decoded_latencies = []
    payload = b""
    for _ in range(SERIALIZATION_REPEATS):
        started = time.perf_counter_ns()
        payload = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        encoded_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        json.loads(payload)
        decoded_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "bytes": len(payload),
        "encode_ms_min": min(encoded_latencies),
        "decode_ms_min": min(decoded_latencies),
    }


def run(
    input_path: Path,
    *,
    iterations: int,
    continuation_iterations: int,
    seed: int,
    prior_strength: float,
    prior_enrichment_path: Path | None = None,
    prior_capture_path: Path | None = None,
) -> dict[str, object]:
    import poke_engine

    native_path = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    engine = {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "native_sha256": _sha256(native_path),
        "distribution_version": "0.0.47",
    }
    roots = []
    enrichment = None
    enriched_roots = {}
    if (prior_enrichment_path is None) != (prior_capture_path is None):
        raise ValueError("prior enrichment and capture panel must be provided together")
    if prior_enrichment_path is not None and prior_capture_path is not None:
        enrichment = load_and_validate(
            prior_enrichment_path, input_path, prior_capture_path
        )
        enriched_roots = {
            json.dumps(root["identity"], separators=(",", ":"), sort_keys=True): root
            for root in enrichment["roots"]
        }
    previous_prior_state = dict(run_foul_play._PRIOR_STATE)
    try:
        for root_index, record in enumerate(shared_root_replay._records(input_path)):
            enriched_root = enriched_roots.pop(
                json.dumps(record["identity"], separators=(",", ":"), sort_keys=True),
                None,
            )
            if enrichment is not None and enriched_root is None:
                raise ValueError("matrix root has no opponent-prior enrichment")
            enriched_schedules = (
                {schedule["schedule_id"]: schedule for schedule in enriched_root["schedules"]}
                if enriched_root is not None
                else {}
            )
            root_schedules = []
            for schedule in record["schedules"]:
                worlds = schedule["worlds"]
                state_strings = [world["sampled_state"] for world in worlds]
                source_weights = [float(world["sample_weight"]) for world in worlds]
                source_weight_sum = math.fsum(source_weights)
                weights = [weight / source_weight_sum for weight in source_weights]
                first_world = worlds[0]
                player_prior = first_world.get("effective_player_priors") or None
                opponent_prior = (
                    enriched_root["source_captured_raw_opponent_priors"]
                    if enriched_root is not None
                    else first_world.get("effective_opponent_priors") or None
                )
                if enriched_root is not None:
                    enriched_worlds = {
                        world["world_index"]: world
                        for world in enriched_schedules[schedule["schedule_id"]]["worlds"]
                    }
                    if any(
                        enriched_worlds[world["world_index"]]["state_sha256"]
                        != world["state_sha256"]
                        for world in worlds
                    ):
                        raise ValueError("matrix worlds differ from opponent-prior enrichment")
                started = time.perf_counter_ns()
                native = poke_engine.shared_information_set_root_search(
                    states=[poke_engine.State.from_string(state) for state in state_strings],
                    particle_weights=weights,
                    iterations=iterations,
                    continuation_iterations=continuation_iterations,
                    seed=seed,
                    prior_strength=prior_strength,
                    s1_prior=player_prior,
                    s2_priors=[opponent_prior for _state in state_strings],
                )
                solve_ms = (time.perf_counter_ns() - started) / 1_000_000
                result = shared_root_result_payload(native)
                action_seed = seed ^ (root_index << 8) ^ int(schedule["schedule_id"])
                recorded_sampling_seed = schedule.get("sampling_seed")
                replay_sampling_seed = (
                    int(recorded_sampling_seed)
                    if recorded_sampling_seed is not None
                    else seed ^ (1 << 32) ^ (root_index << 8) ^ int(schedule["schedule_id"])
                )
                remote_search = {
                    "sampling_seed": replay_sampling_seed,
                    "action_seed": action_seed,
                    "request_ids": [],
                    "engine": engine,
                }
                run_foul_play._PRIOR_STATE.update(
                    {"priors": player_prior, "opp_priors": opponent_prior}
                )
                request_actions = {row["action"] for row in result["policy"]}
                started = time.perf_counter_ns()
                envelope = run_foul_play.build_shared_root_replay_envelope(
                    states=state_strings,
                    source_weights=source_weights,
                    normalized_weights=weights,
                    iterations=iterations,
                    continuation_iterations=continuation_iterations,
                    solver_seed=seed,
                    action_seed=action_seed,
                    result=result,
                    remote_search=remote_search,
                    request_actions=request_actions,
                )
                envelope_build_ms = (time.perf_counter_ns() - started) / 1_000_000
                ordered = sorted(
                    result["policy"],
                    key=lambda row: (
                        -row["probability"],
                        -row["counterfactual_value"],
                        row["action"],
                    ),
                )
                sampled_action, draw = run_foul_play._sample_shared_root_action(
                    [row["action"] for row in ordered],
                    {row["action"]: row["probability"] for row in ordered},
                    action_seed,
                )
                row = {
                    "schema": 4,
                    "context": record["identity"],
                    "choice": sampled_action,
                    "choice_override": {
                        "sampled_action": sampled_action,
                        "mixed_strategy_draw": draw,
                    },
                    "player_priors": player_prior,
                    "opponent_priors": opponent_prior,
                    "remote_search": remote_search,
                    "shared_root": result,
                    "shared_root_replay": envelope,
                }
                started = time.perf_counter_ns()
                validation = validate_search_row(row, rerun=False)
                validation_ms = (time.perf_counter_ns() - started) / 1_000_000
                strategy_aggregates, teacher_repeats = _strategy_metrics(
                    result, schedule
                )
                root_schedules.append(
                    {
                        "schedule_id": schedule["schedule_id"],
                        "recorded_sampling_seed": recorded_sampling_seed,
                        "replay_sampling_seed": replay_sampling_seed,
                        "particles": len(state_strings),
                        "canonical_particles": result["diagnostics"][
                            "canonical_particle_count"
                        ],
                        "payoff_cells": result["diagnostics"]["payoff_cells"],
                        "solve_ms": solve_ms,
                        "capture_sha256": validation["capture_sha256"],
                        "strategy_aggregates": strategy_aggregates,
                        "teacher_repeats": teacher_repeats,
                        "serialization": {
                            "native_capture": _measure_json(
                                result["replay_capture"]
                            ),
                            "envelope": _measure_json(envelope),
                            "full_row": _measure_json(row),
                            "envelope_build_ms": envelope_build_ms,
                            "validation_ms": validation_ms,
                        },
                    }
                )
            roots.append(
                {
                    "identity": record["identity"],
                    "evaluation_sha256": record["evaluation_sha256"],
                    "sampling": record["sampling"],
                    "schedules": root_schedules,
                }
            )
    finally:
        run_foul_play._PRIOR_STATE.clear()
        run_foul_play._PRIOR_STATE.update(previous_prior_state)
    if enriched_roots:
        raise ValueError("opponent-prior enrichment has unmatched roots")

    strategy_names = list(roots[0]["schedules"][0]["strategy_aggregates"]["strategies"])
    root_metrics = []
    for root in roots:
        root_weight = float(root["sampling"]["poststratification_weight"])
        for strategy in strategy_names:
            masses = [
                repeat["strategies"][strategy]["teacher_mass"]
                for schedule in root["schedules"]
                for repeat in schedule["teacher_repeats"]
                if repeat["strategies"][strategy]["teacher_mass"] is not None
            ]
            agreements = [
                repeat["strategies"][strategy]["teacher_argmax_agreement"]
                for schedule in root["schedules"]
                for repeat in schedule["teacher_repeats"]
                if repeat["strategies"][strategy]["teacher_argmax_agreement"] is not None
            ]
            root_metrics.append(
                {
                    "identity": root["identity"],
                    "strategy": strategy,
                    "poststratification_weight": root_weight,
                    "teacher_mass": math.fsum(masses) / len(masses) if masses else None,
                    "teacher_argmax_agreement_fraction": (
                        sum(agreements) / len(agreements) if agreements else None
                    ),
                }
            )
    baseline_by_identity = {
        json.dumps(row["identity"], sort_keys=True): row
        for row in root_metrics
        if row["strategy"] == "rm_policy_argmax"
    }
    strategy_summaries = {}
    for strategy in strategy_names:
        rows = [row for row in root_metrics if row["strategy"] == strategy]
        deltas = []
        for row in rows:
            baseline = baseline_by_identity[json.dumps(row["identity"], sort_keys=True)]
            if row["teacher_mass"] is not None:
                deltas.append(row["teacher_mass"] - baseline["teacher_mass"])
        strategy_summaries[strategy] = {
            "available_roots": sum(row["teacher_mass"] is not None for row in rows),
            "teacher_mass": _summary(
                [row["teacher_mass"] for row in rows if row["teacher_mass"] is not None]
            ),
            "poststratified_teacher_mass": _weighted_mean(rows, "teacher_mass"),
            "teacher_argmax_agreement_fraction": _summary(
                [
                    row["teacher_argmax_agreement_fraction"]
                    for row in rows
                    if row["teacher_argmax_agreement_fraction"] is not None
                ]
            ),
            "poststratified_teacher_argmax_agreement_fraction": _weighted_mean(
                rows, "teacher_argmax_agreement_fraction"
            ),
            "teacher_mass_delta_from_rm_argmax": _summary(deltas),
            "severe_regressions": sum(
                delta <= SEVERE_REGRESSION_DELTA for delta in deltas
            ),
            "by_battle": {
                battle_tag: {
                    "roots": len(battle_rows),
                    "teacher_mass": _summary(
                        [
                            row["teacher_mass"]
                            for row in battle_rows
                            if row["teacher_mass"] is not None
                        ]
                    ),
                    "teacher_argmax_agreement_fraction": _summary(
                        [
                            row["teacher_argmax_agreement_fraction"]
                            for row in battle_rows
                            if row["teacher_argmax_agreement_fraction"] is not None
                        ]
                    ),
                }
                for battle_tag in sorted(
                    {row["identity"]["battle_tag"] for row in rows}
                )
                for battle_rows in [
                    [
                        row
                        for row in rows
                        if row["identity"]["battle_tag"] == battle_tag
                    ]
                ]
            },
        }
    mixed_root_rows = []
    for root in roots:
        alignments = [
            repeat["rm_mixed_alignment"]
            for schedule in root["schedules"]
            for repeat in schedule["teacher_repeats"]
        ]
        mixed_root_rows.append(
            {
                "poststratification_weight": float(
                    root["sampling"]["poststratification_weight"]
                ),
                "alignment": math.fsum(alignments) / len(alignments),
            }
        )
    serialization_rows = [
        schedule["serialization"] for root in roots for schedule in root["schedules"]
    ]
    serialization_summary = {
        artifact: {
            metric: _summary([row[artifact][metric] for row in serialization_rows])
            for metric in ("bytes", "encode_ms_min", "decode_ms_min")
        }
        for artifact in ("native_capture", "envelope", "full_row")
    }
    serialization_summary["envelope_build_ms"] = _summary(
        [row["envelope_build_ms"] for row in serialization_rows]
    )
    serialization_summary["validation_ms"] = _summary(
        [row["validation_ms"] for row in serialization_rows]
    )
    return {
        "schema_version": 1,
        "mode": "shared_root_matrix_diagnostics",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "prior_enrichment": (
            {
                "path": str(prior_enrichment_path),
                "sha256": _sha256(prior_enrichment_path),
                "capture_panel": {
                    "path": str(prior_capture_path),
                    "sha256": _sha256(prior_capture_path),
                },
                "provenance": enrichment["provenance"],
                "counts": enrichment["counts"],
            }
            if enrichment is not None
            else None
        ),
        "engine": engine,
        "configuration": {
            "iterations": iterations,
            "continuation_iterations": continuation_iterations,
            "seed": seed,
            "prior_strength": prior_strength,
            "robust_contamination": list(ROBUST_LAMBDAS),
            "severe_regression_delta": SEVERE_REGRESSION_DELTA,
            "serialization_repeats": SERIALIZATION_REPEATS,
            "teacher": "S-4B all four schedules and all three repeats",
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
            "teacher_comparisons": sum(
                len(schedule["teacher_repeats"])
                for root in roots
                for schedule in root["schedules"]
            ),
            "payoff_cells": sum(
                schedule["payoff_cells"]
                for root in roots
                for schedule in root["schedules"]
            ),
            "schedules_with_complete_opponent_priors": sum(
                schedule["strategy_aggregates"]["strategies"][
                    "opponent_prior_expected"
                ]["selected_action"]
                is not None
                for root in roots
                for schedule in root["schedules"]
            ),
        },
        "current_rm_mixed": {
            "teacher_alignment": _summary(
                [row["alignment"] for row in mixed_root_rows]
            ),
            "poststratified_teacher_alignment": _weighted_mean(
                mixed_root_rows, "alignment"
            ),
        },
        "strategy_summaries": strategy_summaries,
        "serialization_summary": serialization_summary,
        "root_metrics": root_metrics,
        "roots": roots,
        "limitations": [
            "S-4B is a stronger-search proxy, not game-outcome ground truth.",
            "Selectors are evaluated on captured payoff matrices and are not causal action-value estimates.",
            "The panel has 26 roots from five battles and cannot estimate a win rate or authorize games.",
            "Serialization timings are local single-process measurements, not end-to-end remote latency.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--continuation-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--prior-enrichment", type=Path)
    parser.add_argument("--prior-capture-panel", type=Path)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = run(
        args.input.expanduser().resolve(),
        iterations=args.iterations,
        continuation_iterations=args.continuation_iterations,
        seed=args.seed,
        prior_strength=args.prior_strength,
        prior_enrichment_path=(
            args.prior_enrichment.expanduser().resolve()
            if args.prior_enrichment is not None
            else None
        ),
        prior_capture_path=(
            args.prior_capture_panel.expanduser().resolve()
            if args.prior_capture_panel is not None
            else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
