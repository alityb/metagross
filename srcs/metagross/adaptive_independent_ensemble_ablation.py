#!/usr/bin/env python3
"""Evaluate a wire-bounded adaptive independent-MCTS ensemble on frozen roots."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from srcs.metagross.independent_mcts_ensemble_ablation import (
    SEVERE_DELTA,
    _ensemble_action,
    _sha256,
)
from srcs.metagross.shared_root_paired_verifier_experiment import _teacher_mass


MAXIMUM_REPEATS = 3
MAXIMUM_REQUESTS = 64


def adaptive_repeat_count(world_count: int) -> int:
    if world_count <= 0 or world_count > MAXIMUM_REQUESTS:
        raise ValueError("adaptive ensemble world count is outside the wire contract")
    return min(MAXIMUM_REPEATS, MAXIMUM_REQUESTS // world_count)


def production_selection_distribution(
    schedule: dict, repeat_count: int
) -> dict[str, float]:
    """Reproduce Foul Play's >=75%-of-maximum weighted action distribution."""
    _action, mass = _ensemble_action(schedule, repeat_count)
    if not mass:
        raise ValueError("ensemble action distribution is empty")
    threshold = max(mass.values()) * 0.75
    retained = {action: value for action, value in mass.items() if value >= threshold}
    total = math.fsum(retained.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("ensemble action distribution has no positive mass")
    return {action: retained[action] / total for action in sorted(retained)}


def expected_teacher_mass(
    schedule: dict, selection_distribution: dict[str, float]
) -> float:
    return math.fsum(
        probability * _teacher_mass(schedule, action)
        for action, probability in selection_distribution.items()
    )


def _validate_protocol(protocol_path: Path, evaluation_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = Path(__file__).with_name("independent_mcts_ensemble_ablation.py")
    runtime = Path(__file__).with_name("run_foul_play.py")
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("base_dependency_sha256") != _sha256(base)
        or protocol.get("runtime_sha256") != _sha256(runtime)
        or protocol.get("input_sha256") != _sha256(evaluation_path)
        or protocol.get("configuration")
        != {
            "treatment": "S-B",
            "maximum_repeats": MAXIMUM_REPEATS,
            "maximum_requests": MAXIMUM_REQUESTS,
                "baseline_repeat_count": 1,
                "aggregation": "world_weighted_normalized_visit_mass",
                "selection": "deterministic_argmax_lexicographic_tiebreak",
                "severe_regression_delta": SEVERE_DELTA,
        }
    ):
        raise ValueError("adaptive ensemble differs from its frozen protocol")


def analyze(evaluation_path: Path, protocol_path: Path) -> dict[str, object]:
    _validate_protocol(protocol_path, evaluation_path)
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    roots = []
    shape_counts: Counter[tuple[int, int]] = Counter()
    for evaluation in evaluations:
        schedules = []
        for schedule in evaluation["schedules"]:
            world_count = len(schedule["worlds"])
            repeat_count = adaptive_repeat_count(world_count)
            shape_counts[(world_count, repeat_count)] += 1
            baseline_distribution = production_selection_distribution(schedule, 1)
            distribution = production_selection_distribution(schedule, repeat_count)
            baseline_teacher_mass = expected_teacher_mass(
                schedule, baseline_distribution
            )
            teacher_mass = expected_teacher_mass(schedule, distribution)
            schedules.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "world_count": world_count,
                    "effective_repeat_count": repeat_count,
                    "request_count": world_count * repeat_count,
                    "baseline": {
                        "selection_distribution": baseline_distribution,
                        "teacher_mass": baseline_teacher_mass,
                    },
                    "candidate": {
                        "selection_distribution": distribution,
                        "teacher_mass": teacher_mass,
                        "teacher_mass_delta": teacher_mass
                        - baseline_teacher_mass,
                    },
                }
            )
        roots.append(
            {
                "identity": evaluation["identity"],
                "poststratification_weight": float(
                    evaluation["sampling"]["poststratification_weight"]
                ),
                "teacher_mass_delta": math.fsum(
                    row["candidate"]["teacher_mass_delta"]
                    for row in schedules
                )
                / len(schedules),
                "schedules": schedules,
            }
        )

    deltas = [root["teacher_mass_delta"] for root in roots]
    total_weight = math.fsum(root["poststratification_weight"] for root in roots)
    summary = {
        "poststratified_teacher_mass_delta": math.fsum(
            root["teacher_mass_delta"] * root["poststratification_weight"]
            for root in roots
        )
        / total_weight,
        "mean_teacher_mass_delta": math.fsum(deltas) / len(deltas),
        "minimum_root_delta": min(deltas),
        "maximum_root_delta": max(deltas),
        "severe_regressions": sum(delta <= SEVERE_DELTA for delta in deltas),
        "changed_schedules": sum(
            row["candidate"]["selection_distribution"]
            != row["baseline"]["selection_distribution"]
            for root in roots
            for row in root["schedules"]
        ),
        "changed_roots": sum(delta != 0 for delta in deltas),
    }
    conditions = {
        "positive_poststratified_teacher_mass_delta": summary[
            "poststratified_teacher_mass_delta"
        ]
        > 0,
        "zero_severe_root_regressions": summary["severe_regressions"] == 0,
        "at_least_one_changed_root": summary["changed_roots"] > 0,
        "all_schedules_within_wire_bound": all(
            row["request_count"] <= MAXIMUM_REQUESTS
            for root in roots
            for row in root["schedules"]
        ),
    }
    return {
        "schema_version": 1,
        "mode": "adaptive_independent_mcts_ensemble_ablation",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "input": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        "configuration": {
            "treatment": "S-B",
            "maximum_repeats": MAXIMUM_REPEATS,
            "maximum_requests": MAXIMUM_REQUESTS,
            "baseline_repeat_count": 1,
            "aggregation": "world_weighted_normalized_visit_mass",
            "selection": "deterministic_argmax_lexicographic_tiebreak",
            "severe_regression_delta": SEVERE_DELTA,
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
            "shapes": [
                {
                    "world_count": world_count,
                    "effective_repeat_count": repeat_count,
                    "request_count": world_count * repeat_count,
                    "schedules": count,
                }
                for (world_count, repeat_count), count in sorted(shape_counts.items())
            ],
        },
        "summary": summary,
        "gate": {
            "passed": all(conditions.values()),
            "conditions": conditions,
            "latency_evaluation_authorized": all(conditions.values()),
            "new_games_authorized": False,
            "public_ladder_authorized": False,
        },
        "roots": roots,
        "limitations": [
            "The repeats and S-4B teacher belong to the same frozen evaluation family.",
            "Any positive aggregate result remains fragile if driven by few changed roots.",
            "This corpus directly covers 16-world and 32-world schedules only.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.evaluation.expanduser().resolve(), args.protocol.expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
