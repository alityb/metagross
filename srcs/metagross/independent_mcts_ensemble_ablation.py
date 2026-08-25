#!/usr/bin/env python3
"""Evaluate repeated independent-MCTS visit aggregation on frozen roots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from srcs.metagross.shared_root_paired_verifier_experiment import _teacher_mass


REPEAT_COUNTS = (1, 2, 3)
SEVERE_DELTA = -0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensemble_action(schedule: dict, repeat_count: int) -> tuple[str, dict[str, float]]:
    mass: dict[str, float] = defaultdict(float)
    for world in schedule["worlds"]:
        world_weight = float(world["sample_weight"])
        for repeat in world["treatments"]["S-B"][:repeat_count]:
            result = repeat["result"]
            visits = float(result["total_visits"])
            for row in result["side_one"]:
                mass[row["action"]] += (
                    world_weight * float(row["visits"]) / visits / repeat_count
                )
    action = sorted(mass, key=lambda name: (-mass[name], name))[0]
    return action, dict(sorted(mass.items()))


def _validate_protocol(protocol_path: Path, evaluation_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dependency = Path(__file__).resolve().parent / "shared_root_paired_verifier_experiment.py"
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("dependency_sha256") != _sha256(dependency)
        or protocol.get("input_sha256") != _sha256(evaluation_path)
        or protocol.get("configuration")
        != {
            "treatment": "S-B",
            "repeat_counts": list(REPEAT_COUNTS),
            "candidate_repeat_count": 3,
            "baseline_repeat_count": 1,
            "aggregation": "world_weighted_normalized_visit_mass",
            "severe_regression_delta": SEVERE_DELTA,
        }
    ):
        raise ValueError("independent-MCTS ensemble differs from its frozen protocol")


def analyze(evaluation_path: Path, protocol_path: Path) -> dict[str, object]:
    _validate_protocol(protocol_path, evaluation_path)
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    roots = []
    for evaluation in evaluations:
        schedules = []
        for schedule in evaluation["schedules"]:
            repeats = {}
            for repeat_count in REPEAT_COUNTS:
                action, mass = _ensemble_action(schedule, repeat_count)
                repeats[str(repeat_count)] = {
                    "action": action,
                    "visit_mass": mass,
                    "teacher_mass": _teacher_mass(schedule, action),
                }
            baseline = repeats["1"]
            for row in repeats.values():
                row["teacher_mass_delta_from_repeat_1"] = (
                    row["teacher_mass"] - baseline["teacher_mass"]
                )
            schedules.append(
                {"schedule_id": schedule["schedule_id"], "repeats": repeats}
            )
        roots.append(
            {
                "identity": evaluation["identity"],
                "poststratification_weight": float(
                    evaluation["sampling"]["poststratification_weight"]
                ),
                "repeat_deltas": {
                    str(repeat_count): math.fsum(
                        schedule["repeats"][str(repeat_count)][
                            "teacher_mass_delta_from_repeat_1"
                        ]
                        for schedule in schedules
                    )
                    / len(schedules)
                    for repeat_count in REPEAT_COUNTS
                },
                "schedules": schedules,
            }
        )
    total_weight = math.fsum(root["poststratification_weight"] for root in roots)
    summaries = {}
    for repeat_count in REPEAT_COUNTS:
        key = str(repeat_count)
        deltas = [root["repeat_deltas"][key] for root in roots]
        poststratified = math.fsum(
            delta * root["poststratification_weight"]
            for delta, root in zip(deltas, roots, strict=True)
        ) / total_weight
        summaries[key] = {
            "poststratified_teacher_mass_delta": poststratified,
            "mean_teacher_mass_delta": math.fsum(deltas) / len(deltas),
            "minimum_root_delta": min(deltas),
            "maximum_root_delta": max(deltas),
            "severe_regressions": sum(delta <= SEVERE_DELTA for delta in deltas),
            "changed_schedules": sum(
                schedule["repeats"][key]["action"]
                != schedule["repeats"]["1"]["action"]
                for root in roots
                for schedule in root["schedules"]
            ),
            "changed_roots": sum(delta != 0 for delta in deltas),
        }
    candidate = summaries["3"]
    return {
        "schema_version": 1,
        "mode": "independent_mcts_repeat_ensemble_ablation",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "input": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        "configuration": {
            "treatment": "S-B",
            "repeat_counts": list(REPEAT_COUNTS),
            "candidate_repeat_count": 3,
            "baseline_repeat_count": 1,
            "aggregation": "world_weighted_normalized_visit_mass",
            "severe_regression_delta": SEVERE_DELTA,
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
        },
        "summaries": summaries,
        "gate": {
            "passed": candidate["poststratified_teacher_mass_delta"] > 0
            and candidate["severe_regressions"] == 0
            and candidate["changed_roots"] > 0,
            "conditions": {
                "positive_poststratified_teacher_mass_delta": candidate[
                    "poststratified_teacher_mass_delta"
                ]
                > 0,
                "zero_severe_root_regressions": candidate["severe_regressions"] == 0,
                "at_least_one_changed_root": candidate["changed_roots"] > 0,
            },
            "production_implementation_authorized": False,
            "new_games_authorized": False,
        },
        "roots": roots,
        "limitations": [
            "The three repeats and S-4B teacher belong to the same frozen evaluation family.",
            "Only one of 26 roots improves, so the aggregate signal is fragile.",
            "Latency and concurrent execution are not measured by this artifact.",
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
    print(json.dumps({"summaries": report["summaries"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
