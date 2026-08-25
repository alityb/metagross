#!/usr/bin/env python3
"""Audit exact teacher replay and adaptive-ensemble metrics on the reuse panel."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import random

from experimental.src.scripts import evaluate_teacher_root_bundles as evaluator
from experimental.src.scripts import teacher_root_bundle
from srcs.metagross.h2h_audit import _sha256
from srcs.metagross.adaptive_independent_ensemble_ablation import (
    adaptive_repeat_count,
    expected_teacher_mass,
    production_selection_distribution,
)
from srcs.metagross.world_provenance import state_sha256


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def _validate_result(result: dict, iterations: int) -> None:
    if result.get("total_visits") != iterations:
        raise ValueError("exact MCTS result visit count differs from its iteration budget")
    for side in ("side_one", "side_two"):
        options = result.get(side)
        if not isinstance(options, list) or not options:
            raise ValueError("exact MCTS result has no action support")
        actions = [row.get("action") for row in options]
        if len(actions) != len(set(actions)) or any(not action for action in actions):
            raise ValueError("exact MCTS result action support is invalid")
        if sum(int(row.get("visits", -1)) for row in options) != iterations:
            raise ValueError("exact MCTS action visits do not sum to the budget")
        if any(
            not math.isfinite(float(row.get("total_score", math.nan)))
            for row in options
        ):
            raise ValueError("exact MCTS result has nonfinite scores")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    rows: list[dict], clusters: dict[str, str], *, seed: int, resamples: int
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        tag = row["identity"]["battle_tag"]
        cluster = clusters.get(tag)
        if not cluster:
            raise ValueError(f"root has no inference cluster: {tag}")
        grouped.setdefault(cluster, []).append(row["root_delta"])
    cluster_values = {
        cluster: math.fsum(values) / len(values)
        for cluster, values in grouped.items()
    }
    keys = sorted(cluster_values)
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            math.fsum(cluster_values[generator.choice(keys)] for _ in keys) / len(keys)
        )
    return {
        "clusters": len(keys),
        "resamples": resamples,
        "seed": seed,
        "ci95_low": _quantile(estimates, 0.025),
        "ci95_high": _quantile(estimates, 0.975),
        "one_sided_95_lower": _quantile(estimates, 0.05),
    }


def audit(
    *,
    panel_path: Path,
    evaluation_path: Path,
    replay_protocol_path: Path,
    audit_protocol_path: Path,
) -> dict[str, object]:
    replay_protocol = json.loads(replay_protocol_path.read_text(encoding="utf-8"))
    audit_protocol = json.loads(audit_protocol_path.read_text(encoding="utf-8"))
    if (
        audit_protocol.get("status") != "frozen_before_audit"
        or audit_protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or audit_protocol.get("inputs", {}).get("panel_sha256") != _sha256(panel_path)
        or audit_protocol.get("inputs", {}).get("evaluation_sha256")
        != _sha256(evaluation_path)
        or audit_protocol.get("inputs", {}).get("replay_protocol_sha256")
        != _sha256(replay_protocol_path)
    ):
        raise ValueError("reuse replay audit differs from its frozen protocol")
    root = Path(__file__).resolve().parents[2]
    dependencies = {
        "evaluate_teacher_root_bundles.py": root / "experimental" / "src" / "scripts" / "evaluate_teacher_root_bundles.py",
        "teacher_root_bundle.py": root / "experimental" / "src" / "scripts" / "teacher_root_bundle.py",
        "adaptive_independent_ensemble_ablation.py": root / "srcs" / "metagross" / "adaptive_independent_ensemble_ablation.py",
        "independent_mcts_ensemble_ablation.py": root / "srcs" / "metagross" / "independent_mcts_ensemble_ablation.py",
    }
    if any(
        audit_protocol.get("dependency_sha256", {}).get(name) != _sha256(path)
        for name, path in dependencies.items()
    ):
        raise ValueError("reuse replay audit dependency identity mismatch")
    import poke_engine

    native_path = Path(inspect.getfile(poke_engine.poke_engine)).resolve()
    if (
        audit_protocol.get("environment", {}).get("native_sha256")
        != _sha256(native_path)
        or "seed" not in inspect.signature(poke_engine.monte_carlo_tree_search).parameters
    ):
        raise ValueError("reuse replay audit loaded the wrong native engine")

    panel = _rows(panel_path)
    evaluations = _rows(evaluation_path)
    captures = {tuple(sorted(row["identity"].items())): row for row in panel}
    configuration = replay_protocol["configuration"]
    expected_roots = int(configuration.get("expected_roots", 4))
    if not len(panel) == len(evaluations) == expected_roots:
        raise ValueError("reuse replay record count is invalid")
    manifest_sha256 = _sha256(replay_protocol_path)
    root_rows = []
    total_worlds = 0
    total_tree_runs = 0
    all_fresh_hashes = []
    for evaluation in evaluations:
        evaluator.validate_root_evaluation(evaluation)
        identity_key = tuple(sorted(evaluation["identity"].items()))
        capture = captures.get(identity_key)
        if capture is None:
            raise ValueError("evaluation has no source capture")
        expected_configuration = {
            "iterations": configuration["iterations_B"],
            "repeats": configuration["repeats"],
            "deep_multiplier": configuration["deep_multiplier"],
            "base_seed": configuration["tree_base_seed"],
            "c_puct": configuration["c_puct"],
            "threads": configuration["threads"],
            "execution": "offline",
            "input_manifest_sha256": manifest_sha256,
            "source_capture_sha256": capture["capture_sha256"],
            "source_input_manifest_sha256": capture["configuration"][
                "input_manifest_sha256"
            ],
            "primary_side_two_treatment": "equal_legal_priors",
        }
        if evaluation.get("configuration") != expected_configuration:
            raise ValueError("evaluation configuration differs from replay protocol")
        if not len(evaluation["schedules"]) == len(capture["schedules"]) == 4:
            raise ValueError("evaluation schedule count is invalid")
        schedule_deltas = []
        changed_schedules = 0
        root_hashes = []
        for source_schedule, schedule in zip(
            capture["schedules"], evaluation["schedules"], strict=True
        ):
            expected_world_count = configuration["schedule_world_counts"][
                source_schedule["schedule_id"]
            ]
            if not len(schedule["worlds"]) == len(source_schedule["worlds"]) == expected_world_count:
                raise ValueError("evaluation world count is invalid")
            for source_world, world in zip(
                source_schedule["worlds"], schedule["worlds"], strict=True
            ):
                if (
                    world["sampled_state"] != source_world["sampled_state"]
                    or world["state_sha256"] != source_world["state_sha256"]
                    or state_sha256(world["sampled_state"]) != world["state_sha256"]
                    or float(world["sample_weight"])
                    != float(source_world["sample_weight"])
                    or set(world["treatments"]) != {"U-B", "S-B", "S-4B"}
                ):
                    raise ValueError("evaluated world differs from its source capture")
                root_hashes.append(world["state_sha256"])
                all_fresh_hashes.append(world["state_sha256"])
                for treatment, budget in (
                    ("U-B", configuration["iterations_B"]),
                    ("S-B", configuration["iterations_B"]),
                    ("S-4B", configuration["iterations_4B"]),
                ):
                    repeats = world["treatments"][treatment]
                    if len(repeats) != configuration["repeats"]:
                        raise ValueError("teacher treatment repeat count is invalid")
                    for repeat, row in enumerate(repeats):
                        expected_seed = teacher_root_bundle.derive_scheduled_tree_seed(
                            configuration["tree_base_seed"],
                            evaluation["identity"],
                            schedule["schedule_id"],
                            world["state_sha256"],
                            world["world_index"],
                            treatment,
                            repeat,
                        )
                        if (
                            row.get("repeat") != repeat
                            or row.get("seed") != expected_seed
                            or row.get("iterations") != budget
                        ):
                            raise ValueError("teacher treatment seed or budget is invalid")
                        _validate_result(row["result"], budget)
                        total_tree_runs += 1
                total_worlds += 1
            for treatment in ("U-B", "S-B", "S-4B"):
                recomputed = [
                    {
                        "repeat": repeat,
                        "side_one_policy": evaluator._aggregate_schedule_policy(
                            schedule["worlds"], treatment, repeat
                        ),
                    }
                    for repeat in range(configuration["repeats"])
                ]
                if recomputed != schedule["aggregate_treatments"][treatment]:
                    raise ValueError("aggregate treatment policy did not recompute")
            repeat_count = adaptive_repeat_count(expected_world_count)
            baseline_distribution = production_selection_distribution(schedule, 1)
            candidate_distribution = production_selection_distribution(
                schedule, repeat_count
            )
            if baseline_distribution != candidate_distribution:
                changed_schedules += 1
            schedule_deltas.append(
                expected_teacher_mass(schedule, candidate_distribution)
                - expected_teacher_mass(schedule, baseline_distribution)
            )
        screening_hashes = set(
            capture["sampling"].get(
                "screening_state_hashes",
                capture["sampling"].get("original_state_hashes", []),
            )
        )
        root_rows.append(
            {
                "identity": evaluation["identity"],
                "root_delta": math.fsum(schedule_deltas) / len(schedule_deltas),
                "schedule_deltas": schedule_deltas,
                "changed_schedules": changed_schedules,
                "nominal_worlds": len(root_hashes),
                "unique_worlds": len(set(root_hashes)),
                "original_fresh_unique_overlap": len(screening_hashes & set(root_hashes)),
            }
        )

    deltas = [row["root_delta"] for row in root_rows]
    summary = {
        "poststratified_expected_teacher_mass_delta": math.fsum(deltas) / len(deltas),
        "minimum_root_delta": min(deltas),
        "maximum_root_delta": max(deltas),
        "nonnegative_roots": sum(delta >= 0 for delta in deltas),
        "changed_roots": sum(row["changed_schedules"] > 0 for row in root_rows),
        "changed_schedules": sum(row["changed_schedules"] for row in root_rows),
    }
    inference_configuration = replay_protocol.get("inference")
    inference = None
    if inference_configuration is not None:
        cluster_manifest_path = Path(inference_configuration["cluster_manifest_path"])
        if _sha256(cluster_manifest_path) != inference_configuration["cluster_manifest_sha256"]:
            raise ValueError("inference cluster manifest identity mismatch")
        cluster_manifest = json.loads(cluster_manifest_path.read_text(encoding="utf-8"))
        inference = _cluster_bootstrap(
            root_rows,
            cluster_manifest["battle_to_cluster"],
            seed=inference_configuration["bootstrap_seed"],
            resamples=inference_configuration["bootstrap_resamples"],
        )
    minimum_nonnegative = math.ceil(0.55 * len(root_rows))
    conditions = {
        "positive_poststratified_delta": summary[
            "poststratified_expected_teacher_mass_delta"
        ]
        > 0,
        "at_least_55_percent_nonnegative_roots": summary["nonnegative_roots"]
        >= minimum_nonnegative,
        "minimum_root_above_negative_0_02": summary["minimum_root_delta"] > -0.02,
        "at_least_one_changed_root": summary["changed_roots"] > 0,
        "all_evaluations_complete": len(evaluations) == expected_roots,
        "all_worlds_complete": total_worlds
        == len(evaluations) * sum(configuration["schedule_world_counts"]),
        "all_tree_runs_complete": total_tree_runs
        == len(evaluations)
        * sum(configuration["schedule_world_counts"])
        * len(configuration["treatments"])
        * configuration["repeats"],
    }
    if inference is not None:
        conditions.update(
            {
                "cluster_ci95_lower_above_zero": inference["ci95_low"] > 0,
                "at_least_20_percent_changed_roots": summary["changed_roots"]
                >= math.ceil(0.20 * len(root_rows)),
            }
        )
    return {
        "schema_version": 1,
        "mode": "adaptive_independent_ensemble_reuse_replay_audit",
        "inputs": {
            "panel": {"path": str(panel_path), "sha256": _sha256(panel_path)},
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": _sha256(evaluation_path),
            },
            "replay_protocol": {
                "path": str(replay_protocol_path),
                "sha256": _sha256(replay_protocol_path),
            },
            "audit_protocol": {
                "path": str(audit_protocol_path),
                "sha256": _sha256(audit_protocol_path),
            },
        },
        "counts": {
            "roots": len(root_rows),
            "schedules": sum(len(row["schedule_deltas"]) for row in root_rows),
            "worlds": total_worlds,
            "tree_runs": total_tree_runs,
            "nominal_fresh_worlds": len(all_fresh_hashes),
            "unique_fresh_worlds": len(set(all_fresh_hashes)),
        },
        "roots": root_rows,
        "summary": summary,
        "inference": inference,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "authorization": {
            "independent_review_of_larger_local_screen_allowed": all(
                conditions.values()
            ),
            "larger_local_screen_authorized": False,
            "strength_claim_authorized": False,
            "public_ladder_authorized": False,
        },
        "limitations": replay_protocol["limitations"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--replay-protocol", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        panel_path=args.panel.expanduser().resolve(),
        evaluation_path=args.evaluation.expanduser().resolve(),
        replay_protocol_path=args.replay_protocol.expanduser().resolve(),
        audit_protocol_path=args.audit_protocol.expanduser().resolve(),
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "gate": report["gate"]}, sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
