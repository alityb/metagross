#!/usr/bin/env python3
"""Rerun a frozen stratified sample of exact seeded MCTS treatment records."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from experimental.src.scripts import evaluate_teacher_root_bundles as evaluator
from experimental.src.scripts import teacher_root_bundle
from srcs.metagross.h2h_audit import _sha256


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _rank(seed: str, identity: dict, schedule: int, world: int, treatment: str, repeat: int) -> str:
    return hashlib.sha256(
        _canonical([seed, identity, schedule, world, treatment, repeat]).encode("ascii")
    ).hexdigest()


def audit(
    *, evaluation_path: Path, protocol_path: Path
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration = protocol.get("configuration")
    if (
        protocol.get("status") != "frozen_before_spot_audit"
        or protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs", {}).get("evaluation_sha256")
        != _sha256(evaluation_path)
        or configuration
        != {
            "selection_seed": "a" * 64,
            "per_treatment": 16,
            "treatments": ["U-B", "S-B", "S-4B"],
        }
    ):
        raise ValueError("seeded replay spot audit differs from its frozen protocol")
    root = Path(__file__).resolve().parents[2]
    dependencies = {
        "evaluate_teacher_root_bundles.py": root
        / "experimental"
        / "src"
        / "scripts"
        / "evaluate_teacher_root_bundles.py",
        "teacher_root_bundle.py": root
        / "experimental"
        / "src"
        / "scripts"
        / "teacher_root_bundle.py",
        "production_selector.py": root
        / "srcs"
        / "vendor"
        / "foul-play"
        / "fp"
        / "search"
        / "main.py",
        "runtime.py": root / "srcs" / "metagross" / "run_foul_play.py",
        "source_base_audit.json": root
        / "experimental"
        / "runs"
        / "search_native_stage2_20260809"
        / "adaptive-independent-ensemble-n100-base-audit-v7.json",
    }
    if any(
        protocol.get("dependency_sha256", {}).get(name) != _sha256(path)
        for name, path in dependencies.items()
    ):
        raise ValueError("seeded replay spot audit dependency mismatch")
    import poke_engine

    native_path = Path(inspect.getfile(poke_engine.poke_engine)).resolve()
    if (
        protocol.get("environment", {}).get("native_sha256") != _sha256(native_path)
        or "seed" not in inspect.signature(poke_engine.monte_carlo_tree_search).parameters
    ):
        raise ValueError("seeded replay spot audit loaded the wrong native engine")

    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    candidates: dict[str, list[dict]] = {
        treatment: [] for treatment in configuration["treatments"]
    }
    for evaluation in evaluations:
        evaluator.validate_root_evaluation(evaluation)
        for schedule in evaluation["schedules"]:
            for world in schedule["worlds"]:
                for treatment in configuration["treatments"]:
                    for repeat in world["treatments"][treatment]:
                        candidates[treatment].append(
                            {
                                "rank": _rank(
                                    configuration["selection_seed"],
                                    evaluation["identity"],
                                    schedule["schedule_id"],
                                    world["world_index"],
                                    treatment,
                                    repeat["repeat"],
                                ),
                                "identity": evaluation["identity"],
                                "schedule_id": schedule["schedule_id"],
                                "world": world,
                                "treatment": treatment,
                                "repeat": repeat,
                            }
                        )
    selected = []
    for treatment in configuration["treatments"]:
        selected.extend(
            sorted(candidates[treatment], key=lambda row: row["rank"])[
                : configuration["per_treatment"]
            ]
        )
    rows = []
    for selected_row in sorted(selected, key=lambda row: row["rank"]):
        world = selected_row["world"]
        treatment = selected_row["treatment"]
        repeat = selected_row["repeat"]
        state = poke_engine.State.from_string(world["sampled_state"])
        if treatment == "U-B":
            s1_priors = [tuple(row) for row in world["equal_side_one_priors"]]
        else:
            s1_priors = [tuple(row) for row in world["effective_player_priors"]]
        s2_priors = [tuple(row) for row in world["equal_side_two_priors"]]
        result = poke_engine.monte_carlo_tree_search(
            state,
            duration_ms=0,
            iterations=repeat["iterations"],
            threads=1,
            s1_priors=s1_priors,
            s2_priors=s2_priors,
            c_puct=2.0,
            seed=repeat["seed"],
        )
        actual = teacher_root_bundle.snapshot_result(result)
        expected = repeat["result"]
        rows.append(
            {
                "rank": selected_row["rank"],
                "identity": selected_row["identity"],
                "schedule_id": selected_row["schedule_id"],
                "world_index": world["world_index"],
                "treatment": treatment,
                "repeat": repeat["repeat"],
                "seed": repeat["seed"],
                "iterations": repeat["iterations"],
                "byte_identical_snapshot": _canonical(actual) == _canonical(expected),
                "expected_sha256": hashlib.sha256(
                    _canonical(expected).encode("ascii")
                ).hexdigest(),
                "actual_sha256": hashlib.sha256(
                    _canonical(actual).encode("ascii")
                ).hexdigest(),
            }
        )
    passed = (
        len(rows)
        == configuration["per_treatment"] * len(configuration["treatments"])
        and all(row["byte_identical_snapshot"] for row in rows)
    )
    source_base_audit = json.loads(
        dependencies["source_base_audit.json"].read_text(encoding="utf-8")
    )
    return {
        "schema_version": 1,
        "mode": "exact_seeded_native_replay_spot_audit",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
        },
        "counts": {
            "rerun_trees": len(rows),
            "by_treatment": {
                treatment: sum(row["treatment"] == treatment for row in rows)
                for treatment in configuration["treatments"]
            },
        },
        "source_limitations": {
            "historical_base_audit_passed": source_base_audit.get("gate", {}).get(
                "passed"
            ),
            "historical_base_audit_failures": source_base_audit.get("failures"),
            "disposition": "Historical command-provenance failures prohibit a go claim but do not reverse a failed offline gate.",
        },
        "gate": {"passed": passed, "all_snapshots_byte_identical": passed},
        "rows": rows,
        "new_games_authorized": False,
        "public_ladder_authorized": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(evaluation_path=args.evaluation, protocol_path=args.protocol)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": report["counts"], "gate": report["gate"]}, sort_keys=True))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
