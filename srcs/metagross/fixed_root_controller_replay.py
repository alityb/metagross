#!/usr/bin/env python3
"""Verify search-first selection on immutable captured root bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from srcs.metagross import run_foul_play


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") not in {
                "teacher_root_capture",
                "teacher_root_evaluation",
            }:
                raise ValueError(f"record {line_number} is not a teacher root artifact")
            rows.append(row)
    if not rows:
        raise ValueError("root capture file is empty")
    return rows


def _result(snapshot: dict) -> SimpleNamespace:
    def side(name: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                move_choice=row["action"],
                total_score=float(row["total_score"]),
                visits=int(row["visits"]),
            )
            for row in snapshot[name]
        ]

    return SimpleNamespace(
        side_one=side("side_one"),
        side_two=side("side_two"),
        total_visits=int(snapshot["total_visits"]),
    )


def _mcts_results(capture: dict) -> list[tuple[SimpleNamespace, float, int]]:
    schedules = capture["schedules"]
    if capture["record_type"] == "teacher_root_evaluation":
        schedule = next(row for row in schedules if int(row["schedule_id"]) == 0)
        return [
            (
                _result(world["treatments"]["S-B"][0]["result"]),
                float(world["sample_weight"]),
                index,
            )
            for index, world in enumerate(schedule["worlds"])
        ]
    schedule_id = int(capture["behavior_schedule_id"])
    schedule = next(row for row in schedules if int(row["schedule_id"]) == schedule_id)
    return [
        (_result(world["live_result"]), float(world["sample_weight"]), index)
        for index, world in enumerate(schedule["worlds"])
    ]


def _priors(artifact: dict) -> list[tuple[str, float]]:
    if artifact["record_type"] == "teacher_root_evaluation":
        worlds = artifact["schedules"][0]["worlds"]
        return [tuple(row) for row in worlds[0]["effective_player_priors"]]
    return [tuple(row) for row in artifact["recorded_player_priors"]]


def _minimal_public_battle(identity: dict) -> SimpleNamespace:
    side = SimpleNamespace(active=None, reserve=[], side_conditions={})
    return SimpleNamespace(
        battle_tag=identity["battle_tag"],
        user=side,
        opponent=SimpleNamespace(active=None, reserve=[], side_conditions={}),
        force_switch=False,
    )


def verify_files(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        for capture in _records(path):
            results = _mcts_results(capture)
            priors = _priors(capture)
            ordered, _mass = run_foul_play._mcts_actions_and_visit_mass(results)
            expected = ordered[0]
            evidence = {expected: {"qualified": False, "coverage": 1.0}}
            battle = _minimal_public_battle(capture["identity"])
            first, first_telemetry = run_foul_play.select_search_first_choice(
                battle,
                results,
                priors,
                histories={},
                independent_evidence=evidence,
                record_history=False,
            )
            second, second_telemetry = run_foul_play.select_search_first_choice(
                battle,
                results,
                priors,
                histories={},
                independent_evidence=evidence,
                record_history=False,
            )
            snapshot = capture.get("r1_policy_snapshot") or {}
            if snapshot:
                legal = {
                    name
                    for name, index in snapshot["name_table"].items()
                    if snapshot["illegal_actions"][index] is False
                }
            else:
                legal = {action for action, _probability in priors}
            rows.append(
                {
                    "input_path": str(path),
                    "input_sha256": _sha256(path),
                    "artifact_sha256": capture.get("capture_sha256")
                    or capture["evaluation_sha256"],
                    "identity": capture["identity"],
                    "behavior_schedule_id": capture.get("behavior_schedule_id", 0),
                    "canonical_search_choice": expected,
                    "first_choice": first,
                    "second_choice": second,
                    "first_reason": first_telemetry["reason"],
                    "second_reason": second_telemetry["reason"],
                    "canonical_mapping_present": expected in legal,
                    "rejected_certificate_was_shadow_only": not first_telemetry[
                        "verifier_shadow"
                    ]["selection_eligible"],
                    "passed": first == expected == second and expected in legal,
                }
            )
    return {
        "schema_version": 1,
        "mode": "search_first_fixed_root_replay",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "configuration": {
            "controller_mode": "search_first",
            "source_schedule": "captured_behavior_schedule",
            "positive_superiority_evidence": "forced_reject_shadow_only",
        },
        "counts": {
            "roots": len(rows),
            "canonical_search_matches": sum(
                row["first_choice"] == row["canonical_search_choice"] for row in rows
            ),
            "deterministic_repeat_matches": sum(
                row["first_choice"] == row["second_choice"] for row in rows
            ),
            "canonical_mappings": sum(row["canonical_mapping_present"] for row in rows),
            "failures": sum(not row["passed"] for row in rows),
        },
        "roots": rows,
        "gate": {
            "passed": bool(rows) and all(row["passed"] for row in rows),
            "all_search_choices_match_frozen_baseline": all(
                row["first_choice"] == row["canonical_search_choice"] for row in rows
            ),
            "all_repeats_match": all(
                row["first_choice"] == row["second_choice"] for row in rows
            ),
            "all_canonical_mappings_present": all(
                row["canonical_mapping_present"] for row in rows
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = verify_files([path.expanduser().resolve() for path in args.input])
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
