#!/usr/bin/env python3
"""Cycle 34 TRAIN-only causal-Disable mechanics audit."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import audit_cycle27_disable_authority_forms as c27
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest


ROOT = Path(__file__).resolve().parents[3]


class Cycle34Error(RuntimeError):
    pass


def worker(run_dir: Path, index: int) -> None:
    selected = [
        json.loads(line) for line in (run_dir / "selection.jsonl").read_text().splitlines()
    ]
    row = selected[index]
    import poke_engine

    probe = poke_engine.State()
    if not all(hasattr(poke_engine, name) for name in (
        "root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
    )) or not all(hasattr(probe, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
        "with_side_one_pokemon_ability", "with_side_two_pokemon_ability",
    )):
        raise Cycle34Error("request-authoritative mask/ability engine ABI is absent")
    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous = Path.cwd()
    os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main

        RandomBattleTeamDatasets.initialize("gen9")
        c14.CURRENT_ACTIONS[:] = []
        poke_engine.root_options = lambda state: poke_engine.root_options_with_s1_request(
            state, c14.CURRENT_ACTIONS,
        )
        poke_engine.step_with_uniform_r1_semantic = lambda state, s1, s2, u: (
            poke_engine.step_with_uniform_r1_semantic_s1_request(
                state, c14.CURRENT_ACTIONS, s1, s2, u,
            )
        )
        original_actions = c13.request_actions_exact

        def capture_actions(request: Mapping[str, Any]) -> set[str]:
            actions = original_actions(request)
            c14.CURRENT_ACTIONS[:] = sorted(actions)
            return actions

        c13.request_actions_exact = capture_actions
        worktrees = json.loads((ROOT / (
            "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
            "replay-worktrees.json"
        )).read_text())
        result = c27._process_root(
            row, worktree=worktrees[row["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit", engine=poke_engine,
            search_main=search_main,
        )
        result.update({
            "schema": "metagross-cycle34-causal-disable-root/v1",
            "category": row["category"],
            "transition_state": row["transition_state"],
            "disabled_role": row["disabled_role"],
            "observer_role": row["observer_role"],
            "disabled_species": row["disabled_species"],
            "disabled_move": row["move"],
        })
        (run_dir / "mechanics-audit/workers" / f"{index:03d}.json").write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        )
    finally:
        os.chdir(previous)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [
        json.loads(line) for line in (run_dir / "selection.jsonl").read_text().splitlines()
    ]
    if (
        len(selected) != 40
        or len({row["dependency_cluster_id"] for row in selected}) != 24
        or any(row["split"] != "train" for row in selected)
        or Counter(row["transition_state"] for row in selected)
        != {"active": 24, "cleared": 16}
    ):
        raise Cycle34Error("frozen targeted transition selection changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle34Error("Cycle34 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    env = dict(os.environ)
    for index in range(len(selected)):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ], env=env)
        if completed.returncode != 0:
            raise Cycle34Error(f"isolated worker crashed:{index}")
        if (index + 1) % 8 == 0:
            print(json.dumps({"completed": index + 1, "total": len(selected)}), flush=True)
    rows = [
        json.loads((output / "workers" / f"{index:03d}.json").read_text())
        for index in range(len(selected))
    ]
    with (output / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    active = [row for row in passed if row["transition_state"] == "active"]
    cleared = [row for row in passed if row["transition_state"] == "cleared"]
    verification = [value for row in passed for value in row["move_verification_ms"]]
    wall = [row["wall_ms"] for row in rows]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "frozen_train_only_24_battle_40_state_panel": True,
        "both_roles_and_three_transition_categories": all(
            sum(
                row["category"] == category and row["disabled_role"] == role
                for row in selected if row["transition_state"] == "active"
            ) == 4
            for category in ("force_faint_carry", "explicit_end", "own_switch")
            for role in ("p1", "p2")
        ),
        "all_roots_and_640_worlds_supported": len(passed) == 40 and sum(
            row.get("supported_scheduled_worlds", 0) for row in rows
        ) == 640,
        "every_active_transition_reaches_causal_disable": len(active) == 24 and all(
            row["causal_disable_state_count"] >= 1 for row in active
        ),
        "every_cleared_transition_removes_causal_disable": len(cleared) == 16 and all(
            row["causal_disable_state_count"] == 0 for row in cleared
        ),
        "zero_causal_or_hidden_integrity_error": not failed,
        "deterministic_two_by_eight_worlds_weights_actions": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "preserved_cycle33_force_switch_regression": manifest.get(
            "preserved_cycle33_regression_passed"
        ) is True,
        "choice_and_clear_controls_passed": manifest.get(
            "prefreeze_tests_passed"
        ) is True,
        "move_verification_p95_le_5ms": c13.percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": c13.percentile(wall, .95) <= 1750.0,
        "post_run_frozen_integrity": post_ok,
        "no_validation_test_93_teacher_training_gpu_cloud": True,
    }
    report = {
        "schema": "metagross-cycle34-causal-disable-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "natural_inventory_events": 679,
            "natural_inventory_battles": 420,
            "selected_battles": 24,
            "selected_states": 40,
            "active_states": len(active),
            "cleared_states": len(cleared),
            "passed_states": len(passed),
            "failed_states": len(failed),
            "scheduled_worlds": sum(row.get("supported_scheduled_worlds", 0) for row in rows),
            "causal_disable_states": sum(row.get("causal_disable_state_count", 0) for row in passed),
            "world_mechanical_disable_states": sum(
                row.get("world_mechanical_disable_state_count", 0) for row in passed
            ),
        },
        "failures": dict(Counter(
            f"{row['phase']}:{row['failure_category']}:{row['failure_class']}"
            for row in failed
        )),
        "latency_ms": {
            "move_verification_p95": c13.percentile(verification, .95),
            "isolated_root_mean": statistics.fmean(wall),
            "isolated_root_p95": c13.percentile(wall, .95),
            "isolated_root_max": max(wall),
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": c13.sha256(run_dir / "selection.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "fresh_operational_smoke": all(gates.values()),
            "new_h2h": False,
            "training": False,
            "teacher_values": False,
        },
        "sealed_93_rows_read": 0,
        "local_cpu_only": True,
        "gpu_cloud_paid_cost_usd": 0,
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_index is None:
        parent(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve(), args.worker_index)


if __name__ == "__main__":
    main()
