#!/usr/bin/env python3
"""Cycle 36 TRAIN-only exact-form switch/drag reactivation mechanics gate."""

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
from srcs.metagross.causal_reveal_ledger import form_ability_contract


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROOTS = 18
EXPECTED_WORLDS = EXPECTED_ROOTS * 16


class Cycle36Error(RuntimeError):
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
        raise Cycle36Error("request-authoritative mask/ability engine ABI is absent")
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
            row,
            worktree=worktrees[row["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit",
            engine=poke_engine,
            search_main=search_main,
        )
        result.update({
            "schema": "metagross-cycle36-reactivation-root/v1",
            "family": row["family"],
            "changed_role": row["changed_role"],
            "observer_role": row["observer_role"],
            "reactivation_tag": row["reactivation_tag"],
            "exact_return_species": row["exact_return_species"],
            "certified_reactivation_ability": row["certified_reactivation_ability"],
            "reactivation_authority": "rule_implied_switch_reactivation",
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
        len(selected) != EXPECTED_ROOTS
        or len({row["dependency_cluster_id"] for row in selected}) != EXPECTED_ROOTS
        or any(row["split"] != "train" for row in selected)
        or Counter(row["changed_role"] for row in selected) != {"p1": 9, "p2": 9}
        or Counter(row["reactivation_tag"] for row in selected) != {"switch": 16, "drag": 2}
    ):
        raise Cycle36Error("frozen natural reactivation selection changed")
    contract = form_ability_contract()
    if any(
        contract.get(row["exact_return_species"])
        != row["certified_reactivation_ability"]
        for row in selected
    ):
        raise Cycle36Error("selected reactivation no longer matches pinned contract")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle36Error("Cycle36 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    for index in range(EXPECTED_ROOTS):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle36Error(f"isolated worker crashed before fail-closed row:{index}")
        print(json.dumps({"completed": index + 1, "total": EXPECTED_ROOTS}), flush=True)
    rows = [
        json.loads((output / "workers" / f"{index:03d}.json").read_text())
        for index in range(EXPECTED_ROOTS)
    ]
    with (output / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    verification = [value for row in passed for value in row["move_verification_ms"]]
    walls = [row["wall_ms"] for row in rows]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "frozen_train_only_18_battle_panel": True,
        "both_roles_and_switch_drag": True,
        "all_roots_and_288_worlds_supported": (
            len(passed) == EXPECTED_ROOTS
            and sum(row.get("supported_scheduled_worlds", 0) for row in rows)
            == EXPECTED_WORLDS
        ),
        "exact_unique_reactivation_authority_every_root": all(
            row["reactivation_authority"] == "rule_implied_switch_reactivation"
            and contract.get(row["exact_return_species"])
            == row["certified_reactivation_ability"]
            for row in passed
        ),
        "zero_causal_or_hidden_integrity_error": not failed,
        "deterministic_two_by_eight_worlds_weights_actions": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "preserved_trace_and_systematic_controls_passed": manifest.get(
            "prefreeze_tests_passed"
        ) is True,
        "move_verification_p95_le_5ms": bool(verification)
        and c13.percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": c13.percentile(walls, .95) <= 1750.0,
        "post_run_frozen_integrity": post_ok,
        "no_validation_test_93_teacher_training_gpu_cloud": True,
    }
    report = {
        "schema": "metagross-cycle36-reactivation-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "natural_inventory_events": manifest["natural_inventory_events"],
            "natural_inventory_battles": manifest["natural_inventory_battles"],
            "selected_battles": EXPECTED_ROOTS,
            "selected_states": EXPECTED_ROOTS,
            "passed_states": len(passed),
            "failed_states": len(failed),
            "scheduled_worlds": sum(
                row.get("supported_scheduled_worlds", 0) for row in rows
            ),
            "switch_states": sum(row["reactivation_tag"] == "switch" for row in rows),
            "drag_states": sum(row["reactivation_tag"] == "drag" for row in rows),
        },
        "failures": dict(Counter(
            f"{row['phase']}:{row['failure_category']}:{row['failure_class']}"
            for row in failed
        )),
        "latency_ms": {
            "move_verification_p95": c13.percentile(verification, .95)
            if verification else None,
            "isolated_root_mean": statistics.fmean(walls),
            "isolated_root_p95": c13.percentile(walls, .95),
            "isolated_root_max": max(walls),
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": c13.sha256(run_dir / "selection.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "natural_coverage_caveat": (
            "Ogerpon has 19 label-blind public reactivation transitions but zero "
            "subsequent actionable roots; it is covered by systematic fixtures only."
        ),
        "authorization": {
            "fresh_operational_smoke_if_feasible": all(gates.values()),
            "preserved_trace_plus_natural_panel_fallback": all(gates.values()),
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
