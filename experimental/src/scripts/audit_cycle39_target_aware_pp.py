#!/usr/bin/env python3
"""Cycle 39 target-aware causal Pressure PP mechanics gate."""

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
from experimental.src.scripts import audit_cycle37_own_active_resolver as c37
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle39_target_aware_pp_20260816"
BASE_COUNT = 42
PRESSURE_COUNT = 16
TOTAL = BASE_COUNT + PRESSURE_COUNT
WORLDS = TOTAL * 16
ORICORIO_ROOT = "cc266a54937a40d893f86bde29f3c2edd8d65e16621b473665c6ade8221524cd"


class Cycle39Error(RuntimeError):
    pass


def selections(run_dir: Path) -> list[dict[str, Any]]:
    base = [
        {**row, "cycle39_panel": "cycle37_preserved"}
        for row in c37.selections(
            ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816"
        )
    ]
    pressure = [
        {**json.loads(line), "cycle39_panel": "natural_pressure"}
        for line in (run_dir / "pressure-selection.jsonl").read_text().splitlines()
        if line
    ]
    return base + pressure


def worker(run_dir: Path, index: int) -> None:
    selected = selections(run_dir)[index]
    import poke_engine

    probe = poke_engine.State()
    if not all(hasattr(poke_engine, name) for name in (
        "root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
    )) or not all(hasattr(probe, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
        "with_side_one_pokemon_ability", "with_side_two_pokemon_ability",
    )):
        raise Cycle39Error("frozen request/mask/ability engine ABI is absent")
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
            selected,
            worktree=worktrees[selected["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit",
            engine=poke_engine,
            search_main=search_main,
        )
        result.update({
            "schema": "metagross-cycle39-target-aware-pp-root/v1",
            "battle_id": selected["battle_id"],
            "role": selected["role"],
            "model_information_fingerprint_sha256": selected[
                "model_information_fingerprint_sha256"
            ],
            "cycle39_panel": selected["cycle39_panel"],
            "is_preserved_oricorio_control": selected[
                "model_information_fingerprint_sha256"
            ] == ORICORIO_ROOT,
        })
        if selected["cycle39_panel"] == "natural_pressure":
            result.update({
                "pressure_category": selected["category"],
                "pressure_move": selected["move"],
                "pressure_target_semantics": selected["target_semantics"],
                "pressure_mustpressure": selected["mustpressure"],
            })
        (run_dir / "mechanics-audit/workers" / f"{index:03d}.json").write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        )
    finally:
        os.chdir(previous)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = selections(run_dir)
    pressure = [row for row in selected if row["cycle39_panel"] == "natural_pressure"]
    expected_categories = Counter(
        (role, category) for role in ("p1", "p2")
        for category in ("self", "foe", "spread", "mustpressure")
        for _ in range(2)
    )
    if (
        len(selected) != TOTAL
        or len({row["dependency_cluster_id"] for row in selected}) != TOTAL
        or any(row["split"] != "train" for row in selected)
        or Counter((row["role"], row["category"]) for row in pressure)
        != expected_categories
    ):
        raise Cycle39Error("frozen 42+16 TRAIN selection changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle39Error("Cycle39 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    for index in range(TOTAL):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir.resolve()), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle39Error(f"isolated worker crashed before fail-closed row:{index}")
        if (index + 1) % 8 == 0 or index + 1 == TOTAL:
            print(json.dumps({"completed": index + 1, "total": TOTAL}), flush=True)
    rows = [
        json.loads((output / "workers" / f"{index:03d}.json").read_text())
        for index in range(TOTAL)
    ]
    with (output / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    natural_pass = [row for row in passed if row["cycle39_panel"] == "natural_pressure"]
    oricorio = [
        row for row in passed if row["model_information_fingerprint_sha256"] == ORICORIO_ROOT
    ]
    verification = [value for row in passed for value in row["move_verification_ms"]]
    walls = [row["wall_ms"] for row in rows]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "frozen_58_train_unique_clusters": True,
        "all_58_roots_and_928_worlds_supported": (
            len(passed) == TOTAL
            and sum(row.get("supported_scheduled_worlds", 0) for row in passed) == WORLDS
        ),
        "preserved_oricorio_exact_zero_control_repaired": len(oricorio) == 1,
        "all_16_natural_pressure_categories_both_roles_pass": (
            len(natural_pass) == PRESSURE_COUNT
            and Counter(
                (row["role"], row["pressure_category"]) for row in natural_pass
            ) == expected_categories
        ),
        "zero_pp_causal_hidden_action_or_determinism_failure": not failed,
        "deterministic_two_by_eight_worlds_weights_actions": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "all_90_prefreeze_regressions_passed": manifest.get(
            "prefreeze_tests_passed"
        ) is True,
        "move_verification_p95_le_5ms": bool(verification)
        and c13.percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": c13.percentile(walls, .95) <= 1750.0,
        "post_run_frozen_integrity": post_ok,
        "no_h2h_validation_test_93_teacher_training_gpu_cloud": True,
    }
    report = {
        "schema": "metagross-cycle39-target-aware-pp-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "selected_roots": TOTAL, "passed_roots": len(passed),
            "failed_roots": len(failed), "preserved_roots": BASE_COUNT,
            "natural_pressure_roots": PRESSURE_COUNT,
            "supported_scheduled_worlds": sum(
                row.get("supported_scheduled_worlds", 0) for row in passed
            ),
        },
        "natural_pressure_by_role_category": {
            f"{role}:{category}": sum(
                row["role"] == role and row["pressure_category"] == category
                for row in natural_pass
            )
            for role in ("p1", "p2")
            for category in ("self", "foe", "spread", "mustpressure")
        },
        "latency_ms": {
            "move_verification_p95": c13.percentile(verification, .95),
            "isolated_root_mean": statistics.fmean(walls),
            "isolated_root_p95": c13.percentile(walls, .95),
            "isolated_root_max": max(walls),
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "pressure_selection_sha256": c13.sha256(run_dir / "pressure-selection.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "fresh_h2h_protocol_design": all(gates.values()),
            "h2h_started": False, "training": False, "teacher_values": False,
        },
        "sealed_93_rows_read": 0, "local_cpu_only": True,
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
