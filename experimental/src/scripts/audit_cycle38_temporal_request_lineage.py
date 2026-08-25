#!/usr/bin/env python3
"""Cycle 38 temporal private-request/public-form lineage mechanics gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import audit_cycle27_disable_authority_forms as c27
from experimental.src.scripts import audit_cycle37_own_active_resolver as c37
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest


ROOT = Path(__file__).resolve().parents[3]
TOTAL = 42
EXPECTED_WORLDS = 41 * 16
ORICORIO_PP_ROOT = "cc266a54937a40d893f86bde29f3c2edd8d65e16621b473665c6ade8221524cd"
ORICORIO_PP_DETAIL = "12f655bbb5a10bbfbbcc7940d59f9bcd1ac09a5ee753455675c97cf7e5db4359"
TEMPORAL_ROOTS = {
    "5c527b69a9782ca5c89ba7b54271f2cda0e57ecfd14118d97296956d6729d179",
    "701656d78d068b588658a8f73bdb44adde22a999e57b0c4e2ffab5aab2e8ba6c",
    "a274fad4eaf1d58ad58dffde0220830ac0e8b62dfdb070517970f4b241c55ac0",
    "bf333c7fc1f03601baefebc4fbec6d6c85ed2a9c58fc567a40168fcaa52634ce",
    "b3f71466919608c18e4010ab254aafe0ca80e85f65ee03bfb872cb61a081c0e2",
    "e697e827e2b3557367b61767df4cffd6334af4d9856262cefa22b5274eb5b2be",
    "39395381f2e88b24ac7f2a906221cfa28962bbe5fc03842ff9367041781a81ed",
    "9e62af318121055830e5faf23d005bb6f28f4badaf6acc1b8baf4f6f4814b2b2",
    "7669804f7ba3e2b9ea6a775e438ef196a52994aca1b5a2cd91cc5eb3a5f47558",
    "a639321c7053789ffc0eabdd10c9f108fea0ab10d7c416a0d8cd5034dca0e76c",
}


class Cycle38Error(RuntimeError):
    pass


def selected_rows() -> list[dict[str, Any]]:
    return c37.selections(
        ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816"
    )


def worker(run_dir: Path, index: int) -> None:
    selected = selected_rows()[index]
    import poke_engine

    probe = poke_engine.State()
    if not all(hasattr(poke_engine, name) for name in (
        "root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
    )) or not all(hasattr(probe, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
        "with_side_one_pokemon_ability", "with_side_two_pokemon_ability",
    )):
        raise Cycle38Error("frozen request/mask/ability engine ABI is absent")
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
            "schema": "metagross-cycle38-temporal-request-root/v1",
            "battle_id": selected["battle_id"],
            "role": selected["role"],
            "model_information_fingerprint_sha256": selected[
                "model_information_fingerprint_sha256"
            ],
            "cycle37_panel": selected["cycle37_panel"],
            "was_cycle37_temporal_failure": selected[
                "model_information_fingerprint_sha256"
            ] in TEMPORAL_ROOTS,
            "is_expected_oricorio_pp_control": selected[
                "model_information_fingerprint_sha256"
            ] == ORICORIO_PP_ROOT,
        })
        (run_dir / "mechanics-audit/workers" / f"{index:03d}.json").write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        )
    finally:
        os.chdir(previous)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = selected_rows()
    if (
        len(selected) != TOTAL
        or len({row["dependency_cluster_id"] for row in selected}) != TOTAL
        or any(row["split"] != "train" for row in selected)
        or len(TEMPORAL_ROOTS) != 10
        or not TEMPORAL_ROOTS.issubset({
            row["model_information_fingerprint_sha256"] for row in selected
        })
    ):
        raise Cycle38Error("frozen Cycle37 TRAIN selection changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle38Error("Cycle38 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    for index in range(TOTAL):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle38Error(f"isolated worker crashed before fail-closed row:{index}")
        if (index + 1) % 6 == 0 or index + 1 == TOTAL:
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
    expected_failure = [
        row for row in failed
        if row["model_information_fingerprint_sha256"] == ORICORIO_PP_ROOT
        and row.get("failure_class") == "CausalRevealLedgerError"
        and row.get("failure_detail_sha256") == ORICORIO_PP_DETAIL
        and row.get("phase") == "foul_play_rehydration"
    ]
    temporal_pass = [
        row for row in passed
        if row["model_information_fingerprint_sha256"] in TEMPORAL_ROOTS
    ]
    verification = [value for row in passed for value in row["move_verification_ms"]]
    walls = [row["wall_ms"] for row in rows]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "frozen_cycle37_train_42_unique_clusters": True,
        "all_10_temporal_request_failures_repaired": len(temporal_pass) == 10,
        "41_non_pp_roots_and_656_worlds_supported": (
            len(passed) == 41
            and sum(row.get("supported_scheduled_worlds", 0) for row in passed)
            == EXPECTED_WORLDS
        ),
        "oricorio_pressure_pp_control_still_fails_exactly": (
            len(failed) == 1 and len(expected_failure) == 1
        ),
        "zero_new_semantic_hidden_action_or_determinism_failure": (
            len(failed) == len(expected_failure) == 1
        ),
        "deterministic_two_by_eight_worlds_weights_actions": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "all_prefreeze_regressions_passed": manifest.get(
            "prefreeze_tests_passed"
        ) is True,
        "move_verification_p95_le_5ms": bool(verification)
        and c13.percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": c13.percentile(walls, .95) <= 1750.0,
        "post_run_frozen_integrity": post_ok,
        "no_h2h_validation_test_93_teacher_training_gpu_cloud": True,
    }
    report = {
        "schema": "metagross-cycle38-temporal-request-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "selected_roots": TOTAL,
            "passed_roots": len(passed),
            "expected_failed_roots": len(expected_failure),
            "unexpected_failed_roots": len(failed) - len(expected_failure),
            "repaired_temporal_roots": len(temporal_pass),
            "supported_scheduled_worlds": sum(
                row.get("supported_scheduled_worlds", 0) for row in passed
            ),
        },
        "expected_out_of_scope_control": {
            "fingerprint": ORICORIO_PP_ROOT,
            "failure_detail_sha256": ORICORIO_PP_DETAIL,
            "structural_reason": "target-unaware vendor Pressure PP decrement",
        },
        "latency_ms": {
            "move_verification_p95": c13.percentile(verification, .95),
            "isolated_root_mean": statistics.fmean(walls),
            "isolated_root_p95": c13.percentile(walls, .95),
            "isolated_root_max": max(walls),
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "cycle39_pp_gate": all(gates.values()),
            "fresh_h2h": False,
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
        parent(args.run_dir)
    else:
        worker(args.run_dir, args.worker_index)


if __name__ == "__main__":
    main()
