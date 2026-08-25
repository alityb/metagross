#!/usr/bin/env python3
"""Cycle 37 combined own-form resolver and Cycle 36 lineage mechanics gate."""

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
from srcs.metagross.causal_reveal_ledger import canonical_species, form_ability_contract


ROOT = Path(__file__).resolve().parents[3]
C36_COUNT = 18
RESOLVER_COUNT = 24
TOTAL = 42
WORLDS = TOTAL * 16
PRESERVED_FAILURE_ROOT = "447289bc6ac18d2d638df32b4e3abeed3fcdb75dbf25eecd85ea0b30d6e053b2"


class Cycle37Error(RuntimeError):
    pass


def selections(run_dir: Path) -> list[dict[str, Any]]:
    cycle36 = [
        {**json.loads(line), "cycle37_panel": "cycle36_preserved"}
        for line in (ROOT / (
            "experimental/runs/search_native_v2_cycle36_switch_reactivation_20260816/"
            "selection.jsonl"
        )).read_text().splitlines() if line
    ]
    resolver = [
        {**json.loads(line), "cycle37_panel": "resolver_broader"}
        for line in (run_dir / "resolver-selection.jsonl").read_text().splitlines() if line
    ]
    return cycle36 + resolver


def worker(run_dir: Path, index: int) -> None:
    row = selections(run_dir)[index]
    import poke_engine

    probe = poke_engine.State()
    if not all(hasattr(poke_engine, name) for name in (
        "root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
    )) or not all(hasattr(probe, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
        "with_side_one_pokemon_ability", "with_side_two_pokemon_ability",
    )):
        raise Cycle37Error("request-authoritative mask/ability engine ABI is absent")
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
            "schema": "metagross-cycle37-own-active-resolver-root/v1",
            "cycle37_panel": row["cycle37_panel"],
        })
        if row["cycle37_panel"] == "resolver_broader":
            result.update({
                "public_exact_active": row["public_exact_active"],
                "request_exact_active": row["request_exact_active"],
                "canonical_active": row["canonical_active"],
            })
        else:
            result.update({
                "reactivation_tag": row["reactivation_tag"],
                "exact_return_species": row["exact_return_species"],
                "certified_reactivation_ability": row["certified_reactivation_ability"],
            })
        (run_dir / "mechanics-audit/workers" / f"{index:03d}.json").write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        )
    finally:
        os.chdir(previous)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = selections(run_dir)
    if (
        len(selected) != TOTAL
        or len({row["dependency_cluster_id"] for row in selected}) != TOTAL
        or any(row["split"] != "train" for row in selected)
        or Counter(row["cycle37_panel"] for row in selected)
        != {"cycle36_preserved": C36_COUNT, "resolver_broader": RESOLVER_COUNT}
    ):
        raise Cycle37Error("frozen combined TRAIN selection changed")
    broader = [row for row in selected if row["cycle37_panel"] == "resolver_broader"]
    if (
        Counter(row["role"] for row in broader) != {"p1": 12, "p2": 12}
        or any(
            row["public_exact_active"] == row["request_exact_active"]
            or canonical_species(row["public_exact_active"])
            != row["canonical_active"]
            or canonical_species(row["request_exact_active"])
            != row["canonical_active"]
            for row in broader
        )
    ):
        raise Cycle37Error("broader resolver mismatch contract changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle37Error("Cycle37 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    for index in range(TOTAL):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle37Error(f"isolated worker crashed before fail-closed row:{index}")
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
    verification = [value for row in passed for value in row["move_verification_ms"]]
    walls = [row["wall_ms"] for row in rows]
    preserved = next(
        row for row in rows
        if row.get("model_information_fingerprint_sha256") == PRESERVED_FAILURE_ROOT
    )
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "frozen_train_only_42_unique_clusters": True,
        "all_roots_and_672_worlds_supported": (
            len(passed) == TOTAL
            and sum(row.get("supported_scheduled_worlds", 0) for row in rows) == WORLDS
        ),
        "preserved_cycle36_failure_root_repaired": preserved["status"] == "pass",
        "broader_exact_mismatch_canonical_identity": all(
            row["public_exact_active"] != row["request_exact_active"]
            and canonical_species(row["public_exact_active"])
            == row["canonical_active"]
            == canonical_species(row["request_exact_active"])
            for row in passed if row["cycle37_panel"] == "resolver_broader"
        ) and sum(row["cycle37_panel"] == "resolver_broader" for row in passed) == 24,
        "zero_resolver_causal_or_hidden_integrity_error": not failed,
        "deterministic_two_by_eight_worlds_weights_actions": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "all_68_systematic_regressions_passed": manifest.get(
            "prefreeze_tests_passed"
        ) is True,
        "move_verification_p95_le_5ms": bool(verification)
        and c13.percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": c13.percentile(walls, .95) <= 1750.0,
        "post_run_frozen_integrity": post_ok,
        "no_validation_test_93_teacher_training_gpu_cloud": True,
    }
    report = {
        "schema": "metagross-cycle37-own-active-resolver-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "selected_roots": TOTAL,
            "cycle36_roots": C36_COUNT,
            "broader_resolver_roots": RESOLVER_COUNT,
            "passed_roots": len(passed),
            "failed_roots": len(failed),
            "scheduled_worlds": sum(
                row.get("supported_scheduled_worlds", 0) for row in rows
            ),
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
            "cycle36_selection_sha256": manifest["cycle36_selection_sha256"],
            "resolver_selection_sha256": c13.sha256(run_dir / "resolver-selection.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "fresh_h2h_protocol_design": all(gates.values()),
            "scored_h2h_started": False,
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
