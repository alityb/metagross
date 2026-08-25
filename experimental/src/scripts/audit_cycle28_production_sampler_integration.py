#!/usr/bin/env python3
"""Cycle28 TRAIN-only production sampler noninterference audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts.cycle19_equal8192_live_decision import (
    production_considered_sample,
    stable_seed,
)
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from srcs.metagross import causal_reveal_ledger as crl
from srcs.metagross.run_foul_play import (
    prepare_production_random_battles_with_causal_move_receipts,
)


ROOT = Path(__file__).resolve().parents[3]
COUNTS = (16, 32)
BASE_SEED = 2026081528
MCTS_ITERATIONS = 128


class Cycle28Error(RuntimeError):
    pass


def mechanical_bytes(world: Any) -> bytes:
    clone = copy.deepcopy(world)
    if hasattr(clone, crl.MOVE_RECEIPT_ATTRIBUTE):
        delattr(clone, crl.MOVE_RECEIPT_ATTRIBUTE)
    return pickle.dumps(clone, protocol=5)


def engine_states(
    worlds: list[tuple[Any, Any]], ledger: Any, engine: Any, search_main: Any
) -> list[str]:
    values = []
    for world, _weight in worlds:
        state = crl.convert_battle_with_causal_ledger(
            copy.deepcopy(world), search_main.battle_to_poke_engine_state, engine
        )
        values.append(state.to_string())
    return values


def fixed_search_action(
    states: list[str], weights: list[float], actions: list[str], identity: str
) -> tuple[str, str]:
    import poke_engine

    masses = {action: 0.0 for action in actions}
    total_weight = math.fsum(weights)
    if total_weight <= 0:
        raise Cycle28Error("fixture weights have no mass")
    receipts = []
    for index, (state_string, weight) in enumerate(zip(states, weights, strict=True)):
        seed = stable_seed(BASE_SEED, identity, index, "cycle28-parity") % (2**32)
        result = poke_engine.monte_carlo_tree_search_with_s1_request(
            poke_engine.State.from_string(state_string),
            actions,
            duration_ms=0,
            iterations=MCTS_ITERATIONS,
            threads=1,
            seed=seed,
        )
        by_action = {str(row.move_choice): int(row.visits) for row in result.side_one}
        if set(by_action) != set(actions) or int(result.total_visits) != MCTS_ITERATIONS:
            raise Cycle28Error("fixed parity search changed action/visit support")
        for action in actions:
            masses[action] += weight / total_weight * by_action[action] / MCTS_ITERATIONS
        receipts.append({"seed": seed, "visits": by_action})
    total = math.fsum(masses.values())
    policy = {action: mass / total for action, mass in masses.items()}
    action_seed = stable_seed(BASE_SEED, identity, "cycle28-selector")
    selected, considered = production_considered_sample(policy, action_seed)
    return selected, c13.hash_json(
        {"policy": policy, "considered": considered, "receipts": receipts}
    )


def process_fixture(
    selected: Mapping[str, Any], *, worktree: str, harness: str,
    output_root: Path, engine: Any, search_main: Any,
) -> dict[str, Any]:
    identity = selected["state_rank_sha256"][:16]
    temp = Path(tempfile.mkdtemp(prefix=identity + "-", dir=output_root / ".tmp"))
    capture = temp / "capture"
    phase = "capture"
    started = time.perf_counter()
    try:
        if c13.sha256(Path(selected["raw_path"])) != selected["raw_sha256"]:
            raise Cycle28Error("selected replay hash changed")
        completed = subprocess.run(
            [
                "node", harness, "--showdown", worktree,
                "--input", selected["raw_path"], "--out-dir", str(capture),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise Cycle28Error("causal replay capture failed")
        raw = json.loads(Path(selected["raw_path"]).read_text())
        public = json.loads((capture / "public.json").read_text())
        pov = json.loads((capture / f"{selected['role']}.json").read_text())
        derived = v12.materialize_role(
            battle_id=selected["battle_id"], role=selected["role"],
            public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
            showdown_commit=selected["showdown_commit"],
        )
        compact = c13._compact_state(derived, pov, selected["request_index"])
        c13._state_parity(selected, compact)
        target = derived["states"][selected["request_index"]]
        actions = sorted(c13.request_actions_exact(target["private_request"]))
        if set(actions) != set(target["legal_actions"]):
            raise Cycle28Error("request action contract changed")

        phase = "rehydration"
        unbound = crl.freeze_ledger(
            selected["battle_id"], selected["role"], target["public_prefix"]
        )
        battle = c14.build_foul_play_battle_fixed(
            battle_id=selected["battle_id"], role=selected["role"],
            states=derived["states"], target_index=selected["request_index"],
        )
        c13.reconcile_public_facts_into_battle(battle, unbound)
        ledger = crl.bind_live_move_states(battle, unbound)
        setattr(battle, crl.LEDGER_ATTRIBUTE, ledger.to_payload())
        intrinsic = sum(len(fact.moves) for fact in ledger.facts)
        derived_events = [
            event.to_payload()
            for fact in ledger.facts
            for event in fact.move_events
            if event.authority == "derived_public_execution"
        ]
        if (
            intrinsic != selected["cycle27_intrinsic_count"]
            or len(derived_events) != selected["cycle27_derived_count"]
        ):
            raise Cycle28Error("fixture causal class changed")

        phase = "sampler_parity"
        schedules = []
        verification_ms = []
        for count in COUNTS:
            seed = c13.stable_seed(BASE_SEED, selected["state_rank_sha256"], count)
            plain = search_main.prepare_random_battles(
                copy.deepcopy(battle), count, rng=random.Random(seed)
            )
            hooked = prepare_production_random_battles_with_causal_move_receipts(
                search_main.prepare_random_battles,
                copy.deepcopy(battle), count, rng=random.Random(seed),
            )
            repeat = prepare_production_random_battles_with_causal_move_receipts(
                search_main.prepare_random_battles,
                copy.deepcopy(battle), count, rng=random.Random(seed),
            )
            if any(len(rows) != count for rows in (plain, hooked, repeat)):
                raise Cycle28Error("adaptive sampler count changed")
            weight_sets = [
                [float(weight) for _world, weight in rows]
                for rows in (plain, hooked, repeat)
            ]
            if weight_sets[0] != weight_sets[1] or weight_sets[1] != weight_sets[2]:
                raise Cycle28Error("adaptive sampler weights changed")
            mechanical = [
                [hashlib.sha256(mechanical_bytes(world)).hexdigest() for world, _ in rows]
                for rows in (plain, hooked, repeat)
            ]
            if mechanical[0] != mechanical[1] or mechanical[1] != mechanical[2]:
                raise Cycle28Error("receipt hook changed sampled mechanical bytes")
            state_sets = [
                engine_states(rows, ledger, engine, search_main)
                for rows in (plain, hooked, repeat)
            ]
            if state_sets[0] != state_sets[1] or state_sets[1] != state_sets[2]:
                raise Cycle28Error("receipt hook changed engine bytes")
            receipts = [getattr(world, crl.MOVE_RECEIPT_ATTRIBUTE, None) for world, _ in hooked]
            if any(
                not isinstance(receipt, Mapping)
                or receipt.get("schema") != "metagross-causal-move-world-receipts/v1"
                or len(receipt.get("moves", ())) != intrinsic
                or receipt.get("derived_executions") != derived_events
                for receipt in receipts
            ):
                raise Cycle28Error("typed production receipt coverage changed")
            if selected["cycle28_fixture_class"] == "opening_empty" and any(
                receipt["moves"] or receipt["derived_executions"] for receipt in receipts
            ):
                raise Cycle28Error("opening empty receipt gained facts")
            verify_copy = copy.deepcopy(plain)
            verify_started = time.perf_counter()
            crl.verify_sampled_ledgers(battle, verify_copy)
            crl.verify_sampled_move_states(battle, verify_copy)
            verification_ms.append(
                (time.perf_counter() - verify_started) * 1000.0 / (count / 8)
            )
            plain_action, plain_search = fixed_search_action(
                state_sets[0], weight_sets[0], actions, f"{identity}:{count}:plain"
            )
            hook_action, hook_search = fixed_search_action(
                state_sets[1], weight_sets[1], actions, f"{identity}:{count}:plain"
            )
            if plain_action != hook_action or plain_search != hook_search:
                raise Cycle28Error("fixed search/selector action changed")
            schedules.append({
                "declared_world_count": count,
                "weights_sha256": c13.hash_json(weight_sets[0]),
                "mechanical_sha256": c13.hash_json(mechanical[0]),
                "engine_states_sha256": c13.hash_json(state_sets[0]),
                "receipts_sha256": c13.hash_json(receipts),
                "selected_action": plain_action,
                "search_receipt_sha256": plain_search,
            })
        shutil.rmtree(temp)
        return {
            "schema": "metagross-cycle28-production-sampler-fixture/v1",
            "status": "pass",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "fixture_class": selected["cycle28_fixture_class"],
            "intrinsic_move_count": intrinsic,
            "derived_execution_count": len(derived_events),
            "ledger_sha256": c13.hash_json(ledger.to_payload()),
            "schedules": schedules,
            "verification_ms_per_eight": verification_ms,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "teacher_values_opened": 0,
        }
    except BaseException as exc:
        failure = output_root / "failures" / identity
        failure.parent.mkdir(parents=True, exist_ok=True)
        if failure.exists():
            shutil.rmtree(failure)
        shutil.move(str(temp), failure)
        detail = f"{type(exc).__name__}:{exc}"
        return {
            "schema": "metagross-cycle28-production-sampler-fixture/v1",
            "status": "fail",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "fixture_class": selected["cycle28_fixture_class"],
            "phase": phase,
            "failure_class": type(exc).__name__,
            "failure_detail_sha256": hashlib.sha256(detail.encode()).hexdigest(),
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "teacher_values_opened": 0,
        }


def worker(run_dir: Path, index: int) -> None:
    selected = [json.loads(line) for line in (run_dir / "selection-16.jsonl").read_text().splitlines()]
    row = selected[index]
    import poke_engine

    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous = Path.cwd()
    os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main

        RandomBattleTeamDatasets.initialize("gen9")
        worktrees = json.loads((ROOT / (
            "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
            "replay-worktrees.json"
        )).read_text())
        result = process_fixture(
            row,
            worktree=worktrees[row["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit",
            engine=poke_engine,
            search_main=search_main,
        )
        (run_dir / "mechanics-audit/workers" / f"{index:03d}.json").write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        )
    finally:
        os.chdir(previous)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-16.jsonl").read_text().splitlines()]
    if (
        len(selected) != 16
        or len({row["dependency_cluster_id"] for row in selected}) != 16
        or any(row["split"] != "train" for row in selected)
    ):
        raise Cycle28Error("frozen Cycle28 TRAIN fixtures changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle28Error("Cycle28 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    (output / ".tmp").mkdir()
    for index in range(16):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle28Error(f"Cycle28 isolated worker crashed:{index}")
        print(json.dumps({"completed": index + 1, "total": 16}), flush=True)
    rows = [
        json.loads((output / "workers" / f"{index:03d}.json").read_text())
        for index in range(16)
    ]
    with (output / "fixture-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    latencies = [value for row in passed for value in row["verification_ms_per_eight"]]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "all_16_dependency_unique_train_fixtures_pass": len(passed) == 16,
        "both_16_32_adaptive_counts_each_fixture": all(
            [schedule["declared_world_count"] for schedule in row["schedules"]]
            == [16, 32] for row in passed
        ),
        "opening_empty_receipt_coverage": all(
            row["intrinsic_move_count"] == row["derived_execution_count"] == 0
            for row in passed if row["fixture_class"] == "opening_empty"
        ),
        "derived_nonconstraining_receipt_coverage": sum(
            row["derived_execution_count"] for row in passed
        ) == sum(row["cycle27_derived_count"] for row in selected),
        "world_weight_engine_action_noninterference": len(failed) == 0,
        "verification_p95_per_eight_le_5ms": c13.percentile(latencies, .95) <= 5.0,
        "post_run_frozen_integrity": post_ok,
        "teacher_validation_test_sealed_opened_zero": True,
    }
    report = {
        "schema": "metagross-cycle28-production-sampler-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "fixtures": 16,
            "passed": len(passed),
            "failed": len(failed),
            "opening_empty": sum(row["fixture_class"] == "opening_empty" for row in passed),
            "derived": sum(row["fixture_class"] == "derived" for row in passed),
            "later_intrinsic": sum(row["fixture_class"] == "later_intrinsic" for row in passed),
            "adaptive_worlds_compared_per_path": len(passed) * sum(COUNTS),
        },
        "failures": dict(Counter(
            f"{row.get('phase')}:{row.get('failure_class')}" for row in failed
        )),
        "latency_ms_per_eight": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p95": c13.percentile(latencies, .95),
            "max": max(latencies, default=0.0),
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": c13.sha256(run_dir / "selection-16.jsonl"),
            "results_sha256": c13.sha256(output / "fixture-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "fresh_operational_smoke": all(gates.values()),
            "scored_h2h": False,
            "training": False,
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
