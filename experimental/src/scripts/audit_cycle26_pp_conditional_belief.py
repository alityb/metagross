#!/usr/bin/env python3
"""Cycle 26 causal move/PP/disable mechanics over frozen TRAIN roots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import statistics
import struct
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
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from srcs.metagross import causal_reveal_ledger as crl


ROOT = Path(__file__).resolve().parents[3]
WORLD_COUNT = 8
SCHEDULE_COUNT = 2
BASE_SEED = 2026081526


class Cycle26Error(RuntimeError):
    pass


def _weight_bytes(worlds: list[tuple[Any, Any]]) -> bytes:
    values = []
    for _battle, weight in worlds:
        if not isinstance(weight, (float, int)) or isinstance(weight, bool):
            raise Cycle26Error("sampler weight is not numeric")
        values.append(struct.pack(">d", float(weight)))
    return b"".join(values)


def _failure_category(phase: str, exc: BaseException) -> str:
    detail = str(exc).lower()
    if any(token in detail for token in (
        "authority missing", "exact-form mapping", "missing causal move",
        "lacks exact foul play pp state",
    )):
        return "fail_closed_unsupported_move_state"
    if any(token in detail for token in (
        "changed causal pp-disable", "invalid causal live pp-disable",
        "engine move pp-disable state mismatch",
        "derived public", "sampler changed causal ledger", "public item mismatch",
        "public current ability", "reveal mask",
    )):
        return "causal_move_or_reveal_integrity"
    return c13.failure_category(phase, exc)


def _process_root(
    selected: Mapping[str, Any], *, worktree: str, harness: str,
    output_root: Path, engine: Any, search_main: Any,
) -> dict[str, Any]:
    root_id = selected["model_information_fingerprint_sha256"][:16]
    temp_parent = output_root / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=root_id + "-", dir=temp_parent))
    capture = temp / "capture"
    phase = "capture"
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        if c13.sha256(Path(selected["raw_path"])) != selected["raw_sha256"]:
            raise Cycle26Error("selected raw replay hash changed")
        subprocess.run([
            "node", harness, "--showdown", worktree,
            "--input", selected["raw_path"], "--out-dir", str(capture),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        raw = json.loads(Path(selected["raw_path"]).read_text())
        public = json.loads((capture / "public.json").read_text())
        pov = json.loads((capture / f"{selected['role']}.json").read_text())
        phase = "compact_parity"
        derived = v12.materialize_role(
            battle_id=selected["battle_id"], role=selected["role"],
            public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
            showdown_commit=selected["showdown_commit"],
        )
        compact = c13._compact_state(derived, pov, selected["request_index"])
        c13._state_parity(selected, compact)
        target = derived["states"][selected["request_index"]]
        if not target["actionable"] or target["pp_disable_sidecar"].get("revival_prompt"):
            raise Cycle26Error("selected state is nonordinary actionable semantics")
        actions = c13.request_actions_exact(target["private_request"])
        if actions != set(target["legal_actions"]):
            raise Cycle26Error("exact request action extractor disagrees")
        unbound = crl.freeze_ledger(
            selected["battle_id"], selected["role"], target["public_prefix"]
        )

        phase = "foul_play_rehydration"
        battle = c14.build_foul_play_battle_fixed(
            battle_id=selected["battle_id"], role=selected["role"],
            states=derived["states"], target_index=selected["request_index"],
        )
        c13.reconcile_public_facts_into_battle(battle, unbound)
        ledger = crl.bind_live_move_states(battle, unbound)
        setattr(battle, crl.LEDGER_ATTRIBUTE, ledger.to_payload())

        phase = "production_schedules"
        schedule_rows = []
        sampler_ms: list[float] = []
        verification_ms: list[float] = []
        conversion_ms: list[float] = []
        intrinsic_count = sum(len(fact.moves) for fact in ledger.facts)
        derived_count = sum(
            event.authority == "derived_public_execution"
            for fact in ledger.facts for event in fact.move_events
        )
        for schedule_index in range(SCHEDULE_COUNT):
            seed = c13.stable_seed(BASE_SEED, compact["model_information_fingerprint_sha256"], schedule_index)
            repeats = []
            for _repeat in range(2):
                started = time.perf_counter()
                worlds = search_main.prepare_random_battles(
                    copy.deepcopy(battle), WORLD_COUNT, rng=random.Random(seed),
                )
                sampler_ms.append((time.perf_counter() - started) * 1000.0)
                if len(worlds) != WORLD_COUNT:
                    raise Cycle26Error("production sampler returned wrong world count")
                before_weights = _weight_bytes(worlds)
                verify_started = time.perf_counter()
                crl.verify_sampled_ledgers(battle, worlds)
                crl.verify_sampled_move_states(battle, worlds)
                verification_ms.append((time.perf_counter() - verify_started) * 1000.0)
                after_weights = _weight_bytes(worlds)
                if before_weights != after_weights:
                    raise Cycle26Error("move verification changed raw sampler weights")
                payloads = []
                for sampled, _weight in worlds:
                    payload, elapsed, _public_size = c13._world_payload(
                        sampled, ledger, actions, engine, search_main,
                    )
                    payloads.append(payload)
                    conversion_ms.append(elapsed)
                repeats.append({
                    "worlds": payloads,
                    "weights_sha256": hashlib.sha256(after_weights).hexdigest(),
                })
            if repeats[0] != repeats[1]:
                raise Cycle26Error("seeded production schedule repeat disagreed")
            public_hashes = {row["public_sha256"] for row in repeats[0]["worlds"]}
            if len(public_hashes) != 1:
                raise Cycle26Error("hidden completions changed public projection")
            schedule_rows.append({
                "schedule_index": schedule_index,
                "seed": seed,
                "schedule_sha256": c13.hash_json(repeats[0]),
                "public_sha256": next(iter(public_hashes)),
                "weights_sha256": repeats[0]["weights_sha256"],
                "world_count": WORLD_COUNT,
            })
        if len({row["public_sha256"] for row in schedule_rows}) != 1:
            raise Cycle26Error("independent schedules changed public projection")
        result = {
            "schema": "metagross-cycle26-pp-conditional-root/v1",
            "status": "pass",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "model_information_fingerprint_sha256": compact[
                "model_information_fingerprint_sha256"
            ],
            "role": selected["role"],
            "request_index": selected["request_index"],
            "intrinsic_revealed_move_count": intrinsic_count,
            "derived_execution_count": derived_count,
            "move_state_count": sum(len(fact.move_states) for fact in ledger.facts),
            "ledger_sha256": c13.hash_json(ledger.to_payload()),
            "schedules": schedule_rows,
            "sampler_ms": sampler_ms,
            "move_verification_ms": verification_ms,
            "conversion_ms": conversion_ms,
            "supported_scheduled_worlds": SCHEDULE_COUNT * WORLD_COUNT,
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "cpu_ms": (time.process_time() - cpu_started) * 1000.0,
            "teacher_values_opened": 0,
        }
        shutil.rmtree(temp)
        return result
    except BaseException as exc:
        failure = output_root / "failures" / root_id
        failure.parent.mkdir(parents=True, exist_ok=True)
        if failure.exists():
            shutil.rmtree(failure)
        shutil.move(str(temp), failure)
        detail = f"{type(exc).__name__}:{exc}"
        row = {
            "schema": "metagross-cycle26-pp-conditional-root/v1",
            "status": "fail",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "phase": phase,
            "failure_class": type(exc).__name__,
            "failure_category": _failure_category(phase, exc),
            "failure_detail_sha256": hashlib.sha256(detail.encode("utf-8")).hexdigest(),
            "supported_scheduled_worlds": 0,
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "cpu_ms": (time.process_time() - cpu_started) * 1000.0,
            "teacher_values_opened": 0,
        }
        if os.environ.get("METAGROSS_CYCLE26_PREFREEZE_DEBUG") == "1":
            row["prefreeze_debug_detail"] = detail
        (failure / "FAILURE.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        return row


def worker(run_dir: Path, index: int) -> None:
    selected = [json.loads(line) for line in (run_dir / "selection-200.jsonl").read_text().splitlines()]
    row = selected[index]
    import poke_engine
    probe = poke_engine.State()
    if not all(hasattr(poke_engine, name) for name in (
        "root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
    )) or not all(hasattr(probe, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
        "with_side_one_pokemon_ability", "with_side_two_pokemon_ability",
    )):
        raise Cycle26Error("request-authoritative mask/ability engine ABI is absent")
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
        worktrees = json.loads((
            ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
        ).read_text())
        result = _process_root(
            row,
            worktree=worktrees[row["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit",
            engine=poke_engine,
            search_main=search_main,
        )
        path = run_dir / "mechanics-audit/workers" / f"{index:03d}.json"
        path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    finally:
        os.chdir(previous)


def _percentile(values: list[float], q: float) -> float:
    return c13.percentile(values, q)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-200.jsonl").read_text().splitlines()]
    if (
        len(selected) != 200
        or len({row["dependency_cluster_id"] for row in selected}) != 200
        or any(row["split"] != "train" for row in selected)
    ):
        raise Cycle26Error("frozen 200-root TRAIN selection changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle26Error("Cycle26 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    for index in range(200):
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--run-dir", str(run_dir), "--worker-index", str(index),
        ])
        if completed.returncode != 0:
            raise Cycle26Error(f"isolated worker crashed before fail-closed row:{index}")
        if (index + 1) % 20 == 0:
            print(json.dumps({"completed": index + 1, "total": 200}), flush=True)
    rows = [json.loads((output / "workers" / f"{i:03d}.json").read_text()) for i in range(200)]
    with (output / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    support = len(passed) / 200
    world_support = sum(row["supported_scheduled_worlds"] for row in rows) / (200 * 16)
    verification = [x for row in passed for x in row["move_verification_ms"]]
    wall = [row["wall_ms"] for row in rows]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException:
        post_ok = False
    gates = {
        "identical_cluster_unique_200_train_roots": True,
        "root_support_ge_0_95": support >= 0.95,
        "scheduled_world_support_ge_0_95": world_support >= 0.95,
        "zero_causal_move_integrity_failure": not any(
            row.get("failure_category") == "causal_move_or_reveal_integrity"
            for row in failed
        ),
        "exact_intrinsic_move_state_for_every_admitted_world": all(
            row["intrinsic_revealed_move_count"] == row["move_state_count"]
            for row in passed
        ),
        "deterministic_two_by_eight_schedules": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "move_verification_p95_le_5ms": _percentile(verification, .95) <= 5.0,
        "isolated_root_p95_le_1750ms": _percentile(wall, .95) <= 1750.0,
        "teacher_validation_test_sealed_opened_zero": True,
        "post_run_frozen_integrity": post_ok,
    }
    report = {
        "schema": "metagross-cycle26-pp-conditional-belief-report/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "counts": {
            "selected_roots": 200,
            "passed_roots": len(passed),
            "failed_roots": len(failed),
            "root_support": support,
            "scheduled_world_denominator": 3200,
            "supported_scheduled_worlds": sum(row["supported_scheduled_worlds"] for row in rows),
            "scheduled_world_support": world_support,
            "intrinsic_revealed_moves": sum(row.get("intrinsic_revealed_move_count", 0) for row in passed),
            "derived_executions": sum(row.get("derived_execution_count", 0) for row in passed),
        },
        "failures": dict(Counter(
            f"{row['phase']}:{row['failure_category']}:{row['failure_class']}"
            for row in failed
        )),
        "latency_ms": {
            "move_verification": {
                "mean": statistics.fmean(verification) if verification else 0.0,
                "p95": _percentile(verification, .95),
                "max": max(verification, default=0.0),
            },
            "isolated_root": {
                "mean": statistics.fmean(wall),
                "p95": _percentile(wall, .95),
                "max": max(wall),
            },
        },
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": c13.sha256(run_dir / "selection-200.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {
            "fresh_operational_smoke": all(gates.values()),
            "scored_h2h": False,
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
    run_dir = args.run_dir.resolve()
    if args.worker_index is None:
        parent(run_dir)
    else:
        worker(run_dir, args.worker_index)


if __name__ == "__main__":
    main()
