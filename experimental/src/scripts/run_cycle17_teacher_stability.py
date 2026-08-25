#!/usr/bin/env python3
"""Cycle 17 normalized-rqid preflight and unchanged teacher stability gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts import run_cycle15_teacher_stability as c15
from experimental.src.scripts import run_cycle16_teacher_stability as c16
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest

ROOT = Path(__file__).resolve().parents[3]


class Cycle17Error(RuntimeError):
    pass


def invariant_snapshot_without_routing_rqid(response: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = response.get("r1_policy_snapshot")
    if not isinstance(snapshot, dict):
        raise Cycle17Error("dual R1 snapshot missing")
    legality = copy.deepcopy(response["own_legality"])
    if set(legality) != {"authority", "rqid", "force_switch", "trapped", "can_tera", "actions"}:
        raise Cycle17Error("own legality schema changed")
    rqid = legality.pop("rqid")
    if isinstance(rqid, bool) or not isinstance(rqid, int) or rqid < 0:
        raise Cycle17Error("routing rqid is invalid")
    return {
        "text_tokens": snapshot["text_tokens"], "numbers": snapshot["numbers"],
        "illegal_actions": snapshot["illegal_actions"], "name_table": snapshot["name_table"],
        "trajectory_observation_rows": snapshot["trajectory"]["observation_rows"],
        "trajectory_rl2": snapshot["trajectory"]["rl2"],
        "trajectory_time_indices": snapshot["trajectory"]["time_indices"],
        "priors": response["priors"], "opponent_priors": response["opp_priors"],
        "probs": response["probs"], "mechanical_legality": legality,
    }


def _schedule_hash(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_actions": context["request_actions"],
        "exact_duration_ms": context["exact_duration_ms"],
        "exact_world_count": context["exact_world_count"],
        "exact_sha256": c15.canonical_hash(context["exact"]),
        "paired_sha256": c15.canonical_hash(context["paired"]),
    }


def preflight_worker(run_dir: Path, index: int) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-40.jsonl").read_text().splitlines()][index]
    sys.path.insert(0, manifest["engine_import_root"])
    import poke_engine
    if c15.sha256(Path(poke_engine.poke_engine.__file__)) != manifest["engine_binding_sha256"]:
        raise Cycle17Error("wrong engine binding")
    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous = Path.cwd(); os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main
        RandomBattleTeamDatasets.initialize("gen9"); c15.c14.install_monkeypatches(poke_engine)
        first = c15.materialize_schedules(run_dir, selected, poke_engine, search_main)
        second = c15.materialize_schedules(run_dir, selected, poke_engine, search_main)
        first_hash, second_hash = _schedule_hash(first), _schedule_hash(second)
        if first_hash != second_hash:
            raise Cycle17Error("deterministic mechanics/world preflight disagreed")
        original = first["battle"].request_json
        repaired0, _ = c16.with_offline_rqid(
            original, selected["dependency_cluster_id"], selected["role"],
            int(selected["request_index"]), variant=0,
        )
        repaired1, _ = c16.with_offline_rqid(
            original, selected["dependency_cluster_id"], selected["role"],
            int(selected["request_index"]), variant=1,
        )
        stripped0 = {key: value for key, value in repaired0.items() if key != "rqid"}
        stripped1 = {key: value for key, value in repaired1.items() if key != "rqid"}
        stripped_original = {key: value for key, value in original.items() if key != "rqid"}
        if stripped0 != stripped1 or stripped0 != stripped_original:
            raise Cycle17Error("offline rqid changed private request mechanics")
        result = {
            "dependency_cluster_id": selected["dependency_cluster_id"], "status": "pass",
            **first_hash, "causal_prefix_sha256": selected["causal_prefix_sha256"],
            "typed_reveal_ledger_sha256": selected["typed_reveal_ledger_sha256"],
            "legal_action_contract_sha256": selected["legal_action_contract_sha256"],
            "correlation_stripped_request_sha256": c15.canonical_hash(stripped_original),
        }
    finally:
        os.chdir(previous)
    (run_dir / "measurement/preflight" / f"{index:03d}.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )


def search_worker(run_dir: Path, index: int) -> None:
    preflight = json.loads((run_dir / "measurement/preflight" / f"{index:03d}.json").read_text())
    original = c15.materialize_schedules
    def verified(*args, **kwargs):
        context = original(*args, **kwargs)
        if _schedule_hash(context) != {key: preflight[key] for key in (
            "request_actions", "exact_duration_ms", "exact_world_count", "exact_sha256", "paired_sha256"
        )}:
            raise Cycle17Error("search-time state/worlds disagree with frozen preflight")
        return context
    c15.materialize_schedules = verified
    c15.worker(run_dir, index)


def parent(run_dir: Path, workers: int) -> None:
    verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-40.jsonl").read_text().splitlines()]
    measurement = run_dir / "measurement"
    if measurement.exists(): raise Cycle17Error("Cycle17 measurement exists")
    (measurement / "tmp").mkdir(parents=True); (measurement / "workers").mkdir(); (measurement / "preflight").mkdir()

    def run_commands(mode: str):
        commands = [[sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir),
                     f"--{mode}-index", str(i)] for i in range(c15.PANEL_SIZE)]
        def launch(command): return subprocess.run(command, text=True, capture_output=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, result in enumerate(pool.map(launch, commands), 1):
                if result.returncode: raise Cycle17Error(f"{mode} failed: {result.stderr[-3000:]}")
                if completed % 5 == 0: print(json.dumps({"phase": mode, "completed": completed}), flush=True)

    run_commands("preflight")
    c16._invariant_snapshot = invariant_snapshot_without_routing_rqid
    c16.extract_r1_priors(run_dir, selected)
    verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    run_commands("worker")
    report = c15.summarize(run_dir)
    report["schema"] = "metagross-cycle17-teacher-stability-report/v1"
    report["correlation_repair"] = {
        "double_replay_rows": 40, "stripped_field": "own_legality.rqid",
        "all_policy_inputs_outputs_equal": True, "all_mechanics_world_preflights_equal": True,
        "mismatched_and_stale_rejected": True,
    }
    report["hashes"] = {
        "manifest_sha256": c15.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
        "selection_sha256": c15.sha256(run_dir / "selection-40.jsonl"),
        "r1_priors_sha256": c15.sha256(measurement / "r1-priors.jsonl"),
        "correlation_parity_sha256": c15.sha256(measurement / "correlation-parity.jsonl"),
    }
    (measurement / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-index", type=int); parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.preflight_index is not None: preflight_worker(run_dir, args.preflight_index)
    elif args.worker_index is not None: search_worker(run_dir, args.worker_index)
    else: parent(run_dir, args.workers)


if __name__ == "__main__": main()
