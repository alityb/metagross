#!/usr/bin/env python3
"""Cycle 15 TRAIN-only search-budget stability gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from srcs.metagross import causal_reveal_ledger as crl

ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "metagross-cycle15-teacher-stability/v1"
PANEL_SIZE = 40
WORLD_COUNT = 8
SCHEDULES = 2
REPEATS = 2
BASE_SEED = 0xC15A11CE
ARMS = {
    "equal_2048": {"iterations": 2048, "r1": False},
    "equal_8192": {"iterations": 8192, "r1": False},
    "equal_20000": {"iterations": 20000, "r1": False},
    "r1_20000": {"iterations": 20000, "r1": True},
}


class Cycle15Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _capture(run_dir: Path, selected: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    worktrees = json.loads((
        ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
    ).read_text())
    temp = Path(tempfile.mkdtemp(prefix="cycle15-", dir=run_dir / "measurement/tmp"))
    out = temp / "capture"
    subprocess.run([
        "node", str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
        "--showdown", worktrees[selected["showdown_commit"]],
        "--input", selected["raw_path"], "--out-dir", str(out),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    raw = json.loads(Path(selected["raw_path"]).read_text())
    public = json.loads((out / "public.json").read_text())
    pov = json.loads((out / f"{selected['role']}.json").read_text())
    return raw, public, pov


def _derived(selected: Mapping[str, Any], public: dict, pov: dict, raw: dict) -> dict:
    derived = v12.materialize_role(
        battle_id=selected["battle_id"], role=selected["role"],
        public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
        showdown_commit=selected["showdown_commit"],
    )
    compact = c13._compact_state(derived, pov, selected["request_index"])
    c13._state_parity(selected, compact)
    return derived


def _username(prefix: Sequence[str], role: str) -> str:
    matches = []
    for line in prefix:
        fields = line.split("|")
        if len(fields) >= 4 and fields[1] == "player" and fields[2] == role:
            matches.append(fields[3])
    if not matches:
        raise Cycle15Error("causal prefix lacks observer player identity")
    return matches[-1]


def extract_r1_priors(run_dir: Path, selected_rows: Sequence[Mapping[str, Any]]) -> None:
    """Replay genuine public/private decision history through the frozen R1."""
    from srcs.metagross import prior_server as ps

    args = SimpleNamespace(
        local_run_dir=str(ROOT / "srcs/models"), local_run_name="randbats_exit_r1",
        local_base_model="Kakuna", checkpoint=5, agent="Kakuna", username="unused",
        trajectory_mode="causal-history", decision_dump=None,
    )
    server = ps.PriorServer(args)
    output = run_dir / "measurement/r1-priors.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for selected in selected_rows:
            raw, public, pov = _capture(run_dir, selected)
            derived = _derived(selected, public, pov, raw)
            states = derived["states"]
            target_index = int(selected["request_index"])
            username = _username(states[0]["public_prefix"], selected["role"])
            session = ps.BattleSession(
                selected["battle_id"], selected["battle_id"], "cycle15", username, server,
            )
            previous: list[str] = []
            target_response = None
            for index, state in enumerate(states[: target_index + 1]):
                current = list(state["public_prefix"])
                for line in c13.prefix_delta(previous, current):
                    session.feed_line(line)
                request = state["private_request"]
                session.feed_line("|request|" + json.dumps(request, separators=(",", ":")))
                if state.get("actionable"):
                    rqid = request.get("rqid")
                    if isinstance(rqid, bool) or not isinstance(rqid, int):
                        raise Cycle15Error("actionable request lacks integer rqid")
                    request_sha = ps.canonical_request_sha256(request)
                    response = session.compute_priors(
                        requester_username=username, expected_rqid=rqid,
                        expected_request_sha256=request_sha,
                    )
                    if index == target_index:
                        target_response = response
                    else:
                        action = state.get("chosen_action")
                        if not isinstance(action, str) or not action:
                            raise Cycle15Error("causal R1 history lacks observed own command")
                        session.acknowledge_action(
                            action, rqid, request_sha, int(response["decision_idx"]),
                        )
                previous = current
            if target_response is None:
                raise Cycle15Error("selected target did not produce an R1 prior")
            priors = target_response.get("priors")
            if not isinstance(priors, dict) or not priors:
                raise Cycle15Error("R1 target prior is empty")
            row = {
                "dependency_cluster_id": selected["dependency_cluster_id"],
                "checkpoint_sha256": sha256(ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"),
                "trajectory_mode": "causal-history",
                "priors": priors,
                "opponent_priors": target_response.get("opp_priors") or {},
                "request_sha256": target_response["request_sha256"],
                "decision_idx": target_response["decision_idx"],
                "trajectory": target_response["trajectory"],
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()


def _state_string(sampled: Any, ledger: Any, actions: set[str], engine: Any, search_main: Any) -> str:
    state = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled), swap=False)
    bits = c14.compile_slot_aware_bits(state, ledger, swap=False)
    state = crl.install_observer_mask(state, bits, swap=False, engine=engine)
    authoritative, opponent = engine.root_options_with_s1_request(state, sorted(actions))
    if set(authoritative) != actions or not opponent:
        raise Cycle15Error("request-authoritative root options disagree")
    return state.to_string()


def materialize_schedules(run_dir: Path, selected: Mapping[str, Any], engine: Any, search_main: Any):
    raw, public, pov = _capture(run_dir, selected)
    derived = _derived(selected, public, pov, raw)
    target = derived["states"][selected["request_index"]]
    actions = c13.request_actions_exact(target["private_request"])
    ledger = c13.freeze_ledger(selected["battle_id"], selected["role"], target["public_prefix"])
    battle = c14.build_foul_play_battle_fixed(
        battle_id=selected["battle_id"], role=selected["role"],
        states=derived["states"], target_index=selected["request_index"],
    )
    c13.reconcile_public_facts_into_battle(battle, ledger)
    setattr(battle, c13.LEDGER_ATTRIBUTE, ledger.to_payload())

    from config import FoulPlayConfig
    FoulPlayConfig.parallelism = 8
    FoulPlayConfig.search_time_ms = 500
    exact_count, exact_duration = search_main.search_time_num_battles_randombattles(battle)
    if exact_count not in {16, 32} or exact_duration not in {250, 500}:
        raise Cycle15Error("production scheduler left frozen 16/32 + 250/500 contract")

    def make(count: int, channel: str) -> list[dict[str, Any]]:
        rows = []
        for schedule_index in range(SCHEDULES):
            seed = stable_seed(BASE_SEED, selected["dependency_cluster_id"], channel, schedule_index)
            worlds = search_main.prepare_random_battles(
                copy.deepcopy(battle), count, rng=random.Random(seed),
            )
            c13.verify_sampled_ledgers(battle, worlds)
            payload = [
                {"state": _state_string(sampled, ledger, actions, engine, search_main), "weight": float(weight)}
                for sampled, weight in worlds
            ]
            rows.append({"schedule_index": schedule_index, "seed": seed, "worlds": payload})
        return rows

    return {
        "request_actions": sorted(actions),
        "exact_duration_ms": exact_duration,
        "exact_world_count": exact_count,
        "exact": make(exact_count, "production-exact"),
        "paired": make(WORLD_COUNT, "paired-cycle14"),
        "battle": battle,
    }


def _result_payload(result: Any) -> dict[str, Any]:
    def side(rows: Sequence[Any]) -> list[dict[str, Any]]:
        return [{
            "action": str(row.move_choice), "N": int(row.visits),
            "W": float(row.total_score),
            "Q": (float(row.total_score) / int(row.visits)) if int(row.visits) else None,
        } for row in rows]
    return {"total_visits": int(result.total_visits), "side_one": side(result.side_one),
            "side_two": side(result.side_two)}


def _aggregate(world_rows: Sequence[dict[str, Any]], legal: Sequence[str]) -> dict[str, Any]:
    masses = {action: 0.0 for action in legal}
    total_weight = math.fsum(float(row["weight"]) for row in world_rows)
    for row in world_rows:
        result = row["result"]
        denominator = max(1, int(result["total_visits"]))
        by_action = {entry["action"]: entry["N"] for entry in result["side_one"]}
        for action in legal:
            masses[action] += float(row["weight"]) * by_action.get(action, 0) / denominator
    if total_weight <= 0:
        raise Cycle15Error("world schedule has no positive mass")
    masses = {key: value / total_weight for key, value in masses.items()}
    norm = math.fsum(masses.values())
    policy = {key: value / norm for key, value in masses.items()}
    ordered = sorted(legal, key=lambda action: (-policy[action], action))
    return {"policy": policy, "top1": ordered[0], "topk": ordered[: min(3, len(ordered))],
            "gap": policy[ordered[0]] - (policy[ordered[1]] if len(ordered) > 1 else 0.0)}


def run_searches(engine: Any, schedules: Sequence[dict], actions: list[str], arm: str,
                 priors: dict, duration_ms: int = 0) -> list[dict[str, Any]]:
    rows = []
    spec = ARMS.get(arm)
    for schedule in schedules:
        for repeat in range(REPEATS):
            worlds = []
            for world_index, world in enumerate(schedule["worlds"]):
                state = engine.State.from_string(world["state"])
                kwargs: dict[str, Any] = {}
                if arm.startswith("production") or (spec and spec["r1"]):
                    kwargs["s1_priors"] = list(priors["priors"].items())
                    if priors.get("opponent_priors"):
                        kwargs["s2_priors"] = list(priors["opponent_priors"].items())
                    kwargs["c_puct"] = 2.0
                if spec:
                    iterations = int(spec["iterations"])
                    seed = stable_seed(BASE_SEED, arm, schedule["seed"], repeat, world_index)
                    call = {"duration_ms": 0, "iterations": iterations, "threads": 1, "seed": seed}
                else:
                    call = {"duration_ms": duration_ms, "iterations": 0, "threads": 1}
                started = time.perf_counter()
                result = engine.monte_carlo_tree_search_with_s1_request(
                    state, actions, **call, **kwargs,
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                worlds.append({"world_index": world_index, "weight": world["weight"],
                               "latency_ms": elapsed, "result": _result_payload(result)})
            rows.append({"schedule_index": schedule["schedule_index"], "repeat": repeat,
                         "worlds": worlds, **_aggregate(worlds, actions)})
    return rows


def worker(run_dir: Path, index: int) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-40.jsonl").read_text().splitlines()][index]
    prior_map = {row["dependency_cluster_id"]: row for row in map(json.loads, (
        run_dir / "measurement/r1-priors.jsonl").read_text().splitlines())}
    wheel_unpack = Path(manifest["engine_import_root"])
    sys.path.insert(0, str(wheel_unpack))
    import poke_engine
    if sha256(Path(poke_engine.poke_engine.__file__)) != manifest["engine_binding_sha256"]:
        raise Cycle15Error("worker imported wrong engine binding")
    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous = Path.cwd(); os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main
        RandomBattleTeamDatasets.initialize("gen9")
        c14.install_monkeypatches(poke_engine)
        context = materialize_schedules(run_dir, selected, poke_engine, search_main)
        prior = copy.deepcopy(prior_map[selected["dependency_cluster_id"]])
        if set(prior["priors"]) != set(context["request_actions"]):
            raise Cycle15Error("reconstructed R1 prior support disagrees with exact request")
        from srcs.metagross.run_foul_play import sanitize_opponent_priors
        sanitized = sanitize_opponent_priors(context["battle"], prior.get("opponent_priors"))
        prior["opponent_priors"] = dict(sanitized or ())
        exact = run_searches(
            poke_engine, context["exact"], context["request_actions"],
            "production_exact", prior, context["exact_duration_ms"],
        )
        paired = run_searches(
            poke_engine, context["paired"], context["request_actions"],
            "production_paired", prior, context["exact_duration_ms"],
        )
        arms = {name: run_searches(
            poke_engine, context["paired"], context["request_actions"], name, prior,
        ) for name in ARMS}
        row = {
            "schema": RUN_SCHEMA, "status": "pass",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "model_information_fingerprint_sha256": selected["model_information_fingerprint_sha256"],
            "request_actions": context["request_actions"],
            "production_scheduler": {"duration_ms": context["exact_duration_ms"],
                                     "world_count": context["exact_world_count"]},
            "production_exact": exact, "production_paired": paired, "arms": arms,
        }
    finally:
        os.chdir(previous)
    path = run_dir / "measurement/workers" / f"{index:03d}.json"
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _modal(rows: Sequence[dict[str, Any]]) -> str:
    counts = Counter(row["top1"] for row in rows)
    return min(counts, key=lambda action: (-counts[action], action))


def _tv(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return 0.5 * math.fsum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _jsd(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b); m = {k: (a.get(k, 0) + b.get(k, 0)) / 2 for k in keys}
    def kl(p, q):
        return math.fsum(p.get(k, 0) * math.log(p.get(k, 0) / q[k]) for k in keys if p.get(k, 0) > 0)
    return (kl(a, m) + kl(b, m)) / 2


def arm_metrics(root_rows: Sequence[dict], arm: str) -> dict[str, Any]:
    root_entries = []
    for root in root_rows:
        cells = root["arms"][arm]
        production = root["production_exact"]
        top = _modal(cells); prod_top = _modal(production)
        stable = len({cell["top1"] for cell in cells}) == 1
        prod_stable = len({cell["top1"] for cell in production}) == 1
        schedule_top = [_modal([cell for cell in cells if cell["schedule_index"] == s]) for s in range(2)]
        repeat_pairs = [(cells[0], cells[1]), (cells[2], cells[3])]
        root_entries.append({
            "top": top, "production_top": prod_top, "stable": stable,
            "production_stable": prod_stable, "schedule_agree": schedule_top[0] == schedule_top[1],
            "differs": top != prod_top,
            "stable_difference": stable and prod_stable and top != prod_top,
            "repeat_tv": statistics.fmean(_tv(a["policy"], b["policy"]) for a, b in repeat_pairs),
            "repeat_jsd": statistics.fmean(_jsd(a["policy"], b["policy"]) for a, b in repeat_pairs),
            "mean_gap": statistics.fmean(cell["gap"] for cell in cells),
        })
    return {
        "all_cell_top1_stability": statistics.fmean(x["stable"] for x in root_entries),
        "schedule_half_top1_agreement": statistics.fmean(x["schedule_agree"] for x in root_entries),
        "difference_from_production_exact": statistics.fmean(x["differs"] for x in root_entries),
        "stable_differences_from_production_exact": sum(x["stable_difference"] for x in root_entries),
        "mean_repeat_tv": statistics.fmean(x["repeat_tv"] for x in root_entries),
        "mean_repeat_jsd": statistics.fmean(x["repeat_jsd"] for x in root_entries),
        "mean_action_gap": statistics.fmean(x["mean_gap"] for x in root_entries),
        "modal_top1": [x["top"] for x in root_entries],
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    rows = [json.loads((run_dir / "measurement/workers" / f"{i:03d}.json").read_text()) for i in range(PANEL_SIZE)]
    metrics = {arm: arm_metrics(rows, arm) for arm in ARMS}
    reference = metrics["equal_20000"]["modal_top1"]
    for arm in ARMS:
        own = metrics[arm]["modal_top1"]
        metrics[arm]["top1_agreement_with_equal_20000"] = statistics.fmean(a == b for a, b in zip(own, reference))
        del metrics[arm]["modal_top1"]
    eligible = []
    for arm in ("equal_2048", "equal_8192", "equal_20000", "r1_20000"):
        row = metrics[arm]
        if (row["all_cell_top1_stability"] >= 0.70
                and row["schedule_half_top1_agreement"] >= 0.80
                and row["top1_agreement_with_equal_20000"] >= 0.80
                and row["mean_repeat_jsd"] <= 0.10
                and row["stable_differences_from_production_exact"] >= 4):
            eligible.append(arm)
    selected = eligible[0] if eligible else None
    latencies = {}
    for name in ["production_exact", "production_paired", *ARMS]:
        values = []
        for root in rows:
            cells = root[name] if name.startswith("production") else root["arms"][name]
            values.extend(world["latency_ms"] for cell in cells for world in cell["worlds"])
        latencies[name] = {"mean": statistics.fmean(values), "p95": sorted(values)[int(.95 * (len(values)-1))]}
    verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    return {
        "schema": "metagross-cycle15-teacher-stability-report/v1",
        "status": "pass" if selected else "fail",
        "selected_candidate": selected,
        "authorization": {"prospective_h2h": bool(selected), "training": False,
                          "target_collection": False, "validation_test": False, "sealed93": False},
        "counts": {"roots": len(rows), "dependency_clusters": len({x['dependency_cluster_id'] for x in rows})},
        "metrics": metrics, "latency_ms": latencies,
        "interpretation": "Stability and policy difference only; all Q values reuse the hand evaluator and are self-referential, not strength evidence.",
        "production_controls": {
            "P_exact": "actual frozen adaptive 250/500ms + 16/32-world R1-prior scheduler; independent seed tape",
            "P_paired": "R1-prior duration control on the fixed 8-world paired schedules; diagnostic only",
        },
        "local_cpu_only": True, "gpu_cloud_paid_cost_usd": 0,
    }


def parent(run_dir: Path, workers: int) -> None:
    verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-40.jsonl").read_text().splitlines()]
    if len(selected) != PANEL_SIZE or len({x["dependency_cluster_id"] for x in selected}) != PANEL_SIZE:
        raise Cycle15Error("frozen selection is not 40 unique clusters")
    measurement = run_dir / "measurement"
    if measurement.exists():
        raise Cycle15Error("Cycle15 measurement already exists")
    (measurement / "tmp").mkdir(parents=True)
    (measurement / "workers").mkdir()
    extract_r1_priors(run_dir, selected)
    commands = [[sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir),
                 "--worker-index", str(index)] for index in range(PANEL_SIZE)]
    def launch(command):
        return subprocess.run(command, text=True, capture_output=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, completed in enumerate(pool.map(launch, commands), 1):
            if completed.returncode:
                raise Cycle15Error(f"worker failed: {completed.stderr[-2000:]}")
            if index % 5 == 0:
                print(json.dumps({"completed": index, "total": PANEL_SIZE}), flush=True)
    report = summarize(run_dir)
    report["hashes"] = {
        "manifest_sha256": sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
        "selection_sha256": sha256(run_dir / "selection-40.jsonl"),
        "r1_priors_sha256": sha256(measurement / "r1-priors.jsonl"),
    }
    (measurement / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); run_dir = args.run_dir.resolve()
    if args.worker_index is None:
        parent(run_dir, args.workers)
    else:
        worker(run_dir, args.worker_index)


if __name__ == "__main__":
    main()
