#!/usr/bin/env python3
"""Cycle48 Gate A: observed causal-state target corpus for Interior-v1.

Three phases over 64 frozen TRAIN clusters x 8 chronological observed-state
slots x 2 schedules x 8 worlds = 8,192 scheduled world-rows:

1. resolve   - rematerialization #1: pin the 8 ordinary targets per cluster
               (frozen slot rule), verify Cycle12 compact parity, freeze the
               human behavior anchor. No teacher value is opened.
2. r1        - rematerialization #2: byte-agreement with #1, then causal
               R1 root-control priors per target (control only, never a
               search input here).
3. teacher   - rematerialization #3: byte-agreement again, then per-state
               belief schedules, per-world mechanics/leakage admission, and
               equal-prior 8,192 x2 + 20,000 x1 searches per world. Raw
               posterior weights and full legal N/W/Q are preserved.

Human observed actions are behavior anchors, never strength labels. R1
outputs are separate root controls. Hand-leaf teacher targets test
consistency only, never strength.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts import run_cycle10_full_corpus_index as v10
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from srcs.metagross import causal_reveal_ledger as crl

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
EXCLUDE = ROOT / "experimental/runs/search_native_v2_cycle13_train_rehydration_20260815/selection-200.jsonl"
WORKTREES = ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
HARNESS = ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"
R1_CHECKPOINT = ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
SCHEDULES = 2
WORLDS = 8
SLOTS = 8
ROWS_PER_SLOT = SCHEDULES * WORLDS
ROWS_PER_CLUSTER = SLOTS * ROWS_PER_SLOT
BASE = 2026081648
ARMS = (("equal8192_a", 8192), ("equal8192_b", 8192), ("equal20000", 20000))
GATE = {
    "coverage": 0.95, "unique_fingerprints": 512, "battles": 48,
    "top1_agreement": 0.80, "jsd_median": 0.05, "jsd_p90": 0.15,
}


class Cycle48Error(RuntimeError):
    pass


class SlotUnfilledError(Cycle48Error):
    pass


class HumanAnchorError(Cycle48Error):
    pass


class R1ControlError(Cycle48Error):
    pass


class RematerializeTwiceError(Cycle48Error):
    pass


class CarriedFailure(Cycle48Error):
    """Re-reports an earlier phase's failure class without inventing a new one."""

    def __init__(self, carried_class: str, message: str = "") -> None:
        super().__init__(message or carried_class)
        self.carried = carried_class


def row_id(cluster_index: int, slot: int, schedule: int, world: int) -> str:
    return f"c{cluster_index:02d}:s{slot}:sch{schedule}:w{world}"


def failure_row(cluster_index: int, slot: int, schedule: int, world: int,
                phase: str, exc: BaseException) -> dict[str, Any]:
    detail = f"{type(exc).__name__}:{exc}"
    return {
        "row": row_id(cluster_index, slot, schedule, world), "phase": phase,
        "failure_class": getattr(exc, "carried", type(exc).__name__),
        "failure_category": c13.failure_category(phase, exc),
        "failure_detail_sha256": hashlib.sha256(detail.encode("utf-8")).hexdigest(),
    }


def slot_failures(cluster_index: int, slot: int, phase: str, exc: BaseException) -> list[dict[str, Any]]:
    return [
        failure_row(cluster_index, slot, schedule, world, phase, exc)
        for schedule in range(SCHEDULES) for world in range(WORLDS)
    ]


def ordinary_state(state: Mapping[str, Any]) -> bool:
    """Frozen candidate predicate: admitted Cycle13 ordinary semantics
    (actionable, not a revival prompt) plus an observed own command.

    A dangling final request the human never answered has no behavior anchor
    and cannot enter the corpus. The predicate reads only command presence,
    never its content."""
    if not state.get("actionable"):
        return False
    if (state.get("pp_disable_sidecar") or {}).get("revival_prompt"):
        return False
    chosen = state.get("chosen_action")
    return isinstance(chosen, str) and bool(chosen)


def resolve_slots(candidate_indices: Sequence[int], derived_states: Sequence[Mapping[str, Any]],
                  bases: Sequence[int]) -> list[int | None]:
    """Frozen slot rule: forward-then-backward scan from each base position."""
    resolved: list[int | None] = []
    used: set[int] = set()
    for base in bases:
        order = list(range(base, len(candidate_indices))) + list(range(base - 1, -1, -1))
        pick = None
        for position in order:
            request_index = candidate_indices[position]
            if request_index in used:
                continue
            if ordinary_state(derived_states[request_index]):
                pick = request_index
                break
        if pick is not None:
            used.add(pick)
        resolved.append(pick)
    return resolved


def jsd(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    mid = {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2 for k in keys}
    def kl(p: Mapping[str, float]) -> float:
        return math.fsum(p.get(k, 0.0) * math.log(p.get(k, 0.0) / mid[k]) for k in keys if p.get(k, 0.0) > 0)
    return (kl(a) + kl(b)) / 2


def effective_sample_size(weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    square = math.fsum(w * w for w in weights)
    if total <= 0 or square <= 0:
        raise Cycle48Error("world schedule has no positive posterior mass")
    return total * total / square


def aggregate_policy(world_rows: Sequence[Mapping[str, Any]], legal: Sequence[str]) -> dict[str, Any]:
    """Posterior-weighted schedule policy from raw per-world visit counts."""
    masses = {action: 0.0 for action in legal}
    total_weight = math.fsum(float(row["weight"]) for row in world_rows)
    if total_weight <= 0:
        raise Cycle48Error("world schedule has no positive posterior mass")
    for row in world_rows:
        result = row["result"]
        denominator = max(1, int(result["total_visits"]))
        by_action = {entry["action"]: entry["N"] for entry in result["side_one"]}
        for action in legal:
            masses[action] += float(row["weight"]) * by_action.get(action, 0) / denominator
    masses = {key: value / total_weight for key, value in masses.items()}
    norm = math.fsum(masses.values())
    if norm <= 0:
        raise Cycle48Error("aggregated schedule policy has no mass")
    policy = {key: value / norm for key, value in masses.items()}
    top1 = min(legal, key=lambda action: (-policy[action], action))
    return {"policy": policy, "top1": top1}


def top1_of(priors: Mapping[str, float]) -> str:
    return min(priors, key=lambda action: (-float(priors[action]), action))


def install_request_authority(actions: set[str]) -> list[str]:
    """Cycle46 lesson: install the exact request action set before ANY engine
    root/step/search call for this state. Returns the sorted frozen set."""
    if not actions:
        raise Cycle48Error("empty request action set")
    ordered = sorted(actions)
    c14.CURRENT_ACTIONS[:] = ordered
    return ordered


def result_payload(result: Any) -> dict[str, Any]:
    def side(rows: Sequence[Any]) -> list[dict[str, Any]]:
        return [{
            "action": str(row.move_choice), "N": int(row.visits),
            "W": float(row.total_score),
            "Q": (float(row.total_score) / int(row.visits)) if int(row.visits) else None,
        } for row in rows]
    return {"total_visits": int(result.total_visits), "side_one": side(result.side_one),
            "side_two": side(result.side_two)}


def load_selection(run: Path, dev: bool = False) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (run / "selection-64.jsonl").read_text().splitlines() if line]
    if len({row["dependency_cluster_id"] for row in rows}) != len(rows):
        raise Cycle48Error("selection clusters are not unique")
    if any(row["split"] != "train" for row in rows):
        raise Cycle48Error("non-TRAIN cluster in selection")
    if not dev:
        if len(rows) != 64:
            raise Cycle48Error("frozen selection is not 64 unique clusters")
        excluded = {json.loads(line)["dependency_cluster_id"]
                    for line in EXCLUDE.read_text().splitlines() if line}
        if {row["dependency_cluster_id"] for row in rows} & excluded:
            raise Cycle48Error("frozen selection overlaps retired Cycle13 clusters")
    return rows


def capture_and_derive(run: Path, selected: Mapping[str, Any]) -> tuple[dict, dict[int, dict]]:
    """One full pinned rematerialization of the selected POV battle."""
    if c13.sha256(Path(selected["raw_path"])) != selected["raw_sha256"]:
        raise Cycle48Error("selected raw replay hash changed")
    worktrees = json.loads(WORKTREES.read_text())
    temp = Path(tempfile.mkdtemp(prefix="c48-", dir=run / "measurement/tmp"))
    try:
        out = temp / "capture"
        subprocess.run([
            "node", str(HARNESS), "--showdown", worktrees[selected["showdown_commit"]],
            "--input", selected["raw_path"], "--out-dir", str(out),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        raw = json.loads(Path(selected["raw_path"]).read_text())
        public = json.loads((out / "public.json").read_text())
        pov = json.loads((out / f"{selected['role']}.json").read_text())
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    derived = v12.materialize_role(
        battle_id=selected["battle_id"], role=selected["role"],
        public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
        showdown_commit=selected["showdown_commit"],
    )
    compacts = {row["request_index"]: row for row in v10.compact_states(derived, pov)}
    return derived, compacts


def verify_candidates(selected: Mapping[str, Any], compacts: Mapping[int, dict]) -> str:
    """Cycle12 compact parity for every frozen candidate; returns material hash."""
    ordered = []
    for candidate in selected["candidates"]:
        compact = compacts.get(candidate["request_index"])
        if compact is None:
            raise Cycle48Error("frozen candidate does not map to one compact state")
        c13._state_parity(candidate, compact)
        ordered.append(compact)
    return c13.hash_json(ordered)


def resolve_worker(run: Path, cluster_index: int, dev: bool = False) -> dict[str, Any]:
    selected = load_selection(run, dev)[cluster_index]
    row: dict[str, Any] = {
        "cluster_index": cluster_index,
        "dependency_cluster_id": selected["dependency_cluster_id"],
        "status": "resolved", "slots": [],
    }
    try:
        derived, compacts = capture_and_derive(run, selected)
        row["material_sha256"] = verify_candidates(selected, compacts)
        candidate_indices = [candidate["request_index"] for candidate in selected["candidates"]]
        resolved = resolve_slots(candidate_indices, derived["states"], selected["slot_base_positions"])
        for slot, request_index in enumerate(resolved):
            if request_index is None:
                row["slots"].append({
                    "slot": slot, "status": "failed", "phase": "resolve",
                    "failure_class": "SlotUnfilledError",
                })
                continue
            try:
                target = derived["states"][request_index]
                compact = compacts[request_index]
                actions = c13.request_actions_exact(target["private_request"])
                if actions != set(target["legal_actions"]):
                    raise Cycle48Error("exact request action extractor disagrees with frozen bridge")
                chosen = target.get("chosen_action")
                if not isinstance(chosen, str) or not chosen:
                    raise HumanAnchorError("observed human command is absent")
                if chosen not in actions:
                    raise HumanAnchorError("observed human command is outside the exact action mask")
                row["slots"].append({
                    "slot": slot, "status": "resolved", "request_index": request_index,
                    "fingerprint": compact["model_information_fingerprint_sha256"],
                    "typed_reveal_ledger_sha256": compact["typed_reveal_ledger_sha256"],
                    "request_actions": sorted(actions),
                    "human_action": chosen,
                    "target_sha256": c13.hash_json({
                        "private_request": target["private_request"],
                        "public_prefix": list(target["public_prefix"]),
                    }),
                })
            except BaseException as exc:
                row["slots"].append({
                    "slot": slot, "status": "failed", "phase": "resolve",
                    "failure_class": type(exc).__name__,
                    "failure_category": c13.failure_category("resolve", exc),
                    "failure_detail_sha256": hashlib.sha256(
                        f"{type(exc).__name__}:{exc}".encode()).hexdigest(),
                    "request_index": request_index,
                })
    except BaseException as exc:
        row = {
            "cluster_index": cluster_index,
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "status": "failed", "phase": "resolve",
            "failure_class": type(exc).__name__,
            "failure_category": c13.failure_category("capture", exc),
            "failure_detail_sha256": hashlib.sha256(
                f"{type(exc).__name__}:{exc}".encode()).hexdigest(),
            "slots": [],
        }
    return row


def r1_replay_cluster(ps: Any, c15m: Any, c16m: Any, server: Any,
                      selected: Mapping[str, Any], states: Sequence[dict],
                      targets: set[int], variant: int) -> tuple[dict[int, dict], list[dict]]:
    """One causal R1 replay over all frozen targets of a cluster.

    Raw BattleStream requests lack the live client's routing ``rqid``; the
    admitted Cycle16/17 offline-correlation contract supplies a monotone
    transport token that provably never enters the policy (two-variant
    invariance is checked by the caller). Nothing mechanical is fabricated.
    """
    username = c15m._username(states[0]["public_prefix"], selected["role"])
    session = ps.BattleSession(
        f"{selected['battle_id']}-c48v{variant}", selected["battle_id"],
        f"cycle48-v{variant}", username, server,
    )
    previous: list[str] = []
    responses: dict[int, dict] = {}
    correlation: list[dict] = []
    last = max(targets)
    for index, state in enumerate(states[: last + 1]):
        current = list(state["public_prefix"])
        for line in c13.prefix_delta(previous, current):
            session.feed_line(line)
        request, provenance = c16m.with_offline_rqid(
            state["private_request"], selected["dependency_cluster_id"], selected["role"],
            index, variant=variant,
        )
        correlation.append(provenance)
        session.feed_line("|request|" + json.dumps(request, separators=(",", ":")))
        if state.get("actionable"):
            rqid = request["rqid"]
            request_sha = ps.canonical_request_sha256(request)
            response = session.compute_priors(
                requester_username=username, expected_rqid=rqid,
                expected_request_sha256=request_sha,
            )
            if index in targets:
                responses[index] = response
            if index == last:
                # Admitted Cycle16 routing defenses: mismatched and stale
                # correlation tokens must be rejected without mutating the
                # accepted response.
                try:
                    session.compute_priors(
                        requester_username=username, expected_rqid=rqid + 1,
                        expected_request_sha256=request_sha,
                    )
                except RuntimeError:
                    pass
                else:
                    raise R1ControlError("mismatched response correlation was accepted")
                try:
                    ps.request_cache_status(rqid - 1, rqid)
                except ValueError:
                    pass
                else:
                    raise R1ControlError("stale response correlation was accepted")
            else:
                action = state.get("chosen_action")
                if not isinstance(action, str) or not action:
                    raise R1ControlError("causal R1 history lacks observed own command")
                session.acknowledge_action(
                    action, rqid, request_sha, int(response["decision_idx"]),
                )
        previous = current
    if set(responses) != targets:
        raise R1ControlError("R1 replay missed a frozen target")
    return responses, correlation


def r1_pass(run: Path, selection: Sequence[Mapping[str, Any]],
            resolved_rows: Sequence[Mapping[str, Any]]) -> None:
    """Rematerialization #2 plus dual-variant causal R1 root controls."""
    # Admitted production prior-server environment (srcs/metagross/launch.py):
    # eager CPU inference, no dynamo/inductor compilation.
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["ACCELERATE_USE_CPU"] = "true"
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / "srcs/runtime/metamon-cache"))
    from srcs.metagross import prior_server as ps
    from experimental.src.scripts import run_cycle15_teacher_stability as c15m
    from experimental.src.scripts import run_cycle16_teacher_stability as c16m
    from experimental.src.scripts import run_cycle17_teacher_stability as c17m

    args = SimpleNamespace(
        local_run_dir=str(ROOT / "srcs/models"), local_run_name="randbats_exit_r1",
        local_base_model="Kakuna", checkpoint=5, agent="Kakuna", username="unused",
        trajectory_mode="causal-history", decision_dump=None,
    )
    server = ps.PriorServer(args)
    server.dual_r1_capture = True
    checkpoint_sha256 = c13.sha256(R1_CHECKPOINT)
    output = run / "measurement/r1-controls.jsonl"
    parity_path = run / "measurement/r1-correlation-parity.jsonl"
    with output.open("x", encoding="utf-8") as handle, \
            parity_path.open("x", encoding="utf-8") as parity_handle:
        for row in resolved_rows:
            if row["status"] != "resolved":
                continue
            selected = selection[row["cluster_index"]]
            slots_by_ri = {
                slot["request_index"]: slot
                for slot in row["slots"] if slot["status"] == "resolved"
            }
            base = {
                "cluster_index": row["cluster_index"],
                "dependency_cluster_id": row["dependency_cluster_id"],
                "checkpoint_sha256": checkpoint_sha256,
                "trajectory_mode": "causal-history",
            }
            if not slots_by_ri:
                continue
            try:
                derived, compacts = capture_and_derive(run, selected)
                if verify_candidates(selected, compacts) != row["material_sha256"]:
                    raise RematerializeTwiceError("second rematerialization disagrees with resolve pass")
                states = derived["states"]
                targets = set(slots_by_ri)
                first, correlation0 = r1_replay_cluster(
                    ps, c15m, c16m, server, selected, states, targets, 0)
                second, correlation1 = r1_replay_cluster(
                    ps, c15m, c16m, server, selected, states, targets, 1)
            except BaseException as exc:
                for slot in row["slots"]:
                    if slot["status"] != "resolved":
                        continue
                    handle.write(json.dumps({
                        **base, "slot": slot["slot"], "status": "failed",
                        "failure_class": type(exc).__name__,
                        "failure_detail_sha256": hashlib.sha256(
                            f"{type(exc).__name__}:{exc}".encode()).hexdigest(),
                    }, sort_keys=True, separators=(",", ":")) + "\n")
                continue
            invariants = {}
            for request_index, slot in sorted(slots_by_ri.items()):
                out = dict(base)
                out["slot"] = slot["slot"]
                try:
                    invariant0 = c17m.invariant_snapshot_without_routing_rqid(first[request_index])
                    invariant1 = c17m.invariant_snapshot_without_routing_rqid(second[request_index])
                    if invariant0 != invariant1:
                        raise R1ControlError("changing correlation tokens changed policy input/output")
                    priors = first[request_index].get("priors")
                    if not isinstance(priors, dict) or not priors:
                        raise R1ControlError("R1 target prior is empty")
                    if set(priors) != set(slot["request_actions"]):
                        raise R1ControlError("R1 prior support disagrees with exact request")
                    invariants[slot["slot"]] = c15m.canonical_hash(invariant0)
                    out.update({
                        "status": "pass", "priors": priors,
                        "opponent_priors": first[request_index].get("opp_priors") or {},
                        "request_sha256": first[request_index]["request_sha256"],
                        "decision_idx": first[request_index]["decision_idx"],
                        "policy_invariant_sha256": invariants[slot["slot"]],
                    })
                except BaseException as exc:
                    out.update({
                        "status": "failed", "failure_class": type(exc).__name__,
                        "failure_detail_sha256": hashlib.sha256(
                            f"{type(exc).__name__}:{exc}".encode()).hexdigest(),
                    })
                handle.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
            parity_handle.write(json.dumps({
                "cluster_index": row["cluster_index"],
                "dependency_cluster_id": row["dependency_cluster_id"],
                "correlation_variant_0": correlation0,
                "correlation_variant_1": correlation1,
                "policy_invariant_sha256_by_slot": invariants,
            }, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            parity_handle.flush()


def verify_engine_contract_cycle48(engine: Any) -> None:
    """Require exactly the request-authoritative ABI Gate A uses.

    The frozen Cycle17 binding predates synthetic root-level Tera in plain
    ``root_options``; Tera legality is carried by the exact request set, so the
    contract is checked through the ``_with_s1_request`` surface only.
    """
    for name in ("root_options_with_s1_request", "step_with_uniform_r1_semantic_s1_request",
                 "monte_carlo_tree_search_with_s1_request", "root_options",
                 "step_with_uniform_r1_semantic"):
        if not hasattr(engine, name):
            raise Cycle48Error(f"engine lacks request-authoritative ABI: {name}")
    first = engine.Pokemon(id="pikachu", level=80, moves=[engine.Move(id="thunderbolt")],
                           tera_type="electric")
    reserve = engine.Pokemon(id="eevee", level=80, moves=[engine.Move(id="tackle")],
                             tera_type="normal")
    state = engine.State(side_one=engine.Side(pokemon=[first, reserve]),
                         side_two=engine.Side(pokemon=[first, reserve]))
    request = ["switch eevee", "thunderbolt", "thunderbolt-tera"]
    own, opponent = (list(map(str, rows)) for rows in
                     engine.root_options_with_s1_request(state, request))
    if own != request or not opponent:
        raise Cycle48Error("request-authoritative root options failed engine preflight")
    masked = state.with_side_one_public_reveals(1).with_side_two_public_reveals(2)
    if int(masked.s1_public_reveals) != 1 or int(masked.s2_public_reveals) != 2:
        raise Cycle48Error("engine native mask contract failed preflight")
    payloads = []
    for _repeat in range(2):
        result = engine.monte_carlo_tree_search_with_s1_request(
            state, request, duration_ms=0, iterations=64, threads=1, seed=7,
        )
        payloads.append(result_payload(result))
    if payloads[0] != payloads[1]:
        raise Cycle48Error("seeded engine search is nondeterministic in preflight")
    if payloads[0]["total_visits"] != 64 or {
        entry["action"] for entry in payloads[0]["side_one"]
    } != set(request):
        raise Cycle48Error("request-authoritative search failed engine preflight")


def run_teacher_search(engine: Any, state: Any, actions: Sequence[str], iterations: int,
                       seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = engine.monte_carlo_tree_search_with_s1_request(
        state, list(actions), duration_ms=0, iterations=iterations, threads=1, seed=seed,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    payload = result_payload(result)
    if payload["total_visits"] != iterations:
        raise Cycle48Error("teacher visit count mismatch")
    support = {entry["action"] for entry in payload["side_one"]}
    if support != set(actions):
        raise Cycle48Error("teacher legal/visit support mismatch")
    for entry in payload["side_one"] + payload["side_two"]:
        if not math.isfinite(entry["W"]):
            raise Cycle48Error("teacher produced nonfinite W")
        if entry["N"] > 0 and (entry["Q"] is None or not math.isfinite(entry["Q"])):
            raise Cycle48Error("teacher produced visited action without finite Q")
        if entry["N"] == 0 and entry["Q"] is not None:
            raise Cycle48Error("teacher produced unvisited action with non-null Q")
    return {"iterations": iterations, "seed": seed, "latency_ms": elapsed, "result": payload}


def teacher_worker(run: Path, cluster_index: int, dev: bool) -> dict[str, Any]:
    selection = load_selection(run, dev)
    selected = selection[cluster_index]
    resolved = {
        row["cluster_index"]: row
        for row in map(json.loads, (run / "measurement/resolved-targets.jsonl").read_text().splitlines())
    }[cluster_index]
    r1_rows = {
        (row["cluster_index"], row["slot"]): row
        for row in map(json.loads, (run / "measurement/r1-controls.jsonl").read_text().splitlines())
        if row["cluster_index"] == cluster_index
    }
    raw_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if resolved["status"] != "resolved":
        exc = CarriedFailure(resolved.get("failure_class", "Cycle48Error"), "cluster resolve failure")
        for slot in range(SLOTS):
            failures.extend(slot_failures(cluster_index, slot, resolved.get("phase", "resolve"), exc))
        return {"cluster_index": cluster_index, "scheduled": ROWS_PER_CLUSTER,
                "raw_rows": raw_rows, "cells": cells, "failures": failures}

    if not dev:
        manifest = verify_manifest(run / "PREMEASUREMENT_MANIFEST.json")
    else:
        manifest = None
    sys.path.insert(0, str(ENGINE))
    import poke_engine
    if manifest is not None:
        if c13.sha256(Path(poke_engine.poke_engine.__file__)) != manifest["engine_binding_sha256"]:
            raise Cycle48Error("worker imported wrong engine binding")
    verify_engine_contract_cycle48(poke_engine)
    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous_cwd = Path.cwd()
    os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main
        from config import FoulPlayConfig
        RandomBattleTeamDatasets.initialize("gen9")
        c14.install_monkeypatches(poke_engine)

        try:
            derived, compacts = capture_and_derive(run, selected)
            if verify_candidates(selected, compacts) != resolved["material_sha256"]:
                raise RematerializeTwiceError("teacher rematerialization disagrees with resolve pass")
        except BaseException as exc:
            for slot in range(SLOTS):
                failures.extend(slot_failures(cluster_index, slot, "compact_parity", exc))
            return {"cluster_index": cluster_index, "scheduled": ROWS_PER_CLUSTER,
                    "raw_rows": raw_rows, "cells": cells, "failures": failures}

        for slot_row in resolved["slots"]:
            slot = slot_row["slot"]
            if slot_row["status"] != "resolved":
                failures.extend(slot_failures(
                    cluster_index, slot, slot_row.get("phase", "resolve"),
                    CarriedFailure(slot_row.get("failure_class", "SlotUnfilledError")),
                ))
                continue
            r1_row = r1_rows.get((cluster_index, slot))
            if r1_row is None or r1_row.get("status") != "pass":
                failures.extend(slot_failures(
                    cluster_index, slot, "r1_control",
                    CarriedFailure((r1_row or {}).get("failure_class", "R1ControlError"),
                                   "missing or failed R1 control"),
                ))
                continue
            request_index = slot_row["request_index"]
            fingerprint = slot_row["fingerprint"]
            try:
                target = derived["states"][request_index]
                compact = compacts[request_index]
                if c13.hash_json({
                    "private_request": target["private_request"],
                    "public_prefix": list(target["public_prefix"]),
                }) != slot_row["target_sha256"]:
                    raise RematerializeTwiceError("target request/prefix bytes changed between passes")
                actions = c13.request_actions_exact(target["private_request"])
                if sorted(actions) != slot_row["request_actions"]:
                    raise Cycle48Error("frozen request actions changed between passes")
                sorted_actions = install_request_authority(actions)
                ledger = c13.freeze_ledger(selected["battle_id"], selected["role"], target["public_prefix"])
                if c13.hash_json(ledger.to_payload()) != compact["typed_reveal_ledger_sha256"]:
                    raise Cycle48Error("typed causal ledger disagrees with corrected rematerialization")
                battle = c14.build_foul_play_battle_fixed(
                    battle_id=selected["battle_id"], role=selected["role"],
                    states=derived["states"], target_index=request_index,
                )
                c13.reconcile_public_facts_into_battle(battle, ledger)
                setattr(battle, crl.LEDGER_ATTRIBUTE, ledger.to_payload())
                FoulPlayConfig.parallelism = 8
                FoulPlayConfig.search_time_ms = 500
                exact_count, exact_duration = search_main.search_time_num_battles_randombattles(battle)
                if exact_count not in {16, 32} or exact_duration not in {250, 500}:
                    raise Cycle48Error("production scheduler left frozen 16/32 + 250/500 contract")
            except BaseException as exc:
                failures.extend(slot_failures(cluster_index, slot, "foul_play_rehydration", exc))
                continue

            schedule_states: list[dict[str, Any]] = []
            for schedule_index in range(SCHEDULES):
                seed = c13.stable_seed(BASE, fingerprint, schedule_index)
                try:
                    repeats = []
                    for _repeat in range(2):
                        worlds = search_main.prepare_random_battles(
                            copy.deepcopy(battle), WORLDS, rng=random.Random(seed),
                        )
                        if len(worlds) != WORLDS:
                            raise Cycle48Error("production sampler returned wrong world count")
                        crl.verify_sampled_ledgers(battle, worlds)
                        payloads, weights = [], []
                        for sampled, weight in worlds:
                            payload, _elapsed, _size = c13._world_payload(
                                sampled, ledger, actions, poke_engine, search_main,
                            )
                            payloads.append(payload)
                            weights.append(float(weight))
                        repeats.append({"worlds": payloads, "weights": weights, "sampled": worlds})
                    comparable = [
                        {"worlds": repeat["worlds"], "weights": repeat["weights"]}
                        for repeat in repeats
                    ]
                    if comparable[0] != comparable[1]:
                        raise Cycle48Error("seeded production schedule repeat disagreed")
                    public_hashes = {payload["public_sha256"] for payload in repeats[0]["worlds"]}
                    if len(public_hashes) != 1:
                        raise Cycle48Error("hidden completions changed observer public projection")
                    schedule_states.append({
                        "schedule_index": schedule_index, "seed": seed,
                        "worlds": repeats[0]["sampled"], "payloads": repeats[0]["worlds"],
                        "weights": repeats[0]["weights"],
                        "public_sha256": next(iter(public_hashes)),
                    })
                except BaseException as exc:
                    for world in range(WORLDS):
                        failures.append(failure_row(
                            cluster_index, slot, schedule_index, world, "production_schedules", exc,
                        ))
            if len(schedule_states) == SCHEDULES:
                if len({schedule["public_sha256"] for schedule in schedule_states}) != 1:
                    exc = Cycle48Error("hidden completions changed observer public projection")
                    for schedule in schedule_states:
                        for world in range(WORLDS):
                            failures.append(failure_row(
                                cluster_index, slot, schedule["schedule_index"], world,
                                "production_schedules", exc,
                            ))
                    continue

            for schedule in schedule_states:
                schedule_index = schedule["schedule_index"]
                world_rows: list[dict[str, Any]] = []
                for world_index, (sampled, weight) in enumerate(schedule["worlds"]):
                    try:
                        state = c14.convert_slot_aware(
                            copy.deepcopy(sampled), search_main.battle_to_poke_engine_state,
                            poke_engine, swap=False,
                        )
                        payload = schedule["payloads"][world_index]
                        if hashlib.sha256(state.to_string().encode("utf-8")).hexdigest() != payload["state_sha256"]:
                            raise Cycle48Error("search-state conversion disagrees with audited world payload")
                        arm_results = {}
                        for arm_name, iterations in ARMS:
                            arm_results[arm_name] = run_teacher_search(
                                poke_engine, state, sorted_actions, iterations,
                                c13.stable_seed(BASE, "teacher", fingerprint, schedule_index,
                                                arm_name, world_index) % (2 ** 32),
                            )
                        world_rows.append({
                            "row": row_id(cluster_index, slot, schedule_index, world_index),
                            "cluster_index": cluster_index,
                            "dependency_cluster_id": selected["dependency_cluster_id"],
                            "slot": slot, "request_index": request_index,
                            "information_fingerprint": fingerprint,
                            "schedule_index": schedule_index, "schedule_seed": schedule["seed"],
                            "world_index": world_index, "raw_weight": float(weight),
                            "state_sha256": payload["state_sha256"],
                            "public_sha256": payload["public_sha256"],
                            "legal_action_contract_sha256": payload["legal_action_contract_sha256"],
                            "arms": arm_results,
                        })
                    except BaseException as exc:
                        failures.append(failure_row(
                            cluster_index, slot, schedule_index, world_index, "teacher_search", exc,
                        ))
                if len(world_rows) == WORLDS:
                    weighted = lambda arm: [
                        {"weight": row["raw_weight"], "result": row["arms"][arm]["result"]}
                        for row in world_rows
                    ]
                    agg = {arm: aggregate_policy(weighted(arm), sorted_actions) for arm, _ in ARMS}
                    r1_top1 = top1_of(r1_row["priors"])
                    cells.append({
                        "cluster_index": cluster_index,
                        "dependency_cluster_id": selected["dependency_cluster_id"],
                        "slot": slot, "request_index": request_index,
                        "information_fingerprint": fingerprint,
                        "schedule_index": schedule_index, "schedule_seed": schedule["seed"],
                        "raw_weights": schedule["weights"],
                        "effective_sample_size": effective_sample_size(schedule["weights"]),
                        "aggregates": agg,
                        "repeat_jsd": jsd(agg["equal8192_a"]["policy"], agg["equal8192_b"]["policy"]),
                        "agreement_8192_20000": (
                            agg["equal8192_a"]["top1"] == agg["equal20000"]["top1"]
                            and agg["equal8192_b"]["top1"] == agg["equal20000"]["top1"]
                        ),
                        "human_action": slot_row["human_action"],
                        "human_top1_match_20000": slot_row["human_action"] == agg["equal20000"]["top1"],
                        "r1_top1": r1_top1,
                        "r1_top1_match_20000": r1_top1 == agg["equal20000"]["top1"],
                    })
                raw_rows.extend(world_rows)
    finally:
        os.chdir(previous_cwd)
    return {"cluster_index": cluster_index, "scheduled": ROWS_PER_CLUSTER,
            "raw_rows": raw_rows, "cells": cells, "failures": failures}


def evaluate(selection: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
             dev: bool) -> dict[str, Any]:
    raw_rows = [row for result in results for row in result["raw_rows"]]
    cells = [cell for result in results for cell in result["cells"]]
    failures = [row for result in results for row in result["failures"]]
    scheduled = sum(result["scheduled"] for result in results)
    for result in results:
        if len(result["raw_rows"]) + len(result["failures"]) != result["scheduled"]:
            raise Cycle48Error("scheduled row accounting does not balance")
    supported = len(raw_rows)
    states_with_cells = {(cell["cluster_index"], cell["slot"]) for cell in cells}
    fingerprints = {cell["information_fingerprint"] for cell in cells}
    battles = {cell["dependency_cluster_id"] for cell in cells}
    jsd_values = sorted(cell["repeat_jsd"] for cell in cells)
    jsd_median = statistics.median(jsd_values) if jsd_values else None
    jsd_p90 = (jsd_values[min(len(jsd_values) - 1, math.ceil(0.9 * len(jsd_values)) - 1)]
               if jsd_values else None)
    agreement = (statistics.fmean(cell["agreement_8192_20000"] for cell in cells)
                 if cells else 0.0)
    hidden = [
        row for row in failures
        if row["failure_category"] in {"causal_fact_integrity", "hidden_noninterference"}
    ]
    split_ok = (
        all(row["split"] == "train" for row in selection)
        and len({row["dependency_cluster_id"] for row in selection}) == len(selection)
    )
    if not dev:
        excluded = {json.loads(line)["dependency_cluster_id"]
                    for line in EXCLUDE.read_text().splitlines() if line}
        split_ok = split_ok and not (
            {row["dependency_cluster_id"] for row in selection} & excluded)
    gates = {
        "coverage_ge95": scheduled > 0 and supported / scheduled >= GATE["coverage"],
        "unique_fingerprints_ge512": len(fingerprints) >= GATE["unique_fingerprints"],
        "battles_ge48": len(battles) >= GATE["battles"],
        "top1_agreement_ge80": bool(cells) and agreement >= GATE["top1_agreement"],
        "repeat_jsd_median_le05": jsd_median is not None and jsd_median <= GATE["jsd_median"],
        "repeat_jsd_p90_le15": jsd_p90 is not None and jsd_p90 <= GATE["jsd_p90"],
        "zero_hidden_sensitivity": not hidden,
        "zero_split_leakage": split_ok,
    }
    schedule_half: list[float] = []
    by_state: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cell in cells:
        by_state.setdefault((cell["cluster_index"], cell["slot"]), []).append(cell)
    for state_cells in by_state.values():
        if len(state_cells) != SCHEDULES:
            continue
        halves = []
        for cell in sorted(state_cells, key=lambda value: value["schedule_index"]):
            a = cell["aggregates"]["equal8192_a"]["policy"]
            b = cell["aggregates"]["equal8192_b"]["policy"]
            halves.append({key: (a[key] + b[key]) / 2 for key in a})
        schedule_half.append(jsd(halves[0], halves[1]))
    latencies = [
        arm["latency_ms"] for row in raw_rows for arm in row["arms"].values()
    ]
    report = {
        "schema": "metagross-cycle48-gateA-observed-states/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_smoke": dev,
        "counts": {
            "selected_clusters": len(selection), "scheduled": scheduled,
            "supported": supported,
            "complete_cells": len(cells),
            "states_with_complete_cell": len(states_with_cells),
            "unique_fingerprints": len(fingerprints),
            "battles_with_support": len(battles),
            "failures": len(failures),
        },
        "metrics": {
            "coverage": supported / scheduled if scheduled else 0.0,
            "top1_8192_20000_agreement": agreement,
            "repeat_jsd_median": jsd_median, "repeat_jsd_p90": jsd_p90,
            "schedule_half_soft_policy_jsd_median": (
                statistics.median(schedule_half) if schedule_half else None),
            "schedule_half_soft_policy_jsd_p90": (
                sorted(schedule_half)[min(len(schedule_half) - 1,
                                          math.ceil(0.9 * len(schedule_half)) - 1)]
                if schedule_half else None),
            "effective_sample_size_mean": (
                statistics.fmean(cell["effective_sample_size"] for cell in cells)
                if cells else None),
            "effective_sample_size_min": (
                min(cell["effective_sample_size"] for cell in cells) if cells else None),
            "human_top1_match_20000": (
                statistics.fmean(cell["human_top1_match_20000"] for cell in cells)
                if cells else None),
            "r1_top1_match_20000": (
                statistics.fmean(cell["r1_top1_match_20000"] for cell in cells)
                if cells else None),
            "teacher_search_latency_ms_p50": c13.percentile(latencies, 0.5) if latencies else None,
            "teacher_search_latency_ms_p95": c13.percentile(latencies, 0.95) if latencies else None,
        },
        "gates": gates,
        "failure_classes": {},
        "anchors": {
            "human_actions_are_behavior_anchors_not_strength_labels": True,
            "r1_outputs_are_separate_controls": True,
            "raw_posterior_weights_preserved": True,
            "worlds_never_uniformly_averaged_as_posterior": True,
        },
        "authorization": {
            "gateB_tiny_cpu_training": (not dev) and all(gates.values()),
            "h2h": False, "strength_claim": False, "sealed93": False,
            "gpu_cloud_paid": False,
        },
        "local_cpu_only": True, "gpu_cloud_paid_cost_usd": 0,
        "sealed_93_rows_read": 0,
    }
    counter: dict[str, int] = {}
    for row in failures:
        key = f"{row['phase']}:{row['failure_category']}:{row['failure_class']}"
        counter[key] = counter.get(key, 0) + 1
    report["failure_classes"] = dict(sorted(counter.items(), key=lambda item: -item[1]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dev", action="store_true",
                        help="development smoke without a frozen manifest; never a gate result")
    args = parser.parse_args()
    run = args.run.resolve()
    if not args.dev:
        verify_manifest(run / "PREMEASUREMENT_MANIFEST.json")
    selection = load_selection(run, args.dev)
    measurement = run / "measurement"
    measurement.mkdir(parents=False, exist_ok=False)
    (measurement / "tmp").mkdir()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        resolved_rows = list(pool.map(resolve_worker, [run] * len(selection),
                                      range(len(selection)), [args.dev] * len(selection),
                                      chunksize=1))
    with (measurement / "resolved-targets.jsonl").open("x") as handle:
        for row in sorted(resolved_rows, key=lambda value: value["cluster_index"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "resolve", "resolved": sum(
        row["status"] == "resolved" for row in resolved_rows)}), flush=True)

    r1_pass(run, selection, sorted(resolved_rows, key=lambda value: value["cluster_index"]))
    print(json.dumps({"phase": "r1", "rows": len(
        (measurement / "r1-controls.jsonl").read_text().splitlines())}), flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(teacher_worker, [run] * len(selection),
                                range(len(selection)), [args.dev] * len(selection), chunksize=1))
    results.sort(key=lambda value: value["cluster_index"])
    with (measurement / "raw-state-world-targets.jsonl").open("x") as handle:
        for result in results:
            for row in result["raw_rows"]:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
    with (measurement / "cell-aggregates.jsonl").open("x") as handle:
        for result in results:
            for cell in result["cells"]:
                handle.write(json.dumps(cell, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
    with (measurement / "failures.jsonl").open("x") as handle:
        for result in results:
            for row in result["failures"]:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    report = evaluate(selection, results, args.dev)
    if not args.dev:
        verify_manifest(run / "PREMEASUREMENT_MANIFEST.json")
        report["post_run_frozen_integrity"] = True
    report["hashes"] = {
        "selection_sha256": c13.sha256(run / "selection-64.jsonl"),
        "resolved_targets_sha256": c13.sha256(measurement / "resolved-targets.jsonl"),
        "r1_controls_sha256": c13.sha256(measurement / "r1-controls.jsonl"),
        "r1_correlation_parity_sha256": c13.sha256(measurement / "r1-correlation-parity.jsonl"),
        "raw_targets_sha256": c13.sha256(measurement / "raw-state-world-targets.jsonl"),
        "cell_aggregates_sha256": c13.sha256(measurement / "cell-aggregates.jsonl"),
    }
    if not args.dev:
        report["hashes"]["manifest_sha256"] = c13.sha256(run / "PREMEASUREMENT_MANIFEST.json")
    (measurement / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
