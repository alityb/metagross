#!/usr/bin/env python3
"""Cycle 16 rqid-correlation repair plus unchanged Cycle 15 stability gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experimental.src.scripts import run_cycle15_teacher_stability as c15
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest

ROOT = Path(__file__).resolve().parents[3]


class Cycle16Error(RuntimeError):
    pass


def offline_correlation(
    dependency_cluster_id: str, role: str, chronological_request_index: int, *, variant: int,
) -> tuple[dict[str, Any], int]:
    """Return provenance plus a live-compatible monotone transport rqid.

    The namespace never enters the request or model. The integer is monotone
    within one replay and differs by a fixed base only for the invariance audit.
    """
    if role not in {"p1", "p2"} or chronological_request_index < 0 or variant not in {0, 1}:
        raise Cycle16Error("invalid offline correlation input")
    namespace = hashlib.sha256(
        ("cycle16-offline-rqid-v1\0" + dependency_cluster_id + "\0" + role).encode("utf-8")
    ).hexdigest()
    rqid = chronological_request_index + 1 + variant * 1_000_000
    return {"namespace_sha256": namespace, "variant": variant,
            "chronological_request_index": chronological_request_index}, rqid


def with_offline_rqid(
    request: Mapping[str, Any], dependency_cluster_id: str, role: str,
    chronological_request_index: int, *, variant: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance, rqid = offline_correlation(
        dependency_cluster_id, role, chronological_request_index, variant=variant,
    )
    repaired = copy.deepcopy(dict(request))
    repaired["rqid"] = rqid
    return repaired, provenance


def _invariant_snapshot(response: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = response.get("r1_policy_snapshot")
    if not isinstance(snapshot, dict):
        raise Cycle16Error("dual R1 snapshot missing from correlation audit")
    return {
        "text_tokens": snapshot["text_tokens"],
        "numbers": snapshot["numbers"],
        "illegal_actions": snapshot["illegal_actions"],
        "name_table": snapshot["name_table"],
        "trajectory_observation_rows": snapshot["trajectory"]["observation_rows"],
        "trajectory_rl2": snapshot["trajectory"]["rl2"],
        "trajectory_time_indices": snapshot["trajectory"]["time_indices"],
        "priors": response["priors"], "opponent_priors": response["opp_priors"],
        "probs": response["probs"], "own_legality": response["own_legality"],
    }


def _replay_one(server: Any, ps: Any, selected: Mapping[str, Any], states: Sequence[dict], *, variant: int):
    username = c15._username(states[0]["public_prefix"], selected["role"])
    session = ps.BattleSession(
        f"{selected['battle_id']}-v{variant}", selected["battle_id"],
        f"cycle16-v{variant}", username, server,
    )
    previous: list[str] = []
    target_response = None
    correlation_rows = []
    for index, state in enumerate(states[: int(selected["request_index"]) + 1]):
        current = list(state["public_prefix"])
        for line in c15.c13.prefix_delta(previous, current):
            session.feed_line(line)
        request, provenance = with_offline_rqid(
            state["private_request"], selected["dependency_cluster_id"], selected["role"],
            index, variant=variant,
        )
        correlation_rows.append(provenance)
        session.feed_line("|request|" + json.dumps(request, separators=(",", ":")))
        if state.get("actionable"):
            rqid = request["rqid"]
            request_sha = ps.canonical_request_sha256(request)
            response = session.compute_priors(
                requester_username=username, expected_rqid=rqid,
                expected_request_sha256=request_sha,
            )
            if index == int(selected["request_index"]):
                target_response = response
                # Exact routing defenses must reject both mismatched and stale
                # correlation tokens without mutating the accepted response.
                try:
                    session.compute_priors(
                        requester_username=username, expected_rqid=rqid + 1,
                        expected_request_sha256=request_sha,
                    )
                except RuntimeError:
                    pass
                else:
                    raise Cycle16Error("mismatched response correlation was accepted")
                try:
                    ps.request_cache_status(rqid - 1, rqid)
                except ValueError:
                    pass
                else:
                    raise Cycle16Error("stale response correlation was accepted")
            else:
                action = state.get("chosen_action")
                if not isinstance(action, str) or not action:
                    raise Cycle16Error("causal history lacks observed own action")
                session.acknowledge_action(
                    action, rqid, request_sha, int(response["decision_idx"]),
                )
        previous = current
    if target_response is None:
        raise Cycle16Error("target produced no R1 response")
    return target_response, correlation_rows


def extract_r1_priors(run_dir: Path, selected_rows: Sequence[Mapping[str, Any]]) -> None:
    from srcs.metagross import prior_server as ps
    args = SimpleNamespace(
        local_run_dir=str(ROOT / "srcs/models"), local_run_name="randbats_exit_r1",
        local_base_model="Kakuna", checkpoint=5, agent="Kakuna", username="unused",
        trajectory_mode="causal-history", decision_dump=None,
    )
    server = ps.PriorServer(args)
    server.dual_r1_capture = True
    output = run_dir / "measurement/r1-priors.jsonl"
    parity = run_dir / "measurement/correlation-parity.jsonl"
    with output.open("x", encoding="utf-8") as priors_handle, parity.open("x", encoding="utf-8") as parity_handle:
        for selected in selected_rows:
            raw, public, pov = c15._capture(run_dir, selected)
            derived = c15._derived(selected, public, pov, raw)
            states = derived["states"]
            first, first_correlation = _replay_one(server, ps, selected, states, variant=0)
            second, second_correlation = _replay_one(server, ps, selected, states, variant=1)
            first_invariant = _invariant_snapshot(first)
            second_invariant = _invariant_snapshot(second)
            if first_invariant != second_invariant:
                raise Cycle16Error("changing correlation tokens changed policy input/output")
            row = {
                "dependency_cluster_id": selected["dependency_cluster_id"],
                "checkpoint_sha256": c15.sha256(ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"),
                "trajectory_mode": "causal-history", "priors": first["priors"],
                "opponent_priors": first["opp_priors"],
                "request_sha256": first["request_sha256"],
                "decision_idx": first["decision_idx"], "trajectory": first["trajectory"],
            }
            priors_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            parity_handle.write(json.dumps({
                "dependency_cluster_id": selected["dependency_cluster_id"],
                "status": "pass", "correlation_variant_0": first_correlation,
                "correlation_variant_1": second_correlation,
                "policy_invariant_sha256": c15.canonical_hash(first_invariant),
                "causal_prefix_sha256": selected["causal_prefix_sha256"],
                "legal_action_contract_sha256": selected["legal_action_contract_sha256"],
                "private_request_sha256": selected["private_request_sha256"],
            }, sort_keys=True, separators=(",", ":")) + "\n")
            priors_handle.flush(); parity_handle.flush()


def parent(run_dir: Path, workers: int) -> None:
    verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-40.jsonl").read_text().splitlines()]
    if len(selected) != c15.PANEL_SIZE or len({x["dependency_cluster_id"] for x in selected}) != c15.PANEL_SIZE:
        raise Cycle16Error("frozen selection is not 40 unique roots")
    measurement = run_dir / "measurement"
    if measurement.exists():
        raise Cycle16Error("Cycle16 measurement exists")
    (measurement / "tmp").mkdir(parents=True); (measurement / "workers").mkdir()
    extract_r1_priors(run_dir, selected)
    commands = [[sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir),
                 "--worker-index", str(i)] for i in range(c15.PANEL_SIZE)]
    def launch(command): return subprocess.run(command, text=True, capture_output=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, result in enumerate(pool.map(launch, commands), 1):
            if result.returncode:
                raise Cycle16Error(f"worker failed: {result.stderr[-3000:]}")
            if completed % 5 == 0:
                print(json.dumps({"completed": completed, "total": c15.PANEL_SIZE}), flush=True)
    report = c15.summarize(run_dir)
    report["schema"] = "metagross-cycle16-teacher-stability-report/v1"
    report["correlation_repair"] = {
        "double_replay_rows": 40, "policy_input_output_invariant": True,
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
    parser.add_argument("--worker-index", type=int); parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); run_dir = args.run_dir.resolve()
    if args.worker_index is None: parent(run_dir, args.workers)
    else: c15.worker(run_dir, args.worker_index)


if __name__ == "__main__": main()
