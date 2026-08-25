#!/usr/bin/env python3
"""Fail-closed monitor for Cycle27's fresh second-decision live smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    rows,
    to_id,
    validate_teacher,
)
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import (
    attest_registered_battle,
)


CONTROLLER = "metagross-cycle19-equal8192-production-selector/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_rows(search_path: Path) -> list[dict]:
    return [
        row
        for row in rows(search_path)
        if ((row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get(
            "controller_schema"
        )
        == CONTROLLER
    ]


def observer_role(protocol: list[dict], username: str) -> str:
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if (
                len(parts) >= 4
                and parts[1] == "player"
                and to_id(parts[3]) == to_id(username)
                and parts[2] in {"p1", "p2"}
            ):
                return parts[2]
    raise RuntimeError("Cycle27 cannot identify candidate role")


def selected_action_window(
    protocol: list[dict], decision_ns: int, username: str, selected: str
) -> dict:
    requests: list[tuple[int, dict]] = []
    sends: list[tuple[int, list]] = []
    for row in protocol:
        stamp = int(row.get("time_ns", 0))
        if row.get("direction") in {"received", "reconnect_received"}:
            for line in str(row.get("message", "")).splitlines():
                if line.startswith("|request|") and stamp < decision_ns:
                    requests.append(
                        (stamp, json.loads(line.removeprefix("|request|")))
                    )
        if (
            row.get("direction") == "sent"
            and stamp > decision_ns
            and isinstance(row.get("messages"), list)
            and row["messages"]
            and str(row["messages"][0]).startswith(("/choose move ", "/switch "))
        ):
            sends.append((stamp, row["messages"]))
    if not requests or not sends:
        raise RuntimeError("Cycle27 request/send boundary is incomplete")
    request_ns, request = requests[-1]
    send_ns, messages = sends[0]
    rqid = str(request.get("rqid"))
    if not request_ns < decision_ns < send_ns:
        raise RuntimeError("Cycle27 decision lies outside request/send window")
    if len(messages) != 2 or str(messages[1]) != rqid:
        raise RuntimeError("Cycle27 command rqid differs from exact request")

    role = observer_role(protocol, username)
    target = to_id(selected.removeprefix("switch ").removesuffix("-tera"))
    execution_ns = None
    public_line = None
    for row in protocol:
        stamp = int(row.get("time_ns", 0))
        if stamp <= send_ns or row.get("direction") not in {
            "received",
            "reconnect_received",
        }:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if selected.startswith("switch "):
                matched = (
                    len(parts) >= 4
                    and parts[1] in {"switch", "drag", "replace"}
                    and parts[2].startswith(f"{role}a:")
                    and to_id(parts[3].split(",", 1)[0]) == target
                )
            else:
                matched = (
                    len(parts) >= 4
                    and parts[1] == "move"
                    and parts[2].startswith(f"{role}a:")
                    and to_id(parts[3]) == target
                )
            if matched:
                execution_ns, public_line = stamp, line
                break
        if execution_ns is not None:
            break
    if execution_ns is None:
        raise RuntimeError("Cycle27 selected action was not publicly executed")
    return {
        "rqid": rqid,
        "request_time_ns": request_ns,
        "decision_time_ns": decision_ns,
        "send_time_ns": send_ns,
        "public_execution_time_ns": execution_ns,
        "public_line": public_line,
        "observer_role": role,
    }


def opponent_move_before_second_root(
    protocol: list[dict], *, role: str, before_ns: int
) -> dict:
    opponent = "p2" if role == "p1" else "p1"
    found = []
    for row in protocol:
        stamp = int(row.get("time_ns", 0))
        if stamp >= before_ns or row.get("direction") not in {
            "received",
            "reconnect_received",
        }:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if (
                len(parts) >= 4
                and parts[1] == "move"
                and parts[2].startswith(f"{opponent}a:")
            ):
                found.append({"time_ns": stamp, "line": line, "move": to_id(parts[3])})
    if not found:
        raise RuntimeError("Cycle27 second root has no preceding opponent move")
    return found[-1]


def validate_move_receipts(
    run: Path, identity: dict, boundary: dict
) -> dict:
    files = sorted((run / "move-receipts").glob("agenta-*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("Cycle27 requires one candidate move-receipt file")
    production: list[dict] = []
    candidate: list[dict] = []
    protocol_hashes: set[str] = set()
    boundary_ns = int(boundary["public_execution_time_ns"])
    authority_counts = {"causal_disable": 0, "world_mechanical_disable": 0}
    receipt_moves = 0
    for row in rows(files[0]):
        context = row.get("execution_context")
        if not isinstance(context, dict) or any(
            context.get(key) != value for key, value in identity.items()
        ):
            continue
        if row.get("schema") != "metagross-causal-move-conversion-receipt/v1":
            raise RuntimeError("Cycle27 move receipt schema changed")
        if row.get("observer_role") != "p1" or row.get("swap") is not False:
            raise RuntimeError("Cycle27 move receipt has wrong perspective")
        if int(row.get("receipt_time_ns", 0)) > boundary_ns:
            raise RuntimeError("Cycle27 move receipt occurred after public execution")
        protocol_hashes.add(str(row.get("protocol_sha256")))
        move_receipt = row.get("move_receipt") or {}
        if (
            move_receipt.get("protocol_sha256") != row.get("protocol_sha256")
            or move_receipt.get("battle_tag") != identity["battle_tag"]
            or not isinstance(move_receipt.get("moves"), list)
        ):
            raise RuntimeError("Cycle27 nested move receipt identity changed")
        if not move_receipt["moves"]:
            raise RuntimeError("Cycle27 second root did not carry a revealed move")
        for move in move_receipt["moves"]:
            authority = move.get("disable_authority")
            if authority not in authority_counts:
                raise RuntimeError("Cycle27 move has unknown disable authority")
            if (
                isinstance(move.get("current_pp"), bool)
                or not isinstance(move.get("current_pp"), int)
                or isinstance(move.get("max_pp"), bool)
                or not isinstance(move.get("max_pp"), int)
                or not 0 <= move["current_pp"] <= move["max_pp"]
                or not isinstance(move.get("world_disabled"), bool)
            ):
                raise RuntimeError("Cycle27 move PP/disable receipt is malformed")
            if authority == "causal_disable" and move.get("world_disabled") is not True:
                raise RuntimeError("Cycle27 causal disable did not reach search state")
            authority_counts[authority] += 1
            receipt_moves += 1
        phase = context.get("phase")
        cohort = context.get("cohort")
        if phase == "production_control" and cohort == "adaptive_root_search":
            production.append(context)
        elif phase == "equal8192_candidate" and cohort == "fixed_two_by_eight":
            candidate.append(context)
        else:
            raise RuntimeError("Cycle27 move receipt has unknown execution cohort")
    if len(protocol_hashes) != 1:
        raise RuntimeError("Cycle27 second-root receipts span causal prefixes")
    declared = {row.get("declared_world_count") for row in production}
    if len(declared) != 1 or next(iter(declared)) not in {16, 32}:
        raise RuntimeError("Cycle27 production world declaration is invalid")
    production_count = int(next(iter(declared)))
    if (
        len(production) != production_count
        or {row.get("conversion_index") for row in production}
        != set(range(production_count))
    ):
        raise RuntimeError("Cycle27 production move receipts do not reconcile")
    cells = {(row.get("schedule_index"), row.get("world_index")) for row in candidate}
    if (
        len(candidate) != 16
        or cells != {(s, w) for s in range(2) for w in range(8)}
        or {row.get("conversion_index") for row in candidate} != set(range(16))
        or any(row.get("declared_world_count") != 16 for row in candidate)
    ):
        raise RuntimeError("Cycle27 candidate move receipts are not exact 2x8")
    return {
        "path": str(files[0].resolve()),
        "sha256": sha(files[0]),
        "protocol_sha256": next(iter(protocol_hashes)),
        "production_receipts": production_count,
        "candidate_receipts": 16,
        "candidate_cells": sorted([list(cell) for cell in cells]),
        "move_rows": receipt_moves,
        "disable_authority_counts": authority_counts,
        "all_exact_pp_and_engine_world_disable": True,
        "all_receipts_pre_public_execution": True,
    }


def engine_provenance(run: Path, expected_engine_sha: str) -> list[dict] | None:
    result = []
    for namespace in ("agent_a", "agent_b"):
        path = run / "engine-receipts" / f"{namespace}-engine-provenance.json"
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text())
        provenance = receipt.get("provenance") or {}
        if (
            receipt.get("namespace") != namespace
            or provenance.get("native_sha256") != expected_engine_sha
            or provenance.get("native_reveal_masks") is not True
            or provenance.get("mode") != "exact_pinned_experimental_runtime"
        ):
            raise RuntimeError("Cycle27 spawned process engine provenance changed")
        result.append({"path": str(path.resolve()), "sha256": sha(path)})
    return result


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    provenance = engine_provenance(run, expected_engine_sha)
    boundary_path = run / "PUBLIC_EXECUTION_BOUNDARY.json"
    if provenance is None or not boundary_path.is_file():
        return None
    registered = attest_registered_battle(run, pair_manifest)
    boundary = json.loads(boundary_path.read_text())
    candidate_files = []
    for search_path in sorted((run / "smoke-logs").glob("*.search.jsonl")):
        candidates = candidate_rows(search_path)
        if candidates:
            candidate_files.append((search_path, candidates))
    if len(candidate_files) != 1:
        raise RuntimeError("Cycle27 candidate identity is not unique")
    search_path, candidates = candidate_files[0]
    second = [
        row
        for row in candidates
        if (row.get("context") or {}).get("decision_idx") == 1
    ]
    if len(second) != 1:
        return None
    row = second[0]
    teacher = (row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}
    validate_teacher(teacher)
    selected = str(teacher["selected_action"])
    override = row.get("choice_override") or {}
    if row.get("choice") != selected or override.get("final_choice") != selected:
        raise RuntimeError("Cycle27 candidate selected action was not installed")
    if str(teacher.get("reason", "")).startswith("fail_closed"):
        raise RuntimeError("Cycle27 candidate fell back")

    context = row.get("context") or {}
    tag = context.get("tag")
    rqid = context.get("rqid")
    decision_index = context.get("decision_idx")
    root_id = hashlib.sha256(
        f"terminal-mcts-live\0{tag}\0{decision_index}".encode()
    ).hexdigest()
    identity = {
        "battle_tag": tag,
        "rqid": rqid,
        "decision_index": decision_index,
        "root_id": root_id,
    }
    if (
        any(boundary.get(key) != value for key, value in identity.items())
        or boundary.get("selected_action") != selected
        or int(boundary.get("decision_index", -1)) != 1
    ):
        raise RuntimeError("Cycle27 public boundary identity differs from second root")
    protocol_path = search_path.with_name(
        search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
    )
    protocol = rows(protocol_path)
    window = selected_action_window(
        protocol, int(row["time_ns"]), search_path.name.removesuffix(".search.jsonl"), selected
    )
    if int(boundary["public_execution_time_ns"]) < window["public_execution_time_ns"]:
        raise RuntimeError("Cycle27 latch predates public execution")
    preceding = opponent_move_before_second_root(
        protocol, role=window["observer_role"], before_ns=int(row["time_ns"])
    )
    move_receipts = validate_move_receipts(run, identity, boundary)
    if not rows(run / "prior-a.jsonl"):
        return None
    return {
        "schema": "metagross-cycle27-second-root-operational-smoke/v1",
        "status": "pass",
        "completed_scored_games": 0,
        "candidate_search_path": str(search_path.resolve()),
        "candidate_search_sha256": sha(search_path),
        "candidate_protocol_path": str(protocol_path.resolve()),
        "candidate_protocol_sha256": sha(protocol_path),
        "selected_action": selected,
        "production_action": teacher.get("production_action"),
        "decision": teacher.get("decision"),
        "decision_index": 1,
        "worlds": 16,
        "iterations_per_world": 8192,
        "engine_receipts": provenance,
        "registered_battle": registered,
        "request_decision_execution_window": window,
        "preceding_opponent_move": preceding,
        "typed_move_receipts": move_receipts,
        "public_execution_boundary": boundary,
        "exact_selected_action_sent_and_executed": True,
        "fallback_timeout_or_semantic_failure": False,
        "h2h_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        result = try_validate(
            args.run.resolve(), args.pair_manifest.resolve(), args.expected_engine_sha256
        )
        if result is not None:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"status": "pass", "schema": result["schema"]}, sort_keys=True))
            return
        time.sleep(0.01)
    raise TimeoutError("Cycle27 second-root smoke did not pass")


if __name__ == "__main__":
    main()
