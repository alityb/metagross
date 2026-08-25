#!/usr/bin/env python3
"""Fail-closed monitor for Cycle30's first intrinsic-move ordinary root."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle21_registered_form_smoke import (
    attest_registered_battle,
)
from experimental.src.scripts.monitor_cycle29_second_root_smoke import (
    candidate_rows,
    engine_provenance,
    opponent_move_before_second_root,
    rows,
    selected_action_window,
    sha,
    validate_move_receipts,
)
from experimental.src.scripts.monitor_cycle19_operational_smoke import validate_teacher


def decision_identity(row: dict) -> dict:
    context = row.get("context") or {}
    tag = context.get("tag")
    decision_index = context.get("decision_idx")
    return {
        "battle_tag": tag,
        "rqid": context.get("rqid"),
        "decision_index": decision_index,
        "root_id": hashlib.sha256(
            f"terminal-mcts-live\0{tag}\0{decision_index}".encode("utf-8")
        ).hexdigest(),
    }


def receipt_has_causal_move(run: Path, identity: dict) -> bool:
    files = sorted((run / "move-receipts").glob("agenta-*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("Cycle30 requires one candidate move-receipt file")
    payloads = []
    for row in rows(files[0]):
        context = row.get("execution_context")
        if not isinstance(context, dict) or any(
            context.get(key) != value for key, value in identity.items()
        ):
            continue
        receipt = row.get("move_receipt") or {}
        payloads.append(
            (
                json.dumps(receipt.get("moves"), sort_keys=True),
                json.dumps(receipt.get("derived_executions"), sort_keys=True),
            )
        )
    if not payloads:
        raise RuntimeError("Cycle30 decision has no typed move receipts")
    if len(set(payloads)) != 1:
        raise RuntimeError("Cycle30 worlds disagree on causal move receipt payload")
    moves, derived = payloads[0]
    return json.loads(moves) != [] or json.loads(derived) != []


def validate_candidate_row(row: dict) -> tuple[dict, str]:
    teacher = (row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}
    validate_teacher(teacher)
    selected = str(teacher["selected_action"])
    override = row.get("choice_override") or {}
    if row.get("choice") != selected or override.get("final_choice") != selected:
        raise RuntimeError("Cycle30 candidate selected action was not installed")
    if str(teacher.get("reason", "")).startswith("fail_closed"):
        raise RuntimeError("Cycle30 candidate fell back")
    return teacher, selected


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    provenance = engine_provenance(run, expected_engine_sha)
    boundary_path = run / "PUBLIC_EXECUTION_BOUNDARY.json"
    if provenance is None or not boundary_path.is_file():
        return None
    registered = attest_registered_battle(run, pair_manifest)
    boundary = json.loads(boundary_path.read_text())
    evidence = boundary.get("cycle30_dynamic_boundary")
    if not isinstance(evidence, dict) or evidence.get("eligible") is not True:
        raise RuntimeError("Cycle30 latch lacks eligible dynamic-boundary evidence")
    target_index = evidence.get("decision_index")
    if (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or not 0 <= target_index <= 5
        or evidence.get("battle_turn", 7) > 6
        or evidence.get("ordinary") is not True
        or evidence.get("wait") is not False
        or evidence.get("force_switch") is not False
        or evidence.get("automatic_action") is not None
        or int(evidence.get("intrinsic_opponent_move_events", 0)) < 1
    ):
        raise RuntimeError("Cycle30 dynamic boundary violates its frozen eligibility rule")

    candidate_files = []
    for search_path in sorted((run / "smoke-logs").glob("*.search.jsonl")):
        candidates = candidate_rows(search_path)
        if candidates:
            candidate_files.append((search_path, candidates))
    if len(candidate_files) != 1:
        raise RuntimeError("Cycle30 candidate identity is not unique")
    search_path, candidates = candidate_files[0]
    by_index: dict[int, list[dict]] = {}
    for row in candidates:
        index = (row.get("context") or {}).get("decision_idx")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError("Cycle30 candidate row has invalid decision index")
        by_index.setdefault(index, []).append(row)
    if set(by_index) != set(range(target_index + 1)):
        raise RuntimeError("Cycle30 candidate decisions are skipped or extend past target")
    if any(len(group) != 1 for group in by_index.values()):
        raise RuntimeError("Cycle30 candidate decision identity is not unique")

    protocol_path = search_path.with_name(
        search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
    )
    protocol = rows(protocol_path)
    username = search_path.name.removesuffix(".search.jsonl")
    decision_reports = []
    target_teacher = None
    target_selected = None
    target_window = None
    target_receipts = None
    preceding_move = None
    for index in range(target_index + 1):
        row = by_index[index][0]
        teacher, selected = validate_candidate_row(row)
        identity = decision_identity(row)
        window = selected_action_window(
            protocol, int(row["time_ns"]), username, selected
        )
        if str(identity["rqid"]) != str(window["rqid"]):
            raise RuntimeError("Cycle30 decision rqid differs from exact request window")
        has_move = receipt_has_causal_move(run, identity)
        receipts = validate_move_receipts(
            run, identity, boundary, require_revealed_move=has_move
        )
        if has_move:
            move = opponent_move_before_second_root(
                protocol,
                role=window["observer_role"],
                before_ns=int(row["time_ns"]),
            )
        else:
            move = None
        decision_reports.append(
            {
                "decision_index": index,
                "selected_action": selected,
                "production_action": teacher.get("production_action"),
                "decision": teacher.get("decision"),
                "request_decision_execution_window": window,
                "typed_move_receipts": receipts,
                "preceding_opponent_move": move,
            }
        )
        if index == target_index:
            if not has_move:
                raise RuntimeError("Cycle30 target receipts lack a causal opponent move")
            target_teacher = teacher
            target_selected = selected
            target_window = window
            target_receipts = receipts
            preceding_move = move

    identity = decision_identity(by_index[target_index][0])
    if (
        any(boundary.get(key) != value for key, value in identity.items())
        or boundary.get("selected_action") != target_selected
        or int(boundary["public_execution_time_ns"])
        < int(target_window["public_execution_time_ns"])
        or boundary.get("cycle30_dynamic_boundary", {}).get("protocol_sha256")
        != target_receipts.get("protocol_sha256")
    ):
        raise RuntimeError("Cycle30 public boundary differs from target root")
    if not rows(run / "prior-a.jsonl"):
        return None
    return {
        "schema": "metagross-cycle30-dynamic-boundary-operational-smoke/v1",
        "status": "pass",
        "completed_scored_games": 0,
        "candidate_search_path": str(search_path.resolve()),
        "candidate_search_sha256": sha(search_path),
        "candidate_protocol_path": str(protocol_path.resolve()),
        "candidate_protocol_sha256": sha(protocol_path),
        "target_decision_index": target_index,
        "target_battle_turn": evidence["battle_turn"],
        "selected_action": target_selected,
        "production_action": target_teacher.get("production_action"),
        "decision": target_teacher.get("decision"),
        "worlds": 16,
        "iterations_per_world": 8192,
        "engine_receipts": provenance,
        "registered_battle": registered,
        "dynamic_boundary_evidence": evidence,
        "preceding_decisions": decision_reports[:-1],
        "target_request_decision_execution_window": target_window,
        "target_preceding_opponent_move": preceding_move,
        "target_typed_move_receipts": target_receipts,
        "public_execution_boundary": boundary,
        "all_decisions_contiguous_and_publicly_executed": True,
        "all_receipts_pre_target_public_execution": True,
        "fallback_timeout_or_semantic_failure": False,
        "h2h_authorized": True,
        "strength_claim_authorized": False,
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
    raise TimeoutError("Cycle30 dynamic-boundary smoke did not pass")


if __name__ == "__main__":
    main()
