#!/usr/bin/env python3
"""Cycle25 first-decision monitor with complete conversion attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import rows
from experimental.src.scripts.monitor_cycle20_form_smoke import validate_form_transition
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import try_validate as try_validate_cycle21
from experimental.src.scripts.monitor_cycle23_first_decision_smoke import first_request_window


CONTROLLER = "metagross-cycle19-equal8192-production-selector/v1"


def expected_installation(row: dict) -> None:
    matches = [x for x in row.get("installations", []) if x.get("exact_public_species") == "terapagosterastal"]
    if len(matches) != 1:
        raise RuntimeError("Cycle25 exact public form is not unique")
    item = matches[0]
    expected = {
        "authority": "rule_implied_form_transition",
        "installed_base_ability": "terashell",
        "installed_current_ability": "terashell",
        "update_base": True,
    }
    if any(item.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Cycle25 certified ability installation is wrong")
    if not isinstance(item.get("slot"), int) or not 0 <= item["slot"] < 6:
        raise RuntimeError("Cycle25 exact-form slot is invalid")


def validate_attributed_receipts(run: Path, identity: dict, protocol_hash: str, boundary: dict) -> dict:
    files = sorted((run / "ability-receipts").glob("agenta-*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("Cycle25 requires one candidate receipt file")
    parsed = rows(files[0])
    if not parsed:
        raise RuntimeError("Cycle25 has no attributed receipts")
    production, candidate = [], []
    boundary_ns = int(boundary["public_execution_time_ns"])
    for row in parsed:
        if row.get("schema") != "metagross-certified-ability-installation/v2":
            raise RuntimeError("Cycle25 receipt is not attributed v2")
        if row.get("protocol_sha256") != protocol_hash or row.get("observer_role") != "p1" or row.get("swap") is not False:
            raise RuntimeError("Cycle25 receipt left the first observer causal root")
        if int(row.get("receipt_time_ns", 0)) > boundary_ns:
            raise RuntimeError("Cycle25 receipt occurred after public execution")
        context = row.get("execution_context")
        if not isinstance(context, dict) or any(context.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Cycle25 receipt decision identity mismatch")
        expected_installation(row)
        if context.get("phase") == "production_control" and context.get("cohort") == "adaptive_root_search":
            production.append(context)
        elif context.get("phase") == "equal8192_candidate" and context.get("cohort") == "fixed_two_by_eight":
            candidate.append(context)
        else:
            raise RuntimeError("Cycle25 receipt has unknown phase/cohort")
    declared = {row.get("declared_world_count") for row in production}
    if len(declared) != 1 or next(iter(declared)) not in {16, 32}:
        raise RuntimeError("Cycle25 adaptive production declaration is invalid")
    production_count = next(iter(declared))
    if len(production) != production_count or {r.get("conversion_index") for r in production} != set(range(production_count)):
        raise RuntimeError("Cycle25 production cohort does not reconcile")
    if any(r.get("schedule_index") is not None or r.get("world_index") is not None for r in production):
        raise RuntimeError("Cycle25 production cohort has candidate schedule markers")
    cells = {(r.get("schedule_index"), r.get("world_index")) for r in candidate}
    if len(candidate) != 16 or cells != {(s, w) for s in range(2) for w in range(8)}:
        raise RuntimeError("Cycle25 candidate cohort is not exact 2x8")
    if {r.get("conversion_index") for r in candidate} != set(range(16)) or any(r.get("declared_world_count") != 16 for r in candidate):
        raise RuntimeError("Cycle25 candidate conversion indices are incomplete")
    return {"receipt_path": str(files[0].resolve()), "production_receipts": len(production), "candidate_receipts": 16, "candidate_cells": sorted([list(x) for x in cells]), "all_receipts_pre_execution": True}


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    base = try_validate_cycle21(run, pair_manifest, expected_engine_sha)
    boundary_path = run / "PUBLIC_EXECUTION_BOUNDARY.json"
    if base is None or not boundary_path.is_file():
        return None
    boundary = json.loads(boundary_path.read_text())
    search_path = Path(base["candidate_search_path"])
    candidates = [r for r in rows(search_path) if ((r.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get("controller_schema") == CONTROLLER]
    if len(candidates) != 1:
        raise RuntimeError("Cycle25 requires exactly one candidate decision")
    candidate = candidates[0]
    context = candidate.get("context") or {}
    tag, rqid, decision_index = context.get("tag"), context.get("rqid"), context.get("decision_idx")
    root_id = hashlib.sha256(f"terminal-mcts-live\0{tag}\0{decision_index}".encode()).hexdigest()
    identity = {"battle_tag": tag, "rqid": rqid, "decision_index": decision_index, "root_id": root_id}
    if any(boundary.get(k) != v for k, v in identity.items()) or boundary.get("selected_action") != base["selected_action"]:
        raise RuntimeError("Cycle25 public boundary identity mismatch")
    protocol_path = Path(base["candidate_protocol_path"])
    protocol = rows(protocol_path)
    decision_ns = int(candidate["time_ns"])
    username = search_path.name.removesuffix(".search.jsonl")
    lineage = validate_form_transition(protocol, username, decision_ns)
    window = first_request_window(protocol, decision_ns, username, base["selected_action"])
    if int(boundary["public_execution_time_ns"]) < window["public_execution_time_ns"]:
        raise RuntimeError("Cycle25 boundary predates public protocol execution")
    receipt = validate_attributed_receipts(run, identity, lineage["protocol_sha256"], boundary)
    return {**base, "schema": "metagross-cycle25-attributed-first-decision-smoke/v1", "causal_ability_lineage": lineage, "request_decision_execution_window": window, "receipt_attribution": receipt, "public_execution_boundary": boundary, "h2h_authorized": False, "pp_conditional_belief_cycle_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        result = try_validate(args.run.resolve(), args.pair_manifest.resolve(), args.expected_engine_sha256)
        if result is not None:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"status": "pass", "schema": result["schema"]}, sort_keys=True))
            return
        time.sleep(0.005)
    raise TimeoutError("Cycle25 attributed smoke did not pass")


if __name__ == "__main__":
    main()
