#!/usr/bin/env python3
"""Fail-closed Cycle31 monitor layered over the frozen Cycle30 gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle30_dynamic_boundary_smoke import (
    try_validate as validate_cycle30_gate,
)


CONTROLLER = "metagross-cycle19-equal8192-production-selector/v1"


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    result = validate_cycle30_gate(run, pair_manifest, expected_engine_sha)
    if result is None:
        return None
    boundary = result["public_execution_boundary"]
    attribution = boundary.get("cycle31_candidate_attribution")
    if not isinstance(attribution, dict):
        raise RuntimeError("Cycle31 boundary lacks candidate attribution")
    candidate_username = Path(result["candidate_search_path"]).name.removesuffix(
        ".search.jsonl"
    )
    evidence = result["dynamic_boundary_evidence"]
    receipts = result["target_typed_move_receipts"]
    window = result["target_request_decision_execution_window"]
    exact = {
        "battle_tag": boundary.get("battle_tag"),
        "rqid": boundary.get("rqid"),
        "decision_index": boundary.get("decision_index"),
        "root_id": boundary.get("root_id"),
        "protocol_sha256": evidence.get("protocol_sha256"),
    }
    if (
        attribution.get("namespace") != "agent_a"
        or attribution.get("username") != candidate_username
        or attribution.get("observer_role") != window.get("observer_role")
        or any(attribution.get(key) != value for key, value in exact.items())
        or attribution.get("protocol_sha256") != receipts.get("protocol_sha256")
        or attribution.get("controller_schema") != CONTROLLER
        or attribution.get("iterations_per_world") != 8192
        or attribution.get("candidate_cells")
        != [[schedule, world] for schedule in range(2) for world in range(8)]
    ):
        raise RuntimeError("Cycle31 candidate attribution identity is inconsistent")
    registered_roles = result["registered_battle"].get("roles") or {}
    registered_candidate = registered_roles.get(window.get("observer_role")) or {}
    if registered_candidate.get("username") != candidate_username:
        raise RuntimeError("Cycle31 candidate user/role differs from registration")
    result["schema"] = "metagross-cycle31-candidate-boundary-operational-smoke/v1"
    result["candidate_attribution"] = attribution
    result["agent_b_production_only_can_latch"] = False
    result["h2h_authorized"] = True
    result["strength_claim_authorized"] = False
    return result


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
    raise TimeoutError("Cycle31 candidate-attributed smoke did not pass")


if __name__ == "__main__":
    main()
