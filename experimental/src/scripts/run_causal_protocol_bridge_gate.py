#!/usr/bin/env python3
"""Run frozen Cycle 3 causal bridge root/successor gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from belief.causal_protocol_bridge import (
    CausalProtocolBridgeError,
    canonical_species,
    parse_causal_protocol,
    reconcile_causal_facts,
)
from scripts.audit_public_search_state_gate_a_selection import select_without_representation
from scripts.run_public_search_state_gate_a import (
    EXPECTED,
    GateAError,
    hidden_perturbation,
    load_rows,
    rank,
    sha256,
)
from search.public_search_state_v1 import (
    canonical_action_table,
    canonical_bytes,
    extract_public_search_state,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "metagross-search-native-v2-cycle3-causal-bridge-gate/v1"


def _request_actions(request: Mapping[str, Any]) -> set[str]:
    """Derive ordinary legal actions from the observer-private request."""
    if not isinstance(request.get("rqid"), int) or isinstance(request.get("rqid"), bool):
        raise CausalProtocolBridgeError("invalid private request rqid")
    forced_rows = request.get("forceSwitch", [False])
    if not isinstance(forced_rows, list) or not forced_rows or not isinstance(forced_rows[0], bool):
        raise CausalProtocolBridgeError("invalid private request forceSwitch")
    forced = forced_rows[0]
    active_rows = request.get("active") or []
    if not isinstance(active_rows, list):
        raise CausalProtocolBridgeError("invalid private request active rows")
    active = active_rows[0] if active_rows else {}
    if not isinstance(active, Mapping):
        raise CausalProtocolBridgeError("invalid private request active row")
    trapped = active.get("trapped", False)
    if not isinstance(trapped, bool):
        raise CausalProtocolBridgeError("invalid private request trapped flag")
    can_tera = bool(active.get("canTerastallize", False))
    actions: set[str] = set()
    if not forced:
        moves = active.get("moves", [])
        if not isinstance(moves, list):
            raise CausalProtocolBridgeError("invalid private request moves")
        for move in moves:
            if not isinstance(move, Mapping):
                raise CausalProtocolBridgeError("invalid private request move")
            move_id = "".join(character for character in str(move.get("id", "")).lower() if character.isalnum())
            if move_id and not move.get("disabled", False) and move.get("pp") != 0:
                actions.add(move_id)
                if can_tera:
                    actions.add(move_id + "-tera")
    side = request.get("side")
    party = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(party, list):
        raise CausalProtocolBridgeError("invalid private request party")
    if forced or not trapped:
        for pokemon in party:
            if not isinstance(pokemon, Mapping):
                raise CausalProtocolBridgeError("invalid private request Pokemon")
            if pokemon.get("active") is True:
                continue
            condition = pokemon.get("condition")
            details = pokemon.get("details")
            if not isinstance(condition, str) or not isinstance(details, str):
                raise CausalProtocolBridgeError("invalid private request Pokemon fields")
            hp = condition.split(" ", 1)[0].split("/", 1)[0]
            if condition.endswith(" fnt") or hp == "0":
                continue
            actions.add("switch " + canonical_species(details.split(",", 1)[0]))
    if not actions:
        raise CausalProtocolBridgeError("private request has no legal action")
    return actions


def _schedule(root: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    pairs = [
        (left, right)
        for left in root["_side_one_actions"]
        for right in root["_side_two_actions"]
    ]
    pairs.sort(key=lambda pair: rank([root["capture_sha256"], *pair]))
    return [(left, right, uniform) for left, right in pairs[:4] for uniform in (0.25, 0.75)]


def _repeat_step(engine: Any, state: Any, left: str, right: str, uniform: float) -> tuple[Any, Any]:
    first = engine.step_with_uniform_r1_semantic(state, left, right, uniform)
    second = engine.step_with_uniform_r1_semantic(state, left, right, uniform)
    if first.state.to_string() != second.state.to_string():
        raise GateAError("step_nondeterminism")
    if list(first.selected_instructions) != list(second.selected_instructions):
        raise GateAError("instruction_nondeterminism")
    return first, second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "cycle3-bridge-gate-report.json"
    if output.exists():
        raise GateAError("Cycle 3 report already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise GateAError(f"frozen input hash mismatch: {relative}")

    import poke_engine

    if sha256(Path(poke_engine.poke_engine.__file__)) != EXPECTED["engine"]:
        raise GateAError("engine binding hash mismatch")
    source_paths = [
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    ]
    if sha256(source_paths[0]) != EXPECTED["agent_a"] or sha256(source_paths[1]) != EXPECTED["agent_b"]:
        raise GateAError("source hash mismatch")
    rows = load_rows(source_paths[0]) + load_rows(source_paths[1])
    selected, selection_rejections = select_without_representation(rows, poke_engine)

    root_supported = 0
    scheduled = 0
    successor_supported = 0
    root_failures: Counter[str] = Counter()
    successor_failures: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    certified_counts: Counter[str] = Counter()
    for root in selected:
        schedule = _schedule(root)
        scheduled += len(schedule)
        try:
            snapshot = root["r1_policy_snapshot"]
            request = snapshot["player_information_state"]["private_request"]
            request_actions = _request_actions(request)
            if request_actions != set(root["_side_one_actions"]):
                raise CausalProtocolBridgeError("private request/engine legality mismatch")
            if request_actions != set(snapshot["own_legality"]["actions"]):
                raise CausalProtocolBridgeError("private request/snapshot legality mismatch")
            if canonical_action_table(sorted(request_actions))["name_table"] != snapshot["name_table"]:
                raise CausalProtocolBridgeError("private request action-map mismatch")
            facts = parse_causal_protocol(
                snapshot["protocol_prefix"],
                player_role=snapshot["player_role"],
                private_request=request,
            )
            reconciled = reconcile_causal_facts(root["_state"], poke_engine, facts)
            state = reconciled.state
            public_root = extract_public_search_state(state, poke_engine)
            if public_root["action_table"]["name_table"] != snapshot["name_table"]:
                raise CausalProtocolBridgeError("reconciled root action-map mismatch")
            root_supported += 1
            for repair in reconciled.archival_repairs:
                repair_counts[repair.split(":", 1)[0]] += 1
            for reveal in facts.opponent:
                certified_counts["species"] += 1
                certified_counts["move"] += len(reveal.moves)
                certified_counts["item"] += int(reveal.item_status_revealed)
                certified_counts["ability"] += int(reveal.ability is not None)
        except Exception as exc:
            root_failures[f"{type(exc).__name__}:{exc}"] += 1
            continue

        for left, right, uniform in schedule:
            try:
                root_string = state.to_string()
                root_public = canonical_bytes(extract_public_search_state(state, poke_engine))
                step, _ = _repeat_step(poke_engine, state, left, right, uniform)
                child = step.state
                restored = child.reverse_instructions(step.selected_instructions)
                if restored.to_string() != root_string:
                    raise GateAError("engine_reverse_mismatch")
                if canonical_bytes(extract_public_search_state(restored, poke_engine)) != root_public:
                    raise GateAError("public_reverse_mismatch")
                child_public = canonical_bytes(extract_public_search_state(child, poke_engine))
                perturbation = hidden_perturbation(child, poke_engine)
                if perturbation is not None and canonical_bytes(
                    extract_public_search_state(perturbation, poke_engine)
                ) != child_public:
                    raise GateAError("hidden_noninterference")
                successor_supported += 1
            except Exception as exc:
                successor_failures[f"{type(exc).__name__}:{exc}"] += 1

    root_coverage = root_supported / len(selected)
    successor_coverage = successor_supported / scheduled
    sampler_path = run_dir / "production-sampler-preservation.json"
    sampler = json.loads(sampler_path.read_text()) if sampler_path.exists() else {"status": "pending"}
    status = "pass" if root_coverage >= 0.95 and successor_coverage >= 0.95 else "fail"
    report = {
        "schema": SCHEMA,
        "status": status,
        "gates": {
            "root_reconciliation_ge_0.95": root_coverage >= 0.95,
            "scheduled_successor_support_ge_0.95": successor_coverage >= 0.95,
        },
        "counts": {
            "source_rows": len(rows),
            "selected_roots": len(selected),
            "physical_battles": len({row["identity"]["battle_tag"] for row in selected}),
            "root_supported": root_supported,
            "root_unsupported": len(selected) - root_supported,
            "scheduled_successors": scheduled,
            "successor_supported": successor_supported,
            "successor_unsupported": scheduled - successor_supported,
            "root_coverage": root_coverage,
            "successor_coverage": successor_coverage,
        },
        "root_failures": dict(root_failures.most_common()),
        "successor_failures": dict(successor_failures.most_common()),
        "selection_rejections": dict(sorted(selection_rejections.items())),
        "archival_repairs": dict(sorted(repair_counts.items())),
        "event_certified_facts": dict(sorted(certified_counts.items())),
        "scope_guard": {
            "bridge_mechanics_only": True,
            "placeholder_pp_repairs": repair_counts["move"],
            "long_horizon_target_collection_authorized": False,
            "required_before_long_horizon_collection": (
                "Freeze and pass a causal opponent PP/disable contract (event-counted PP or "
                "fail-closed abstention) whenever placeholder move hydration occurs."
            ),
        },
        "production_sampler_preservation": sampler,
        "information_contract": {
            "opponent_protocol_events_only": True,
            "own_private_request_only": True,
            "transformer_defaults_forbidden": True,
            "hidden_truth_fill_forbidden": True,
            "root_r1_used_at_interior": False,
        },
        "sealed_confirmation_panel_rows_read": 0,
        "training_runs": 0,
        "new_games": 0,
        "h2h_games": 0,
        "local_cpu_only": True,
        "paid_compute_usd": 0,
        "hashes": {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "agent_a_source_sha256": EXPECTED["agent_a"],
            "agent_b_source_sha256": EXPECTED["agent_b"],
            "engine_binding_sha256": EXPECTED["engine"],
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
