#!/usr/bin/env python3
"""Run the separately frozen Cycle 4 systematic bridge gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from belief.causal_protocol_bridge import CausalProtocolBridgeError, norm
from belief.causal_protocol_bridge_v2 import (
    parse_causal_protocol_v2,
    reconcile_causal_facts_v2,
)
from belief.public_form_contract import load_public_form_contract
from scripts.audit_public_search_state_gate_a_selection import select_without_representation
from scripts.run_causal_protocol_bridge_gate import _schedule
from scripts.run_public_search_state_gate_a import (
    EXPECTED,
    GateAError,
    hidden_perturbation,
    load_rows,
    sha256,
)
from search.public_search_state_v1 import (
    canonical_action_table,
    canonical_bytes,
    extract_public_search_state,
)


ROOT = Path(__file__).resolve().parents[3]


def request_actions_exact(request: Mapping[str, Any]) -> set[str]:
    """Own-private legal actions; exact form IDs never use public aliases."""
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
            move_id = norm(move.get("id", ""))
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
            actions.add("switch " + norm(details.split(",", 1)[0]))
    if not actions:
        raise CausalProtocolBridgeError("private request has no legal action")
    return actions


def instruction_signature(value: Any) -> tuple[float, tuple[str, ...]]:
    percentage = getattr(value, "percentage", None)
    rows = getattr(value, "instruction_list", None)
    if not isinstance(percentage, (float, int)) or rows is None:
        raise GateAError("unsupported StateInstructions API")
    return float(percentage), tuple(str(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "cycle4-bridge-gate-report.json"
    if output.exists():
        raise GateAError("Cycle 4 report already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise GateAError(f"frozen input hash mismatch: {relative}")

    import poke_engine

    if sha256(Path(poke_engine.poke_engine.__file__)) != EXPECTED["engine"]:
        raise GateAError("engine binding hash mismatch")
    paths = [
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    ]
    rows = load_rows(paths[0]) + load_rows(paths[1])
    selected, selection_rejections = select_without_representation(rows, poke_engine)
    contract = load_public_form_contract()

    root_supported = successor_supported = scheduled = 0
    root_failures: Counter[str] = Counter()
    successor_failures: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    certified: Counter[str] = Counter()
    for root in selected:
        schedule = _schedule(root)
        scheduled += len(schedule)
        try:
            snapshot = root["r1_policy_snapshot"]
            request = snapshot["player_information_state"]["private_request"]
            actions = request_actions_exact(request)
            if actions != set(root["_side_one_actions"]):
                raise CausalProtocolBridgeError("private request/engine legality mismatch")
            if actions != set(snapshot["own_legality"]["actions"]):
                raise CausalProtocolBridgeError("private request/snapshot legality mismatch")
            if canonical_action_table(sorted(actions))["name_table"] != snapshot["name_table"]:
                raise CausalProtocolBridgeError("private request action-map mismatch")
            facts = parse_causal_protocol_v2(
                snapshot["protocol_prefix"], player_role=snapshot["player_role"],
                private_request=request, contract=contract,
            )
            reconciled = reconcile_causal_facts_v2(root["_state"], poke_engine, facts, contract)
            state = reconciled.state
            root_public = extract_public_search_state(state, poke_engine)
            if root_public["action_table"]["name_table"] != snapshot["name_table"]:
                raise CausalProtocolBridgeError("reconciled root action-map mismatch")
            root_supported += 1
            for repair in reconciled.archival_repairs:
                repairs[repair.split(":", 1)[0]] += 1
            for reveal in facts.opponent:
                certified["species"] += 1
                certified["move"] += len(reveal.moves)
                certified["item"] += int(reveal.item_status_revealed)
                certified["ability"] += int(reveal.ability is not None)
        except Exception as exc:
            root_failures[f"{type(exc).__name__}:{exc}"] += 1
            continue

        for left, right, uniform in schedule:
            try:
                root_string = state.to_string()
                root_bytes = canonical_bytes(extract_public_search_state(state, poke_engine))
                first = poke_engine.step_with_uniform_r1_semantic(state, left, right, uniform)
                second = poke_engine.step_with_uniform_r1_semantic(state, left, right, uniform)
                if first.state.to_string() != second.state.to_string():
                    raise GateAError("step_nondeterminism")
                if instruction_signature(first.selected_instructions) != instruction_signature(second.selected_instructions):
                    raise GateAError("instruction_nondeterminism")
                first_public = canonical_bytes(extract_public_search_state(first.state, poke_engine))
                if first_public != canonical_bytes(extract_public_search_state(second.state, poke_engine)):
                    raise GateAError("public_step_nondeterminism")
                restored = first.state.reverse_instructions(first.selected_instructions)
                if restored.to_string() != root_string:
                    raise GateAError("engine_reverse_mismatch")
                if canonical_bytes(extract_public_search_state(restored, poke_engine)) != root_bytes:
                    raise GateAError("public_reverse_mismatch")
                perturbation = hidden_perturbation(first.state, poke_engine)
                if perturbation is not None and canonical_bytes(
                    extract_public_search_state(perturbation, poke_engine)
                ) != first_public:
                    raise GateAError("hidden_noninterference")
                successor_supported += 1
            except Exception as exc:
                successor_failures[f"{type(exc).__name__}:{exc}"] += 1

    root_coverage = root_supported / len(selected)
    successor_coverage = successor_supported / scheduled
    status = "pass" if root_coverage >= 0.95 and successor_coverage >= 0.95 else "fail"
    cycle3_sampler = ROOT / "experimental/runs/search_native_v2_cycle3_bridge_repair_20260815/production-sampler-preservation.json"
    report = {
        "schema": "metagross-search-native-v2-cycle4-systematic-bridge-gate/v1",
        "status": status,
        "gates": {
            "root_reconciliation_ge_0.95": root_coverage >= 0.95,
            "scheduled_successor_support_ge_0.95": successor_coverage >= 0.95,
        },
        "counts": {
            "source_rows": len(rows), "selected_roots": len(selected),
            "physical_battles": len({row["identity"]["battle_tag"] for row in selected}),
            "root_supported": root_supported, "root_unsupported": len(selected) - root_supported,
            "root_coverage": root_coverage, "scheduled_successors": scheduled,
            "successor_supported": successor_supported,
            "successor_unsupported": scheduled - successor_supported,
            "successor_coverage": successor_coverage,
        },
        "root_failures": dict(root_failures.most_common()),
        "successor_failures": dict(successor_failures.most_common()),
        "archival_repairs": dict(sorted(repairs.items())),
        "event_certified_facts": dict(sorted(certified.items())),
        "selection_rejections": dict(sorted(selection_rejections.items())),
        "public_form_contract": {
            "mapping_count": len(contract.mapping),
            "battle_only_count": sum(value == "battleOnly" for value in contract.authority.values()),
            "cosmetic_count": sum(value == "cosmeticFormes" for value in contract.authority.values()),
            "ogerpon_wellspring_tera_target": contract.canonical("ogerponwellspringtera"),
        },
        "production_sampler_diagnostic": {
            "status": "live_capture_contract_bug",
            "raw_recall": 1.0,
            "typed_recall": 0.0,
            "source_report_sha256": sha256(cycle3_sampler),
        },
        "authorization": {
            "freeze_native_live_capture_contract": status == "pass",
            "implementation": False, "training": False,
            "target_collection": False, "h2h": False,
            "sealed_confirmation": False,
        },
        "sealed_confirmation_panel_rows_read": 0,
        "local_cpu_only": True, "paid_compute_usd": 0,
        "hashes": {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "engine_binding_sha256": EXPECTED["engine"],
            "agent_a_source_sha256": EXPECTED["agent_a"],
            "agent_b_source_sha256": EXPECTED["agent_b"],
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
