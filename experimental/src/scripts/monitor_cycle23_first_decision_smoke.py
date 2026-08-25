#!/usr/bin/env python3
"""Cycle23 monitor bound to the first candidate causal root and action."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import rows
from experimental.src.scripts.monitor_cycle20_form_smoke import validate_form_transition
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import try_validate as try_validate_cycle21


CONTROLLER_SCHEMA = "metagross-cycle19-equal8192-production-selector/v1"


def first_request_window(protocol: list[dict], decision_ns: int, username: str, selected: str) -> dict:
    requests = []
    sends = []
    for row in protocol:
        stamp = int(row.get("time_ns", 0))
        if row.get("direction") in {"received", "reconnect_received"}:
            for line in str(row.get("message", "")).splitlines():
                if line.startswith("|request|") and stamp < decision_ns:
                    requests.append((stamp, json.loads(line.removeprefix("|request|"))))
        if row.get("direction") == "sent" and isinstance(row.get("messages"), list):
            messages = row["messages"]
            if messages and str(messages[0]).startswith(("/choose move ", "/switch ")):
                sends.append((stamp, messages))
    if not requests or not sends:
        raise RuntimeError("Cycle23 request/send boundary is incomplete")
    request_ns, request = requests[-1]
    send_ns, messages = sends[0]
    if not request_ns < decision_ns < send_ns:
        raise RuntimeError("Cycle23 decision is outside its request/send time window")
    rqid = str(request.get("rqid"))
    if len(messages) != 2 or str(messages[1]) != rqid:
        raise RuntimeError("Cycle23 command rqid does not match its exact request")

    role = None
    for row in protocol:
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "player" and parts[3].lower() == username.lower():
                role = parts[2]
    if role not in {"p1", "p2"}:
        raise RuntimeError("Cycle23 cannot resolve candidate role")
    target = selected.removeprefix("switch ").removesuffix("-tera").replace("-", "").lower()
    execution_ns = None
    for row in protocol:
        stamp = int(row.get("time_ns", 0))
        if stamp <= send_ns or row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if selected.startswith("switch "):
                match = len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"} and parts[2].startswith(f"{role}a:")
                actual = parts[3].split(",", 1)[0].replace("-", "").replace(" ", "").lower() if match else ""
            else:
                match = len(parts) >= 4 and parts[1] == "move" and parts[2].startswith(f"{role}a:")
                actual = parts[3].replace("-", "").replace(" ", "").lower() if match else ""
            if match and actual == target:
                execution_ns = stamp
                break
        if execution_ns is not None:
            break
    if execution_ns is None or execution_ns <= send_ns:
        raise RuntimeError("Cycle23 selected action is not publicly executed after send")
    return {
        "rqid": rqid, "request_time_ns": request_ns, "decision_time_ns": decision_ns,
        "send_time_ns": send_ns, "public_execution_time_ns": execution_ns,
    }


def validate_bound_receipts(run: Path, protocol_sha256: str) -> dict:
    files = sorted((run / "ability-receipts").glob("agenta-*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("Cycle23 requires one candidate ability receipt file")
    all_rows = rows(files[0])
    if not all_rows:
        raise RuntimeError("Cycle23 has no ability installation receipts")
    if any(row.get("protocol_sha256") != protocol_sha256 for row in all_rows):
        raise RuntimeError("Cycle23 receipt escaped the first causal root")
    slots = []
    for row in all_rows:
        if row.get("schema") != "metagross-certified-ability-installation/v1":
            raise RuntimeError("Cycle23 ability receipt schema mismatch")
        if row.get("observer_role") != "p1" or row.get("swap") is not False:
            raise RuntimeError("Cycle23 receipt has wrong observer orientation")
        matches = [
            item for item in row.get("installations", [])
            if item.get("exact_public_species") == "terapagosterastal"
        ]
        if len(matches) != 1:
            raise RuntimeError("Cycle23 exact public form is not unique in receipt")
        item = matches[0]
        expected = {
            "authority": "rule_implied_form_transition",
            "exact_public_species": "terapagosterastal",
            "installed_base_ability": "terashell",
            "installed_current_ability": "terashell",
            "update_base": True,
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Cycle23 installed the wrong certified ability")
        slot = item.get("slot")
        if not isinstance(slot, int) or not 0 <= slot < 6:
            raise RuntimeError("Cycle23 certified exact-form slot is invalid")
        slots.append(slot)
    if len(all_rows) < 16:
        raise RuntimeError("Cycle23 has fewer receipts than candidate worlds")
    return {
        "receipt_path": str(files[0].resolve()), "receipt_rows": len(all_rows),
        "causal_root_protocol_sha256": protocol_sha256,
        "all_current_and_base_terashell": True,
        "unique_exact_form_slot_per_receipt": True,
        "slot_values": sorted(set(slots)),
    }


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    base = try_validate_cycle21(run, pair_manifest, expected_engine_sha)
    if base is None:
        return None
    search_path = Path(base["candidate_search_path"])
    candidates = [
        row for row in rows(search_path)
        if ((row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get("controller_schema") == CONTROLLER_SCHEMA
    ]
    if len(candidates) != 1:
        raise RuntimeError("Cycle23 requires exactly one candidate decision")
    candidate = candidates[0]
    decision_ns = int(candidate.get("time_ns", 0))
    protocol_path = Path(base["candidate_protocol_path"])
    protocol = rows(protocol_path)
    username = search_path.name.removesuffix(".search.jsonl")
    lineage = validate_form_transition(protocol, username, decision_ns)
    window = first_request_window(protocol, decision_ns, username, str(base["selected_action"]))
    ability = validate_bound_receipts(run, lineage["protocol_sha256"])
    return {
        **base, "schema": "metagross-cycle23-first-decision-smoke/v1",
        "causal_ability_lineage": lineage,
        "certified_ability_installation": ability,
        "request_decision_execution_window": window,
        "termination_boundary": "immediately_after_first_public_execution",
        "h2h_authorized": False,
        "pp_conditional_belief_cycle_only": True,
    }


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
    raise TimeoutError("Cycle23 first-decision smoke did not pass in time")


if __name__ == "__main__":
    main()
