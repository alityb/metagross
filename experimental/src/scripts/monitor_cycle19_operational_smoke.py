#!/usr/bin/env python3
"""Fail-closed monitor for Cycle 19's one-decision live operational smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    parsed = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def to_id(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(character.lower() for character in normalized if character.isalnum())


def received_request(protocol: list[dict], before_ns: int) -> dict:
    candidates = []
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        if int(row.get("time_ns", 0)) > before_ns:
            continue
        for line in str(row.get("message", "")).splitlines():
            if line.startswith("|request|"):
                candidates.append(json.loads(line.removeprefix("|request|")))
    if not candidates:
        raise RuntimeError("candidate sent an action without a preceding private request")
    return candidates[-1]


def expected_command(action: str, request: dict) -> tuple[str, str]:
    rqid = str(request.get("rqid"))
    if action.startswith("switch "):
        target = to_id(action.removeprefix("switch "))
        matches = []
        for index, pokemon in enumerate((request.get("side") or {}).get("pokemon") or [], 1):
            details = str(pokemon.get("details") or "").split(",", 1)[0]
            ident = str(pokemon.get("ident") or "").split(":", 1)[-1]
            if target in {to_id(details), to_id(ident)}:
                matches.append(index)
        if len(matches) != 1:
            raise RuntimeError("selected switch does not map uniquely to private request")
        return f"/switch {matches[0]}", rqid
    tera = action.endswith("-tera")
    move = action.removesuffix("-tera")
    command = f"/choose move {move}"
    if tera:
        command += " terastallize"
    return command, rqid


def public_action_confirmed(
    protocol: list[dict], after_ns: int, username: str, selected: str
) -> bool:
    role = None
    player_pattern = re.compile(r"^\|player\|(p[12])\|([^|]+)\|")
    for row in protocol:
        for line in str(row.get("message", "")).splitlines():
            match = player_pattern.match(line)
            if match and to_id(match.group(2)) == to_id(username):
                role = match.group(1)
    if role is None:
        return False
    target = to_id(selected.removeprefix("switch ").removesuffix("-tera"))
    for row in protocol:
        if int(row.get("time_ns", 0)) <= after_ns or row.get("direction") not in {
            "received",
            "reconnect_received",
        }:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if selected.startswith("switch "):
                if len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"}:
                    if parts[2].startswith(f"{role}a:") and to_id(parts[3].split(",", 1)[0]) == target:
                        return True
            elif len(parts) >= 4 and parts[1] == "move":
                if parts[2].startswith(f"{role}a:") and to_id(parts[3]) == target:
                    return True
    return False


def validate_teacher(teacher: dict) -> None:
    if teacher.get("controller_schema") != "metagross-cycle19-equal8192-production-selector/v1":
        raise RuntimeError("wrong Cycle19 controller schema")
    if teacher.get("schedule_count") != 2 or teacher.get("world_count") != 16:
        raise RuntimeError("candidate did not use two eight-world schedules")
    if teacher.get("iterations_per_world") != 8192:
        raise RuntimeError("candidate iteration contract changed")
    receipts = teacher.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != 16:
        raise RuntimeError("candidate receipts are incomplete")
    if any(row.get("total_visits") != 8192 for row in receipts):
        raise RuntimeError("candidate receipt has the wrong visit count")
    cells = {(row.get("schedule_index"), row.get("world_index")) for row in receipts}
    if cells != {(schedule, world) for schedule in range(2) for world in range(8)}:
        raise RuntimeError("candidate schedule/world cells are incomplete")
    policy = teacher.get("prefilter_aggregate_policy")
    considered = teacher.get("considered_choices")
    if not isinstance(policy, dict) or not isinstance(considered, list) or not policy:
        raise RuntimeError("candidate did not preserve policy/considered receipts")
    if not math.isclose(math.fsum(map(float, policy.values())), 1.0, abs_tol=1e-12):
        raise RuntimeError("candidate aggregate policy is not normalized")
    maximum = max(map(float, policy.values()))
    expected = [
        action
        for action, mass in sorted(policy.items(), key=lambda row: row[1], reverse=True)
        if float(mass) >= 0.75 * maximum
    ]
    actual = [row.get("action") for row in considered]
    if actual != expected or teacher.get("selected_action") not in actual:
        raise RuntimeError("candidate considered-choice contract is wrong")
    legal = set(policy)
    for receipt in receipts:
        side = receipt.get("side_one")
        if not isinstance(side, list) or {row.get("action") for row in side} != legal:
            raise RuntimeError("candidate per-world legal action support changed")


def try_validate(run: Path, expected_engine_sha: str) -> dict | None:
    engine_receipts = []
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
            raise RuntimeError("spawned process engine provenance mismatch")
        engine_receipts.append({"path": str(path.resolve()), "sha256": sha(path)})

    for search_path in sorted((run / "smoke-logs").glob("*.search.jsonl")):
        for search_row in rows(search_path):
            override = search_row.get("choice_override") or {}
            teacher = override.get("terminal_mcts_teacher") or {}
            if teacher.get("controller_schema") != "metagross-cycle19-equal8192-production-selector/v1":
                continue
            validate_teacher(teacher)
            selected = str(teacher["selected_action"])
            if search_row.get("choice") != selected or override.get("final_choice") != selected:
                raise RuntimeError("candidate selected action was not installed")
            if str(teacher.get("reason", "")).startswith("fail_closed"):
                raise RuntimeError("candidate fell back")
            protocol_path = search_path.with_name(
                search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
            )
            protocol = rows(protocol_path)
            sent = [
                row
                for row in protocol
                if row.get("direction") == "sent"
                and isinstance(row.get("messages"), list)
                and row["messages"]
                and str(row["messages"][0]).startswith(("/choose move ", "/switch "))
            ]
            if not sent:
                return None
            sent_row = sent[-1]
            request = received_request(protocol, int(sent_row["time_ns"]))
            expected_command_value, expected_rqid = expected_command(selected, request)
            if sent_row["messages"] != [expected_command_value, expected_rqid]:
                raise RuntimeError("Showdown command differs from selected candidate action")
            username = search_path.name.removesuffix(".search.jsonl")
            if not public_action_confirmed(
                protocol, int(sent_row["time_ns"]), username, selected
            ):
                return None
            # The candidate is fixed to agent_a; require at least one durable R1 row.
            if not rows(run / "prior-a.jsonl"):
                return None
            return {
                "schema": "metagross-cycle19-one-decision-smoke/v1",
                "status": "pass",
                "completed_scored_games": 0,
                "candidate_search_path": str(search_path.resolve()),
                "candidate_search_sha256": sha(search_path),
                "candidate_protocol_path": str(protocol_path.resolve()),
                "candidate_protocol_sha256": sha(protocol_path),
                "selected_action": selected,
                "production_action": teacher.get("production_action"),
                "decision": teacher.get("decision"),
                "worlds": 16,
                "iterations_per_world": 8192,
                "engine_receipts": engine_receipts,
                "exact_selected_action_sent": True,
                "public_execution_confirmed": True,
                "fallback_timeout_or_semantic_failure": False,
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        result = try_validate(args.run.resolve(), args.expected_engine_sha256.lower())
        if result is not None:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True))
            return
        time.sleep(0.25)
    raise TimeoutError("Cycle19 one-decision operational smoke did not pass in time")


if __name__ == "__main__":
    main()
