#!/usr/bin/env python3
"""Cycle 9's narrowly repaired replay canonicalization and POV contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from experimental.src.scripts import cycle8_replay_audit as v8
from srcs.metagross.causal_reveal_ledger import freeze_ledger, norm


HISTORICAL_PROTOCOL_COMMIT = "a4b6781ad07ff854a5e25c73fc04b699caa3d96a"
PUBLIC_BROADCAST_PREFIX = (
    '|raw|<div class="broadcast-blue"><strong>This battle is required to be public '
)
ReplayAuditError = v8.ReplayAuditError
canonical_json = v8.canonical_json
sha256_bytes = v8.sha256_bytes
sha256_path = v8.sha256_path
model_information_fingerprint = v8.model_information_fingerprint


def _forced_loser_name(inputlog: str, lines: Sequence[str]) -> str | None:
    role = None
    for line in inputlog.splitlines():
        if line.startswith(">forcelose "):
            role = line.split(" ", 1)[1].strip()
    if role not in {"p1", "p2"}:
        return None
    for line in lines:
        fields = line.split("|")
        if len(fields) >= 4 and fields[1] == "player" and fields[2] == role:
            return fields[3]
    return None


def canonical_public_lines(
    lines: Sequence[str], *, inputlog: str, showdown_commit: str,
) -> list[str]:
    base = v8.canonical_public_lines(lines, inputlog=inputlog)
    loser = _forced_loser_name(inputlog, lines)
    seen_player_roles: set[str] = set()
    result: list[str] = []
    for line in base:
        fields = line.split("|")
        tag = fields[1] if len(fields) > 1 else ""
        if tag == "badge":
            continue
        if line.startswith(PUBLIC_BROADCAST_PREFIX) and line.endswith("</div>"):
            continue
        if tag == "player" and len(fields) >= 3 and fields[2] in {"p1", "p2"}:
            if fields[2] in seen_player_roles:
                continue
            seen_player_roles.add(fields[2])
        if tag == "-message" and loser and line == f"|-message|{loser} lost due to inactivity.":
            continue
        if showdown_commit == HISTORICAL_PROTOCOL_COMMIT:
            if tag in {"-damage", "-heal"} and len(fields) >= 4:
                if fields[3] == "0 fnt":
                    fields[3] = "0"
            if tag == "-weather" and len(fields) >= 3 and fields[2] in {"Rain Dance", "RainDance"}:
                fields[2] = "RainDance"
            line = "|".join(fields)
        result.append(line)
    return result


def public_lines_from_capture(
    public_capture: Mapping[str, Any], *, inputlog: str, showdown_commit: str,
) -> list[str]:
    chunks = public_capture.get("public_chunks")
    if not isinstance(chunks, list):
        raise ReplayAuditError("public capture lacks chunks")
    lines: list[str] = []
    for row in chunks:
        if not isinstance(row, Mapping) or not isinstance(row.get("data"), str):
            raise ReplayAuditError("malformed public chunk")
        lines.extend(row["data"].splitlines())
    return canonical_public_lines(
        lines, inputlog=inputlog, showdown_commit=showdown_commit,
    )


def _revival_prompt(request: Mapping[str, Any]) -> bool:
    side = request.get("side")
    rows = side.get("pokemon") if isinstance(side, Mapping) else None
    return isinstance(rows, list) and any(
        isinstance(row, Mapping)
        and row.get("active") is True
        and row.get("reviving") is True
        for row in rows
    )


def _request_support(
    request: Mapping[str, Any],
) -> tuple[set[str], dict[str, int], dict[str, Any], dict[str, str]]:
    if not _revival_prompt(request):
        actions, table, sidecar = v8._request_support(request)
        semantics = {
            action: "switch" if action.startswith("switch ")
            else "tera" if action.endswith("-tera") else "move"
            for action in actions
        }
        sidecar["revival_prompt"] = False
        return actions, table, sidecar, semantics

    force_rows = request.get("forceSwitch")
    if not isinstance(force_rows, list) or force_rows != [True]:
        raise ReplayAuditError("Revival Blessing prompt is not forced")
    side = request.get("side")
    rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(rows, list):
        raise ReplayAuditError("Revival Blessing prompt lacks own team")
    actions: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReplayAuditError("invalid Revival Blessing target row")
        if row.get("active") is True:
            continue
        condition = row.get("condition")
        details = row.get("details")
        if not isinstance(condition, str) or not isinstance(details, str):
            raise ReplayAuditError("invalid Revival Blessing target fields")
        hp_text = condition.split(" ", 1)[0].split("/", 1)[0]
        fainted = condition.endswith(" fnt") or hp_text == "0"
        if fainted:
            species = norm(details.split(",", 1)[0])
            if species:
                actions.add("switch " + species)
    if not actions or len(actions) > 5:
        raise ReplayAuditError("Revival Blessing prompt has invalid target count")
    table = {action: index + 4 for index, action in enumerate(sorted(actions))}
    active_rows = request.get("active")
    active = active_rows[0] if isinstance(active_rows, list) and active_rows else {}
    moves = active.get("moves", []) if isinstance(active, Mapping) else []
    sidecar = {
        "wait": False,
        "forced_switch": True,
        "trapped": False,
        "can_tera": False,
        "revival_prompt": True,
        "moves": [
            {
                "id": norm(move.get("id", "")), "pp": move.get("pp"),
                "maxpp": move.get("maxpp"),
                "disabled": move.get("disabled", False),
            }
            for move in moves if isinstance(move, Mapping)
        ],
    }
    return actions, table, sidecar, {action: "revival_target" for action in actions}


def materialize_role(
    *, battle_id: str, role: str, public_capture: Mapping[str, Any],
    pov_capture: Mapping[str, Any], inputlog: str, showdown_commit: str,
) -> dict[str, Any]:
    if role not in {"p1", "p2"} or pov_capture.get("role") != role:
        raise ReplayAuditError("POV role mismatch")
    requests = pov_capture.get("requests")
    commands = pov_capture.get("commands")
    errors = pov_capture.get("errors")
    chunks = pov_capture.get("sideupdate_chunks")
    if not all(isinstance(value, list) for value in (requests, commands, errors, chunks)):
        raise ReplayAuditError("POV capture is incomplete")
    if errors:
        raise ReplayAuditError("Showdown emitted a private error")
    public_chunks = public_capture.get("public_chunks")
    if not isinstance(public_chunks, list):
        raise ReplayAuditError("public chunks are absent")

    command_by_request: dict[int, Mapping[str, Any]] = {}
    for command in commands:
        if not isinstance(command, Mapping):
            raise ReplayAuditError("invalid command row")
        request_index = command.get("preceding_request_index")
        if isinstance(request_index, bool) or not isinstance(request_index, int):
            raise ReplayAuditError("command has null preceding request")
        if request_index in command_by_request:
            raise ReplayAuditError("request was reused by multiple commands")
        if request_index < 0 or request_index >= len(requests):
            raise ReplayAuditError("command request index is out of range")
        command_by_request[request_index] = command

    states = []
    for request_index, row in enumerate(requests):
        if not isinstance(row, Mapping) or not isinstance(row.get("request"), Mapping):
            raise ReplayAuditError("invalid request row")
        request = row["request"]
        side = request.get("side")
        if not isinstance(side, Mapping) or side.get("id") != role:
            raise ReplayAuditError("opposite-side or missing private request")
        chunk_count = row.get("public_chunk_count")
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or not 0 <= chunk_count <= len(public_chunks):
            raise ReplayAuditError("invalid public event boundary")
        command = command_by_request.get(request_index)
        if command is not None:
            command_input_index = command.get("input_index")
            if isinstance(command_input_index, bool) or not isinstance(command_input_index, int):
                raise ReplayAuditError("command has invalid input index")
            command_chunk_count = sum(
                isinstance(chunk, Mapping)
                and isinstance(chunk.get("input_index"), int)
                and chunk["input_index"] < command_input_index
                for chunk in public_chunks
            )
            if command_chunk_count < chunk_count:
                raise ReplayAuditError("command-time public prefix regressed")
            chunk_count = command_chunk_count
        raw_prefix = [
            line for chunk in public_chunks[:chunk_count]
            for line in str(chunk.get("data", "")).splitlines()
        ]
        public_prefix = canonical_public_lines(
            raw_prefix, inputlog=inputlog, showdown_commit=showdown_commit,
        )
        ledger = freeze_ledger(battle_id, role, public_prefix)
        wait = request.get("wait", False)
        if not isinstance(wait, bool):
            raise ReplayAuditError("invalid wait metadata")
        if wait:
            if command is not None:
                raise ReplayAuditError("wait request received a recorded command")
            actions: set[str] = set()
            action_table: dict[str, int] = {}
            action_semantics: dict[str, str] = {}
            sidecar = {
                "wait": True, "forced_switch": False, "trapped": False,
                "can_tera": False, "revival_prompt": False, "moves": [],
            }
        else:
            actions, action_table, sidecar, action_semantics = _request_support(request)
            sidecar["wait"] = False
        chosen = None
        chosen_index = None
        chosen_semantics = None
        if command is not None:
            chosen = v8._command_action(str(command.get("command", "")), request)
            if chosen not in actions or chosen not in action_table:
                raise ReplayAuditError("recorded action is illegal in its exact request")
            chosen_index = action_table[chosen]
            chosen_semantics = action_semantics[chosen]
        state = {
            "schema": "metagross-cycle9-rematerialized-pov/v1",
            "battle_id": battle_id,
            "role": role,
            "request_index": request_index,
            "actionable": not wait,
            "public_event_index": len(public_prefix),
            "public_prefix": public_prefix,
            "private_request": request,
            "legal_actions": sorted(actions),
            "action_table": action_table,
            "action_semantics": action_semantics,
            "chosen_action": chosen,
            "chosen_action_index": chosen_index,
            "chosen_action_semantics": chosen_semantics,
            "typed_reveal_ledger": ledger.to_payload(),
            "pp_disable_sidecar": sidecar,
        }
        state["model_information_fingerprint_sha256"] = model_information_fingerprint(
            role=role, public_event_index=len(public_prefix),
            public_prefix=public_prefix, private_request=request,
            ledger_payload=ledger.to_payload(),
        )
        state["provenance_record_sha256"] = sha256_bytes(canonical_json(state).encode("ascii"))
        states.append(state)
    return {
        "schema": "metagross-cycle9-rematerialized-pov/v1",
        "battle_id": battle_id, "role": role, "states": states,
    }
