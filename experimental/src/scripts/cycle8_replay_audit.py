#!/usr/bin/env python3
"""Frozen canonicalization and POV checks for the Cycle 8 replay audit.

The functions in this module consume only spectator output plus one role's
sideupdate capture.  They never receive the opposite role's capture or an
omniscient Showdown Battle object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from srcs.metagross.causal_reveal_ledger import freeze_ledger, norm


SCHEMA = "metagross-cycle8-rematerialized-pov/v1"
TRANSPORT_TAGS = frozenset({
    "c", "c:", "chat", "html", "inactive", "inactiveoff", "init",
    "j", "J", "join", "l", "L", "leave", "n", "name", "request",
    "t:", "title", "uhtml", "uhtmlchange",
})
HP_FIELDS = {"switch": 4, "drag": 4, "replace": 4, "-damage": 3, "-heal": 3}
RATING_RAW = re.compile(r"^\|raw\|.*'s rating: .*&rarr;.*$")


class ReplayAuditError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_public_prefix(public_prefix: Sequence[str]) -> list[str]:
    """Remove server/player identity while preserving causal battle semantics."""
    player_roles: dict[str, str] = {}
    result = []
    for line in public_prefix:
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "player" and len(parts) >= 4 and parts[2] in {"p1", "p2"}:
            player_roles[parts[3]] = parts[2]
            result.append(f"|player|{parts[2]}")
        elif tag == "win" and len(parts) >= 3:
            result.append("|win|" + player_roles.get(parts[2], "winner"))
        else:
            result.append(line)
    return result


def _model_private_request(private_request: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(canonical_json(private_request))
    result.pop("rqid", None)
    side = result.get("side")
    if isinstance(side, dict):
        side.pop("name", None)
    return result


def model_information_fingerprint(
    *, role: str, public_event_index: int, public_prefix: Sequence[str],
    private_request: Mapping[str, Any], ledger_payload: Mapping[str, Any],
) -> str:
    """Hash model-visible information only; provenance identity stays outside."""
    ledger = {
        key: value for key, value in ledger_payload.items()
        if key not in {"battle_tag", "protocol_sha256"}
    }
    payload = {
        "schema": "metagross-cycle8-model-information/v1",
        "role": role,
        "public_event_index": public_event_index,
        "public_prefix": _model_public_prefix(public_prefix),
        "private_request": _model_private_request(private_request),
        "typed_reveal_ledger": ledger,
    }
    return sha256_bytes(canonical_json(payload).encode("ascii"))


def _forfeit_name(inputlog: str, public_lines: Sequence[str]) -> str | None:
    force_side = None
    for line in inputlog.splitlines():
        if line.startswith(">forcelose "):
            force_side = line.split(" ", 1)[1].strip()
    if force_side not in {"p1", "p2"}:
        return None
    for line in public_lines:
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "player" and parts[2] == force_side:
            return parts[3]
    return None


def canonical_public_lines(lines: Sequence[str], *, inputlog: str) -> list[str]:
    """Drop only frozen server transport and normalize exact HP to public HP."""
    forfeiter = _forfeit_name(inputlog, lines)
    result: list[str] = []
    for line in lines:
        if not line.startswith("|"):
            continue
        fields = line.split("|")
        tag = fields[1] if len(fields) > 1 else ""
        if tag in TRANSPORT_TAGS or (tag == "" and line == "|"):
            continue
        if tag == "-message" and forfeiter and line == f"|-message|{forfeiter} forfeited.":
            continue
        if RATING_RAW.fullmatch(line):
            continue
        hp_field = HP_FIELDS.get(tag)
        if hp_field is not None and len(fields) > hp_field:
            match = re.fullmatch(r"(\d+)/(\d+)(.*)", fields[hp_field])
            if match and int(match.group(2)) > 0:
                hp, maximum = int(match.group(1)), int(match.group(2))
                displayed = math.ceil(100 * hp / maximum)
                if displayed == 100 and hp < maximum:
                    displayed = 99
                fields[hp_field] = f"{displayed}/100{match.group(3)}"
        result.append("|".join(fields))
        # Replay files may append room reset/rematch traffic after the battle's
        # unique terminal event.  No post-terminal line is battle mechanics.
        if tag == "win" or line == "|tie":
            break
    return result


def public_lines_from_capture(public_capture: Mapping[str, Any], *, inputlog: str) -> list[str]:
    chunks = public_capture.get("public_chunks")
    if not isinstance(chunks, list):
        raise ReplayAuditError("public capture lacks chunks")
    lines: list[str] = []
    for row in chunks:
        if not isinstance(row, Mapping) or not isinstance(row.get("data"), str):
            raise ReplayAuditError("malformed public chunk")
        lines.extend(row["data"].splitlines())
    return canonical_public_lines(lines, inputlog=inputlog)


def _request_support(request: Mapping[str, Any]) -> tuple[set[str], dict[str, int], dict[str, Any]]:
    force_rows = request.get("forceSwitch", [False])
    if not isinstance(force_rows, list) or not force_rows or not isinstance(force_rows[0], bool):
        raise ReplayAuditError("invalid forceSwitch metadata")
    forced = force_rows[0]
    active_rows = request.get("active", [])
    if active_rows is None:
        active_rows = []
    if not isinstance(active_rows, list):
        raise ReplayAuditError("invalid active metadata")
    active = active_rows[0] if active_rows else {}
    if not isinstance(active, Mapping):
        raise ReplayAuditError("invalid active row")
    trapped = active.get("trapped", False)
    if not isinstance(trapped, bool):
        raise ReplayAuditError("invalid trapped metadata")
    can_tera = active.get("canTerastallize", False)
    if can_tera is None:
        can_tera = False
    if not isinstance(can_tera, (bool, str)):
        raise ReplayAuditError("invalid Tera metadata")
    can_tera = bool(can_tera)

    actions: set[str] = set()
    if not forced:
        moves = active.get("moves", [])
        if not isinstance(moves, list):
            raise ReplayAuditError("invalid move metadata")
        for move in moves:
            if not isinstance(move, Mapping):
                raise ReplayAuditError("invalid move row")
            move_id = norm(move.get("id", ""))
            disabled = move.get("disabled", False)
            pp = move.get("pp")
            if not isinstance(disabled, bool) or isinstance(pp, bool) or (
                pp is not None and not isinstance(pp, int)
            ):
                raise ReplayAuditError("invalid PP/disable metadata")
            if move_id and not disabled and pp != 0:
                actions.add(move_id)
                if can_tera:
                    actions.add(move_id + "-tera")

    side = request.get("side")
    if not isinstance(side, Mapping) or not isinstance(side.get("pokemon"), list):
        raise ReplayAuditError("request lacks own side")
    if forced or not trapped:
        for pokemon in side["pokemon"]:
            if not isinstance(pokemon, Mapping):
                raise ReplayAuditError("invalid own Pokemon row")
            if pokemon.get("active") is True:
                continue
            condition = pokemon.get("condition")
            details = pokemon.get("details")
            if not isinstance(condition, str) or not isinstance(details, str):
                raise ReplayAuditError("invalid own switch row")
            hp_text = condition.split(" ", 1)[0].split("/", 1)[0]
            if condition.endswith(" fnt") or hp_text == "0":
                continue
            species = norm(details.split(",", 1)[0])
            if species:
                actions.add("switch " + species)
    if not actions:
        raise ReplayAuditError("request contains no supported action")

    learned_moves = sorted({
        action.removesuffix("-tera") for action in actions
        if not action.startswith("switch ") and action != "struggle"
    })
    switches = sorted(action for action in actions if action.startswith("switch "))
    if len(learned_moves) > 4 or len(switches) > 5:
        raise ReplayAuditError("request exceeds 13-action space")
    table: dict[str, int] = {}
    for index, move_id in enumerate(learned_moves):
        table[move_id] = index
        if move_id + "-tera" in actions:
            table[move_id + "-tera"] = index + 9
    table.update({action: index + 4 for index, action in enumerate(switches)})
    if "struggle" in actions:
        table["struggle"] = 0
    sidecar = {
        "forced_switch": forced,
        "trapped": trapped,
        "can_tera": can_tera,
        "moves": [
            {
                "id": norm(move.get("id", "")),
                "pp": move.get("pp"),
                "maxpp": move.get("maxpp"),
                "disabled": move.get("disabled", False),
            }
            for move in active.get("moves", [])
        ],
    }
    return actions, table, sidecar


def _command_action(command: str, request: Mapping[str, Any]) -> str:
    tokens = command.strip().lower().split()
    if not tokens:
        raise ReplayAuditError("empty command")
    if tokens[0] == "team":
        raise ReplayAuditError("team-preview command is outside the frozen 13-action gate")
    if tokens[0] == "move" and len(tokens) >= 2:
        move_token = tokens[1]
        active = request.get("active")
        active_row = active[0] if isinstance(active, list) and active else {}
        moves = active_row.get("moves") if isinstance(active_row, Mapping) else None
        if move_token.isdigit():
            slot = int(move_token) - 1
            if not isinstance(moves, list) or slot < 0 or slot >= len(moves):
                raise ReplayAuditError("move slot is absent from exact request")
            move_token = str(moves[slot].get("id", ""))
        action = norm(move_token)
        if "terastallize" in tokens[2:]:
            action += "-tera"
        return action
    if tokens[0] == "switch" and len(tokens) == 2:
        side = request.get("side")
        rows = side.get("pokemon") if isinstance(side, Mapping) else None
        if tokens[1].isdigit():
            slot = int(tokens[1]) - 1
            if not isinstance(rows, list) or slot < 0 or slot >= len(rows):
                raise ReplayAuditError("switch slot is absent from exact request")
            details = rows[slot].get("details") if isinstance(rows[slot], Mapping) else None
            if not isinstance(details, str):
                raise ReplayAuditError("switch details are absent")
            return "switch " + norm(details.split(",", 1)[0])
        return "switch " + norm(tokens[1])
    raise ReplayAuditError("unsupported recorded command")


def materialize_role(
    *, battle_id: str, role: str, public_capture: Mapping[str, Any],
    pov_capture: Mapping[str, Any], inputlog: str,
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
        raw_prefix = [
            line
            for chunk in public_chunks[:chunk_count]
            for line in str(chunk.get("data", "")).splitlines()
        ]
        public_prefix = canonical_public_lines(raw_prefix, inputlog=inputlog)
        ledger = freeze_ledger(battle_id, role, public_prefix)
        command = command_by_request.get(request_index)
        wait = request.get("wait", False)
        if not isinstance(wait, bool):
            raise ReplayAuditError("invalid wait metadata")
        if wait:
            if command is not None:
                raise ReplayAuditError("wait request received a recorded command")
            actions: set[str] = set()
            action_table: dict[str, int] = {}
            sidecar = {
                "wait": True, "forced_switch": False, "trapped": False,
                "can_tera": False, "moves": [],
            }
        else:
            actions, action_table, sidecar = _request_support(request)
            sidecar["wait"] = False
        chosen = None
        chosen_index = None
        if command is not None:
            chosen = _command_action(str(command.get("command", "")), request)
            if chosen not in actions or chosen not in action_table:
                raise ReplayAuditError("recorded action is illegal in its exact request")
            chosen_index = action_table[chosen]
        state = {
            "schema": SCHEMA,
            "battle_id": battle_id,
            "role": role,
            "request_index": request_index,
            "actionable": not wait,
            "public_event_index": len(public_prefix),
            "public_prefix": public_prefix,
            "private_request": request,
            "legal_actions": sorted(actions),
            "action_table": action_table,
            "chosen_action": chosen,
            "chosen_action_index": chosen_index,
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
    return {"schema": SCHEMA, "battle_id": battle_id, "role": role, "states": states}
