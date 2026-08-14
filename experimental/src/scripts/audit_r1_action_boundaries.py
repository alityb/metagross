#!/usr/bin/env python3
"""Audit exact RL2 action labels recoverable from saved Foul Play protocols.

This is a deployment-contract audit, not a strength evaluation. It joins each
outbound Showdown choice to the private request with the same battle room and
rqid, then maps the chosen command to the frozen 13-action Metamon convention.
No public ``|move|``/``|switch|`` event is used as an action label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from srcs.metagross.prior_server import (
    UNLEARNED_REQUEST_ACTIONS,
    norm,
    private_request_move_name_table,
    private_request_switch_name_table,
    request_action_support,
)


class ActionBoundaryAuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requests_from_message(message: str) -> Iterable[tuple[str, dict]]:
    lines = message.splitlines()
    if not lines or not lines[0].startswith(">battle-"):
        return
    room = lines[0][1:].strip()
    for line in lines[1:]:
        if not line.startswith("|request|"):
            continue
        try:
            request = json.loads(line.removeprefix("|request|"))
        except json.JSONDecodeError as exc:
            raise ActionBoundaryAuditError("invalid private request JSON") from exc
        if not isinstance(request, dict):
            raise ActionBoundaryAuditError("private request is not an object")
        yield room, request


def _selected_request_action(command: str, request: dict) -> str | None:
    command = command.strip().lower()
    move = re.fullmatch(r"/choose move ([a-z0-9]+)( terastallize)?", command)
    if move:
        action = norm(move.group(1))
        if move.group(2):
            action += "-tera"
        return action
    switch = re.fullmatch(r"/switch ([1-6])", command)
    if switch:
        slot = int(switch.group(1)) - 1
        side = request.get("side")
        rows = side.get("pokemon") if isinstance(side, dict) else None
        if not isinstance(rows, list) or slot >= len(rows):
            raise ActionBoundaryAuditError("selected switch has no private team slot")
        row = rows[slot]
        details = row.get("details") if isinstance(row, dict) else None
        if not isinstance(details, str):
            raise ActionBoundaryAuditError("selected switch has invalid private details")
        return "switch " + norm(details.split(",", 1)[0])
    if command == "/choose default":
        return None
    return None


def action_index(request: dict, action: str) -> int:
    support = request_action_support(request)
    actions = set(support["actions"])
    if action not in actions:
        raise ActionBoundaryAuditError("selected action is absent from private support")
    if action == "struggle":
        return 0
    if action in UNLEARNED_REQUEST_ACTIONS:
        raise ActionBoundaryAuditError("selected action lacks a frozen RL2 index")
    table = private_request_move_name_table(actions)
    table.update(private_request_switch_name_table(actions))
    if action not in table:
        raise ActionBoundaryAuditError("selected action is absent from the frozen table")
    return table[action]


def audit_protocols(paths: list[Path]) -> dict:
    if not paths:
        raise ActionBoundaryAuditError("no protocol files supplied")
    requests: dict[tuple[str, int], dict] = {}
    choices: list[tuple[str, int, str]] = []
    files = []
    received_public_action_events = 0
    for path in paths:
        files.append({"path": str(path), "sha256": _sha256(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ActionBoundaryAuditError(f"invalid JSONL in {path}") from exc
                if row.get("direction") == "received":
                    message = row.get("message")
                    if not isinstance(message, str):
                        raise ActionBoundaryAuditError("received row lacks a message")
                    received_public_action_events += sum(
                        event.startswith(("|move|", "|switch|"))
                        for event in message.splitlines()
                    )
                    for room, request in _requests_from_message(message):
                        rqid = request.get("rqid")
                        if isinstance(rqid, bool) or not isinstance(rqid, int):
                            continue
                        key = (room, rqid)
                        previous = requests.get(key)
                        if previous is not None and previous != request:
                            raise ActionBoundaryAuditError("rqid request payload changed")
                        requests[key] = request
                elif row.get("direction") == "sent":
                    messages = row.get("messages")
                    room = row.get("room")
                    if (
                        not isinstance(room, str)
                        or not isinstance(messages, list)
                        or len(messages) < 2
                        or not isinstance(messages[0], str)
                    ):
                        continue
                    command = messages[0].strip().lower()
                    if not command.startswith(("/choose ", "/switch ")):
                        continue
                    try:
                        rqid = int(messages[1])
                    except (TypeError, ValueError):
                        raise ActionBoundaryAuditError("choice has invalid rqid") from None
                    choices.append((room, rqid, command))

    mapped = []
    skipped_default = 0
    seen: dict[tuple[str, int], tuple[str, int]] = {}
    for room, rqid, command in choices:
        key = (room, rqid)
        request = requests.get(key)
        if request is None:
            raise ActionBoundaryAuditError("choice has no correlated private request")
        action = _selected_request_action(command, request)
        if action is None:
            skipped_default += 1
            continue
        index = action_index(request, action)
        prior = seen.get(key)
        if prior is not None and prior != (action, index):
            raise ActionBoundaryAuditError("duplicate rqid has conflicting choices")
        if prior is None:
            seen[key] = (action, index)
            mapped.append({"room": room, "rqid": rqid, "action": action, "action_idx": index})

    counts = {
        "files": len(files),
        "private_requests": len(requests),
        "choice_rows": len(choices),
        "unique_mapped_choices": len(mapped),
        "idempotent_duplicate_choices": len(choices) - len(mapped) - skipped_default,
        "skipped_default_choices": skipped_default,
        "public_action_events_ignored_as_labels": received_public_action_events,
    }
    if not mapped:
        raise ActionBoundaryAuditError("no exact action boundaries were mapped")
    return {
        "schema_version": 1,
        "audit": "r1_selected_action_boundary_v1",
        "status": "pass",
        "counts": counts,
        "action_index_histogram": {
            str(index): sum(row["action_idx"] == index for row in mapped)
            for index in range(13)
        },
        "inputs": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_protocols([path.expanduser().resolve() for path in args.protocol])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
