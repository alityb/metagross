#!/usr/bin/env python3
"""Replay one captured client protocol through a running R1 prior server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experimental.src.scripts.audit_r1_action_boundaries import (  # noqa: E402
    _selected_request_action,
)
from srcs.metagross.prior_server import (  # noqa: E402
    canonical_request_sha256,
    request_action_support,
)


class HistoryReplayError(RuntimeError):
    pass


def _json_request(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=(json.dumps(payload).encode() if payload is not None else None),
        headers=({"Content-Type": "application/json"} if payload is not None else {}),
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise HistoryReplayError("server response is not an object")
    return value


def replay_protocol(
    protocol_path: Path,
    server_url: str,
    username: str,
    namespace: str,
) -> dict:
    pending: dict[tuple[str, int], tuple[dict, dict]] = {}
    ordinary_responses: list[dict] = []
    acknowledgements = []
    received_rows = 0
    with protocol_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            direction = row.get("direction")
            if direction == "received":
                message = row.get("message")
                if not isinstance(message, str) or not message.startswith(">battle-"):
                    continue
                received_rows += 1
                lines = message.splitlines()
                tag = lines[0][1:].strip()
                _json_request(
                    f"{server_url}/lines",
                    {"tag": tag, "namespace": namespace, "lines": lines[1:]},
                )
                for event in lines[1:]:
                    if not event.startswith("|request|"):
                        continue
                    request = json.loads(event.removeprefix("|request|"))
                    if request.get("wait") is True or request.get("teamPreview") is True:
                        continue
                    try:
                        support = request_action_support(request)
                    except RuntimeError:
                        continue
                    if not support["actions"]:
                        continue
                    rqid = support["rqid"]
                    digest = canonical_request_sha256(request)
                    query = urllib.parse.urlencode(
                        {
                            "tag": tag,
                            "namespace": namespace,
                            "username": username,
                            "rqid": rqid,
                            "request_sha256": digest,
                        }
                    )
                    response = _json_request(f"{server_url}/priors?{query}")
                    if response.get("rqid") != rqid or response.get("request_sha256") != digest:
                        raise HistoryReplayError("prior response identity mismatch")
                    trajectory = response.get("trajectory")
                    if not isinstance(trajectory, dict) or trajectory.get("reset_reason") is not None:
                        raise HistoryReplayError("trajectory reset during replay")
                    pending[(tag, rqid)] = (request, response)
                    if trajectory.get("automatic_action") is None:
                        ordinary_responses.append(response)
            elif direction == "sent":
                room = row.get("room")
                messages = row.get("messages")
                if (
                    not isinstance(room, str)
                    or not isinstance(messages, list)
                    or len(messages) < 2
                    or not isinstance(messages[0], str)
                    or not messages[0].startswith(("/choose ", "/switch "))
                ):
                    continue
                rqid = int(messages[1])
                pair = pending.get((room, rqid))
                if pair is None:
                    raise HistoryReplayError("outbound choice lacks served priors")
                request, response = pair
                action = _selected_request_action(messages[0], request)
                if action is None:
                    raise HistoryReplayError("outbound choice has no canonical action")
                ack = _json_request(
                    f"{server_url}/action",
                    {
                        "tag": room,
                        "namespace": namespace,
                        "rqid": rqid,
                        "request_sha256": response["request_sha256"],
                        "decision_idx": response["decision_idx"],
                        "action": action,
                    },
                )
                if ack.get("ok") is not True:
                    raise HistoryReplayError("action acknowledgement failed")
                acknowledgements.append(ack)

    for index, response in enumerate(ordinary_responses, start=1):
        trajectory = response["trajectory"]
        if trajectory.get("observations") != index:
            raise HistoryReplayError("observation history is not monotone")
        if trajectory.get("transitions") != index - 1:
            raise HistoryReplayError("transition history is not causal")
        if trajectory.get("inference_length") != index:
            raise HistoryReplayError("inference history is not full length")
        receipts = trajectory.get("action_receipts")
        if not isinstance(receipts, list) or len(receipts) != index - 1:
            raise HistoryReplayError("action receipts are not aligned")
        if any(receipt.get("source") != "selected_action_ack" for receipt in receipts):
            raise HistoryReplayError("trajectory contains a non-causal action source")
    if len(acknowledgements) < len(ordinary_responses):
        raise HistoryReplayError("not every served decision was acknowledged")
    return {
        "schema_version": 1,
        "audit": "r1_live_server_history_replay_v1",
        "status": "pass",
        "protocol": str(protocol_path.resolve()),
        "server_url": server_url,
        "received_rows": received_rows,
        "ordinary_decisions": len(ordinary_responses),
        "acknowledgements": len(acknowledgements),
        "maximum_inference_length": max(
            response["trajectory"]["inference_length"] for response in ordinary_responses
        ),
        "resets": 0,
        "missing_receipts": 0,
    }


def audit_decision_dump(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise HistoryReplayError("decision dump is empty")
    for expected_length, row in enumerate(rows, start=1):
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, dict):
            raise HistoryReplayError("decision dump lacks trajectory evidence")
        rl2 = trajectory.get("rl2")
        time_indices = trajectory.get("time_indices")
        receipts = trajectory.get("action_receipts")
        if (
            not isinstance(rl2, list)
            or len(rl2) != expected_length
            or not isinstance(time_indices, list)
            or time_indices != list(range(expected_length))
            or not isinstance(receipts, list)
            or len(receipts) != expected_length - 1
        ):
            raise HistoryReplayError("dumped trajectory arrays are misaligned")
        if any(not isinstance(vector, list) or len(vector) != 14 for vector in rl2):
            raise HistoryReplayError("dumped RL2 vector has invalid width")
        if any(value != 0.0 for value in rl2[0]):
            raise HistoryReplayError("initial RL2 vector is not zero")
        for index, receipt in enumerate(receipts, start=1):
            action_idx = receipt.get("action_idx")
            if (
                isinstance(action_idx, bool)
                or not isinstance(action_idx, int)
                or not 0 <= action_idx < 13
                or rl2[index][1 + action_idx] != 1.0
                or sum(rl2[index][1:]) != 1.0
            ):
                raise HistoryReplayError("dumped RL2 action disagrees with receipt")
    return {
        "decision_dump": str(path.resolve()),
        "decision_dump_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensor_rows": len(rows),
        "maximum_tensor_length": len(rows[-1]["trajectory"]["rl2"]),
        "rl2_receipt_mismatches": 0,
        "time_index_mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--namespace", default="causal-history-replay")
    parser.add_argument("--decision-dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay_protocol(
        args.protocol.expanduser().resolve(),
        args.server_url.rstrip("/"),
        args.username,
        args.namespace,
    )
    if args.decision_dump is not None:
        report.update(audit_decision_dump(args.decision_dump.expanduser().resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
