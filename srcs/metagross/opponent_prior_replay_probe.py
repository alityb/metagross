#!/usr/bin/env python3
"""Replay one frozen public-state request against the local opponent-prior server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

from srcs.metagross.h2h_audit import _read_jsonl, _sha256


def _json_request(url: str, payload: object | None = None) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def probe(
    protocol_log: Path,
    search_log: Path,
    probe_protocol_path: Path,
    prior_url: str,
) -> dict[str, object]:
    protocol = json.loads(probe_protocol_path.read_text(encoding="utf-8"))
    target = protocol["target"]
    root = Path(__file__).resolve().parents[2]
    if (
        protocol.get("status") != "frozen_before_probe"
        or protocol.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs", {}).get("protocol_log_sha256")
        != _sha256(protocol_log)
        or protocol.get("inputs", {}).get("search_log_sha256") != _sha256(search_log)
        or protocol.get("source_identity", {}).get("prior_server.py")
        != _sha256(root / "srcs" / "metagross" / "prior_server.py")
        or protocol.get("source_identity", {}).get("policy_checkpoint")
        != _sha256(
            root
            / "srcs"
            / "models"
            / "randbats_exit_r1"
            / "ckpts"
            / "policy_weights"
            / "policy_epoch_5.pt"
        )
    ):
        raise ValueError("opponent-prior probe differs from its frozen protocol")
    health = _json_request(f"{prior_url}/health")
    if (
        health.get("ok") is not True
        or health.get("identity", {}).get("nonce") != protocol["instance_nonce"]
        or health.get("identity", {}).get("checkpoint_sha256")
        != protocol["source_identity"]["policy_checkpoint"]
    ):
        raise RuntimeError("opponent-prior probe server identity mismatch")
    search_rows = _read_jsonl(search_log)
    matches = [
        row
        for row in search_rows
        if row.get("context", {}).get("tag") == target["battle_tag"]
        and row.get("context", {}).get("decision_idx") == target["decision_idx"]
        and row.get("context", {}).get("rqid") == target["rqid"]
    ]
    if len(matches) != 1:
        raise ValueError("opponent-prior probe target search is not unique")
    namespace = protocol["namespace"]
    result = None
    for row in _read_jsonl(protocol_log):
        if row.get("direction") != "received":
            continue
        message = row.get("message", "")
        lines = message.splitlines()
        if not lines or lines[0] != f">{target['battle_tag']}":
            continue
        _json_request(
            f"{prior_url}/lines",
            {"tag": target["battle_tag"], "namespace": namespace, "lines": lines[1:]},
        )
        for line in lines[1:]:
            if not line.startswith("|request|"):
                continue
            request_payload = json.loads(line.split("|", 2)[2])
            if request_payload.get("rqid") != target["rqid"]:
                continue
            query = urllib.parse.urlencode(
                {
                    "tag": target["battle_tag"],
                    "namespace": namespace,
                    "username": target["username"],
                    "rqid": target["rqid"],
                }
            )
            result = _json_request(f"{prior_url}/priors?{query}")
            break
        if result is not None:
            break
    opponent_priors = result.get("opp_priors") if isinstance(result, dict) else None
    if not isinstance(opponent_priors, dict) or not opponent_priors:
        raise RuntimeError("opponent-prior probe returned no opponent priors")
    masses = [float(value) for value in opponent_priors.values()]
    mass_sum = math.fsum(masses)
    conditions = {
        "target_request_replayed": result is not None,
        "nonempty_opponent_priors": bool(opponent_priors),
        "finite_nonnegative_mass": all(
            math.isfinite(value) and value >= 0 for value in masses
        ),
        "normalized_within_float32_tolerance": abs(mass_sum - 1.0) <= 1e-6,
    }
    return {
        "schema_version": 1,
        "mode": "source_bound_opponent_prior_replay_probe",
        "protocol": {
            "path": str(probe_protocol_path),
            "sha256": _sha256(probe_protocol_path),
        },
        "server_identity": health["identity"],
        "target": target,
        "result": {
            "opponent_priors": opponent_priors,
            "action_count": len(opponent_priors),
            "mass_sum": mass_sum,
            "session_decision_idx": result.get("decision_idx"),
        },
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "authorization": {
            "opponent_prior_fix_verified": all(conditions.values()),
            "new_games_authorized": False,
            "public_ladder_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-log", type=Path, required=True)
    parser.add_argument("--search-log", type=Path, required=True)
    parser.add_argument("--probe-protocol", type=Path, required=True)
    parser.add_argument("--prior-url", default="http://127.0.0.1:8977")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = probe(
        args.protocol_log.expanduser().resolve(),
        args.search_log.expanduser().resolve(),
        args.probe_protocol.expanduser().resolve(),
        args.prior_url,
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
