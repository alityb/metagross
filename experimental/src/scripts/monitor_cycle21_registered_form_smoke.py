#!/usr/bin/env python3
"""Cycle21 registered-team, ability-lineage, and equal8192 smoke monitor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import rows, to_id
from experimental.src.scripts.monitor_cycle20_form_smoke import try_validate as try_validate_form
from srcs.metagross.causal_reveal_ledger import canonical_species


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_roster(packed: str) -> list[str]:
    roster = []
    for member in packed.split("]"):
        fields = member.split("|")
        roster.append(canonical_species(to_id(fields[1] or fields[0])))
    return roster


def player_roles(protocol: list[dict]) -> dict[str, str]:
    found = {}
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "player" and parts[2] in {"p1", "p2"}:
                found[to_id(parts[3])] = parts[2]
    return found


def first_private_roster(protocol: list[dict]) -> list[str]:
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            if not line.startswith("|request|"):
                continue
            request = json.loads(line.removeprefix("|request|"))
            pokemon = (request.get("side") or {}).get("pokemon") or []
            if len(pokemon) == 6:
                return [
                    canonical_species(to_id(str(mon.get("details", "")).split(",", 1)[0]))
                    for mon in pokemon
                ]
    raise RuntimeError("Cycle21 protocol has no genuine six-Pokemon private request")


def public_leads(protocol: list[dict]) -> dict[str, str]:
    leads = {}
    for row in protocol:
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "switch":
                role = parts[2][:2]
                if role in {"p1", "p2"} and role not in leads:
                    leads[role] = canonical_species(to_id(parts[3].split(",", 1)[0]))
    return leads


def attest_registered_battle(run: Path, pair_manifest: Path) -> dict:
    pair_payload = json.loads(pair_manifest.read_text())
    pair = pair_payload["pairs"][0]
    receipt_path = run / "REGISTRATION_CONSUMPTION.json"
    launch_path = run / "SHOWDOWN_LAUNCH.json"
    if not receipt_path.is_file() or not launch_path.is_file():
        raise RuntimeError("Cycle21 registration or Showdown launch receipt is absent")
    receipt = json.loads(receipt_path.read_text())
    launch = json.loads(launch_path.read_text())
    pair_dir = (run / "smoke-registrations").resolve()
    if (
        receipt.get("status") != "pass"
        or receipt.get("pair_manifest_sha256") != sha(pair_manifest)
        or receipt.get("battle_seed") != pair["battle_seed"]
        or receipt.get("registrations_observed") != 2
        or receipt.get("registrations_consumed") != 2
        or receipt.get("registration_reappearances") != 0
        or receipt.get("remaining_files") != []
        or any(pair_dir.glob("*.json"))
    ):
        raise RuntimeError("Cycle21 registrations were not consumed exactly")
    if (
        launch.get("ready") is not True
        or Path(launch.get("pair_directory", "")).resolve() != pair_dir
        or (launch.get("environment") or {}).get("METAGROSS_EVAL_PAIR_DIR") != str(pair_dir)
        or launch.get("port") != 8010
    ):
        raise RuntimeError("Cycle21 registration-aware Showdown launch is not attested")
    protocol_by_user = {}
    for path in (run / "smoke-logs").glob("*.protocol.jsonl"):
        protocol_by_user[path.name.removesuffix(".protocol.jsonl")] = rows(path)
    role_to_team = {}
    for registration in receipt["registrations"]:
        username = registration["username"]
        protocol = protocol_by_user.get(username)
        if protocol is None:
            raise RuntimeError("Cycle21 registration has no matching player protocol")
        role = player_roles(protocol).get(to_id(username))
        expected_role = registration["orientation"]
        expected_roster = packed_roster(registration["payload"]["packed_team"])
        if role != expected_role or first_private_roster(protocol) != expected_roster:
            raise RuntimeError("Cycle21 private roster/player orientation mismatch")
        role_to_team[role] = {
            "username": username,
            "assigned_team_sha256": registration["payload"]["assigned_team_sha256"],
            "ordered_private_roster": expected_roster,
        }
    any_protocol = next(iter(protocol_by_user.values()))
    leads = public_leads(any_protocol)
    for role, team in role_to_team.items():
        if leads.get(role) != team["ordered_private_roster"][0]:
            raise RuntimeError("Cycle21 public lead differs from registered private team")
    return {
        "registration_receipt_sha256": sha(receipt_path),
        "showdown_launch_sha256": sha(launch_path),
        "battle_seed": pair["battle_seed"], "roles": role_to_team,
        "public_leads": leads, "packed_teams_and_seed_attested": True,
    }


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    form = try_validate_form(run, expected_engine_sha)
    if form is None:
        return None
    registered = attest_registered_battle(run, pair_manifest)
    return {
        **form, "schema": "metagross-cycle21-registered-form-smoke/v1",
        "registered_battle": registered,
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
        time.sleep(0.25)
    raise TimeoutError("Cycle21 registered form smoke did not pass in time")


if __name__ == "__main__":
    main()
