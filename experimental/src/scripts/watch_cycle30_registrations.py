#!/usr/bin/env python3
"""Cycle30 registration witness for the fresh dynamic-boundary smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_registration(payload: dict, pair: dict, username: str) -> str:
    if not re.fullmatch(r"c30smk[xy][0-9a-f]{7}", username):
        raise RuntimeError("unexpected Cycle30 registration username")
    shared = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "leg": 1,
        "format": "gen9randombattle",
        "battle_seed": pair["battle_seed"],
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
    }
    if any(payload.get(key) != value for key, value in shared.items()):
        raise RuntimeError("Cycle30 registration identity differs from frozen pair")
    assigned = payload.get("assigned_team_sha256")
    if assigned == pair["team_1_sha256"]:
        expected_team, orientation = pair["team_1_packed"], "p1"
    elif assigned == pair["team_2_sha256"]:
        expected_team, orientation = pair["team_2_packed"], "p2"
    else:
        raise RuntimeError("Cycle30 registration has unknown assigned team")
    packed = payload.get("packed_team")
    if (
        not isinstance(packed, str)
        or packed != expected_team
        or sha_bytes(packed.encode()) != assigned
    ):
        raise RuntimeError("Cycle30 packed team differs from frozen pair")
    return orientation


def watch(directory: Path, pair_manifest: Path, timeout_seconds: float) -> dict:
    pair_payload = json.loads(pair_manifest.read_text())
    pairs = pair_payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise RuntimeError("Cycle30 watcher requires one frozen pair")
    pair = pairs[0]
    seen: dict[str, dict] = {}
    deleted: set[str] = set()
    empty_since = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = (
            {path.stem: path for path in directory.glob("*.json")}
            if directory.exists()
            else {}
        )
        if len(current) > 2:
            raise RuntimeError("Cycle30 registration directory contains extras")
        for username, path in current.items():
            if username in deleted:
                raise RuntimeError("Cycle30 registration reappeared")
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            payload = json.loads(raw)
            record = {
                "username": username,
                "path": str(path.resolve()),
                "sha256": sha_bytes(raw),
                "orientation": validate_registration(payload, pair, username),
                "payload": payload,
            }
            if username in seen and seen[username] != record:
                raise RuntimeError("Cycle30 registration mutated")
            seen[username] = record
        for username in set(seen) - set(current) - deleted:
            deleted.add(username)
        if len(seen) == 2 and len(deleted) == 2 and not current:
            empty_since = empty_since or time.monotonic()
            if time.monotonic() - empty_since >= 1.0:
                if {row["orientation"] for row in seen.values()} != {"p1", "p2"}:
                    raise RuntimeError("Cycle30 registrations do not cover p1/p2")
                return {
                    "schema": "metagross-cycle30-registration-consumption/v1",
                    "status": "pass",
                    "pair_manifest_sha256": sha_bytes(pair_manifest.read_bytes()),
                    "pair_id": pair["pair_id"],
                    "battle_seed": pair["battle_seed"],
                    "registrations_observed": 2,
                    "registrations_consumed": 2,
                    "registration_reappearances": 0,
                    "remaining_files": [],
                    "registrations": sorted(seen.values(), key=lambda row: row["orientation"]),
                }
        else:
            empty_since = None
        time.sleep(0.01)
    raise TimeoutError("Cycle30 registration consumption was not observed exactly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = watch(args.directory.resolve(), args.pair_manifest.resolve(), args.timeout_seconds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "consumed": 2}, sort_keys=True))


if __name__ == "__main__":
    main()
