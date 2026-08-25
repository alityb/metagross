#!/usr/bin/env python3
"""Cycle32 registration witness with a fresh username domain."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experimental.src.scripts import watch_cycle31_registrations as base


def validate_registration(payload: dict, pair: dict, username: str) -> str:
    if not re.fullmatch(r"c32smk[xy][0-9a-f]{7}", username):
        raise RuntimeError("unexpected Cycle32 registration username")
    shared = {
        "schema_version": 1, "pair_id": pair["pair_id"], "leg": 1,
        "format": "gen9randombattle", "battle_seed": pair["battle_seed"],
        "team_1_sha256": pair["team_1_sha256"], "team_2_sha256": pair["team_2_sha256"],
    }
    if any(payload.get(key) != value for key, value in shared.items()):
        raise RuntimeError("Cycle32 registration identity differs from frozen pair")
    assigned = payload.get("assigned_team_sha256")
    if assigned == pair["team_1_sha256"]:
        expected_team, orientation = pair["team_1_packed"], "p1"
    elif assigned == pair["team_2_sha256"]:
        expected_team, orientation = pair["team_2_packed"], "p2"
    else:
        raise RuntimeError("Cycle32 registration has unknown assigned team")
    packed = payload.get("packed_team")
    if not isinstance(packed, str) or packed != expected_team or base.sha_bytes(packed.encode()) != assigned:
        raise RuntimeError("Cycle32 packed team differs from frozen pair")
    return orientation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = base.validate_registration
    base.validate_registration = validate_registration
    try:
        result = base.watch(
            args.directory.resolve(), args.pair_manifest.resolve(), args.timeout_seconds
        )
    finally:
        base.validate_registration = original
    result["schema"] = "metagross-cycle32-registration-consumption/v1"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "consumed": 2}, sort_keys=True))


if __name__ == "__main__":
    main()
