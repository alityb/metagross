#!/usr/bin/env python3
"""Witness exact Cycle40 registration consumption with fresh identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from experimental.src.scripts import watch_cycle33_registrations as base


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate(payload: dict, pair: dict, username: str) -> dict:
    match = re.fullmatch(r"c40h2h([xy])[0-9a-f]{7}", username)
    if match is None:
        raise RuntimeError("unexpected Cycle40 registration username")
    expected = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "format": "gen9randombattle",
        "battle_seed": pair["battle_seed"],
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
    }
    if payload.get("leg") not in {1, 2} or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Cycle40 registration differs from frozen pair")
    assigned = payload.get("assigned_team_sha256")
    if assigned == pair["team_1_sha256"]:
        packed = pair["team_1_packed"]
    elif assigned == pair["team_2_sha256"]:
        packed = pair["team_2_packed"]
    else:
        raise RuntimeError("Cycle40 registration assigned unknown team")
    if payload.get("packed_team") != packed or sha_bytes(packed.encode()) != assigned:
        raise RuntimeError("Cycle40 packed team mismatch")
    return {
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "leg": payload["leg"],
        "side": "p1" if match.group(1) == "x" else "p2",
        "assigned_team_sha256": assigned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base.validate = validate
    result = base.watch(
        args.directory.resolve(), args.pair_manifest.resolve(), args.timeout_seconds
    )
    result["schema"] = "metagross-cycle40-registration-consumption/v1"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "consumed": 40}, sort_keys=True))


if __name__ == "__main__":
    main()
