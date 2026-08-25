#!/usr/bin/env python3
"""Cycle22 registered smoke plus durable certified-ability receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experimental.src.scripts.monitor_cycle21_registered_form_smoke import (
    try_validate as try_validate_cycle21,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ability_receipts(run: Path) -> dict:
    files = sorted((run / "ability-receipts").glob("agenta-*.jsonl"))
    if not files:
        raise RuntimeError("Cycle22 has no candidate ability-installation receipt")
    rows = []
    for path in files:
        for raw in path.read_text().splitlines():
            payload = json.loads(raw)
            if payload.get("schema") != "metagross-certified-ability-installation/v1":
                raise RuntimeError("Cycle22 ability receipt schema mismatch")
            rows.append(payload)
    matches = []
    for row in rows:
        for installation in row.get("installations", []):
            if installation.get("exact_public_species") != "terapagosterastal":
                continue
            expected = {
                "authority": "rule_implied_form_transition",
                "exact_public_species": "terapagosterastal",
                "installed_base_ability": "terashell",
                "installed_current_ability": "terashell",
                "slot": 0,
                "update_base": True,
            }
            if installation != expected:
                raise RuntimeError("Cycle22 installed the wrong Terapagos ability state")
            matches.append({
                "battle_tag": row.get("battle_tag"),
                "observer_role": row.get("observer_role"),
                "protocol_sha256": row.get("protocol_sha256"),
                "swap": row.get("swap"),
            })
    if not matches or any(row["observer_role"] != "p1" or row["swap"] is not False for row in matches):
        raise RuntimeError("Cycle22 candidate did not attest the expected observer hydration")
    return {
        "receipt_files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "terapagos_installation_receipts": len(matches),
        "all_current_and_base_terashell": True,
        "sampled_preinstallation_values_recorded": False,
    }


def try_validate(run: Path, pair_manifest: Path, expected_engine_sha: str) -> dict | None:
    result = try_validate_cycle21(run, pair_manifest, expected_engine_sha)
    if result is None:
        return None
    return {
        **result,
        "schema": "metagross-cycle22-certified-ability-smoke/v1",
        "certified_ability_installation": validate_ability_receipts(run),
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
    raise TimeoutError("Cycle22 certified-ability smoke did not pass in time")


if __name__ == "__main__":
    main()
