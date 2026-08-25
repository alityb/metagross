#!/usr/bin/env python3
"""Select Cycle20's first label-blind mirror seed with team-2 lead Terapagos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
START_SEED = 202_620_000_000
MAX_CANDIDATES = 100_000
HELPER = ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs"


def derived_seed(master_seed: int, label: str) -> str:
    digest = hashlib.sha256(f"{master_seed}:1:{label}".encode("ascii")).digest()
    return ",".join(
        str(int.from_bytes(digest[offset : offset + 2], "big"))
        for offset in range(0, 8, 2)
    )


def lead_species(packed: str) -> str:
    first = packed.split("]", 1)[0]
    fields = first.split("|")
    return fields[1] or fields[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Cycle20 selection output already exists")
    for offset in range(MAX_CANDIDATES):
        mirror_seed = START_SEED + offset
        team_1_seed = derived_seed(mirror_seed, "team-1")
        team_2_seed = derived_seed(mirror_seed, "team-2")
        completed = subprocess.run(
            ["node", str(HELPER), "gen9randombattle", team_1_seed, team_2_seed],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        generated = json.loads(completed.stdout)
        if lead_species(generated["team_2_packed"]) != "Terapagos":
            continue
        payload = {
            "schema": "metagross-cycle20-form-smoke-selection/v1",
            "selection_is_label_blind": True,
            "selection_rule": "first ascending mirror seed with team_2 lead base Terapagos",
            "start_mirror_seed": START_SEED,
            "selected_mirror_seed": mirror_seed,
            "tested_seed_count": offset + 1,
            "format": "gen9randombattle",
            "pair_index": 1,
            "team_1_seed": team_1_seed,
            "team_2_seed": team_2_seed,
            "team_1_sha256": generated["team_1_sha256"],
            "team_2_sha256": generated["team_2_sha256"],
            "team_2_lead": "Terapagos",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return
    raise RuntimeError("no Terapagos lead found in frozen Cycle20 scan range")


if __name__ == "__main__":
    main()
