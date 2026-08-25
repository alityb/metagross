#!/usr/bin/env python3
"""Select Cycle23's first label-blind fresh team-2 Terapagos lead."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
START_SEED = 202_623_000_000
MAX_CANDIDATES = 100_000
HELPER = ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs"


def derived_seed(master: int, label: str) -> str:
    digest = hashlib.sha256(f"{master}:1:{label}".encode("ascii")).digest()
    return ",".join(str(int.from_bytes(digest[i:i + 2], "big")) for i in range(0, 8, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for offset in range(MAX_CANDIDATES):
        master = START_SEED + offset
        first, second = derived_seed(master, "team-1"), derived_seed(master, "team-2")
        generated = json.loads(subprocess.check_output(
            ["node", str(HELPER), "gen9randombattle", first, second], cwd=ROOT, text=True
        ))
        fields = generated["team_2_packed"].split("]", 1)[0].split("|")
        if (fields[1] or fields[0]) != "Terapagos":
            continue
        payload = {
            "schema": "metagross-cycle23-form-smoke-selection/v1",
            "selection_is_label_blind": True,
            "selection_rule": "first ascending mirror seed with team_2 lead base Terapagos",
            "start_mirror_seed": START_SEED,
            "selected_mirror_seed": master,
            "tested_seed_count": offset + 1,
            "format": "gen9randombattle",
            "pair_index": 1,
            "team_1_seed": first,
            "team_2_seed": second,
            "team_1_sha256": generated["team_1_sha256"],
            "team_2_sha256": generated["team_2_sha256"],
            "team_2_lead": "Terapagos",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return
    raise RuntimeError("no Terapagos lead found in frozen Cycle23 scan range")


if __name__ == "__main__":
    main()
