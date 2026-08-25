#!/usr/bin/env python3
"""Freeze Cycle 18 H2H code, teams, runtimes and gates before games."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle18_h2h_protocol_20260815"


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PREMEASUREMENT_MANIFEST.json"
    if output.exists(): raise RuntimeError("Cycle18 already frozen")
    pair_path = RUN / "preflight-result.json.pairs.json"
    pair = json.loads(pair_path.read_text())
    pairs = pair.get("pairs", [])
    if len(pairs) != 10 or len({row["pair_id"] for row in pairs}) != 10:
        raise RuntimeError("Cycle18 pair manifest is not ten unique pairs")
    team_pairs = {tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pairs}
    if len(team_pairs) != 10: raise RuntimeError("Cycle18 contains duplicate unordered teams")
    prior_pairs = set()
    for path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if path.resolve() == pair_path.resolve(): continue
        try: payload = json.loads(path.read_text())
        except Exception: continue
        for row in payload.get("pairs", []):
            if "team_1_sha256" in row and "team_2_sha256" in row:
                prior_pairs.add(tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))))
    if team_pairs & prior_pairs: raise RuntimeError("Cycle18 team pair overlaps prior artifacts")
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True).strip()
    extension = next((ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked/poke_engine").glob("poke_engine*.so"))
    files = [
        RUN / "PROTOCOL.md", pair_path,
        ROOT / "experimental/src/scripts/run_cycle18_equal8192_h2h.sh",
        ROOT / "experimental/src/scripts/cycle18_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/verify_cycle18_h2h_freeze.py",
        ROOT / "experimental/src/eval/run.py", ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
        ROOT / "experimental/src/scripts/start_showdown.sh",
        ROOT / "srcs/metagross/run_foul_play.py", ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt", extension,
    ]
    manifest = {
        "schema": "metagross-cycle18-h2h-freeze/v1", "status": "frozen_before_games",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "showdown": {"path": str(showdown.resolve()), "commit": commit,
                     "dist_tree_sha256": tree_sha256(showdown / "dist")},
        "pair_manifest": {"path": str(pair_path.resolve()), "sha256": sha(pair_path),
                          "config_sha256": pair["config_sha256"], "pairs": 10,
                          "unordered_team_pairs_disjoint_from_prior": True},
        "candidate": {"iterations_per_world": 8192, "schedules": 2, "worlds_per_schedule": 8,
                      "priors": "equal", "request_authoritative_root": True,
                      "engine_binding_sha256": sha(extension)},
        "production": {"checkpoint_sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
                       "search_time_ms": 500, "parallelism": 8, "threads": 1, "c_puct": 2.0},
        "seeds": {"mirror": 202618081501,
                  "production": "1818181818181818181818181818181818181818181818181818181818181818"},
        "gate": {"games": 20, "mirrored_pairs": 10, "candidate_wins_to_continue": 13,
                 "void_semantic_operational_failures": 0},
        "authorization": {"games": 20, "training": False, "target_collection": False,
                          "continuation": False, "sealed93": False, "gpu_cloud_paid": False},
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "pair_sha256": sha(pair_path),
                      "protocol_sha256": sha(RUN / "PROTOCOL.md")}, sort_keys=True))


if __name__ == "__main__": main()
