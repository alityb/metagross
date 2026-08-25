#!/usr/bin/env python3
"""Freeze Cycle 19 before its one-decision live smoke."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle19_operational_repair_20260815"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PRESMOKE_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle19 presmoke is already frozen")
    engine_root = ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    pair = RUN / "smoke-result.json.pairs.json"
    pair_payload = json.loads(pair.read_text())
    if len(pair_payload.get("pairs", [])) != 1:
        raise RuntimeError("Cycle19 smoke requires exactly one fresh pair")
    if sha(pair) != "efbfa7d3df5621374e7e53ccbb8a54b81f35279a288c383b064b7de2810528f5":
        raise RuntimeError("Cycle19 smoke pair changed")
    smoke_row = pair_payload["pairs"][0]
    smoke_team_pair = tuple(
        sorted((smoke_row["team_1_sha256"], smoke_row["team_2_sha256"]))
    )
    prior_team_pairs = set()
    for prior_path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if prior_path.resolve() == pair.resolve():
            continue
        try:
            prior_payload = json.loads(prior_path.read_text())
        except Exception:
            continue
        for row in prior_payload.get("pairs", []):
            if "team_1_sha256" in row and "team_2_sha256" in row:
                prior_team_pairs.add(
                    tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
                )
    if smoke_team_pair in prior_team_pairs:
        raise RuntimeError("Cycle19 smoke team pair overlaps an earlier artifact")
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
    ).strip()
    files = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        pair,
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/run_cycle19_operational_smoke.sh",
        ROOT / "experimental/src/scripts/freeze_cycle19_presmoke.py",
        ROOT / "experimental/src/scripts/verify_cycle19_presmoke.py",
        ROOT / "experimental/src/scripts/tests/test_cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/tests/test_monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/eval/run.py",
        ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
        ROOT / "experimental/src/scripts/start_showdown.sh",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
        extension,
    ]
    manifest = {
        "schema": "metagross-cycle19-presmoke-freeze/v1",
        "status": "frozen_before_operational_smoke",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "smoke_pair_sha256": sha(pair),
        "smoke_team_pair_disjoint_from_prior_artifacts": True,
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {
            "import_root": str(engine_root.resolve()),
            "native_path": str(extension.resolve()),
            "native_sha256": sha(extension),
            "general_environment_mutated": False,
        },
        "showdown": {
            "path": str(showdown.resolve()),
            "commit": commit,
            "dist_tree_sha256": tree_sha256(showdown / "dist"),
        },
        "smoke_gate": {
            "candidate_decisions": 1,
            "schedules": 2,
            "worlds_per_schedule": 8,
            "iterations_per_world": 8192,
            "considered_fraction": 0.75,
            "require_public_action_confirmation": True,
            "fallback_timeout_semantic_failures": 0,
        },
        "authorization": {
            "operational_smoke": True,
            "scored_pair_generation_before_pass": False,
            "h2h_games": 0,
            "training": False,
            "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "protocol_sha256": manifest["protocol_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
