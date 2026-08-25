#!/usr/bin/env python3
"""Freeze Cycle 19 prospective H2H after smoke pass and before games."""

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
    output = RUN / "H2H_PREMEASUREMENT_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle19 H2H already frozen")
    smoke_evidence = RUN / "SMOKE_EVIDENCE_MANIFEST.json"
    if sha(smoke_evidence) != "8a0bb940cdbe51006fc115cc4cd5c68df76894606959a71da7c5176359c59356":
        raise RuntimeError("Cycle19 smoke evidence changed")
    smoke = json.loads(smoke_evidence.read_text())
    if smoke.get("status") != "pass" or smoke.get("authorization", {}).get("fresh_scored_pair_generation") is not True:
        raise RuntimeError("Cycle19 smoke did not authorize pair generation")
    pair = RUN / "h2h-result.json.pairs.json"
    if sha(pair) != "689ed2dedb0baa29d9a686f37200a42e58a02eeb4db5752327299bc0ef034160":
        raise RuntimeError("Cycle19 H2H pair manifest changed")
    pairs = json.loads(pair.read_text()).get("pairs", [])
    current = {tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pairs}
    if len(pairs) != 10 or len(current) != 10:
        raise RuntimeError("Cycle19 H2H pairs are not ten unique unordered pairs")
    prior = set()
    for path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if path.resolve() == pair.resolve():
            continue
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        for row in value.get("pairs", []):
            if "team_1_sha256" in row and "team_2_sha256" in row:
                prior.add(tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))))
    if current & prior:
        raise RuntimeError("Cycle19 H2H pair overlaps prior local artifacts")
    engine_root = ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True).strip()
    files = [
        RUN / "H2H_PROTOCOL.md", RUN / "H2H_PREFREEZE_TESTS.json", pair, smoke_evidence,
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/summarize_cycle19_h2h.py",
        ROOT / "experimental/src/scripts/run_cycle19_h2h.sh",
        ROOT / "experimental/src/scripts/freeze_cycle19_h2h.py",
        ROOT / "experimental/src/scripts/verify_cycle19_h2h_freeze.py",
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
        "schema": "metagross-cycle19-h2h-premeasurement/v1",
        "status": "frozen_before_scored_games",
        "protocol_sha256": sha(RUN / "H2H_PROTOCOL.md"),
        "pair_sha256": sha(pair),
        "smoke_evidence_sha256": sha(smoke_evidence),
        "team_pairs_disjoint_from_all_prior_local_artifacts": True,
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {"import_root": str(engine_root.resolve()), "native_sha256": sha(extension)},
        "showdown": {"path": str(showdown.resolve()), "commit": commit, "dist_tree_sha256": tree_sha256(showdown / "dist")},
        "fixed_gate": {"games": 20, "pairs": 10, "candidate_wins": 13, "all_failures": 0},
        "authorization": {"h2h_games": 20, "continuation": False, "training": False, "sealed93": False, "gpu_cloud_paid": False},
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "protocol_sha256": manifest["protocol_sha256"], "pair_sha256": manifest["pair_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
