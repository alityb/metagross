#!/usr/bin/env python3
"""Freeze Cycle48 Gate A selection/code/source provenance before any teacher value."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle48_gateA_observed_states_20260816"
ENGINE = ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
EXCLUDE = ROOT / "experimental/runs/search_native_v2_cycle13_train_rehydration_20260815/selection-200.jsonl"
WORKTREES = ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
MASTER = ROOT / "experimental/runs/search_native_v2_cycle12_transport_repair_20260815/full-corpus-index/eligible-battles.jsonl.gz"
CORPUS = ROOT / "experimental/runs/search_native_v2_cycle11_full_corpus_repair_20260815/corpus-20564.jsonl"
R1_CHECKPOINT = ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "PREMEASUREMENT_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle48 Gate A manifest already exists")
    if (run / "measurement").exists():
        raise RuntimeError("Cycle48 Gate A measurement already exists; freeze must precede it")

    selection_path = run / "selection-64.jsonl"
    selected = [json.loads(line) for line in selection_path.read_text().splitlines() if line]
    excluded = {json.loads(line)["dependency_cluster_id"] for line in EXCLUDE.read_text().splitlines() if line}
    if (
        len(selected) != 64
        or len({row["dependency_cluster_id"] for row in selected}) != 64
        or any(row["split"] != "train" for row in selected)
        or {row["dependency_cluster_id"] for row in selected} & excluded
        or len({row["raw_sha256"] for row in selected}) != 64
        or any(len(row["candidates"]) < 8 or len(row["slot_base_positions"]) != 8 for row in selected)
        or any("observed_action" in candidate for row in selected for candidate in row["candidates"])
    ):
        raise RuntimeError("selection is not 64 fresh disjoint TRAIN clusters with 8 label-blind slots")

    worktrees = json.loads(WORKTREES.read_text())
    runtime = []
    for commit in sorted({row["showdown_commit"] for row in selected}):
        path = Path(worktrees[commit])
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError("Showdown worktree commit drift")
        runtime.append({"path": str(path.resolve()), "commit": commit,
                        "dist_tree_sha256": tree_sha256(path / "dist")})

    ext = next((ENGINE / "poke_engine").glob("poke_engine*.so"))
    files = [
        run / "PROTOCOL.md", selection_path, run / "selection-report.json",
        EXCLUDE, WORKTREES, MASTER, CORPUS, R1_CHECKPOINT,
        ROOT / "experimental/src/scripts/select_cycle48_gateA_observed_states.py",
        ROOT / "experimental/src/scripts/freeze_cycle48_gateA_observed_states.py",
        ROOT / "experimental/src/scripts/run_cycle48_gateA_observed_states.py",
        ROOT / "experimental/src/scripts/tests/test_cycle48_gateA_observed_states.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/run_cycle15_teacher_stability.py",
        ROOT / "experimental/src/scripts/cycle12_replay_audit.py",
        ROOT / "experimental/src/scripts/run_cycle10_full_corpus_index.py",
        ROOT / "experimental/src/scripts/run_cycle8_replay_audit.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "experimental/src/search/public_search_state_v1.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/prior_server.py",
        ext,
        *[Path(row["raw_path"]) for row in selected],
    ]
    manifest = {
        "schema": "metagross-cycle48-gateA-premeasurement/v1",
        "status": "frozen_before_teacher_targets",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "engine_import_root": str(ENGINE.resolve()),
        "engine_binding_sha256": sha(ext),
        "selection_sha256": sha(selection_path),
        "r1_checkpoint_sha256": sha(R1_CHECKPOINT),
        "counts": {
            "dependency_clusters": 64, "slots_per_cluster": 8, "frozen_state_slots": 512,
            "schedules": 2, "worlds_per_schedule": 8, "scheduled_world_rows": 8192,
            "rematerializations_per_cluster": 3,
        },
        "measurement_contract": {
            "base_seed": 2026081648,
            "teacher_arms": {"equal8192_a": 8192, "equal8192_b": 8192, "equal20000": 20000},
            "equal_prior_via_request_authoritative_search": True,
            "aggregation": "posterior-weighted per (state, schedule); schedules never merged",
            "human_actions": "behavior anchors, never strength labels",
            "r1_outputs": "separate root controls, never search inputs",
        },
        "files": [{"path": str(p.resolve()), "sha256": sha(p)} for p in files],
        "showdown_runtime": runtime,
        "gate": {
            "coverage": 0.95, "unique_fingerprints": 512, "battles": 48,
            "top1_agreement": 0.80, "jsd_median": 0.05, "jsd_p90": 0.15,
            "zero_hidden_sensitivity": True, "zero_split_leakage": True,
        },
        "authorization": {
            "training": False, "h2h": False, "sealed93": False,
            "validation_or_dev_test": False, "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_sha256": sha(output), "selection_sha256": sha(selection_path),
        "engine_sha256": sha(ext), "protocol_sha256": sha(run / "PROTOCOL.md"),
        "file_count": len(manifest["files"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
