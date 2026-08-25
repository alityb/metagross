#!/usr/bin/env python3
"""Freeze Cycle 16 after its separately tested rqid repair."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experimental.src.scripts.freeze_cycle15_teacher_stability import ROOT, sha
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

RUN = ROOT / "experimental/runs/search_native_v2_cycle16_teacher_stability_20260815"


def main() -> None:
    manifest_path = RUN / "PREMEASUREMENT_MANIFEST.json"
    if manifest_path.exists():
        raise RuntimeError("Cycle16 is already frozen")
    selection = RUN / "selection-40.jsonl"
    c15_selection = ROOT / "experimental/runs/search_native_v2_cycle15_teacher_stability_20260815/selection-40.jsonl"
    if sha(selection) != sha(c15_selection):
        raise RuntimeError("Cycle16 selection is not byte-identical to Cycle15")
    rows = [json.loads(line) for line in selection.read_text().splitlines()]
    worktrees = json.loads((
        ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
    ).read_text())
    runtime = []
    for commit in sorted({row["showdown_commit"] for row in rows}):
        path = Path(worktrees[commit]).resolve()
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != commit:
            raise RuntimeError("Showdown worktree mismatch")
        runtime.append({"path": str(path), "commit": commit,
                        "dist_tree_sha256": tree_sha256(path / "dist")})
    extension = next((RUN / "engine-binding/unpacked/poke_engine").glob("poke_engine*.so"))
    files = [
        ROOT / "experimental/src/scripts/run_cycle15_teacher_stability.py",
        ROOT / "experimental/src/scripts/run_cycle16_teacher_stability.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/cycle12_replay_audit.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "srcs/metagross/prior_server.py", ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
        RUN / "PROTOCOL.md", selection, extension,
        *(Path(row["raw_path"]) for row in rows),
    ]
    manifest = {
        "schema": "metagross-cycle16-premeasurement-manifest/v1",
        "status": "frozen-before-cycle16-values",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "selection_sha256": sha(selection),
        "cycle15_selection_sha256": sha(c15_selection),
        "engine_import_root": str((RUN / "engine-binding/unpacked").resolve()),
        "engine_binding_sha256": sha(extension),
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "showdown_runtime": runtime,
        "fixed_measurement": {
            "roots": 40, "correlation_variants": 2, "schedules": 2,
            "search_repeats": 2, "paired_worlds": 8,
            "production_worlds": [16, 32], "production_duration_ms": [250, 500],
            "arms": ["P_exact", "P_paired", "equal_2048", "equal_8192", "equal_20000", "r1_20000"],
            "gates": {"stability": .70, "schedule_agreement": .80,
                      "agreement_with_equal_20k": .80, "mean_repeat_jsd_max": .10,
                      "stable_production_differences_min": 4},
        },
        "authorization": {"r1_parity_and_teacher_stability": True, "training": False,
                          "target_collection": False, "h2h": False,
                          "validation_test": False, "sealed93": False,
                          "gpu_cloud_paid": False},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(manifest_path),
                      "protocol_sha256": sha(RUN / "PROTOCOL.md"),
                      "selection_sha256": sha(selection)}, sort_keys=True))


if __name__ == "__main__": main()
