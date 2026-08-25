#!/usr/bin/env python3
"""Freeze Cycle 15 selection and immutable premeasurement inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle15_teacher_stability_20260815"
C14 = ROOT / "experimental/runs/search_native_v2_cycle14_mechanics_repair_20260815"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selection_path = RUN / "selection-40.jsonl"
    manifest_path = RUN / "PREMEASUREMENT_MANIFEST.json"
    if selection_path.exists() or manifest_path.exists():
        raise RuntimeError("Cycle15 is already frozen")
    selected = [json.loads(line) for line in (C14 / "selection-200.jsonl").read_text().splitlines()]
    admitted = {
        row["dependency_cluster_id"] for row in map(
            json.loads, (C14 / "mechanics-audit/root-results.jsonl").read_text().splitlines()
        ) if row["status"] == "pass"
    }
    rows = [row for row in selected if row["dependency_cluster_id"] in admitted]
    rows.sort(key=lambda row: hashlib.sha256(
        ("cycle15-selection-v1\0" + row["dependency_cluster_id"]).encode("utf-8")
    ).hexdigest())
    rows = rows[:40]
    if len(rows) != 40 or len({row["dependency_cluster_id"] for row in rows}) != 40:
        raise RuntimeError("cannot freeze 40 cluster-unique admitted roots")
    with selection_path.open("x", encoding="ascii") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    worktrees = json.loads((
        ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
    ).read_text())
    commits = sorted({row["showdown_commit"] for row in rows})
    runtime = []
    for commit in commits:
        path = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError("Showdown worktree commit mismatch")
        runtime.append({"path": str(path), "commit": commit,
                        "dist_tree_sha256": tree_sha256(path / "dist")})
    extension = next((RUN / "engine-binding/unpacked/poke_engine").glob("poke_engine*.so"))
    files = [
        ROOT / "experimental/src/scripts/run_cycle15_teacher_stability.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/cycle12_replay_audit.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
        RUN / "PROTOCOL.md", selection_path, extension,
        *(Path(row["raw_path"]) for row in rows),
    ]
    manifest = {
        "schema": "metagross-cycle15-premeasurement-manifest/v1",
        "status": "frozen-before-values",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "selection_sha256": sha(selection_path),
        "cycle14_report_sha256": sha(C14 / "mechanics-audit/REPORT.json"),
        "engine_import_root": str((RUN / "engine-binding/unpacked").resolve()),
        "engine_binding_sha256": sha(extension),
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "showdown_runtime": runtime,
        "fixed_measurement": {
            "roots": 40, "schedules": 2, "search_repeats": 2,
            "paired_worlds": 8, "production_worlds": [16, 32],
            "production_duration_ms": [250, 500],
            "arms": ["P_exact", "P_paired", "equal_2048", "equal_8192", "equal_20000", "r1_20000"],
        },
        "authorization": {"teacher_stability_values": True, "training": False,
                          "target_collection": False, "h2h": False,
                          "validation_test": False, "sealed93": False,
                          "gpu_cloud_paid": False},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(manifest_path),
                      "selection_sha256": sha(selection_path),
                      "protocol_sha256": sha(RUN / "PROTOCOL.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
