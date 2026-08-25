#!/usr/bin/env python3
"""Freeze Cycle 26 selection, sources, pinned runtimes, and gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle26_pp_conditional_belief_20260815"
SOURCE_SELECTION = ROOT / "experimental/runs/search_native_v2_cycle14_mechanics_repair_20260815/selection-200.jsonl"
SELECTION = RUN / "selection-200.jsonl"
MANIFEST = RUN / "PREMEASUREMENT_MANIFEST.json"
ENGINE = ROOT / (
    "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815/"
    "engine-binding/unpacked/poke_engine/poke_engine.cpython-311-darwin.so"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit("Cycle26 manifest already exists")
    if sha256(SELECTION) != sha256(SOURCE_SELECTION):
        raise SystemExit("Cycle26 selection is not byte-identical to Cycle14")
    selected = [json.loads(line) for line in SELECTION.read_text().splitlines()]
    if (
        len(selected) != 200
        or len({row["dependency_cluster_id"] for row in selected}) != 200
        or any(row["split"] != "train" for row in selected)
    ):
        raise SystemExit("Cycle26 selection violates frozen TRAIN/cluster contract")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    source_paths = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        SELECTION,
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "experimental/src/scripts/audit_cycle26_pp_conditional_belief.py",
        ROOT / "experimental/src/scripts/freeze_cycle26_pp_conditional_belief.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "srcs/metagross/tests/test_cycle26_pp_conditional_belief.py",
        ENGINE,
        Path(shutil.which("node") or ""),
    ]
    source_paths.extend(Path(row["raw_path"]) for row in selected)
    if any(not path.is_file() for path in source_paths):
        raise SystemExit("Cycle26 frozen source is missing")
    files = [
        {"path": str(path.resolve()), "sha256": sha256(path)}
        for path in sorted(set(source_paths), key=lambda value: str(value.resolve()))
    ]
    commits = sorted({row["showdown_commit"] for row in selected})
    runtimes = []
    for commit in commits:
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle26 Showdown worktree commit mismatch")
        runtimes.append({
            "commit": commit,
            "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    payload = {
        "schema": "metagross-cycle26-premeasurement-manifest/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "selection_sha256": sha256(SELECTION),
        "cycle14_selection_sha256": sha256(SOURCE_SELECTION),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "python": "/opt/homebrew/bin/python3.11",
        "root_count": 200,
        "dependency_cluster_count": 200,
        "split": "train",
        "files": files,
        "showdown_runtime": runtimes,
        "gates": {
            "root_support_min": 0.95,
            "scheduled_world_support_min": 0.95,
            "causal_integrity_failures_max": 0,
            "move_verification_p95_ms_max": 5.0,
            "isolated_root_p95_ms_max": 1750.0
        },
        "authorization": {
            "mechanics_measurement": True,
            "fresh_operational_smoke_after_pass": True,
            "scored_h2h": False,
            "teacher_values": False,
            "training": False,
            "validation_dev_test": False,
            "sealed93": False,
            "gpu_cloud_paid": False
        }
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verify_manifest(MANIFEST)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": payload["protocol_sha256"],
        "selection_sha256": payload["selection_sha256"],
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(files),
        "showdown_runtimes": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
