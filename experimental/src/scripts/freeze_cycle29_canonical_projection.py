#!/usr/bin/env python3
"""Freeze Cycle29 TRAIN fixtures, implementation, runtimes, and gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle29_canonical_projection_20260815"
SOURCE_SELECTION = ROOT / "experimental/runs/search_native_v2_cycle28_production_sampler_integration_20260815/selection-16.jsonl"
SELECTION = RUN / "selection-16.jsonl"
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
        raise SystemExit("Cycle29 manifest already exists")
    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if tests.get("status") != "pass" or tests.get("passed") != 19:
        raise SystemExit("Cycle29 tests did not pass")
    if sha256(SELECTION) != sha256(SOURCE_SELECTION):
        raise SystemExit("Cycle29 selection differs from Cycle28")
    selected = [json.loads(line) for line in SELECTION.read_text().splitlines()]
    classes = {name: sum(row["cycle28_fixture_class"] == name for row in selected) for name in ("opening_empty", "derived", "later_intrinsic")}
    if (
        len(selected) != 16
        or len({row["dependency_cluster_id"] for row in selected}) != 16
        or any(row["split"] != "train" for row in selected)
        or classes != {"opening_empty": 4, "derived": 7, "later_intrinsic": 5}
    ):
        raise SystemExit("Cycle29 fixture selection violates frozen contract")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    sources = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        SELECTION,
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/metagross/production_sampler_projection.py",
        ROOT / "experimental/src/scripts/audit_cycle29_canonical_projection.py",
        ROOT / "experimental/src/scripts/freeze_cycle29_canonical_projection.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "srcs/metagross/tests/test_cycle27_disable_authority_forms.py",
        ROOT / "srcs/metagross/tests/test_cycle28_production_sampler_integration.py",
        ROOT / "srcs/metagross/tests/test_cycle29_canonical_projection.py",
        ENGINE,
        Path(shutil.which("node") or ""),
    ]
    sources.extend(Path(row["raw_path"]) for row in selected)
    if any(not path.is_file() for path in sources):
        raise SystemExit("Cycle29 frozen source is missing")
    commits = sorted({row["showdown_commit"] for row in selected})
    runtimes = []
    for commit in commits:
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle29 Showdown worktree commit mismatch")
        runtimes.append({
            "commit": commit,
            "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    payload = {
        "schema": "metagross-cycle29-canonical-projection-manifest/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "selection_sha256": sha256(SELECTION),
        "cycle28_selection_sha256": sha256(SOURCE_SELECTION),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "python": "/opt/homebrew/bin/python3.11",
        "fixture_count": 16,
        "dependency_cluster_count": 16,
        "classes": classes,
        "split": "train",
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(sources), key=lambda value: str(value.resolve()))
        ],
        "showdown_runtime": runtimes,
        "gates": {
            "fixture_pass_rate": 1.0,
            "adaptive_world_counts": [16, 32],
            "world_weight_engine_action_discrepancies_max": 0,
            "empty_receipt_failures_max": 0,
            "derived_provenance_failures_max": 0,
            "move_verification_p95_ms_per_eight_max": 5.0,
            "post_run_integrity": True,
        },
        "authorization": {
            "mechanics_measurement": True,
            "fresh_operational_smoke_after_pass": True,
            "scored_h2h": False,
            "teacher_values": False,
            "training": False,
            "validation_dev_test": False,
            "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verify_manifest(MANIFEST)
    print(json.dumps({
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": payload["protocol_sha256"],
        "selection_sha256": payload["selection_sha256"],
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(payload["files"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
