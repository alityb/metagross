#!/usr/bin/env python3
"""Freeze the Cycle 13 TRAIN-only rehydration manifest before measurement."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import sha256_path, tree_sha256


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle13_train_rehydration_20260815"
BINDING = RUN / "engine-binding/unpacked/poke_engine/poke_engine.cpython-311-darwin.so"


def include_tree(paths: set[Path], root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            paths.add(path.resolve())


def main() -> None:
    output = RUN / "PREMEASUREMENT_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle 13 manifest already exists")
    selection_path = RUN / "selection-200.jsonl"
    selected = [json.loads(line) for line in selection_path.read_text().splitlines() if line]
    if (
        len(selected) != 200
        or any(row.get("split") != "train" for row in selected)
        or len({row["dependency_cluster_id"] for row in selected}) != 200
    ):
        raise RuntimeError("Cycle 13 selection is not 200 unique TRAIN clusters")

    paths = {
        RUN / "PROTOCOL.md",
        selection_path,
        RUN / "selection-report.json",
        RUN / "engine-binding/poke_engine-0.0.47-cp311-cp311-macosx_11_0_arm64.whl",
        RUN / "engine-binding/unpacked/poke_engine/__init__.py",
        RUN / "engine-binding/unpacked/poke_engine/poke_engine.pyi",
        BINDING,
        ROOT / "experimental/runs/search_native_v2_cycle12_transport_repair_20260815/full-corpus-index/REPORT.json",
        ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json",
        ROOT / "experimental/src/scripts/select_cycle13_train_states.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/freeze_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/tests/test_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "experimental/src/scripts/cycle8_replay_audit.py",
        ROOT / "experimental/src/scripts/cycle9_replay_audit.py",
        ROOT / "experimental/src/scripts/cycle11_replay_audit.py",
        ROOT / "experimental/src/scripts/cycle12_replay_audit.py",
        ROOT / "experimental/src/scripts/run_cycle10_full_corpus_index.py",
        ROOT / "experimental/src/scripts/run_cycle8_replay_audit.py",
        ROOT / "experimental/src/search/public_search_state_v1.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/export_showdown_public_form_contract.cjs",
        ROOT / "srcs/metagross/tests/test_causal_reveal_ledger.py",
        ROOT / "experimental/engine/pe_v3_learned_priors/Cargo.toml",
        ROOT / "experimental/engine/pe_v3_learned_priors/Cargo.lock",
        ROOT / "experimental/engine/pe_v3_learned_priors/poke-engine-py/Cargo.toml",
    }
    paths.update(Path(row["raw_path"]).resolve() for row in selected)
    include_tree(paths, ROOT / "srcs/vendor/foul-play/fp")
    include_tree(paths, ROOT / "srcs/vendor/foul-play/data")
    include_tree(paths, ROOT / "experimental/engine/pe_v3_learned_priors/src")
    include_tree(paths, ROOT / "experimental/engine/pe_v3_learned_priors/poke-engine-py/src")
    for executable in ("node", "/opt/homebrew/bin/python3.11"):
        found = shutil.which(executable) if "/" not in executable else executable
        if not found:
            raise RuntimeError(f"missing runtime executable: {executable}")
        paths.add(Path(found).resolve())
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"missing frozen files: {missing[:5]}")

    worktrees = json.loads((
        ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
    ).read_text())
    runtime_paths = {
        commit: Path(worktrees[commit]).resolve()
        for commit in sorted({row["showdown_commit"] for row in selected})
    }
    external = (ROOT / "external/pokemon-showdown").resolve()
    external_commit = subprocess.check_output(
        ["git", "-C", str(external), "rev-parse", "HEAD"], text=True,
    ).strip()
    runtime_paths.setdefault(external_commit, external)
    runtimes = []
    for commit, path in sorted(runtime_paths.items()):
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise RuntimeError(f"Showdown worktree commit mismatch: {path}")
        runtimes.append({
            "commit": commit, "path": str(path),
            "dist_tree_sha256": tree_sha256(path / "dist"),
        })

    manifest = {
        "schema": "metagross-cycle13-premeasurement-manifest/v1",
        "status": "frozen_before_measurement",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_path(RUN / "PROTOCOL.md"),
        "selection_sha256": sha256_path(selection_path),
        "selection_count": 200,
        "dependency_cluster_count": 200,
        "engine_binding_path": str(BINDING.resolve()),
        "engine_binding_sha256": sha256_path(BINDING),
        "engine_build_contract": {
            "python": "CPython 3.11 arm64",
            "release": True,
            "default_features": False,
            "features": ["poke-engine/terastallization"],
            "source_has_cycle6_symmetric_native_masks": True,
        },
        "files": [
            {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
             "sha256": sha256_path(path)}
            for path in sorted(paths, key=str)
        ],
        "showdown_runtime": runtimes,
        "fixed_measurement": {
            "base_seed": 2026081513,
            "schedules": 2, "worlds_per_schedule": 8,
            "exact_repeats": 2, "minimum_passed_roots": 190,
        },
        "authorization": {
            "train_split_mechanics_only": True,
            "validation_or_dev_test_labels": False,
            "teacher_values": False, "training": False, "h2h": False,
            "sealed_93": False, "gpu_cloud_paid": False,
        },
        "premeasurement_checks": {
            "selection_label_blind": True,
            "one_root_per_dependency_cluster": True,
            "validation_index_files_opened": 0,
            "dev_test_index_files_opened": 0,
            "teacher_q_visit_outcome_fields_opened": 0,
            "debug_smoke_excluded_from_counts": True,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(output), "manifest_sha256": sha256_path(output),
        "protocol_sha256": manifest["protocol_sha256"],
        "selection_sha256": manifest["selection_sha256"],
        "engine_binding_sha256": manifest["engine_binding_sha256"],
        "file_count": len(manifest["files"]),
        "showdown_runtime_count": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
