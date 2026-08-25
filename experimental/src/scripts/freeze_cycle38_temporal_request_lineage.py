#!/usr/bin/env python3
"""Freeze Cycle 38 temporal request-lineage mechanics gate."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.scripts.audit_cycle38_temporal_request_lineage import (
    ORICORIO_PP_ROOT,
    TEMPORAL_ROOTS,
    selected_rows,
)
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle38_temporal_request_20260816"
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
        raise SystemExit("Cycle38 manifest already exists")
    selected = selected_rows()
    ids = {row["model_information_fingerprint_sha256"] for row in selected}
    if (
        len(selected) != 42 or len({row["dependency_cluster_id"] for row in selected}) != 42
        or any(row["split"] != "train" for row in selected)
        or not TEMPORAL_ROOTS.issubset(ids) or ORICORIO_PP_ROOT not in ids
    ):
        raise SystemExit("Cycle38 fixed TRAIN selection changed")
    pretests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if pretests.get("total_passed") != 78 or pretests.get("failed") != 0:
        raise SystemExit("Cycle38 prefreeze tests did not pass")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    source_paths = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        ROOT / "srcs/vendor/foul-play/fp/battle.py",
        ROOT / "srcs/vendor/foul-play/fp/battle_modifier.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/vendor/foul-play/fp/search/main.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/tests/test_cycle38_temporal_form_request_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle37_own_active_resolver.py",
        ROOT / "srcs/metagross/tests/test_cycle36_switch_reactivation_lineage.py",
        ROOT / "experimental/src/scripts/audit_cycle38_temporal_request_lineage.py",
        ROOT / "experimental/src/scripts/freeze_cycle38_temporal_request_lineage.py",
        ROOT / "experimental/src/scripts/audit_cycle37_own_active_resolver.py",
        ROOT / "experimental/src/scripts/audit_cycle27_disable_authority_forms.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "experimental/runs/search_native_v2_cycle36_switch_reactivation_20260816/selection.jsonl",
        ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816/resolver-selection.jsonl",
        ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816/mechanics-audit/root-results.jsonl",
        ENGINE,
        Path(shutil.which("node") or ""),
    ]
    source_paths.extend(Path(row["raw_path"]) for row in selected)
    if any(not path.is_file() for path in source_paths):
        raise SystemExit("Cycle38 frozen source is missing")
    runtimes = []
    for commit in sorted({row["showdown_commit"] for row in selected}):
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle38 Showdown runtime commit mismatch")
        runtimes.append({
            "commit": commit,
            "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    payload = {
        "schema": "metagross-cycle38-temporal-request-freeze/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "python": "/opt/homebrew/bin/python3.11",
        "root_count": 42,
        "expected_supported_root_count": 41,
        "expected_supported_world_count": 656,
        "split": "train",
        "temporal_failure_fingerprints": sorted(TEMPORAL_ROOTS),
        "expected_oricorio_pp_failure_fingerprint": ORICORIO_PP_ROOT,
        "prefreeze_tests_passed": True,
        "prefreeze_test_count": 78,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(source_paths), key=lambda value: str(value.resolve()))
        ],
        "showdown_runtime": runtimes,
        "gates": {
            "temporal_roots_pass": 10,
            "non_pp_roots_pass": 41,
            "supported_worlds": 656,
            "expected_pp_control_failures": 1,
            "unexpected_failures_max": 0,
            "move_verification_p95_ms_max": 5.0,
            "isolated_root_p95_ms_max": 1750.0,
        },
        "authorization": {
            "cycle39_pp_gate_after_pass": True,
            "fresh_h2h": False,
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
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(payload["files"]),
        "showdown_runtimes": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
