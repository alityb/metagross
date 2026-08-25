#!/usr/bin/env python3
"""Freeze Cycle 39 target-aware PP panels, code, runtimes, and gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from experimental.src.scripts.audit_cycle39_target_aware_pp import selections
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle39_target_aware_pp_20260816"
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
        raise SystemExit("Cycle39 manifest already exists")
    selected = selections(RUN)
    pressure = [row for row in selected if row["cycle39_panel"] == "natural_pressure"]
    expected = Counter(
        (role, category) for role in ("p1", "p2")
        for category in ("self", "foe", "spread", "mustpressure")
        for _ in range(2)
    )
    if (
        len(selected) != 58
        or len({row["dependency_cluster_id"] for row in selected}) != 58
        or any(row["split"] != "train" for row in selected)
        or Counter((row["role"], row["category"]) for row in pressure) != expected
    ):
        raise SystemExit("Cycle39 frozen TRAIN panel changed")
    pretests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    report = json.loads((RUN / "PRESSURE_SELECTION_REPORT.json").read_text())
    if (
        pretests.get("total_passed") != 90 or pretests.get("failed") != 0
        or report.get("selected_roots") != 16
        or report.get("outcomes_used") is not False
        or report.get("teacher_values_opened") != 0
    ):
        raise SystemExit("Cycle39 tests/selection report changed")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    paths = [
        RUN / "PROTOCOL.md", RUN / "PREFREEZE_TESTS.json",
        RUN / "PRESSURE_SELECTION_REPORT.json", RUN / "pressure-selection.jsonl",
        ROOT / "srcs/vendor/foul-play/fp/battle.py",
        ROOT / "srcs/vendor/foul-play/fp/battle_modifier.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/vendor/foul-play/fp/search/main.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/export_showdown_pressure_target_contract.cjs",
        ROOT / "srcs/metagross/tests/test_cycle39_target_aware_causal_pp.py",
        ROOT / "srcs/metagross/tests/test_cycle38_temporal_form_request_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle37_own_active_resolver.py",
        ROOT / "srcs/metagross/tests/test_cycle36_switch_reactivation_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle27_disable_authority_forms.py",
        ROOT / "experimental/src/scripts/select_cycle39_pressure_panel.py",
        ROOT / "experimental/src/scripts/audit_cycle39_target_aware_pp.py",
        ROOT / "experimental/src/scripts/freeze_cycle39_target_aware_pp.py",
        ROOT / "experimental/src/scripts/audit_cycle37_own_active_resolver.py",
        ROOT / "experimental/src/scripts/audit_cycle27_disable_authority_forms.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / "experimental/runs/search_native_v2_cycle36_switch_reactivation_20260816/selection.jsonl",
        ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816/resolver-selection.jsonl",
        ROOT / "experimental/runs/search_native_v2_cycle38c_temporal_switch_20260816/mechanics-audit/REPORT.json",
        ENGINE, Path(shutil.which("node") or ""),
    ]
    paths.extend(Path(row["raw_path"]) for row in selected)
    if any(not path.is_file() for path in paths):
        raise SystemExit("Cycle39 frozen source missing")
    runtimes = []
    for commit in sorted({row["showdown_commit"] for row in selected}):
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle39 Showdown runtime mismatch")
        runtimes.append({
            "commit": commit, "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    pressure_contract = json.loads(subprocess.check_output([
        "node", str(ROOT / "srcs/metagross/export_showdown_pressure_target_contract.cjs")
    ], text=True))
    if (
        pressure_contract.get("showdown_commit")
        != "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5"
        or pressure_contract.get("row_count", 0) < 950
    ):
        raise SystemExit("Cycle39 pinned Showdown Pressure contract invalid")
    payload = {
        "schema": "metagross-cycle39-target-aware-pp-freeze/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "canonical_absolute_run_dir": str(RUN.resolve()),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "pressure_contract_showdown_commit": pressure_contract["showdown_commit"],
        "pressure_contract_row_count": pressure_contract["row_count"],
        "python": "/opt/homebrew/bin/python3.11",
        "root_count": 58, "scheduled_world_count": 928, "split": "train",
        "preserved_root_count": 42, "natural_pressure_root_count": 16,
        "prefreeze_tests_passed": True, "prefreeze_test_count": 90,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(paths), key=lambda value: str(value.resolve()))
        ],
        "showdown_runtime": runtimes,
        "gates": {
            "root_support": 1.0, "scheduled_world_support": 1.0,
            "preserved_oricorio_pass": True, "natural_pressure_pass": 16,
            "causal_hidden_action_failures_max": 0,
            "move_verification_p95_ms_max": 5.0,
            "isolated_root_p95_ms_max": 1750.0,
        },
        "authorization": {
            "fresh_h2h_protocol_design_after_pass": True, "h2h": False,
            "teacher_values": False, "training": False,
            "validation_dev_test": False, "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verify_manifest(MANIFEST)
    print(json.dumps({
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": payload["protocol_sha256"],
        "pressure_selection_sha256": sha256(RUN / "pressure-selection.jsonl"),
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(payload["files"]), "showdown_runtimes": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
