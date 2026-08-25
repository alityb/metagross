#!/usr/bin/env python3
"""Freeze Cycle 36 sources, natural panel, runtimes, and fixed gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle36_switch_reactivation_20260816"
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
        raise SystemExit("Cycle36 manifest already exists")
    selection = [
        json.loads(line) for line in (RUN / "selection.jsonl").read_text().splitlines()
    ]
    report = json.loads((RUN / "SELECTION_REPORT.json").read_text())
    if (
        len(selection) != 18
        or len({row["dependency_cluster_id"] for row in selection}) != 18
        or any(row["split"] != "train" for row in selection)
        or report.get("selected_battles") != 18
        or report.get("selected_states") != 18
        or sum(row["changed_role"] == "p1" for row in selection) != 9
        or sum(row["changed_role"] == "p2" for row in selection) != 9
        or sum(row["reactivation_tag"] == "switch" for row in selection) != 16
        or sum(row["reactivation_tag"] == "drag" for row in selection) != 2
    ):
        raise SystemExit("Cycle36 selection contract changed")
    pretests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if pretests.get("passed") != 50 or pretests.get("failed") != 0:
        raise SystemExit("Cycle36 prefreeze tests did not pass")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    source_paths = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        RUN / "SELECTION_REPORT.json",
        RUN / "selection.jsonl",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/export_showdown_form_ability_contract.cjs",
        ROOT / "srcs/metagross/export_showdown_public_form_contract.cjs",
        ROOT / "srcs/metagross/tests/test_cycle36_switch_reactivation_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle20_ability_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle22_certified_ability_install.py",
        ROOT / "srcs/metagross/tests/test_cycle27_disable_authority_forms.py",
        ROOT / "srcs/metagross/tests/test_cycle34_causal_disable_population.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/vendor/foul-play/fp/search/main.py",
        ROOT / "experimental/src/scripts/select_cycle36_reactivation_panel.py",
        ROOT / "experimental/src/scripts/audit_cycle36_switch_reactivation.py",
        ROOT / "experimental/src/scripts/freeze_cycle36_switch_reactivation.py",
        ROOT / "experimental/src/scripts/audit_cycle27_disable_authority_forms.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
        ROOT / (
            "experimental/runs/search_native_v2_cycle35_fresh_h2h_20260815/"
            "h2h-logs/c35h2hy01428dd.protocol.jsonl"
        ),
        ROOT / (
            "experimental/runs/search_native_v2_cycle35_fresh_h2h_20260815/"
            "STOP_REPORT.json"
        ),
        ROOT / (
            "experimental/runs/search_native_v2_cycle12_transport_repair_20260815/"
            "full-corpus-index/eligible-battles.jsonl.gz"
        ),
        ROOT / (
            "experimental/runs/search_native_v2_cycle11_full_corpus_repair_20260815/"
            "corpus-20564.jsonl"
        ),
        ENGINE,
        Path(shutil.which("node") or ""),
    ]
    source_paths.extend(Path(row["raw_path"]) for row in selection)
    if any(not path.is_file() for path in source_paths):
        raise SystemExit("Cycle36 frozen source is missing")
    commits = sorted({row["showdown_commit"] for row in selection})
    runtimes = []
    for commit in commits:
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle36 Showdown worktree commit mismatch")
        runtimes.append({
            "commit": commit,
            "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    payload = {
        "schema": "metagross-cycle36-switch-reactivation-freeze/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "selection_sha256": sha256(RUN / "selection.jsonl"),
        "selection_report_sha256": sha256(RUN / "SELECTION_REPORT.json"),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "python": "/opt/homebrew/bin/python3.11",
        "battle_count": 18,
        "state_count": 18,
        "scheduled_world_count": 288,
        "split": "train",
        "natural_inventory_events": report["natural_events"],
        "natural_inventory_battles": report["natural_battles"],
        "natural_ogerpon_public_transitions": 19,
        "natural_ogerpon_actionable_roots": 0,
        "prefreeze_tests_passed": True,
        "preserved_cycle35_trace_passed": True,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(source_paths), key=lambda value: str(value.resolve()))
        ],
        "showdown_runtime": runtimes,
        "gates": {
            "root_support": 1.0,
            "scheduled_world_support": 1.0,
            "causal_hidden_integrity_failures_max": 0,
            "move_verification_p95_ms_max": 5.0,
            "isolated_root_p95_ms_max": 1750.0
        },
        "authorization": {
            "mechanics_measurement": True,
            "fresh_operational_smoke_after_pass_if_feasible": True,
            "preserved_trace_natural_panel_fallback_after_pass": True,
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
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": payload["protocol_sha256"],
        "selection_sha256": payload["selection_sha256"],
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(payload["files"]),
        "showdown_runtimes": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
