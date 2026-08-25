#!/usr/bin/env python3
"""Freeze Cycle 37 resolver protocol, panels, sources, and runtimes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816"
C36 = ROOT / "experimental/runs/search_native_v2_cycle36_switch_reactivation_20260816"
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
        raise SystemExit("Cycle37 manifest already exists")
    cycle36 = [json.loads(line) for line in (C36 / "selection.jsonl").read_text().splitlines()]
    resolver = [
        json.loads(line) for line in (RUN / "resolver-selection.jsonl").read_text().splitlines()
    ]
    selection = cycle36 + resolver
    selection_report = json.loads((RUN / "RESOLVER_SELECTION_REPORT.json").read_text())
    if (
        len(cycle36) != 18 or len(resolver) != 24
        or len({row["dependency_cluster_id"] for row in selection}) != 42
        or any(row["split"] != "train" for row in selection)
        or selection_report.get("selected_by_role") != {"p1": 12, "p2": 12}
        or selection_report.get("selected_states") != 24
    ):
        raise SystemExit("Cycle37 frozen selection contract changed")
    pretests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if pretests.get("total_passed") != 68 or pretests.get("failed") != 0:
        raise SystemExit("Cycle37 prefreeze tests did not pass")
    worktrees = json.loads((ROOT / (
        "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/"
        "replay-worktrees.json"
    )).read_text())
    source_paths = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        RUN / "RESOLVER_SELECTION_REPORT.json",
        RUN / "resolver-selection.jsonl",
        C36 / "selection.jsonl",
        C36 / "PREMEASUREMENT_MANIFEST.json",
        C36 / "mechanics-audit/REPORT.json",
        ROOT / "srcs/vendor/foul-play/fp/battle.py",
        ROOT / "srcs/vendor/foul-play/fp/battle_modifier.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/vendor/foul-play/fp/search/main.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/export_showdown_public_form_contract.cjs",
        ROOT / "srcs/metagross/export_showdown_form_ability_contract.cjs",
        ROOT / "srcs/metagross/tests/test_cycle37_own_active_resolver.py",
        ROOT / "srcs/metagross/tests/test_cycle36_switch_reactivation_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle20_ability_lineage.py",
        ROOT / "srcs/metagross/tests/test_cycle22_certified_ability_install.py",
        ROOT / "srcs/metagross/tests/test_cycle27_disable_authority_forms.py",
        ROOT / "srcs/metagross/tests/test_cycle34_causal_disable_population.py",
        ROOT / "experimental/src/scripts/select_cycle37_resolver_panel.py",
        ROOT / "experimental/src/scripts/audit_cycle37_own_active_resolver.py",
        ROOT / "experimental/src/scripts/freeze_cycle37_own_active_resolver.py",
        ROOT / "experimental/src/scripts/audit_cycle27_disable_authority_forms.py",
        ROOT / "experimental/src/scripts/audit_cycle14_mechanics_repair.py",
        ROOT / "experimental/src/scripts/audit_cycle13_train_rehydration.py",
        ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs",
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
        raise SystemExit("Cycle37 frozen source is missing")
    runtimes = []
    for commit in sorted({row["showdown_commit"] for row in selection}):
        worktree = Path(worktrees[commit]).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != commit:
            raise SystemExit("Cycle37 Showdown worktree commit mismatch")
        runtimes.append({
            "commit": commit,
            "path": str(worktree),
            "dist_tree_sha256": tree_sha256(worktree / "dist"),
        })
    payload = {
        "schema": "metagross-cycle37-own-active-resolver-freeze/v1",
        "protocol_sha256": sha256(RUN / "PROTOCOL.md"),
        "cycle36_selection_sha256": sha256(C36 / "selection.jsonl"),
        "resolver_selection_sha256": sha256(RUN / "resolver-selection.jsonl"),
        "resolver_selection_report_sha256": sha256(RUN / "RESOLVER_SELECTION_REPORT.json"),
        "engine_binding_path": str(ENGINE.resolve()),
        "engine_binding_sha256": sha256(ENGINE),
        "python": "/opt/homebrew/bin/python3.11",
        "cycle36_root_count": 18,
        "resolver_root_count": 24,
        "root_count": 42,
        "scheduled_world_count": 672,
        "split": "train",
        "prefreeze_tests_passed": True,
        "prefreeze_test_count": 68,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(source_paths), key=lambda value: str(value.resolve()))
        ],
        "showdown_runtime": runtimes,
        "gates": {
            "root_support": 1.0,
            "scheduled_world_support": 1.0,
            "resolver_causal_hidden_failures_max": 0,
            "move_verification_p95_ms_max": 5.0,
            "isolated_root_p95_ms_max": 1750.0
        },
        "authorization": {
            "mechanics_measurement": True,
            "fresh_h2h_protocol_design_after_pass": True,
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
        "cycle36_selection_sha256": payload["cycle36_selection_sha256"],
        "resolver_selection_sha256": payload["resolver_selection_sha256"],
        "engine_binding_sha256": payload["engine_binding_sha256"],
        "files": len(payload["files"]),
        "showdown_runtimes": len(runtimes),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
