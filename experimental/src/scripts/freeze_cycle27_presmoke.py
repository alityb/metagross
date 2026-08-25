#!/usr/bin/env python3
"""Freeze Cycle27 before the one fresh second-root operational smoke."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.cycle27_canonical_smoke import config_identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle27_disable_authority_forms_20260815"
BASE = ROOT / "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
PAIR_SHA = "5792a08e8a906970af94741a937a3879d3e5e2ec825a29216d86d820ed3f1b9c"
ENGINE_SHA = "c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PRESMOKE_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle27 smoke is already frozen")
    mechanics = json.loads((RUN / "mechanics-audit/REPORT.json").read_text())
    if (
        mechanics.get("status") != "pass"
        or mechanics.get("counts", {}).get("passed_roots") != 199
        or mechanics.get("counts", {}).get("supported_scheduled_worlds") != 3184
        or mechanics.get("authorization", {}).get("fresh_operational_smoke") is not True
    ):
        raise RuntimeError("Cycle27 mechanics did not authorize this smoke")
    tests = json.loads((RUN / "PRESMOKE_TESTS.json").read_text())
    if tests.get("status") != "pass" or tests.get("passed") != 12:
        raise RuntimeError("Cycle27 presmoke tests did not pass")
    pair = RUN / "smoke-result.json.pairs.json"
    if sha(pair) != PAIR_SHA:
        raise RuntimeError("Cycle27 fresh pair changed")
    payload = json.loads(pair.read_text())
    if len(payload.get("pairs", [])) != 1:
        raise RuntimeError("Cycle27 smoke must contain one fresh pair")
    row = payload["pairs"][0]
    current = tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
    prior: set[tuple[str, str]] = set()
    for path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if path.resolve() == pair.resolve():
            continue
        try:
            old_rows = json.loads(path.read_text()).get("pairs", [])
        except Exception:
            continue
        for old in old_rows:
            if "team_1_sha256" in old and "team_2_sha256" in old:
                prior.add(tuple(sorted((old["team_1_sha256"], old["team_2_sha256"]))))
    if current in prior:
        raise RuntimeError("Cycle27 smoke pair overlaps a prior pair")
    canonical = RUN / "CANONICAL_SMOKE_ARGV.json"
    prepare = config_identity(canonical, "prepare")
    live = config_identity(canonical, "live", PAIR_SHA)
    if prepare != live or prepare != payload["config_sha256"]:
        raise RuntimeError("Cycle27 preparation/live config identity changed")
    registration_dir = RUN / "smoke-registrations"
    if not registration_dir.is_dir() or any(registration_dir.iterdir()):
        raise RuntimeError("Cycle27 registration domain is not fresh and empty")

    engine_root = BASE / "engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    if sha(extension) != ENGINE_SHA:
        raise RuntimeError("Cycle27 pinned engine changed")
    files = [
        RUN / "PROTOCOL.md",
        RUN / "PREMEASUREMENT_MANIFEST.json",
        RUN / "mechanics-audit/REPORT.json",
        RUN / "mechanics-audit/root-results.jsonl",
        RUN / "selection-200.jsonl",
        RUN / "PRESMOKE_TESTS.json",
        RUN / "CANONICAL_SMOKE_ARGV.json",
        RUN / "SHOWDOWN_RUNTIME_MANIFEST.json",
        pair,
        ROOT / "experimental/src/scripts/cycle27_canonical_smoke.py",
        ROOT / "experimental/src/scripts/watch_cycle27_registrations.py",
        ROOT / "experimental/src/scripts/monitor_cycle27_second_root_smoke.py",
        ROOT / "experimental/src/scripts/run_cycle27_second_root_smoke.sh",
        ROOT / "experimental/src/scripts/freeze_cycle27_presmoke.py",
        ROOT / "experimental/src/scripts/verify_cycle27_presmoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/eval/run.py",
        ROOT / "srcs/metagross/showdown_runtime_server.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/metagross/tests/test_cycle27_disable_authority_forms.py",
        ROOT / "srcs/metagross/tests/test_cycle25_receipt_attribution.py",
        extension,
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
    ]
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema": "metagross-cycle27-second-root-presmoke-freeze/v1",
        "status": "frozen_before_fresh_second_root_smoke",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "mechanics_report_sha256": sha(RUN / "mechanics-audit/REPORT.json"),
        "pair_sha256": PAIR_SHA,
        "canonical_argv_sha256": sha(canonical),
        "config_sha256": prepare,
        "files": [
            {"path": str(path.resolve()), "sha256": sha(path)} for path in files
        ],
        "engine": {
            "import_root": str(engine_root.resolve()),
            "native_path": str(extension.resolve()),
            "native_sha256": sha(extension),
        },
        "showdown": {
            "path": str(showdown.resolve()),
            "commit": commit,
            "dist_tree_sha256": tree_sha256(showdown / "dist"),
        },
        "gate": {
            "fresh_pair": True,
            "registration_consumed_exactly_once": True,
            "target_decision_index": 1,
            "preceding_public_opponent_move": True,
            "production_declared_count": [16, 32],
            "candidate_cells": "2x8",
            "candidate_iterations_per_world": 8192,
            "typed_pp_disable_receipt_each_candidate_world": True,
            "receipt_authorities": [
                "causal_disable",
                "world_mechanical_disable",
            ],
            "same_root_rqid_decision": True,
            "post_execution_receipts": 0,
            "exact_selected_action_publicly_executed": True,
            "semantic_operational_failures": 0,
        },
        "authorization_on_pass": {
            "scored_h2h": False,
            "training": False,
            "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": sha(output),
                "pair_sha256": PAIR_SHA,
                "config_sha256": prepare,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
