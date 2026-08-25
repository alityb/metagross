#!/usr/bin/env python3
"""Freeze Cycle23 after monitor tests and before live execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
from experimental.src.scripts.verify_cycle22_presmoke import verify as verify_cycle22


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle23_first_decision_monitor_20260815"
BASE = ROOT / "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PRESMOKE_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle23 presmoke already frozen")
    verify_cycle22(BASE / "PRESMOKE_MANIFEST.json")
    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if tests.get("status") != "pass" or tests.get("passed") != 55:
        raise RuntimeError("Cycle23 tests did not pass exactly")
    selection = json.loads((RUN / "SMOKE_SELECTION.json").read_text())
    if (
        selection.get("start_mirror_seed") != 202_623_000_000
        or selection.get("selected_mirror_seed") != 202_623_000_260
        or selection.get("tested_seed_count") != 261
        or selection.get("team_2_lead") != "Terapagos"
    ):
        raise RuntimeError("Cycle23 label-blind selection changed")
    pair = RUN / "smoke-result.json.pairs.json"
    pair_payload = json.loads(pair.read_text())
    if len(pair_payload.get("pairs", [])) != 1:
        raise RuntimeError("Cycle23 requires one smoke pair")
    row = pair_payload["pairs"][0]
    if (
        row["team_1_sha256"] != selection["team_1_sha256"]
        or row["team_2_sha256"] != selection["team_2_sha256"]
        or not row["team_2_packed"].startswith("Terapagos||")
    ):
        raise RuntimeError("Cycle23 pair disagrees with frozen selection")
    current = tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
    prior_pairs = set()
    for prior in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if prior.resolve() == pair.resolve():
            continue
        try:
            value = json.loads(prior.read_text())
        except Exception:
            continue
        for previous in value.get("pairs", []):
            if "team_1_sha256" in previous and "team_2_sha256" in previous:
                prior_pairs.add(tuple(sorted((previous["team_1_sha256"], previous["team_2_sha256"]))))
    if current in prior_pairs:
        raise RuntimeError("Cycle23 smoke pair overlaps a prior artifact")
    pair_dir = RUN / "smoke-registrations"
    if pair_dir.exists() and any(pair_dir.iterdir()):
        raise RuntimeError("Cycle23 registration directory is not fresh")

    engine_root = BASE / "engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    if sha(extension) != "c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055":
        raise RuntimeError("Cycle23 inherited native binding changed")
    files = [
        RUN / "PROTOCOL.md", RUN / "PREFREEZE_TESTS.json", RUN / "SMOKE_SELECTION.json",
        RUN / "SHOWDOWN_RUNTIME_MANIFEST.json", pair, BASE / "PRESMOKE_MANIFEST.json",
        ROOT / "experimental/src/scripts/select_cycle23_form_smoke_pair.py",
        ROOT / "experimental/src/scripts/watch_cycle23_registrations.py",
        ROOT / "experimental/src/scripts/monitor_cycle23_first_decision_smoke.py",
        ROOT / "experimental/src/scripts/run_cycle23_first_decision_smoke.sh",
        ROOT / "experimental/src/scripts/freeze_cycle23_presmoke.py",
        ROOT / "experimental/src/scripts/verify_cycle23_presmoke.py",
        ROOT / "experimental/src/scripts/tests/test_cycle23_registration_attestation.py",
        ROOT / "experimental/src/scripts/tests/test_monitor_cycle23_first_decision_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle20_form_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/eval/run.py", ROOT / "srcs/metagross/showdown_runtime_server.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py", ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/prior_server.py", extension,
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
    ]
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "metagross-cycle23-presmoke-freeze/v1",
        "status": "frozen_before_first_decision_smoke",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "selection_sha256": sha(RUN / "SMOKE_SELECTION.json"),
        "pair_sha256": sha(pair),
        "cycle22_base_manifest_sha256": sha(BASE / "PRESMOKE_MANIFEST.json"),
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {"import_root": str(engine_root.resolve()), "native_path": str(extension.resolve()), "native_sha256": sha(extension)},
        "showdown": {"path": str(showdown.resolve()), "commit": commit, "dist_tree_sha256": tree_sha256(showdown / "dist")},
        "gate": {
            "registration_files_observed": 2, "registration_files_consumed": 2,
            "registration_reappearances": 0, "remaining_registration_files": 0,
            "private_rosters_match": True, "public_leads_match": True,
            "first_candidate_decisions": 1, "request_rqid_exact": True,
            "causal_root_protocol_hash_exact": True, "unique_exact_form_slot": True,
            "required_current_ability": "terashell", "required_base_ability": "terashell",
            "schedules": 2, "worlds_per_schedule": 8, "iterations_per_world": 8192,
            "terminate_after_first_public_execution": True, "semantic_operational_failures": 0,
        },
        "authorization_on_pass": {"pp_conditional_belief_mechanics_cycle": True, "h2h_games": 0, "training": False, "sealed93": False, "gpu_cloud_paid": False},
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "protocol_sha256": manifest["protocol_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
