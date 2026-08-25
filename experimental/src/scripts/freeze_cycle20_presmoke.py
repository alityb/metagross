#!/usr/bin/env python3
"""Freeze Cycle20 after mechanics admission and before live smoke."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle20_ability_lineage_20260815"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PRESMOKE_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle20 presmoke is already frozen")
    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if tests.get("status") != "pass" or tests.get("passed") != 38:
        raise RuntimeError("Cycle20 prefreeze tests did not pass exactly")
    selection = json.loads((RUN / "SMOKE_SELECTION.json").read_text())
    if (
        selection.get("start_mirror_seed") != 202_620_000_000
        or selection.get("selected_mirror_seed") != 202_620_000_142
        or selection.get("tested_seed_count") != 143
        or selection.get("team_2_lead") != "Terapagos"
    ):
        raise RuntimeError("Cycle20 label-blind smoke selection changed")
    pair = RUN / "smoke-result.json.pairs.json"
    pair_payload = json.loads(pair.read_text())
    if len(pair_payload.get("pairs", [])) != 1:
        raise RuntimeError("Cycle20 smoke requires exactly one pair")
    row = pair_payload["pairs"][0]
    if (
        pair_payload.get("showdown_commit") != "4880d3693580bd33652797cf31179c6fcdf87e50"
        or row.get("team_1_sha256") != selection["team_1_sha256"]
        or row.get("team_2_sha256") != selection["team_2_sha256"]
        or not row.get("team_2_packed", "").startswith("Terapagos||")
    ):
        raise RuntimeError("Cycle20 smoke pair disagrees with frozen selection")
    current_pair = tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
    prior_pairs = set()
    for prior in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if prior.resolve() == pair.resolve():
            continue
        try:
            prior_payload = json.loads(prior.read_text())
        except Exception:
            continue
        for prior_row in prior_payload.get("pairs", []):
            if "team_1_sha256" in prior_row and "team_2_sha256" in prior_row:
                prior_pairs.add(tuple(sorted((prior_row["team_1_sha256"], prior_row["team_2_sha256"]))))
    if current_pair in prior_pairs:
        raise RuntimeError("Cycle20 smoke team pair overlaps an earlier artifact")
    engine_root = ROOT / (
        "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/"
        "engine-binding/unpacked"
    )
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    showdown = ROOT / "external/pokemon-showdown"
    transcript = ROOT / (
        "experimental/runs/search_native_v2_cycle19_operational_repair_20260815/"
        "h2h-logs/c19h2hx002cb97.protocol.jsonl"
    )
    files = [
        RUN / "PROTOCOL.md", RUN / "PREFREEZE_TESTS.json",
        RUN / "SMOKE_SELECTION.json", pair,
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/export_showdown_form_ability_contract.cjs",
        ROOT / "srcs/metagross/export_showdown_public_form_contract.cjs",
        ROOT / "srcs/metagross/tests/test_causal_reveal_ledger.py",
        ROOT / "srcs/metagross/tests/test_cycle20_ability_lineage.py",
        transcript,
        ROOT / "experimental/src/scripts/select_cycle20_form_smoke_pair.py",
        ROOT / "experimental/src/scripts/monitor_cycle20_form_smoke.py",
        ROOT / "experimental/src/scripts/run_cycle20_form_smoke.sh",
        ROOT / "experimental/src/scripts/freeze_cycle20_presmoke.py",
        ROOT / "experimental/src/scripts/verify_cycle20_presmoke.py",
        ROOT / "experimental/src/scripts/tests/test_monitor_cycle20_form_smoke.py",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/tests/test_cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/tests/test_monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/eval/run.py",
        ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
        ROOT / "experimental/src/scripts/start_showdown.sh",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
        extension,
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema": "metagross-cycle20-presmoke-freeze/v1",
        "status": "frozen_before_live_form_transition_smoke",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "selection_sha256": sha(RUN / "SMOKE_SELECTION.json"),
        "smoke_pair_sha256": sha(pair),
        "cycle19_failure_transcript_sha256": sha(transcript),
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {
            "import_root": str(engine_root.resolve()),
            "native_path": str(extension.resolve()),
            "native_sha256": sha(extension),
            "general_environment_mutated": False,
        },
        "showdown": {
            "path": str(showdown.resolve()), "commit": commit,
            "dist_tree_sha256": tree_sha256(showdown / "dist"),
        },
        "smoke_gate": {
            "required_exact_form": "terapagosterastal",
            "required_ordered_abilities": ["terashift", "terashell"],
            "candidate_decisions": 1, "schedules": 2,
            "worlds_per_schedule": 8, "iterations_per_world": 8192,
            "considered_fraction": 0.75,
            "require_public_action_confirmation": True,
            "fallback_timeout_semantic_failures": 0,
        },
        "authorization": {
            "operational_smoke": True, "h2h_games": 0, "training": False,
            "sealed93": False, "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "protocol_sha256": manifest["protocol_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
