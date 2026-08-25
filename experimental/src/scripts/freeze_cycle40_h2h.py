#!/usr/bin/env python3
"""Freeze Cycle40's integrated prospective H2H before any scored outcome."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"
C32 = ROOT / "experimental/runs/search_native_v2_cycle32_authenticated_identity_20260815"
C34 = ROOT / "experimental/runs/search_native_v2_cycle34_causal_disable_repair_20260815"
C38 = ROOT / "experimental/runs/search_native_v2_cycle38c_temporal_switch_20260816"
C39 = ROOT / "experimental/runs/search_native_v2_cycle39_target_aware_pp_20260816"
C30 = ROOT / "experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815"
BASE = ROOT / "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
ENGINE_SHA = "c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def argv_value(payload: dict, flag: str) -> str:
    argv = payload["argv"]
    index = argv.index(flag)
    return argv[index + 1]


def main() -> None:
    output = RUN / "H2H_PREMEASUREMENT_MANIFEST.json"
    if output.exists():
        raise RuntimeError("Cycle40 already frozen")

    smoke = C32 / "SMOKE_RESULT.json"
    if sha(smoke) != "ed8b8cf3ab829e28166678d2adfb56d1af021b10135747a2d8a7d4bbaed4e426":
        raise RuntimeError("Cycle32 smoke changed")
    if json.loads(smoke.read_text()).get("status") != "pass":
        raise RuntimeError("Cycle32 smoke is not admitted")
    reports = {
        "cycle34": C34 / "mechanics-audit/REPORT.json",
        "cycle38c": C38 / "mechanics-audit/REPORT.json",
        "cycle39": C39 / "mechanics-audit/REPORT.json",
    }
    for name, path in reports.items():
        payload = json.loads(path.read_text())
        if payload.get("status") != "pass" or not all(payload.get("gates", {}).values()):
            raise RuntimeError(f"{name} mechanics is not admitted")

    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if tests.get("passed") != 73 or tests.get("failed") != 0:
        raise RuntimeError("Cycle40 prefreeze tests failed")
    junit = RUN / "prefreeze-junit.xml"
    if sha(junit) != tests.get("junit_sha256"):
        raise RuntimeError("Cycle40 JUnit receipt changed")

    registry_path = RUN / "PRIOR_IDENTITY_REGISTRY_V2.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("status") != "frozen_before_cycle40_pair_generation":
        raise RuntimeError("Cycle40 prior identity registry invalid")
    if sha(registry_path) != tests.get("prior_registry_sha256"):
        raise RuntimeError("Cycle40 prior identity registry changed")

    pair_path = RUN / "h2h-result.json.pairs.json"
    if sha(pair_path) != tests.get("pair_manifest_sha256"):
        raise RuntimeError("Cycle40 pair manifest changed after tests")
    pair_payload = json.loads(pair_path.read_text())
    pairs = pair_payload.get("pairs", [])
    current_pairs = {canon(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pairs}
    current_teams = {value for row in pairs for value in (row["team_1_sha256"], row["team_2_sha256"])}
    current_ids = {row["pair_id"] for row in pairs}
    current_seeds = {canon(row["battle_seed"]) for row in pairs}
    if not (len(pairs) == len(current_pairs) == len(current_ids) == len(current_seeds) == 10):
        raise RuntimeError("Cycle40 requires ten unique pairs/ids/seeds")
    if len(current_teams) != 20:
        raise RuntimeError("Cycle40 requires twenty unique individual teams")
    if current_pairs & set(registry["unordered_team_pairs"]):
        raise RuntimeError("Cycle40 pair overlaps prior artifact")
    if current_teams & set(registry["individual_team_sha256"]):
        raise RuntimeError("Cycle40 individual team overlaps prior artifact")
    if current_ids & set(registry["pair_ids"]) or current_seeds & set(registry["battle_seeds"]):
        raise RuntimeError("Cycle40 pair id or battle seed overlaps prior artifact")

    canonical = RUN / "CANONICAL_H2H_ARGV.json"
    canonical_payload = json.loads(canonical.read_text())
    for flag, key in (
        ("--mirror-seed", "mirror_seeds"),
        ("--production-run-seed", "production_run_seeds"),
        ("--run-id", "run_ids"),
        ("--username-prefix", "username_prefixes"),
    ):
        if argv_value(canonical_payload, flag) in set(registry[key]):
            raise RuntimeError(f"Cycle40 {flag} overlaps prior artifact")
    prefix = argv_value(canonical_payload, "--username-prefix")
    if any(value.startswith(prefix) for value in registry["usernames"]):
        raise RuntimeError("Cycle40 username namespace overlaps prior artifact")
    pair_sha = sha(pair_path)
    prepared = identity(canonical, "prepare")
    live = identity(canonical, "live", pair_sha)
    if prepared != live or prepared != pair_payload["config_sha256"]:
        raise RuntimeError("Cycle40 canonical prepare/live identity mismatch")
    registration = RUN / "h2h-registrations"
    if not registration.is_dir() or any(registration.iterdir()):
        raise RuntimeError("Cycle40 registration domain is not fresh")

    engine_root = BASE / "engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    if sha(extension) != ENGINE_SHA:
        raise RuntimeError("Cycle40 engine binding changed")
    files = [
        RUN / "PROTOCOL.md", RUN / "PREFREEZE_TESTS.json", junit,
        canonical, pair_path, registry_path,
        smoke,
        reports["cycle34"], reports["cycle38c"], reports["cycle39"],
        C34 / "PREMEASUREMENT_MANIFEST.json",
        C38 / "PREMEASUREMENT_MANIFEST.json",
        C39 / "PREMEASUREMENT_MANIFEST.json",
        C30 / "SHOWDOWN_RUNTIME_MANIFEST.json",
        ROOT / "experimental/src/scripts/build_cycle40_prior_identity_registry.py",
        ROOT / "experimental/src/scripts/freeze_cycle40_h2h.py",
        ROOT / "experimental/src/scripts/run_cycle40_h2h.sh",
        ROOT / "experimental/src/scripts/watch_cycle40_registrations.py",
        ROOT / "experimental/src/scripts/summarize_cycle40_h2h.py",
        ROOT / "experimental/src/scripts/tests/test_cycle40_h2h.py",
        ROOT / "experimental/src/scripts/cycle33_canonical_h2h.py",
        ROOT / "experimental/src/scripts/verify_cycle33_h2h_freeze.py",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",
        ROOT / "experimental/src/scripts/summarize_cycle19_h2h.py",
        ROOT / "experimental/src/scripts/summarize_cycle33_h2h.py",
        ROOT / "experimental/src/eval/run.py",
        ROOT / "srcs/metagross/showdown_runtime_server.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/vendor/foul-play/fp/battle.py",
        ROOT / "srcs/vendor/foul-play/fp/battle_modifier.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        ROOT / "srcs/vendor/foul-play/fp/search/main.py",
        extension,
        ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
    ]
    if any(not path.is_file() for path in files):
        raise RuntimeError("Cycle40 frozen file missing")
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema": "metagross-cycle33-h2h-premeasurement/v1",
        "cycle": 40,
        "status": "frozen_before_scored_outcomes",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "pair_sha256": pair_sha,
        "config_sha256": prepared,
        "prior_identity_registry_sha256": sha(registry_path),
        "freshness": {
            "prior_json_sources_scanned": registry["counts"]["pair_sources"],
            "prior_unordered_pairs": registry["counts"]["unordered_team_pairs"],
            "prior_individual_teams": registry["counts"]["individual_teams"],
            "prior_battle_seeds": registry["counts"]["battle_seeds"],
            "all_current_identity_classes_disjoint": True,
        },
        "admitted_chain": {name: sha(path) for name, path in reports.items()},
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {"import_root": str(engine_root.resolve()), "native_sha256": sha(extension)},
        "showdown": {
            "path": str(showdown.resolve()),
            "commit": commit,
            "dist_tree_sha256": tree_sha256(showdown / "dist"),
        },
        "gate": {
            "games": 20,
            "mirrored_pairs": 10,
            "candidate_wins_to_continue": 13,
            "interim_outcome_looks": 0,
            "all_failures": 0,
        },
        "authorization": {
            "stage1_games": 20,
            "continuation": False,
            "training": False,
            "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_sha256": sha(output),
        "protocol_sha256": manifest["protocol_sha256"],
        "pair_sha256": pair_sha,
        "prior_identity_registry_sha256": manifest["prior_identity_registry_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
