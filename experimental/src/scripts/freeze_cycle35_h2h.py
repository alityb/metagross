#!/usr/bin/env python3
"""Freeze Cycle35 before any scored outcomes."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle35_fresh_h2h_20260815"
C32 = ROOT / "experimental/runs/search_native_v2_cycle32_authenticated_identity_20260815"
C34 = ROOT / "experimental/runs/search_native_v2_cycle34_causal_disable_repair_20260815"
C30 = ROOT / "experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815"
BASE = ROOT / "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
ENGINE_SHA = "c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    output = RUN / "H2H_PREMEASUREMENT_MANIFEST.json"
    if output.exists(): raise RuntimeError("Cycle35 already frozen")
    smoke = C32 / "SMOKE_RESULT.json"
    if sha(smoke) != "ed8b8cf3ab829e28166678d2adfb56d1af021b10135747a2d8a7d4bbaed4e426" or json.loads(smoke.read_text()).get("status") != "pass":
        raise RuntimeError("Cycle32 smoke changed")
    repair = json.loads((C34 / "mechanics-audit/REPORT.json").read_text())
    if repair.get("status") != "pass" or not all(repair.get("gates", {}).values()):
        raise RuntimeError("Cycle34 repair is not admitted")
    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    if tests.get("passed") != 15 or tests.get("failed") != 0:
        raise RuntimeError("Cycle35 tests failed")
    pair_path = RUN / "h2h-result.json.pairs.json"
    pair_sha = sha(pair_path)
    pair_payload = json.loads(pair_path.read_text())
    pairs = pair_payload.get("pairs", [])
    current = {tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pairs}
    if len(pairs) != 10 or len(current) != 10:
        raise RuntimeError("Cycle35 requires ten unique pairs")
    prior = set()
    for path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if path.resolve() == pair_path.resolve(): continue
        try: old = json.loads(path.read_text()).get("pairs", [])
        except Exception: continue
        prior |= {tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
                  for row in old if "team_1_sha256" in row and "team_2_sha256" in row}
    if current & prior: raise RuntimeError("Cycle35 pair overlaps prior H2H/smoke")
    canonical = RUN / "CANONICAL_H2H_ARGV.json"
    prepared, live = identity(canonical, "prepare"), identity(canonical, "live", pair_sha)
    if prepared != live or prepared != pair_payload["config_sha256"]:
        raise RuntimeError("Cycle35 config identity mismatch")
    registration = RUN / "h2h-registrations"
    if not registration.is_dir() or any(registration.iterdir()):
        raise RuntimeError("Cycle35 registration domain not fresh")
    engine_root = BASE / "engine-binding/unpacked"
    extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    if sha(extension) != ENGINE_SHA: raise RuntimeError("Cycle35 engine changed")
    files = [
        RUN / "PROTOCOL.md", RUN / "PREFREEZE_TESTS.json", canonical, pair_path,
        smoke, C34 / "PROTOCOL.md", C34 / "PREMEASUREMENT_MANIFEST.json",
        C34 / "mechanics-audit/REPORT.json", C34 / "mechanics-audit/root-results.jsonl",
        C30 / "SHOWDOWN_RUNTIME_MANIFEST.json",
        ROOT / "experimental/src/scripts/cycle33_canonical_h2h.py",
        ROOT / "experimental/src/scripts/watch_cycle35_registrations.py",
        ROOT / "experimental/src/scripts/summarize_cycle35_h2h.py",
        ROOT / "experimental/src/scripts/run_cycle35_h2h.sh",
        ROOT / "experimental/src/scripts/freeze_cycle35_h2h.py",
        ROOT / "experimental/src/scripts/verify_cycle33_h2h_freeze.py",
        ROOT / "experimental/src/scripts/tests/test_cycle33_h2h.py",
        ROOT / "experimental/src/scripts/tests/test_cycle35_h2h.py",
        ROOT / "experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",
        ROOT / "experimental/src/scripts/summarize_cycle19_h2h.py",
        ROOT / "experimental/src/scripts/summarize_cycle33_h2h.py",
        ROOT / "experimental/src/eval/run.py",
        ROOT / "srcs/metagross/showdown_runtime_server.py",
        ROOT / "srcs/metagross/causal_reveal_ledger.py",
        ROOT / "srcs/metagross/run_foul_play.py", ROOT / "srcs/metagross/prior_server.py",
        ROOT / "srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT / "srcs/vendor/foul-play/fp/search/random_battles.py",
        extension, ROOT / "srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
    ]
    if any(not path.is_file() for path in files): raise RuntimeError("Cycle35 frozen file missing")
    showdown = ROOT / "external/pokemon-showdown"
    commit = subprocess.check_output(["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "metagross-cycle33-h2h-premeasurement/v1",
        "cycle": 35, "status": "frozen_before_scored_outcomes",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "cycle32_smoke_sha256": sha(smoke),
        "cycle34_report_sha256": sha(C34 / "mechanics-audit/REPORT.json"),
        "pair_sha256": pair_sha, "config_sha256": prepared,
        "team_pairs_disjoint_from_all_prior_h2h_smokes": True,
        "retired_cycle33_data_reused": False,
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "engine": {"import_root": str(engine_root.resolve()), "native_sha256": sha(extension)},
        "showdown": {"path": str(showdown.resolve()), "commit": commit,
                     "dist_tree_sha256": tree_sha256(showdown / "dist")},
        "gate": {"games": 20, "mirrored_pairs": 10, "candidate_wins_to_continue": 13,
                 "interim_outcome_looks": 0, "all_failures": 0},
        "authorization": {"stage1_games": 20, "continuation": False, "training": False,
                          "sealed93": False, "gpu_cloud_paid": False}
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(output), "protocol_sha256": manifest["protocol_sha256"],
                      "pair_sha256": pair_sha}, sort_keys=True))

if __name__ == "__main__": main()
