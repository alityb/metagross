#!/usr/bin/env python3
"""Freeze Cycle32 before its fresh authenticated-identity smoke."""

from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
from experimental.src.scripts.cycle32_canonical_smoke import config_identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experimental/runs/search_native_v2_cycle32_authenticated_identity_20260815"
C29 = ROOT / "experimental/runs/search_native_v2_cycle29_canonical_projection_20260815"
C30 = ROOT / "experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815"
C31 = ROOT / "experimental/runs/search_native_v2_cycle31_candidate_attribution_20260815"
BASE = ROOT / "experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
PAIR_SHA = "1faeea684cca8ee83be1ba66b7e938d7ec91e967c7d5cd72c9b6eb273ad2023c"
ENGINE_SHA = "c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    output = RUN / "PRESMOKE_MANIFEST.json"
    if output.exists(): raise RuntimeError("Cycle32 smoke is already frozen")
    mechanics = json.loads((C29 / "mechanics-audit/REPORT.json").read_text())
    if mechanics.get("status") != "pass" or mechanics.get("counts", {}).get("passed") != 16: raise RuntimeError("Cycle29 mechanics changed")
    if json.loads((C31 / "SMOKE_FAILURE.json").read_text()).get("failure_class") != "candidate_external_username_attribution": raise RuntimeError("Cycle31 diagnosis changed")
    if json.loads((RUN / "PRESMOKE_TESTS.json").read_text()).get("passed") != 27: raise RuntimeError("Cycle32 tests failed")
    pair = RUN / "smoke-result.json.pairs.json"
    if sha(pair) != PAIR_SHA: raise RuntimeError("Cycle32 pair changed")
    payload = json.loads(pair.read_text()); row = payload["pairs"][0]
    current = tuple(sorted((row["team_1_sha256"], row["team_2_sha256"])))
    for path in (ROOT / "experimental/runs").rglob("*.pairs.json"):
        if path.resolve() == pair.resolve(): continue
        try: old_rows = json.loads(path.read_text()).get("pairs", [])
        except Exception: continue
        if any("team_1_sha256" in old and "team_2_sha256" in old and tuple(sorted((old["team_1_sha256"], old["team_2_sha256"]))) == current for old in old_rows): raise RuntimeError(f"Cycle32 pair overlap: {path}")
    canonical = RUN / "CANONICAL_SMOKE_ARGV.json"
    prepare = config_identity(canonical, "prepare"); live = config_identity(canonical, "live", PAIR_SHA)
    if prepare != live or prepare != payload["config_sha256"]: raise RuntimeError("Cycle32 config identity changed")
    reg = RUN / "smoke-registrations"
    if not reg.is_dir() or any(reg.iterdir()): raise RuntimeError("Cycle32 registration domain not fresh")
    engine_root = BASE / "engine-binding/unpacked"; extension = next((engine_root / "poke_engine").glob("poke_engine*.so"))
    if sha(extension) != ENGINE_SHA: raise RuntimeError("Cycle32 engine changed")
    files = [
        RUN/"PROTOCOL.md", RUN/"PRESMOKE_TESTS.json", canonical, pair,
        C30/"SHOWDOWN_RUNTIME_MANIFEST.json", C31/"SMOKE_FAILURE.json", C29/"mechanics-audit/REPORT.json",
        ROOT/"experimental/src/scripts/cycle32_canonical_smoke.py", ROOT/"experimental/src/scripts/watch_cycle32_registrations.py",
        ROOT/"experimental/src/scripts/watch_cycle31_registrations.py", ROOT/"experimental/src/scripts/monitor_cycle32_authenticated_boundary_smoke.py",
        ROOT/"experimental/src/scripts/monitor_cycle30_dynamic_boundary_smoke.py", ROOT/"experimental/src/scripts/run_cycle32_authenticated_boundary_smoke.sh",
        ROOT/"experimental/src/scripts/freeze_cycle32_presmoke.py", ROOT/"experimental/src/scripts/verify_cycle32_presmoke.py",
        ROOT/"experimental/src/scripts/monitor_cycle29_second_root_smoke.py", ROOT/"experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT/"experimental/src/scripts/monitor_cycle21_registered_form_smoke.py", ROOT/"experimental/src/scripts/cycle19_equal8192_live_decision.py",
        ROOT/"experimental/src/eval/run.py", ROOT/"srcs/metagross/showdown_runtime_server.py", ROOT/"srcs/metagross/causal_reveal_ledger.py",
        ROOT/"srcs/metagross/run_foul_play.py", ROOT/"srcs/metagross/prior_server.py", ROOT/"srcs/vendor/foul-play/fp/search/helpers.py",
        ROOT/"srcs/vendor/foul-play/fp/search/random_battles.py", ROOT/"srcs/metagross/tests/test_cycle30_dynamic_boundary.py",
        ROOT/"srcs/metagross/tests/test_cycle31_candidate_attribution.py", ROOT/"srcs/metagross/tests/test_cycle32_authenticated_identity.py",
        extension, ROOT/"srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt",
    ]
    if any(not path.is_file() for path in files): raise RuntimeError("Cycle32 frozen file missing")
    showdown = ROOT/"external/pokemon-showdown"; commit = subprocess.check_output(["git","-C",str(showdown),"rev-parse","HEAD"],text=True).strip()
    manifest = {
        "schema":"metagross-cycle32-authenticated-identity-presmoke-freeze/v1", "status":"frozen_before_fresh_authenticated_smoke",
        "protocol_sha256":sha(RUN/"PROTOCOL.md"), "cycle29_mechanics_report_sha256":sha(C29/"mechanics-audit/REPORT.json"),
        "cycle31_failure_sha256":sha(C31/"SMOKE_FAILURE.json"), "pair_sha256":PAIR_SHA, "config_sha256":prepare,
        "files":[{"path":str(path.resolve()),"sha256":sha(path)} for path in files],
        "engine":{"import_root":str(engine_root.resolve()),"native_path":str(extension.resolve()),"native_sha256":sha(extension)},
        "showdown":{"path":str(showdown.resolve()),"commit":commit,"dist_tree_sha256":tree_sha256(showdown/"dist")},
        "gate":{"fresh_pair":True,"candidate_namespace":"agent_a","internal_role_and_external_username_distinct":True,"public_config_registration_join":True,"first_eligible_boundary":"ordinary_and_intrinsic_opponent_move","max_decision_index":5,"max_battle_turn":6,"timeout_seconds":600,"candidate_cells":"2x8","candidate_iterations_per_world":8192,"semantic_operational_failures":0},
        "authorization_on_pass":{"separately_frozen_scored_h2h":True,"strength_claim":False,"training":False,"sealed93":False,"gpu_cloud_paid":False}
    }
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"manifest_sha256":sha(output),"pair_sha256":PAIR_SHA,"config_sha256":prepare},sort_keys=True))
if __name__ == "__main__": main()
