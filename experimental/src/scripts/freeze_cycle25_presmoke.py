#!/usr/bin/env python3
"""Freeze Cycle25 before its one attributed live decision."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
from experimental.src.scripts.cycle25_canonical_eval import config_identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/"experimental/runs/search_native_v2_cycle25_receipt_attribution_20260815"
BASE=ROOT/"experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->None:
    output=RUN/"PRESMOKE_MANIFEST.json"
    if output.exists(): raise RuntimeError("Cycle25 already frozen")
    tests=json.loads((RUN/"PREFREEZE_TESTS.json").read_text())
    if tests.get("status")!="pass" or tests.get("passed")!=57: raise RuntimeError("Cycle25 tests did not pass")
    selection=json.loads((RUN/"SMOKE_SELECTION.json").read_text())
    if (selection.get("start_mirror_seed"),selection.get("selected_mirror_seed"),selection.get("tested_seed_count"),selection.get("team_2_lead")) != (202_625_000_000,202_625_000_549,550,"Terapagos"): raise RuntimeError("Cycle25 selection changed")
    pair=RUN/"smoke-result.json.pairs.json"; pair_sha=sha(pair); payload=json.loads(pair.read_text()); row=payload["pairs"][0]
    if len(payload["pairs"])!=1 or row["team_1_sha256"]!=selection["team_1_sha256"] or row["team_2_sha256"]!=selection["team_2_sha256"] or not row["team_2_packed"].startswith("Terapagos||"): raise RuntimeError("Cycle25 pair differs")
    canonical=RUN/"CANONICAL_EVAL_ARGV.json"; prepare=config_identity(canonical,"prepare"); live=config_identity(canonical,"live",pair_sha)
    if prepare!=live or prepare!=payload["config_sha256"]: raise RuntimeError("Cycle25 config identity differs")
    current=tuple(sorted((row["team_1_sha256"],row["team_2_sha256"]))); prior=set()
    for path in (ROOT/"experimental/runs").rglob("*.pairs.json"):
        if path.resolve()==pair.resolve(): continue
        try: value=json.loads(path.read_text())
        except Exception: continue
        for old in value.get("pairs",[]):
            if "team_1_sha256" in old and "team_2_sha256" in old: prior.add(tuple(sorted((old["team_1_sha256"],old["team_2_sha256"]))))
    if current in prior: raise RuntimeError("Cycle25 pair overlaps prior")
    directory=RUN/"smoke-registrations"
    if directory.exists() and any(directory.iterdir()): raise RuntimeError("Cycle25 registrations not fresh")
    engine_root=BASE/"engine-binding/unpacked"; extension=next((engine_root/"poke_engine").glob("poke_engine*.so"))
    if sha(extension)!="c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055": raise RuntimeError("Cycle25 engine changed")
    files=[RUN/"PROTOCOL.md",RUN/"PREFREEZE_TESTS.json",RUN/"SMOKE_SELECTION.json",RUN/"CANONICAL_EVAL_ARGV.json",RUN/"SHOWDOWN_RUNTIME_MANIFEST.json",pair,
      ROOT/"experimental/src/scripts/select_cycle25_form_smoke_pair.py",ROOT/"experimental/src/scripts/cycle25_canonical_eval.py",ROOT/"experimental/src/scripts/watch_cycle25_registrations.py",ROOT/"experimental/src/scripts/monitor_cycle25_attributed_smoke.py",ROOT/"experimental/src/scripts/run_cycle25_attributed_smoke.sh",ROOT/"experimental/src/scripts/freeze_cycle25_presmoke.py",ROOT/"experimental/src/scripts/verify_cycle25_presmoke.py",ROOT/"srcs/metagross/tests/test_cycle25_receipt_attribution.py",
      ROOT/"experimental/src/scripts/monitor_cycle23_first_decision_smoke.py",ROOT/"experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",ROOT/"experimental/src/scripts/monitor_cycle20_form_smoke.py",ROOT/"experimental/src/scripts/monitor_cycle19_operational_smoke.py",ROOT/"experimental/src/scripts/cycle19_equal8192_live_decision.py",ROOT/"experimental/src/eval/run.py",ROOT/"srcs/metagross/showdown_runtime_server.py",ROOT/"srcs/metagross/causal_reveal_ledger.py",ROOT/"srcs/metagross/run_foul_play.py",ROOT/"srcs/metagross/prior_server.py",extension,ROOT/"srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"]
    showdown=ROOT/"external/pokemon-showdown"; commit=subprocess.check_output(["git","-C",str(showdown),"rev-parse","HEAD"],text=True).strip()
    manifest={"schema":"metagross-cycle25-presmoke-freeze/v1","status":"frozen_before_attributed_first_decision_smoke","protocol_sha256":sha(RUN/"PROTOCOL.md"),"pair_sha256":pair_sha,"canonical_argv_sha256":sha(canonical),"config_sha256":prepare,"files":[{"path":str(p.resolve()),"sha256":sha(p)} for p in files],"engine":{"import_root":str(engine_root.resolve()),"native_path":str(extension.resolve()),"native_sha256":sha(extension)},"showdown":{"path":str(showdown.resolve()),"commit":commit,"dist_tree_sha256":tree_sha256(showdown/"dist")},"gate":{"receipt_schema":"v2","production_declared_count":[16,32],"candidate_cells":"2x8","same_root_rqid_decision":True,"post_execution_receipts":0,"boundary_latch_after_public_action":True,"candidate_iterations_per_world":8192,"semantic_operational_failures":0},"authorization_on_pass":{"pp_conditional_belief_mechanics_cycle":True,"h2h_games":0,"training":False,"sealed93":False,"gpu_cloud_paid":False}}
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"manifest_sha256":sha(output),"protocol_sha256":manifest["protocol_sha256"],"config_sha256":prepare},sort_keys=True))
if __name__=="__main__": main()
