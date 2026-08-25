#!/usr/bin/env python3
"""Freeze Cycle33 before any scored outcomes."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"experimental/runs/search_native_v2_cycle33_prospective_h2h_20260815"; C32=ROOT/"experimental/runs/search_native_v2_cycle32_authenticated_identity_20260815"; C30=ROOT/"experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815"; BASE=ROOT/"experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815"
PAIR_SHA="2e3b59a2c2124e5e6f14b2fe9d1a0b277d5e23bf082a3edaa44ab5738600a9db"; ENGINE_SHA="c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=RUN/"H2H_PREMEASUREMENT_MANIFEST.json"
 if out.exists():raise RuntimeError("Cycle33 already frozen")
 smoke=RUN.parent/"search_native_v2_cycle32_authenticated_identity_20260815"/"SMOKE_RESULT.json"
 if sha(smoke)!="ed8b8cf3ab829e28166678d2adfb56d1af021b10135747a2d8a7d4bbaed4e426" or json.loads(smoke.read_text()).get("status")!="pass":raise RuntimeError("Cycle32 smoke changed")
 tests=json.loads((RUN/"PREFREEZE_TESTS.json").read_text())
 if tests.get("status")!="pass" or tests.get("passed")!=30:raise RuntimeError("Cycle33 tests failed")
 pair=RUN/"h2h-result.json.pairs.json"
 if sha(pair)!=PAIR_SHA:raise RuntimeError("Cycle33 pair changed")
 payload=json.loads(pair.read_text()); pairs=payload.get("pairs",[]); current={tuple(sorted((r["team_1_sha256"],r["team_2_sha256"]))) for r in pairs}
 if len(pairs)!=10 or len(current)!=10:raise RuntimeError("Cycle33 requires ten unique pairs")
 prior=set()
 for path in (ROOT/"experimental/runs").rglob("*.pairs.json"):
  if path.resolve()==pair.resolve():continue
  try:old=json.loads(path.read_text()).get("pairs",[])
  except Exception:continue
  prior|={tuple(sorted((r["team_1_sha256"],r["team_2_sha256"]))) for r in old if "team_1_sha256" in r and "team_2_sha256" in r}
 if current&prior:raise RuntimeError("Cycle33 pair overlap")
 canonical=RUN/"CANONICAL_H2H_ARGV.json"; prepared=identity(canonical,"prepare"); live=identity(canonical,"live",PAIR_SHA)
 if prepared!=live or prepared!=payload["config_sha256"]:raise RuntimeError("Cycle33 config identity mismatch")
 reg=RUN/"h2h-registrations"
 if not reg.is_dir() or any(reg.iterdir()):raise RuntimeError("Cycle33 registration domain not fresh")
 engine_root=BASE/"engine-binding/unpacked"; extension=next((engine_root/"poke_engine").glob("poke_engine*.so"))
 if sha(extension)!=ENGINE_SHA:raise RuntimeError("Cycle33 engine changed")
 files=[RUN/"PROTOCOL.md",RUN/"PREFREEZE_TESTS.json",canonical,pair,smoke,C30/"SHOWDOWN_RUNTIME_MANIFEST.json",ROOT/"experimental/src/scripts/cycle33_canonical_h2h.py",ROOT/"experimental/src/scripts/watch_cycle33_registrations.py",ROOT/"experimental/src/scripts/summarize_cycle33_h2h.py",ROOT/"experimental/src/scripts/run_cycle33_h2h.sh",ROOT/"experimental/src/scripts/freeze_cycle33_h2h.py",ROOT/"experimental/src/scripts/verify_cycle33_h2h_freeze.py",ROOT/"experimental/src/scripts/tests/test_cycle33_h2h.py",ROOT/"experimental/src/scripts/cycle19_equal8192_live_decision.py",ROOT/"experimental/src/scripts/monitor_cycle19_operational_smoke.py",ROOT/"experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",ROOT/"experimental/src/scripts/summarize_cycle19_h2h.py",ROOT/"experimental/src/eval/run.py",ROOT/"srcs/metagross/showdown_runtime_server.py",ROOT/"srcs/metagross/causal_reveal_ledger.py",ROOT/"srcs/metagross/run_foul_play.py",ROOT/"srcs/metagross/prior_server.py",ROOT/"srcs/vendor/foul-play/fp/search/helpers.py",ROOT/"srcs/vendor/foul-play/fp/search/random_battles.py",extension,ROOT/"srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"]
 if any(not p.is_file() for p in files):raise RuntimeError("Cycle33 frozen file missing")
 showdown=ROOT/"external/pokemon-showdown"; commit=subprocess.check_output(["git","-C",str(showdown),"rev-parse","HEAD"],text=True).strip()
 manifest={"schema":"metagross-cycle33-h2h-premeasurement/v1","status":"frozen_before_scored_outcomes","protocol_sha256":sha(RUN/"PROTOCOL.md"),"cycle32_smoke_sha256":sha(smoke),"pair_sha256":PAIR_SHA,"config_sha256":prepared,"team_pairs_disjoint_from_prior":True,"files":[{"path":str(p.resolve()),"sha256":sha(p)} for p in files],"engine":{"import_root":str(engine_root.resolve()),"native_sha256":sha(extension)},"showdown":{"path":str(showdown.resolve()),"commit":commit,"dist_tree_sha256":tree_sha256(showdown/"dist")},"gate":{"games":20,"mirrored_pairs":10,"candidate_wins_to_continue":13,"interim_outcome_looks":0,"all_failures":0},"authorization":{"stage1_games":20,"continuation":False,"training":False,"sealed93":False,"gpu_cloud_paid":False}}
 out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"manifest_sha256":sha(out),"protocol_sha256":manifest["protocol_sha256"],"pair_sha256":PAIR_SHA},sort_keys=True))
if __name__=="__main__":main()
