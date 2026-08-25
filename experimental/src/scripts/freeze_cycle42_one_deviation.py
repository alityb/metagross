#!/usr/bin/env python3
"""Freeze Cycle42 protocol, identities, controller, and mechanics before outcomes."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from srcs.metagross.terminal_mcts_one_deviation import assignment_manifest
from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/"experimental/runs/search_native_v2_cycle42_one_deviation_20260816";ENGINE_ROOT=ROOT/"experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815/engine-binding/unpacked";ENGINE_SHA="c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"
def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"))
def main():
 global RUN
 parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,default=RUN);parser.add_argument("--seed",default="202642160842");args=parser.parse_args();RUN=args.run.resolve()
 out=RUN/"PREMEASUREMENT_MANIFEST.json"
 if out.exists():raise FileExistsError(out)
 junit=ET.parse(RUN/"prefreeze-junit.xml").getroot(); failures=sum(int(x.attrib.get("failures",0))+int(x.attrib.get("errors",0)) for x in junit.iter("testsuite")); tests=max([int(x.attrib.get("tests",0)) for x in junit.iter("testsuite")],default=0)
 if failures or tests<15:raise RuntimeError("Cycle42 prefreeze tests failed/incomplete")
 expected=assignment_manifest(args.seed);stored=json.loads((RUN/"ASSIGNMENT_MANIFEST.json").read_text())
 if canon(expected)!=canon(stored):raise RuntimeError("assignment manifest differs")
 reg=json.loads((RUN/"PRIOR_IDENTITY_REGISTRY.json").read_text()); pairs=json.loads((RUN/"h2h-result.json.pairs.json").read_text());rows=pairs["pairs"]
 cp={canon(sorted((x["team_1_sha256"],x["team_2_sha256"]))) for x in rows};ct={y for x in rows for y in(x["team_1_sha256"],x["team_2_sha256"])};ci={x["pair_id"] for x in rows};cs={canon(x["battle_seed"]) for x in rows}
 if (len(rows),len(cp),len(ct),len(ci),len(cs))!=(10,10,20,10,10) or cp&set(reg["unordered_team_pairs"]) or ct&set(reg["individual_team_sha256"]) or ci&set(reg["pair_ids"]) or cs&set(reg["battle_seeds"]):raise RuntimeError("pair freshness failed")
 pairpath=RUN/"h2h-result.json.pairs.json";pairsha=sha(pairpath);canonical=RUN/"CANONICAL_H2H_ARGV.json"
 if identity(canonical,"prepare")!=identity(canonical,"live",pairsha) or identity(canonical,"prepare")!=pairs["config_sha256"]:raise RuntimeError("canonical identity failed")
 if list((RUN/"h2h-registrations").iterdir()):raise RuntimeError("registration domain not empty")
 argv=json.loads(canonical.read_text())["argv"]
 def av(flag):return argv[argv.index(flag)+1]
 for flag,key in (("--mirror-seed","mirror_seeds"),("--production-run-seed","production_run_seeds"),("--run-id","run_ids"),("--username-prefix","username_prefixes")):
  if av(flag) in set(reg[key]):raise RuntimeError(f"fresh identity overlap: {flag}")
 if any(name.startswith(av("--username-prefix")) for name in reg["usernames"]):raise RuntimeError("username namespace overlap")
 reports=[ROOT/"experimental/runs/search_native_v2_cycle34_causal_disable_repair_20260815/mechanics-audit/REPORT.json",ROOT/"experimental/runs/search_native_v2_cycle38c_temporal_switch_20260816/mechanics-audit/REPORT.json",ROOT/"experimental/runs/search_native_v2_cycle39_target_aware_pp_20260816/mechanics-audit/REPORT.json"]
 for p in reports:
  v=json.loads(p.read_text())
  if v.get("status")!="pass" or not all(v.get("gates",{}).values()):raise RuntimeError(f"admitted mechanics changed: {p}")
 ext=next((ENGINE_ROOT/"poke_engine").glob("poke_engine*.so"))
 if sha(ext)!=ENGINE_SHA:raise RuntimeError("engine changed")
 files=[RUN/"PROTOCOL.md",RUN/"CANONICAL_H2H_ARGV.json",RUN/"ASSIGNMENT_MANIFEST.json",RUN/"PRIOR_IDENTITY_REGISTRY.json",pairpath,RUN/"prefreeze-junit.xml",*reports,ROOT/"experimental/src/scripts/cycle19_equal8192_live_decision.py",ROOT/"experimental/src/scripts/cycle33_canonical_h2h.py",ROOT/"experimental/src/scripts/run_cycle42_one_deviation.sh",ROOT/"experimental/src/scripts/summarize_cycle42_one_deviation.py",ROOT/"experimental/src/scripts/watch_cycle42_registrations.py",ROOT/"experimental/src/scripts/cycle43_outcome_blind_watcher.py",ROOT/"experimental/src/scripts/freeze_cycle42_one_deviation.py",ROOT/"experimental/src/scripts/build_cycle42_identity_registry.py",ROOT/"experimental/src/scripts/tests/test_cycle42_one_deviation.py",ROOT/"experimental/src/scripts/tests/test_cycle42_freeze.py",ROOT/"experimental/src/scripts/tests/test_cycle43_outcome_blind_watcher.py",ROOT/"srcs/metagross/terminal_mcts_one_deviation.py",ROOT/"srcs/metagross/run_foul_play.py",ROOT/"srcs/metagross/causal_reveal_ledger.py",ext,ROOT/"srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"]
 showdown=ROOT/"external/pokemon-showdown"; commit=subprocess.check_output(["git","-C",str(showdown),"rev-parse","HEAD"],text=True).strip()
 payload={"schema":"metagross-cycle33-h2h-premeasurement/v1","cycle":43 if "cycle43" in RUN.name else 42,"status":"frozen_before_outcomes","protocol_sha256":sha(RUN/"PROTOCOL.md"),"pair_sha256":pairsha,"config_sha256":identity(canonical,"prepare"),"assignment_schedule_sha256":expected["schedule_sha256"],"randomization_seed":args.seed,"files":[{"path":str(p.resolve()),"sha256":sha(p)} for p in files],"engine":{"import_root":str(ENGINE_ROOT.resolve()),"native_sha256":sha(ext)},"showdown":{"path":str(showdown.resolve()),"commit":commit,"dist_tree_sha256":tree_sha256(showdown/"dist")},"gate":{"games":20,"pairs":10,"minimum_eligible":14,"minimum_per_arm":6,"minimum_effect":.25,"nonnegative_each_role":True,"maximum_one_sided_fisher_p":.20,"all_failures":0},"authorization":{"powered_replication_only_on_pass":True,"training":False,"sealed93":False,"gpu_cloud_paid":False}}
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"manifest_sha256":sha(out),"protocol_sha256":payload["protocol_sha256"],"pair_sha256":pairsha,"assignment_sha256":expected["schedule_sha256"],"tests":tests},sort_keys=True))
if __name__=="__main__":main()
