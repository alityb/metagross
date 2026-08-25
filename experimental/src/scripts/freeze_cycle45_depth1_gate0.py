#!/usr/bin/env python3
"""Freeze Cycle45 selection/code/source provenance before teacher targets."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/"experimental/runs/search_native_v2_cycle45_depth1_gate0_20260816";SOURCE=ROOT/"experimental/runs/search_native_v2_cycle13_train_rehydration_20260815/selection-200.jsonl";ENGINE=ROOT/"experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 global RUN
 parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,default=RUN);parser.add_argument("--offset",type=int,default=0);args=parser.parse_args();RUN=args.run.resolve()
 selected=[json.loads(x) for x in SOURCE.read_text().splitlines()][args.offset:args.offset+64]
 if len(selected)!=64 or len({x["dependency_cluster_id"] for x in selected})!=64 or any(x["split"]!="train" for x in selected):raise RuntimeError("selection not 64 disjoint TRAIN clusters")
 out=RUN/"selection-64.jsonl";out.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in selected))
 if len({x["raw_sha256"] for x in selected})!=64:raise RuntimeError("battle raw dependency overlap")
 worktrees=json.loads((ROOT/"experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json").read_text());commits=sorted({x["showdown_commit"] for x in selected});runtime=[]
 for commit in commits:
  path=Path(worktrees[commit]);actual=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
  if actual!=commit:raise RuntimeError("Showdown worktree commit drift")
  runtime.append({"path":str(path.resolve()),"commit":commit,"dist_tree_sha256":tree_sha256(path/"dist")})
 ext=next((ENGINE/"poke_engine").glob("poke_engine*.so"));files=[RUN/"PROTOCOL.md",out,SOURCE,ROOT/"experimental/src/scripts/run_cycle45_depth1_gate0.py",ROOT/"experimental/src/scripts/freeze_cycle45_depth1_gate0.py",ROOT/"experimental/src/scripts/run_cycle15_teacher_stability.py",ROOT/"experimental/src/scripts/audit_cycle13_train_rehydration.py",ROOT/"experimental/src/scripts/audit_cycle14_mechanics_repair.py",ROOT/"experimental/src/scripts/collect_cycle7_causal_child_targets.py",ROOT/"experimental/src/scripts/cycle12_replay_audit.py",ROOT/"experimental/src/scripts/replay_cycle8_inputlog.cjs",ROOT/"experimental/src/search/causal_child_target_v1.py",ROOT/"experimental/src/search/public_search_state_v1.py",ROOT/"srcs/metagross/causal_reveal_ledger.py",ext,*[Path(x["raw_path"]) for x in selected]]
 manifest={"schema":"metagross-cycle45-depth1-premeasurement/v1","status":"frozen_before_teacher_targets","engine_import_root":str(ENGINE.resolve()),"engine_binding_sha256":sha(ext),"selection_sha256":sha(out),"counts":{"battles":64,"dependency_clusters":64,"schedules":2,"worlds_per_schedule":8,"paths_per_world":8,"scheduled_rows":8192},"files":[{"path":str(p.resolve()),"sha256":sha(p)} for p in files],"showdown_runtime":runtime,"gate":{"unique_fingerprints":512,"battles":48,"support":.95,"top1_agreement":.8,"jsd_median":.05,"jsd_p90":.15},"authorization":{"training":False,"h2h":False,"sealed93":False,"gpu_cloud_paid":False}}
 path=RUN/"PREMEASUREMENT_MANIFEST.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"manifest_sha256":sha(path),"selection_sha256":sha(out),"engine_sha256":sha(ext)},sort_keys=True))
if __name__=="__main__":main()
