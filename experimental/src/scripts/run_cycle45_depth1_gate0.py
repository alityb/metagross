#!/usr/bin/env python3
"""Cycle45 depth-one target mechanics/stability pilot."""
from __future__ import annotations
import argparse,concurrent.futures,copy,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
from typing import Any
from experimental.src.scripts import run_cycle15_teacher_stability as c15
from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts.collect_cycle7_causal_child_targets import child_reveal_sidecar
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from search.causal_child_target_v1 import child_information_state
from search.public_search_state_v1 import canonical_bytes
from srcs.metagross import causal_reveal_ledger as crl
ROOT=Path(__file__).resolve().parents[3];SCHEDULES=2;WORLDS=8;PATHS=8;BASE=2026451608
class GateError(RuntimeError):pass
def h(v:object)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def seed(*p)->int:return int.from_bytes(hashlib.sha256("\0".join(map(str,p)).encode()).digest()[:8],"big")%(2**32)
def result_payload(result):
 def side(rows):return [{"action":str(x.move_choice),"N":int(x.visits),"W":float(x.total_score),"Q":float(x.total_score)/int(x.visits) if int(x.visits) else None} for x in rows]
 return {"total_visits":int(result.total_visits),"side_one":side(result.side_one),"side_two":side(result.side_two)}
def policy(result,legal):
 by={x["action"]:x["N"] for x in result["side_one"]}; total=sum(by.values())
 if set(by)!=set(legal) or total<=0:raise GateError("teacher legal/visit support mismatch")
 p={a:by[a]/total for a in legal};top=min(legal,key=lambda a:(-p[a],a));return p,top
def jsd(a,b):
 keys=set(a)|set(b);m={k:(a.get(k,0)+b.get(k,0))/2 for k in keys}
 def kl(p):return math.fsum(p.get(k,0)*math.log(p.get(k,0)/m[k]) for k in keys if p.get(k,0)>0)
 return (kl(a)+kl(b))/2
def search(engine,state,legal,iterations,tag):
 first,second=map(list,engine.root_options(state));
 if first!=legal or not second:raise GateError("interior legal orientation mismatch")
 p1=[(x,1/len(first)) for x in first];p2=[(x,1/len(second)) for x in second]
 r=engine.monte_carlo_tree_search(state,duration_ms=0,iterations=iterations,threads=1,s1_priors=p1,s2_priors=p2,c_puct=2.0,seed=seed(BASE,tag,iterations));out=result_payload(r)
 if out["total_visits"]!=iterations:raise GateError("teacher visit count mismatch")
 pol,top=policy(out,legal);return {"iterations":iterations,"policy":pol,"top1":top,"result":out}
def worker(run:Path,index:int)->dict:
 manifest=verify_manifest(run/"PREMEASUREMENT_MANIFEST.json"); selected=json.loads((run/"selection-64.jsonl").read_text().splitlines()[index]);sys.path.insert(0,manifest["engine_import_root"]);import poke_engine
 sys.path.insert(0,str(ROOT/"srcs/vendor/foul-play"));old=Path.cwd();os.chdir(ROOT/"srcs/vendor/foul-play")
 rows=[];failures=[];scheduled=SCHEDULES*WORLDS*PATHS
 try:
  from data.pkmn_sets import RandomBattleTeamDatasets
  from fp.search import main as search_main
  RandomBattleTeamDatasets.initialize("gen9");c14.install_monkeypatches(poke_engine);ctx=c15.materialize_schedules(run,selected,poke_engine,search_main);c14.CURRENT_ACTIONS[:]=list(ctx["request_actions"]);ledger=crl.attached_ledger(ctx["battle"])
  for sch in ctx["paired"]:
   for wi,w in enumerate(sch["worlds"]):
    root=poke_engine.State.from_string(w["state"]);rootstr=root.to_string();own,opp=map(list,poke_engine.root_options_with_s1_request(root,ctx["request_actions"]));pairs=[(a,b) for a in own for b in opp];pairs=sorted(pairs,key=lambda x:seed(selected["dependency_cluster_id"],"path",*x))[:PATHS]
    if len(pairs)<PATHS:
     failures.extend({"row":f"{index}:{sch['schedule_index']}:{wi}:{pi}","class":"insufficient_joint_paths","detail_sha256":h(len(pairs))} for pi in range(len(pairs),PATHS))
    for pi,(a,b) in enumerate(pairs):
     rowid=f"{index}:{sch['schedule_index']}:{wi}:{pi}"
     try:
      first=poke_engine.step_with_uniform_r1_semantic(root,a,b,0.5);second=poke_engine.step_with_uniform_r1_semantic(root,a,b,0.5)
      sig=lambda x:(x.state.to_string(),float(x.selected_instructions.percentage),tuple(map(str,x.selected_instructions.instruction_list)))
      if sig(first)!=sig(second):raise GateError("semantic step nondeterminism")
      if first.state.reverse_instructions(first.selected_instructions).to_string()!=rootstr:raise GateError("apply/reverse mismatch")
      if float(poke_engine.terminal_value(first.state))!=0:raise GateError("terminal child")
      legal,opplegal=map(list,poke_engine.root_options(first.state))
      if not legal or not opplegal or "none" in [x.lower() for x in legal+opplegal]:raise GateError("automatic/unsupported child")
      sidecar=child_reveal_sidecar(root,first,ledger);public=child_information_state(first.state,poke_engine);pb=canonical_bytes(public)
      from scripts.run_public_search_state_gate_a import hidden_perturbation
      perturb=hidden_perturbation(first.state,poke_engine)
      if perturb is not None and canonical_bytes(child_information_state(perturb,poke_engine))!=pb:raise GateError("hidden sensitivity")
      info={"public_sha256":hashlib.sha256(pb).hexdigest(),"root_ledger_sha256":sidecar["root_ledger_sha256"],"semantic_events":sidecar["semantic_events"],"added_reveals":sidecar["added_reveals"],"root_mask":sidecar["root_mask"],"child_mask":sidecar["child_mask"],"legal_actions":legal,"observer":"side_one","public_path":{"own":a,"opponent":b,"uniform":0.5}}
      fingerprint=h(info);t8a=search(poke_engine,first.state,legal,8192,rowid+":a");t8b=search(poke_engine,first.state,legal,8192,rowid+":b");t20=search(poke_engine,first.state,legal,20000,rowid+":c")
      rows.append({"dependency_cluster_id":selected["dependency_cluster_id"],"battle_provenance_sha256":h(selected["dependency_cluster_id"]),"schedule_index":sch["schedule_index"],"world_index":wi,"path_index":pi,"raw_weight":w["weight"],"information_fingerprint":fingerprint,"information":info,"state_sha256":hashlib.sha256(first.state.to_string().encode()).hexdigest(),"teachers":{"equal8192_a":t8a,"equal8192_b":t8b,"equal20000":t20},"repeat_jsd":jsd(t8a["policy"],t8b["policy"]),"agreement_8192_20000":t8a["top1"]==t20["top1"] and t8b["top1"]==t20["top1"]})
     except Exception as exc:failures.append({"row":rowid,"class":type(exc).__name__,"detail_sha256":hashlib.sha256(str(exc).encode()).hexdigest()})
 except Exception as exc:
  remaining=scheduled-len(rows)-len(failures);failures.extend({"row":f"{index}:root:{i}","class":type(exc).__name__,"detail_sha256":hashlib.sha256(str(exc).encode()).hexdigest()} for i in range(max(0,remaining)))
 finally:os.chdir(old)
 return {"index":index,"cluster":selected["dependency_cluster_id"],"scheduled":scheduled,"supported":len(rows),"failures":failures,"rows":rows}
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--workers",type=int,default=8);a=p.parse_args();run=a.run.resolve();verify_manifest(run/"PREMEASUREMENT_MANIFEST.json");out=run/"measurement";out.mkdir();(out/"tmp").mkdir()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:results=list(ex.map(worker,[run]*64,range(64),chunksize=1))
 raw=[row for r in results for row in r["rows"]];fail=[x for r in results for x in r["failures"]];scheduled=sum(r["scheduled"] for r in results);supported=len(raw);fingerprints={r["information_fingerprint"] for r in raw};battles={r["dependency_cluster_id"] for r in raw};js=[r["repeat_jsd"] for r in raw];agree=sum(r["agreement_8192_20000"] for r in raw)/supported if supported else 0
 ordered=sorted(js);p90=ordered[min(len(ordered)-1,math.ceil(.9*len(ordered))-1)] if ordered else None;median=statistics.median(js) if js else None
 gates={"unique_fingerprints_ge512":len(fingerprints)>=512,"battles_ge48":len(battles)>=48,"zero_hidden_split_leakage":not any("hidden" in json.dumps(x).lower() for x in fail),"ordinary_support_ge95":supported/scheduled>=.95,"top1_agreement_ge80":agree>=.8,"repeat_jsd_median_le05":median is not None and median<=.05,"repeat_jsd_p90_le15":p90 is not None and p90<=.15}
 with (out/"raw-path-world-targets.jsonl").open("x") as f:
  for row in raw:f.write(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
 report={"schema":"metagross-cycle45-depth1-gate0/v1","status":"pass" if all(gates.values()) else "fail","counts":{"selected_battles":64,"scheduled":scheduled,"supported":supported,"unique_fingerprints":len(fingerprints),"battles_with_support":len(battles),"failures":len(fail)},"metrics":{"coverage":supported/scheduled,"top1_8192_20000_agreement":agree,"repeat_jsd_median":median,"repeat_jsd_p90":p90},"gates":gates,"failure_classes":{x["class"]:sum(y["class"]==x["class"] for y in fail) for x in fail},"authorization":{"tiny_cpu_training_smoke":all(gates.values()),"h2h":False,"sealed93":False,"gpu_cloud_paid":False}}
 (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
