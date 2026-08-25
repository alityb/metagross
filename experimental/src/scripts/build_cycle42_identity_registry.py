#!/usr/bin/env python3
"""Build a label-blind registry of all prior live identities for Cycle42."""
from __future__ import annotations
import argparse,hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v:object)->str:return json.dumps(v,sort_keys=True,separators=(",",":"))
def av(argv:list,flag:str):
    try:return argv[argv.index(flag)+1]
    except (ValueError,IndexError):return None
def rows(v):
    if isinstance(v,dict):
        if any(k in v for k in ("team_1_sha256","team_2_sha256","pair_id","battle_seed")):yield v
        for x in v.values():yield from rows(x)
    elif isinstance(v,list):
        for x in v:yield from rows(x)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,default=ROOT/"experimental/runs/search_native_v2_cycle42_one_deviation_20260816");args=parser.parse_args();global RUN;RUN=args.run.resolve()
    out=RUN/"PRIOR_IDENTITY_REGISTRY.json"
    if out.exists():raise FileExistsError(out)
    pairs=set();teams=set();ids=set();seeds=set();sources=[]
    for p in sorted((ROOT/"experimental/runs").rglob("*.json")):
        if RUN in p.parents or ("sealed" in str(p).lower() and "93" in str(p)):continue
        try:v=json.loads(p.read_text())
        except Exception:continue
        n=0
        for r in rows(v):
            a,b=r.get("team_1_sha256"),r.get("team_2_sha256")
            if isinstance(a,str) and isinstance(b,str):pairs.add(canon(sorted((a,b))));teams|={a,b};n+=1
            if isinstance(r.get("pair_id"),str):ids.add(r["pair_id"])
            if r.get("battle_seed") is not None:seeds.add(canon(r["battle_seed"]))
        if n:sources.append({"path":str(p.resolve()),"sha256":sha(p),"rows":n})
    ms=set();ps=set();runs=set();prefixes=set();configs=[]
    for p in sorted((ROOT/"experimental/runs").rglob("*ARGV.json")):
        if RUN in p.parents:continue
        try:argv=json.loads(p.read_text()).get("argv")
        except Exception:continue
        if not isinstance(argv,list):continue
        for flag,sink in (("--mirror-seed",ms),("--production-run-seed",ps),("--run-id",runs),("--username-prefix",prefixes)):
            x=av(argv,flag)
            if isinstance(x,str):sink.add(x)
        configs.append({"path":str(p.resolve()),"sha256":sha(p)})
    names=set()
    for p in (ROOT/"experimental/runs").rglob("REGISTRATION_CONSUMPTION.json"):
        if RUN in p.parents:continue
        try:v=json.loads(p.read_text())
        except Exception:continue
        for r in v.get("registrations",[]):
            if isinstance(r,dict) and isinstance(r.get("username"),str):names.add(r["username"])
    for p in (ROOT/"experimental/runs").rglob("*.protocol.jsonl"):
        if RUN not in p.parents:names.add(p.name.removesuffix(".protocol.jsonl"))
    payload={"schema":"metagross-cycle42-prior-identity-registry/v1","status":"frozen_before_cycle42_pair_generation","labels_or_outcomes_read":False,"sources":sources,"config_sources":configs,"unordered_team_pairs":sorted(pairs),"individual_team_sha256":sorted(teams),"pair_ids":sorted(ids),"battle_seeds":sorted(seeds),"mirror_seeds":sorted(ms),"production_run_seeds":sorted(ps),"run_ids":sorted(runs),"username_prefixes":sorted(prefixes),"usernames":sorted(names)}
    payload["counts"]={k:len(payload[k]) for k in ("unordered_team_pairs","individual_team_sha256","pair_ids","battle_seeds","usernames")}
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload["counts"],sort_keys=True))
if __name__=="__main__":main()
