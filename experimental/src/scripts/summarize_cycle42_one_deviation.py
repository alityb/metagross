#!/usr/bin/env python3
"""Integrity-audit and score Cycle42's randomized first disagreement."""
from __future__ import annotations
import argparse,hashlib,json,math,re
from collections import Counter
from pathlib import Path
from srcs.metagross.terminal_mcts_one_deviation import EQUAL8192_SCHEMA,assignment_manifest
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import player_roles
from experimental.src.scripts.monitor_cycle19_operational_smoke import rows,to_id

def load(p:Path):return json.loads(p.read_text())
def fisher(tw,tn,pw,pn):
    totalw=tw+pw; total=tn+pn; den=math.comb(total,tn); upper=min(tn,totalw)
    return sum(math.comb(totalw,x)*math.comb(total-totalw,tn-x)/den for x in range(tw,upper+1))
def role(path:Path,user:str):
    r=player_roles(rows(path)); found=r.get(to_id(user))
    if found not in {"p1","p2"}:raise RuntimeError("missing authenticated public role")
    return found
def opportunity_files(run:Path):
    errors=[]; result=[]
    for p in sorted((run/"h2h-logs").glob("*.search.jsonl")):
        user=p.name.removesuffix(".search.jsonl"); telemetry=[]; bare=[]
        for i,row in enumerate(rows(p),1):
            override=row.get("choice_override") or {}; one=override.get("terminal_mcts_one_deviation"); teacher=override.get("terminal_mcts_teacher")
            if isinstance(teacher,dict) and not isinstance(one,dict):bare.append(i)
            if isinstance(one,dict):telemetry.append((i,row,override,one,teacher))
        if not telemetry:continue
        if bare:errors.append(f"{user}: teacher telemetry after lock or outside one-deviation controller")
        assigns={json.dumps(x[3].get("assignment"),sort_keys=True) for x in telemetry}
        if len(assigns)!=1:errors.append(f"{user}: assignment changed");continue
        a=telemetry[0][3]["assignment"]; eligible=[x for x in telemetry if x[3].get("eligible") is True]
        if len(eligible)>1:errors.append(f"{user}: multiple eligible opportunities")
        locked=[i for i,x in enumerate(telemetry) if x[3].get("locked_after_decision") is True]
        if locked and locked[0]!=len(telemetry)-1:errors.append(f"{user}: queried teacher after lock")
        failures=[]
        for line,row,override,one,teacher in telemetry:
            if one.get("schema")!=EQUAL8192_SCHEMA or one.get("teacher_contract")!="cycle41_equal8192":errors.append(f"{user}:{line}: wrong controller schema")
            if one.get("production_action")!=override.get("terminal_mcts_production_choice"):errors.append(f"{user}:{line}: production mismatch")
            if not isinstance(teacher,dict) or teacher.get("controller_schema")!="metagross-cycle19-equal8192-production-selector/v1":errors.append(f"{user}:{line}: wrong teacher")
            elif teacher.get("production_action")!=one.get("production_action") or teacher.get("world_count")!=16 or teacher.get("schedule_count")!=2 or teacher.get("iterations_per_world")!=8192 or len(teacher.get("receipts",[]))!=16 or any(x.get("total_visits")!=8192 for x in teacher["receipts"]):errors.append(f"{user}:{line}: equal8192 receipt violation")
            if one.get("integrity_failure") is not None:failures.append(one["integrity_failure"])
        opp=eligible[0] if eligible else None
        if opp:
            _,row,override,one,teacher=opp; expected=one["teacher_action"] if a["arm"]=="teacher" else one["production_action"]
            if override.get("final_choice")!=expected or row.get("choice")!=expected:errors.append(f"{user}: assigned action not selected")
            if teacher.get("production_action")!=override.get("terminal_mcts_production_choice") or teacher.get("selected_action")==teacher.get("production_action"):errors.append(f"{user}: opportunity is not certified disagreement")
        result.append({"username":user,"assignment":a,"queries":len(telemetry),"eligible":bool(opp),"opportunity":opp[3] if opp else None,"integrity_failures":failures,"public_role":role(p.with_name(user+".protocol.jsonl"),user)})
    return result,errors
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seed",default="202642160842");a=ap.parse_args(); run=a.run.resolve(); result=load(run/"h2h-result.json"); games=result.get("games",[]); records,errors=opportunity_files(run); expected=assignment_manifest(a.seed); byidx={x["game_index"]:x for x in games}; exp={x["game_index"]:x for x in expected["assignments"]}
    if len(games)!=20 or len(byidx)!=20:errors.append("not exactly 20 unique games")
    if len(records)!=20:errors.append("not exactly 20 candidate streams")
    witness=load(run/"REGISTRATION_CONSUMPTION.json"); regs=witness.get("registrations",[])
    if len(regs)!=40:errors.append("not exactly 40 consumed registrations")
    analyzed=[]
    for r in records:
        i=r["assignment"].get("game_index"); g=byidx.get(i); e=exp.get(i)
        if not g or not e:errors.append(f"game {i}: missing game/assignment");continue
        if any(r["assignment"].get(k)!=e[k] for k in ("game_index","pair_index","pair_leg","arm")):errors.append(f"game {i}: assignment mismatch")
        if r["assignment"].get("schedule_sha256")!=expected["schedule_sha256"]:errors.append(f"game {i}: schedule hash mismatch")
        if g.get("void") or g.get("error") is not None or g.get("winner") not in {"agent_a","agent_b"}:errors.append(f"game {i}: nondecisive/error")
        if r["integrity_failures"]:errors.append(f"game {i}: controller integrity failure")
        expected_role="p1" if i%2 else "p2"
        if r["public_role"]!=expected_role:errors.append(f"game {i}: candidate public role mismatch")
        analyzed.append({**r,"candidate_win":g.get("winner")=="agent_a","battle_tag":g.get("battle_tag"),"pair_id":g.get("pair_id")})
    if Counter(x["assignment"]["arm"] for x in analyzed)!=Counter({"teacher":10,"production":10}):errors.append("arm imbalance")
    if Counter(x["assignment"]["pair_leg"] for x in analyzed if x["assignment"]["arm"]=="teacher")!=Counter({1:5,2:5}):errors.append("mirror-leg assignment imbalance")
    eligible=[x for x in analyzed if x["eligible"]]; arms={k:[x for x in eligible if x["assignment"]["arm"]==k] for k in ("teacher","production")}; tn,pn=map(len,(arms["teacher"],arms["production"]));tw=sum(x["candidate_win"] for x in arms["teacher"]);pw=sum(x["candidate_win"] for x in arms["production"]);tr=tw/tn if tn else None;pr=pw/pn if pn else None;effect=tr-pr if tr is not None and pr is not None else None;p=fisher(tw,tn,pw,pn) if tn and pn else None
    role_effect={}
    for rr in ("p1","p2"):
        t=[x for x in arms["teacher"] if x["public_role"]==rr];q=[x for x in arms["production"] if x["public_role"]==rr];role_effect[rr]=(sum(x["candidate_win"] for x in t)/len(t)-sum(x["candidate_win"] for x in q)/len(q)) if t and q else None
    passed=not errors and len(eligible)>=14 and min(tn,pn)>=6 and effect is not None and effect>=.25 and all(v is not None and v>=0 for v in role_effect.values()) and p is not None and p<=.20
    report={"schema":"metagross-cycle42-one-deviation-result/v1","integrity":{"ok":not errors,"errors":errors},"all_games":{"n":len(analyzed)},"eligible":{"n":len(eligible),"teacher_n":tn,"production_n":pn,"teacher_wins":tw,"production_wins":pw,"teacher_win_rate":tr,"production_win_rate":pr,"effect":effect,"role_effect":role_effect,"one_sided_fisher_p":p},"gate":{"minimum_eligible":14,"minimum_per_arm":6,"minimum_effect":.25,"nonnegative_each_public_role":True,"maximum_one_sided_fisher_p":.20,"decision":"PASS_POWERED_REPLICATION_ONLY" if passed else "FAIL_RETIRE_ROOT_ONLY"},"games":sorted(analyzed,key=lambda x:x["assignment"]["game_index"])}
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"decision":report["gate"]["decision"],"eligible":len(eligible),"effect":effect,"p":p,"errors":errors},sort_keys=True))
if __name__=="__main__":main()
