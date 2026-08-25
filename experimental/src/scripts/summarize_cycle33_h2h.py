#!/usr/bin/env python3
"""Fail-closed scorer for Cycle33's fixed prospective 20-game gate."""

from __future__ import annotations
import argparse, hashlib, json, math, statistics
from pathlib import Path
from experimental.src.scripts.monitor_cycle19_operational_smoke import rows, to_id, validate_teacher
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import first_private_roster, packed_roster, player_roles, public_leads
from experimental.src.scripts.summarize_cycle19_h2h import engine_provenance, validate_candidate_file, wilson
from experimental.src.scripts.verify_cycle33_h2h_freeze import verify

CONTROLLER="metagross-cycle19-equal8192-production-selector/v1"
def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(row: dict)->tuple:
    context=row.get("context") or {}; tag=context.get("tag"); idx=context.get("decision_idx")
    root=hashlib.sha256(f"terminal-mcts-live\0{tag}\0{idx}".encode()).hexdigest()
    return tag,context.get("rqid"),idx,root

def load_receipts(run: Path)->dict[tuple,list[dict]]:
    indexed={}
    files=sorted((run/"move-receipts").glob("*.jsonl"))
    if len(files)!=40: raise RuntimeError(f"Cycle33 expected 40 move-receipt files, found {len(files)}")
    for path in files:
        file_rows=rows(path)
        if not file_rows: raise RuntimeError(f"empty Cycle33 receipt file: {path}")
        for row in file_rows:
            context=row.get("execution_context") or {}; phase=context.get("phase"); cohort=context.get("cohort")
            if (phase,cohort) not in {("production_control","adaptive_root_search"),("equal8192_candidate","fixed_two_by_eight")}: raise RuntimeError("Cycle33 unknown receipt cohort")
            if row.get("schema")!="metagross-causal-move-conversion-receipt/v1" or row.get("observer_role")!="p1" or row.get("swap") is not False: raise RuntimeError("Cycle33 malformed receipt envelope")
            nested=row.get("move_receipt") or {}
            if nested.get("protocol_sha256")!=row.get("protocol_sha256") or nested.get("battle_tag")!=context.get("battle_tag") or not isinstance(nested.get("moves"),list) or not isinstance(nested.get("derived_executions"),list): raise RuntimeError("Cycle33 nested receipt identity changed")
            for move in nested["moves"]:
                if move.get("disable_authority") not in {"causal_disable","world_mechanical_disable"} or isinstance(move.get("current_pp"),bool) or not isinstance(move.get("current_pp"),int) or isinstance(move.get("max_pp"),bool) or not isinstance(move.get("max_pp"),int) or not 0<=move["current_pp"]<=move["max_pp"] or not isinstance(move.get("world_disabled"),bool): raise RuntimeError("Cycle33 invalid PP/disable receipt")
                if move["disable_authority"]=="causal_disable" and move["world_disabled"] is not True: raise RuntimeError("Cycle33 causal disable missing from world")
            for event in nested["derived_executions"]:
                if not isinstance(event,dict) or event.get("authority")!="derived_public_execution" or not event.get("derived_cause"): raise RuntimeError("Cycle33 invalid derived execution")
            key=(phase,context.get("battle_tag"),context.get("rqid"),context.get("decision_index"),context.get("root_id"))
            indexed.setdefault(key,[]).append(row)
    return indexed

def validate_cohort(indexed: dict, base: tuple, phase: str, decision_ns: int)->dict:
    key=(phase,*base); cohort=indexed.pop(key,[])
    if not cohort: raise RuntimeError(f"Cycle33 missing {phase} receipts")
    if any(int(row.get("receipt_time_ns",0))>decision_ns for row in cohort): raise RuntimeError("Cycle33 receipt occurred after decision")
    protocols={row.get("protocol_sha256") for row in cohort}
    if len(protocols)!=1: raise RuntimeError("Cycle33 cohort spans causal prefixes")
    contexts=[row["execution_context"] for row in cohort]
    if phase=="production_control":
        declared={row.get("declared_world_count") for row in contexts}
        if len(declared)!=1 or next(iter(declared)) not in {16,32}: raise RuntimeError("Cycle33 production declaration invalid")
        count=int(next(iter(declared)))
        if len(cohort)!=count or {row.get("conversion_index") for row in contexts}!=set(range(count)): raise RuntimeError("Cycle33 production receipts incomplete")
    else:
        count=16; cells={(row.get("schedule_index"),row.get("world_index")) for row in contexts}
        if len(cohort)!=16 or cells!={(s,w) for s in range(2) for w in range(8)} or {row.get("conversion_index") for row in contexts}!=set(range(16)) or any(row.get("declared_world_count")!=16 for row in contexts): raise RuntimeError("Cycle33 candidate receipts incomplete")
    return {"count":count,"protocol_sha256":next(iter(protocols))}

def registration_audit(run: Path,pair_path: Path,protocol_paths: list[Path])->dict:
    witness=json.loads((run/"REGISTRATION_CONSUMPTION.json").read_text())
    if witness.get("status")!="pass" or witness.get("pair_manifest_sha256")!=sha(pair_path) or witness.get("registrations_consumed")!=40 or witness.get("remaining_files")!=[] or any((run/"h2h-registrations").iterdir()): raise RuntimeError("Cycle33 registration witness failed")
    pairs={row["pair_id"]:row for row in json.loads(pair_path.read_text())["pairs"]}
    protocols={path.name.removesuffix(".protocol.jsonl"):rows(path) for path in protocol_paths}
    for registration in witness["registrations"]:
        username=registration["username"]; protocol=protocols.get(username); pair=pairs[registration["pair_id"]]
        if protocol is None: raise RuntimeError("Cycle33 registration lacks protocol")
        role=registration["side"]; roles=player_roles(protocol)
        if roles.get(to_id(username))!=role: raise RuntimeError("Cycle33 public username/role mismatch")
        packed=pair["team_1_packed"] if registration["assigned_team_sha256"]==pair["team_1_sha256"] else pair["team_2_packed"]
        roster=packed_roster(packed)
        if first_private_roster(protocol)!=roster or public_leads(protocol).get(role)!=roster[0]: raise RuntimeError("Cycle33 private roster/public lead mismatch")
    return {"witness_sha256":sha(run/"REGISTRATION_CONSUMPTION.json"),"registrations":40,"packed_team_seed_role_lead_parity":True}

def summarize(run: Path,manifest_path: Path)->dict:
    manifest=verify(manifest_path); pair_path=run/"h2h-result.json.pairs.json"; result_path=run/"h2h-result.json"; payload=json.loads(result_path.read_text()); games=payload.get("games") or []; summary=payload.get("summary") or {}
    if len(games)!=20 or summary.get("completed_games")!=20 or summary.get("void_games")!=0 or summary.get("decisive_games")!=20: raise RuntimeError("Cycle33 outcome denominator invalid")
    if any(g.get("void") or g.get("error") or g.get("winner") not in {"agent_a","agent_b"} for g in games): raise RuntimeError("Cycle33 failed game row")
    if sum(g.get("challenger")=="agent_a" for g in games)!=10 or sum(g.get("acceptor")=="agent_a" for g in games)!=10: raise RuntimeError("Cycle33 role balance invalid")
    by_pair={}
    for game in games: by_pair.setdefault(int(game["pair_index"]),[]).append(game)
    if set(by_pair)!=set(range(1,11)): raise RuntimeError("Cycle33 pair indices incomplete")
    pair_results=[]
    for index,group in sorted(by_pair.items()):
        if len(group)!=2 or {g["pair_leg"] for g in group}!={1,2} or len({g["pair_id"] for g in group})!=1 or len({tuple(sorted((g["team_1_sha256"],g["team_2_sha256"]))) for g in group})!=1 or len({(g["agent_a_team_sha256"],g["agent_b_team_sha256"]) for g in group})!=2: raise RuntimeError(f"Cycle33 pair {index} mirror invalid")
        ordered=sorted(group,key=lambda g:g["pair_leg"]); pair_results.append({"pair_index":index,"pair_id":ordered[0]["pair_id"],"leg_1_winner":ordered[0]["winner"],"leg_2_winner":ordered[1]["winner"],"candidate_wins":sum(g["winner"]=="agent_a" for g in group)})
    logs=sorted((run/"h2h-logs").glob("*.log")); searches=sorted((run/"h2h-logs").glob("*.search.jsonl")); protocols=sorted((run/"h2h-logs").glob("*.protocol.jsonl"))
    if len(logs)!=40 or len(searches)!=40 or len(protocols)!=40: raise RuntimeError("Cycle33 spawned artifact count invalid")
    for log in logs: engine_provenance(log,manifest["engine"]["native_sha256"])
    for path in protocols:
        if any(row.get("direction") in {"send_failure","send_rejected","reconnect"} for row in rows(path)): raise RuntimeError("Cycle33 protocol operational failure")
    registration=registration_audit(run,pair_path,protocols); indexed=load_receipts(run); candidate_paths=[]; comparator_paths=[]
    for path in searches:
        has=any(((row.get("choice_override") or {}).get("terminal_mcts_teacher") or {}).get("controller_schema")==CONTROLLER for row in rows(path)); (candidate_paths if has else comparator_paths).append(path)
    if len(candidate_paths)!=20 or len(comparator_paths)!=20: raise RuntimeError("Cycle33 candidate/comparator assignment invalid")
    decisions=overrides=0; latencies=[]; production_receipts=candidate_receipts=0
    for path in candidate_paths:
        count,lats,ovs=validate_candidate_file(path); decisions+=count; latencies+=lats; overrides+=ovs
        for row in rows(path):
            base=identity(row); prod=validate_cohort(indexed,base,"production_control",int(row["time_ns"])); cand=validate_cohort(indexed,base,"equal8192_candidate",int(row["time_ns"]));
            if prod["protocol_sha256"]!=cand["protocol_sha256"]: raise RuntimeError("Cycle33 candidate/production prefixes differ")
            production_receipts+=prod["count"]; candidate_receipts+=cand["count"]
    for path in comparator_paths:
        for row in rows(path):
            if (row.get("choice_override") or {}).get("terminal_mcts_teacher"): raise RuntimeError("Cycle33 candidate leaked into comparator")
            production_receipts+=validate_cohort(indexed,identity(row),"production_control",int(row["time_ns"]))["count"]
    if indexed: raise RuntimeError(f"Cycle33 unjoined receipt cohorts remain: {len(indexed)}")
    wins=int(summary.get("agent_a_wins")); losses=int(summary.get("agent_a_losses"));
    if wins+losses!=20: raise RuntimeError("Cycle33 result accounting incomplete")
    low,high=wilson(wins,20); role={"candidate_as_challenger":{"games":10,"wins":sum(g["winner"]=="agent_a" and g["challenger"]=="agent_a" for g in games)},"candidate_as_acceptor":{"games":10,"wins":sum(g["winner"]=="agent_a" and g["acceptor"]=="agent_a" for g in games)}}
    return {"schema":"metagross-cycle33-h2h-result/v1","status":"pass" if wins>=13 else "fail","games":20,"mirrored_pairs":10,"candidate_wins":wins,"candidate_losses":losses,"candidate_win_rate":wins/20,"wilson95":[low,high],"pair_results":pair_results,"role_results":role,"candidate_decisions":decisions,"candidate_overrides":overrides,"candidate_pass_through":decisions-overrides,"candidate_mean_latency_ms":statistics.fmean(latencies),"candidate_p95_latency_ms":sorted(latencies)[max(0,math.ceil(.95*len(latencies))-1)],"production_conversion_receipts":production_receipts,"candidate_conversion_receipts":candidate_receipts,"registration":registration,"spawned_engine_logs":40,"semantic_operational_integrity_failures":0,"result_sha256":sha(result_path),"postrun_manifest_integrity":"pass","gate":"continue_only_if_at_least_13_wins","strength_claim_authorized":False}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); report=summarize(a.run.resolve(),a.manifest.resolve()); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
