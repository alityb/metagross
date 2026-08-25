#!/usr/bin/env python3
"""Witness exact Cycle42 registration consumption."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from experimental.src.scripts import watch_cycle33_registrations as base
PREFIX="c42od"
def validate(payload:dict,pair:dict,username:str)->dict:
    m=re.fullmatch(rf"{re.escape(PREFIX)}([xy])[0-9a-f]{{7}}",username)
    if m is None:raise RuntimeError("unexpected Cycle42 registration username")
    for k,v in {"schema_version":1,"pair_id":pair["pair_id"],"format":"gen9randombattle","battle_seed":pair["battle_seed"],"team_1_sha256":pair["team_1_sha256"],"team_2_sha256":pair["team_2_sha256"]}.items():
        if payload.get(k)!=v:raise RuntimeError("Cycle42 registration differs from frozen pair")
    assigned=payload.get("assigned_team_sha256")
    packed=pair["team_1_packed"] if assigned==pair["team_1_sha256"] else pair["team_2_packed"] if assigned==pair["team_2_sha256"] else None
    if packed is None or payload.get("packed_team")!=packed or hashlib.sha256(packed.encode()).hexdigest()!=assigned:raise RuntimeError("Cycle42 packed team mismatch")
    return {"pair_id":pair["pair_id"],"pair_index":pair["pair_index"],"leg":payload.get("leg"),"side":"p1" if m.group(1)=="x" else "p2","assigned_team_sha256":assigned,"username":username}
def main():
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True);p.add_argument("--pair-manifest",type=Path,required=True);p.add_argument("--timeout-seconds",type=float,default=7200);p.add_argument("--output",type=Path,required=True);p.add_argument("--username-prefix",default="c42od");a=p.parse_args();global PREFIX;PREFIX=a.username_prefix
    if a.output.exists():raise FileExistsError(a.output)
    base.validate=validate; result=base.watch(a.directory.resolve(),a.pair_manifest.resolve(),a.timeout_seconds);result["schema"]="metagross-cycle42-registration-consumption/v1";a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
