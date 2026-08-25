#!/usr/bin/env python3
"""Witness exact consumption of all Cycle33 mirrored-pair registrations."""

from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path

def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()

def validate(payload: dict, pair: dict, username: str) -> dict:
    match=re.fullmatch(r"c33h2h([xy])[0-9a-f]{7}",username)
    if match is None: raise RuntimeError("unexpected Cycle33 registration username")
    leg=payload.get("leg")
    expected={"schema_version":1,"pair_id":pair["pair_id"],"format":"gen9randombattle","battle_seed":pair["battle_seed"],"team_1_sha256":pair["team_1_sha256"],"team_2_sha256":pair["team_2_sha256"]}
    if leg not in {1,2} or any(payload.get(key)!=value for key,value in expected.items()): raise RuntimeError("Cycle33 registration differs from frozen pair")
    assigned=payload.get("assigned_team_sha256")
    if assigned==pair["team_1_sha256"]: packed=pair["team_1_packed"]
    elif assigned==pair["team_2_sha256"]: packed=pair["team_2_packed"]
    else: raise RuntimeError("Cycle33 registration assigned unknown team")
    if payload.get("packed_team")!=packed or sha_bytes(packed.encode())!=assigned: raise RuntimeError("Cycle33 packed team mismatch")
    return {"pair_id":pair["pair_id"],"pair_index":pair["pair_index"],"leg":leg,"side":"p1" if match.group(1)=="x" else "p2","assigned_team_sha256":assigned}

def watch(directory: Path,pair_manifest: Path,timeout: float)->dict:
    pairs=json.loads(pair_manifest.read_text()).get("pairs")
    if not isinstance(pairs,list) or len(pairs)!=10: raise RuntimeError("Cycle33 requires ten frozen pairs")
    by_id={row["pair_id"]:row for row in pairs}; seen={}; deleted=set(); deadline=time.monotonic()+timeout; empty_since=None
    while time.monotonic()<deadline:
        current={path.stem:path for path in directory.glob("*.json")} if directory.exists() else {}
        if len(current)>2: raise RuntimeError("Cycle33 registration directory contains extras")
        for username,path in current.items():
            if username in deleted: raise RuntimeError("Cycle33 registration reappeared")
            try: raw=path.read_bytes()
            except FileNotFoundError: continue
            payload=json.loads(raw); pair=by_id.get(payload.get("pair_id"))
            if pair is None: raise RuntimeError("Cycle33 registration has unknown pair")
            record={"username":username,"registration_sha256":sha_bytes(raw),**validate(payload,pair,username)}
            if username in seen and seen[username]!=record: raise RuntimeError("Cycle33 registration mutated")
            seen[username]=record
        for username in set(seen)-set(current)-deleted: deleted.add(username)
        if len(seen)==40 and len(deleted)==40 and not current:
            empty_since=empty_since or time.monotonic()
            if time.monotonic()-empty_since>=1:
                groups={}
                for row in seen.values(): groups.setdefault((row["pair_id"],row["leg"]),[]).append(row)
                if len(groups)!=20 or any(len(rows)!=2 or {r["side"] for r in rows}!={"p1","p2"} or {r["assigned_team_sha256"] for r in rows}!={by_id[key[0]]["team_1_sha256"],by_id[key[0]]["team_2_sha256"]} for key,rows in groups.items()): raise RuntimeError("Cycle33 registration pair/leg coverage changed")
                return {"schema":"metagross-cycle33-registration-consumption/v1","status":"pass","pair_manifest_sha256":sha_bytes(pair_manifest.read_bytes()),"pairs":10,"legs":20,"registrations_observed":40,"registrations_consumed":40,"registration_reappearances":0,"remaining_files":[],"registrations":sorted(seen.values(),key=lambda r:(r["pair_index"],r["leg"],r["side"]))}
        else: empty_since=None
        time.sleep(.01)
    raise TimeoutError("Cycle33 did not consume exactly 40 registrations")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--directory",type=Path,required=True); p.add_argument("--pair-manifest",type=Path,required=True); p.add_argument("--timeout-seconds",type=float,default=3600); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=watch(a.directory.resolve(),a.pair_manifest.resolve(),a.timeout_seconds); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":"pass","consumed":40},sort_keys=True))
if __name__=="__main__": main()
