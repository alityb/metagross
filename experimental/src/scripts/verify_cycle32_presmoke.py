#!/usr/bin/env python3
import hashlib, json, subprocess, sys
from pathlib import Path
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(path: Path) -> dict:
    payload=json.loads(path.read_text())
    if payload.get("schema")!="metagross-cycle32-authenticated-identity-presmoke-freeze/v1": raise RuntimeError("wrong Cycle32 manifest schema")
    for row in payload["files"]:
        source=Path(row["path"])
        if not source.is_file() or sha(source)!=row["sha256"]: raise RuntimeError(f"Cycle32 frozen file mismatch: {source}")
    showdown=payload["showdown"]; root=Path(showdown["path"]); commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if commit!=showdown["commit"] or tree_sha256(root/"dist")!=showdown["dist_tree_sha256"]: raise RuntimeError("Cycle32 Showdown changed")
    return payload
if __name__=="__main__":
    result=verify(Path(sys.argv[1]).resolve()); print(json.dumps({"status":"pass","schema":result["schema"]},sort_keys=True))
