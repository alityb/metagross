#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(path:Path)->dict:
 payload=json.loads(path.read_text())
 if payload.get("schema")!="metagross-cycle33-h2h-premeasurement/v1":raise RuntimeError("wrong Cycle33 manifest")
 for row in payload["files"]:
  source=Path(row["path"])
  if not source.is_file() or sha(source)!=row["sha256"]:raise RuntimeError(f"Cycle33 frozen file mismatch: {source}")
 s=payload["showdown"];root=Path(s["path"]);commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
 if commit!=s["commit"] or tree_sha256(root/"dist")!=s["dist_tree_sha256"]:raise RuntimeError("Cycle33 Showdown changed")
 return payload
if __name__=="__main__":
 result=verify(Path(sys.argv[1]).resolve());print(json.dumps({"status":"pass","schema":result["schema"]},sort_keys=True))
