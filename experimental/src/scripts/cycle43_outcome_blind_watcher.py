#!/usr/bin/env python3
"""Cycle43 progress snapshots from receipt/registration metadata only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
FORBIDDEN_PATH_PARTS={"result","runner","protocol","showdown","decision","prior"}
FORBIDDEN_KEYS={"winner","win","score","outcome","result","terminal","games"}
ALLOWED_DIRS={"move-receipts","ability-receipts","engine-receipts","h2h-registrations"}
def validate_path(run:Path,path:Path)->None:
    rel=path.resolve().relative_to(run.resolve())
    if not rel.parts or rel.parts[0] not in ALLOWED_DIRS or any(any(token in part.lower() for token in FORBIDDEN_PATH_PARTS) for part in rel.parts):raise RuntimeError("outcome-blind watcher rejected non-whitelisted path")
def reject_outcome_keys(value:object)->None:
    if isinstance(value,dict):
        for key,child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:raise RuntimeError("outcome-bearing key rejected")
            reject_outcome_keys(child)
    elif isinstance(value,list):
        for child in value:reject_outcome_keys(child)
def snapshot(run:Path)->dict:
    counts={}
    for directory in sorted(ALLOWED_DIRS):
        root=run/directory; files=[] if not root.exists() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:validate_path(run,path)
        counts[directory]={"files":len(files),"bytes":sum(p.stat().st_size for p in files)}
    payload={"schema":"metagross-cycle43-outcome-blind-heartbeat/v1","receipt_counts":counts,"outcome_fields_read":False,"runner_stdout_read":False,"result_bytes_read":False,"protocol_bytes_read":False}
    reject_outcome_keys(payload);return payload
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path);a=p.parse_args();v=snapshot(a.run.resolve());rendered=json.dumps(v,indent=2,sort_keys=True)+"\n";(a.output.write_text(rendered) if a.output else print(rendered,end=""))
if __name__=="__main__":main()
