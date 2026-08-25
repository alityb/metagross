#!/usr/bin/env python3
"""Derive Cycle33 prepare/live H2H argv from one canonical object."""

from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path: Path):
    payload=json.loads(path.read_text()); evaluator=Path(payload["evaluator"]).resolve(); argv=payload.get("argv")
    if payload.get("schema")!="metagross-cycle33-canonical-h2h-argv/v1" or evaluator!=(ROOT/"experimental/src/eval/run.py").resolve(): raise RuntimeError("wrong Cycle33 canonical schema/evaluator")
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv) or {"--prepare-mirrored-pairs-only","--pair-manifest-sha256"}.intersection(argv): raise RuntimeError("malformed Cycle33 canonical argv")
    return evaluator, argv
def derived(path: Path, phase: str, pair_sha: str|None=None):
    evaluator, argv=load(path); value=list(argv)
    if phase=="prepare": value.append("--prepare-mirrored-pairs-only")
    elif phase=="live":
        if pair_sha is None or len(pair_sha)!=64: raise RuntimeError("Cycle33 live requires pair SHA")
        value += ["--pair-manifest-sha256",pair_sha]
    else: raise RuntimeError("unknown Cycle33 phase")
    return evaluator,value
def identity(path: Path, phase: str, pair_sha: str|None=None):
    from experimental.src.eval.run import parse_args, resume_config_sha256
    _,argv=derived(path,phase,pair_sha); return resume_config_sha256(parse_args(argv))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--canonical",type=Path,required=True); parser.add_argument("--phase",choices=("prepare","live","verify"),required=True); parser.add_argument("--pair-sha256"); parser.add_argument("--pair-manifest",type=Path); args=parser.parse_args(); path=args.canonical.resolve()
    if args.phase=="verify":
        if args.pair_manifest is None or args.pair_sha256 is None or sha(args.pair_manifest)!=args.pair_sha256: raise RuntimeError("Cycle33 pair verification failed")
        prepared=identity(path,"prepare"); live=identity(path,"live",args.pair_sha256); stored=json.loads(args.pair_manifest.read_text())["config_sha256"]
        if prepared!=live or prepared!=stored: raise RuntimeError("Cycle33 config identity differs")
        print(json.dumps({"status":"pass","config_sha256":stored},sort_keys=True)); return
    evaluator,argv=derived(path,args.phase,args.pair_sha256); command=[sys.executable,str(evaluator),*argv]
    if args.phase=="prepare": subprocess.run(command,cwd=ROOT,check=True)
    else: os.execv(sys.executable,command)
if __name__=="__main__": main()
