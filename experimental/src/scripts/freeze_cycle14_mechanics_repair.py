#!/usr/bin/env python3
"""Freeze Cycle 14 files and runtimes before measurement."""
from __future__ import annotations
import json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from experimental.src.scripts.run_cycle8_replay_audit import sha256_path, tree_sha256

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experimental/runs/search_native_v2_cycle14_mechanics_repair_20260815'
BIND=RUN/'engine-binding/unpacked/poke_engine/poke_engine.cpython-311-darwin.so'

def add_tree(paths, root):
    paths.update(p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')

def main():
    out=RUN/'PREMEASUREMENT_MANIFEST.json'
    if out.exists(): raise RuntimeError('Cycle14 manifest exists')
    sel=[json.loads(x) for x in (RUN/'selection-200.jsonl').read_text().splitlines()]
    if len(sel)!=200 or len({x['dependency_cluster_id'] for x in sel})!=200 or any(x['split']!='train' for x in sel): raise RuntimeError('selection changed')
    if sha256_path(RUN/'selection-200.jsonl')!='d1d31d96f807fcdc1b5c3ae60e5feb28e9f828be2a829b04136dffe525864031': raise RuntimeError('not identical Cycle13 selection')
    paths={RUN/'PROTOCOL.md',RUN/'selection-200.jsonl',RUN/'selection-report.json',BIND,
      RUN/'engine-binding/poke_engine-0.0.47-cp311-cp311-macosx_11_0_arm64.whl',
      ROOT/'experimental/src/scripts/audit_cycle14_mechanics_repair.py',ROOT/'experimental/src/scripts/freeze_cycle14_mechanics_repair.py',
      ROOT/'experimental/src/scripts/audit_cycle13_train_rehydration.py',ROOT/'experimental/src/scripts/tests/test_cycle14_mechanics_repair.py',
      ROOT/'experimental/src/scripts/tests/test_cycle13_train_rehydration.py',ROOT/'experimental/src/scripts/replay_cycle8_inputlog.cjs',
      ROOT/'experimental/src/scripts/cycle8_replay_audit.py',ROOT/'experimental/src/scripts/cycle9_replay_audit.py',
      ROOT/'experimental/src/scripts/cycle11_replay_audit.py',ROOT/'experimental/src/scripts/cycle12_replay_audit.py',
      ROOT/'experimental/src/scripts/run_cycle10_full_corpus_index.py',ROOT/'experimental/src/scripts/run_cycle8_replay_audit.py',
      ROOT/'experimental/src/search/public_search_state_v1.py',ROOT/'srcs/metagross/causal_reveal_ledger.py',
      ROOT/'srcs/metagross/export_showdown_public_form_contract.cjs',ROOT/'experimental/engine/pe_v3_learned_priors/Cargo.toml',
      ROOT/'experimental/engine/pe_v3_learned_priors/Cargo.lock',ROOT/'experimental/engine/pe_v3_learned_priors/poke-engine-py/Cargo.toml',
      ROOT/'experimental/engine/pe_v3_learned_priors/poke-engine-py/python/tests/test_poke_engine.py',
      ROOT/'experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json'}
    paths.update(Path(x['raw_path']).resolve() for x in sel)
    for d in ('srcs/vendor/foul-play/fp','srcs/vendor/foul-play/data','experimental/engine/pe_v3_learned_priors/src','experimental/engine/pe_v3_learned_priors/poke-engine-py/src'): add_tree(paths,ROOT/d)
    paths.update(Path(shutil.which(x) if '/' not in x else x).resolve() for x in ('node','/opt/homebrew/bin/python3.11'))
    if any(not p.is_file() for p in paths): raise RuntimeError('missing frozen file')
    wt=json.loads((ROOT/'experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json').read_text())
    runtimes=[]
    for commit in sorted({x['showdown_commit'] for x in sel}):
      p=Path(wt[commit]).resolve(); actual=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
      if actual!=commit: raise RuntimeError('Showdown commit mismatch')
      runtimes.append({'commit':commit,'path':str(p),'dist_tree_sha256':tree_sha256(p/'dist')})
    m={'schema':'metagross-cycle14-premeasurement-manifest/v1','status':'frozen_before_measurement','frozen_at':datetime.now(timezone.utc).isoformat(),
      'protocol_sha256':sha256_path(RUN/'PROTOCOL.md'),'selection_sha256':sha256_path(RUN/'selection-200.jsonl'),
      'engine_binding_path':str(BIND.resolve()),'engine_binding_sha256':sha256_path(BIND),
      'files':[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha256_path(p)} for p in sorted(paths,key=str)],
      'showdown_runtime':runtimes,'fixed_measurement':{'roots':200,'minimum_passed':190,'schedules':2,'worlds':8,'repeats':2,'fresh_subprocesses':200},
      'authorization':{'teacher':False,'training':False,'h2h':False,'validation_test':False,'sealed93':False,'gpu_cloud_paid':False}}
    out.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest_sha256':sha256_path(out),'protocol_sha256':m['protocol_sha256'],'selection_sha256':m['selection_sha256'],'engine_binding_sha256':m['engine_binding_sha256'],'files':len(m['files'])},indent=2))
if __name__=='__main__': main()
