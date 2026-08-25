#!/usr/bin/env python3
"""Derive Cycle27 prepare/live smoke argv from one canonical object."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_canonical(path: Path) -> tuple[Path, list[str]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "metagross-cycle27-canonical-smoke-argv/v1":
        raise RuntimeError("wrong Cycle27 canonical argv schema")
    evaluator = Path(payload["evaluator"]).resolve()
    argv = payload.get("argv")
    if evaluator != (ROOT / "experimental/src/eval/run.py").resolve():
        raise RuntimeError("Cycle27 evaluator path changed")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise RuntimeError("Cycle27 canonical argv is malformed")
    if {"--prepare-mirrored-pairs-only", "--pair-manifest-sha256"}.intersection(argv):
        raise RuntimeError("Cycle27 canonical argv contains projected fields")
    return evaluator, argv


def derived_argv(
    canonical: Path, phase: str, pair_sha256: str | None = None,
) -> tuple[Path, list[str]]:
    evaluator, argv = load_canonical(canonical)
    derived = list(argv)
    if phase == "prepare":
        derived.append("--prepare-mirrored-pairs-only")
    elif phase == "live":
        if pair_sha256 is None or len(pair_sha256) != 64:
            raise RuntimeError("Cycle27 live phase requires frozen pair SHA")
        derived += ["--pair-manifest-sha256", pair_sha256]
    else:
        raise RuntimeError("unknown Cycle27 phase")
    return evaluator, derived


def config_identity(canonical: Path, phase: str, pair_sha256: str | None = None) -> str:
    from experimental.src.eval.run import parse_args, resume_config_sha256
    _, argv = derived_argv(canonical, phase, pair_sha256)
    return resume_config_sha256(parse_args(argv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "live", "verify"), required=True)
    parser.add_argument("--pair-sha256")
    parser.add_argument("--pair-manifest", type=Path)
    args = parser.parse_args()
    canonical = args.canonical.resolve()
    if args.phase == "verify":
        if args.pair_manifest is None or args.pair_sha256 is None:
            raise RuntimeError("Cycle27 verify requires pair manifest and SHA")
        if sha(args.pair_manifest) != args.pair_sha256:
            raise RuntimeError("Cycle27 pair bytes differ from frozen SHA")
        prepared = config_identity(canonical, "prepare")
        live = config_identity(canonical, "live", args.pair_sha256)
        stored = json.loads(args.pair_manifest.read_text())["config_sha256"]
        if prepared != live or prepared != stored:
            raise RuntimeError("Cycle27 preparation/live identity mismatch")
        print(json.dumps({"status": "pass", "config_sha256": stored}, sort_keys=True))
        return
    evaluator, argv = derived_argv(canonical, args.phase, args.pair_sha256)
    command = [sys.executable, str(evaluator), *argv]
    if args.phase == "prepare":
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
