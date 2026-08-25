#!/usr/bin/env python3
"""Verify immutable Cycle 18 inputs without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        source = Path(row["path"])
        if not source.is_file() or sha(source) != row["sha256"]:
            raise RuntimeError(f"frozen file mismatch: {source}")
    showdown = manifest["showdown"]
    repo = Path(showdown["path"])
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != showdown["commit"]:
        raise RuntimeError("Showdown commit changed")
    if tree_sha256(repo / "dist") != showdown["dist_tree_sha256"]:
        raise RuntimeError("Showdown dist changed")
    pair = json.loads(Path(manifest["pair_manifest"]["path"]).read_text())
    if len(pair.get("pairs", [])) != 10 or pair.get("config_sha256") != manifest["pair_manifest"]["config_sha256"]:
        raise RuntimeError("pair manifest contract changed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", type=Path)
    result = verify(parser.parse_args().manifest.resolve())
    print(json.dumps({"status": "pass", "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__": main()
