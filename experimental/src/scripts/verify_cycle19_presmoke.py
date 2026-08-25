#!/usr/bin/env python3
"""Verify Cycle 19 immutable presmoke artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experimental.src.scripts.run_cycle8_replay_audit import tree_sha256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        source = Path(row["path"])
        if not source.is_file() or sha(source) != row["sha256"]:
            raise RuntimeError(f"Cycle19 presmoke frozen file mismatch: {source}")
    showdown = manifest["showdown"]
    repo = Path(showdown["path"])
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != showdown["commit"] or tree_sha256(repo / "dist") != showdown["dist_tree_sha256"]:
        raise RuntimeError("Cycle19 Showdown runtime changed")
    return manifest


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]).resolve())
    print(json.dumps({"status": "pass", "schema": result["schema"]}, sort_keys=True))
