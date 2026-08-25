#!/usr/bin/env python3
"""Verify the frozen Cycle20 pre-smoke manifest."""

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
    payload = json.loads(path.read_text())
    if payload.get("schema") != "metagross-cycle20-presmoke-freeze/v1":
        raise RuntimeError("wrong Cycle20 manifest schema")
    for row in payload.get("files", []):
        source = Path(row["path"])
        if not source.is_file() or sha(source) != row["sha256"]:
            raise RuntimeError(f"Cycle20 frozen file mismatch: {source}")
    showdown = payload["showdown"]
    root = Path(showdown["path"])
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != showdown["commit"] or tree_sha256(root / "dist") != showdown["dist_tree_sha256"]:
        raise RuntimeError("Cycle20 Showdown runtime changed")
    return payload


if __name__ == "__main__":
    manifest = verify(Path(sys.argv[1]).resolve())
    print(json.dumps({"status": "pass", "schema": manifest["schema"]}, sort_keys=True))
