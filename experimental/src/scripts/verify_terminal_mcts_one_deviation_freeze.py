#!/usr/bin/env python3
"""Fail closed unless every frozen Cycle 1b artifact matches its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    actual_manifest_sha = hashlib.sha256(raw).hexdigest()
    if actual_manifest_sha != args.expected_manifest_sha256:
        raise SystemExit("frozen manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema") != "metagross-terminal-mcts-one-deviation-freeze/v1":
        raise SystemExit("wrong frozen-manifest schema")
    if manifest.get("execution_started") is not False:
        raise SystemExit("manifest no longer represents a pre-game freeze")
    for artifact in manifest.get("artifacts", []):
        path = (args.root / artifact["path"]).resolve()
        try:
            path.relative_to(args.root.resolve())
        except ValueError as exc:
            raise SystemExit("manifest artifact escapes repository root") from exc
        data = path.read_bytes()
        if artifact.get("hash_mode") == "normalize_manifest_sha_placeholder":
            data = re.sub(
                rb'EXPECTED_MANIFEST_SHA256="[0-9a-f_]+"',
                b'EXPECTED_MANIFEST_SHA256="__MANIFEST_SHA256__"',
                data,
                count=1,
            )
        elif artifact.get("hash_mode", "raw") != "raw":
            raise SystemExit(f"unknown artifact hash mode: {artifact['path']}")
        if len(data) != artifact["bytes"]:
            raise SystemExit(f"artifact size mismatch: {artifact['path']}")
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise SystemExit(f"artifact SHA-256 mismatch: {artifact['path']}")
    print(actual_manifest_sha)


if __name__ == "__main__":
    main()
