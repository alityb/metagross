#!/usr/bin/env python3
"""Build the exact local Showdown identity consumed by the verified supervisor."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from experimental.src.eval.experiment_manifest import hash_tree
from srcs.metagross.h2h_audit import _sha256


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    showdown = ROOT / "external/pokemon-showdown"
    files = {
        "package_json": showdown / "package.json",
        "package_lock": showdown / "package-lock.json",
        "launcher": showdown / "pokemon-showdown",
        "ladders_source": showdown / "server/ladders.ts",
        "registration_source": showdown / "server/eval-pair-registration.ts",
        "ladders_runtime": showdown / "dist/server/ladders.js",
        "registration_runtime": showdown / "dist/server/eval-pair-registration.js",
        "config": showdown / "config/config.js",
        "team_generator": ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
    }
    node = Path(shutil.which("node") or "").resolve()
    if not node.is_file():
        raise RuntimeError("Node executable is unavailable")
    identity = json.loads(subprocess.check_output([
        str(node), "-p",
        "JSON.stringify({node:process.version,v8:process.versions.v8,modules:process.versions.modules,platform:process.platform,arch:process.arch})",
    ], text=True))
    identity.update({"executable": str(node), "executable_sha256": _sha256(node)})
    payload = {
        "schema": "metagross-cycle21-showdown-runtime/v1",
        "files": {name: _sha256(path) for name, path in files.items()},
        "trees": {"dist": hash_tree(showdown / "dist"), "node_modules": hash_tree(showdown / "node_modules")},
        "node": identity,
        "execution_constraints": {
            "argv": [str(node), "pokemon-showdown", "start", "--no-security", "--skip-build", "8010"],
            "environment_required": "METAGROSS_EVAL_PAIR_DIR",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "written", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
