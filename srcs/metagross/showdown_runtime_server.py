#!/usr/bin/env python3
"""Verify and launch the exact loopback Showdown runtime for local evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from experimental.src.eval.experiment_manifest import hash_tree
from srcs.metagross.h2h_audit import _sha256


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--pair-directory", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.runtime_manifest.expanduser().resolve()
    pair_directory = args.pair_directory.expanduser().resolve()
    launch_record = args.launch_record.expanduser().resolve()
    server_log = args.server_log.expanduser().resolve()
    if launch_record.exists() or server_log.exists():
        raise FileExistsError("Showdown launch outputs must not exist")
    root = Path(__file__).resolve().parents[2]
    showdown = root / "external" / "pokemon-showdown"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_paths = {
        "package_json": showdown / "package.json",
        "package_lock": showdown / "package-lock.json",
        "launcher": showdown / "pokemon-showdown",
        "ladders_source": showdown / "server" / "ladders.ts",
        "registration_source": showdown / "server" / "eval-pair-registration.ts",
        "ladders_runtime": showdown / "dist" / "server" / "ladders.js",
        "registration_runtime": showdown / "dist" / "server" / "eval-pair-registration.js",
        "config": showdown / "config" / "config.js",
        "team_generator": root / "experimental" / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs",
    }
    observed_files = {name: _sha256(path) for name, path in file_paths.items()}
    if (
        observed_files != manifest.get("files")
        or hash_tree(showdown / "dist") != manifest.get("trees", {}).get("dist")
        or hash_tree(showdown / "node_modules") != manifest.get("trees", {}).get("node_modules")
        or pair_directory.exists() and any(pair_directory.iterdir())
    ):
        raise RuntimeError("Showdown runtime or pair directory failed launch identity checks")
    node_executable = Path(manifest["node"]["executable"]).resolve()
    if _sha256(node_executable) != manifest["node"]["executable_sha256"]:
        raise RuntimeError("Node executable differs from the runtime manifest")
    try:
        with socket.create_connection(("127.0.0.1", 8010), timeout=1):
            raise RuntimeError("Showdown port 8010 already has a listener")
    except ConnectionRefusedError:
        pass
    except OSError as exc:
        if getattr(exc, "errno", None) not in {61, 111}:
            raise
    pair_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    command = [
        str(node_executable),
        "pokemon-showdown",
        "start",
        "--no-security",
        "--skip-build",
        "8010",
    ]
    environment = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "METAGROSS_EVAL_PAIR_DIR": str(pair_directory),
    }
    with server_log.open("xb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=showdown,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Showdown exited during startup: {process.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", 8010), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Showdown did not bind loopback port 8010")
                    time.sleep(0.25)
            record = {
                "schema_version": 1,
                "mode": "verified_showdown_runtime_launch",
                "supervisor_source_sha256": _sha256(Path(__file__).resolve()),
                "runtime_manifest_sha256": _sha256(manifest_path),
                "pid": process.pid,
                "cwd": str(showdown),
                "argv": command,
                "node_executable_sha256": _sha256(node_executable),
                "host": "127.0.0.1",
                "port": 8010,
                "pair_directory": str(pair_directory),
                "server_log": str(server_log),
                "environment": environment,
                "files": observed_files,
                "dist": hash_tree(showdown / "dist"),
                "node_modules": hash_tree(showdown / "node_modules"),
                "ready": True,
            }
            launch_record.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            def terminate(_signum, _frame):
                process.terminate()

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            returncode = process.wait()
            if returncode not in {0, -signal.SIGTERM, -signal.SIGINT}:
                raise RuntimeError(f"Showdown exited unexpectedly: {returncode}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
