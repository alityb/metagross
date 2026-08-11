#!/usr/bin/env python3
"""Launch the frozen N=100 evaluation only after exact preflight verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

from srcs.metagross.adaptive_ensemble_local_preflight import preflight
from srcs.metagross.h2h_audit import _sha256


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve()
    authorization_path = args.authorization.expanduser().resolve()
    preflight_path = args.preflight.expanduser().resolve()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    stored_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    run_root = root / "experimental" / "runs" / "search_native_stage2_20260809"
    runtime_manifest_path = run_root / "showdown-runtime-identity-v3.json"
    python_manifest_path = run_root / "python-runtime-identity-v1.json"
    remote_preflight_path = run_root / prereg["artifacts"]["remote_preflight"]
    live_preflight = preflight(
        prereg_path,
        authorization_path,
        runtime_manifest_path,
        python_manifest_path,
        root / "external" / "pokemon-showdown",
        root / prereg["artifacts"]["pair_directory"],
        [prereg["prior_servers"][mode]["url"] for mode in ("candidate", "comparator")],
        [prereg["prior_servers"][mode]["nonce"] for mode in ("candidate", "comparator")],
        remote_preflight_path,
    )
    launch_path = run_root / prereg["artifacts"]["showdown_launch"]
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    try:
        os.kill(int(launch["pid"]), 0)
        with socket.create_connection((launch["host"], int(launch["port"])), timeout=2):
            pass
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("frozen Showdown process is not live") from exc
    showdown_exact = (
        launch.get("argv") == prereg["execution"]["showdown"]["argv"]
        and launch.get("environment") == prereg["execution"]["showdown_environment"]
        and launch.get("runtime_manifest_sha256") == _sha256(runtime_manifest_path)
        and launch.get("supervisor_source_sha256")
        == prereg["source_identity"]["showdown_supervisor.py"]
        and launch.get("ready") is True
    )
    if (
        authorization.get("status") != "authorized"
        or authorization.get("preregistration_sha256") != _sha256(prereg_path)
        or authorization.get("local_n100_authorized") is not True
        or authorization.get("public_ladder_authorized") is not False
        or stored_preflight.get("passed") is not True
        or stored_preflight.get("preregistration_sha256") != _sha256(prereg_path)
        or stored_preflight.get("authorization_artifact", {}).get("sha256")
        != _sha256(authorization_path)
        or not all(stored_preflight.get("gates", {}).values())
        or live_preflight.get("passed") is not True
        or not all(live_preflight.get("gates", {}).values())
        or live_preflight.get("prior_servers") != stored_preflight.get("prior_servers")
        or not showdown_exact
        or prereg.get("source_identity", {}).get("n100_runner.py")
        != _sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("N=100 runner identity or authorization check failed")
    frozen = prereg["execution"]["evaluation_argv"]
    executable = (root / frozen[0]).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    environment = prereg["execution"]["environment"]
    os.execve(str(executable), [str(executable), *frozen[1:]], environment)


if __name__ == "__main__":
    main()
