#!/usr/bin/env python3
"""Launch the accepted r1 prior server and ladder client."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def wait_for_health(url: str, process: subprocess.Popen, timeout: int = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"prior server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if json.loads(response.read()).get("ok") is True:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"prior server did not become healthy within {timeout}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--port", type=int, default=8977)
    parser.add_argument(
        "--websocket-uri",
        default="wss://sim3.psim.us/showdown/websocket",
    )
    parser.add_argument(
        "--metamon-python",
        type=Path,
        default=ROOT / ".venv-metamon" / "bin" / "python",
    )
    parser.add_argument(
        "--foul-play-python",
        type=Path,
        default=ROOT / ".venv-fp-priors" / "bin" / "python",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "srcs" / "models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
    if not password:
        raise RuntimeError("set METAGROSS_SHOWDOWN_PASSWORD before launching")
    if args.games <= 0:
        raise ValueError("--games must be positive")

    prior_url = f"http://127.0.0.1:{args.port}"
    common_env = os.environ.copy()
    common_env.update(
        {
            "METAMON_CACHE_DIR": str(ROOT / "srcs" / "runtime" / "metamon-cache"),
            "TORCHDYNAMO_DISABLE": "1",
            "ACCELERATE_USE_CPU": "true",
            "WANDB_MODE": "disabled",
        }
    )
    prior = subprocess.Popen(
        [
            str(args.metamon_python),
            "-u",
            str(ROOT / "srcs" / "metagross" / "prior_server.py"),
            "--local-run-dir",
            str(args.checkpoint_root),
            "--local-run-name",
            "randbats_exit_r1",
            "--checkpoint",
            "5",
            "--port",
            str(args.port),
            "--username",
            args.username,
        ],
        cwd=ROOT,
        env=common_env,
    )

    client = None
    try:
        wait_for_health(prior_url, prior)
        client_env = os.environ.copy()
        client_env.update(
            {
                "FOUL_PLAY_DIR": str(ROOT / "srcs" / "vendor" / "foul-play"),
                "METAGROSS_PRIOR_SERVER": prior_url,
                "METAGROSS_CPUCT": "2.0",
                "METAGROSS_REQUIRE_PRIORS": "1",
            }
        )
        client = subprocess.Popen(
            [
                str(args.foul_play_python),
                "-u",
                str(ROOT / "srcs" / "metagross" / "run_foul_play.py"),
                "--websocket-uri",
                args.websocket_uri,
                "--ps-username",
                args.username,
                "--bot-mode",
                "search_ladder",
                "--pokemon-format",
                "gen9randombattle",
                "--run-count",
                str(args.games),
                "--search-time-ms",
                "500",
                "--search-parallelism",
                "8",
                "--search-threads",
                "1",
                "--log-level",
                "INFO",
            ],
            cwd=ROOT,
            env=client_env,
        )
        raise SystemExit(client.wait())
    finally:
        for process in (client, prior):
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in (client, prior):
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
