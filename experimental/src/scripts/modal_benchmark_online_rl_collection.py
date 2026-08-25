"""Benchmark concurrent online-RL self-play collection on a Modal GPU."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 3 and (SCRIPT_PATH.parents[3] / "srcs").is_dir()
ROOT = SCRIPT_PATH.parents[3] if IS_LOCAL_CHECKOUT else Path("/workspace")
SCRIPTS = ROOT / "experimental" / "src" / "scripts"
METAMON = ROOT / "srcs" / "vendor" / "metamon" / "metamon"
AMAGO = ROOT / ".venv-metamon" / "lib" / "python3.11" / "site-packages" / "amago"
SHOWDOWN_COMMIT = "4880d3693580bd33652797cf31179c6fcdf87e50"
G4_RUN = "randbats_online_g4_autonomous_freshfix_20260729"
G4_SHA256 = "cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505"

APP = modal.App("metagross-online-rl-collection-benchmark")
VOLUME = modal.Volume.from_name("metagross-online-rl", create_if_missing=False)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "nodejs", "npm")
    .pip_install(
        "torch",
        "numpy",
        "gymnasium<=0.29.1",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "lz4",
        "termcolor",
        "rich",
        "huggingface_hub",
        "datasets",
        "pandas",
        "scipy",
        "ratarmountcore",
        "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git",
        "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
    )
    .run_commands(
        "git clone https://github.com/smogon/pokemon-showdown.git /workspace/external/pokemon-showdown",
        f"cd /workspace/external/pokemon-showdown && git checkout {SHOWDOWN_COMMIT}",
        "cd /workspace/external/pokemon-showdown && npm ci && npm run build",
        "mkdir -p /workspace/external/metamon_cache /workspace/external/pokemon-showdown/databases",
    )
)
if IS_LOCAL_CHECKOUT:
    IMAGE = IMAGE.add_local_dir(
        AMAGO,
        "/usr/local/lib/python3.11/site-packages/amago",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    ).add_local_dir(
        METAMON,
        "/usr/local/lib/python3.11/site-packages/metamon",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    ).add_local_dir(
        SCRIPTS,
        "/workspace/experimental/src/scripts",
        copy=True,
        ignore=["__pycache__", "*.pyc", "tests"],
    )


def wait_for_port(process: subprocess.Popen[bytes], port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Pokemon Showdown exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Pokemon Showdown did not become ready")


@APP.function(
    image=IMAGE,
    gpu="L4",
    cpu=16,
    memory=32768,
    timeout=3600,
    volumes={"/data": VOLUME},
)
def benchmark(games: int = 100, workers: int = 4, chunk_games: int = 25) -> dict[str, object]:
    if games <= 0 or workers <= 0 or chunk_games <= 0:
        raise ValueError("games, workers, and chunk_games must be positive")
    root = Path("/tmp/online_rl_collection_benchmark")
    output = root / "collection"
    root.mkdir(parents=True, exist_ok=True)
    profile = {
        "kind": "local",
        "run_dir": "/data/online_rl/checkpoints",
        "run_name": G4_RUN,
        "checkpoint": 1,
        "checkpoint_sha256": G4_SHA256,
        "base_model": "Kakuna",
        "temperature": 1.0,
        "alias_to": "",
    }
    pool = {
        "schema_version": 1,
        "format": "gen9randombattle",
        "profiles": {"online_g4_frozen": profile},
        "pfsp": {
            "learner": "online_g4_frozen",
            "pool": [{"id": "online_g4_frozen", "base_weight": 1.0}],
            "target_winrate": [0.4, 0.6],
            "min_pool_weight": 0.05,
        },
    }
    schedule = {
        "schema_version": 1,
        "format": "gen9randombattle",
        "learner": "online_g4_frozen",
        "opponents": ["online_g4_frozen"] * games,
    }
    pool_path = root / "pool.json"
    schedule_path = root / "schedule.json"
    pool_path.write_text(json.dumps(pool), encoding="utf-8")
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

    showdown_dir = Path("/workspace/external/pokemon-showdown")
    showdown_log = (root / "showdown.log").open("wb")
    showdown = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "8011"],
        cwd=showdown_dir,
        stdout=showdown_log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_port(showdown, 8011)
        started = time.monotonic()
        command = [
                "/usr/local/bin/python",
                "/workspace/experimental/src/scripts/online_rl_generate.py",
                "--pool", str(pool_path),
                "--schedule", str(schedule_path),
                "--out-dir", str(output),
                "--showdown-port", "8011",
                "--workers", str(workers),
                "--chunk-games", str(chunk_games),
                "--torch-threads", "2",
                "--collection-kind", "fresh",
                "--python", "/usr/local/bin/python",
            ]
        try:
            subprocess.run(command, check=True, cwd="/workspace/experimental/src")
        except subprocess.CalledProcessError as error:
            logs = {
                str(path.relative_to(output)): path.read_text(errors="replace")[-8000:]
                for path in output.glob("**/*.out")
            }
            raise RuntimeError(json.dumps({"command": command, "logs": logs}, indent=2)) from error
        elapsed = time.monotonic() - started
    finally:
        showdown.terminate()
        try:
            showdown.wait(timeout=15)
        except subprocess.TimeoutExpired:
            showdown.kill()
        showdown_log.close()

    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    completed = int(manifest["completed_battles"])
    if completed != games or manifest.get("failed_shards"):
        raise RuntimeError("benchmark collection was incomplete")
    return {
        "games": completed,
        "workers": workers,
        "chunk_games": chunk_games,
        "elapsed_seconds": elapsed,
        "games_per_minute": completed / elapsed * 60,
        "learner_trajectory_count": manifest["learner_trajectory_count"],
        "failed_shards": manifest["failed_shards"],
    }


@APP.local_entrypoint()
def main(games: int = 100, workers: int = 4, chunk_games: int = 25) -> None:
    print(json.dumps(benchmark.remote(games, workers, chunk_games), indent=2, sort_keys=True))
