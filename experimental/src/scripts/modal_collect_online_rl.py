"""Collect pinned online-RL trajectories into a persistent Modal Volume."""

from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
from modal.types import FileEntryType


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 3 and (SCRIPT_PATH.parents[3] / "srcs").is_dir()
ROOT = SCRIPT_PATH.parents[3] if IS_LOCAL_CHECKOUT else Path("/workspace")
SCRIPTS = ROOT / "experimental" / "src" / "scripts"
METAMON = ROOT / "srcs" / "vendor" / "metamon" / "metamon"
AMAGO = ROOT / ".venv-metamon" / "lib" / "python3.11" / "site-packages" / "amago"
SHOWDOWN_COMMIT = "4880d3693580bd33652797cf31179c6fcdf87e50"

APP = modal.App("metagross-online-rl-collection")
VOLUME = modal.Volume.from_name("metagross-online-rl", create_if_missing=False)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "nodejs", "npm")
    .pip_install(
        "torch", "numpy", "gymnasium<=0.29.1", "gin-config", "wandb", "einops",
        "tqdm", "lz4", "termcolor", "rich", "huggingface_hub", "datasets",
        "pandas", "scipy", "ratarmountcore",
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
        AMAGO, "/usr/local/lib/python3.11/site-packages/amago", copy=True,
        ignore=["__pycache__", "*.pyc"],
    ).add_local_dir(
        METAMON, "/usr/local/lib/python3.11/site-packages/metamon", copy=True,
        ignore=["__pycache__", "*.pyc"],
    ).add_local_dir(
        SCRIPTS, "/workspace/experimental/src/scripts", copy=True,
        ignore=["__pycache__", "*.pyc", "tests"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_port(process: subprocess.Popen[bytes], port: int, timeout: float = 60) -> None:
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


def _download_collection(
    collection_id: str, destination: Path, manifest: dict[str, object]
) -> None:
    volume = modal.Volume.from_name("metagross-online-rl", create_if_missing=False)
    remote_root = Path("online_rl/collections") / collection_id
    destination.mkdir(parents=True, exist_ok=True)

    def retry(operation):
        for attempt in range(8):
            try:
                return operation()
            except Exception as error:
                if "rate limit" not in str(error).lower() or attempt == 7:
                    raise
                time.sleep(min(30, 2 ** attempt))

    def list_directory(path: str):
        return retry(lambda: list(volume.iterdir(path, recursive=False)))

    def download(remote_path: str, size: int | None = None) -> None:
        relative = Path(remote_path).relative_to(remote_root)
        target = destination / relative
        if size is not None and target.is_file() and target.stat().st_size == size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")

        def transfer() -> None:
            with temporary.open("wb") as handle:
                for payload in volume.read_file(f"/{remote_path}"):
                    handle.write(payload)
            temporary.replace(target)

        retry(transfer)

    chunks = manifest.get("chunks")
    learner = manifest.get("learner")
    if not isinstance(chunks, list) or not isinstance(learner, str):
        raise RuntimeError("collection manifest cannot drive trajectory transfer")
    phase_directories = []
    for chunk in chunks:
        chunk_dir = f"chunk_{chunk['chunk_index']:05d}_{learner}_vs_{chunk['opponent']}"
        for phase in chunk.get("phases", []):
            phase_directories.append(
                f"/{remote_root}/{chunk_dir}/{phase['phase']}/learner_trajectories/gen9randombattle"
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        directory_entries = list(executor.map(
            list_directory, phase_directories
        ))
    trajectories = [
        entry for entries in directory_entries for entry in entries
        if entry.type == FileEntryType.FILE and entry.path.endswith(".lz4")
    ]
    if len(trajectories) != manifest.get("learner_trajectory_count"):
        raise RuntimeError("remote collection trajectory total does not match manifest")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(
            lambda entry: download(entry.path, entry.size), trajectories
        ))
    for name in ("MANIFEST.json", "BATTLE_LEDGER.jsonl", "pool.json", "schedule.json"):
        download(f"{remote_root}/{name}")
    local_count = sum(1 for path in destination.glob("**/*.lz4") if path.is_file())
    if local_count != manifest.get("learner_trajectory_count"):
        raise RuntimeError("downloaded collection trajectory total does not match manifest")


@APP.function(
    image=IMAGE, gpu="L4", cpu=16, memory=32768, timeout=21600,
    volumes={"/data": VOLUME},
)
def collect(collection_id: str, workers: int, chunk_games: int, torch_threads: int) -> dict[str, object]:
    root = Path("/data/online_rl/collections") / collection_id
    pool_path = root / "pool.json"
    schedule_path = root / "schedule.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    for profile in pool["profiles"].values():
        if profile.get("kind", "local") == "local":
            profile["run_dir"] = "/data/online_rl/checkpoints"
    pool_path.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    showdown_log = (root / "showdown.log").open("ab")
    showdown = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "8011"],
        cwd="/workspace/external/pokemon-showdown", stdout=showdown_log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(showdown, 8011)
        subprocess.run([
            "/usr/local/bin/python", "/workspace/experimental/src/scripts/online_rl_generate.py",
            "--pool", str(pool_path), "--schedule", str(schedule_path),
            "--out-dir", str(root), "--showdown-port", "8011",
            "--workers", str(workers), "--chunk-games", str(chunk_games),
            "--torch-threads", str(torch_threads), "--collection-kind", "fresh",
            "--python", "/usr/local/bin/python",
        ], check=True, cwd="/workspace/experimental/src")
    finally:
        showdown.terminate()
        try:
            showdown.wait(timeout=15)
        except subprocess.TimeoutExpired:
            showdown.kill()
        showdown_log.close()
        VOLUME.commit()

    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("failed_shards") or manifest.get("completed_battles") != manifest.get("requested_battles"):
        raise RuntimeError("persistent Modal collection was incomplete")
    return manifest


@APP.local_entrypoint()
def main(pool: str, schedule: str, collection_id: str, local_out: str = "", workers: int = 4, chunk_games: int = 25, torch_threads: int = 2) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", collection_id):
        raise ValueError("collection_id contains unsafe characters")
    if min(workers, chunk_games, torch_threads) <= 0:
        raise ValueError("worker, chunk, and thread counts must be positive")
    pool_path = Path(pool).resolve()
    schedule_path = Path(schedule).resolve()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    remote_root = f"/online_rl/collections/{collection_id}"
    with VOLUME.batch_upload(force=True) as batch:
        batch.put_file(str(pool_path), f"{remote_root}/pool.json")
        batch.put_file(str(schedule_path), f"{remote_root}/schedule.json")
        uploaded: set[tuple[str, int, str]] = set()
        for profile in payload["profiles"].values():
            if profile.get("kind", "local") != "local":
                continue
            key = (
                str(profile["run_name"]), int(profile["checkpoint"]),
                str(profile["checkpoint_sha256"]).lower(),
            )
            if key in uploaded:
                continue
            uploaded.add(key)
            checkpoint = (
                Path(profile["run_dir"]).expanduser().resolve() / profile["run_name"] /
                "ckpts" / "policy_weights" / f"policy_epoch_{profile['checkpoint']}.pt"
            )
            if _sha256(checkpoint) != str(profile["checkpoint_sha256"]).lower():
                raise ValueError(f"checkpoint SHA-256 mismatch: {checkpoint}")
            batch.put_file(
                str(checkpoint),
                f"/online_rl/checkpoints/{profile['run_name']}/ckpts/policy_weights/{checkpoint.name}",
            )
    report = collect.remote(collection_id, workers, chunk_games, torch_threads)
    if local_out:
        _download_collection(
            collection_id,
            Path(local_out).resolve(),
            report,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
