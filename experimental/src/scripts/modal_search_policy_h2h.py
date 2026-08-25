#!/usr/bin/env python3
"""Distributed, counterbalanced search-policy H2H on Modal CPU workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 3 and (SCRIPT_PATH.parents[3] / "srcs").is_dir()
ROOT = SCRIPT_PATH.parents[3] if IS_LOCAL_CHECKOUT else Path("/workspace")
AMAGO = ROOT / ".venv-metamon" / "lib" / "python3.11" / "site-packages" / "amago"
SHOWDOWN_COMMIT = "4880d3693580bd33652797cf31179c6fcdf87e50"
VOLUME_NAME = "metagross-online-rl"
PERSIST_ROOT = Path("/data/search_policy_h2h")
PRODUCTION_RUN_SEED = "2026081320260813202608132026081320260813202608132026081320260813"

PROFILES = {
    "action": {
        "root": "/data/search_policy_student/checkpoints",
        "run_name": "search_policy_action_1k_seed20260812",
        "checkpoint": 1,
        "sha256": "efeda02ece3c4fd2f28172450b3ea0d6488bc33447308a7d0140e24962422d6f",
    },
    "visits": {
        "root": "/data/search_policy_student/checkpoints",
        "run_name": "search_policy_visits_1k_seed20260812",
        "checkpoint": 1,
        "sha256": "a2b27bebcabea97a9ec01a84a3f371c7d7902e64b1eb302668a05f27b39d3c34",
    },
    "r1": {
        "root": "/data/accepted",
        "run_name": "randbats_exit_r1",
        "checkpoint": 5,
        "sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
    },
}

APP = modal.App("metagross-search-policy-h2h")
app = APP
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "curl", "git", "nodejs", "npm")
        .pip_install(
            "accelerate",
            "datasets",
            "einops",
            "gin-config",
            "gymnasium==0.29.1",
            "huggingface_hub",
            "lz4",
            "maturin>=1.0,<2.0",
            "numpy==2.4.6",
            "pandas",
            "ratarmountcore",
            "rich",
            "scipy",
            "termcolor",
            "torch==2.12.1",
            "tqdm",
            "wandb",
            "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git@e1268d270c3f2bd32c7ff5713e01062302020579",
            "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
            "git clone https://github.com/smogon/pokemon-showdown.git /workspace/external/pokemon-showdown",
            f"cd /workspace/external/pokemon-showdown && git checkout {SHOWDOWN_COMMIT}",
            "cd /workspace/external/pokemon-showdown && npm ci && npm run build",
            "mkdir -p /workspace/external/pokemon-showdown/databases /workspace/external/pokemon-showdown/logs/repl /workspace/external/pokemon-showdown/logs/modlog /workspace/external/pokemon-showdown/logs/ladder",
        )
    )
    if not IS_LOCAL_CHECKOUT:
        return image
    return (
        image.add_local_dir(
            ROOT / "srcs" / "metagross",
            "/workspace/srcs/metagross",
            copy=True,
            ignore=["__pycache__", "*.pyc", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "metamon" / "metamon",
            "/usr/local/lib/python3.11/site-packages/metamon",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            AMAGO,
            "/usr/local/lib/python3.11/site-packages/amago",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "foul-play",
            "/workspace/srcs/vendor/foul-play",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "external", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "poke-engine",
            "/workspace/srcs/vendor/poke-engine",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "linux_wheels", "release", "target"],
        )
        .add_local_dir(
            ROOT / "experimental" / "src" / "eval",
            "/workspace/experimental/src/eval",
            copy=True,
            ignore=["__pycache__", "*.pyc", "tests"],
        )
        .add_local_file(
            ROOT / "experimental" / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs",
            "/workspace/experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
            copy=True,
        )
        .add_local_file(
            ROOT / "experimental" / "src" / "scripts" / "summarize_counterbalanced_ab.py",
            "/workspace/experimental/src/scripts/summarize_counterbalanced_ab.py",
            copy=True,
        )
        .run_commands(
            "python -m venv --system-site-packages /workspace/.venv-foul-play",
            "/workspace/.venv-foul-play/bin/python -m pip install maturin requests==2.33.0 websockets==15.0.1 python-dateutil==2.8.0",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/metagross-poke-engine /workspace/.venv-foul-play/bin/python -m pip install --no-cache-dir /workspace/srcs/vendor/poke-engine --config-settings='build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization'",
            "rm -rf /tmp/metagross-poke-engine",
        )
    )


IMAGE = _build_image()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(profile: dict[str, object]) -> Path:
    return (
        Path(str(profile["root"]))
        / str(profile["run_name"])
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{profile['checkpoint']}.pt"
    )


def _persistent_unit_path(run_id: str, unit_index: int) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id must contain only letters, numbers, hyphens, and underscores")
    return PERSIST_ROOT / run_id / "units" / f"{unit_index:04d}.json"


def _validated_unit(row: object, payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(row, dict):
        raise RuntimeError("persisted unit is not an object")
    expected = {
        "candidate": str(payload["candidate"]),
        "comparator": str(payload["comparator"]),
        "unit_index": int(payload["unit_index"]),
        "mirror_seed": int(payload["mirror_seed"]),
        "games": 4,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError("persisted unit does not match its requested identity")
    wins = int(row.get("candidate_wins", -1))
    losses = int(row.get("comparator_wins", -1))
    if wins < 0 or losses < 0 or wins + losses != 4:
        raise RuntimeError("persisted unit has an invalid result")
    return row


def _commit_unit(run_id: str, payload: dict[str, object], row: dict[str, object]) -> None:
    destination = _persistent_unit_path(run_id, int(payload["unit_index"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    VOLUME.commit()


def _wait_for_port(process: subprocess.Popen, port: int, timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service on port {port} exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"service on port {port} did not become ready")


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _start_prior(profile_id: str, port: int, output: Path) -> subprocess.Popen:
    profile = PROFILES[profile_id]
    checkpoint = _checkpoint_path(profile)
    actual = _sha256(checkpoint)
    if actual != profile["sha256"]:
        raise RuntimeError(f"{profile_id} checkpoint hash mismatch: {actual}")
    environment = os.environ.copy()
    environment.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
        ACCELERATE_USE_CPU="true",
    )
    command = [
        sys.executable,
        "-u",
        "/workspace/srcs/metagross/prior_server.py",
        "--local-run-dir", str(profile["root"]),
        "--local-run-name", str(profile["run_name"]),
        "--local-base-model", "Kakuna",
        "--checkpoint", str(profile["checkpoint"]),
        "--checkpoint-sha256", str(profile["sha256"]),
        "--username", profile_id,
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    return subprocess.Popen(
        command,
        cwd="/workspace",
        env=environment,
        stdout=output.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_orientation(
    root: Path,
    name: str,
    mirror_seed: int,
    first_port: int,
    second_port: int,
) -> dict[str, object]:
    output = root / name / "result.json"
    logs = root / name / "logs"
    registrations = root / name / "registrations"
    logs.mkdir(parents=True, exist_ok=True)
    registrations.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "/workspace/experimental/src/eval/run.py",
        "--mode", "h2h",
        "--server", "local",
        "--format", "gen9randombattle",
        "--websocket-uri", "ws://localhost:8011/showdown/websocket",
        "--paired",
        "--mirrored-pairs",
        "--mirror-seed", str(mirror_seed),
        "--showdown-dir", "/workspace/external/pokemon-showdown",
        "--mirrored-team-generator", "/workspace/experimental/src/scripts/generate_mirrored_randbats_pair.cjs",
        "--pair-registration-dir", str(registrations),
        "--agent-a", "production_r1_search_first",
        "--agent-b", "production_r1_search_first",
        "--agent-a-prior-server-url", f"http://127.0.0.1:{first_port}",
        "--agent-b-prior-server-url", f"http://127.0.0.1:{second_port}",
        "--agent-a-require-priors",
        "--agent-b-require-priors",
        "--strict-isolated-priors",
        "--foul-play-python", "/workspace/.venv-foul-play/bin/python",
        "--foul-play-search-time-ms", "500",
        "--foul-play-search-parallelism", "8",
        "--foul-play-search-threads", "1",
        "--cpuct", "2.0",
        "--production-run-seed", PRODUCTION_RUN_SEED,
        "--concurrent-games", "1",
        "--fail-fast",
        "--game-timeout-seconds", "900",
        "--n-games", "2",
        "--username-prefix", name[:4],
        "--run-id", f"search_policy_cloud_{name}_{mirror_seed}",
        "--json-out", str(output),
        "--log-dir", str(logs),
    ]
    with (root / name / "eval.log").open("w") as eval_log:
        subprocess.run(
            command,
            check=True,
            cwd="/workspace",
            timeout=2400,
            stdout=eval_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return json.loads(output.read_text(encoding="utf-8"))


@APP.function(
    image=IMAGE,
    cpu=(16.0, 16.0),
    memory=(24576, 24576),
    timeout=3600,
    max_containers=32,
    volumes={"/data": VOLUME},
)
def run_counterbalanced_unit(payload: dict[str, object]) -> dict[str, object]:
    candidate = str(payload["candidate"])
    comparator = str(payload["comparator"])
    unit_index = int(payload["unit_index"])
    mirror_seed = int(payload["mirror_seed"])
    run_id = str(payload["run_id"])
    if candidate not in PROFILES or comparator not in PROFILES or candidate == comparator:
        raise ValueError("invalid H2H profiles")
    persisted = _persistent_unit_path(run_id, unit_index)
    if persisted.exists():
        return _validated_unit(json.loads(persisted.read_text(encoding="utf-8")), payload)
    root = Path(tempfile.mkdtemp(prefix=f"search-policy-h2h-{unit_index:04d}-"))
    showdown_log = (root / "showdown.log").open("w")
    showdown = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "8011"],
        cwd="/workspace/external/pokemon-showdown",
        stdout=showdown_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    prior_candidate = None
    prior_comparator = None
    started = time.monotonic()
    try:
        _wait_for_port(showdown, 8011)
        prior_candidate = _start_prior(candidate, 8977, root / "prior-candidate.log")
        prior_comparator = _start_prior(comparator, 8978, root / "prior-comparator.log")
        _wait_for_port(prior_candidate, 8977)
        _wait_for_port(prior_comparator, 8978)
        candidate_as_a = _run_orientation(root, "candidate-as-a", mirror_seed, 8977, 8978)
        candidate_as_b = _run_orientation(root, "candidate-as-b", mirror_seed, 8978, 8977)
    # Do not intercept BaseException: Modal uses interruption to preempt and
    # transparently retry workers with the same input.
    except Exception as error:
        diagnostic_paths = [
            root / "showdown.log",
            root / "prior-candidate.log",
            root / "prior-comparator.log",
            root / "candidate-as-a" / "eval.log",
            root / "candidate-as-b" / "eval.log",
        ]
        diagnostics = {
            str(path.relative_to(root)): path.read_text(errors="replace")[-12000:]
            for path in diagnostic_paths
            if path.exists()
        }
        raise RuntimeError(json.dumps({"error": repr(error), "logs": diagnostics})) from error
    finally:
        _stop(prior_candidate)
        _stop(prior_comparator)
        _stop(showdown)
        showdown_log.close()

    games_a = candidate_as_a["games"]
    games_b = candidate_as_b["games"]
    match_fields = (
        "game_index", "pair_index", "pair_leg", "battle_seed",
        "team_1_sha256", "team_2_sha256",
    )
    keys_a = [tuple(game[field] for field in match_fields) for game in games_a]
    keys_b = [tuple(game[field] for field in match_fields) for game in games_b]
    if keys_a != keys_b:
        raise RuntimeError("counterbalanced orientations did not use identical games")
    if any(game["void"] or game["error"] is not None for game in games_a + games_b):
        raise RuntimeError("counterbalanced unit contains a void or error")
    candidate_wins = sum(game["winner"] == "agent_a" for game in games_a)
    candidate_wins += sum(game["winner"] == "agent_b" for game in games_b)
    row = {
        "schema_version": 1,
        "candidate": candidate,
        "comparator": comparator,
        "unit_index": unit_index,
        "mirror_seed": mirror_seed,
        "games": 4,
        "candidate_wins": candidate_wins,
        "comparator_wins": 4 - candidate_wins,
        "elapsed_seconds": time.monotonic() - started,
        "match_keys": [
            {field: game[field] for field in match_fields} for game in games_a
        ],
    }
    _commit_unit(run_id, payload, row)
    return row


def _wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = rate + z * z / (2 * games)
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    return (center - margin) / denominator, (center + margin) / denominator


def _summary(candidate: str, comparator: str, rows: list[dict[str, object]]) -> dict[str, object]:
    games = sum(int(row["games"]) for row in rows)
    wins = sum(int(row["candidate_wins"]) for row in rows)
    low, high = _wilson(wins, games)
    cluster_scores = [int(row["candidate_wins"]) / int(row["games"]) for row in rows]
    if len(cluster_scores) < 2:
        cluster_low, cluster_high = 0.0, 1.0
    else:
        cluster_mean = statistics.mean(cluster_scores)
        cluster_margin = 1.959963984540054 * statistics.stdev(cluster_scores) / math.sqrt(len(cluster_scores))
        cluster_low = max(0.0, cluster_mean - cluster_margin)
        cluster_high = min(1.0, cluster_mean + cluster_margin)
    return {
        "schema_version": 1,
        "candidate": candidate,
        "comparator": comparator,
        "units": len(rows),
        "games": games,
        "candidate_wins": wins,
        "comparator_wins": games - wins,
        "candidate_winrate": wins / games,
        "wilson95": [low, high],
        "cluster_normal95": [cluster_low, cluster_high],
        "candidate_advantage_established": low > 0.5 and cluster_low > 0.5,
        "rows": rows,
    }


@APP.local_entrypoint()
def main(
    candidate: str,
    comparator: str,
    games: int = 20,
    seed: int = 2026081401,
    out: str = "",
    run_id: str = "",
    skip_indices: str = "",
) -> None:
    if candidate not in PROFILES or comparator not in PROFILES or candidate == comparator:
        raise ValueError("candidate and comparator must be distinct known profiles")
    if games <= 0 or games % 4:
        raise ValueError("games must be a positive multiple of four")
    run_id = run_id or f"{candidate}-vs-{comparator}-{games}-{seed}"
    _persistent_unit_path(run_id, 0)
    skipped = {
        int(value)
        for value in skip_indices.split(",")
        if value.strip()
    }
    unit_count = games // 4
    if any(index < 0 or index >= unit_count for index in skipped):
        raise ValueError("skip_indices contains an out-of-range unit")
    payloads = [
        {
            "candidate": candidate,
            "comparator": comparator,
            "unit_index": index,
            "mirror_seed": seed + index,
            "run_id": run_id,
        }
        for index in range(games // 4)
        if index not in skipped
    ]
    rows = list(run_counterbalanced_unit.map(payloads, order_outputs=False))
    rows.sort(key=lambda row: int(row["unit_index"]))
    report = _summary(candidate, comparator, rows)
    if out:
        destination = Path(out).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
