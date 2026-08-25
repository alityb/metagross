#!/usr/bin/env python3
"""Resumable counterbalanced Action-student versus R1 gate on one VM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_RUN_SEED = "2026081320260813202608132026081320260813202608132026081320260813"
PROFILES = {
    "action": {
        "root": ROOT / "experimental" / "releases" / "search_policy_student",
        "run_name": "search_policy_action_1k_seed20260812",
        "checkpoint": 1,
        "sha256": "efeda02ece3c4fd2f28172450b3ea0d6488bc33447308a7d0140e24962422d6f",
    },
    "r1": {
        "root": ROOT / "srcs" / "models",
        "run_name": "randbats_exit_r1",
        "checkpoint": 5,
        "sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(profile: dict[str, object]) -> Path:
    return (
        Path(profile["root"])
        / str(profile["run_name"])
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{profile['checkpoint']}.pt"
    )


def wait_for_port(process: subprocess.Popen, port: int, timeout: float = 240) -> None:
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


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_prior(profile_id: str, port: int, log: Path) -> subprocess.Popen:
    profile = PROFILES[profile_id]
    checkpoint = checkpoint_path(profile)
    actual = sha256(checkpoint)
    if actual != profile["sha256"]:
        raise RuntimeError(f"{profile_id} checkpoint hash mismatch: {actual}")
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT),
        METAMON_CACHE_DIR=str(ROOT / "srcs" / "runtime" / "metamon-cache"),
        HF_HOME=str(ROOT / "srcs" / "runtime" / "hf-home"),
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
        CUDA_VISIBLE_DEVICES="0",
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "srcs" / "metagross" / "prior_server.py"),
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
        cwd=ROOT,
        env=environment,
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_orientation(
    unit_dir: Path,
    name: str,
    mirror_seed: int,
    first_port: int,
    second_port: int,
    showdown_port: int,
) -> dict[str, object]:
    root = unit_dir / name
    output = root / "result.json"
    logs = root / "logs"
    registrations = root / "registrations"
    logs.mkdir(parents=True)
    registrations.mkdir(parents=True)
    command = [
        sys.executable,
        str(ROOT / "experimental" / "src" / "eval" / "run.py"),
        "--mode", "h2h",
        "--server", "local",
        "--format", "gen9randombattle",
        "--websocket-uri", f"ws://localhost:{showdown_port}/showdown/websocket",
        "--paired",
        "--mirrored-pairs",
        "--mirror-seed", str(mirror_seed),
        "--showdown-dir", str(ROOT / "external" / "pokemon-showdown"),
        "--mirrored-team-generator", str(ROOT / "experimental" / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs"),
        "--pair-registration-dir", str(registrations),
        "--agent-a", "production_r1_search_first",
        "--agent-b", "production_r1_search_first",
        "--agent-a-prior-server-url", f"http://127.0.0.1:{first_port}",
        "--agent-b-prior-server-url", f"http://127.0.0.1:{second_port}",
        "--agent-a-require-priors",
        "--agent-b-require-priors",
        "--strict-isolated-priors",
        "--foul-play-python", sys.executable,
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
        "--run-id", f"search_policy_vm_{name}_{mirror_seed}",
        "--json-out", str(output),
        "--log-dir", str(logs),
    ]
    with (root / "eval.log").open("w") as eval_log:
        subprocess.run(
            command,
            check=True,
            cwd=ROOT,
            timeout=2400,
            stdout=eval_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return json.loads(output.read_text(encoding="utf-8"))


def wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = rate + z * z / (2 * games)
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    return (center - margin) / denominator, (center + margin) / denominator


def summary(rows: list[dict[str, object]], requested_games: int) -> dict[str, object]:
    games = sum(int(row["games"]) for row in rows)
    wins = sum(int(row["candidate_wins"]) for row in rows)
    low, high = wilson(wins, games)
    scores = [int(row["candidate_wins"]) / int(row["games"]) for row in rows]
    if len(scores) < 2:
        cluster_low, cluster_high = 0.0, 1.0
    else:
        margin = 1.959963984540054 * statistics.stdev(scores) / math.sqrt(len(scores))
        cluster_low = max(0.0, statistics.mean(scores) - margin)
        cluster_high = min(1.0, statistics.mean(scores) + margin)
    return {
        "schema_version": 1,
        "candidate": "action",
        "comparator": "r1",
        "requested_games": requested_games,
        "complete": games == requested_games,
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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def completed_unit(unit_dir: Path, index: int, mirror_seed: int) -> dict[str, object] | None:
    path = unit_dir / "unit.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (
        row.get("unit_index") != index
        or row.get("mirror_seed") != mirror_seed
        or row.get("games") != 4
        or int(row.get("candidate_wins", -1)) + int(row.get("comparator_wins", -1)) != 4
    ):
        raise RuntimeError(f"invalid completed unit: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--showdown-port", type=int, default=8011)
    parser.add_argument("--candidate-port", type=int, default=8977)
    parser.add_argument("--r1-port", type=int, default=8978)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 4:
        raise ValueError("games must be a positive multiple of four")
    args.run_dir = args.run_dir.expanduser().resolve()
    units_root = args.run_dir / "units"
    units_root.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES.values():
        checkpoint = checkpoint_path(profile)
        if sha256(checkpoint) != profile["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")

    showdown_log = (args.run_dir / "showdown.log").open("a")
    showdown = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--skip-build", "--no-security", str(args.showdown_port)],
        cwd=ROOT / "external" / "pokemon-showdown",
        stdout=showdown_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    candidate = None
    r1 = None
    try:
        wait_for_port(showdown, args.showdown_port)
        candidate = start_prior("action", args.candidate_port, args.run_dir / "prior-action.log")
        r1 = start_prior("r1", args.r1_port, args.run_dir / "prior-r1.log")
        wait_for_port(candidate, args.candidate_port)
        wait_for_port(r1, args.r1_port)
        rows: list[dict[str, object]] = []
        for index in range(args.games // 4):
            mirror_seed = args.seed + index
            unit_dir = units_root / f"{index:04d}"
            row = completed_unit(unit_dir, index, mirror_seed)
            if row is None:
                if unit_dir.exists():
                    shutil.rmtree(unit_dir)
                unit_dir.mkdir()
                started = time.monotonic()
                as_a = run_orientation(
                    unit_dir, "candidate-as-a", mirror_seed,
                    args.candidate_port, args.r1_port, args.showdown_port,
                )
                as_b = run_orientation(
                    unit_dir, "candidate-as-b", mirror_seed,
                    args.r1_port, args.candidate_port, args.showdown_port,
                )
                games_a = as_a["games"]
                games_b = as_b["games"]
                match_fields = (
                    "game_index", "pair_index", "pair_leg", "battle_seed",
                    "team_1_sha256", "team_2_sha256",
                )
                keys_a = [tuple(game[field] for field in match_fields) for game in games_a]
                keys_b = [tuple(game[field] for field in match_fields) for game in games_b]
                if keys_a != keys_b:
                    raise RuntimeError("counterbalanced orientations used different games")
                if any(game["void"] or game["error"] is not None for game in games_a + games_b):
                    raise RuntimeError("counterbalanced unit contains a void or error")
                wins = sum(game["winner"] == "agent_a" for game in games_a)
                wins += sum(game["winner"] == "agent_b" for game in games_b)
                row = {
                    "schema_version": 1,
                    "candidate": "action",
                    "comparator": "r1",
                    "unit_index": index,
                    "mirror_seed": mirror_seed,
                    "games": 4,
                    "candidate_wins": wins,
                    "comparator_wins": 4 - wins,
                    "elapsed_seconds": time.monotonic() - started,
                    "match_keys": [
                        {field: game[field] for field in match_fields} for game in games_a
                    ],
                }
                atomic_json(unit_dir / "unit.json", row)
            rows.append(row)
            report = summary(rows, args.games)
            atomic_json(args.run_dir / "summary.partial.json", report)
            print(
                f"PROGRESS units={len(rows)}/{args.games // 4} "
                f"games={report['games']}/{args.games} "
                f"action_wins={report['candidate_wins']}",
                flush=True,
            )
        atomic_json(args.run_dir / "summary.json", summary(rows, args.games))
    finally:
        stop(candidate)
        stop(r1)
        stop(showdown)
        showdown_log.close()


if __name__ == "__main__":
    main()
