#!/usr/bin/env python3
"""Collect current-policy trajectories against a reproducible PFSP schedule.

This is deliberately a staged collector, not a continuously updating actor.
Every shard pins both checkpoints for its entire lifetime and writes learner-POV
Metamon trajectories plus a manifest suitable for the next training generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SRC_ROOT.parents[1]
RUNNER = SRC_ROOT / "scripts" / "run_kakuna_challenge.py"
DEFAULT_PYTHON = WORKSPACE_ROOT / ".venv-metamon" / "bin" / "python"


class CollectionError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_workspace_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve())


def validate_profile(profile_id: str, profile: dict[str, Any]) -> None:
    if profile.get("kind", "local") not in {"local", "pretrained"}:
        raise CollectionError(f"profile {profile_id!r} has unsupported kind")
    if profile.get("kind", "local") == "local":
        missing = [
            key
            for key in ("run_dir", "run_name", "checkpoint", "checkpoint_sha256")
            if profile.get(key) is None
        ]
        if missing:
            raise CollectionError(f"profile {profile_id!r} is missing {', '.join(missing)}")
    elif not profile.get("agent"):
        raise CollectionError(f"pretrained profile {profile_id!r} requires agent")


def verify_local_checkpoints(profiles: dict[str, dict[str, Any]], profile_ids: set[str]) -> None:
    verified: set[tuple[Path, str]] = set()
    for profile_id in sorted(profile_ids):
        profile = profiles[profile_id]
        if profile.get("kind", "local") != "local":
            continue
        checkpoint = (
            Path(resolve_workspace_path(str(profile["run_dir"])))
            / str(profile["run_name"])
            / "ckpts"
            / "policy_weights"
            / f"policy_epoch_{profile['checkpoint']}.pt"
        )
        expected = str(profile["checkpoint_sha256"]).lower()
        key = (checkpoint, expected)
        if key in verified:
            continue
        if not checkpoint.is_file():
            raise CollectionError(f"profile {profile_id!r} checkpoint is missing: {checkpoint}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise CollectionError(f"profile {profile_id!r} checkpoint SHA-256 does not match")
        verified.add(key)


def load_plan(pool_path: Path, schedule_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    profiles = pool.get("profiles")
    learner = schedule.get("learner")
    opponents = schedule.get("opponents")
    if not isinstance(profiles, dict) or learner not in profiles:
        raise CollectionError("pool does not contain the scheduled learner profile")
    if not isinstance(opponents, list) or not opponents:
        raise CollectionError("schedule must contain at least one opponent")
    for profile_id in {learner, *opponents}:
        if profile_id not in profiles:
            raise CollectionError(f"schedule references unknown profile {profile_id!r}")
        validate_profile(profile_id, profiles[profile_id])
    return pool, schedule


def policy_command(
    *,
    python: str,
    profile: dict[str, Any],
    username: str,
    opponent_username: str,
    role: str,
    battle_format: str,
    battles: int,
    results_dir: Path,
    trajectories_dir: Path | None,
    showdown_port: int,
) -> list[str]:
    command = [
        python,
        str(RUNNER),
        "--username", username,
        "--opponent-username", opponent_username,
        "--role", role,
        "--battle-format", battle_format,
        "--total-battles", str(battles),
        "--temperature", str(profile.get("temperature", 1.0)),
        "--save-results-to", str(results_dir),
        "--showdown-port", str(showdown_port),
    ]
    # Local r1 was trained on the real gen9randombattle token. Passing an empty
    # alias explicitly overrides the legacy Kakuna runner's gen9ou default.
    alias = profile.get("alias_to", "gen9ou")
    command.extend(["--alias-to", str(alias or "")])
    if trajectories_dir is not None:
        command.extend(["--save-trajectories-to", str(trajectories_dir)])
    if profile.get("kind", "local") == "local":
        command.extend(
            [
                "--local-run-dir", resolve_workspace_path(str(profile["run_dir"])),
                "--local-run-name", str(profile["run_name"]),
                "--local-base-model", str(profile.get("base_model", "Kakuna")),
                "--checkpoint", str(profile["checkpoint"]),
            ]
        )
    else:
        command.extend(["--agent", str(profile["agent"])])
        if profile.get("checkpoint") is not None:
            command.extend(["--checkpoint", str(profile["checkpoint"])])
    return command


def result_rows(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[3].strip().upper() for row in reader if len(row) >= 4]


def result_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def outcome_totals(rows: list[str]) -> tuple[int, int]:
    unknown = sorted(set(rows) - {"WIN", "LOSS"})
    if unknown:
        raise CollectionError(f"results contain ties or unknown outcomes: {unknown}")
    return rows.count("WIN"), rows.count("LOSS")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_until_ready(process: subprocess.Popen, log_path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CollectionError(f"acceptor exited before checkpoint validation; see {log_path}")
        if log_path.is_file() and "Checkpoint validated" in log_path.read_text(errors="ignore"):
            return
        time.sleep(1)
    raise CollectionError(f"acceptor did not validate its checkpoint; see {log_path}")


def run_phase(
    *,
    args: argparse.Namespace,
    learner: dict[str, Any],
    opponent: dict[str, Any],
    opponent_id: str,
    shard_index: int,
    learner_role: str,
    battles: int,
    output: Path,
) -> dict[str, Any]:
    phase = f"learner_{learner_role}"
    phase_dir = output / phase
    results_dir = phase_dir / "results"
    trajectories_dir = phase_dir / "learner_trajectories"
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(
        f"{output.resolve()}:{opponent_id}:{shard_index}:{learner_role}".encode()
    ).hexdigest()[:8]
    learner_user = f"orl{token}l"
    opponent_user = f"orl{token}o"
    opponent_role = "challenger" if learner_role == "acceptor" else "acceptor"
    learner_command = policy_command(
        python=args.python,
        profile=learner,
        username=learner_user,
        opponent_username=opponent_user,
        role=learner_role,
        battle_format=args.format,
        battles=battles,
        results_dir=results_dir,
        trajectories_dir=trajectories_dir,
        showdown_port=args.showdown_port,
    )
    opponent_command = policy_command(
        python=args.python,
        profile=opponent,
        username=opponent_user,
        opponent_username=learner_user,
        role=opponent_role,
        battle_format=args.format,
        battles=battles,
        results_dir=phase_dir / "opponent_results",
        trajectories_dir=None,
        showdown_port=args.showdown_port,
    )
    record: dict[str, Any] = {
        "phase": phase,
        "requested_battles": battles,
        "learner_role": learner_role,
        "learner_username": learner_user,
        "opponent_username": opponent_user,
        "learner_command": learner_command,
        "opponent_command": opponent_command,
    }
    if args.dry_run:
        return record

    environment = os.environ | {
        "METAMON_CACHE_DIR": str(WORKSPACE_ROOT / "external" / "metamon_cache"),
        "WANDB_MODE": "disabled",
        "TORCHDYNAMO_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(args.torch_threads),
        "MKL_NUM_THREADS": str(args.torch_threads),
    }
    learner_log_path = phase_dir / "learner.out"
    opponent_log_path = phase_dir / "opponent.out"
    learner_process = opponent_process = None
    try:
        if learner_role == "acceptor":
            with learner_log_path.open("w", encoding="utf-8") as log:
                learner_process = subprocess.Popen(learner_command, cwd=SRC_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            wait_until_ready(learner_process, learner_log_path, args.load_timeout_seconds)
            with opponent_log_path.open("w", encoding="utf-8") as log:
                opponent_process = subprocess.Popen(opponent_command, cwd=SRC_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        else:
            with opponent_log_path.open("w", encoding="utf-8") as log:
                opponent_process = subprocess.Popen(opponent_command, cwd=SRC_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            wait_until_ready(opponent_process, opponent_log_path, args.load_timeout_seconds)
            with learner_log_path.open("w", encoding="utf-8") as log:
                learner_process = subprocess.Popen(learner_command, cwd=SRC_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)

        result_path = results_dir / f"battle_log_{learner_user}_{args.format}.csv"
        deadline = time.monotonic() + args.phase_timeout_seconds
        last_count = 0
        last_progress = time.monotonic()
        while time.monotonic() < deadline:
            rows = result_rows(result_path)
            if len(rows) >= battles:
                break
            if len(rows) > last_count:
                last_count = len(rows)
                last_progress = time.monotonic()
            if learner_process.poll() is not None and opponent_process.poll() is not None:
                break
            if time.monotonic() - last_progress > args.stall_timeout_seconds:
                raise CollectionError(f"{phase} stalled after {last_count}/{battles} battles")
            time.sleep(2)
        rows = result_rows(result_path)
        if len(rows) != battles:
            raise CollectionError(f"{phase} completed {len(rows)}/{battles} battles")
        wins, losses = outcome_totals(rows)
        trajectory_paths = list(trajectories_dir.glob("**/*.lz4"))
        if len(trajectory_paths) != battles:
            raise CollectionError(
                f"{phase} wrote {len(trajectory_paths)}/{battles} learner trajectories"
            )
        if any("_WIN.json.lz4" not in path.name and "_LOSS.json.lz4" not in path.name for path in trajectory_paths):
            raise CollectionError(f"{phase} wrote a trajectory without a terminal outcome")
        trajectory_wins = sum("_WIN.json.lz4" in path.name for path in trajectory_paths)
        trajectory_losses = sum("_LOSS.json.lz4" in path.name for path in trajectory_paths)
        if (trajectory_wins, trajectory_losses) != (wins, losses):
            raise CollectionError(f"{phase} result and trajectory outcomes do not match")
        record.update(
            completed_battles=len(rows),
            learner_wins=wins,
            learner_losses=losses,
            learner_trajectory_count=len(trajectory_paths),
            learner_result_csv=str(result_path.relative_to(args.out_dir)),
        )
        return record
    finally:
        terminate(learner_process)
        terminate(opponent_process)


def _totals(chunks: list[dict[str, Any]]) -> dict[str, int]:
    phases = [phase for chunk in chunks for phase in chunk.get("phases", [])]
    return {
        "completed_battles": sum(int(phase.get("completed_battles", 0)) for phase in phases),
        "learner_wins": sum(int(phase.get("learner_wins", 0)) for phase in phases),
        "learner_losses": sum(int(phase.get("learner_losses", 0)) for phase in phases),
        "learner_trajectory_count": sum(int(phase.get("learner_trajectory_count", 0)) for phase in phases),
    }


def _completed_chunk(path: Path, requested: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    chunk = json.loads(path.read_text(encoding="utf-8"))
    totals = _totals([chunk])
    if (
        not chunk.get("error")
        and chunk.get("requested_battles") == requested
        and totals["completed_battles"] == requested
        and totals["learner_wins"] + totals["learner_losses"] == requested
        and totals["learner_trajectory_count"] == requested
    ):
        return chunk
    return None


def write_battle_ledger(output: Path, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for chunk in chunks:
        for phase in chunk.get("phases", []):
            csv_path = output / str(phase.get("learner_result_csv", ""))
            rows = result_records(csv_path)
            if len(rows) != int(phase.get("completed_battles", -1)):
                raise CollectionError(f"battle ledger source count does not match: {csv_path}")
            for battle_index, result in enumerate(rows):
                records.append({
                    "schema_version": 1,
                    "chunk_index": chunk["chunk_index"],
                    "opponent": chunk["opponent"],
                    "phase": phase["phase"],
                    "learner_role": phase["learner_role"],
                    "battle_index": battle_index,
                    "result": result,
                })
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    ledger_path = output / "BATTLE_LEDGER.jsonl"
    if ledger_path.is_file() and ledger_path.read_text(encoding="utf-8") != payload:
        raise CollectionError(f"immutable battle ledger changed: {ledger_path}")
    if not ledger_path.is_file():
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(ledger_path)
    return {
        "path": ledger_path.name,
        "records": len(records),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def run_chunk(
    args: argparse.Namespace,
    profiles: dict[str, dict[str, Any]],
    learner_id: str,
    chunk_index: int,
    opponent_id: str,
    requested: int,
) -> dict[str, Any]:
    chunk_dir = args.out_dir / f"chunk_{chunk_index:05d}_{learner_id}_vs_{opponent_id}"
    manifest_path = chunk_dir / "MANIFEST.json"
    resumed = _completed_chunk(manifest_path, requested)
    if resumed is not None:
        return resumed
    if chunk_dir.exists():
        # Completed chunks are immutable; failed partial attempts must not leak
        # stale CSV rows or trajectories into a retry's admission totals.
        shutil.rmtree(chunk_dir)
    chunk: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "staged_online_rl_collection_chunk",
        "chunk_index": chunk_index,
        "opponent": opponent_id,
        "requested_battles": requested,
        "phases": [],
    }
    first = requested // 2 + (requested % 2 if chunk_index % 2 else 0)
    try:
        for learner_role, battles in (("acceptor", first), ("challenger", requested - first)):
            if battles:
                chunk["phases"].append(run_phase(
                    args=args, learner=profiles[learner_id], opponent=profiles[opponent_id],
                    opponent_id=opponent_id, shard_index=chunk_index,
                    learner_role=learner_role, battles=battles, output=chunk_dir,
                ))
        chunk.update(_totals([chunk]))
        if not args.dry_run and not (
            chunk["completed_battles"] == requested
            == chunk["learner_wins"] + chunk["learner_losses"]
            == chunk["learner_trajectory_count"]
        ):
            raise CollectionError(f"chunk {chunk_index} admission totals do not match")
    except Exception as exc:
        chunk["error"] = str(exc)
    chunk["completed_at"] = utc_now()
    atomic_json(manifest_path, chunk)
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--load-timeout-seconds", type=float, default=900)
    parser.add_argument("--phase-timeout-seconds", type=float, default=21600)
    parser.add_argument("--stall-timeout-seconds", type=float, default=1800)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-games", type=int, default=1)
    parser.add_argument("--collection-kind", choices=("fresh", "arena"), default="fresh")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.torch_threads, args.showdown_port, args.load_timeout_seconds, args.phase_timeout_seconds, args.stall_timeout_seconds, args.workers, args.chunk_games) <= 0:
        parser.error("thread and timeout values must be positive")
    if not args.dry_run and not Path(args.python).is_file():
        parser.error(f"Metamon Python does not exist: {args.python}")
    args.out_dir = args.out_dir.resolve()

    pool, schedule = load_plan(args.pool, args.schedule)
    args.format = pool.get("format", "gen9randombattle")
    profiles = pool["profiles"]
    learner_id = schedule["learner"]
    counts = Counter(schedule["opponents"])
    verify_local_checkpoints(profiles, {learner_id, *counts})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "staged_online_rl_collection",
        "created_at": utc_now(),
        "pool": str(args.pool.resolve()),
        "schedule": str(args.schedule.resolve()),
        "format": args.format,
        "learner": learner_id,
        "collection_kind": args.collection_kind,
        "dry_run": args.dry_run,
        "requested_battles": len(schedule["opponents"]),
        "chunks": [],
        "shards": [],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for opponent_id, requested in sorted(counts.items()):
        while requested:
            battles = min(requested, args.chunk_games)
            work.append((len(work), opponent_id, battles))
            requested -= battles
    chunks: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_chunk, args, profiles, learner_id, index, opponent, battles): index
            for index, opponent, battles in work
        }
        for future in as_completed(futures):
            chunks[futures[future]] = future.result()
    manifest["chunks"] = [chunks[index] for index in sorted(chunks)]
    # Keep the old key for readers of the smoke manifest while chunks become the unit of resume.
    manifest["shards"] = manifest["chunks"]
    failures = [chunk for chunk in manifest["chunks"] if chunk.get("error")]

    manifest["completed_at"] = utc_now()
    manifest["failed_shards"] = len(failures)
    manifest.update(_totals(manifest["chunks"]))
    if not args.dry_run and not failures:
        manifest["battle_ledger"] = write_battle_ledger(args.out_dir, manifest["chunks"])
    atomic_json(args.out_dir / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"{len(failures)} collection shard(s) failed; see MANIFEST.json")


if __name__ == "__main__":
    main()
