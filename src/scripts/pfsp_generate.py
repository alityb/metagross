#!/usr/bin/env python3
"""Execute a PFSP-lite schedule as strict paired H2H generation shards.

Profiles are read from the JSON template used by `pfsp_plan.py`. Each matchup
is role-balanced by `eval.run --paired`, records MCTS visits per side, and
captures one full raw replay per battle (agent A only) to avoid duplicate POV
protocol files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def add_profile_args(cmd: list[str], prefix: str, profile: dict, out: Path) -> None:
    cmd.extend([f"--agent-{prefix}", profile["agent"]])
    if profile.get("prior_server_url"):
        cmd.extend([f"--agent-{prefix}-prior-server-url", profile["prior_server_url"]])
    if profile.get("require_priors"):
        cmd.append(f"--agent-{prefix}-require-priors")
    if profile.get("python"):
        cmd.extend([f"--agent-{prefix}-python", profile["python"]])
    if prefix == "a":
        cmd.extend([f"--agent-{prefix}-replay-dir", str(out / "replays")])
    # Only learner decisions become training targets. Opponents may be stock
    # Foul Play, which has no learned-prior telemetry to validate.
    if prefix == "a":
        cmd.extend([f"--agent-{prefix}-decision-log", str(out / "agent_a_decisions.jsonl")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent matchup shards; each shard keeps games serial.")
    parser.add_argument("--shards-per-matchup", type=int, default=1,
                        help="Split each opponent matchup into isolated paired subshards.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.shards_per_matchup <= 0:
        parser.error("--shards-per-matchup must be positive")

    pool = json.loads(args.pool.read_text())
    schedule = json.loads(args.schedule.read_text())
    learner_id = schedule["learner"]
    profiles = pool["profiles"]
    learner = profiles[learner_id]

    counts = Counter(schedule["opponents"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "schedule": str(args.schedule),
        "learner": learner_id,
        "search_time_ms": args.search_ms,
        "parallelism": args.parallelism,
        "matchups": [],
    }
    run_nonce = secrets.token_hex(8)
    manifest["run_nonce"] = run_nonce

    def run_matchup(opponent_id: str, games: int, shard_index: int) -> dict:
        if opponent_id == learner_id:
            opponent = learner
        else:
            opponent = profiles[opponent_id]
        # Paired H2H requires an even count. The schedule is a desired count;
        # round up to retain role balance and record the actual count.
        paired_games = games + (games % 2)
        out = args.out_dir / f"{learner_id}_vs_{opponent_id}"
        if args.shards_per_matchup > 1:
            out /= f"shard_{shard_index:02d}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "eval.run",
            "--mode", "h2h", "--server", "local", "--format", pool["format"],
            "--paired", "--n-games", str(paired_games),
            "--foul-play-search-time-ms", str(args.search_ms),
            "--foul-play-search-parallelism", str(args.parallelism),
            "--foul-play-search-threads", "1",
            "--log-dir", str(out / "logs"),
            "--json-out", str(out / "result.json"),
        ]
        username_prefix = hashlib.blake2s(
            f"{run_nonce}\0{learner_id}\0{opponent_id}\0{shard_index}".encode(),
            digest_size=4,
        ).hexdigest()
        cmd.extend(["--username-prefix", username_prefix])
        add_profile_args(cmd, "a", learner, out)
        add_profile_args(cmd, "b", opponent, out)
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return {
            "opponent": opponent_id,
            "shard_index": shard_index,
            "requested_games": games,
            "paired_games": paired_games,
            "out": str(out),
            "username_prefix": username_prefix,
        }

    jobs = []
    for opponent_id, total_games in sorted(counts.items()):
        base_games, extra_games = divmod(total_games, args.shards_per_matchup)
        for shard_index in range(args.shards_per_matchup):
            games = base_games + (1 if shard_index < extra_games else 0)
            if games:
                jobs.append((opponent_id, games, shard_index))
    if args.workers == 1:
        completed = [run_matchup(opponent_id, games, shard_index) for opponent_id, games, shard_index in jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_matchup, opponent_id, games, shard_index)
                for opponent_id, games, shard_index in jobs
            ]
            completed = [future.result() for future in as_completed(futures)]
    manifest["matchups"] = sorted(
        completed, key=lambda row: (row["opponent"], row["shard_index"])
    )

    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
