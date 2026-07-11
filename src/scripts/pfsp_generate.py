#!/usr/bin/env python3
"""Execute a PFSP-lite schedule as strict paired H2H generation shards.

Profiles are read from the JSON template used by `pfsp_plan.py`. Each matchup
is role-balanced by `eval.run --paired`, records MCTS visits per side, and
captures one full raw replay per battle (agent A only) to avoid duplicate POV
protocol files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
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
    cmd.extend([f"--agent-{prefix}-decision-log", str(out / f"agent_{prefix}_decisions.jsonl")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args()

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

    for opponent_id, games in sorted(counts.items()):
        if opponent_id == learner_id:
            opponent = learner
        else:
            opponent = profiles[opponent_id]
        # Paired H2H requires an even count. The schedule is a desired count;
        # round up to retain role balance and record the actual count.
        paired_games = games + (games % 2)
        out = args.out_dir / f"{learner_id}_vs_{opponent_id}"
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
        add_profile_args(cmd, "a", learner, out)
        add_profile_args(cmd, "b", opponent, out)
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        manifest["matchups"].append(
            {"opponent": opponent_id, "requested_games": games, "paired_games": paired_games, "out": str(out)}
        )

    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
