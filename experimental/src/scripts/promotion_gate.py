#!/usr/bin/env python3
"""Run and evaluate a paired candidate-vs-accepted H2H promotion gate."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - margin, center + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-prior-url", required=True)
    parser.add_argument("--accepted-prior-url", required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args()
    if args.games % 2:
        raise SystemExit("--games must be even for paired H2H")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    cmd = [
        sys.executable, "-m", "eval.run",
        "--mode", "h2h", "--server", "local", "--format", "gen9randombattle",
        "--agent-a", "foul_play_root_priors_opp",
        "--agent-b", "foul_play_root_priors_opp",
        "--agent-a-prior-server-url", args.candidate_prior_url,
        "--agent-b-prior-server-url", args.accepted_prior_url,
        "--foul-play-search-time-ms", str(args.search_ms),
        "--foul-play-search-parallelism", str(args.parallelism),
        "--foul-play-search-threads", "1",
        "--n-games", str(args.games), "--paired",
        "--json-out", str(result_path),
        "--log-dir", str(args.out_dir / "logs"),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    result = json.loads(result_path.read_text())
    decisive = int(result.get("decisive_games", 0))
    wins = int(result.get("agent_a_wins", 0))
    low, high = wilson(wins, decisive)
    report = {
        "candidate": args.candidate_prior_url,
        "accepted": args.accepted_prior_url,
        "paired_games_requested": args.games,
        "decisive_games": decisive,
        "candidate_wins": wins,
        "candidate_winrate": wins / decisive if decisive else None,
        "wilson95": [low, high],
        "promote": decisive == args.games and low > 0.5,
        "raw_result": result,
    }
    (args.out_dir / "promotion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
