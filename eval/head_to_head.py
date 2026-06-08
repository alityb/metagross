from __future__ import annotations

import argparse
import json
import math


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom
    return center - margin, center + margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head result summarizer")
    parser.add_argument("--wins", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    args = parser.parse_args()
    low, high = wilson_interval(args.wins, args.games)
    print(json.dumps({"wins": args.wins, "games": args.games, "winrate": args.wins / args.games, "wilson_95": [low, high]}, indent=2))


if __name__ == "__main__":
    main()
