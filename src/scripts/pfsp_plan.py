#!/usr/bin/env python3
"""Create a reproducible PFSP-lite matchup schedule.

Input pool JSON keeps a frozen base weight per opponent. Historical paired H2H
results optionally refine it: opponents where the learner's win rate is in the
target band are preferred, while every pool member retains a minimum quota.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--battles", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text())
    members = pool["pfsp"]["pool"]
    low, high = pool["pfsp"].get("target_winrate", [0.4, 0.6])
    min_weight = pool["pfsp"].get("min_pool_weight", 0.05)
    learner = pool["pfsp"]["learner"]

    records: dict[str, list[int]] = defaultdict(list)
    if args.history and args.history.exists():
        for line in args.history.read_text().splitlines():
            row = json.loads(line)
            if row.get("learner") != learner or row.get("void"):
                continue
            opponent = row.get("opponent")
            winner = row.get("winner")
            if opponent and winner in {"learner", "opponent"}:
                records[opponent].append(1 if winner == "learner" else 0)

    weights = []
    report = []
    for member in members:
        opponent = member["id"]
        base = float(member["base_weight"])
        outcomes = records[opponent]
        rate = sum(outcomes) / len(outcomes) if outcomes else None
        # Unknown matchups keep their base priority. Known 40-60% matchups get
        # a 2x boost; clearly lopsided matchups retain the floor only.
        priority = base if rate is None else (base * 2.0 if low <= rate <= high else min_weight)
        weights.append(max(min_weight, priority))
        report.append({"opponent": opponent, "n": len(outcomes), "winrate": rate, "weight": weights[-1]})

    rng = random.Random(args.seed)
    picks = rng.choices([m["id"] for m in members], weights=weights, k=args.battles)
    schedule = {
        "schema_version": 1,
        "seed": args.seed,
        "learner": learner,
        "target_winrate": [low, high],
        "battle_count": args.battles,
        "paired_matchups": True,
        "selection_report": report,
        "opponents": picks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2) + "\n")
    print(json.dumps(schedule["selection_report"], indent=2))


if __name__ == "__main__":
    main()
