#!/usr/bin/env python3
"""Compute preregistered pair-level inference for mirrored H2H results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def pair_scores(payload: dict, candidate: str, comparator: str) -> list[float]:
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("H2H artifact has no games")
    pairs: dict[str, list[dict]] = {}
    for game in games:
        if not isinstance(game, dict):
            raise ValueError("H2H artifact contains a malformed game")
        if game.get("agent_a") != candidate or game.get("agent_b") != comparator:
            raise ValueError("H2H artifact does not match the frozen treatment order")
        pair_id = game.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("H2H game has no mirrored pair ID")
        pairs.setdefault(pair_id, []).append(game)
    scores = []
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {game.get("pair_leg") for game in pair} != {1, 2}:
            raise ValueError(f"mirrored pair {pair_id} is incomplete")
        if any(game.get("void") or game.get("error") for game in pair):
            raise ValueError(f"mirrored pair {pair_id} contains a void or error")
        winners = [game.get("winner") for game in pair]
        if any(winner not in {"agent_a", "agent_b"} for winner in winners):
            raise ValueError(f"mirrored pair {pair_id} has an unknown winner")
        scores.append(sum(winner == "agent_a" for winner in winners) / 2.0)
    return scores


def bootstrap_interval(
    scores: list[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    if not scores or resamples < 1:
        raise ValueError("bootstrap requires scores and positive resamples")
    rng = random.Random(seed)
    count = len(scores)
    means = sorted(
        math.fsum(scores[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    lower = means[math.floor(0.025 * (resamples - 1))]
    upper = means[math.ceil(0.975 * (resamples - 1))]
    return lower, upper


def analyze(
    payload: dict,
    *,
    candidate: str,
    comparator: str,
    resamples: int,
    seed: int,
) -> dict:
    scores = pair_scores(payload, candidate, comparator)
    mean = math.fsum(scores) / len(scores)
    lower, upper = bootstrap_interval(scores, resamples=resamples, seed=seed)
    return {
        "schema_version": 1,
        "candidate": candidate,
        "comparator": comparator,
        "games": len(scores) * 2,
        "complete_pairs": len(scores),
        "pair_sweeps_candidate": sum(score == 1.0 for score in scores),
        "pair_splits": sum(score == 0.5 for score in scores),
        "pair_sweeps_comparator": sum(score == 0.0 for score in scores),
        "pair_score_mean": mean,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "ci95_low": lower,
            "ci95_high": upper,
        },
        "claims": {
            "positive_point_estimate": mean > 0.5,
            "statistically_better_than_search_guided_r1": (
                len(scores) >= 250 and lower > 0.5
            ),
            "practical_margin_passed": len(scores) >= 250 and lower > 0.52,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="production_r1_shared_rm_plus")
    parser.add_argument("--comparator", default="production_r1_search_first")
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=8675315)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.input.expanduser().resolve().read_text())
    result = analyze(
        payload,
        candidate=args.candidate,
        comparator=args.comparator,
        resamples=args.resamples,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
