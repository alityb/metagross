#!/usr/bin/env python3
"""Validate and summarize a matched, model-slot-counterbalanced evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


class CounterbalanceError(RuntimeError):
    pass


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    rate = wins / games
    denominator = 1 + z * z / games
    center = rate + z * z / (2 * games)
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    return (center - margin) / denominator, (center + margin) / denominator


def _load_games(path: Path, expected_games: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload.get("games")
    if not isinstance(games, list) or len(games) != expected_games:
        raise CounterbalanceError(f"{path} does not contain exactly {expected_games} games")
    if any(game.get("void") or game.get("error") is not None for game in games):
        raise CounterbalanceError(f"{path} contains a void or error game")
    if [game.get("game_index") for game in games] != list(range(1, expected_games + 1)):
        raise CounterbalanceError(f"{path} has a non-contiguous game schedule")
    for offset in range(0, expected_games, 2):
        pair = games[offset:offset + 2]
        if len(pair) != 2 or pair[0].get("pair_id") != pair[1].get("pair_id"):
            raise CounterbalanceError(f"{path} contains an incomplete mirrored pair")
        if {pair[0].get("pair_leg"), pair[1].get("pair_leg")} != {1, 2}:
            raise CounterbalanceError(f"{path} contains invalid mirrored pair legs")
    return games


def _match_key(game: dict[str, Any]) -> tuple[Any, ...]:
    return (
        game.get("game_index"),
        game.get("pair_index"),
        game.get("pair_leg"),
        game.get("battle_seed"),
        game.get("team_1_sha256"),
        game.get("team_2_sha256"),
    )


def summarize(
    g4_as_a_path: Path,
    g4_as_b_path: Path,
    games_per_orientation: int,
) -> dict[str, Any]:
    if games_per_orientation <= 0 or games_per_orientation % 2:
        raise CounterbalanceError("games_per_orientation must be a positive even integer")
    g4_as_a = _load_games(g4_as_a_path, games_per_orientation)
    g4_as_b = _load_games(g4_as_b_path, games_per_orientation)
    if [_match_key(game) for game in g4_as_a] != [_match_key(game) for game in g4_as_b]:
        raise CounterbalanceError("the two model-slot orientations do not use identical matchups")

    g4_wins_as_a = sum(game["winner"] == "agent_a" for game in g4_as_a)
    g4_wins_as_b = sum(game["winner"] == "agent_b" for game in g4_as_b)
    total_games = games_per_orientation * 2
    g4_wins = g4_wins_as_a + g4_wins_as_b
    ci_low, ci_high = wilson_interval(g4_wins, total_games)
    return {
        "schema_version": 1,
        "record_type": "counterbalanced_full_stack_ab",
        "matched_design_valid": True,
        "games_per_orientation": games_per_orientation,
        "matched_team_seed_pairs": games_per_orientation // 2,
        "total_games": total_games,
        "g4_wins": g4_wins,
        "r1_wins": total_games - g4_wins,
        "g4_winrate": g4_wins / total_games,
        "g4_wilson95": [ci_low, ci_high],
        "g4_as_agent_a": {
            "wins": g4_wins_as_a,
            "losses": games_per_orientation - g4_wins_as_a,
            "winrate": g4_wins_as_a / games_per_orientation,
        },
        "g4_as_agent_b": {
            "wins": g4_wins_as_b,
            "losses": games_per_orientation - g4_wins_as_b,
            "winrate": g4_wins_as_b / games_per_orientation,
        },
        "slot_winrate_gap": abs(g4_wins_as_a - g4_wins_as_b) / games_per_orientation,
        "candidate_advantage_established": ci_low > 0.5,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g4-as-a", type=Path, required=True)
    parser.add_argument("--g4-as-b", type=Path, required=True)
    parser.add_argument("--games-per-orientation", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.g4_as_a, args.g4_as_b, args.games_per_orientation)
    atomic_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
