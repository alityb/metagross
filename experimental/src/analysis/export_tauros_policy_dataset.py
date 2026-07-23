#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import lz4.frame

from tauros_autopsy import action_kind, action_name, alive_count, bucket_state, hp_pct, move_names, pokemon_name, status


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "metamon_results").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle, skipinitialspace=True))
    return rows


def read_lz4_json(path: Path) -> dict[str, Any]:
    with lz4.frame.open(path, "rb") as handle:
        return json.loads(handle.read())


def feature_row(
    run_id: str,
    game_index: int,
    turn_index: int,
    result: dict[str, Any] | None,
    state: dict[str, Any],
    action_idx: int,
) -> dict[str, Any]:
    active = state.get("player_active_pokemon") or {}
    opponent = state.get("opponent_active_pokemon") or {}
    action = action_name(state, action_idx)
    active_moves = move_names(active)
    opponent_moves = move_names(opponent)
    return {
        "run_id": run_id,
        "game_index": game_index,
        "turn_index": turn_index,
        "battle_id": None if result is None else result.get("Battle ID"),
        "winner_label": 1 if result is not None and result.get("Result") == "WIN" else 0,
        "turn_count": int(result.get("Turn Count") or 0) if result else None,
        "bucket": bucket_state(state),
        "action_bucket": bucket_state(state, action),
        "action": action,
        "action_kind": action_kind(action),
        "action_idx": action_idx,
        "active": pokemon_name(active),
        "opponent_active": pokemon_name(opponent),
        "active_hp": hp_pct(active),
        "opponent_hp": hp_pct(opponent),
        "active_status": status(active) or "none",
        "opponent_status": status(opponent) or "none",
        "player_alive": alive_count(state, "player"),
        "opponent_alive": alive_count(state, "opponent"),
        "forced_switch": bool(state.get("forced_switch")),
        "active_moves": active_moves,
        "opponent_revealed_moves": opponent_moves,
        "has_sleep_move": any(move in {"sleeppowder", "hypnosis", "lovelykiss", "sing", "spore"} for move in active_moves),
        "has_para_move": any(move in {"thunderwave", "bodyslam", "stunspore", "glare"} for move in active_moves),
        "has_boom_move": any(move in {"explosion", "selfdestruct"} for move in active_moves),
        "has_recovery_move": any(move in {"recover", "softboiled", "rest"} for move in active_moves),
    }


def export_examples(run_dirs: list[Path], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    action_kinds = Counter()
    buckets = Counter()
    games = 0
    examples = 0
    wins = 0

    with output.open("w", encoding="utf-8") as handle:
        for run_dir in run_dirs:
            results = load_results(run_dir)
            trajectories = sorted((run_dir / "metamon_trajectories").glob("**/*.json.lz4"))
            games += len(results)
            wins += sum(1 for row in results if row.get("Result") == "WIN")
            for game_index, path in enumerate(trajectories, start=1):
                payload = read_lz4_json(path)
                result = results[game_index - 1] if game_index - 1 < len(results) else None
                for turn_index, (state, action_idx) in enumerate(
                    zip(payload.get("states", []), payload.get("actions", [])), start=1
                ):
                    row = feature_row(run_dir.name, game_index, turn_index, result, state, int(action_idx))
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    examples += 1
                    action_kinds[row["action_kind"]] += 1
                    buckets[row["bucket"]] += 1
            counts[run_dir.name] = len(trajectories)

    return {
        "output": str(output),
        "run_dirs": [str(path) for path in run_dirs],
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "examples": examples,
        "trajectories_by_run": dict(counts),
        "action_kinds": dict(action_kinds.most_common()),
        "buckets": dict(buckets.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tauros trajectory policy examples")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    summary = export_examples(args.run_dirs, args.output)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
