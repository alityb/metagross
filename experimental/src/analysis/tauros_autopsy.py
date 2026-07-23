#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lz4.frame


STATUS_NONE = {None, "", "none", "nostatus"}
RECOVERY_MOVES = {"recover", "softboiled", "rest"}
SLEEP_MOVES = {"sleeppowder", "hypnosis", "lovelykiss", "sing", "spore"}
PARALYSIS_MOVES = {"thunderwave", "bodyslam", "stunspore", "glare"}
BOOM_MOVES = {"explosion", "selfdestruct"}


@dataclass
class BattleResult:
    result: str
    turns: int
    battle_id: str
    team_file: str


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = wins / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_results(run_dir: Path) -> list[BattleResult]:
    rows: list[BattleResult] = []
    for path in sorted((run_dir / "metamon_results").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            for row in reader:
                rows.append(
                    BattleResult(
                        result=row.get("Result", ""),
                        turns=int(row.get("Turn Count") or 0),
                        battle_id=row.get("Battle ID", ""),
                        team_file=row.get("Team File", ""),
                    )
                )
    return rows


def read_lz4_json(path: Path) -> dict[str, Any]:
    with lz4.frame.open(path, "rb") as handle:
        return json.loads(handle.read())


def pokemon_name(pokemon: dict[str, Any] | None) -> str:
    if not pokemon:
        return "none"
    return str(pokemon.get("name") or pokemon.get("species") or "none").lower()


def hp_pct(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 0.0
    try:
        return float(pokemon.get("hp_pct", 0.0))
    except (TypeError, ValueError):
        return 0.0


def status(pokemon: dict[str, Any] | None) -> str | None:
    if not pokemon:
        return None
    value = pokemon.get("status")
    if value in STATUS_NONE:
        return None
    return str(value).lower()


def move_names(pokemon: dict[str, Any] | None) -> list[str]:
    if not pokemon:
        return []
    return [str(move.get("name", "")).lower() for move in pokemon.get("moves", [])]


def alive_count(state: dict[str, Any], side: str) -> int:
    if side == "player":
        mons = [state.get("player_active_pokemon")] + list(state.get("available_switches", []))
    else:
        # Metamon only fully knows revealed opponent state. Use the public remaining
        # counter as the best available opponent-side proxy.
        try:
            return int(state.get("opponents_remaining", 0))
        except (TypeError, ValueError):
            return 0
    return sum(1 for mon in mons if hp_pct(mon) > 0)


def action_name(state: dict[str, Any], action_idx: int) -> str:
    if action_idx < 0:
        return "noop"
    active = state.get("player_active_pokemon") or {}
    moves = move_names(active)
    if 0 <= action_idx < len(moves):
        return f"move:{moves[action_idx]}"
    switch_idx = action_idx - 4
    switches = state.get("available_switches", [])
    if 0 <= switch_idx < len(switches):
        return f"switch:{pokemon_name(switches[switch_idx])}"
    return f"unknown:{action_idx}"


def action_kind(action: str) -> str:
    if action.startswith("switch:"):
        return "switch"
    if not action.startswith("move:"):
        return "unknown"
    move = action.split(":", 1)[1]
    if move in SLEEP_MOVES:
        return "sleep"
    if move in PARALYSIS_MOVES:
        return "paralysis"
    if move in BOOM_MOVES:
        return "boom"
    if move in RECOVERY_MOVES:
        return "recovery"
    return "attack_or_other"


def bucket_state(state: dict[str, Any], action: str | None = None) -> str:
    active = state.get("player_active_pokemon")
    opponent = state.get("opponent_active_pokemon")
    turn_action = action or ""
    player_alive = alive_count(state, "player")
    opponent_alive = alive_count(state, "opponent")
    active_name = pokemon_name(active)
    opponent_name = pokemon_name(opponent)
    active_moves = set(move_names(active))

    if state.get("forced_switch"):
        return "forced_switch"
    if player_alive <= 2 or opponent_alive <= 2:
        if active_name == "tauros" or opponent_name == "tauros":
            return "tauros_endgame"
        return "low_hp_endgame"
    if action_kind(turn_action) == "sleep" or active_moves & SLEEP_MOVES:
        return "sleep_pressure"
    if action_kind(turn_action) == "paralysis" or active_moves & PARALYSIS_MOVES:
        return "paralysis_spread"
    if action_kind(turn_action) == "boom" or active_moves & BOOM_MOVES:
        return "explosion_opportunity"
    if active_name == "chansey" and opponent_name == "chansey":
        return "chansey_mirror"
    if active_name == "snorlax" or opponent_name == "snorlax":
        return "snorlax_trade"
    if action_kind(turn_action) == "recovery" or active_moves & RECOVERY_MOVES:
        return "recovery_loop"
    if status(active) is not None:
        return "statused_active"
    return "other"


def top_foul_play_move(row: dict[str, Any]) -> str:
    visits = row.get("mcts_visits") or {}
    if not visits:
        return "unknown"
    return max(visits.items(), key=lambda item: item[1])[0]


def load_foul_play_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def battle_number_from_tag(tag: str) -> int:
    match = re.search(r"-(\d+)$", tag or "")
    return int(match.group(1)) if match else 0


def summarize(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(run_dir)
    result_by_index = {idx + 1: result for idx, result in enumerate(results)}
    wins = sum(1 for result in results if result.result == "WIN")
    losses = sum(1 for result in results if result.result == "LOSS")
    ci_low, ci_high = wilson_ci(wins, wins + losses)

    tauros_bucket_counts: Counter[str] = Counter()
    tauros_bucket_wins: Counter[str] = Counter()
    tauros_actions: Counter[str] = Counter()
    tauros_action_kinds: Counter[str] = Counter()

    trajectory_paths = sorted((run_dir / "metamon_trajectories").glob("**/*.json.lz4"))
    for game_idx, path in enumerate(trajectory_paths, start=1):
        payload = read_lz4_json(path)
        result = result_by_index.get(game_idx)
        won = result is not None and result.result == "WIN"
        for state, action_idx in zip(payload.get("states", []), payload.get("actions", [])):
            action = action_name(state, int(action_idx))
            bucket = bucket_state(state, action)
            tauros_bucket_counts[bucket] += 1
            tauros_actions[action] += 1
            tauros_action_kinds[action_kind(action)] += 1
            if won:
                tauros_bucket_wins[bucket] += 1

    fp_rows = load_foul_play_rows(run_dir / "foul_play_decisions.jsonl")
    fp_bucket_counts: Counter[str] = Counter()
    fp_bucket_wins: Counter[str] = Counter()
    fp_top_moves: Counter[str] = Counter()
    for row in fp_rows:
        state_string = row.get("state")
        state = {}
        # Foul Play rows currently store poke-engine strings, not Metamon dicts.
        # We still bucket by top-move type and outcome until a common state schema
        # is added in the next trace iteration.
        top_move = top_foul_play_move(row)
        bucket = "foul_play_" + action_kind(f"move:{str(top_move).lower()}")
        fp_bucket_counts[bucket] += 1
        fp_top_moves[str(top_move)] += 1
        if row.get("label") == 1:
            fp_bucket_wins[bucket] += 1

    bucket_rows = []
    for bucket, count in sorted(tauros_bucket_counts.items()):
        bucket_rows.append(
            {
                "agent": "TaurosV0",
                "bucket": bucket,
                "count": count,
                "wins": tauros_bucket_wins[bucket],
                "winrate": tauros_bucket_wins[bucket] / count if count else 0.0,
            }
        )
    for bucket, count in sorted(fp_bucket_counts.items()):
        bucket_rows.append(
            {
                "agent": "FoulPlay",
                "bucket": bucket,
                "count": count,
                "wins": fp_bucket_wins[bucket],
                "winrate": fp_bucket_wins[bucket] / count if count else 0.0,
            }
        )

    with (out_dir / "bucket_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["agent", "bucket", "count", "wins", "winrate"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(bucket_rows)

    summary = {
        "run_dir": str(run_dir),
        "games": len(results),
        "tauros_wins": wins,
        "tauros_losses": losses,
        "tauros_winrate": wins / (wins + losses) if wins + losses else 0.0,
        "tauros_ci95": [ci_low, ci_high],
        "metamon_trajectory_files": len(trajectory_paths),
        "foul_play_decision_rows": len(fp_rows),
        "tauros_action_kinds": dict(tauros_action_kinds.most_common()),
        "tauros_top_actions": dict(tauros_actions.most_common(25)),
        "foul_play_top_moves": dict(fp_top_moves.most_common(25)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# Tauros Autopsy: {run_dir.name}",
        "",
        f"Games: {len(results)}",
        f"TaurosV0: {wins}-{losses} ({summary['tauros_winrate']:.3f}), CI95 [{ci_low:.3f}, {ci_high:.3f}]",
        f"Metamon trajectories: {len(trajectory_paths)}",
        f"Foul Play decision rows: {len(fp_rows)}",
        "",
        "## Tauros Action Kinds",
    ]
    for key, value in tauros_action_kinds.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Foul Play Moves"])
    for key, value in fp_top_moves.most_common(15):
        lines.append(f"- {key}: {value}")
    (out_dir / "top_disagreements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Tauros/Kakuna trace shards")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or Path("experiments") / "tauros_autopsy" / args.run_dir.name
    print(json.dumps(summarize(args.run_dir, out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
