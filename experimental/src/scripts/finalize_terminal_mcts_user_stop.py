#!/usr/bin/env python3
"""Freeze and audit a paired terminal-MCTS run stopped at a pair boundary.

This finalizer deliberately accepts only the atomic progress artifact.  Logs or
registrations for an in-flight pair are reported but excluded from outcomes and
telemetry.  It never reads the sealed confirmation panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from summarize_terminal_mcts_h2h import summarize_log


EXPECTED_CONFIG_SHA256 = (
    "719f92a56c564838a8298811c7e946ed7eb0b7266694acc185dabdfdb6e3542f"
)
EXPECTED_FUTILITY_SHA256 = (
    "4ba46a21d892f9b5ed0760f4bef582aac2dd8744abd3e77153e4a7e08601f26c"
)
EXPECTED_H2H_SOURCE_SHA256 = (
    "a569b5896f5af4d50dda517417d6de0acaf4a24ede8221dd1331c9a32e31104a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frozen-futility", required=True, type=Path)
    parser.add_argument("--h2h-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
        / denominator
    )
    return center - margin, center + margin


def main() -> None:
    args = parse_args()
    progress_path = args.run_dir / "result.json.progress.json"
    plan_path = args.run_dir / "result.json.pairs.json"
    progress = load_json(progress_path)
    plan = load_json(plan_path)
    futility = load_json(args.frozen_futility)

    if sha256(args.frozen_futility) != EXPECTED_FUTILITY_SHA256:
        raise ValueError("frozen 50-game prefix hash mismatch")
    if sha256(args.h2h_source) != EXPECTED_H2H_SOURCE_SHA256:
        raise ValueError("H2H evaluator source hash mismatch")
    if progress.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("progress config hash mismatch")
    if plan.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("pair-plan config hash mismatch")

    games = progress.get("games")
    pairs = plan.get("pairs")
    if not isinstance(games, list) or not isinstance(pairs, list):
        raise ValueError("invalid progress or pair-plan schema")
    if not games or len(games) % 2:
        raise ValueError("stop is not at a non-empty paired boundary")
    if len(games) > 150:
        raise ValueError("success continuation exceeded frozen maximum")

    for expected_index, game in enumerate(games, 1):
        if game.get("game_index") != expected_index:
            raise ValueError("game indexes are not contiguous")
        if game.get("void") or game.get("error") is not None:
            raise ValueError(f"void/error in completed game {expected_index}")
        if game.get("winner") not in {"agent_a", "agent_b"}:
            raise ValueError(f"invalid winner in game {expected_index}")
        expected_pair = pairs[(expected_index - 1) // 2]
        if game.get("pair_id") != expected_pair.get("pair_id"):
            raise ValueError(f"pair identity mismatch in game {expected_index}")
        if game.get("battle_seed") != expected_pair.get("battle_seed"):
            raise ValueError(f"battle-seed mismatch in game {expected_index}")
        if game.get("pair_leg") != 1 + ((expected_index - 1) % 2):
            raise ValueError(f"pair-leg mismatch in game {expected_index}")
    for left, right in zip(games[::2], games[1::2], strict=True):
        if left["pair_id"] != right["pair_id"]:
            raise ValueError("pair was split at stop boundary")
        if left["agent_a_team_sha256"] != right["agent_b_team_sha256"]:
            raise ValueError("agent A/B teams were not mirrored")
        if left["agent_b_team_sha256"] != right["agent_a_team_sha256"]:
            raise ValueError("agent B/A teams were not mirrored")
        if {left["challenger"], right["challenger"]} != {"agent_a", "agent_b"}:
            raise ValueError("challenger roles were not mirrored")

    completed = len(games)
    pattern = re.compile(r"^tmy1[xy](\d{3})[a-z0-9]+\.search\.jsonl$")
    completed_telemetry: dict[int, dict[str, Any]] = {}
    excluded_indexes: set[int] = set()
    for path in sorted((args.run_dir / "logs").glob("*.search.jsonl")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected search-log domain: {path.name}")
        index = int(match.group(1))
        telemetry = summarize_log(path)
        if telemetry["teacher_calls"] == 0:
            continue
        if index > completed:
            excluded_indexes.add(index)
            continue
        if index in completed_telemetry:
            raise ValueError(f"duplicate treatment telemetry for game {index}")
        telemetry["candidate_won"] = games[index - 1]["winner"] == "agent_a"
        completed_telemetry[index] = telemetry
    if set(completed_telemetry) != set(range(1, completed + 1)):
        raise ValueError("missing treatment telemetry for a completed game")

    registration_names = {
        path.stem for path in (args.run_dir / "registrations").glob("*.json")
    }
    if any(not name.startswith("tmy1") for name in registration_names):
        raise ValueError("registration escaped frozen tmy1 domain")
    if len(registration_names) < completed * 2:
        raise ValueError("missing registrations for completed games")

    telemetry = list(completed_telemetry.values())
    extension_wins = sum(game["winner"] == "agent_a" for game in games)
    prefix_games = int(futility["games"])
    prefix_wins = int(futility["candidate_wins"])
    aggregate_games = prefix_games + completed
    aggregate_wins = prefix_wins + extension_wins
    lower, upper = wilson_interval(aggregate_wins, aggregate_games)
    override_games = [row for row in telemetry if row["override_count"] > 0]
    passthrough_games = [row for row in telemetry if row["override_count"] == 0]
    combined_override_games = int(futility["override_games"]) + len(override_games)
    combined_override_wins = int(futility["override_game_wins"]) + sum(
        row["candidate_won"] for row in override_games
    )
    combined_passthrough_games = int(futility["passthrough_games"]) + len(
        passthrough_games
    )
    combined_passthrough_wins = int(futility["passthrough_game_wins"]) + sum(
        row["candidate_won"] for row in passthrough_games
    )

    pair_scores = {"candidate_2_0": 0, "split_1_1": 0, "candidate_0_2": 0}
    for left, right in zip(games[::2], games[1::2], strict=True):
        wins = int(left["winner"] == "agent_a") + int(right["winner"] == "agent_a")
        key = ("candidate_0_2", "split_1_1", "candidate_2_0")[wins]
        pair_scores[key] += 1

    report = {
        "schema": "metagross-terminal-mcts-user-directed-stop/v1",
        "status": "user_directed_early_stop_not_a_preregistered_success_look",
        "completed_extension_games": completed,
        "completed_extension_pairs": completed // 2,
        "planned_extension_games": 150,
        "excluded_incomplete_game_indexes": sorted(excluded_indexes),
        "extension_candidate_wins": extension_wins,
        "extension_candidate_losses": completed - extension_wins,
        "extension_winrate": extension_wins / completed,
        "extension_pair_scores": pair_scores,
        "extension_candidate_as_challenger": {
            "games": sum(game["challenger"] == "agent_a" for game in games),
            "wins": sum(
                game["challenger"] == "agent_a" and game["winner"] == "agent_a"
                for game in games
            ),
        },
        "extension_candidate_as_acceptor": {
            "games": sum(game["acceptor"] == "agent_a" for game in games),
            "wins": sum(
                game["acceptor"] == "agent_a" and game["winner"] == "agent_a"
                for game in games
            ),
        },
        "aggregate_games_including_frozen_50": aggregate_games,
        "aggregate_candidate_wins": aggregate_wins,
        "aggregate_candidate_losses": aggregate_games - aggregate_wins,
        "aggregate_winrate": aggregate_wins / aggregate_games,
        "aggregate_wilson_ci95": [lower, upper],
        "preregistered_success_boundary": "wilson_95pct_lower_gt_0.5",
        "preregistered_success_boundary_met": lower > 0.5,
        "cycle1_admitted": False,
        "cycle1_admission_reason": "user stopped before max-200; success boundary not met",
        "extension_teacher_calls": sum(row["teacher_calls"] for row in telemetry),
        "extension_override_count": sum(row["override_count"] for row in telemetry),
        "extension_fail_closed_count": sum(
            row["fail_closed_count"] for row in telemetry
        ),
        "extension_override_games": len(override_games),
        "extension_override_game_wins": sum(
            row["candidate_won"] for row in override_games
        ),
        "extension_passthrough_games": len(passthrough_games),
        "extension_passthrough_game_wins": sum(
            row["candidate_won"] for row in passthrough_games
        ),
        "aggregate_override_games": combined_override_games,
        "aggregate_override_game_wins": combined_override_wins,
        "aggregate_passthrough_games": combined_passthrough_games,
        "aggregate_passthrough_game_wins": combined_passthrough_wins,
        "conditional_outcome_warning": (
            "override/pass-through outcomes are descriptive and confounded by "
            "teacher eligibility; they are not a causal override-effect estimate"
        ),
        "frozen_futility_sha256": sha256(args.frozen_futility),
        "progress_sha256": sha256(progress_path),
        "pair_plan_sha256": sha256(plan_path),
        "h2h_source_sha256": sha256(args.h2h_source),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "username_prefix": "tmy1",
        "run_id": "terminal-mcts-cycle1-success150",
        "mirror_seed": "2026101531",
        "production_seed": "27" * 32,
        "sealed_confirmation_panel_rows_read": 0,
        "local_cpu_only": True,
        "paid_compute_usd": 0,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
