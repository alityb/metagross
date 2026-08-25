#!/usr/bin/env python3
"""Aggregate the frozen 10+40 terminal-MCTS prospective H2H boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from summarize_terminal_mcts_h2h import summarize_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-dir", required=True, type=Path)
    parser.add_argument("--extension-dir", required=True, type=Path)
    parser.add_argument("--expected-canary-result-sha256", required=True)
    parser.add_argument("--expected-canary-telemetry-sha256", required=True)
    parser.add_argument("--success-dir", type=Path)
    parser.add_argument("--frozen-futility-result", type=Path)
    parser.add_argument("--expected-futility-sha256")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def argument_value(result: dict[str, Any], flag: str) -> str:
    arguments = result.get("execution_identity", {}).get("arguments", [])
    if not isinstance(arguments, list) or arguments.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag}")
    index = arguments.index(flag)
    try:
        return str(arguments[index + 1])
    except IndexError as exc:
        raise ValueError(f"missing value for {flag}") from exc


def wilson_interval(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        raise ValueError("games must be positive")
    z = 1.959963984540054
    proportion = wins / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / games + z * z / (4 * games * games)
        )
        / denominator
    )
    return center - margin, center + margin


def registration_usernames(run_dir: Path) -> set[str]:
    usernames = {path.stem for path in (run_dir / "registrations").glob("*.json")}
    if not usernames:
        raise ValueError(f"no registrations in {run_dir}")
    return usernames


def validate_run(
    run_dir: Path,
    expected_games: int,
    expected_prefix: str,
    expected_run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result = load_json(run_dir / "result.json")
    games = result.get("games")
    summary = result.get("summary")
    if not isinstance(games, list) or not isinstance(summary, dict):
        raise ValueError(f"invalid result schema in {run_dir}")
    if len(games) != expected_games or summary.get("completed_games") != expected_games:
        raise ValueError(f"unexpected completed game count in {run_dir}")
    if summary.get("void_games") != 0 or any(row.get("void") for row in games):
        raise ValueError(f"void game in {run_dir}")
    if any(row.get("error") is not None for row in games):
        raise ValueError(f"game error in {run_dir}")
    if summary.get("mirrored_pairs") is not True or summary.get("paired") is not True:
        raise ValueError(f"run is not paired and mirrored: {run_dir}")
    if summary.get("agent_a_as_challenger_games") != expected_games // 2:
        raise ValueError(f"challenger imbalance in {run_dir}")
    if summary.get("agent_a_as_acceptor_games") != expected_games // 2:
        raise ValueError(f"acceptor imbalance in {run_dir}")
    if argument_value(result, "--username-prefix") != expected_prefix:
        raise ValueError(f"unexpected username prefix in {run_dir}")
    if argument_value(result, "--run-id") != expected_run_id:
        raise ValueError(f"unexpected run ID in {run_dir}")

    registrations = registration_usernames(run_dir)
    if len(registrations) != expected_games * 2:
        raise ValueError(f"unexpected registration count in {run_dir}")
    if any(not username.startswith(expected_prefix) for username in registrations):
        raise ValueError(f"registration escaped prefix in {run_dir}")

    treatment_logs: list[dict[str, Any]] = []
    by_game_index: dict[int, dict[str, Any]] = {}
    pattern = re.compile(
        rf"^{re.escape(expected_prefix)}[xy](\d{{3}})[a-z0-9]+\.search\.jsonl$"
    )
    for path in sorted((run_dir / "logs").glob("*.search.jsonl")):
        telemetry = summarize_log(path)
        if telemetry["teacher_calls"] == 0:
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unrecognized treatment log name: {path.name}")
        game_index = int(match.group(1))
        if game_index in by_game_index:
            raise ValueError(f"duplicate treatment log for game {game_index}")
        telemetry["candidate_username"] = path.name.removesuffix(".search.jsonl")
        telemetry["game_index"] = game_index
        by_game_index[game_index] = telemetry
        treatment_logs.append(telemetry)
    if set(by_game_index) != set(range(1, expected_games + 1)):
        raise ValueError(f"missing treatment telemetry in {run_dir}")

    for game in games:
        game_index = int(game["game_index"])
        telemetry = by_game_index[game_index]
        telemetry["candidate_won"] = (
            game.get("winner_username") == telemetry["candidate_username"]
        )
    return result, games, treatment_logs


def main() -> None:
    args = parse_args()
    canary_result_path = args.canary_dir / "result.json"
    canary_telemetry_path = args.canary_dir / "teacher-telemetry-summary.json"
    if sha256(canary_result_path) != args.expected_canary_result_sha256:
        raise ValueError("canary result hash mismatch")
    if sha256(canary_telemetry_path) != args.expected_canary_telemetry_sha256:
        raise ValueError("canary telemetry hash mismatch")

    canary, canary_games, canary_logs = validate_run(
        args.canary_dir, 10, "tmc1", "terminal-mcts-cycle1-canary"
    )
    extension, extension_games, extension_logs = validate_run(
        args.extension_dir, 40, "tmx1", "terminal-mcts-cycle1-extension40"
    )
    if canary["execution_identity"]["source_sha256"] != extension["execution_identity"]["source_sha256"]:
        raise ValueError("H2H evaluator source changed between stages")
    canary_seed = argument_value(canary, "--mirror-seed")
    extension_seed = argument_value(extension, "--mirror-seed")
    if canary_seed == extension_seed:
        raise ValueError("mirror seeds are not disjoint")
    if argument_value(canary, "--production-run-seed") == argument_value(
        extension, "--production-run-seed"
    ):
        raise ValueError("production RNG seeds are not disjoint")
    stages = [
        (args.canary_dir, canary, canary_games, canary_logs),
        (args.extension_dir, extension, extension_games, extension_logs),
    ]
    if args.success_dir is not None:
        if args.frozen_futility_result is None or args.expected_futility_sha256 is None:
            raise ValueError("success aggregation requires frozen futility artifact/hash")
        if sha256(args.frozen_futility_result) != args.expected_futility_sha256:
            raise ValueError("frozen futility result hash mismatch")
        success, success_games, success_logs = validate_run(
            args.success_dir, 150, "tmy1", "terminal-mcts-cycle1-success150"
        )
        if (
            success["execution_identity"]["source_sha256"]
            != canary["execution_identity"]["source_sha256"]
        ):
            raise ValueError("H2H evaluator source changed in success continuation")
        stages.append((args.success_dir, success, success_games, success_logs))
    elif args.frozen_futility_result is not None or args.expected_futility_sha256 is not None:
        raise ValueError("futility artifact/hash only applies with --success-dir")

    username_sets = [registration_usernames(stage[0]) for stage in stages]
    pair_sets = [{row["pair_id"] for row in stage[2]} for stage in stages]
    battle_seed_sets = [
        {tuple(row["battle_seed"].split(",")) for row in stage[2]} for stage in stages
    ]
    mirror_seeds = [argument_value(stage[1], "--mirror-seed") for stage in stages]
    production_seeds = [
        argument_value(stage[1], "--production-run-seed") for stage in stages
    ]
    if len(set(mirror_seeds)) != len(mirror_seeds):
        raise ValueError("mirror seeds are not disjoint")
    if len(set(production_seeds)) != len(production_seeds):
        raise ValueError("production RNG seeds are not disjoint")
    for left in range(len(stages)):
        for right in range(left + 1, len(stages)):
            if username_sets[left] & username_sets[right]:
                raise ValueError("username domains overlap")
            if pair_sets[left] & pair_sets[right]:
                raise ValueError("pair identities overlap")
            if battle_seed_sets[left] & battle_seed_sets[right]:
                raise ValueError("battle seeds overlap")

    all_games = [game for stage in stages for game in stage[2]]
    all_logs = [row for stage in stages for row in stage[3]]
    wins = sum(int(row["candidate_won"]) for row in all_logs)
    if wins != sum(int(row.get("winner") == "agent_a") for row in all_games):
        raise ValueError("candidate win attribution disagrees with agent_a result")
    lower, upper = wilson_interval(wins, len(all_games))
    override_game_logs = [row for row in all_logs if row["override_count"] > 0]
    passthrough_game_logs = [row for row in all_logs if row["override_count"] == 0]
    is_final = len(all_games) == 200
    report = {
        "schema": (
            "metagross-terminal-mcts-success-look/v1"
            if is_final
            else "metagross-terminal-mcts-futility-look/v1"
        ),
        "games": len(all_games),
        "candidate_wins": wins,
        "candidate_losses": len(all_games) - wins,
        "winrate": wins / len(all_games),
        "wilson_ci95": [lower, upper],
        "futility_boundary": "stop_if_candidate_wins_lte_22_of_50",
        "futility_stop": (wins <= 22) if not is_final else False,
        "success_boundary": "admit_only_if_wilson_95pct_lower_gt_0.5",
        "cycle1_admitted": lower > 0.5,
        "teacher_calls": sum(row["teacher_calls"] for row in all_logs),
        "override_count": sum(row["override_count"] for row in all_logs),
        "fail_closed_count": sum(row["fail_closed_count"] for row in all_logs),
        "override_games": len(override_game_logs),
        "override_game_wins": sum(int(row["candidate_won"]) for row in override_game_logs),
        "passthrough_games": len(passthrough_game_logs),
        "passthrough_game_wins": sum(
            int(row["candidate_won"]) for row in passthrough_game_logs
        ),
        "canary_result_sha256": sha256(canary_result_path),
        "canary_telemetry_sha256": sha256(canary_telemetry_path),
        "extension_result_sha256": sha256(args.extension_dir / "result.json"),
        "h2h_source_sha256": canary["execution_identity"]["source_sha256"],
        "mirror_seeds": mirror_seeds,
        "sealed_confirmation_panel_rows_read": 0,
        "local_cpu_only": True,
        "paid_compute_usd": 0,
    }
    if args.success_dir is not None:
        report["frozen_futility_sha256"] = sha256(args.frozen_futility_result)
        report["success_extension_result_sha256"] = sha256(
            args.success_dir / "result.json"
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
