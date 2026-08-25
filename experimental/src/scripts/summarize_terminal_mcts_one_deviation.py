#!/usr/bin/env python3
"""Audit and score the frozen Cycle 1b one-deviation canary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from srcs.metagross.terminal_mcts_one_deviation import (
    CANARY_GAMES,
    SCHEMA,
    assignment_manifest,
)


MIN_ELIGIBLE = 12
MIN_ELIGIBLE_PER_ARM = 5
MIN_WIN_RATE_EFFECT = 0.30
MAX_ONE_SIDED_FISHER_P = 0.20


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def fisher_greater(
    teacher_wins: int, teacher_n: int, production_wins: int, production_n: int
) -> float:
    """One-sided Fisher exact p-value for teacher win rate > production."""
    total_wins = teacher_wins + production_wins
    total_n = teacher_n + production_n
    denominator = math.comb(total_n, teacher_n)
    lower = max(0, teacher_n - (total_n - total_wins))
    upper = min(teacher_n, total_wins)
    return sum(
        math.comb(total_wins, wins) * math.comb(total_n - total_wins, teacher_n - wins)
        / denominator
        for wins in range(max(teacher_wins, lower), upper + 1)
    )


def _candidate_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors: list[str] = []
    for path in sorted((run_dir / "logs").glob("*.search.jsonl")):
        source_rows = _load_jsonl(path)
        telemetry = []
        bare_teacher_lines = []
        for row in source_rows:
            override = row.get("choice_override")
            if not isinstance(override, dict):
                continue
            one = override.get("terminal_mcts_one_deviation")
            teacher = override.get("terminal_mcts_teacher")
            if isinstance(teacher, dict) and not isinstance(one, dict):
                bare_teacher_lines.append(row["_line_number"])
            if isinstance(one, dict):
                telemetry.append((row, override, one, teacher))
        if not telemetry:
            continue
        if bare_teacher_lines:
            errors.append(f"{path.name}: teacher rows lack Cycle 1b telemetry")
        assignments = {
            json.dumps(one.get("assignment"), sort_keys=True)
            for _row, _override, one, _teacher in telemetry
        }
        if len(assignments) != 1:
            errors.append(f"{path.name}: assignment changed within a game")
            continue
        assignment = telemetry[0][2].get("assignment")
        if not isinstance(assignment, dict):
            errors.append(f"{path.name}: missing assignment")
            continue
        eligible = [entry for entry in telemetry if entry[2].get("eligible") is True]
        failures = [
            entry[2].get("integrity_failure")
            for entry in telemetry
            if entry[2].get("integrity_failure") is not None
        ]
        if len(eligible) > 1:
            errors.append(f"{path.name}: more than one eligible opportunity")
        locked_positions = [
            index
            for index, entry in enumerate(telemetry)
            if entry[2].get("locked_after_decision") is True
        ]
        if locked_positions and locked_positions[0] != len(telemetry) - 1:
            errors.append(f"{path.name}: teacher queried after the game was locked")
        for row, override, one, teacher in telemetry:
            line = row["_line_number"]
            if one.get("schema") != SCHEMA:
                errors.append(f"{path.name}:{line}: wrong telemetry schema")
            if not isinstance(teacher, dict):
                errors.append(f"{path.name}:{line}: teacher decision is missing")
            if one.get("production_action") != override.get(
                "terminal_mcts_production_choice"
            ):
                errors.append(f"{path.name}:{line}: production action mismatch")
            if one.get("eligible") is True:
                arm = assignment.get("arm")
                expected = (
                    one.get("teacher_action")
                    if arm == "teacher"
                    else one.get("production_action")
                )
                if override.get("final_choice") != expected:
                    errors.append(f"{path.name}:{line}: randomized action was not obeyed")
                if bool(one.get("intervention_applied")) != (arm == "teacher"):
                    errors.append(f"{path.name}:{line}: intervention flag disagrees with arm")
        records.append(
            {
                "path": path.name,
                "assignment": assignment,
                "teacher_queries": len(telemetry),
                "eligible": bool(eligible),
                "opportunity": eligible[0][2] if eligible else None,
                "integrity_failures": failures,
            }
        )
    return records, errors


def summarize(run_dir: Path, *, seed: str) -> dict[str, Any]:
    result = _load_json(run_dir / "result.json")
    games = result.get("games")
    if not isinstance(games, list):
        raise ValueError("result.json has no games array")
    records, errors = _candidate_records(run_dir)
    games_by_index = {game.get("game_index"): game for game in games if isinstance(game, dict)}
    expected_manifest = assignment_manifest(seed)
    expected_by_index = {
        row["game_index"]: row for row in expected_manifest["assignments"]
    }
    if len(games) != CANARY_GAMES or len(games_by_index) != CANARY_GAMES:
        errors.append(f"expected exactly {CANARY_GAMES} unique games")
    if len(records) != CANARY_GAMES:
        errors.append(f"expected exactly {CANARY_GAMES} candidate telemetry logs")

    analyzed = []
    seen_indices = set()
    for record in records:
        assignment = record["assignment"]
        game_index = assignment.get("game_index")
        if game_index in seen_indices:
            errors.append(f"duplicate assignment for game {game_index}")
            continue
        seen_indices.add(game_index)
        expected = expected_by_index.get(game_index)
        if expected is None or any(
            assignment.get(field) != expected.get(field)
            for field in ("game_index", "pair_index", "pair_leg", "arm")
        ):
            errors.append(f"game {game_index}: assignment differs from frozen schedule")
            continue
        if assignment.get("schedule_sha256") != expected_manifest["schedule_sha256"]:
            errors.append(f"game {game_index}: assignment schedule hash mismatch")
        game = games_by_index.get(game_index)
        if not isinstance(game, dict):
            errors.append(f"game {game_index}: missing outcome")
            continue
        if game.get("void") or game.get("winner") not in {"agent_a", "agent_b"}:
            errors.append(f"game {game_index}: non-decisive outcome")
        if game.get("pair_index") != expected["pair_index"] or game.get(
            "pair_leg"
        ) != expected["pair_leg"]:
            errors.append(f"game {game_index}: mirrored-pair metadata mismatch")
        if record["integrity_failures"]:
            errors.append(f"game {game_index}: teacher integrity failure")
        analyzed.append(
            {
                **record,
                "candidate_win": game.get("winner") == "agent_a",
                "battle_tag": game.get("battle_tag"),
                "pair_id": game.get("pair_id"),
            }
        )

    arm_counts = Counter(row["assignment"]["arm"] for row in analyzed)
    teacher_leg_counts = Counter(
        row["assignment"]["pair_leg"]
        for row in analyzed
        if row["assignment"]["arm"] == "teacher"
    )
    if arm_counts != {"teacher": 10, "production": 10}:
        errors.append("assignment is not exactly 10/10")
    if teacher_leg_counts != {1: 5, 2: 5}:
        errors.append("teacher assignment is not exactly 5/5 across mirror legs")
    if len({row["battle_tag"] for row in analyzed}) != len(analyzed):
        errors.append("battle tags are not unique within the fresh run")
    if len({row["pair_id"] for row in analyzed}) != CANARY_GAMES // 2:
        errors.append("expected exactly ten fresh mirrored pair ids")

    eligible = [row for row in analyzed if row["eligible"]]
    by_arm = {
        arm: [row for row in eligible if row["assignment"]["arm"] == arm]
        for arm in ("teacher", "production")
    }
    teacher_n = len(by_arm["teacher"])
    production_n = len(by_arm["production"])
    teacher_wins = sum(row["candidate_win"] for row in by_arm["teacher"])
    production_wins = sum(row["candidate_win"] for row in by_arm["production"])
    teacher_rate = teacher_wins / teacher_n if teacher_n else None
    production_rate = production_wins / production_n if production_n else None
    effect = (
        teacher_rate - production_rate
        if teacher_rate is not None and production_rate is not None
        else None
    )
    fisher_p = (
        fisher_greater(teacher_wins, teacher_n, production_wins, production_n)
        if teacher_n and production_n
        else None
    )
    integrity_ok = not errors
    pass_gate = bool(
        integrity_ok
        and len(eligible) >= MIN_ELIGIBLE
        and min(teacher_n, production_n) >= MIN_ELIGIBLE_PER_ARM
        and effect is not None
        and effect >= MIN_WIN_RATE_EFFECT
        and fisher_p is not None
        and fisher_p <= MAX_ONE_SIDED_FISHER_P
    )

    return {
        "schema": "metagross-terminal-mcts-one-deviation-canary-summary/v1",
        "run_dir": str(run_dir.resolve()),
        "estimand": (
            "candidate real-game win-rate difference at the first certified "
            "deviation: teacher action minus production action"
        ),
        "assignment_schedule_sha256": expected_manifest["schedule_sha256"],
        "integrity": {"ok": integrity_ok, "errors": errors},
        "all_randomized_games": {
            "n": len(analyzed),
            "arm_counts": dict(arm_counts),
            "candidate_wins_by_arm": {
                arm: sum(
                    row["candidate_win"]
                    for row in analyzed
                    if row["assignment"]["arm"] == arm
                )
                for arm in ("teacher", "production")
            },
        },
        "eligible_games": {
            "n": len(eligible),
            "teacher_n": teacher_n,
            "production_n": production_n,
            "teacher_wins": teacher_wins,
            "production_wins": production_wins,
            "teacher_win_rate": teacher_rate,
            "production_win_rate": production_rate,
            "win_rate_effect": effect,
            "fisher_exact_greater_p": fisher_p,
        },
        "frozen_gate": {
            "minimum_eligible_games": MIN_ELIGIBLE,
            "minimum_eligible_per_arm": MIN_ELIGIBLE_PER_ARM,
            "minimum_win_rate_effect": MIN_WIN_RATE_EFFECT,
            "maximum_one_sided_fisher_p": MAX_ONE_SIDED_FISHER_P,
            "decision": "PASS_TO_POWERED_REPLICATION" if pass_gate else "STOP_CYCLE1B",
            "deployment_authorized": False,
        },
        "games": sorted(analyzed, key=lambda row: row["assignment"]["game_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", default="2026081507")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir, seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
