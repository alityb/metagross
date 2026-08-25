#!/usr/bin/env python3
"""Synchronous local exact-terminal teacher for one live production root."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import sys
import time
from typing import Any

from scripts.collect_shallow_search_statistics import summarize_worlds
from train.direct_long_horizon_controller import decide, shortlist_top_two
from train.outcome_grounded import RESULT_SCHEMA, stable_u64, stable_uniform, weighted_choice


def _selected(options: list[Any]) -> str:
    if not options:
        raise RuntimeError("continuation search returned no actions")
    return str(max(options, key=lambda option: (int(option.visits), str(option.move_choice))).move_choice)


def _root_task(payload: dict[str, Any]) -> dict[str, Any]:
    import poke_engine

    state = poke_engine.State.from_string(payload["state"])
    result = poke_engine.monte_carlo_tree_search(
        state,
        duration_ms=0,
        iterations=20_000,
        threads=1,
        seed=stable_u64(payload["root_id"], payload["schedule_id"], payload["world_index"], "live-root") % (2**32),
    )
    denominator = max(1, int(result.total_visits))
    policy = {str(option.move_choice): int(option.visits) / denominator for option in result.side_one}
    values = {
        str(option.move_choice): float(option.total_score) / int(option.visits)
        for option in result.side_one if int(option.visits) > 0
    }
    opponent = [(str(option.move_choice), float(option.visits)) for option in result.side_two]
    return {**payload, "policy": policy, "values": values, "opponent_policy": opponent}


def _outcome_task(payload: dict[str, Any]) -> dict[str, Any]:
    import poke_engine

    outcomes = {action: [] for action in payload["actions"]}
    searches = 0
    for rollout in range(payload["rollout_start"], payload["rollout_start"] + payload["rollouts"]):
        opponent_action = weighted_choice(
            payload["opponent_policy"],
            stable_uniform(payload["seed"], payload["root_id"], payload["schedule_id"], payload["world_index"], rollout, "root-opponent"),
        )
        for action in payload["actions"]:
            state = poke_engine.State.from_string(payload["state"])
            chance = stable_uniform(payload["seed"], payload["root_id"], payload["schedule_id"], payload["world_index"], rollout, 0, "chance")
            state = poke_engine.step_with_uniform(state, action, opponent_action, chance)[0]
            value = float(poke_engine.terminal_value(state))
            outcome = None if value == 0.0 else (value + 1.0) / 2.0
            decisions = 1
            while outcome is None and decisions < 192:
                result = poke_engine.monte_carlo_tree_search(
                    state,
                    duration_ms=0,
                    iterations=2_048,
                    threads=1,
                    seed=stable_u64(payload["seed"], payload["root_id"], payload["schedule_id"], payload["world_index"], rollout, decisions, "continuation") % (2**32),
                )
                searches += 1
                side_one = _selected(result.side_one)
                side_two = _selected(result.side_two)
                chance = stable_uniform(payload["seed"], payload["root_id"], payload["schedule_id"], payload["world_index"], rollout, decisions, "chance")
                state = poke_engine.step_with_uniform(state, side_one, side_two, chance)[0]
                decisions += 1
                value = float(poke_engine.terminal_value(state))
                outcome = None if value == 0.0 else (value + 1.0) / 2.0
            outcomes[action].append({
                "world_index": int(payload["world_index"]),
                "rollout": rollout,
                "outcome": outcome,
                "decisions": decisions,
            })
    return {
        "schedule_id": int(payload["schedule_id"]),
        "world_index": int(payload["world_index"]),
        "action_outcomes": outcomes,
        "continuation_searches": searches,
    }


def _result_rows(
    root_id: str,
    battle_id: str,
    actions: tuple[str, str],
    parts: list[dict[str, Any]],
    rollouts: int,
) -> list[dict[str, Any]]:
    rows = []
    for schedule_id in (0, 1):
        selected = sorted(
            (part for part in parts if part["schedule_id"] == schedule_id),
            key=lambda part: part["world_index"],
        )
        if len(selected) != 8:
            raise RuntimeError("live teacher has incomplete world outcomes")
        action_outcomes = {
            action: [sample for part in selected for sample in part["action_outcomes"][action]]
            for action in actions
        }
        rows.append({
            "schema": RESULT_SCHEMA,
            "battle_id": battle_id,
            "root_id": root_id,
            "schedule_id": schedule_id,
            "baseline_action": actions[0],
            "candidate_actions": list(actions),
            "action_outcomes": action_outcomes,
            "configuration": {"rollouts": rollouts},
        })
    return rows


def evaluate(payload: dict[str, Any], workers: int) -> dict[str, Any]:
    started = time.monotonic()
    root_id = str(payload["root_id"])
    battle_id = str(payload["battle_id"])
    live_choice = str(payload["production_choice"])
    schedules = payload.get("schedules")
    if not isinstance(schedules, list) or len(schedules) != 2:
        raise ValueError("live teacher requires exactly two schedules")
    root_payloads = []
    for schedule_id, schedule in enumerate(schedules):
        worlds = schedule.get("worlds")
        if not isinstance(worlds, list) or len(worlds) != 8:
            raise ValueError("live teacher requires eight worlds per schedule")
        for world_index, world in enumerate(worlds):
            root_payloads.append({
                "root_id": root_id,
                "schedule_id": schedule_id,
                "world_index": world_index,
                "state": str(world["state"]),
                "weight": float(world["weight"]),
            })
    with ProcessPoolExecutor(max_workers=workers) as executor:
        roots = list(executor.map(_root_task, root_payloads))
    schedule_rows = []
    for schedule_id in (0, 1):
        worlds = [row for row in roots if row["schedule_id"] == schedule_id]
        summary = summarize_worlds(worlds)
        schedule_rows.append({"root_id": root_id, "schedule_id": schedule_id, **summary})
    try:
        actions = shortlist_top_two(schedule_rows)
    except ValueError as exc:
        return {
            "schema": "metagross-terminal-mcts-live-decision/v1",
            "decision": "abstain",
            "selected_action": live_choice,
            "reason": f"ineligible:{exc}",
            "elapsed_seconds": time.monotonic() - started,
        }
    if actions[0] != live_choice:
        return {
            "schema": "metagross-terminal-mcts-live-decision/v1",
            "decision": "abstain",
            "selected_action": live_choice,
            "reason": "two_schedule_baseline_disagrees_with_production",
            "teacher_baseline": actions[0],
            "elapsed_seconds": time.monotonic() - started,
        }

    def collect(rollout_start: int, rollouts: int) -> list[dict[str, Any]]:
        tasks = [
            {
                **root,
                "actions": actions,
                "seed": int(payload["seed"]),
                "rollout_start": rollout_start,
                "rollouts": rollouts,
            }
            for root in roots
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_outcome_task, tasks))

    prefix = collect(0, 4)
    stage4_rows = _result_rows(root_id, battle_id, actions, prefix, 4)
    stage4 = decide(stage4_rows)
    final = stage4
    continuation_searches = sum(part["continuation_searches"] for part in prefix)
    if stage4["decision"] != "override":
        suffix = collect(4, 12)
        continuation_searches += sum(part["continuation_searches"] for part in suffix)
        merged = []
        by_key = {(part["schedule_id"], part["world_index"]): part for part in prefix}
        for part in suffix:
            target = by_key[(part["schedule_id"], part["world_index"])]
            for action in actions:
                target["action_outcomes"][action].extend(part["action_outcomes"][action])
        merged = list(by_key.values())
        final = decide(_result_rows(root_id, battle_id, actions, merged, 16))
    return {
        "schema": "metagross-terminal-mcts-live-decision/v1",
        "decision": final["decision"],
        "selected_action": final["selected_action"],
        "baseline_action": actions[0],
        "alternative_action": actions[1],
        "reason": "frozen_terminal_gate",
        "rollouts": final["rollouts"],
        "terminal_coverage": final["terminal_coverage"],
        "paired_coverage": final["paired_coverage"],
        "schedule_advantages": final["schedule_advantages"],
        "cluster_bootstrap_ci95": final["cluster_bootstrap_ci95"],
        "checks": final["checks"],
        "continuation_searches": continuation_searches,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    report = evaluate(payload, args.workers)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
