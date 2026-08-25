#!/usr/bin/env python3
"""Frozen Cycle 18 equal-prior 8,192-iteration live root controller."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import random
import sys
import time
from typing import Any

ITERATIONS = 8192
SCHEDULES = 2
WORLDS = 8


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")


def task(payload: dict[str, Any]) -> dict[str, Any]:
    import poke_engine
    state = poke_engine.State.from_string(payload["state"])
    started = time.perf_counter()
    result = poke_engine.monte_carlo_tree_search_with_s1_request(
        state, payload["request_actions"], duration_ms=0, iterations=ITERATIONS,
        threads=1, seed=payload["search_seed"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    def side(rows):
        return [{"action": str(row.move_choice), "N": int(row.visits),
                 "W": float(row.total_score),
                 "Q": float(row.total_score) / int(row.visits) if int(row.visits) else None}
                for row in rows]
    return {"schedule_index": payload["schedule_index"], "world_index": payload["world_index"],
            "weight": payload["weight"], "state_sha256": hashlib.sha256(payload["state"].encode()).hexdigest(),
            "search_seed": payload["search_seed"], "latency_ms": elapsed_ms,
            "total_visits": int(result.total_visits), "side_one": side(result.side_one),
            "side_two": side(result.side_two)}


def evaluate(payload: dict[str, Any], workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    schedules = payload.get("schedules"); actions = payload.get("request_actions")
    if not isinstance(schedules, list) or len(schedules) != SCHEDULES:
        raise ValueError("Cycle18 requires two schedules")
    if not isinstance(actions, list) or not actions or len(actions) != len(set(actions)):
        raise ValueError("Cycle18 requires unique exact request actions")
    tasks = []
    for schedule_index, schedule in enumerate(schedules):
        worlds = schedule.get("worlds")
        if not isinstance(worlds, list) or len(worlds) != WORLDS:
            raise ValueError("Cycle18 requires eight worlds per schedule")
        for world_index, world in enumerate(worlds):
            weight = float(world["weight"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("invalid belief weight")
            tasks.append({"state": str(world["state"]), "weight": weight,
                "request_actions": actions, "schedule_index": schedule_index, "world_index": world_index,
                "search_seed": stable_seed(payload["seed"], payload["root_id"], schedule_index, world_index,
                                           "cycle18-equal8192") % (2**32)})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(task, tasks))
    masses = {action: 0.0 for action in actions}
    for receipt in receipts:
        by_action = {row["action"]: row["N"] for row in receipt["side_one"]}
        if set(by_action) != set(actions) or receipt["total_visits"] != ITERATIONS:
            raise RuntimeError("Cycle18 root search violated action/visit contract")
        # Each schedule has normalized weight; mix the two schedules equally.
        for action in actions:
            masses[action] += 0.5 * receipt["weight"] * by_action[action] / ITERATIONS
    total = math.fsum(masses.values())
    if total <= 0: raise RuntimeError("Cycle18 aggregate visit mass is empty")
    policy = {action: mass / total for action, mass in masses.items()}
    action_seed = stable_seed(payload["seed"], payload["root_id"], "cycle18-action")
    ordered = sorted(actions)
    selected = random.Random(action_seed).choices(ordered, weights=[policy[a] for a in ordered], k=1)[0]
    production = str(payload["production_choice"])
    return {
        "schema": "metagross-terminal-mcts-live-decision/v1",
        "controller_schema": "metagross-cycle18-equal8192-live/v1",
        "decision": "override" if selected != production else "abstain",
        "selected_action": selected, "production_action": production,
        "reason": "frozen_equal8192_visit_policy",
        "iterations_per_world": ITERATIONS, "schedule_count": SCHEDULES,
        "world_count": len(receipts), "action_seed": action_seed,
        "aggregate_policy": policy, "receipts": receipts,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=8)
    report = evaluate(json.load(sys.stdin), parser.parse_args().workers)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__": main()
