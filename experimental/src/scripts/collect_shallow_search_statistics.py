#!/usr/bin/env python3
"""Capture deployment-budget search statistics for a frozen causal root panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts.run_public_mcts_leaf_gate import _load_panel, _seed
from search.selective_shared_root import compute_selective_shared_root_metrics


SCHEMA = "metagross-shallow-search-statistics/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    mass = math.fsum(weight for weight, _ in values)
    return math.fsum(weight * value for weight, value in values) / mass if mass > 0 else 0.0


def _weighted_std(values: list[tuple[float, float]], mean: float) -> float:
    mass = math.fsum(weight for weight, _ in values)
    return math.sqrt(
        max(0.0, math.fsum(weight * (value - mean) ** 2 for weight, value in values) / mass)
    ) if mass > 0 else 0.0


def summarize_worlds(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    policies = [world["policy"] for world in worlds]
    weights = [float(world["weight"]) for world in worlds]
    metrics = compute_selective_shared_root_metrics(policies, weights)
    actions = sorted({action for policy in policies for action in policy})
    aggregate_visits = {
        action: math.fsum(weight * policy.get(action, 0.0) for weight, policy in zip(weights, policies))
        for action in actions
    }
    visit_total = math.fsum(aggregate_visits.values())
    aggregate_visits = {action: mass / visit_total for action, mass in aggregate_visits.items()}
    selected = max(actions, key=lambda action: (aggregate_visits[action], action))
    entropy = -math.fsum(
        probability * math.log(max(probability, 1e-12)) for probability in aggregate_visits.values()
    ) / math.log(len(actions)) if len(actions) > 1 else 0.0

    action_statistics = {}
    for action in actions:
        visit_rows = [(weight, policy.get(action, 0.0)) for weight, policy in zip(weights, policies)]
        value_rows = [
            (float(world["weight"]), float(world["values"][action]))
            for world in worlds
            if action in world["values"]
        ]
        visit_mean = aggregate_visits[action]
        value_mean = _weighted_mean(value_rows)
        action_statistics[action] = {
            "visit_mass": visit_mean,
            "visit_std": _weighted_std(visit_rows, visit_mean),
            "mean_value": value_mean,
            "value_std": _weighted_std(value_rows, value_mean),
            "world_support": math.fsum(weight for weight, _ in value_rows),
            "world_top_vote": math.fsum(
                weight
                for weight, policy in zip(weights, policies)
                if max(actions, key=lambda candidate: (policy.get(candidate, 0.0), candidate)) == action
            ),
        }
    return {
        "selected_action": selected,
        "action_statistics": action_statistics,
        "root_statistics": {
            "visit_entropy": entropy,
            "weighted_top_action_disagreement": metrics.weighted_top_action_disagreement,
            "weighted_js_divergence": metrics.weighted_js_divergence,
            "aggregate_top_visit_mass": metrics.aggregate_top_visit_mass,
            "aggregate_top_two_margin": metrics.aggregate_top_two_margin,
            "effective_world_count": metrics.effective_world_count,
            "action_count": metrics.action_count,
        },
    }


def _task(payload: tuple[dict[str, Any], int, int, float]) -> dict[str, Any]:
    schedule, iterations, budget_ms, budget_fraction = payload
    import poke_engine

    worlds = []
    total_visits = 0
    for world in schedule["worlds"]:
        state = poke_engine.State.from_string(world["state"])
        result = poke_engine.monte_carlo_tree_search(
            state,
            duration_ms=0,
            iterations=iterations,
            threads=1,
            seed=_seed(schedule["pair_id"], int(world["world_index"]), "shallow-500ms"),
        )
        denominator = max(1, int(result.total_visits))
        total_visits += denominator
        policy = {str(option.move_choice): int(option.visits) / denominator for option in result.side_one}
        values = {
            str(option.move_choice): float(option.total_score) / int(option.visits)
            for option in result.side_one
            if int(option.visits) > 0
        }
        worlds.append({
            "world_index": int(world["world_index"]),
            "weight": float(world["weight"]),
            "total_visits": denominator,
            "policy": policy,
            "values": values,
        })
    summary = summarize_worlds(worlds)
    return {
        "schema": SCHEMA,
        "battle_id": schedule["battle_id"],
        "root_id": schedule["root_id"],
        "pair_id": schedule["pair_id"],
        "schedule_id": schedule["schedule_id"],
        "budget_ms": budget_ms,
        "budget_fraction": budget_fraction,
        "iterations_per_world": iterations,
        "total_visits": total_visits,
        "worlds": worlds,
        **summary,
    }


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.budget_ms != 500 or not math.isclose(args.budget_fraction, 0.72) or args.iterations != 20_000:
        raise ValueError("shallow-search contract is frozen at 500 ms / 0.72 / 20k iterations per world")
    os.environ.pop("METAGROSS_VALUE_MODEL", None)
    os.environ.pop("METAGROSS_LEARNED_VALUE_WEIGHT", None)
    panel, panel_hash = _load_panel(args.panel)
    import poke_engine

    audited_worlds = [
        world
        for root in panel[:32]
        for schedule in root["schedules"][:1]
        for world in schedule["worlds"][:1]
    ]
    if not any(
        any(action.endswith("-tera") for action in poke_engine.root_options(poke_engine.State.from_string(world["state"]))[0])
        for world in audited_worlds
    ):
        raise RuntimeError("loaded engine lacks Gen 9 terastallization root options")
    engine_binary = Path(poke_engine.poke_engine.__file__)
    tasks = []
    for root in panel:
        for schedule in root["schedules"]:
            tasks.append(({
                **schedule,
                "battle_id": root["battle_id"],
                "root_id": root["root_id"],
                "pair_id": f"{root['root_id']}:{schedule['schedule_id']}",
            }, args.iterations, args.budget_ms, args.budget_fraction))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_task, tasks))
    rows.sort(key=lambda row: row["pair_id"])
    _write(rows, args.output)
    report = {
        "schema": "metagross-shallow-search-statistics-report/v1",
        "panel_sha256": panel_hash,
        "rows": len(rows),
        "battles": len({row["battle_id"] for row in rows}),
        "budget_ms": args.budget_ms,
        "budget_fraction": args.budget_fraction,
        "iterations_per_world": args.iterations,
        "engine_feature_contract": "gen9+terastallization+causal-public-reveals",
        "engine_binary_sha256": sha256(engine_binary),
        "mean_total_visits": math.fsum(row["total_visits"] for row in rows) / len(rows),
        "output_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--budget-ms", type=int, default=500)
    parser.add_argument("--budget-fraction", type=float, default=0.72)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    print(json.dumps(collect(args), sort_keys=True))


if __name__ == "__main__":
    main()
