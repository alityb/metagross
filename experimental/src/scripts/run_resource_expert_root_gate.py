#!/usr/bin/env python3
"""Run and score the resource-aware long-horizon root teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from scripts.run_public_mcts_leaf_gate import _load_oracle, _load_panel, _seed, _sha256


RESULT_SCHEMA = "metagross-resource-expert-root-arm/v1"
REPORT_SCHEMA = "metagross-resource-expert-root-gate/v1"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _arm_task(task: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
    schedule, iterations, arm = task
    import poke_engine

    visits: dict[str, float] = {}
    values: dict[str, float] = {}
    masses: dict[str, float] = {}
    total = 0
    pair_id = schedule["pair_id"]
    for world in schedule["worlds"]:
        state = poke_engine.State.from_string(world["state"])
        result = poke_engine.monte_carlo_tree_search(
            state,
            iterations=iterations,
            threads=1,
            seed=_seed(pair_id, int(world["world_index"]), f"{arm}-teacher"),
        )
        weight = float(world["weight"])
        denominator = max(1, result.total_visits)
        total += result.total_visits
        for option in result.side_one:
            visits[option.move_choice] = visits.get(option.move_choice, 0.0) + weight * option.visits / denominator
            if option.visits > 0:
                values[option.move_choice] = values.get(option.move_choice, 0.0) + weight * option.total_score / option.visits
                masses[option.move_choice] = masses.get(option.move_choice, 0.0) + weight
    selected = max(visits, key=lambda action: (visits[action], action))
    return {
        "schema": RESULT_SCHEMA,
        "arm": arm,
        "battle_id": schedule["battle_id"],
        "root_id": schedule["root_id"],
        "pair_id": pair_id,
        "selected_action": selected,
        "historical_500ms_action": schedule["historical_500ms_action"],
        "resource_stratum": schedule["resource_stratum"],
        "iterations_per_world": iterations,
        "total_iterations": total,
        "visit_policy": visits,
        "mean_action_values": {action: values[action] / masses[action] for action in values},
    }


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm == "resource":
        if args.model is None or not args.model.is_file() or not 0 < args.learned_weight <= 1:
            raise ValueError("resource arm requires a model and positive blend")
        os.environ["METAGROSS_VALUE_MODEL"] = str(args.model.resolve())
        os.environ["METAGROSS_LEARNED_VALUE_WEIGHT"] = str(args.learned_weight)
    else:
        os.environ.pop("METAGROSS_VALUE_MODEL", None)
        os.environ.pop("METAGROSS_LEARNED_VALUE_WEIGHT", None)
    panel, panel_hash = _load_panel(args.panel)
    tasks = []
    for root in panel:
        for schedule in root["schedules"]:
            tasks.append(
                (
                    {
                        **schedule,
                        "battle_id": root["battle_id"],
                        "root_id": root["root_id"],
                        "pair_id": f"{root['root_id']}:{schedule['schedule_id']}",
                        "historical_500ms_action": root["historical_500ms_action"],
                        "resource_stratum": root["resource_stratum"],
                    },
                    args.iterations,
                    args.arm,
                )
            )
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_arm_task, tasks))
    rows.sort(key=lambda row: row["pair_id"])
    _write(args.output, rows)
    return {
        "arm": args.arm,
        "pairs": len(rows),
        "battles": len({row["battle_id"] for row in rows}),
        "panel_sha256": panel_hash,
        "model_sha256": _sha256(args.model) if args.model else None,
        "learned_weight": args.learned_weight if args.arm == "resource" else 0.0,
        "iterations_per_world": args.iterations,
        "output_sha256": _sha256(args.output),
    }


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bootstrap(values: list[float], seed: int, samples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    estimates = [math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(samples)]
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def report(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    arm_rows = {arm: {row["pair_id"]: row for row in _read(path)} for arm, path in (("hand", args.hand), ("resource", args.resource))}
    roots = {row["root_id"]: row for row in panel}
    units = []
    for root_id, root in roots.items():
        for schedule in root["schedules"]:
            pair_id = f"{root_id}:{schedule['schedule_id']}"
            reference = oracle[pair_id]
            action_values = reference["action_values"]
            row = {
                "battle_id": root["battle_id"],
                "pair_id": pair_id,
                "stratum": root["resource_stratum"],
                "oracle_action": reference["oracle_action"],
            }
            for arm, action in (
                ("historical", root["historical_500ms_action"]),
                ("hand", arm_rows["hand"][pair_id]["selected_action"]),
                ("resource", arm_rows["resource"][pair_id]["selected_action"]),
            ):
                if action not in action_values:
                    raise ValueError(f"{arm} action is absent from corrected oracle: {action}")
                row[f"{arm}_action"] = action
                row[f"{arm}_regret"] = reference["oracle_best_value"] - action_values[action]
            units.append(row)

    metrics = {}
    for arm in ("historical", "hand", "resource"):
        regrets = [row[f"{arm}_regret"] for row in units]
        metrics[arm] = {
            "mean_regret": math.fsum(regrets) / len(regrets),
            "oracle_top1": sum(row[f"{arm}_action"] == row["oracle_action"] for row in units) / len(units),
            "tera_rate": sum(row[f"{arm}_action"].endswith("-tera") for row in units) / len(units),
            "catastrophic_regret_count": sum(value >= 0.10 for value in regrets),
        }
    battle_deltas = []
    for battle_id in sorted({row["battle_id"] for row in units}):
        rows = [row for row in units if row["battle_id"] == battle_id]
        battle_deltas.append(
            math.fsum(row["hand_regret"] - row["resource_regret"] for row in rows) / len(rows)
        )
    interval = _bootstrap(battle_deltas, args.seed)
    metrics["resource_vs_hand"] = {
        "mean_regret_improvement": math.fsum(battle_deltas) / len(battle_deltas),
        "battle_bootstrap_ci95": interval,
        "changed_units": sum(row["resource_action"] != row["hand_action"] for row in units),
    }
    gate = {
        "beats_historical_500ms": metrics["resource"]["mean_regret"] < metrics["historical"]["mean_regret"],
        "beats_equal_iteration_hand": metrics["resource"]["mean_regret"] < metrics["hand"]["mean_regret"],
        "positive_battle_ci": interval[0] > 0.0,
        "no_added_catastrophes_vs_hand": metrics["resource"]["catastrophic_regret_count"] <= metrics["hand"]["catastrophic_regret_count"],
    }
    result = {
        "schema": REPORT_SCHEMA,
        "panel_sha256": panel_hash,
        "oracle_sha256": oracle_hash,
        "hand_sha256": _sha256(args.hand),
        "resource_sha256": _sha256(args.resource),
        "pairs": len(units),
        "battles": len(roots),
        "metrics": metrics,
        "gate": gate,
        "passed": all(gate.values()),
        "next": "run_frozen_holdout" if all(gate.values()) else "stop_before_holdout_h2h_and_distillation",
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["report_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    arm = sub.add_parser("arm")
    arm.add_argument("--panel", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--arm", choices=("hand", "resource"), required=True)
    arm.add_argument("--model", type=Path)
    arm.add_argument("--learned-weight", type=float, default=0.25)
    arm.add_argument("--iterations", type=int, default=20_000)
    arm.add_argument("--workers", type=int, default=8)
    score = sub.add_parser("report")
    score.add_argument("--panel", type=Path, required=True)
    score.add_argument("--oracle", type=Path, required=True)
    score.add_argument("--hand", type=Path, required=True)
    score.add_argument("--resource", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    print(json.dumps(run_arm(args) if args.mode == "arm" else report(args), sort_keys=True))


if __name__ == "__main__":
    main()
