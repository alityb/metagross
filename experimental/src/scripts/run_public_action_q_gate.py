#!/usr/bin/env python3
"""Run an equal-budget root-prior A/B using a frozen public action-Q model."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from eval.action_q_root_gate import SCHEMA
from scripts.run_public_mcts_leaf_gate import (
    _load_oracle,
    _load_panel,
    _seed,
    _sha256,
    _write_jsonl,
)
from train.public_action_q import (
    action_features,
    load_model,
    load_move_database,
    predict_priors,
)


def run(args: argparse.Namespace) -> dict:
    if args.budget_ms != 500 or args.prior_temperature != 0.05 or args.c_puct != 2.0:
        raise ValueError("action-Q gate configuration differs from the frozen protocol")
    os.environ.pop("METAGROSS_VALUE_MODEL", None)
    os.environ.pop("METAGROSS_LEARNED_VALUE_WEIGHT", None)
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    model_hash = None
    model = mean = std = move_database = None
    if args.arm == "candidate":
        if args.model is None or not args.model.is_file():
            raise ValueError("candidate arm requires an action-Q model")
        model_hash = _sha256(args.model)
        model, mean, std, metadata = load_model(args.model)
        if metadata.get("prior_temperature") != 0.05 or metadata.get("c_puct") != 2.0:
            raise ValueError("model metadata differs from frozen deployment")
        move_database = load_move_database(args.move_database)
    import poke_engine

    rows = []
    for root in panel:
        for schedule in root["schedules"]:
            pair_id = f"{root['root_id']}:{schedule['schedule_id']}"
            reference = oracle.get(pair_id)
            if reference is None:
                raise ValueError("panel pair is missing from oracle")
            actions = sorted(reference["action_values"])
            started = time.perf_counter()
            per_world_ms = max(1, int(args.budget_ms * 0.72 / len(schedule["worlds"])))
            visits: dict[str, float] = {}
            guidance_queries = 0
            legal_action_queries = 0
            total_visits = 0
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                priors = None
                legal_action_queries += len(actions)
                if args.arm == "candidate":
                    features = np.stack([
                        action_features(
                            state,
                            action,
                            poke_engine=poke_engine,
                            move_database=move_database,
                        )
                        for action in actions
                    ])
                    priors = predict_priors(
                        model, mean, std, features, actions, args.prior_temperature
                    )
                    if len(priors) != len(actions) or not math.isclose(
                        math.fsum(probability for _, probability in priors), 1.0, abs_tol=1e-6
                    ):
                        raise ValueError("action-Q prior does not cover legal support")
                    guidance_queries += len(priors)
                result = poke_engine.monte_carlo_tree_search(
                    state,
                    duration_ms=per_world_ms,
                    threads=1,
                    s1_priors=priors,
                    c_puct=args.c_puct,
                    seed=_seed(pair_id, int(world["world_index"]), "tree"),
                )
                weight = float(world["weight"])
                denominator = max(1, result.total_visits)
                total_visits += result.total_visits
                for option in result.side_one:
                    visits[option.move_choice] = visits.get(option.move_choice, 0.0) + (
                        weight * option.visits / denominator
                    )
            elapsed = (time.perf_counter() - started) * 1000.0
            selected = max(visits, key=lambda action: (visits[action], action))
            values = reference["action_values"]
            if selected not in values:
                raise ValueError("selected action is absent from oracle")
            rows.append({
                "schema": SCHEMA,
                "battle_id": root["battle_id"],
                "root_id": root["root_id"],
                "pair_id": pair_id,
                "arm": args.arm,
                "budget_ms": args.budget_ms,
                "elapsed_ms": elapsed,
                "selected_action": selected,
                "oracle_action": reference["oracle_action"],
                "oracle_best_value": reference["oracle_best_value"],
                "selected_oracle_value": values[selected],
                "oracle_artifact_sha256": oracle_hash,
                "action_q_model_sha256": model_hash,
                "guidance_queries": guidance_queries,
                "legal_action_queries": legal_action_queries,
                "total_mcts_visits": total_visits,
                "prior_temperature": args.prior_temperature if args.arm == "candidate" else None,
                "c_puct": args.c_puct if args.arm == "candidate" else None,
                "panel_sha256": panel_hash,
            })
    _write_jsonl(args.output, rows)
    return {
        "arm": args.arm,
        "pairs": len(rows),
        "battles": len({row["battle_id"] for row in rows}),
        "panel_sha256": panel_hash,
        "oracle_sha256": oracle_hash,
        "model_sha256": model_hash,
        "output_sha256": _sha256(args.output),
        "guidance_queries": sum(row["guidance_queries"] for row in rows),
        "legal_action_queries": sum(row["legal_action_queries"] for row in rows),
        "total_mcts_visits": sum(row["total_mcts_visits"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--move-database", type=Path, required=True)
    parser.add_argument("--budget-ms", type=int, default=500)
    parser.add_argument("--prior-temperature", type=float, default=0.05)
    parser.add_argument("--c-puct", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
