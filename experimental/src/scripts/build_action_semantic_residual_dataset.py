#!/usr/bin/env python3
"""Build belief-aggregated, leak-free action-semantic development examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train.action_semantic_residual import (
    ENRICHED_FEATURE_NAMES, SCHEMA, aggregate_search_features, development_freeze,
    json_dump, residualize_semantics, sha256, summarize_semantics, transition_signals,
)
from train.outcome_grounded import stable_u64, stable_uniform, weighted_choice


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--analysis-8", type=Path, required=True)
    parser.add_argument("--analysis-16", type=Path, required=True)
    parser.add_argument(
        "--label-mode",
        choices=("intersection", "second_stable", "all_second_stable"),
        default="intersection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--root-iterations", type=int, default=20_000)
    parser.add_argument("--probes", type=int, default=4)
    args = parser.parse_args()

    import poke_engine

    panel = read_jsonl(args.panel)
    search = read_jsonl(args.search)
    analysis8 = json.loads(args.analysis_8.read_text())
    analysis16 = json.loads(args.analysis_16.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = development_freeze(panel, args.panel.resolve())
    json_dump(args.output_dir / "development-freeze.json", freeze)

    search_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in search:
        search_by_root.setdefault(str(row["root_id"]), []).append(row)
    old_stable = {(row["root_id"], row["stable_action"]) for row in analysis8["root_results"] if row["stable_action"]}
    new_stable = {(row["root_id"], row["stable_action"]) for row in analysis16["root_results"] if row["stable_action"]}
    all_new_stable = {
        (row["root_id"], alternative["action"])
        for row in analysis16["root_results"]
        for alternative in row["alternatives"]
        if alternative["stable_correction"]
    }
    persistent = (
        old_stable & new_stable
        if args.label_mode == "intersection"
        else all_new_stable if args.label_mode == "all_second_stable" else new_stable
    )
    outcome = {
        (root["root_id"], alt["action"]): float(alt["mean_advantage"])
        for root in analysis16["root_results"] for alt in root["alternatives"]
    }

    examples = []
    for root_index, root in enumerate(panel, 1):
        root_search_rows = sorted(search_by_root[root["root_id"]], key=lambda row: row["schedule_id"])
        if len(root_search_rows) != 2:
            raise ValueError(f"root {root['root_id']} does not have exactly two live-search schedules")
        transition_by_action = {action: [] for action in root["candidate_actions"]}
        policy_by_action = {action: [] for action in root["candidate_actions"]}
        values_by_action = {action: [] for action in root["candidate_actions"]}
        attack_by_action = {action: False for action in root["candidate_actions"]}
        for schedule in root["schedules"]:
            live = next(row for row in root_search_rows if int(row["schedule_id"]) == int(schedule["schedule_id"]))
            live_world = {int(row["world_index"]): row for row in live["worlds"]}
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                search_result = poke_engine.monte_carlo_tree_search(
                    state, duration_ms=0, iterations=args.root_iterations, threads=1,
                    seed=stable_u64(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], "root") % 2**32,
                )
                opponent_policy = [(str(option.move_choice), float(option.visits)) for option in search_result.side_two]
                before_resource = poke_engine.compute_resource_features(state)
                before_value = poke_engine.compute_value_features(state)
                live_stats = live_world[int(world["world_index"])]
                for action in root["candidate_actions"]:
                    policy_by_action[action].append(float(live_stats["policy"].get(action, 0.0)))
                    values_by_action[action].append(float(live_stats["values"].get(action, 0.0)))
                    for probe in range(args.probes):
                        opponent_action = weighted_choice(opponent_policy, stable_uniform(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], probe, "semantic-opponent"))
                        if not action.startswith("switch "):
                            damage = poke_engine.calculate_damage(state, action.removesuffix("-tera"), opponent_action.removesuffix("-tera"), True)[0]
                            attack_by_action[action] = attack_by_action[action] or any(float(value) > 0 for value in damage)
                        chance = stable_uniform(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], probe, "semantic-chance")
                        after = poke_engine.step_with_uniform(poke_engine.State.from_string(world["state"]), action, opponent_action, chance)[0]
                        transition_by_action[action].append(transition_signals(before_resource, poke_engine.compute_resource_features(after), before_value, poke_engine.compute_value_features(after), is_switch=action.startswith("switch ")))
        absolute_semantics = {
            action: summarize_semantics(
                action, transition_by_action[action], policy_by_action[action], values_by_action[action],
                is_attack=attack_by_action[action],
            )
            for action in root["candidate_actions"]
        }
        for action in root["candidate_actions"]:
            if action == root["baseline_action"]:
                continue
            baseline_features = aggregate_search_features(root_search_rows, root, action)
            semantic_features = residualize_semantics(absolute_semantics[action], absolute_semantics[root["baseline_action"]])
            advantage = outcome[(root["root_id"], action)]
            examples.append({
                "schema": SCHEMA,
                "root_id": root["root_id"], "battle_id": root["battle_id"],
                "action": action, "baseline_action": root["baseline_action"],
                "persistent_correction": (root["root_id"], action) in persistent,
                "durable_correction": (root["root_id"], action) in persistent,
                "outcome_advantage": advantage,
                "harmful": advantage < 0.0,
                "baseline_features": baseline_features,
                "semantic_features": semantic_features,
                "enriched_features": [*baseline_features, *semantic_features],
            })
        print(f"semantic roots {root_index}/{len(panel)}", flush=True)

    output = args.output_dir / "dataset.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in examples))
    report = {
        "schema": "metagross-action-semantic-dataset-report/v1",
        "claim_status": "development_only_not_final_confirmation",
        "panel_sha256": sha256(args.panel), "search_sha256": sha256(args.search),
        "analysis_8_sha256": sha256(args.analysis_8), "analysis_16_sha256": sha256(args.analysis_16),
        "engine_binary_sha256": sha256(Path(poke_engine.poke_engine.__file__)),
        "dataset_sha256": sha256(output), "roots": len(panel), "examples": len(examples),
        "persistent_corrections": len(persistent), "harmful_alternatives": sum(row["harmful"] for row in examples),
        "label_mode": args.label_mode,
        "root_iterations": args.root_iterations, "semantic_probes": args.probes,
        "serialization_contract": "only_mean_std_lower_tail_across_hidden_worlds_no_per_world_transition_features",
        "feature_count_baseline": len(examples[0]["baseline_features"]),
        "feature_count_enriched": len(ENRICHED_FEATURE_NAMES),
    }
    json_dump(args.output_dir / "dataset-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
