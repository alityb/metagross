#!/usr/bin/env python3
"""Build the frozen belief-aggregated candidate-switch development dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train.action_semantic_residual import json_dump, sha256
from train.candidate_switch_residual import (
    CANDIDATE_FEATURE_NAMES, SCHEMA, matchup_features, residual_features,
    static_features, summarize_matchups, switch_target, type_vector,
)
from train.outcome_grounded import stable_u64, stable_uniform, weighted_choice


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-iterations", type=int, default=2_048)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    import poke_engine

    panel = read_jsonl(args.panel)
    source = read_jsonl(args.source_dataset)
    if len(panel) != 200 or len(source) != 383:
        raise ValueError(f"frozen corpus size changed: panel={len(panel)} rows={len(source)}")
    if sum(bool(row["durable_correction"]) for row in source) != 56:
        raise ValueError("frozen durable action-label count changed")
    if sum(bool(row["harmful"]) for row in source) != 155:
        raise ValueError("frozen harmful-alternative count changed")
    source_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        source_by_root.setdefault(str(row["root_id"]), []).append(row)
    if set(source_by_root) != {str(row["root_id"]) for row in panel}:
        raise ValueError("source dataset and panel roots differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for root_index, root in enumerate(panel, 1):
        source_rows = source_by_root[str(root["root_id"])]
        actions = sorted({str(root["baseline_action"]), *(str(row["action"]) for row in source_rows)})
        type_rows: dict[str, list[list[float]]] = {action: [] for action in actions}
        static_rows: dict[str, list[list[float]]] = {action: [] for action in actions}
        matchups: dict[str, list[list[float]]] = {action: [] for action in actions}
        for schedule in root["schedules"]:
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                result = poke_engine.monte_carlo_tree_search(
                    state, duration_ms=0, iterations=args.root_iterations, threads=1,
                    seed=stable_u64(
                        args.seed, root["root_id"], schedule["schedule_id"],
                        world["world_index"], "candidate-root",
                    ) % 2**32,
                )
                opponent_policy = [
                    (str(option.move_choice), float(option.visits))
                    for option in result.side_two
                ]
                for action in actions:
                    target = switch_target(state, action)
                    type_rows[action].append(type_vector(target))
                    static_rows[action].append(static_features(target))
                for probe in range(args.probes):
                    opponent_action = weighted_choice(
                        opponent_policy,
                        stable_uniform(
                            args.seed, root["root_id"], schedule["schedule_id"],
                            world["world_index"], probe, "candidate-opponent",
                        ),
                    )
                    chance = stable_uniform(
                        args.seed, root["root_id"], schedule["schedule_id"],
                        world["world_index"], probe, "candidate-chance",
                    )
                    for action in actions:
                        state_copy = poke_engine.State.from_string(world["state"])
                        before_target = switch_target(state_copy, action)
                        after = poke_engine.step_with_uniform(
                            state_copy, action, opponent_action, chance,
                        )[0]
                        matchups[action].append(
                            matchup_features(before_target, after, poke_engine=poke_engine)
                        )
        absolute = {}
        for action in actions:
            types = np.asarray(type_rows[action], dtype=float)
            statics = np.asarray(static_rows[action], dtype=float)
            # Side one's full team is known and must not vary with a hidden-world sample.
            if not np.allclose(types, types[0], atol=0.0, rtol=0.0):
                raise ValueError(f"own candidate types vary across worlds: {root['root_id']} {action}")
            if not np.allclose(statics, statics[0], atol=0.0, rtol=0.0):
                raise ValueError(f"own candidate state varies across worlds: {root['root_id']} {action}")
            absolute[action] = (types[0].tolist(), statics[0].tolist(), summarize_matchups(matchups[action]))
        baseline = absolute[str(root["baseline_action"])]
        for source_row in source_rows:
            candidate = absolute[str(source_row["action"])]
            candidate_features = residual_features(
                candidate[0], baseline[0], candidate[1], baseline[1], candidate[2], baseline[2],
            )
            output_rows.append({
                "schema": SCHEMA,
                "root_id": source_row["root_id"],
                "battle_id": source_row["battle_id"],
                "action": source_row["action"],
                "baseline_action": source_row["baseline_action"],
                "durable_correction": source_row["durable_correction"],
                "persistent_correction": source_row["persistent_correction"],
                "outcome_advantage": source_row["outcome_advantage"],
                "harmful": source_row["harmful"],
                "search_features": source_row["baseline_features"],
                "candidate_features": candidate_features,
                "enriched_features": [*source_row["baseline_features"], *candidate_features],
            })
        print(f"candidate-switch roots {root_index}/{len(panel)}", flush=True)

    output = args.output_dir / "dataset.jsonl"
    output.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in output_rows
    ))
    report = {
        "schema": "metagross-candidate-switch-dataset-report/v1",
        "claim_status": "development_only_not_final_confirmation",
        "panel_sha256": sha256(args.panel),
        "source_dataset_sha256": sha256(args.source_dataset),
        "engine_binary_sha256": sha256(Path(poke_engine.poke_engine.__file__)),
        "dataset_sha256": sha256(output),
        "roots": len(panel),
        "battles": len({row["battle_id"] for row in output_rows}),
        "examples": len(output_rows),
        "durable_corrections": sum(bool(row["durable_correction"]) for row in output_rows),
        "harmful_alternatives": sum(bool(row["harmful"]) for row in output_rows),
        "root_iterations": args.root_iterations,
        "opponent_probes": args.probes,
        "search_feature_count": len(output_rows[0]["search_features"]),
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "serialization_contract": (
            "relative_owned_candidate_and_mean_lower_tail_across_causal_worlds;"
            "no_species_identity_no_opponent_identity_no_per_world_features"
        ),
    }
    json_dump(args.output_dir / "dataset-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
