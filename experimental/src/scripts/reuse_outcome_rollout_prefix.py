#!/usr/bin/env python3
"""Reuse an exactly keyed rollout prefix when panel/action/config identities match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.collect_outcome_grounded_continuations import read_panel, sha256, write_private
from train.outcome_grounded import RESULT_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    panel = read_panel(args.panel)
    expected = {(row["root_id"], int(schedule["schedule_id"])): row for row in panel for schedule in row["schedules"]}
    source = [json.loads(line) for line in args.source_results.read_text().splitlines() if line.strip()]
    reused = []
    rejected = {"not_in_panel": 0, "identity_or_actions": 0, "configuration": 0, "short_prefix": 0}
    for row in source:
        key = (str(row.get("root_id")), int(row.get("schedule_id", -1)))
        root = expected.get(key)
        if root is None:
            rejected["not_in_panel"] += 1
            continue
        if (
            row.get("schema") != RESULT_SCHEMA
            or row.get("battle_id") != root["battle_id"]
            or row.get("baseline_action") != root["baseline_action"]
            or row.get("candidate_actions") != root["candidate_actions"]
        ):
            rejected["identity_or_actions"] += 1
            continue
        configuration = row.get("configuration", {})
        if (
            configuration.get("root_iterations") != 20_000
            or configuration.get("continuation_iterations") != 2_048
            or configuration.get("max_decisions") != 128
            or configuration.get("seed") != args.seed
            or int(configuration.get("rollouts", 0)) < args.rollouts
        ):
            rejected["configuration"] += 1
            continue
        action_outcomes: dict[str, list[dict[str, Any]]] = {}
        valid = True
        for action in root["candidate_actions"]:
            samples = [sample for sample in row["action_outcomes"].get(action, []) if int(sample["rollout"]) < args.rollouts]
            expected_samples = len(root["schedules"][key[1]]["worlds"]) * args.rollouts
            if len(samples) != expected_samples:
                valid = False
                break
            action_outcomes[action] = samples
        if not valid:
            rejected["short_prefix"] += 1
            continue
        continuation_searches = sum(max(0, int(sample["decisions"]) - 1) for samples in action_outcomes.values() for sample in samples)
        reused.append({
            **row,
            "action_outcomes": action_outcomes,
            "configuration": {**configuration, "rollouts": args.rollouts},
            "continuation_searches": continuation_searches,
            "prefix_reuse": {
                "source_results_sha256": sha256(args.source_results),
                "rollouts": list(range(args.rollouts)),
                "identity": "root_schedule_action_world_rollout_seed_configuration",
            },
        })
    reused.sort(key=lambda row: (row["root_id"], int(row["schedule_id"])))
    write_private(args.output, reused)
    report = {
        "schema": "metagross-outcome-rollout-prefix-reuse/v1",
        "panel_sha256": sha256(args.panel), "source_results_sha256": sha256(args.source_results),
        "output_sha256": sha256(args.output), "source_rows": len(source), "reused_rows": len(reused),
        "reused_roots": len({row["root_id"] for row in reused}), "rollouts": args.rollouts,
        "seed": args.seed, "rejected": rejected,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
