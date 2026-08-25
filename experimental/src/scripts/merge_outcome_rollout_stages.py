#!/usr/bin/env python3
"""Merge disjoint, exactly configured rollout stages into a 0..N result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.collect_outcome_grounded_continuations import read_panel, sha256, write_private
from train.outcome_grounded import RESULT_SCHEMA


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--total-rollouts", type=int, default=16)
    args = parser.parse_args()
    panel = read_panel(args.panel)
    expected = {(row["root_id"], int(schedule["schedule_id"])): row for row in panel for schedule in row["schedules"]}
    prefix = {(row["root_id"], int(row["schedule_id"])): row for row in read(args.prefix) if row["root_id"] in {root["root_id"] for root in panel}}
    suffix = {(row["root_id"], int(row["schedule_id"])): row for row in read(args.suffix)}
    if set(prefix) != set(expected) or set(suffix) != set(expected):
        raise ValueError("rollout stages do not exactly cover the promoted panel")
    merged = []
    for key, root in expected.items():
        first, second = prefix[key], suffix[key]
        if any(
            first.get(name) != second.get(name)
            for name in ("schema", "battle_id", "root_id", "schedule_id", "baseline_action", "candidate_actions")
        ) or first.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"stage identity mismatch: {key}")
        if first["configuration"].get("seed") != second["configuration"].get("seed"):
            raise ValueError(f"stage seed mismatch: {key}")
        outcomes = {}
        for action in root["candidate_actions"]:
            samples = first["action_outcomes"][action] + second["action_outcomes"][action]
            keys = [(int(sample["world_index"]), int(sample["rollout"])) for sample in samples]
            expected_keys = [(world, rollout) for world in range(8) for rollout in range(args.total_rollouts)]
            if sorted(keys) != expected_keys:
                raise ValueError(f"rollout coverage mismatch: {key} {action}")
            outcomes[action] = sorted(samples, key=lambda sample: (int(sample["world_index"]), int(sample["rollout"])))
        merged.append({
            **second, "action_outcomes": outcomes,
            "configuration": {**second["configuration"], "rollouts": args.total_rollouts, "rollout_start": 0},
            "continuation_searches": int(first["continuation_searches"]) + int(second["continuation_searches"]),
            "stage_merge": {"prefix_sha256": sha256(args.prefix), "suffix_sha256": sha256(args.suffix)},
        })
    merged.sort(key=lambda row: (row["root_id"], int(row["schedule_id"])))
    write_private(args.output, merged)
    report = {
        "schema": "metagross-outcome-rollout-stage-merge/v1", "panel_sha256": sha256(args.panel),
        "prefix_sha256": sha256(args.prefix), "suffix_sha256": sha256(args.suffix),
        "output_sha256": sha256(args.output), "rows": len(merged), "rollouts": args.total_rollouts,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
