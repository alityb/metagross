#!/usr/bin/env python3
"""Apply the frozen four-rollout promotion rule without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.collect_outcome_grounded_continuations import read_panel, sha256, write_private
from train.outcome_grounded import RESULT_SCHEMA, stable_u64


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else float("nan")


def best(values: dict[str, float]) -> str:
    return max(values, key=lambda action: (values[action], action))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-roots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    panel = read_panel(args.panel)
    results = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError("invalid screen result schema")
        by_root.setdefault(str(row["root_id"]), []).append(row)
    ranked = []
    diagnostics = []
    for root in panel:
        rows = sorted(by_root.get(root["root_id"], []), key=lambda row: int(row["schedule_id"]))
        if len(rows) != 2 or {int(row["schedule_id"]) for row in rows} != {0, 1}:
            raise ValueError(f"incomplete four-rollout screen for {root['root_id']}")
        samples = [sample for row in rows for action in root["candidate_actions"] for sample in row["action_outcomes"][action]]
        if len(samples) != 2 * len(root["candidate_actions"]) * 8 * 4:
            raise ValueError(f"screen sample count changed for {root['root_id']}")
        terminal_rate = sum(sample["outcome"] is not None for sample in samples) / len(samples)
        schedule_q = []
        for row in rows:
            schedule_q.append({
                action: mean([float(sample["outcome"]) for sample in row["action_outcomes"][action] if sample["outcome"] is not None])
                for action in root["candidate_actions"]
            })
        aggregate_q = {action: mean([schedule[action] for schedule in schedule_q]) for action in root["candidate_actions"]}
        aggregate_best = best(aggregate_q)
        baseline = root["baseline_action"]
        advantage = aggregate_q[aggregate_best] - aggregate_q[baseline]
        schedule_advantages = [schedule[aggregate_best] - schedule[baseline] for schedule in schedule_q]
        half_q = []
        for half in ({0, 1}, {2, 3}):
            half_q.append({
                action: mean([
                    float(sample["outcome"])
                    for row in rows for sample in row["action_outcomes"][action]
                    if int(sample["rollout"]) in half and sample["outcome"] is not None
                ])
                for action in root["candidate_actions"]
            })
        half_best = [best(values) for values in half_q]
        schedule_best = [best(values) for values in schedule_q]
        ordered = sorted(aggregate_q.values(), reverse=True)
        top_margin = ordered[0] - ordered[1]
        promising = (
            aggregate_best != baseline and advantage >= 0.03
            and all(value > 0.0 for value in schedule_advantages)
        )
        uncertain = len(set(half_best)) > 1 or len(set(schedule_best)) > 1 or top_margin <= 0.05
        eligible = terminal_rate >= 0.99 and (promising or uncertain)
        diagnostic = {
            "root_id": root["root_id"], "battle_id": root["battle_id"],
            "terminal_rate": terminal_rate, "aggregate_best_action": aggregate_best,
            "baseline_action": baseline, "aggregate_best_advantage": advantage,
            "schedule_advantages": schedule_advantages, "half_best_actions": half_best,
            "schedule_best_actions": schedule_best, "top_margin": top_margin,
            "promising": promising, "uncertain": uncertain, "eligible": eligible,
        }
        diagnostics.append(diagnostic)
        if eligible:
            # Promise first, then effect/uncertainty, then a deterministic battle key.
            ranked.append((int(promising), advantage, -top_margin, stable_u64(args.seed, root["battle_id"]), root))
    ranked.sort(key=lambda item: item[:4], reverse=True)
    promoted = [item[4] for item in ranked[: args.maximum_roots]]
    write_private(args.output, promoted)
    promoted_ids = {row["root_id"] for row in promoted}
    report = {
        "schema": "metagross-staged-outcome-promotion-report/v1",
        "panel_sha256": sha256(args.panel), "results_sha256": sha256(args.results),
        "output_sha256": sha256(args.output), "source_roots": len(panel),
        "eligible_roots": len(ranked), "promoted_roots": len(promoted),
        "promising_promoted": sum(row["promising"] for row in diagnostics if row["root_id"] in promoted_ids),
        "uncertain_promoted": sum(row["uncertain"] for row in diagnostics if row["root_id"] in promoted_ids),
        "maximum_roots": args.maximum_roots, "seed": args.seed,
        "rule": {
            "minimum_terminal_rate": 0.99, "minimum_promising_advantage": 0.03,
            "positive_both_schedules": True, "uncertainty_top_margin_maximum": 0.05,
        },
        "diagnostics": diagnostics,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("source_roots", "eligible_roots", "promoted_roots", "promising_promoted", "uncertain_promoted")}, sort_keys=True))


if __name__ == "__main__":
    main()
