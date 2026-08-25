#!/usr/bin/env python3
"""Analyze matched terminal outcomes and qualify outcome-grounded targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from train.outcome_grounded import RESULT_SCHEMA, bootstrap_ci, stable_u64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(args: argparse.Namespace) -> dict:
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != RESULT_SCHEMA for row in rows):
        raise ValueError("invalid outcome-grounded results")
    by_root = {}
    for row in rows:
        by_root.setdefault(row["root_id"], []).append(row)
    roots = []
    total_samples = terminal_samples = 0
    half_agreements = []
    stable_corrections = 0
    for root_id, schedules in sorted(by_root.items()):
        if len(schedules) != 2 or {row["schedule_id"] for row in schedules} != {0, 1}:
            raise ValueError("outcome root lacks two schedules")
        schedules.sort(key=lambda row: row["schedule_id"])
        baseline = schedules[0]["baseline_action"]
        actions = schedules[0]["candidate_actions"]
        action_clusters = {action: {} for action in actions}
        action_samples = {action: {} for action in actions}
        action_halves = {action: [[], []] for action in actions}
        action_completed = {action: 0 for action in actions}
        action_total = {action: 0 for action in actions}
        schedule_q = {action: [] for action in actions}
        for schedule in schedules:
            for action in actions:
                samples = schedule["action_outcomes"][action]
                completed_values = [float(sample["outcome"]) for sample in samples if sample["outcome"] is not None]
                schedule_q[action].append(math.fsum(completed_values) / len(completed_values) if completed_values else None)
                for sample in samples:
                    action_total[action] += 1; total_samples += 1
                    outcome = sample["outcome"]
                    if outcome is None:
                        continue
                    value = float(outcome)
                    action_completed[action] += 1; terminal_samples += 1
                    key = (int(schedule["schedule_id"]), int(sample["world_index"]))
                    action_clusters[action].setdefault(key, []).append(value)
                    action_samples[action][(*key, int(sample["rollout"]))] = value
                    action_halves[action][int(sample["rollout"]) % 2].append(value)
        q = {}
        for action in actions:
            count = math.fsum(len(values) for values in action_clusters[action].values())
            q[action] = (
                math.fsum(math.fsum(values) for values in action_clusters[action].values()) / count
                if count > 0 else None
            )
        half_best = []
        for half in (0, 1):
            available = [action for action in actions if action_halves[action][half]]
            half_best.append(
                max(
                    available,
                    key=lambda action: (math.fsum(action_halves[action][half]) / len(action_halves[action][half]), action),
                )
                if available else None
            )
        half_agreements.append(half_best[0] is not None and half_best[0] == half_best[1])
        alternatives = []
        for action in actions:
            if action == baseline:
                continue
            paired_keys = sorted(set(action_samples[action]).intersection(action_samples[baseline]))
            paired_by_cluster = {}
            for key in paired_keys:
                cluster = key[:2]
                paired_by_cluster.setdefault(cluster, []).append(
                    action_samples[action][key] - action_samples[baseline][key]
                )
            deltas = [math.fsum(values) / len(values) for values in paired_by_cluster.values()]
            interval = (
                bootstrap_ci(deltas, stable_u64(args.seed, root_id, action) % (2**32))
                if deltas else [None, None]
            )
            schedule_deltas = []
            for schedule_index in range(2):
                values = [
                    action_samples[action][key] - action_samples[baseline][key]
                    for key in paired_keys if key[0] == schedule_index
                ]
                schedule_deltas.append(math.fsum(values) / len(values) if values else None)
            terminal_rate = action_completed[action] / action_total[action]
            baseline_terminal_rate = action_completed[baseline] / action_total[baseline]
            paired_rate = len(paired_keys) / action_total[action]
            stable = (
                terminal_rate >= 0.95
                and baseline_terminal_rate >= 0.95
                and paired_rate >= 0.90
                and bool(deltas)
                and interval[0] is not None
                and interval[0] > 0
                and all(value is not None and math.isfinite(value) and value > 0.01 for value in schedule_deltas)
            )
            alternatives.append({
                "action": action,
                "mean_advantage": math.fsum(deltas) / len(deltas) if deltas else None,
                "cluster_bootstrap_ci95": interval,
                "schedule_advantages": schedule_deltas,
                "terminal_rate": terminal_rate,
                "baseline_terminal_rate": baseline_terminal_rate,
                "paired_terminal_rate": paired_rate,
                "stable_correction": stable,
            })
        stable = [row for row in alternatives if row["stable_correction"]]
        stable_corrections += bool(stable)
        available_q = [action for action in actions if q[action] is not None]
        terminal_best = max(available_q, key=lambda action: (q[action], action)) if available_q else None
        roots.append({
            "root_id": root_id,
            "battle_id": schedules[0]["battle_id"],
            "baseline_action": baseline,
            "terminal_best_action": terminal_best,
            "terminal_q": q,
            "terminal_rate": math.fsum(action_completed.values()) / math.fsum(action_total.values()),
            "half_split_best_agreement": half_agreements[-1],
            "alternatives": alternatives,
            "stable_action": max(stable, key=lambda row: row["mean_advantage"])["action"] if stable else None,
        })
    terminal_rate = terminal_samples / total_samples
    half_agreement = sum(half_agreements) / len(half_agreements)
    raw_disagreements = sum(
        root["terminal_best_action"] is not None
        and root["terminal_best_action"] != root["baseline_action"]
        for root in roots
    )
    admitted = terminal_rate >= 0.95 and half_agreement >= 0.70 and stable_corrections >= 10
    report = {
        "schema": "metagross-outcome-grounded-analysis/v1",
        "results_sha256": sha256(args.results),
        "roots": len(roots),
        "samples": total_samples,
        "terminal_rate": terminal_rate,
        "half_split_best_agreement": half_agreement,
        "raw_terminal_disagreements": raw_disagreements,
        "stable_corrections": stable_corrections,
        "target_admitted_for_scale": admitted,
        "root_results": roots,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    report = analyze(args)
    print(json.dumps({key: report[key] for key in ("roots", "samples", "terminal_rate", "half_split_best_agreement", "raw_terminal_disagreements", "stable_corrections", "target_admitted_for_scale")}, sort_keys=True))
    raise SystemExit(0 if report["target_admitted_for_scale"] else 2)


if __name__ == "__main__":
    main()
