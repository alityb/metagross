#!/usr/bin/env python3
"""Compare matched outcome-teacher stability at 8 and 16 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from train.outcome_grounded import RESULT_SCHEMA


SCHEMA = "metagross-outcome-rollout-stability-diagnostic/v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != RESULT_SCHEMA for row in rows):
        raise ValueError(f"invalid outcome rows: {path}")
    return rows


def _keyed_samples(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], tuple[Any, int]]:
    keyed: dict[tuple[Any, ...], tuple[Any, int]] = {}
    for row in rows:
        for action in row["candidate_actions"]:
            for sample in row["action_outcomes"][action]:
                key = (
                    row["root_id"], int(row["schedule_id"]), action,
                    int(sample["world_index"]), int(sample["rollout"]),
                )
                if key in keyed:
                    raise ValueError(f"duplicate outcome sample: {key}")
                keyed[key] = (sample.get("outcome"), int(sample["decisions"]))
    return keyed


def verify_prefix(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, int]:
    old = _keyed_samples(old_rows)
    new = _keyed_samples(new_rows)
    missing = [key for key in old if key not in new]
    changed = [key for key, value in old.items() if new.get(key) != value]
    if missing or changed:
        raise ValueError(
            f"16-rollout result does not reproduce the frozen prefix: "
            f"missing={len(missing)} changed={len(changed)}"
        )
    if set(key[0] for key in old) != set(key[0] for key in new):
        raise ValueError("8- and 16-rollout root sets differ")
    if {key[4] for key in old} != set(range(8)):
        raise ValueError("frozen result does not contain rollout indices 0 through 7")
    if {key[4] for key in new} != set(range(16)):
        raise ValueError("diagnostic result does not contain rollout indices 0 through 15")
    return {"prefix_samples": len(old), "missing": 0, "changed": 0}


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _best(means: dict[str, float | None]) -> str | None:
    available = [action for action, value in means.items() if value is not None]
    return max(available, key=lambda action: (float(means[action]), action)) if available else None


def root_diagnostics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["root_id"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for root_id, schedules in grouped.items():
        schedules.sort(key=lambda row: int(row["schedule_id"]))
        if len(schedules) != 2 or {int(row["schedule_id"]) for row in schedules} != {0, 1}:
            raise ValueError(f"root lacks two schedules: {root_id}")
        actions = list(schedules[0]["candidate_actions"])
        if any(row["candidate_actions"] != actions for row in schedules):
            raise ValueError(f"candidate actions differ across schedules: {root_id}")
        baseline = str(schedules[0]["baseline_action"])
        all_means: dict[str, float | None] = {}
        half_means: list[dict[str, float | None]] = [{}, {}]
        schedule_means: list[dict[str, float | None]] = []
        total = terminal = 0
        for action in actions:
            samples = [sample for row in schedules for sample in row["action_outcomes"][action]]
            values = [float(sample["outcome"]) for sample in samples if sample.get("outcome") is not None]
            all_means[action] = _mean(values)
            total += len(samples)
            terminal += len(values)
            for parity in (0, 1):
                half_means[parity][action] = _mean([
                    float(sample["outcome"])
                    for sample in samples
                    if sample.get("outcome") is not None and int(sample["rollout"]) % 2 == parity
                ])
        for row in schedules:
            schedule_means.append({
                action: _mean([
                    float(sample["outcome"])
                    for sample in row["action_outcomes"][action]
                    if sample.get("outcome") is not None
                ])
                for action in actions
            })
        aggregate_best = _best(all_means)
        ordered = sorted(
            (float(value), action) for action, value in all_means.items() if value is not None
        )
        top_margin = ordered[-1][0] - ordered[-2][0] if len(ordered) >= 2 else None
        cluster_best: list[str] = []
        cluster_values: list[float] = []
        for row in schedules:
            for world in range(8):
                means = {
                    action: _mean([
                        float(sample["outcome"])
                        for sample in row["action_outcomes"][action]
                        if int(sample["world_index"]) == world and sample.get("outcome") is not None
                    ])
                    for action in actions
                }
                best = _best(means)
                if best is not None:
                    cluster_best.append(best)
                if aggregate_best is not None and means[aggregate_best] is not None:
                    cluster_values.append(float(means[aggregate_best]))
        half_best = [_best(means) for means in half_means]
        schedule_best = [_best(means) for means in schedule_means]
        result[root_id] = {
            "battle_id": schedules[0]["battle_id"],
            "baseline_action": baseline,
            "aggregate_best_action": aggregate_best,
            "aggregate_best_type": (
                "switch" if aggregate_best is not None and aggregate_best.startswith("switch ") else "move"
            ),
            "baseline_type": "switch" if baseline.startswith("switch ") else "move",
            "half_best_actions": half_best,
            "half_agreement": half_best[0] is not None and half_best[0] == half_best[1],
            "schedule_best_actions": schedule_best,
            "schedule_agreement": schedule_best[0] is not None and schedule_best[0] == schedule_best[1],
            "terminal_rate": terminal / total,
            "top_margin": top_margin,
            "world_top_vote_mass": (
                sum(action == aggregate_best for action in cluster_best) / len(cluster_best)
                if aggregate_best is not None and cluster_best else None
            ),
            "aggregate_best_cluster_std": (
                statistics.pstdev(cluster_values) if len(cluster_values) >= 2 else 0.0
            ),
            "aggregate_best_advantage_over_baseline": (
                float(all_means[aggregate_best]) - float(all_means[baseline])
                if aggregate_best is not None
                and all_means[aggregate_best] is not None
                and all_means[baseline] is not None else None
            ),
        }
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    metrics = {}
    for name in (
        "terminal_rate", "top_margin", "world_top_vote_mass",
        "aggregate_best_cluster_std", "aggregate_best_advantage_over_baseline",
    ):
        current = values(name)
        metrics[name] = {
            "mean": _mean(current),
            "median": statistics.median(current) if current else None,
        }
    return {
        "roots": len(rows),
        "schedule_agreement_rate": (
            sum(bool(row["schedule_agreement"]) for row in rows) / len(rows) if rows else None
        ),
        "baseline_is_aggregate_best_rate": (
            sum(row["baseline_action"] == row["aggregate_best_action"] for row in rows) / len(rows)
            if rows else None
        ),
        "aggregate_best_type": {
            kind: sum(row["aggregate_best_type"] == kind for row in rows)
            for kind in ("move", "switch")
        },
        "metrics": metrics,
    }


def _stable(analysis: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["root_id"]): str(row["stable_action"])
        for row in analysis["root_results"] if row.get("stable_action") is not None
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    old_rows = read_rows(args.old_results)
    new_rows = read_rows(args.new_results)
    prefix = verify_prefix(old_rows, new_rows)
    old = root_diagnostics(old_rows)
    new = root_diagnostics(new_rows)
    if set(old) != set(new):
        raise ValueError("root diagnostics differ in membership")
    old_disagree = {root_id for root_id, row in old.items() if not row["half_agreement"]}
    old_agree = set(old) - old_disagree
    new_disagree = {root_id for root_id, row in new.items() if not row["half_agreement"]}
    new_agree = set(new) - new_disagree
    old_analysis = json.loads(args.old_analysis.read_text())
    new_analysis = json.loads(args.new_analysis.read_text())
    stable_old = _stable(old_analysis)
    stable_new = _stable(new_analysis)
    stabilized = sorted(old_disagree & new_agree)
    destabilized = sorted(old_agree & new_disagree)
    remained_unstable = sorted(old_disagree & new_disagree)
    remained_stable = sorted(old_agree & new_agree)
    old_disagreement_rows = [old[root_id] for root_id in sorted(old_disagree)]
    old_agreement_rows = [old[root_id] for root_id in sorted(old_agree)]
    report = {
        "schema": SCHEMA,
        "claim_status": "post_gate_same_roots_diagnostic_not_confirmation",
        "old_results_sha256": sha256(args.old_results),
        "new_results_sha256": sha256(args.new_results),
        "old_analysis_sha256": sha256(args.old_analysis),
        "new_analysis_sha256": sha256(args.new_analysis),
        "prefix_reproduction": prefix,
        "roots": len(old),
        "rollouts": {"old": 8, "new": 16},
        "half_split": {
            "old_agreements": len(old_agree),
            "old_rate": len(old_agree) / len(old),
            "new_agreements": len(new_agree),
            "new_rate": len(new_agree) / len(new),
            "stabilized_from_old_disagreement": len(stabilized),
            "destabilized_from_old_agreement": len(destabilized),
            "remained_unstable": len(remained_unstable),
            "remained_stable": len(remained_stable),
            "mechanically_stabilized_at_16": len(new_agree) / len(new) >= 0.70,
        },
        "stable_corrections": {
            "old": len(stable_old),
            "new": len(stable_new),
            "same_root_and_action": len(set(stable_old.items()) & set(stable_new.items())),
            "old_roots_retained": len(set(stable_old) & set(stable_new)),
            "new_only_roots": len(set(stable_new) - set(stable_old)),
            "lost_old_roots": len(set(stable_old) - set(stable_new)),
        },
        "old_groups": {
            "half_agreement": _summary(old_agreement_rows),
            "half_disagreement": _summary(old_disagreement_rows),
        },
        "root_transitions": {
            "stabilized": [{"root_id": root_id, "old": old[root_id], "new": new[root_id]} for root_id in stabilized],
            "destabilized": [{"root_id": root_id, "old": old[root_id], "new": new[root_id]} for root_id in destabilized],
            "remained_unstable": [{"root_id": root_id, "old": old[root_id], "new": new[root_id]} for root_id in remained_unstable],
        },
        "recommended_next_step": (
            "new_disjoint_16_rollout_confirmation_then_action_semantic_abstaining_residual"
            if len(new_agree) / len(new) >= 0.70
            else "change_continuation_teacher_or_uncertainty_model_before_more_policy_data"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--old-analysis", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--new-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = diagnose(parser.parse_args())
    print(json.dumps({
        "roots": report["roots"],
        "half_split": report["half_split"],
        "stable_corrections": report["stable_corrections"],
        "recommended_next_step": report["recommended_next_step"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
