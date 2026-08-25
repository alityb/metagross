#!/usr/bin/env python3
"""Measure the same-evaluator ceiling and compute cost of ambiguity-triggered 50k deepening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from train.outcome_grounded import bootstrap_ci, stable_u64
from train.shallow_search_residual import battle_split, is_ambiguous


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(args: argparse.Namespace) -> dict:
    shallow = [json.loads(line) for line in args.shallow.read_text().splitlines() if line.strip()]
    oracle = {row["pair_id"]: row for row in map(json.loads, args.oracle.open())}
    split_reports = {}
    for split in ("train", "calibration", "test"):
        rows = []
        for search in shallow:
            if battle_split(search["battle_id"]) != split:
                continue
            teacher = oracle[search["pair_id"]]
            baseline = search["selected_action"]
            triggered = is_ambiguous(search["root_statistics"])
            adaptive = teacher["oracle_action"] if triggered else baseline
            values = teacher["action_values"]
            rows.append({
                "battle_id": search["battle_id"],
                "triggered": triggered,
                "changed": adaptive != baseline,
                "baseline_regret": teacher["oracle_best_value"] - values[baseline],
                "adaptive_regret": teacher["oracle_best_value"] - values[adaptive],
            })
        by_battle = {}
        for row in rows:
            by_battle.setdefault(row["battle_id"], []).append(row)
        deltas = [
            math.fsum(row["baseline_regret"] - row["adaptive_regret"] for row in current) / len(current)
            for current in by_battle.values()
        ]
        ambiguity_rate = sum(row["triggered"] for row in rows) / len(rows)
        split_reports[split] = {
            "battles": len(by_battle),
            "units": len(rows),
            "ambiguity_rate": ambiguity_rate,
            "changed_units": sum(row["changed"] for row in rows),
            "baseline_mean_regret": math.fsum(row["baseline_regret"] for row in rows) / len(rows),
            "ideal_adaptive_mean_regret": math.fsum(row["adaptive_regret"] for row in rows) / len(rows),
            "ideal_improvement_ci95": bootstrap_ci(deltas, stable_u64(args.seed, split) % (2**32)),
            "average_iterations_if_tree_continuable": 20_000 + ambiguity_rate * 30_000,
            "average_iterations_if_50k_rerun": 20_000 + ambiguity_rate * 50_000,
            "continuable_compute_ratio": (20_000 + ambiguity_rate * 30_000) / 20_000,
            "rerun_compute_ratio": (20_000 + ambiguity_rate * 50_000) / 20_000,
        }
    report = {
        "schema": "metagross-adaptive-deepening-ceiling/v1",
        "estimand": "idealized same-evaluator upper bound; uses 50k oracle action when frozen 20k ambiguity fires",
        "not_a_candidate_gate": True,
        "shallow_sha256": sha256(args.shallow),
        "oracle_sha256": sha256(args.oracle),
        "splits": split_reports,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    report = analyze(args)
    print(json.dumps(report["splits"], sort_keys=True))


if __name__ == "__main__":
    main()
