#!/usr/bin/env python3
"""Development-only threshold sweep for the adaptive search-budget controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from srcs.metagross.known_team_search_failure_attribution import adaptive_budget_decision


SCHEMA = "metagross-adaptive-budget-development/v1"
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
MINIMUM_GAIN_CAPTURE = 0.80
MINIMUM_COMPUTE_SAVING = 0.40
MAXIMUM_HARMFUL = 0
FULL_COST = 5_000 + 20_000 + 80_000


def analyze(attribution: Mapping[str, Any]) -> dict[str, Any]:
    rescued = {
        tuple(root["identity"])
        for root in attribution["roots"]
        if root["attribution"] == "finite_search_budget"
    }
    rows = []
    for threshold in THRESHOLDS:
        total_cost = 0
        selected_gain = 0.0
        fixed_high_gain = 0.0
        harmful = 0
        decisions = 0
        for root in attribution["roots"]:
            for repeat in range(2):
                checkpoints = [
                    (int(budget), root["budgets"][budget]["repeats"][repeat]["visit_policy"])
                    for budget in ("5000", "20000", "80000")
                ]
                stop = adaptive_budget_decision(checkpoints, threshold)
                total_cost += 25_000 if stop == 20_000 else FULL_COST
                delta = float(root["budgets"][str(stop)]["visit"]["teacher_deltas"][repeat])
                harmful += delta <= -0.02
                if tuple(root["identity"]) in rescued:
                    selected_gain += max(0.0, delta)
                    fixed_high_gain += max(
                        0.0,
                        float(root["budgets"]["80000"]["visit"]["teacher_deltas"][repeat]),
                    )
                decisions += 1
        mean_cost = total_cost / decisions
        gain_capture = selected_gain / fixed_high_gain if fixed_high_gain else 0.0
        saving = 1.0 - mean_cost / FULL_COST
        passes = (
            gain_capture >= MINIMUM_GAIN_CAPTURE
            and saving >= MINIMUM_COMPUTE_SAVING
            and harmful <= MAXIMUM_HARMFUL
        )
        rows.append(
            {
                "margin_threshold": threshold,
                "mean_iterations_per_particle": mean_cost,
                "compute_saving_vs_fixed_ladder": saving,
                "fixed_80k_rescue_gain_captured": gain_capture,
                "harmful_repeat_decisions": harmful,
                "passes": passes,
            }
        )
    passing = [row for row in rows if row["passes"]]
    return {
        "schema": SCHEMA,
        "status": "complete_development_only",
        "configuration": {
            "thresholds": list(THRESHOLDS),
            "minimum_gain_capture": MINIMUM_GAIN_CAPTURE,
            "minimum_compute_saving": MINIMUM_COMPUTE_SAVING,
            "maximum_harmful_repeat_decisions": MAXIMUM_HARMFUL,
            "cost_accounting": "cumulative_5k_then_20k_then_80k",
        },
        "rows": rows,
        "decision": {
            "gate_passed": bool(passing),
            "selected_threshold": passing[0]["margin_threshold"] if passing else None,
            "next": "confirmation_panel" if passing else "information_state_value_dataset",
            "public_ladder_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.attribution.read_text(encoding="utf-8"))
    args.output.write_text(json.dumps(analyze(payload), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
