#!/usr/bin/env python3
"""Frozen nested OOF gate for the prospective switch-to-switch corpus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.evaluate_action_semantic_residual import read_jsonl
from scripts.evaluate_compact_action_semantic_residual import grouped_folds
from scripts.evaluate_specialist_action_residual import action_family, nested_family_oof
from train.action_semantic_residual import json_dump, sha256


SCHEMA = "metagross-targeted-switch-residual-oof/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-compact-report", type=Path, required=True)
    parser.add_argument("--prior-specialist-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    durable = sum(row["durable_correction"] for row in rows)
    if not 50 <= durable <= 100:
        raise ValueError(f"targeted switch gate requires 50-100 durable action corrections, got {durable}")
    families = {action_family(row) for row in rows}
    if families != {"switch_option"} or any(
        not row["action"].startswith("switch ") or not row["baseline_action"].startswith("switch ")
        for row in rows
    ):
        raise ValueError(f"targeted dataset is not switch-to-switch only: {families}")
    outer = grouped_folds(rows, 10, args.seed)
    candidate = nested_family_oof(rows, "switch_option", outer, args.seed)
    result = candidate["metrics"]
    minimum = math.ceil(0.30 * durable)
    compact = json.loads(args.prior_compact_report.read_text())["compact"]["metrics"]
    old_switch = json.loads(args.prior_specialist_report.read_text())["specialists"]["switch_option"]["metrics"]
    beats_compact = result["harmful_overrides"] <= compact["harmful_overrides"] and (
        result["persistent_corrections_identified"] > compact["persistent_corrections_identified"]
        or (result["persistent_corrections_identified"] == compact["persistent_corrections_identified"]
            and result["summed_development_advantage"] > compact["summed_development_advantage"] + 1e-12)
    )
    beats_old_switch = result["harmful_overrides"] <= old_switch["harmful_overrides"] and (
        result["persistent_corrections_identified"] > old_switch["persistent_corrections_identified"]
        or (result["persistent_corrections_identified"] == old_switch["persistent_corrections_identified"]
            and result["summed_development_advantage"] > old_switch["summed_development_advantage"] + 1e-12)
    )
    passed = (result["harmful_overrides"] == 0
              and result["persistent_corrections_identified"] >= minimum
              and beats_compact and beats_old_switch)
    report = {"schema": SCHEMA, "claim_status": "development_only_not_confirmation",
        "dataset_sha256": sha256(args.dataset), "seed": args.seed,
        "correction_unit": "independently_stable_root_action_pair",
        "durable_corrections": durable, "minimum_recovered": minimum,
        "model": "frozen_switch_specific_squared_error_boosted_stumps",
        "grouping": "ten_outer_folds_with_five_fold_inner_battle_grouped_oof",
        "thresholding": "inner_oof_positive_score_maximize_durable_corrections_subject_to_zero_harm",
        "candidate": candidate, "prior_compact_metrics": compact,
        "prior_switch_specialist_metrics": old_switch,
        "strictly_beats_prior_compact": beats_compact,
        "strictly_beats_prior_switch_specialist": beats_old_switch,
        "passed": passed}
    json_dump(args.output, report)
    print(json.dumps({"durable_corrections": durable, "minimum_recovered": minimum,
        "metrics": result, "strictly_beats_prior_compact": beats_compact,
        "strictly_beats_prior_switch_specialist": beats_old_switch,
        "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
