#!/usr/bin/env python3
"""Freeze a label-blind switch-to-switch outcome panel from live search."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from scripts.build_outcome_grounded_panel import read_rows, write_private
from scripts.run_public_mcts_leaf_gate import _load_panel
from train.outcome_grounded import PANEL_SCHEMA, stable_u64
from train.shallow_search_residual import is_ambiguous
from train.action_semantic_residual import sha256


SCHEMA = "metagross-switch-to-switch-panel-report/v1"


def aggregate(rows: list[dict], action: str, field: str) -> float:
    return math.fsum(float(row["action_statistics"][action][field]) for row in rows) / len(rows)


def build(args: argparse.Namespace) -> dict:
    panel, source_hash = _load_panel(args.panel)
    search_rows = read_rows(args.shallow)
    search: dict[str, list[dict]] = collections.defaultdict(list)
    for row in search_rows:
        search[str(row["root_id"])].append(row)

    denied_roots: set[str] = set()
    denied_battles: set[str] = set()
    freeze_hashes = []
    for path in args.development_freeze:
        payload = json.loads(path.read_text())
        denied_roots.update(str(value) for value in payload["root_ids"])
        denied_battles.update(str(value) for value in payload["battle_ids"])
        freeze_hashes.append(sha256(path))
    root_overlap = {str(row["root_id"]) for row in panel} & denied_roots
    battle_overlap = {str(row["battle_id"]) for row in panel} & denied_battles
    if root_overlap or battle_overlap:
        raise ValueError(
            f"source corpus overlaps opened outcome development: roots={len(root_overlap)} battles={len(battle_overlap)}"
        )

    candidates = []
    attrition: collections.Counter[str] = collections.Counter()
    for root in panel:
        schedules = sorted(search.get(str(root["root_id"]), []), key=lambda row: int(row["schedule_id"]))
        if len(schedules) != 2:
            attrition["missing_search_schedule"] += 1; continue
        attrition["complete_search"] += 1
        if not all(is_ambiguous(row["root_statistics"]) for row in schedules):
            attrition["not_ambiguous_both"] += 1; continue
        attrition["ambiguous_both"] += 1
        support = set.intersection(*(set(row["action_statistics"]) for row in schedules))
        if len(support) < 2:
            attrition["insufficient_common_support"] += 1; continue
        ranked = sorted(
            support,
            key=lambda action: (aggregate(schedules, action, "visit_mass"),
                                aggregate(schedules, action, "mean_value"), action),
            reverse=True,
        )
        baseline = ranked[0]
        if not baseline.startswith("switch "):
            attrition["baseline_not_switch"] += 1; continue
        alternatives = [action for action in ranked[1:] if action.startswith("switch ")]
        if not alternatives:
            attrition["no_alternative_switch"] += 1; continue
        attrition["eligible_switch_to_switch"] += 1
        actions = [baseline, *alternatives[:2]]
        ambiguity = math.fsum(
            row["root_statistics"]["weighted_js_divergence"]
            + row["root_statistics"]["weighted_top_action_disagreement"]
            + (1.0 - row["root_statistics"]["aggregate_top_visit_mass"])
            for row in schedules
        ) / 2.0
        candidates.append((ambiguity, stable_u64(args.seed, root["battle_id"]), {
            "schema": PANEL_SCHEMA,
            "battle_id": root["battle_id"], "root_id": root["root_id"],
            "baseline_action": baseline, "teacher_action": baseline,
            "candidate_actions": actions,
            "selection": {
                "purpose": "development_only_prospective_switch_to_switch",
                "outcome_labels_opened_at_selection": False,
                "oracle_50k_consulted": False,
                "both_schedules_ambiguous": True,
                "baseline_rule": "highest_mean_20k_visit_mass_across_two_schedules",
                "candidate_rule": "top_two_nonbaseline_switches_by_mean_visits_value_name",
                "ambiguity_score": ambiguity,
                "schedule_selected_actions": [row["selected_action"] for row in schedules],
            },
            "source_context": {
                "decision_idx": root.get("decision_idx"),
                "public_reveal_fractions": root.get("public_reveal_fractions"),
                "r1_selection": root.get("selection"),
                "causal_history": root.get("causal_history"),
                "source_path": root.get("source_path"), "source_line": root.get("source_line"),
            },
            "schedules": root["schedules"],
        }))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    eligible = len(candidates)
    selected = [item[2] for item in candidates[: args.maximum_roots]]
    write_private(args.output, selected)
    report = {
        "schema": SCHEMA, "purpose": "development_only_never_confirmation",
        "source_panel_sha256": source_hash, "shallow_sha256": sha256(args.shallow),
        "development_freeze_sha256": freeze_hashes,
        "source_roots": len(panel), "source_root_overlap": len(root_overlap),
        "source_battle_overlap": len(battle_overlap), "eligible_roots": eligible,
        "selected_roots": len(selected), "maximum_roots": args.maximum_roots,
        "candidate_actions": sum(len(row["candidate_actions"]) for row in selected),
        "schedules": len(selected) * 2, "worlds": len(selected) * 16,
        "selection_used_oracle_50k": False, "selection_used_terminal_outcomes": False,
        "attrition": dict(attrition), "seed": args.seed,
        "output_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, action="append", required=True)
    parser.add_argument("--maximum-roots", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
