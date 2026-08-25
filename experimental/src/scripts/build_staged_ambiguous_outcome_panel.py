#!/usr/bin/env python3
"""Freeze all two-schedule ambiguous roots for staged outcome screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.build_outcome_grounded_panel import read_rows, write_private
from train.action_semantic_residual import assert_confirmation_disjoint
from train.outcome_grounded import PANEL_SCHEMA, stable_u64
from train.shallow_search_residual import is_ambiguous


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_action(rows: list[dict[str, Any]], action: str, field: str) -> float:
    return math.fsum(float(row["action_statistics"].get(action, {}).get(field, 0.0)) for row in rows) / len(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    roots = read_rows(args.panel)
    search_rows = read_rows(args.shallow)
    freeze = json.loads(args.development_freeze.read_text())
    denied_roots = set(freeze["root_ids"])
    denied_battles = set(freeze["battle_ids"])
    search: dict[str, list[dict[str, Any]]] = {}
    for row in search_rows:
        search.setdefault(str(row["root_id"]), []).append(row)
    consumed = set()
    for panel_path in args.consumed_panel:
        consumed.update(str(row["root_id"]) for row in read_rows(panel_path))

    selected = []
    excluded = {"development_denylist": 0, "missing_search_schedule": 0, "not_ambiguous_both": 0}
    for root in roots:
        if root["root_id"] in denied_roots or root["battle_id"] in denied_battles:
            excluded["development_denylist"] += 1
            continue
        schedules = sorted(search.get(root["root_id"], []), key=lambda row: int(row["schedule_id"]))
        if len(schedules) != 2:
            excluded["missing_search_schedule"] += 1
            continue
        if not all(is_ambiguous(row["root_statistics"]) for row in schedules):
            excluded["not_ambiguous_both"] += 1
            continue
        support = set.intersection(*(set(row["action_statistics"]) for row in schedules))
        if len(support) < 3:
            raise ValueError(f"ambiguous root {root['root_id']} has fewer than three shared actions")
        ranked = sorted(
            support,
            key=lambda action: (
                aggregate_action(schedules, action, "visit_mass"),
                aggregate_action(schedules, action, "mean_value"),
                action,
            ),
            reverse=True,
        )
        candidates = ranked[:3]
        baseline = candidates[0]
        schedule_actions = [row["selected_action"] for row in schedules]
        selected.append({
            "schema": PANEL_SCHEMA,
            "battle_id": root["battle_id"],
            "root_id": root["root_id"],
            "baseline_action": baseline,
            "teacher_action": baseline,
            "candidate_actions": candidates,
            "selection": {
                "purpose": "development_only_staged_outcome_screen",
                "ambiguous_both_schedules": True,
                "baseline_rule": "mean_visit_mass_across_two_20k_schedules",
                "shallow_schedule_agreement": len(set(schedule_actions)) == 1,
                "shallow_schedule_actions": schedule_actions,
                "previously_outcome_opened": root["root_id"] in consumed,
                "tie_breaker": stable_u64(args.seed, root["battle_id"]),
            },
            "source_context": {
                "decision_idx": root.get("decision_idx"),
                "public_reveal_fractions": root.get("public_reveal_fractions"),
                "r1_selection": root.get("selection"),
                "causal_history": root.get("causal_history"),
                "source_path": root.get("source_path"),
                "source_line": root.get("source_line"),
            },
            "schedules": root["schedules"],
        })
    selected.sort(key=lambda row: (stable_u64(args.seed, row["battle_id"]), row["root_id"]))
    if not args.minimum_roots <= len(selected) <= args.maximum_roots:
        raise ValueError(f"staged panel has {len(selected)} roots outside frozen range {args.minimum_roots}-{args.maximum_roots}")
    assert_confirmation_disjoint(selected, freeze)
    write_private(args.output, selected)
    report = {
        "schema": "metagross-staged-ambiguous-outcome-panel-report/v1",
        "purpose": "development_only_never_confirmation",
        "source_panel_sha256": sha256(args.panel),
        "shallow_sha256": sha256(args.shallow),
        "development_freeze_sha256": sha256(args.development_freeze),
        "output_sha256": sha256(args.output),
        "source_roots": len(roots), "roots": len(selected),
        "battles": len({row["battle_id"] for row in selected}),
        "schedules": len(selected) * 2, "worlds": len(selected) * 16,
        "previously_outcome_opened": sum(row["selection"]["previously_outcome_opened"] for row in selected),
        "shallow_schedule_agreement": sum(row["selection"]["shallow_schedule_agreement"] for row in selected),
        "excluded": excluded, "seed": args.seed,
        "minimum_roots": args.minimum_roots, "maximum_roots": args.maximum_roots,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, required=True)
    parser.add_argument("--consumed-panel", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--minimum-roots", type=int, default=500)
    parser.add_argument("--maximum-roots", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
