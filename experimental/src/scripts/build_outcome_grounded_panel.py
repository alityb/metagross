#!/usr/bin/env python3
"""Freeze shared-error ambiguous roots for outcome-grounded continuation study."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from scripts.run_public_mcts_leaf_gate import _load_oracle, _load_panel
from train.action_semantic_residual import assert_confirmation_disjoint
from train.outcome_grounded import PANEL_SCHEMA, stable_u64
from train.shallow_search_residual import battle_split, is_ambiguous


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_private(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict:
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    shallow_rows = read_rows(args.shallow)
    shallow = {row["pair_id"]: row for row in shallow_rows}
    excluded_root_ids: set[str] = set()
    excluded_battle_ids: set[str] = set()
    for excluded_panel in args.exclude_panel:
        excluded_rows = read_rows(excluded_panel)
        excluded_root_ids.update(row["root_id"] for row in excluded_rows)
        excluded_battle_ids.update(row["battle_id"] for row in excluded_rows)
    freezes = []
    for freeze_path in getattr(args, "development_freeze", []):
        freeze = json.loads(freeze_path.read_text())
        freezes.append(freeze)
        excluded_root_ids.update(freeze["root_ids"])
        excluded_battle_ids.update(freeze["battle_ids"])
    if getattr(args, "final_confirmation", False) and not freezes:
        raise ValueError("final confirmation requires at least one --development-freeze denylist")
    candidates = []
    attrition: collections.Counter[str] = collections.Counter()
    for root in panel:
        if root["root_id"] in excluded_root_ids or root["battle_id"] in excluded_battle_ids:
            attrition["excluded_root"] += 1
            continue
        if battle_split(root["battle_id"]) != "train":
            attrition["non_training_split"] += 1
            continue
        attrition["training_roots"] += 1
        schedules = []
        for schedule in root["schedules"]:
            pair_id = f"{root['root_id']}:{schedule['schedule_id']}"
            if pair_id not in shallow or pair_id not in oracle:
                raise ValueError("panel is missing shallow/oracle schedule")
            schedules.append((schedule, shallow[pair_id], oracle[pair_id]))
        if len(schedules) != 2:
            raise ValueError("panel root does not contain exactly two schedules")
        if not all(is_ambiguous(row[1]["root_statistics"]) for row in schedules):
            attrition["not_ambiguous_both_schedules"] += 1
            continue
        attrition["ambiguous_both_schedules"] += 1
        shallow_actions = {row[1]["selected_action"] for row in schedules}
        teacher_actions = {row[2]["oracle_action"] for row in schedules}
        if len(shallow_actions) != 1:
            attrition["shallow_schedule_disagreement"] += 1
            continue
        attrition["shallow_schedules_agree"] += 1
        if len(teacher_actions) != 1:
            attrition["oracle_schedule_disagreement"] += 1
            continue
        attrition["oracle_schedules_agree"] += 1
        if shallow_actions != teacher_actions:
            attrition["20k_50k_action_disagreement"] += 1
            continue
        attrition["four_way_agreement"] += 1
        baseline = next(iter(shallow_actions))
        support = set.intersection(*(set(row[1]["action_statistics"]) for row in schedules))
        teacher_values = {
            action: math.fsum(float(row[2]["action_values"][action]) for row in schedules) / 2
            for action in support
        }
        visit_values = {
            action: math.fsum(float(row[1]["action_statistics"][action]["visit_mass"]) for row in schedules) / 2
            for action in support
        }
        teacher_alternatives = sorted(
            (action for action in support if action != baseline),
            key=lambda action: (teacher_values[action], action), reverse=True,
        )[:3]
        visit_alternatives = sorted(
            (action for action in support if action != baseline),
            key=lambda action: (visit_values[action], action), reverse=True,
        )[:2]
        action_set = [baseline] + [
            action for action in teacher_alternatives + visit_alternatives if action != baseline
        ]
        action_set = list(dict.fromkeys(action_set))[: args.max_candidate_actions]
        ambiguity_score = math.fsum(
            row[1]["root_statistics"]["weighted_js_divergence"]
            + row[1]["root_statistics"]["weighted_top_action_disagreement"]
            + (1.0 - row[1]["root_statistics"]["aggregate_top_visit_mass"])
            for row in schedules
        ) / 2
        candidates.append((ambiguity_score, stable_u64(args.seed, root["battle_id"]), {
            "schema": PANEL_SCHEMA,
            "battle_id": root["battle_id"],
            "root_id": root["root_id"],
            "baseline_action": baseline,
            "teacher_action": baseline,
            "candidate_actions": action_set,
            "selection": {
                "source_split": "train",
                "both_schedules_ambiguous": True,
                "20k_50k_all_agree": True,
                "ambiguity_score": ambiguity_score,
            },
            "source_context": {
                "decision_idx": root.get("decision_idx"),
                "public_reveal_fractions": root.get("public_reveal_fractions"),
                "r1_selection": root.get("selection"),
                "causal_history": root.get("causal_history"),
            },
            "schedules": [row[0] for row in schedules],
        }))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    if args.roots is not None and len(candidates) < args.roots:
        raise ValueError(f"only {len(candidates)} eligible roots for {args.roots}")
    rows = [row[2] for row in (candidates if args.roots is None else candidates[:args.roots])]
    if getattr(args, "final_confirmation", False):
        for freeze in freezes:
            assert_confirmation_disjoint(rows, freeze)
    write_private(args.output, rows)
    report = {
        "schema": "metagross-outcome-grounded-panel-report/v1",
        "roots": len(rows),
        "schedules": len(rows) * 2,
        "worlds": len(rows) * 16,
        "candidate_actions": sum(len(row["candidate_actions"]) for row in rows),
        "eligible_roots": len(candidates),
        "requested_roots": "all_eligible" if args.roots is None else args.roots,
        "attrition": dict(attrition),
        "excluded_roots": len(excluded_root_ids),
        "excluded_battles": len(excluded_battle_ids),
        "development_freezes": len(freezes),
        "purpose": "final_confirmation" if getattr(args, "final_confirmation", False) else "development",
        "max_candidate_actions": args.max_candidate_actions,
        "seed": args.seed,
        "panel_sha256": sha256(args.output),
        "source_panel_sha256": panel_hash,
        "shallow_sha256": sha256(args.shallow),
        "oracle_sha256": oracle_hash,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="freeze every root passing the four-way agreement screen",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--development-freeze", type=Path, action="append", default=[])
    parser.add_argument(
        "--final-confirmation", action="store_true",
        help="require development denylist(s) and reject root or battle overlap",
    )
    parser.add_argument("--max-candidate-actions", type=int, default=6)
    args = parser.parse_args()
    if args.max_candidate_actions < 2:
        parser.error("--max-candidate-actions must be at least 2")
    if args.all_eligible:
        args.roots = None
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
