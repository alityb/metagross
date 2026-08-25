#!/usr/bin/env python3
"""Audit and replay the frozen terminal-MCTS controller on opened corpora."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from train.direct_long_horizon_controller import decide
from train.outcome_grounded import RESULT_SCHEMA


PINNED_ENGINE_SHA256 = "cf71fbba541c9e7b4f3c891bf9b25dca863196708b7131f77d1e0016c1073f69"
EXPECTED_POLICY = "seeded_exact_mcts_argmax_both_sides"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def project_top_two(row: dict[str, Any], rollouts: int) -> dict[str, Any]:
    projected = copy.deepcopy(row)
    actions = list(row["candidate_actions"][:2])
    if len(actions) != 2 or actions[0] != row["baseline_action"]:
        raise ValueError("source result does not preserve frozen top-two ordering")
    projected["candidate_actions"] = actions
    projected["action_outcomes"] = {
        action: [
            sample
            for sample in row["action_outcomes"][action]
            if int(sample["rollout"]) < rollouts
        ]
        for action in actions
    }
    projected["configuration"] = {**row["configuration"], "rollouts": rollouts}
    return projected


def paired_mean(rows: list[dict[str, Any]], baseline: str, alternative: str) -> float:
    values = []
    for row in rows:
        maps = {}
        for action in (baseline, alternative):
            maps[action] = {
                (int(sample["world_index"]), int(sample["rollout"])): float(sample["outcome"])
                for sample in row["action_outcomes"][action]
                if sample["outcome"] is not None
            }
        keys = set(maps[baseline]).intersection(maps[alternative])
        values.extend(maps[alternative][key] - maps[baseline][key] for key in keys)
    return math.fsum(values) / len(values) if values else float("nan")


def audit_corpus(
    name: str, results_path: Path, merge_report_path: Path, analysis_path: Path
) -> tuple[dict[str, Any], set[str]]:
    merge = json.loads(merge_report_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    result_hash = sha256(results_path)
    if (
        merge.get("output_sha256") != result_hash
        or analysis.get("results_sha256") != result_hash
        or int(merge.get("rollouts", 0)) != 16
    ):
        raise ValueError(f"{name}: merged-result hash contract failed")
    root_analysis = {row["root_id"]: row for row in analysis["root_results"]}
    rows = read_jsonl(results_path)
    if any(row.get("schema") != RESULT_SCHEMA for row in rows):
        raise ValueError(f"{name}: result schema mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    battle_ids = set()
    terminal = total = 0
    for row in rows:
        config = row.get("configuration", {})
        if (
            config.get("continuation_policy") != EXPECTED_POLICY
            or int(config.get("continuation_iterations", 0)) != 2048
            or int(config.get("root_iterations", 0)) != 20_000
            or config.get("root_opponent_policy") != "20k_mcts_visit_distribution"
            or int(config.get("rollouts", 0)) != 16
            or int(config.get("rollout_start", 0)) != 0
            or int(config.get("max_decisions", 0)) not in {128, 192}
        ):
            raise ValueError(f"{name}: continuation configuration mismatch")
        grouped.setdefault(str(row["root_id"]), []).append(row)
        battle_ids.add(str(row["battle_id"]))
        for samples in row["action_outcomes"].values():
            total += len(samples)
            terminal += sum(sample["outcome"] is not None for sample in samples)
    if len(rows) != 2 * len(grouped) or len(battle_ids) != len(grouped):
        raise ValueError(f"{name}: physical-root grouping mismatch")

    decisions = []
    early_overrides = retained_positive = harmful = 0
    advantages = []
    for root_id, source_rows in sorted(grouped.items()):
        stage4_rows = [project_top_two(row, 4) for row in source_rows]
        stage16_rows = [project_top_two(row, 16) for row in source_rows]
        stage4 = decide(stage4_rows)
        stage16 = decide(stage16_rows)
        chosen = stage4 if stage4["decision"] == "override" else stage16
        override = chosen["decision"] == "override"
        full_mean = paired_mean(
            stage16_rows, stage16["baseline_action"], stage16["alternative_action"]
        )
        if stage4["decision"] == "override":
            early_overrides += 1
            retained_positive += full_mean > 0.0
        durable_harm = (
            override
            and all(value is not None and value < -0.02 for value in stage16["schedule_advantages"])
            and stage16["cluster_bootstrap_ci95"][1] is not None
            and stage16["cluster_bootstrap_ci95"][1] < -0.01
        )
        harmful += durable_harm
        if override:
            advantages.append(full_mean)
        expected = root_analysis.get(root_id)
        if expected is None:
            raise ValueError(f"{name}: analysis join incomplete")
        decisions.append({
            "root_id": root_id,
            "battle_id": source_rows[0]["battle_id"],
            "stage4_decision": stage4["decision"],
            "stage16_decision": stage16["decision"],
            "final_decision": chosen["decision"],
            "selected_action": chosen["selected_action"],
            "baseline_action": chosen["baseline_action"],
            "full16_paired_advantage": full_mean,
            "durably_harmful": durable_harm,
            "source_stable_action": expected.get("stable_action"),
        })
    overrides = len(advantages)
    report = {
        "name": name,
        "inputs": {
            "results_sha256": result_hash,
            "merge_report_sha256": sha256(merge_report_path),
            "analysis_sha256": sha256(analysis_path),
            "engine_binary_sha256": PINNED_ENGINE_SHA256,
        },
        "roots": len(grouped),
        "battles": len(battle_ids),
        "terminal_coverage": terminal / total,
        "overrides": overrides,
        "early_overrides": early_overrides,
        "early_retained_positive": retained_positive,
        "early_retained_positive_rate": retained_positive / early_overrides if early_overrides else None,
        "durably_harmful_overrides": harmful,
        "mean_full16_advantage": math.fsum(advantages) / overrides if overrides else None,
        "decisions": decisions,
    }
    return report, battle_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-results", type=Path, required=True)
    parser.add_argument("--generic-merge-report", type=Path, required=True)
    parser.add_argument("--generic-analysis", type=Path, required=True)
    parser.add_argument("--switch-results", type=Path, required=True)
    parser.add_argument("--switch-merge-report", type=Path, required=True)
    parser.add_argument("--switch-analysis", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()

    generic, generic_ids = audit_corpus(
        "opened_generic_300", args.generic_results, args.generic_merge_report, args.generic_analysis
    )
    switch, switch_ids = audit_corpus(
        "opened_switch_200", args.switch_results, args.switch_merge_report, args.switch_analysis
    )
    if generic_ids.intersection(switch_ids):
        raise ValueError("opened corpora overlap by physical battle")
    exclusions = {
        "schema": "metagross-terminal-mcts-controller-exclusions/v1",
        "claim": "development_only_no_confirmation_identity_read",
        "battle_ids": sorted(generic_ids | switch_ids),
        "sources": ["opened_generic_300", "opened_switch_200"],
        "sealed_test_split": {
            "materialized": False,
            "identities_read": 0,
            "disjointness_authority": "distinct_collection_scope_and_unopened_identity_split",
        },
    }
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.write_text(json.dumps(exclusions, indent=2, sort_keys=True) + "\n")

    corpora = [generic, switch]
    roots = sum(row["roots"] for row in corpora)
    terminal_coverage = sum(row["roots"] * row["terminal_coverage"] for row in corpora) / roots
    overrides = sum(row["overrides"] for row in corpora)
    harmful = sum(row["durably_harmful_overrides"] for row in corpora)
    early = sum(row["early_overrides"] for row in corpora)
    retained = sum(row["early_retained_positive"] for row in corpora)
    advantages = [
        decision["full16_paired_advantage"]
        for corpus in corpora
        for decision in corpus["decisions"]
        if decision["final_decision"] == "override"
    ]
    checks = {
        "root_count_500": roots == 500,
        "corpora_disjoint": not generic_ids.intersection(switch_ids),
        "terminal_coverage": terminal_coverage >= 0.95,
        "override_coverage": overrides >= 5,
        "zero_durable_harm": harmful == 0,
        "positive_mean_advantage": bool(advantages) and math.fsum(advantages) / len(advantages) > 0,
        "early_retention": early > 0 and retained / early >= 0.80,
        "sealed_confirmation_unread": True,
        "latency_canary": None,
    }
    report = {
        "schema": "metagross-terminal-mcts-controller-opened-integration/v1",
        "protocol_sha256": sha256(args.protocol),
        "engine_binary_sha256": PINNED_ENGINE_SHA256,
        "corpora": corpora,
        "aggregate": {
            "roots": roots,
            "terminal_coverage": terminal_coverage,
            "overrides": overrides,
            "durably_harmful_overrides": harmful,
            "mean_full16_advantage": math.fsum(advantages) / len(advantages) if advantages else None,
            "early_overrides": early,
            "early_retained_positive": retained,
            "early_retained_positive_rate": retained / early if early else None,
        },
        "checks": checks,
        "admitted_except_latency": all(value for value in checks.values() if value is not None),
        "exclusions_sha256": sha256(args.exclusions),
        "confirmation": {"rows_read": 0, "identities_read": 0, "preserved": True},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "checks": checks, "admitted_except_latency": report["admitted_except_latency"]}, sort_keys=True))
    raise SystemExit(0 if report["admitted_except_latency"] else 2)


if __name__ == "__main__":
    main()
