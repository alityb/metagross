#!/usr/bin/env python3
"""Evaluate direct-authoritative shared-root consensus overrides offline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SEVERE_DELTA = -0.20
SELECTORS = (
    "rm_policy_argmax",
    "opponent_prior_expected",
    "worst_case_endpoint",
    "bounded_robust_0.10",
    "bounded_robust_0.25",
    "bounded_robust_0.50",
)
RULE_GROUPS = {
    "rm_prior_consensus": ("rm_policy_argmax", "opponent_prior_expected"),
    "rm_worst_consensus": ("rm_policy_argmax", "worst_case_endpoint"),
    "prior_worst_consensus": ("opponent_prior_expected", "worst_case_endpoint"),
    "rm_prior_worst_consensus": (
        "rm_policy_argmax",
        "opponent_prior_expected",
        "worst_case_endpoint",
    ),
    "prior_robust_consensus": (
        "opponent_prior_expected",
        "bounded_robust_0.10",
        "bounded_robust_0.25",
        "bounded_robust_0.50",
    ),
    "non_rm_consensus": (
        "opponent_prior_expected",
        "worst_case_endpoint",
        "bounded_robust_0.10",
        "bounded_robust_0.25",
        "bounded_robust_0.50",
    ),
    "all_search_consensus": SELECTORS,
}
RULES = ("direct_only", *RULE_GROUPS, "strict_search_majority")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_key(identity: dict) -> str:
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)


def _validate_protocol(protocol_path: Path, matrix_path: Path, evaluation_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "evaluation_sha256": _sha256(evaluation_path),
            "matrix_diagnostics_sha256": _sha256(matrix_path),
        }
        or protocol.get("configuration")
        != {
            "fallback": "direct_r1_argmax",
            "rules": list(RULES),
            "selectors": list(SELECTORS),
            "severe_regression_delta": SEVERE_DELTA,
            "cross_validation": "leave_one_battle_out_rule_selection",
        }
    ):
        raise ValueError("consensus override analysis differs from its frozen protocol")


def _override(rule: str, actions: dict[str, str | None], direct: str) -> str:
    if rule == "direct_only":
        return direct
    if rule == "strict_search_majority":
        available = [actions[name] for name in SELECTORS if actions[name] is not None]
        if not available:
            return direct
        action, count = sorted(Counter(available).items(), key=lambda row: (-row[1], row[0]))[0]
        return action if count > len(available) / 2 else direct
    selected = [actions[name] for name in RULE_GROUPS[rule]]
    return selected[0] if selected[0] is not None and len(set(selected)) == 1 else direct


def _weighted_mean(rows: list[dict], field: str) -> float:
    total = math.fsum(row["weight"] for row in rows)
    return math.fsum(row[field] * row["weight"] for row in rows) / total


def _metrics(rows: list[dict], rule: str) -> dict:
    values = [
        {
            **row,
            "delta": row["rules"][rule]["teacher_mass_delta"],
            "overrides": row["rules"][rule]["override_schedules"],
        }
        for row in rows
    ]
    return {
        "roots": len(values),
        "override_schedules": sum(row["overrides"] for row in values),
        "poststratified_teacher_mass_delta": _weighted_mean(values, "delta"),
        "mean_teacher_mass_delta": math.fsum(row["delta"] for row in values) / len(values),
        "severe_regressions": sum(row["delta"] <= SEVERE_DELTA for row in values),
        "minimum_delta": min(row["delta"] for row in values),
        "maximum_delta": max(row["delta"] for row in values),
    }


def analyze(
    matrix_path: Path, evaluation_path: Path, protocol_path: Path
) -> dict[str, object]:
    _validate_protocol(protocol_path, matrix_path, evaluation_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    evaluation_roots = {_identity_key(root["identity"]): root for root in evaluations}
    matrix_roots = {_identity_key(root["identity"]): root for root in matrix["roots"]}
    if (
        len(evaluation_roots) != len(evaluations)
        or len(matrix_roots) != len(matrix["roots"])
        or evaluation_roots.keys() != matrix_roots.keys()
    ):
        raise ValueError("consensus override root join is incomplete")

    roots = []
    for key, matrix_root in matrix_roots.items():
        evaluation = evaluation_roots[key]
        evaluation_schedules = {
            schedule["schedule_id"]: schedule for schedule in evaluation["schedules"]
        }
        rule_masses = {rule: [] for rule in RULES}
        direct_masses = []
        override_counts = {rule: 0 for rule in RULES}
        schedules = []
        for schedule in matrix_root["schedules"]:
            source = evaluation_schedules[schedule["schedule_id"]]
            direct = max(
                source["worlds"][0]["effective_player_priors"],
                key=lambda row: (float(row[1]), str(row[0]).lower()),
            )[0].lower()
            actions = {
                name: schedule["strategy_aggregates"]["strategies"][name]["selected_action"]
                for name in SELECTORS
            }
            teacher_policies = [
                {row["action"]: float(row["probability"]) for row in repeat["side_one_policy"]}
                for repeat in source["aggregate_treatments"]["S-4B"]
            ]

            def teacher_mass(action: str) -> float:
                return math.fsum(policy.get(action, 0.0) for policy in teacher_policies) / len(
                    teacher_policies
                )

            direct_mass = teacher_mass(direct)
            direct_masses.append(direct_mass)
            decisions = {}
            for rule in RULES:
                action = _override(rule, actions, direct)
                mass = teacher_mass(action)
                rule_masses[rule].append(mass)
                override_counts[rule] += action != direct
                decisions[rule] = {
                    "action": action,
                    "overrode_direct": action != direct,
                    "teacher_mass": mass,
                    "teacher_mass_delta": mass - direct_mass,
                }
            schedules.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "direct_action": direct,
                    "selector_actions": actions,
                    "rules": decisions,
                }
            )
        direct_root_mass = math.fsum(direct_masses) / len(direct_masses)
        roots.append(
            {
                "identity": matrix_root["identity"],
                "battle_tag": matrix_root["identity"]["battle_tag"],
                "weight": float(matrix_root["sampling"]["poststratification_weight"]),
                "direct_teacher_mass": direct_root_mass,
                "rules": {
                    rule: {
                        "teacher_mass": math.fsum(rule_masses[rule]) / len(rule_masses[rule]),
                        "teacher_mass_delta": math.fsum(rule_masses[rule]) / len(rule_masses[rule])
                        - direct_root_mass,
                        "override_schedules": override_counts[rule],
                    }
                    for rule in RULES
                },
                "schedules": schedules,
            }
        )

    rule_metrics = {rule: _metrics(roots, rule) for rule in RULES}
    folds = []
    held_out_rows = []
    for battle in sorted({root["battle_tag"] for root in roots}):
        training = [root for root in roots if root["battle_tag"] != battle]
        holdout = [root for root in roots if root["battle_tag"] == battle]
        eligible = [
            (rule, _metrics(training, rule))
            for rule in RULES
            if _metrics(training, rule)["severe_regressions"] == 0
        ]
        selected_rule, training_metrics = sorted(
            eligible,
            key=lambda row: (
                -row[1]["poststratified_teacher_mass_delta"],
                row[1]["override_schedules"],
                row[0],
            ),
        )[0]
        if training_metrics["poststratified_teacher_mass_delta"] <= 0:
            selected_rule = "direct_only"
            training_metrics = _metrics(training, selected_rule)
        holdout_metrics = _metrics(holdout, selected_rule)
        for root in holdout:
            held_out_rows.append(
                {
                    "weight": root["weight"],
                    "delta": root["rules"][selected_rule]["teacher_mass_delta"],
                }
            )
        folds.append(
            {
                "held_out_battle": battle,
                "selected_rule": selected_rule,
                "training": training_metrics,
                "holdout": holdout_metrics,
            }
        )
    cross_validated = {
        "roots": len(held_out_rows),
        "poststratified_teacher_mass_delta": _weighted_mean(held_out_rows, "delta"),
        "mean_teacher_mass_delta": math.fsum(row["delta"] for row in held_out_rows)
        / len(held_out_rows),
        "severe_regressions": sum(row["delta"] <= SEVERE_DELTA for row in held_out_rows),
        "minimum_delta": min(row["delta"] for row in held_out_rows),
    }
    return {
        "schema_version": 1,
        "mode": "exploratory_direct_authoritative_consensus_overrides",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "inputs": {
            "matrix_diagnostics": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        },
        "configuration": {
            "fallback": "direct_r1_argmax",
            "rules": list(RULES),
            "selectors": list(SELECTORS),
            "severe_regression_delta": SEVERE_DELTA,
            "cross_validation": "leave_one_battle_out_rule_selection",
        },
        "counts": {"roots": len(roots), "battles": len(folds), "schedules": len(roots) * 4},
        "rule_metrics": rule_metrics,
        "cross_validated": cross_validated,
        "folds": folds,
        "roots": roots,
        "gate": {
            "passed": cross_validated["poststratified_teacher_mass_delta"] > 0
            and cross_validated["severe_regressions"] == 0,
            "conditions": {
                "positive_cross_validated_delta": cross_validated[
                    "poststratified_teacher_mass_delta"
                ]
                > 0,
                "zero_cross_validated_severe_regressions": cross_validated[
                    "severe_regressions"
                ]
                == 0,
            },
            "new_candidate_supported": False,
            "new_games_authorized": False,
        },
        "limitations": [
            "This rule family and cross-validation are exploratory on 26 roots from five battles.",
            "The S-4B teacher is a development proxy, not game-outcome ground truth.",
            "A passing result requires confirmation on a separately collected source-bound root panel.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.matrix.expanduser().resolve(),
        args.evaluation.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cross_validated": report["cross_validated"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
