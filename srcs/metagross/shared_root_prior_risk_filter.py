#!/usr/bin/env python3
"""Evaluate one-feature prior-selector fallbacks with held-out battles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SEVERE_DELTA = -0.20
FEATURES = (
    "raw_prior_action_count",
    "raw_prior_total_mass",
    "raw_prior_entropy",
    "raw_prior_top_probability",
    "mean_matched_raw_prior_mass",
    "mean_effective_prior_entropy",
    "mean_effective_prior_positive_actions",
    "mean_opponent_support_size",
    "unique_opponent_supports",
    "mean_canonical_particles",
    "particle_2_tv_mean",
    "particle_2_tv_p95",
    "particle_2_argmax_mismatch_fraction",
    "particle_4_tv_mean",
    "particle_4_tv_p95",
    "particle_4_argmax_mismatch_fraction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_protocol(protocol_path: Path, attribution_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_configuration = {
        "cross_validation": "leave one battle out",
        "fallback": "rm_policy_argmax",
        "rules": "one feature, one threshold, at_least or at_most",
        "severe_regression_delta": SEVERE_DELTA,
        "strategy": "opponent_prior_expected",
        "training_objective": "maximize poststratified teacher-mass delta subject to zero severe training regressions; fall back everywhere if gain is nonpositive",
    }
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("input", {}).get("sha256") != _sha256(attribution_path)
        or protocol.get("configuration") != expected_configuration
    ):
        raise ValueError("prior-risk filter differs from its frozen protocol")
    return protocol


def _weighted_mean(rows: list[dict], field: str) -> float:
    denominator = math.fsum(float(row["weight"]) for row in rows)
    return math.fsum(float(row[field]) * float(row["weight"]) for row in rows) / denominator


def _applies(row: dict, rule: dict | None) -> bool:
    if rule is None or row.get("delta") is None:
        return False
    value = float(row[rule["feature"]])
    return value >= rule["threshold"] if rule["direction"] == "at_least" else value <= rule["threshold"]


def _evaluate(rows: list[dict], rule: dict | None) -> dict:
    selected = [row for row in rows if _applies(row, rule)]
    outcomes = [
        {**row, "filtered_delta": float(row["delta"]) if _applies(row, rule) else 0.0}
        for row in rows
    ]
    return {
        "roots": len(rows),
        "prior_selected_roots": len(selected),
        "prior_coverage_fraction": len(selected) / len(rows),
        "poststratified_teacher_mass_delta": _weighted_mean(outcomes, "filtered_delta"),
        "severe_regressions": sum(row["filtered_delta"] <= SEVERE_DELTA for row in outcomes),
        "minimum_delta": min(row["filtered_delta"] for row in outcomes),
        "root_outcomes": [
            {
                "identity": row["identity"],
                "prior_selected": _applies(row, rule),
                "teacher_mass_delta": row["filtered_delta"],
            }
            for row in outcomes
        ],
    }


def _candidate_rules(rows: list[dict]) -> list[dict | None]:
    rules: list[dict | None] = [None]
    for feature in FEATURES:
        values = sorted({float(row[feature]) for row in rows if row.get(feature) is not None})
        thresholds = values + [
            (left + right) / 2.0 for left, right in zip(values, values[1:], strict=False)
        ]
        for threshold in sorted(set(thresholds)):
            for direction in ("at_least", "at_most"):
                rules.append({"feature": feature, "direction": direction, "threshold": threshold})
    return rules


def _select_rule(rows: list[dict]) -> tuple[dict | None, dict]:
    candidates = []
    for rule in _candidate_rules(rows):
        metrics = _evaluate(rows, rule)
        if metrics["severe_regressions"] == 0:
            candidates.append((rule, metrics))
    rule, metrics = sorted(
        candidates,
        key=lambda item: (
            -item[1]["poststratified_teacher_mass_delta"],
            -item[1]["prior_selected_roots"],
            json.dumps(item[0], sort_keys=True),
        ),
    )[0]
    if metrics["poststratified_teacher_mass_delta"] <= 0:
        return None, _evaluate(rows, None)
    return rule, metrics


def analyze(attribution_path: Path, protocol_path: Path) -> dict:
    _validate_protocol(protocol_path, attribution_path)
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    if attribution.get("mode") != "exploratory_shared_root_prior_failure_attribution":
        raise ValueError("prior-risk filter input has an invalid contract")
    rows = []
    for root in attribution["roots"]:
        rows.append(
            {
                "identity": root["identity"],
                "battle_tag": root["identity"]["battle_tag"],
                "weight": float(root["poststratification_weight"]),
                "delta": root["strategies"]["opponent_prior_expected"][
                    "teacher_mass_delta_from_rm_argmax"
                ],
                **{feature: root.get(feature) for feature in FEATURES},
            }
        )
    battles = sorted({row["battle_tag"] for row in rows})
    folds = []
    all_outcomes = []
    for battle in battles:
        training = [row for row in rows if row["battle_tag"] != battle]
        holdout = [row for row in rows if row["battle_tag"] == battle]
        rule, training_metrics = _select_rule(training)
        holdout_metrics = _evaluate(holdout, rule)
        all_outcomes.extend(holdout_metrics["root_outcomes"])
        folds.append(
            {
                "held_out_battle": battle,
                "selected_rule": rule,
                "training": {key: value for key, value in training_metrics.items() if key != "root_outcomes"},
                "holdout": holdout_metrics,
            }
        )
    outcome_by_identity = {
        json.dumps(row["identity"], separators=(",", ":"), sort_keys=True): row
        for row in all_outcomes
    }
    joined = [
        {
            **row,
            "prior_selected": outcome_by_identity[
                json.dumps(row["identity"], separators=(",", ":"), sort_keys=True)
            ]["prior_selected"],
            "filtered_delta": outcome_by_identity[
                json.dumps(row["identity"], separators=(",", ":"), sort_keys=True)
            ]["teacher_mass_delta"],
        }
        for row in rows
    ]
    cross_validated = {
        "roots": len(joined),
        "battles": len(battles),
        "prior_selected_roots": sum(row["prior_selected"] for row in joined),
        "poststratified_teacher_mass_delta": _weighted_mean(joined, "filtered_delta"),
        "severe_regressions": sum(row["filtered_delta"] <= SEVERE_DELTA for row in joined),
        "minimum_delta": min(row["filtered_delta"] for row in joined),
    }
    available = [row for row in rows if row["delta"] is not None]
    always_prior = _evaluate(rows, {"feature": FEATURES[0], "direction": "at_least", "threshold": -math.inf})
    return {
        "schema_version": 1,
        "mode": "exploratory_leave_one_battle_out_prior_risk_filter",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "input": {"path": str(attribution_path), "sha256": _sha256(attribution_path)},
        "configuration": {
            "strategy": "opponent_prior_expected",
            "fallback": "rm_policy_argmax",
            "features": list(FEATURES),
            "rules": "one feature, one threshold, at_least or at_most",
            "training_objective": "maximize poststratified teacher-mass delta subject to zero severe training regressions; fall back everywhere if gain is nonpositive",
            "severe_regression_delta": SEVERE_DELTA,
            "cross_validation": "leave one battle out",
        },
        "counts": {
            "roots": len(rows),
            "battles": len(battles),
            "roots_with_available_prior_selector": len(available),
        },
        "cross_validated": cross_validated,
        "comparators": {
            "always_rm_policy_argmax": {
                "poststratified_teacher_mass_delta": 0.0,
                "severe_regressions": 0,
                "prior_selected_roots": 0,
            },
            "always_opponent_prior_expected": {
                key: value
                for key, value in always_prior.items()
                if key != "root_outcomes"
            },
        },
        "folds": folds,
        "gate": {
            "passed": cross_validated["poststratified_teacher_mass_delta"] > 0
            and cross_validated["severe_regressions"] == 0,
            "conditions": {
                "positive_poststratified_teacher_mass_delta": cross_validated[
                    "poststratified_teacher_mass_delta"
                ]
                > 0,
                "zero_held_out_severe_regressions": cross_validated["severe_regressions"] == 0,
            },
            "new_candidate_supported": False,
            "new_games_authorized": False,
        },
        "limitations": [
            "Only five battles and 26 roots are available; folds are small and trajectory-correlated.",
            "Feature and threshold selection is nested within each training fold but the entire analysis family is post-hoc.",
            "The S-4B teacher is a development proxy, not game-outcome ground truth.",
            "A passing exploratory gate would still require a separately collected held-out root panel and preregistration.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.attribution.expanduser().resolve(), args.protocol.expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cross_validated": report["cross_validated"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
