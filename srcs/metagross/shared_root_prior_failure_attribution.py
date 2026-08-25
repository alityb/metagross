#!/usr/bin/env python3
"""Attribute enriched opponent-prior selector regressions to frozen root features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PRIOR_STRATEGIES = (
    "opponent_prior_expected",
    "bounded_robust_0.10",
    "bounded_robust_0.25",
    "bounded_robust_0.50",
)
SEVERE_DELTA = -0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_protocol(
    protocol_path: Path, matrix_path: Path, enrichment_path: Path, particle_path: Path
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "matrix_diagnostics_sha256": _sha256(matrix_path),
            "particle_ablation_sha256": _sha256(particle_path),
            "prior_enrichment_sha256": _sha256(enrichment_path),
        }
        or protocol.get("configuration")
        != {
            "correlations": ["pearson", "spearman_with_average_tie_ranks"],
            "prior_strategies": list(PRIOR_STRATEGIES),
            "severe_regression_delta": SEVERE_DELTA,
        }
    ):
        raise ValueError("prior-failure attribution differs from its frozen protocol")
    return protocol


def _identity_key(identity: object) -> str:
    if not isinstance(identity, dict):
        raise ValueError("root identity is invalid")
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _entropy(probabilities: list[float]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _correlations(rows: list[dict], outcome: str, features: list[str]) -> dict:
    result = {}
    for feature in features:
        pairs = [
            (float(row[feature]), float(row[outcome]))
            for row in rows
            if row.get(feature) is not None and row.get(outcome) is not None
        ]
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        result[feature] = {
            "count": len(pairs),
            "pearson": _pearson(left, right),
            "spearman": _pearson(_ranks(left), _ranks(right)),
        }
    return result


def _particle_features(root: dict, size: str) -> dict[str, float | int | None]:
    cohort = root["cohorts"][size]
    rows = cohort["rows"]
    return {
        f"particle_{size}_cohort_count": len(rows),
        f"particle_{size}_tv_mean": _mean([float(row["tv_from_full"]) for row in rows]),
        f"particle_{size}_tv_p95": _percentile(
            [float(row["tv_from_full"]) for row in rows], 0.95
        ),
        f"particle_{size}_argmax_mismatch_fraction": _mean(
            [float(row["argmax_changed_from_full"]) for row in rows]
        ),
        f"particle_{size}_canonical_collapse_fraction": _mean(
            [
                float(row["diagnostics"]["canonical_particle_count"] < int(size))
                for row in rows
            ]
        ),
    }


def _prior_features(root: dict) -> dict[str, float | int | None]:
    raw = root["source_captured_raw_opponent_priors"]
    total = math.fsum(float(row[1]) for row in raw)
    normalized = [float(row[1]) / total for row in raw]
    worlds = [world for schedule in root["schedules"] for world in schedule["worlds"]]
    available = [world for world in worlds if world["effective_opponent_priors"] is not None]
    effective_entropies = []
    effective_positive_actions = []
    for world in available:
        probabilities = [float(row[1]) for row in world["effective_opponent_priors"]]
        effective_entropies.append(_entropy(probabilities))
        effective_positive_actions.append(sum(value > 0 for value in probabilities))
    return {
        "raw_prior_action_count": len(raw),
        "raw_prior_total_mass": total,
        "raw_prior_entropy": _entropy(normalized),
        "raw_prior_top_probability": max(normalized),
        "prior_available_world_fraction": len(available) / len(worlds),
        "mean_matched_raw_prior_mass": _mean(
            [float(world["matched_raw_prior_mass"]) for world in worlds]
        ),
        "minimum_matched_raw_prior_mass": min(
            float(world["matched_raw_prior_mass"]) for world in worlds
        ),
        "mean_effective_prior_entropy": _mean(effective_entropies),
        "mean_effective_prior_positive_actions": _mean(effective_positive_actions),
        "mean_opponent_support_size": _mean(
            [float(len(world["opponent_action_support"])) for world in worlds]
        ),
        "unique_opponent_supports": len(
            {tuple(world["opponent_action_support"]) for world in worlds}
        ),
    }


def analyze(
    matrix_path: Path,
    enrichment_path: Path,
    particle_path: Path,
    protocol_path: Path,
) -> dict:
    _validate_protocol(protocol_path, matrix_path, enrichment_path, particle_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))
    particle = json.loads(particle_path.read_text(encoding="utf-8"))
    if (
        matrix.get("mode") != "shared_root_matrix_diagnostics"
        or enrichment.get("mode") != "source_captured_opponent_prior_enrichment"
        or particle.get("mode") != "stage2_shared_root_particle_cohort_ablation"
        or matrix.get("prior_enrichment", {}).get("sha256") != _sha256(enrichment_path)
        or matrix.get("input", {}).get("sha256") != particle.get("input", {}).get("sha256")
    ):
        raise ValueError("prior-failure attribution inputs do not share a frozen panel")

    matrix_roots = {_identity_key(root["identity"]): root for root in matrix["roots"]}
    enriched_roots = {_identity_key(root["identity"]): root for root in enrichment["roots"]}
    particle_roots = {_identity_key(root["identity"]): root for root in particle["roots"]}
    if (
        len(matrix_roots) != len(matrix["roots"])
        or len(enriched_roots) != len(enrichment["roots"])
        or len(particle_roots) != len(particle["roots"])
        or matrix_roots.keys() != enriched_roots.keys()
        or matrix_roots.keys() != particle_roots.keys()
    ):
        raise ValueError("prior-failure attribution root join is incomplete")

    metrics = {
        (_identity_key(row["identity"]), row["strategy"]): row
        for row in matrix["root_metrics"]
    }
    if len(metrics) != len(matrix["root_metrics"]):
        raise ValueError("matrix root metrics contain duplicate identities or strategies")
    root_rows = []
    for key, matrix_root in matrix_roots.items():
        enriched_root = enriched_roots[key]
        particle_root = particle_roots[key]
        baseline = metrics[(key, "rm_policy_argmax")]["teacher_mass"]
        prior_features = _prior_features(enriched_root)
        complete_prior = prior_features["prior_available_world_fraction"] == 1.0
        strategies = {}
        for strategy in PRIOR_STRATEGIES:
            metric = metrics[(key, strategy)]
            teacher_mass = metric["teacher_mass"] if complete_prior else None
            strategies[strategy] = {
                "teacher_mass": teacher_mass,
                "teacher_mass_delta_from_rm_argmax": (
                    teacher_mass - baseline if teacher_mass is not None else None
                ),
                "severe_regression": (
                    teacher_mass - baseline <= SEVERE_DELTA
                    if teacher_mass is not None
                    else None
                ),
                "selected_actions": sorted(
                    {
                        schedule["strategy_aggregates"]["strategies"][strategy][
                            "selected_action"
                        ]
                        for schedule in matrix_root["schedules"]
                        if schedule["strategy_aggregates"]["strategies"][strategy][
                            "selected_action"
                        ]
                        is not None
                    }
                ),
            }
        root_rows.append(
            {
                "identity": matrix_root["identity"],
                "poststratification_weight": float(
                    matrix_root["sampling"]["poststratification_weight"]
                ),
                "rm_policy_argmax_teacher_mass": baseline,
                "mean_canonical_particles": _mean(
                    [float(schedule["canonical_particles"]) for schedule in matrix_root["schedules"]]
                ),
                **prior_features,
                **_particle_features(particle_root, "2"),
                **_particle_features(particle_root, "4"),
                "strategies": strategies,
            }
        )

    feature_names = [
        "raw_prior_action_count",
        "raw_prior_total_mass",
        "raw_prior_entropy",
        "raw_prior_top_probability",
        "mean_matched_raw_prior_mass",
        "minimum_matched_raw_prior_mass",
        "mean_effective_prior_entropy",
        "mean_effective_prior_positive_actions",
        "mean_opponent_support_size",
        "unique_opponent_supports",
        "mean_canonical_particles",
        "particle_2_tv_mean",
        "particle_2_tv_p95",
        "particle_2_argmax_mismatch_fraction",
        "particle_2_canonical_collapse_fraction",
        "particle_4_tv_mean",
        "particle_4_tv_p95",
        "particle_4_argmax_mismatch_fraction",
        "particle_4_canonical_collapse_fraction",
    ]
    strategy_reports = {}
    for strategy in PRIOR_STRATEGIES:
        available_rows = []
        for root in root_rows:
            delta = root["strategies"][strategy]["teacher_mass_delta_from_rm_argmax"]
            if delta is not None:
                available_rows.append({**root, "delta": delta})
        severe = [row for row in available_rows if row["delta"] <= SEVERE_DELTA]
        nonsevere = [row for row in available_rows if row["delta"] > SEVERE_DELTA]
        strategy_reports[strategy] = {
            "available_roots": len(available_rows),
            "severe_regressions": len(severe),
            "delta": {
                "mean": _mean([row["delta"] for row in available_rows]),
                "minimum": min((row["delta"] for row in available_rows), default=None),
                "maximum": max((row["delta"] for row in available_rows), default=None),
            },
            "feature_correlations_with_delta": _correlations(
                available_rows, "delta", feature_names
            ),
            "severe_vs_other_feature_means": {
                feature: {
                    "severe": _mean([float(row[feature]) for row in severe if row[feature] is not None]),
                    "other": _mean(
                        [float(row[feature]) for row in nonsevere if row[feature] is not None]
                    ),
                }
                for feature in feature_names
            },
            "severe_root_identities": [row["identity"] for row in severe],
        }

    return {
        "schema_version": 1,
        "mode": "exploratory_shared_root_prior_failure_attribution",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "inputs": {
            "matrix_diagnostics": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "prior_enrichment": {
                "path": str(enrichment_path),
                "sha256": _sha256(enrichment_path),
            },
            "particle_ablation": {"path": str(particle_path), "sha256": _sha256(particle_path)},
        },
        "configuration": {
            "prior_strategies": list(PRIOR_STRATEGIES),
            "severe_regression_delta": SEVERE_DELTA,
            "correlations": ["pearson", "spearman_with_average_tie_ranks"],
            "correlation_weighting": "unweighted_root_level_exploratory",
        },
        "counts": {
            "roots": len(root_rows),
            "roots_with_complete_opponent_priors": sum(
                root["prior_available_world_fraction"] == 1.0 for root in root_rows
            ),
        },
        "strategy_reports": strategy_reports,
        "roots": root_rows,
        "conclusion": {
            "architecture_gate_passed": False,
            "new_candidate_supported": False,
            "new_games_authorized": False,
            "interpretation": "Post-hoc associations may prioritize a mechanism test but cannot rescue any rejected selector or establish causality.",
        },
        "limitations": [
            "This is exploratory post-hoc attribution on 26 roots from five battles.",
            "The S-4B teacher is a stronger-search proxy, not game-outcome ground truth.",
            "Features are correlated and no p-values or multiplicity-adjusted claims are made.",
            "Correlations and descriptive feature means are unweighted by poststratification weight.",
            "The particle ablation used an older frozen no-effective-opponent-prior treatment, so its stability features are context rather than matched causal interventions.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prior-enrichment", type=Path, required=True)
    parser.add_argument("--particle-ablation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.matrix.expanduser().resolve(),
        args.prior_enrichment.expanduser().resolve(),
        args.particle_ablation.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
