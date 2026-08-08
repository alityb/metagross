"""Deterministic statistics for aggregate paired holdout results.

The functions in this module operate only on per-world aggregates.  They do
not reconstruct rollout-level observations or infer missing catastrophe data.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence


_HASH_RE = re.compile(r"[0-9a-f]{64}")
_REQUIRED_FIELDS = {
    "pairs",
    "baseline_sum",
    "candidate_sum",
    "delta_sum",
    "delta_squared_sum",
    "catastrophic_count",
    "baseline_catastrophic_count",
    "candidate_catastrophic_severity_sum",
    "baseline_catastrophic_severity_sum",
    "candidate_better_count",
    "baseline_better_count",
    "equal_count",
}
_OPTIONAL_COUNT_FIELDS = {
    "candidate_catastrophic_count",
    "baseline_terminal_count",
    "candidate_terminal_count",
    "baseline_nonterminal_count",
    "candidate_nonterminal_count",
    "continuation_iterations_executed",
}
_OPTIONAL_NUMERIC_FIELDS = {
    "baseline_nonterminal_evaluation_delta_sum",
    "candidate_nonterminal_evaluation_delta_sum",
}
_NUMERIC_FIELDS = {
    "baseline_sum",
    "candidate_sum",
    "delta_sum",
    "delta_squared_sum",
    "candidate_catastrophic_severity_sum",
    "baseline_catastrophic_severity_sum",
}


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _count(value: object, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} cannot exceed pairs")
    return value


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _validated_row(raw: object, index: int) -> dict[str, float | int]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"row {index} must be a mapping")
    if any(not isinstance(field, str) for field in raw):
        raise ValueError(f"row {index} field names must be strings")
    fields = set(raw)
    if not _REQUIRED_FIELDS <= fields:
        missing = sorted(_REQUIRED_FIELDS - fields)
        raise ValueError(f"row {index} is missing fields: {', '.join(missing)}")
    unknown = (
        fields - _REQUIRED_FIELDS - _OPTIONAL_COUNT_FIELDS - _OPTIONAL_NUMERIC_FIELDS
    )
    if unknown:
        raise ValueError(
            f"row {index} has unknown fields: {', '.join(sorted(unknown))}"
        )

    pairs = _count(raw["pairs"], f"row {index} pairs")
    if pairs == 0:
        raise ValueError(f"row {index} pairs must be positive")

    row: dict[str, float | int] = {"pairs": pairs}
    for name in _NUMERIC_FIELDS:
        row[name] = _finite_number(raw[name], f"row {index} {name}")
    for name in (
        "catastrophic_count",
        "baseline_catastrophic_count",
        "candidate_better_count",
        "baseline_better_count",
        "equal_count",
    ):
        row[name] = _count(raw[name], f"row {index} {name}", pairs)
    for name in _OPTIONAL_COUNT_FIELDS:
        if name in raw:
            maximum = None if name == "continuation_iterations_executed" else pairs
            row[name] = _count(raw[name], f"row {index} {name}", maximum)
    for name in _OPTIONAL_NUMERIC_FIELDS:
        if name in raw:
            row[name] = _finite_number(raw[name], f"row {index} {name}")

    baseline_sum = float(row["baseline_sum"])
    candidate_sum = float(row["candidate_sum"])
    delta_sum = float(row["delta_sum"])
    delta_squared_sum = float(row["delta_squared_sum"])
    tolerance = 1e-10 * max(1, pairs)
    if not -tolerance <= baseline_sum <= pairs + tolerance:
        raise ValueError(f"row {index} baseline_sum is outside [0, pairs]")
    if not -tolerance <= candidate_sum <= pairs + tolerance:
        raise ValueError(f"row {index} candidate_sum is outside [0, pairs]")
    if abs(candidate_sum - baseline_sum - delta_sum) > tolerance:
        raise ValueError(f"row {index} candidate, baseline, and delta sums disagree")
    if not -tolerance <= delta_squared_sum <= pairs + tolerance:
        raise ValueError(f"row {index} delta_squared_sum is outside [0, pairs]")
    if delta_squared_sum + tolerance < delta_sum * delta_sum / pairs:
        raise ValueError(f"row {index} delta moments are inconsistent")
    if (
        int(row["candidate_better_count"])
        + int(row["baseline_better_count"])
        + int(row["equal_count"])
        != pairs
    ):
        raise ValueError(f"row {index} comparison counts do not sum to pairs")
    if (
        "candidate_catastrophic_count" in row
        and row["candidate_catastrophic_count"] != row["catastrophic_count"]
    ):
        raise ValueError(f"row {index} candidate catastrophe alias disagrees")

    for arm in ("baseline", "candidate"):
        terminal_name = f"{arm}_terminal_count"
        nonterminal_name = f"{arm}_nonterminal_count"
        if (
            terminal_name in row
            and nonterminal_name in row
            and int(row[terminal_name]) + int(row[nonterminal_name]) != pairs
        ):
            raise ValueError(
                f"row {index} {arm} terminal and nonterminal counts do not sum to pairs"
            )

    for arm, count_name in (
        ("candidate", "catastrophic_count"),
        ("baseline", "baseline_catastrophic_count"),
    ):
        severity = float(row[f"{arm}_catastrophic_severity_sum"])
        count = int(row[count_name])
        if not 0.0 <= severity <= count:
            raise ValueError(
                f"row {index} {arm} catastrophe severity is outside [0, count]"
            )
    return row


def _lower_tail_cvar(
    records: Sequence[dict[str, object]], tail_mass: float, direction: float
) -> float:
    ordered = sorted(
        records,
        key=lambda record: (
            direction * float(record["delta_mean"]),
            str(record["cluster_hash"]),
            str(record["state_hash"]),
            record["sort_key"],
        ),
    )
    remaining = tail_mass
    total = 0.0
    for record in ordered:
        weight = min(float(record["normalized_weight"]), remaining)
        total += weight * direction * float(record["delta_mean"])
        remaining -= weight
        if remaining <= 1e-15:
            break
    if remaining > 1e-12:
        raise ValueError("posterior weights do not cover the requested CVaR tail")
    return total / tail_mass


def _median_of_means(
    clusters: Sequence[dict[str, object]], groups: int, direction: float
) -> tuple[float, list[float]]:
    buckets: list[list[dict[str, object]]] = [[] for _ in range(groups)]
    for index, cluster in enumerate(clusters):
        buckets[index % groups].append(cluster)
    means = []
    for bucket in buckets:
        mass = math.fsum(float(cluster["normalized_weight"]) for cluster in bucket)
        means.append(
            math.fsum(
                float(cluster["normalized_weight"])
                * direction
                * float(cluster["delta_mean"])
                for cluster in bucket
            )
            / mass
        )
    return float(statistics.median(means)), means


def compute_holdout_metrics(
    rows: Sequence[Mapping[str, object]],
    posterior_weights: Sequence[float],
    state_hashes: Sequence[str],
    cluster_hashes: Sequence[str],
    *,
    alpha: float,
    cvar_tail_mass: float = 0.1,
    mom_groups: int | None = None,
) -> dict[str, object]:
    """Validate aggregate worlds and return deterministic holdout metrics.

    ``catastrophic_count`` is intentionally the candidate catastrophe count to
    preserve the existing holdout row contract.  Severity is a value in
    ``[0, 1]`` per catastrophe and is reported conditional on catastrophe.
    The one-sided tail bounds use weighted Hoeffding with effective sample size
    capped by the effective number of exact clusters.
    """
    for label, values in (
        ("rows", rows),
        ("posterior_weights", posterior_weights),
        ("state_hashes", state_hashes),
        ("cluster_hashes", cluster_hashes),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{label} must be a sequence")
    if not rows:
        raise ValueError("at least one holdout row is required")
    if not (
        len(rows) == len(posterior_weights) == len(state_hashes) == len(cluster_hashes)
    ):
        raise ValueError("rows, weights, state hashes, and cluster hashes must align")

    alpha_value = _finite_number(alpha, "alpha")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    tail_mass = _finite_number(cvar_tail_mass, "cvar_tail_mass")
    if not 0.0 < tail_mass <= 1.0:
        raise ValueError("cvar_tail_mass must be in (0, 1]")
    if mom_groups is not None and (
        isinstance(mom_groups, bool)
        or not isinstance(mom_groups, int)
        or mom_groups < 1
    ):
        raise ValueError("mom_groups must be a positive integer")

    records: list[dict[str, object]] = []
    state_clusters: dict[str, str] = {}
    for index, (raw, raw_weight, raw_state_hash, raw_cluster_hash) in enumerate(
        zip(rows, posterior_weights, state_hashes, cluster_hashes, strict=True)
    ):
        row = _validated_row(raw, index)
        weight = _finite_number(raw_weight, f"posterior weight {index}")
        if weight < 0.0:
            raise ValueError("posterior weights must be non-negative")
        state_hash = _hash(raw_state_hash, f"state hash {index}")
        cluster_hash = _hash(raw_cluster_hash, f"cluster hash {index}")
        previous_cluster = state_clusters.setdefault(state_hash, cluster_hash)
        if previous_cluster != cluster_hash:
            raise ValueError("an exact state hash cannot belong to multiple clusters")
        pairs = int(row["pairs"])
        delta_mean = float(row["delta_sum"]) / pairs
        delta_second_moment = float(row["delta_squared_sum"]) / pairs
        sort_key = tuple(row[name] for name in sorted(row))
        records.append(
            {
                "state_hash": state_hash,
                "cluster_hash": cluster_hash,
                "weight": weight,
                "pairs": pairs,
                "candidate_mean": float(row["candidate_sum"]) / pairs,
                "baseline_mean": float(row["baseline_sum"]) / pairs,
                "delta_mean": delta_mean,
                "within_variance": max(
                    0.0, delta_second_moment - delta_mean * delta_mean
                ),
                "candidate_catastrophe_rate": int(row["catastrophic_count"]) / pairs,
                "baseline_catastrophe_rate": int(row["baseline_catastrophic_count"])
                / pairs,
                "candidate_catastrophe_severity_rate": float(
                    row["candidate_catastrophic_severity_sum"]
                )
                / pairs,
                "baseline_catastrophe_severity_rate": float(
                    row["baseline_catastrophic_severity_sum"]
                )
                / pairs,
                "candidate_better_rate": int(row["candidate_better_count"]) / pairs,
                "baseline_better_rate": int(row["baseline_better_count"]) / pairs,
                "equal_rate": int(row["equal_count"]) / pairs,
                "baseline_nonterminal_evaluation_delta_mass": (
                    float(row["baseline_nonterminal_evaluation_delta_sum"]) / pairs
                    if "baseline_nonterminal_evaluation_delta_sum" in row
                    else None
                ),
                "candidate_nonterminal_evaluation_delta_mass": (
                    float(row["candidate_nonterminal_evaluation_delta_sum"]) / pairs
                    if "candidate_nonterminal_evaluation_delta_sum" in row
                    else None
                ),
                "baseline_nonterminal_mass": (
                    int(row["baseline_nonterminal_count"]) / pairs
                    if "baseline_nonterminal_count" in row
                    else None
                ),
                "candidate_nonterminal_mass": (
                    int(row["candidate_nonterminal_count"]) / pairs
                    if "candidate_nonterminal_count" in row
                    else None
                ),
                "sort_key": sort_key,
            }
        )

    total_weight = math.fsum(float(record["weight"]) for record in records)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("posterior weights must contain positive finite mass")
    for record in records:
        record["normalized_weight"] = float(record["weight"]) / total_weight
    records.sort(
        key=lambda record: (
            str(record["cluster_hash"]),
            str(record["state_hash"]),
            record["sort_key"],
            float(record["normalized_weight"]),
        )
    )

    normalized_weights = [float(record["normalized_weight"]) for record in records]
    mean_delta = math.fsum(
        weight * float(record["delta_mean"])
        for weight, record in zip(normalized_weights, records, strict=True)
    )
    candidate_mean = math.fsum(
        weight * float(record["candidate_mean"])
        for weight, record in zip(normalized_weights, records, strict=True)
    )
    baseline_mean = math.fsum(
        weight * float(record["baseline_mean"])
        for weight, record in zip(normalized_weights, records, strict=True)
    )

    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["cluster_hash"]), []).append(record)
    clusters: list[dict[str, object]] = []
    for cluster_hash in sorted(grouped):
        members = grouped[cluster_hash]
        mass = math.fsum(float(member["normalized_weight"]) for member in members)
        if mass == 0.0:
            delta_mean = 0.0
        else:
            delta_mean = (
                math.fsum(
                    float(member["normalized_weight"]) * float(member["delta_mean"])
                    for member in members
                )
                / mass
            )
        clusters.append(
            {
                "cluster_hash": cluster_hash,
                "state_hashes": sorted(
                    {str(member["state_hash"]) for member in members}
                ),
                "world_count": len(members),
                "pairs": sum(int(member["pairs"]) for member in members),
                "normalized_weight": mass,
                "delta_mean": delta_mean,
            }
        )
    active_clusters = [
        cluster for cluster in clusters if float(cluster["normalized_weight"]) > 0.0
    ]
    if mom_groups is None:
        group_count = min(3, len(active_clusters))
    elif mom_groups > len(active_clusters):
        raise ValueError("mom_groups cannot exceed positive-mass clusters")
    else:
        group_count = mom_groups

    cluster_weight_squares = math.fsum(
        float(cluster["normalized_weight"]) ** 2 for cluster in active_clusters
    )
    effective_clusters = 1.0 / cluster_weight_squares
    centered_cluster_variance = math.fsum(
        float(cluster["normalized_weight"])
        * (float(cluster["delta_mean"]) - mean_delta) ** 2
        for cluster in active_clusters
    )
    correction = 1.0 - cluster_weight_squares
    between_cluster_variance = (
        centered_cluster_variance / correction
        if len(active_clusters) > 1 and correction > 0.0
        else 0.0
    )
    between_mean_variance = between_cluster_variance * cluster_weight_squares
    within_world_mean_variance = math.fsum(
        float(record["normalized_weight"]) ** 2
        * float(record["within_variance"])
        / int(record["pairs"])
        for record in records
    )
    standard_error = math.sqrt(
        max(0.0, between_mean_variance + within_world_mean_variance)
    )
    pair_weight_squares = math.fsum(
        float(record["normalized_weight"]) ** 2 / int(record["pairs"])
        for record in records
    )
    effective_pairs = 1.0 / pair_weight_squares

    def weighted(field: str) -> float:
        return math.fsum(
            float(record["normalized_weight"]) * float(record[field])
            for record in records
        )

    def weighted_nonterminal_evaluation_delta_mean(arm: str) -> float | None:
        mass_field = f"{arm}_nonterminal_mass"
        delta_field = f"{arm}_nonterminal_evaluation_delta_mass"
        if any(
            record[mass_field] is None or record[delta_field] is None
            for record in records
        ):
            return None
        nonterminal_mass = weighted(mass_field)
        if nonterminal_mass == 0.0:
            return None
        return weighted(delta_field) / nonterminal_mass

    candidate_catastrophe_rate = weighted("candidate_catastrophe_rate")
    baseline_catastrophe_rate = weighted("baseline_catastrophe_rate")
    candidate_severity_mass = weighted("candidate_catastrophe_severity_rate")
    baseline_severity_mass = weighted("baseline_catastrophe_severity_rate")
    candidate_severity_mean = (
        candidate_severity_mass / candidate_catastrophe_rate
        if candidate_catastrophe_rate > 0.0
        else 0.0
    )
    baseline_severity_mean = (
        baseline_severity_mass / baseline_catastrophe_rate
        if baseline_catastrophe_rate > 0.0
        else 0.0
    )
    candidate_nonterminal_evaluation_delta_mean = (
        weighted_nonterminal_evaluation_delta_mean("candidate")
    )
    baseline_nonterminal_evaluation_delta_mean = (
        weighted_nonterminal_evaluation_delta_mean("baseline")
    )
    nonterminal_evaluation_delta_mean_difference = (
        candidate_nonterminal_evaluation_delta_mean
        - baseline_nonterminal_evaluation_delta_mean
        if candidate_nonterminal_evaluation_delta_mean is not None
        and baseline_nonterminal_evaluation_delta_mean is not None
        else None
    )
    tail_effective_sample_size = min(effective_pairs, effective_clusters)
    tail_radius = math.sqrt(
        math.log(1.0 / alpha_value) / (2.0 * tail_effective_sample_size)
    )
    candidate_mom, candidate_group_means = _median_of_means(
        active_clusters, group_count, 1.0
    )
    baseline_mom, baseline_group_means = _median_of_means(
        active_clusters, group_count, -1.0
    )

    world_summaries = [
        {
            "state_hash": record["state_hash"],
            "cluster_hash": record["cluster_hash"],
            "pairs": record["pairs"],
            "normalized_weight": record["normalized_weight"],
            "delta_mean": record["delta_mean"],
        }
        for record in records
    ]
    return {
        "schema_version": 1,
        "world_count": len(records),
        "cluster_count": len(clusters),
        "positive_mass_cluster_count": len(active_clusters),
        "total_pairs": sum(int(record["pairs"]) for record in records),
        "normalized_weights": normalized_weights,
        "worlds": world_summaries,
        "cluster_aggregates": clusters,
        "weighted_candidate_mean": candidate_mean,
        "weighted_baseline_mean": baseline_mean,
        "weighted_mean_delta": mean_delta,
        "weighted_candidate_nonterminal_evaluation_delta_mean": (
            candidate_nonterminal_evaluation_delta_mean
        ),
        "weighted_baseline_nonterminal_evaluation_delta_mean": (
            baseline_nonterminal_evaluation_delta_mean
        ),
        "weighted_nonterminal_evaluation_delta_mean_difference": (
            nonterminal_evaluation_delta_mean_difference
        ),
        "between_cluster_variance": between_cluster_variance,
        "between_mean_variance": between_mean_variance,
        "within_world_mean_variance": within_world_mean_variance,
        "standard_error": standard_error,
        "effective_clusters": effective_clusters,
        "effective_pairs": effective_pairs,
        "candidate_catastrophe_rate": candidate_catastrophe_rate,
        "baseline_catastrophe_rate": baseline_catastrophe_rate,
        "candidate_catastrophe_severity_mean": candidate_severity_mean,
        "baseline_catastrophe_severity_mean": baseline_severity_mean,
        "candidate_better_rate": weighted("candidate_better_rate"),
        "baseline_better_rate": weighted("baseline_better_rate"),
        "equal_rate": weighted("equal_rate"),
        "alpha": alpha_value,
        "tail_bound_method": "weighted_hoeffding_cluster_capped",
        "tail_effective_sample_size": tail_effective_sample_size,
        "candidate_tail_upper_confidence_bound": min(
            1.0, candidate_catastrophe_rate + tail_radius
        ),
        "baseline_tail_upper_confidence_bound": min(
            1.0, baseline_catastrophe_rate + tail_radius
        ),
        "positive_cluster_mass": math.fsum(
            float(cluster["normalized_weight"])
            for cluster in active_clusters
            if float(cluster["delta_mean"]) > 0.0
        ),
        "cvar_tail_mass": tail_mass,
        "candidate_lower_tail_cvar": _lower_tail_cvar(records, tail_mass, 1.0),
        "baseline_lower_tail_cvar": _lower_tail_cvar(records, tail_mass, -1.0),
        "mom_groups": group_count,
        "candidate_median_of_means": candidate_mom,
        "baseline_median_of_means": baseline_mom,
        "candidate_mom_group_means": candidate_group_means,
        "baseline_mom_group_means": baseline_group_means,
    }
