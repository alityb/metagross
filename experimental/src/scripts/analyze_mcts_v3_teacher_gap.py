#!/usr/bin/env python3
"""Audit embedded parent policies against finalized schema-v3 MCTS targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

N_ACTIONS = 13
MASS_TOLERANCE = 1e-4
SMOOTHING_EPSILON = 1e-12
TOP_K = 3
METRIC_NAMES = (
    "top1_fractional_agreement",
    "top1_set_overlap",
    "top3_agreement",
    "parent_top_tie_rate",
    "search_top_tie_rate",
    "top_action_disjoint_rate",
    "selected_action_outside_parent_top_set_rate",
    "cross_entropy_nats",
    "kl_target_parent_nats",
    "jensen_shannon_nats",
    "total_variation",
    "parent_entropy_nats",
    "search_entropy_nats",
    "search_top_mass",
    "search_top_margin",
    "parent_top_mass",
    "parent_probability_on_search_top_action",
    "parent_probability_on_selected_action",
)
PRIMARY_METRICS = METRIC_NAMES
REPORT_COUNT_FIELDS = (
    "groups_total",
    "groups_admitted",
    "groups_rejected",
    "targets_written",
    "target_files",
    "targets",
    "target_groups",
    "learner_trajectories",
    "target_groups_without_parsed_trajectory",
    "parse_only_trajectories",
    "decision_records",
    "raw_replay_files",
    "unique_replay_battles",
    "root_prior_decisions",
    "opponent_prior_decisions",
    "decision_battles",
    "battle_result_records",
)
REPORT_COUNT_MAPS = (
    "rejection_reasons",
    "dump_stats",
    "decision_stats",
)
REPORT_MATCH_STATS = (
    "skipped_forced_action_decisions",
    "ignored_forced_visit_mass",
    "mask_fallback_decisions",
)


class AnalysisError(ValueError):
    """A fail-closed dataset or command-line validation error."""


@dataclass(frozen=True)
class InputSpec:
    name: str
    parent_policy_identity: str
    parent_policy_sha256: str
    jsonl_path: Path
    report_paths: tuple[Path, ...] = ()


@dataclass
class MetricTotal:
    decision_count: int = 0
    sums: list[float] = field(
        default_factory=lambda: [0.0] * len(METRIC_NAMES)
    )

    def add(self, values: tuple[float, ...]) -> None:
        self.decision_count += 1
        for index, value in enumerate(values):
            self.sums[index] += value


@dataclass
class Accumulator(MetricTotal):
    battle_tags: set[str] = field(default_factory=set)

    def add_row(self, values: tuple[float, ...], battle_tag: str) -> None:
        self.add(values)
        self.battle_tags.add(battle_tag)

    def result(self) -> dict[str, float | int]:
        if not self.decision_count:
            raise AnalysisError("cannot summarize an empty accumulator")
        battle_count = len(self.battle_tags)
        return {
            "decision_count": self.decision_count,
            "battle_count": battle_count,
            "mean_decisions_per_battle": self.decision_count / battle_count,
            **{
                name: self.sums[index] / self.decision_count
                for index, name in enumerate(METRIC_NAMES)
            },
        }


@dataclass
class DatasetSummary:
    spec: InputSpec
    input_sha256: str
    aggregate: Accumulator
    battles: dict[str, MetricTotal]
    strata: dict[str, dict[str, Accumulator]]


def _error(path: Path, line_number: int, message: str) -> AnalysisError:
    return AnalysisError(f"{path}:{line_number}: {message}")


def _validate_spec(spec: InputSpec) -> None:
    if not spec.name.strip():
        raise AnalysisError("dataset name must not be empty")
    if not spec.parent_policy_identity.strip():
        raise AnalysisError(f"{spec.name}: parent policy identity must not be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", spec.parent_policy_sha256):
        raise AnalysisError(
            f"{spec.name}: parent policy SHA-256 must be exactly 64 hexadecimal characters"
        )


def _distribution(
    row: dict[str, Any], field_name: str, illegal: list[bool], path: Path, line_number: int
) -> list[float]:
    raw_values = row.get(field_name)
    if not isinstance(raw_values, list) or len(raw_values) != N_ACTIONS:
        raise _error(path, line_number, f"{field_name} must be a {N_ACTIONS}-wide list")
    values: list[float] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise _error(path, line_number, f"{field_name} contains a non-number")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise _error(
                path, line_number, f"{field_name} contains non-finite or negative mass"
            )
        values.append(value)
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise _error(path, line_number, f"{field_name} mass is {total!r}, expected 1")
    if any(flag and value > 0.0 for flag, value in zip(illegal, values)):
        raise _error(path, line_number, f"{field_name} has mass on an illegal action")
    return [value / total for value in values]


def _entropy(distribution: list[float]) -> float:
    return -math.fsum(value * math.log(value) for value in distribution if value > 0.0)


def _top_set(distribution: list[float], legal_indices: list[int]) -> set[int]:
    top = max(distribution[index] for index in legal_indices)
    return {index for index in legal_indices if distribution[index] == top}


def _smoothed(
    distribution: list[float], legal_indices: list[int]
) -> list[float]:
    denominator = (
        math.fsum(distribution[index] for index in legal_indices)
        + SMOOTHING_EPSILON * len(legal_indices)
    )
    return [
        (distribution[index] + SMOOTHING_EPSILON) / denominator
        if index in legal_indices
        else 0.0
        for index in range(N_ACTIONS)
    ]


def calculate_row_metrics(
    parent: list[float],
    target: list[float],
    illegal: list[bool],
    selected_action_index: int,
) -> tuple[float, ...]:
    """Reduce normalized distributions to the fixed metric tuple exactly once."""
    legal_indices = [index for index, flag in enumerate(illegal) if not flag]
    parent_top = _top_set(parent, legal_indices)
    search_top = _top_set(target, legal_indices)
    intersection = parent_top & search_top
    fractional_top1 = len(intersection) / (len(parent_top) * len(search_top))
    top1_overlap = float(bool(intersection))

    ranked_parent = sorted((parent[index] for index in legal_indices), reverse=True)
    top_k_boundary = ranked_parent[min(TOP_K, len(ranked_parent)) - 1]
    parent_top_k = {
        index for index in legal_indices if parent[index] >= top_k_boundary
    }
    top_k_agreement = len(search_top & parent_top_k) / len(search_top)

    smoothed_parent = _smoothed(parent, legal_indices)
    smoothed_target = _smoothed(target, legal_indices)
    cross_entropy = -math.fsum(
        smoothed_target[index] * math.log(smoothed_parent[index])
        for index in legal_indices
    )
    kl = math.fsum(
        smoothed_target[index]
        * math.log(smoothed_target[index] / smoothed_parent[index])
        for index in legal_indices
    )
    midpoint = [(parent[index] + target[index]) / 2.0 for index in range(N_ACTIONS)]
    js = 0.5 * math.fsum(
        target[index] * math.log(target[index] / midpoint[index])
        for index in legal_indices
        if target[index] > 0.0
    ) + 0.5 * math.fsum(
        parent[index] * math.log(parent[index] / midpoint[index])
        for index in legal_indices
        if parent[index] > 0.0
    )
    ranked_target = sorted((target[index] for index in legal_indices), reverse=True)
    search_margin = ranked_target[0] - ranked_target[1] if len(ranked_target) > 1 else 1.0
    search_top_mass = target[next(iter(search_top))]
    parent_top_mass = parent[next(iter(parent_top))]
    parent_on_search_top = math.fsum(parent[index] for index in search_top) / len(search_top)
    values = {
        "top1_fractional_agreement": fractional_top1,
        "top1_set_overlap": top1_overlap,
        "top3_agreement": top_k_agreement,
        "parent_top_tie_rate": float(len(parent_top) > 1),
        "search_top_tie_rate": float(len(search_top) > 1),
        "top_action_disjoint_rate": 1.0 - top1_overlap,
        "selected_action_outside_parent_top_set_rate": float(
            selected_action_index not in parent_top
        ),
        "cross_entropy_nats": cross_entropy,
        "kl_target_parent_nats": max(0.0, kl),
        "jensen_shannon_nats": max(0.0, js),
        "total_variation": 0.5
        * math.fsum(abs(parent[index] - target[index]) for index in legal_indices),
        "parent_entropy_nats": _entropy(parent),
        "search_entropy_nats": _entropy(target),
        "search_top_mass": search_top_mass,
        "search_top_margin": search_margin,
        "parent_top_mass": parent_top_mass,
        "parent_probability_on_search_top_action": parent_on_search_top,
        "parent_probability_on_selected_action": parent[selected_action_index],
    }
    return tuple(values[name] for name in METRIC_NAMES)


def _turn_bin(turn: int | None) -> str:
    if turn is None:
        return "unknown"
    if turn <= 5:
        return "0-5"
    if turn <= 10:
        return "6-10"
    if turn <= 20:
        return "11-20"
    if turn <= 30:
        return "21-30"
    return "31+"


def _action_kind(index: int) -> str:
    if index <= 3:
        return "normal_move"
    if index <= 8:
        return "switch"
    return "tera_move"


def _add_stratum(
    strata: dict[str, dict[str, Accumulator]],
    dimension: str,
    category: str,
    values: tuple[float, ...],
    battle_tag: str,
) -> None:
    strata[dimension].setdefault(category, Accumulator()).add_row(values, battle_tag)


def load_dataset(spec: InputSpec) -> DatasetSummary:
    """Stream one finalized dataset into compact aggregate and battle summaries."""
    _validate_spec(spec)
    aggregate = Accumulator()
    battles: dict[str, MetricTotal] = {}
    strata: dict[str, dict[str, Accumulator]] = {
        "battle_turn_bin": {},
        "legal_action_count": {},
        "selected_action_kind": {},
        "label": {},
    }
    seen: dict[tuple[str, str, int], int] = {}
    labels_by_pov: dict[tuple[str, str], int | None] = {}
    input_hash = hashlib.sha256()
    try:
        source = spec.jsonl_path.open("rb")
    except OSError as exc:
        raise AnalysisError(f"cannot read dataset {spec.jsonl_path}: {exc}") from exc
    with source:
        for line_number, raw_line in enumerate(source, 1):
            input_hash.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _error(spec.jsonl_path, line_number, "invalid UTF-8 JSON") from exc
            if not isinstance(row, dict):
                raise _error(spec.jsonl_path, line_number, "record must be a JSON object")
            if type(row.get("schema")) is not int or row["schema"] != 3:
                raise _error(spec.jsonl_path, line_number, "schema must be integer 3")

            battle_tag = row.get("battle_tag")
            username = row.get("username")
            decision_idx = row.get("decision_idx")
            if not isinstance(battle_tag, str) or not battle_tag.strip():
                raise _error(spec.jsonl_path, line_number, "invalid battle_tag")
            if not isinstance(username, str) or not username.strip():
                raise _error(spec.jsonl_path, line_number, "invalid username")
            if (
                isinstance(decision_idx, bool)
                or not isinstance(decision_idx, int)
                or decision_idx < 0
            ):
                raise _error(spec.jsonl_path, line_number, "invalid decision_idx")
            key = (battle_tag, username, decision_idx)
            if key in seen:
                raise _error(
                    spec.jsonl_path,
                    line_number,
                    f"duplicate key {key!r}; first seen on line {seen[key]}",
                )
            seen[key] = line_number

            illegal_raw = row.get("illegal_actions")
            if (
                not isinstance(illegal_raw, list)
                or len(illegal_raw) != N_ACTIONS
                or not all(isinstance(value, bool) for value in illegal_raw)
            ):
                raise _error(
                    spec.jsonl_path,
                    line_number,
                    f"illegal_actions must be a {N_ACTIONS}-wide boolean list",
                )
            illegal = list(illegal_raw)
            if all(illegal):
                raise _error(spec.jsonl_path, line_number, "record has no legal actions")
            parent = _distribution(
                row, "policy_probs", illegal, spec.jsonl_path, line_number
            )
            target = _distribution(
                row, "visit_target_13", illegal, spec.jsonl_path, line_number
            )

            selected = row.get("selected_action_index")
            if (
                isinstance(selected, bool)
                or not isinstance(selected, int)
                or not 0 <= selected < N_ACTIONS
            ):
                raise _error(
                    spec.jsonl_path, line_number, "selected_action_index is out of range"
                )
            if illegal[selected]:
                raise _error(
                    spec.jsonl_path, line_number, "selected_action_index is illegal"
                )
            if target[selected] <= 0.0:
                raise _error(
                    spec.jsonl_path,
                    line_number,
                    "selected_action_index has zero visit-target mass",
                )

            label = row.get("label")
            if label is not None and (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in (0, 1)
            ):
                raise _error(spec.jsonl_path, line_number, "label must be 0, 1, or null")
            pov = (battle_tag, username)
            if pov in labels_by_pov and labels_by_pov[pov] != label:
                raise _error(
                    spec.jsonl_path, line_number, f"conflicting label state for {pov!r}"
                )
            labels_by_pov[pov] = label

            turn = row.get("battle_turn")
            if turn is not None and (
                isinstance(turn, bool) or not isinstance(turn, int) or turn < 0
            ):
                raise _error(
                    spec.jsonl_path,
                    line_number,
                    "battle_turn must be a nonnegative integer or null",
                )

            values = calculate_row_metrics(parent, target, illegal, selected)
            aggregate.add_row(values, battle_tag)
            battles.setdefault(battle_tag, MetricTotal()).add(values)
            _add_stratum(
                strata, "battle_turn_bin", _turn_bin(turn), values, battle_tag
            )
            _add_stratum(
                strata,
                "legal_action_count",
                str(illegal.count(False)),
                values,
                battle_tag,
            )
            _add_stratum(
                strata,
                "selected_action_kind",
                _action_kind(selected),
                values,
                battle_tag,
            )
            _add_stratum(
                strata,
                "label",
                "missing" if label is None else str(label),
                values,
                battle_tag,
            )
    if not aggregate.decision_count:
        raise AnalysisError(f"{spec.jsonl_path}: dataset contains no records")
    return DatasetSummary(
        spec=spec,
        input_sha256=input_hash.hexdigest(),
        aggregate=aggregate,
        battles=battles,
        strata=strata,
    )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    battles: dict[str, MetricTotal], repeats: int, seed: int
) -> dict[str, dict[str, float | int | str]]:
    """Bootstrap precomputed metric sums by whole battle, without row replay."""
    if not battles:
        raise AnalysisError("cannot bootstrap an empty dataset")
    if repeats < 1:
        raise AnalysisError("bootstrap repeats must be positive")
    clusters = [battles[tag] for tag in sorted(battles)]
    rng = random.Random(seed)
    samples = [[] for _ in METRIC_NAMES]
    for _ in range(repeats):
        decision_count = 0
        sums = [0.0] * len(METRIC_NAMES)
        for _ in clusters:
            selected = clusters[rng.randrange(len(clusters))]
            decision_count += selected.decision_count
            for index, value in enumerate(selected.sums):
                sums[index] += value
        for index, value in enumerate(sums):
            samples[index].append(value / decision_count)
    return {
        name: {
            "lower": _percentile(samples[index], 0.025),
            "upper": _percentile(samples[index], 0.975),
            "confidence": 0.95,
            "repeats": repeats,
            "cluster_unit": "battle_tag",
        }
        for index, name in enumerate(METRIC_NAMES)
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _numeric_count_map(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): count
        for key, count in value.items()
        if isinstance(count, (int, float))
        and not isinstance(count, bool)
        and math.isfinite(float(count))
        and count >= 0
    }


def _report_provenance(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    provenance = []
    for path in paths:
        report_hash = _sha256_file(path)
        try:
            with path.open("r", encoding="utf-8") as source:
                content = json.load(source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"cannot read report {path}: {exc}") from exc
        if not isinstance(content, dict):
            raise AnalysisError(f"report {path} must contain a JSON object")
        counts = {
            key: content[key]
            for key in REPORT_COUNT_FIELDS
            if isinstance(content.get(key), int)
            and not isinstance(content[key], bool)
            and content[key] >= 0
        }
        count_maps = {
            key: extracted
            for key in REPORT_COUNT_MAPS
            if (extracted := _numeric_count_map(content.get(key)))
        }
        match_stats = _numeric_count_map(content.get("match_stats"))
        selected_match_stats = {
            key: match_stats[key] for key in REPORT_MATCH_STATS if key in match_stats
        }
        if selected_match_stats:
            count_maps["match_stats"] = selected_match_stats
        if isinstance(content.get("errors"), list):
            counts["errors_count"] = len(content["errors"])
        provenance.append(
            {
                "path": str(path),
                "sha256": report_hash,
                "counts_included_in_analyzed_rows": False,
                "counts": counts,
                "count_maps": count_maps,
            }
        )
    return provenance


def _derived_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _dataset_analysis(
    summary: DatasetSummary, seed: int, repeats: int
) -> dict[str, Any]:
    spec = summary.spec
    return {
        "name": spec.name,
        "parent_policy": {
            "identity": spec.parent_policy_identity,
            "sha256": spec.parent_policy_sha256.lower(),
        },
        "input": {
            "path": str(spec.jsonl_path),
            "sha256": summary.input_sha256,
        },
        "metrics": summary.aggregate.result(),
        "confidence_intervals": cluster_bootstrap(
            summary.battles, repeats, _derived_seed(seed, spec.name)
        ),
        "strata": {
            dimension: {
                category: accumulator.result()
                for category, accumulator in sorted(groups.items())
            }
            for dimension, groups in summary.strata.items()
        },
        "provenance_reports": _report_provenance(spec.report_paths),
    }


def analyze(specs: list[InputSpec], seed: int = 0, repeats: int = 2000) -> dict[str, Any]:
    if not specs:
        raise AnalysisError("at least one input specification is required")
    if repeats < 1:
        raise AnalysisError("bootstrap repeats must be positive")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AnalysisError("dataset names must be unique")
    summaries = [load_dataset(spec) for spec in specs]
    datasets = [_dataset_analysis(summary, seed, repeats) for summary in summaries]
    return {
        "analysis_schema_version": 2,
        "distribution_validation": {
            "mass_absolute_tolerance": MASS_TOLERANCE,
            "post_validation_normalization": True,
            "illegal_mass_allowed": False,
        },
        "divergence_smoothing": {
            "method": (
                "add epsilon to every legal action in both normalized distributions, "
                "then renormalize; CE and KL use these smoothed distributions"
            ),
            "epsilon": SMOOTHING_EPSILON,
            "jensen_shannon_uses_unsmoothed_distributions": True,
        },
        "bootstrap": {
            "seed": seed,
            "repeats": repeats,
            "confidence": 0.95,
            "cluster_unit": "battle_tag within each dataset",
        },
        "metric_definitions": {
            "top1_fractional_agreement": (
                "expected exact agreement under independent uniform tie-breaking: "
                "intersection_size / (parent_top_set_size * search_top_set_size)"
            ),
            "top1_set_overlap": "indicator that parent and search argmax sets overlap",
            "top3_agreement": (
                "fraction of search argmax actions in the parent's tie-inclusive top-3 set"
            ),
            "top_action_disjoint_rate": "parent and search argmax sets have no overlap",
            "selected_action_outside_parent_top_set_rate": (
                "recorded selected action is not in the parent argmax set"
            ),
            "parent_probability_on_search_top_action": (
                "mean parent probability over the search argmax set"
            ),
            "parent_probability_on_selected_action": (
                "parent probability on the recorded selected_action_index"
            ),
            "search_top_margin": (
                "largest minus second-largest legal visit-target probability; zero on a top tie"
            ),
        },
        "stratification_definitions": {
            "battle_turn_bins": ["0-5", "6-10", "11-20", "21-30", "31+", "unknown"],
            "canonical_action_kinds": {
                "normal_move": "indices 0-3",
                "switch": "indices 4-8",
                "tera_move": "indices 9-12",
            },
            "label": ["0", "1", "missing"],
        },
        "datasets": datasets,
        "combined_counts": {
            "dataset_count": len(datasets),
            "decision_count": sum(
                dataset["metrics"]["decision_count"] for dataset in datasets
            ),
            "battle_count": sum(dataset["metrics"]["battle_count"] for dataset in datasets),
        },
        "provenance_note": (
            "Allowlisted report counts are provenance only. They are not added to analyzed "
            "counts, and absent rejection categories are not inferred. No report payload is embedded."
        ),
    }


def _format(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.4f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    bootstrap = analysis["bootstrap"]
    lines = [
        "# Schema-v3 Teacher-Gap Audit",
        "",
        (
            f"Battle-clustered 95% percentile bootstrap: seed `{bootstrap['seed']}`, "
            f"{bootstrap['repeats']} repeats. Metrics are never pooled across datasets."
        ),
        "",
        "| Dataset | Parent policy | Decisions | Battles | Fractional top-1 (95% CI) | KL | JS | TV | Selected outside parent top |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in analysis["datasets"]:
        metrics = dataset["metrics"]
        interval = dataset["confidence_intervals"]["top1_fractional_agreement"]
        lines.append(
            "| {name} | `{identity}` (`{sha}`) | {decisions} | {battles} | "
            "{top1} [{lower}, {upper}] | {kl} | {js} | {tv} | {selected} |".format(
                name=dataset["name"],
                identity=dataset["parent_policy"]["identity"],
                sha=dataset["parent_policy"]["sha256"][:12],
                decisions=metrics["decision_count"],
                battles=metrics["battle_count"],
                top1=_format(metrics["top1_fractional_agreement"]),
                lower=_format(interval["lower"]),
                upper=_format(interval["upper"]),
                kl=_format(metrics["kl_target_parent_nats"]),
                js=_format(metrics["jensen_shannon_nats"]),
                tv=_format(metrics["total_variation"]),
                selected=_format(
                    metrics["selected_action_outside_parent_top_set_rate"]
                ),
            )
        )
    for dataset in analysis["datasets"]:
        lines.extend(["", f"## {dataset['name']} Strata"])
        for dimension, groups in dataset["strata"].items():
            lines.extend(
                [
                    "",
                    f"### {dimension.replace('_', ' ').title()}",
                    "",
                    "| Stratum | Decisions | Battles | Fractional top-1 | Top-3 | Search entropy | Top mass |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name, metrics in groups.items():
                lines.append(
                    f"| {name} | {metrics['decision_count']} | {metrics['battle_count']} | "
                    f"{_format(metrics['top1_fractional_agreement'])} | "
                    f"{_format(metrics['top3_agreement'])} | "
                    f"{_format(metrics['search_entropy_nats'])} | "
                    f"{_format(metrics['search_top_mass'])} |"
                )
    lines.extend(["", "## Provenance", "", analysis["provenance_note"]])
    for dataset in analysis["datasets"]:
        for report in dataset["provenance_reports"]:
            lines.append(
                f"- `{dataset['name']}`: `{report['path']}` SHA-256 `{report['sha256']}`; "
                f"counts `{json.dumps(report['counts'], sort_keys=True)}`; "
                f"count maps `{json.dumps(report['count_maps'], sort_keys=True)}`"
            )
    if not any(dataset["provenance_reports"] for dataset in analysis["datasets"]):
        lines.append("- No finalization or shard reports supplied.")
    return "\n".join(lines) + "\n"


def _parse_input(values: list[str]) -> InputSpec:
    if len(values) < 4:
        raise AnalysisError(
            "each --input requires DATASET PARENT_IDENTITY POLICY_SHA256 JSONL [REPORT ...]"
        )
    return InputSpec(
        values[0], values[1], values[2], Path(values[3]), tuple(map(Path, values[4:]))
    )


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_output_paths(
    specs: list[InputSpec], output_json: Path, output_markdown: Path, force: bool
) -> None:
    output_paths = [_resolved(output_json), _resolved(output_markdown)]
    if output_paths[0] == output_paths[1]:
        raise AnalysisError("JSON and Markdown output paths must differ")
    source_paths = {
        _resolved(path)
        for spec in specs
        for path in (spec.jsonl_path, *spec.report_paths)
    }
    collisions = [path for path in output_paths if path in source_paths]
    if collisions:
        raise AnalysisError(f"output path collides with an input or report: {collisions[0]}")
    if not force:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise AnalysisError(f"output path already exists: {existing[0]}; use --force")


def _atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        if force:
            os.replace(temporary_name, path)
            temporary_name = None
        else:
            try:
                os.link(temporary_name, path)
            except FileExistsError as exc:
                raise AnalysisError(f"output path already exists: {path}; use --force") from exc
            os.unlink(temporary_name)
            temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        nargs="+",
        required=True,
        metavar="VALUE",
        help=(
            "DATASET PARENT_IDENTITY POLICY_SHA256 JSONL [REPORT ...]; repeat for each "
            "dataset. A following option or --input terminates the specification."
        ),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Atomically replace outputs.")
    args = parser.parse_args()
    try:
        specs = [_parse_input(values) for values in args.input]
        for spec in specs:
            _validate_spec(spec)
        validate_output_paths(
            specs, args.output_json, args.output_markdown, args.force
        )
        result = analyze(specs, args.bootstrap_seed, args.bootstrap_repeats)
        _atomic_write(
            args.output_json,
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            args.force,
        )
        _atomic_write(args.output_markdown, render_markdown(result), args.force)
    except (AnalysisError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "datasets": len(result["datasets"]),
                "decisions": result["combined_counts"]["decision_count"],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
