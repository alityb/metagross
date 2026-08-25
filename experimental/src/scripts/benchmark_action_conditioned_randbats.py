#!/usr/bin/env python3
"""Benchmark externally supplied action likelihoods over Randbats active sets.

Each JSONL row requires ``active_candidates``, ``legal_actions``,
``observed_action``, and ``action_likelihoods``. Candidate records use
``candidate_id`` and optional ``prior_weight``. ``label`` is optional and must
be a candidate ID revealed only after the action; label-named fields anywhere
inside active candidates are rejected to prevent leakage.

Likelihoods are an adapter boundary: provide P(action | candidate) from a
candidate-conditioned frozen policy. This script intentionally does not use
public-only opponent priors as a substitute.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import CandidateValidationError, Posterior, load_active_candidates, update_from_action


def _canonical_action(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or any(ord(char) < 32 for char in value):
        raise CandidateValidationError(f"{field} must be a non-empty canonical action string")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CandidateValidationError(f"{field} must be finite")
    return float(value)


def validate_row(row: Any) -> tuple[Posterior, str | None, dict[str, Any]]:
    """Strictly validate one benchmark row and compute its posterior."""
    if not isinstance(row, Mapping):
        raise CandidateValidationError("row must be an object")
    required = {"active_candidates", "legal_actions", "observed_action", "action_likelihoods"}
    missing = required - set(row)
    if missing:
        raise CandidateValidationError(f"row missing required fields: {sorted(missing)}")
    candidates = load_active_candidates(row["active_candidates"])
    legal_actions = row["legal_actions"]
    if not isinstance(legal_actions, list) or not legal_actions:
        raise CandidateValidationError("legal_actions must be a non-empty list")
    legal_actions = [_canonical_action(action, "legal_actions entry") for action in legal_actions]
    if len(set(legal_actions)) != len(legal_actions):
        raise CandidateValidationError("legal_actions must not contain duplicates")
    observed_action = _canonical_action(row["observed_action"], "observed_action")
    if observed_action not in legal_actions:
        raise CandidateValidationError("observed_action is not legal for this pre-action state")
    likelihoods = row["action_likelihoods"]
    if not isinstance(likelihoods, Mapping):
        raise CandidateValidationError("action_likelihoods must be an object")
    posterior = update_from_action(candidates, likelihoods)
    label = row.get("label")
    if label is not None:
        if not isinstance(label, str) or label not in posterior.posterior:
            raise CandidateValidationError("label must be an active candidate_id")
    metadata: dict[str, Any] = {}
    metadata["evidence_type"] = (
        "switch" if observed_action.startswith("switch ")
        else "tera_move" if observed_action.endswith("-tera")
        else "move"
    )
    has_replay = "replay_id" in row or "time" in row
    if has_replay:
        if not isinstance(row.get("replay_id"), str) or not row["replay_id"]:
            raise CandidateValidationError("replay_id must be a non-empty string when replay metadata is present")
        metadata["replay_id"] = row["replay_id"]
        metadata["time"] = _finite_number(row.get("time"), "time")
    return posterior, label, metadata


def _metric_summary(rows: Iterable[tuple[Posterior, str | None, dict[str, Any]]]) -> dict[str, Any]:
    rows = list(rows)
    labeled = [(posterior, label, metadata) for posterior, label, metadata in rows if label is not None]
    result: dict[str, Any] = {"rows": len(rows), "labeled_rows": len(labeled), "coverage": len(labeled) / len(rows) if rows else 0.0}
    for name, posterior_mode in (
        ("generator_only", False),
        ("posterior", True),
        ("move_only_posterior", "move_only"),
    ):
        if not labeled:
            result[name] = {key: None for key in ("top1", "top3", "mrr", "mean_label_probability", "brier", "nll", "ece")}
            continue
        topk_counts = {k: 0 for k in (1, 3, 5, 10)}
        reciprocal_ranks: list[float] = []
        label_probabilities: list[float] = []
        briers: list[float] = []
        nlls: list[float] = []
        zero_label_probability = 0
        reliability = [dict(count=0, confidence_sum=0.0, correct=0) for _ in range(10)]
        for belief, label, metadata in labeled:
            assert label is not None
            use_posterior = posterior_mode is True or (
                posterior_mode == "move_only" and metadata.get("evidence_type") != "switch"
            )
            probabilities = belief.posterior if use_posterior else belief.prior
            ranking = belief.ranking(posterior=use_posterior)
            rank = next(index for index, (candidate_id, _) in enumerate(ranking, start=1) if candidate_id == label)
            for k in topk_counts:
                topk_counts[k] += rank <= min(k, len(ranking))
            reciprocal_ranks.append(1.0 / rank)
            label_probability = probabilities[label]
            label_probabilities.append(label_probability)
            if label_probability > 0.0:
                nlls.append(-math.log(label_probability))
            else:
                zero_label_probability += 1
            briers.append(sum((probability - (candidate_id == label)) ** 2 for candidate_id, probability in probabilities.items()))
            top_candidate, confidence = ranking[0]
            bin_index = min(9, int(confidence * 10))
            reliability[bin_index]["count"] += 1
            reliability[bin_index]["confidence_sum"] += confidence
            reliability[bin_index]["correct"] += top_candidate == label
        count = len(labeled)
        reliability_rows = []
        ece = 0.0
        for index, values in enumerate(reliability):
            bin_count = values["count"]
            mean_confidence = values["confidence_sum"] / bin_count if bin_count else None
            accuracy = values["correct"] / bin_count if bin_count else None
            if bin_count:
                ece += bin_count / count * abs(mean_confidence - accuracy)
            reliability_rows.append({
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": bin_count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            })
        result[name] = {
            "top1": topk_counts[1] / count,
            "top3": topk_counts[3] / count,
            "topk": {str(k): value / count for k, value in topk_counts.items()},
            "mrr": sum(reciprocal_ranks) / count,
            "mean_label_probability": sum(label_probabilities) / count,
            "brier": sum(briers) / count,
            "nll": sum(nlls) / count if not zero_label_probability else "infinity",
            "zero_label_probability": zero_label_probability,
            "ece": ece,
            "reliability": reliability_rows,
        }
    if labeled:
        evidence = [belief.evidence for belief, _label, _metadata in labeled]
        bayes_factors = [belief.posterior[label] / belief.prior[label] for belief, label, _metadata in labeled]
        finite_log_gains = [
            math.log(belief.posterior[label]) - math.log(belief.prior[label])
            for belief, label, _metadata in labeled if belief.posterior[label] > 0.0
        ]
        result["action_evidence"] = {
            "mean": sum(evidence) / len(evidence),
            "median": median(evidence),
            "mean_truth_bayes_factor": sum(bayes_factors) / len(bayes_factors),
            "mean_truth_log_probability_gain": (
                sum(finite_log_gains) / len(finite_log_gains) if finite_log_gains else None
            ),
            "zero_posterior_truth_rows": len(labeled) - len(finite_log_gains),
        }
        prior_metrics = result["generator_only"]
        posterior_metrics = result["posterior"]
        move_only_metrics = result["move_only_posterior"]
        result["delta_posterior_minus_generator"] = {
            "brier": posterior_metrics["brier"] - prior_metrics["brier"],
            "ece": posterior_metrics["ece"] - prior_metrics["ece"],
            "nll": (
                posterior_metrics["nll"] - prior_metrics["nll"]
                if isinstance(posterior_metrics["nll"], float) and isinstance(prior_metrics["nll"], float)
                else None
            ),
            "topk": {
                key: posterior_metrics["topk"][key] - prior_metrics["topk"][key]
                for key in prior_metrics["topk"]
            },
        }
        result["delta_move_only_minus_generator"] = {
            "brier": move_only_metrics["brier"] - prior_metrics["brier"],
            "ece": move_only_metrics["ece"] - prior_metrics["ece"],
            "nll": (
                move_only_metrics["nll"] - prior_metrics["nll"]
                if isinstance(move_only_metrics["nll"], float) and isinstance(prior_metrics["nll"], float)
                else None
            ),
            "topk": {
                key: move_only_metrics["topk"][key] - prior_metrics["topk"][key]
                for key in prior_metrics["topk"]
            },
        }
    return result


def _slice_summary(rows: list[tuple[Posterior, str | None, dict[str, Any]]]) -> dict[str, Any]:
    result = _metric_summary(rows)
    result["evidence_strata"] = {
        evidence_type: _metric_summary([
            row for row in rows if row[2].get("evidence_type") == evidence_type
        ])
        for evidence_type in ("move", "switch", "tera_move")
    }
    return result


def benchmark_rows(rows: list[tuple[Posterior, str | None, dict[str, Any]]], holdout_fraction: float = 0.2) -> dict[str, Any]:
    """Report all rows and, with complete metadata, a chronological replay holdout."""
    report = {"all": _slice_summary(rows)}
    metadata_present = ["replay_id" in metadata for _, _, metadata in rows]
    if any(metadata_present) and not all(metadata_present):
        raise CandidateValidationError("replay_id and time must be present on every row or no rows")
    if not all(metadata_present):
        return report
    replay_times: dict[str, float] = {}
    for _, _, metadata in rows:
        replay_times[metadata["replay_id"]] = min(replay_times.get(metadata["replay_id"], metadata["time"]), metadata["time"])
    ordered_replays = sorted(replay_times, key=lambda replay_id: (replay_times[replay_id], replay_id))
    if len(ordered_replays) < 2:
        report["chronological_holdout"] = {"available": False, "reason": "fewer than two replays"}
        return report
    holdout_count = max(1, math.ceil(len(ordered_replays) * holdout_fraction))
    holdout_count = min(holdout_count, len(ordered_replays) - 1)
    held_out = set(ordered_replays[-holdout_count:])
    report["chronological_holdout"] = {
        "available": True,
        "holdout_replay_ids": sorted(held_out),
        "train": _slice_summary([row for row in rows if row[2]["replay_id"] not in held_out]),
        "holdout": _slice_summary([row for row in rows if row[2]["replay_id"] in held_out]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL benchmark input")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.holdout_fraction) or not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be finite and in (0, 1)")
    validated = []
    try:
        for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise CandidateValidationError(f"line {line_number}: blank JSONL rows are not allowed")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateValidationError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            try:
                validated.append(validate_row(row))
            except CandidateValidationError as exc:
                raise CandidateValidationError(f"line {line_number}: {exc}") from exc
        if not validated:
            raise CandidateValidationError("benchmark input contains no rows")
        report = benchmark_rows(validated, args.holdout_fraction)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    except (OSError, CandidateValidationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
