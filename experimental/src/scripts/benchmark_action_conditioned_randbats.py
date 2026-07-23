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
    has_replay = "replay_id" in row or "time" in row
    if has_replay:
        if not isinstance(row.get("replay_id"), str) or not row["replay_id"]:
            raise CandidateValidationError("replay_id must be a non-empty string when replay metadata is present")
        metadata["replay_id"] = row["replay_id"]
        metadata["time"] = _finite_number(row.get("time"), "time")
    return posterior, label, metadata


def _metric_summary(rows: Iterable[tuple[Posterior, str | None, dict[str, Any]]]) -> dict[str, Any]:
    rows = list(rows)
    labeled = [(posterior, label) for posterior, label, _ in rows if label is not None]
    result: dict[str, Any] = {"rows": len(rows), "labeled_rows": len(labeled), "coverage": len(labeled) / len(rows) if rows else 0.0}
    for name, posterior_mode in (("generator_only", False), ("posterior", True)):
        if not labeled:
            result[name] = {key: None for key in ("top1", "top3", "mrr", "mean_label_probability", "brier")}
            continue
        top1 = top3 = 0
        reciprocal_ranks: list[float] = []
        label_probabilities: list[float] = []
        briers: list[float] = []
        for belief, label in labeled:
            assert label is not None
            probabilities = belief.posterior if posterior_mode else belief.prior
            ranking = belief.ranking(posterior=posterior_mode)
            rank = next(index for index, (candidate_id, _) in enumerate(ranking, start=1) if candidate_id == label)
            top1 += rank == 1
            top3 += rank <= 3
            reciprocal_ranks.append(1.0 / rank)
            label_probabilities.append(probabilities[label])
            briers.append(sum((probability - (candidate_id == label)) ** 2 for candidate_id, probability in probabilities.items()))
        count = len(labeled)
        result[name] = {
            "top1": top1 / count,
            "top3": top3 / count,
            "mrr": sum(reciprocal_ranks) / count,
            "mean_label_probability": sum(label_probabilities) / count,
            "brier": sum(briers) / count,
        }
    return result


def benchmark_rows(rows: list[tuple[Posterior, str | None, dict[str, Any]]], holdout_fraction: float = 0.2) -> dict[str, Any]:
    """Report all rows and, with complete metadata, a chronological replay holdout."""
    report = {"all": _metric_summary(rows)}
    metadata_present = [bool(metadata) for _, _, metadata in rows]
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
        "train": _metric_summary([row for row in rows if row[2]["replay_id"] not in held_out]),
        "holdout": _metric_summary([row for row in rows if row[2]["replay_id"] in held_out]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL benchmark input")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
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
        print(json.dumps(benchmark_rows(validated, args.holdout_fraction), sort_keys=True))
    except (OSError, CandidateValidationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
