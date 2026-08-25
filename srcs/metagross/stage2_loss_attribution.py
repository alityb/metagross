#!/usr/bin/env python3
"""Attribute Stage 2 N=20 outcomes to recorded shared-root decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_class(action: str | None) -> str:
    if not action:
        return "missing"
    if action.startswith("switch "):
        return "switch"
    if action.endswith("-tera"):
        return "tera_move"
    return "ordinary_move"


def _entropy(policy: list[dict]) -> tuple[float, float]:
    entropy = -math.fsum(
        probability * math.log(probability)
        for entry in policy
        if (probability := float(entry["probability"])) > 0
    )
    normalized = entropy / math.log(len(policy)) if len(policy) > 1 else 0.0
    return entropy, normalized


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _aggregate(decisions: list[dict]) -> dict:
    divergent = [decision for decision in decisions if not decision["baseline_agreement"]]
    deltas = [decision["counterfactual_delta"] for decision in decisions]
    divergent_deltas = [decision["counterfactual_delta"] for decision in divergent]
    transitions: dict[str, int] = {}
    correction_reasons: dict[str, int] = {}
    for decision in decisions:
        transition = f'{decision["baseline_class"]}->{decision["sampled_class"]}'
        transitions[transition] = transitions.get(transition, 0) + 1
        if decision["selection_class"] == "deterministic_correction":
            reason = decision["selection_reason"]
            correction_reasons[reason] = correction_reasons.get(reason, 0) + 1
    return {
        "decisions": len(decisions),
        "baseline_agreements": len(decisions) - len(divergent),
        "root_divergences": len(divergent),
        "sampled_policy_argmax": sum(decision["sampled_is_policy_argmax"] for decision in decisions),
        "sampled_non_argmax": sum(not decision["sampled_is_policy_argmax"] for decision in decisions),
        "entropy": {
            "mean": _mean([decision["entropy"] for decision in decisions]),
            "median": _median([decision["entropy"] for decision in decisions]),
        },
        "normalized_entropy": {
            "mean": _mean([decision["normalized_entropy"] for decision in decisions]),
            "median": _median([decision["normalized_entropy"] for decision in decisions]),
        },
        "top_probability": {
            "mean": _mean([decision["top_probability"] for decision in decisions]),
            "median": _median([decision["top_probability"] for decision in decisions]),
        },
        "sampled_probability": {
            "mean": _mean([decision["sampled_probability"] for decision in decisions]),
            "median": _median([decision["sampled_probability"] for decision in decisions]),
        },
        "sampled_counterfactual_value": {
            "mean": _mean([decision["sampled_counterfactual_value"] for decision in decisions]),
            "median": _median([decision["sampled_counterfactual_value"] for decision in decisions]),
        },
        "baseline_counterfactual_value": {
            "mean": _mean([decision["baseline_counterfactual_value"] for decision in decisions]),
            "median": _median([decision["baseline_counterfactual_value"] for decision in decisions]),
        },
        "counterfactual_delta": {
            "mean": _mean(deltas),
            "median": _median(deltas),
            "divergent_mean": _mean(divergent_deltas),
            "divergent_median": _median(divergent_deltas),
            "positive": sum(delta > 0 for delta in deltas),
            "zero": sum(delta == 0 for delta in deltas),
            "negative": sum(delta < 0 for delta in deltas),
            "minimum": min(deltas, default=None),
            "maximum": max(deltas, default=None),
        },
        "actions": {
            "baseline_switches": sum(decision["baseline_class"] == "switch" for decision in decisions),
            "sampled_switches": sum(decision["sampled_class"] == "switch" for decision in decisions),
            "executed_switches": sum(decision["executed_class"] == "switch" for decision in decisions),
            "baseline_tera": sum(decision["baseline_class"] == "tera_move" for decision in decisions),
            "sampled_tera": sum(decision["sampled_class"] == "tera_move" for decision in decisions),
            "executed_tera": sum(decision["executed_class"] == "tera_move" for decision in decisions),
            "transitions": dict(sorted(transitions.items())),
        },
        "deterministic_corrections": {
            "count": sum(decision["selection_class"] == "deterministic_correction" for decision in decisions),
            "reasons": dict(sorted(correction_reasons.items())),
        },
    }


def analyze(results_path: Path, log_dir: Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    games = results["games"]
    games_by_tag = {game["battle_tag"]: game for game in games}
    decisions = []
    search_inputs = []
    for path in sorted(log_dir.glob("*.search.jsonl")):
        candidate_rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row.get("shared_root"), dict):
                    continue
                candidate_rows.append(row)
                context = row["context"]
                game = games_by_tag[context["tag"]]
                override = row["choice_override"]
                policy = override["mixed_strategy"]
                by_action = {entry["action"]: entry for entry in policy}
                baseline = override["baseline"]
                sampled = override["sampled_action"]
                executed = override["final_choice"]
                top = max(
                    policy,
                    key=lambda entry: (
                        float(entry["probability"]),
                        float(entry["counterfactual_value"]),
                        entry["action"],
                    ),
                )
                entropy, normalized_entropy = _entropy(policy)
                sampled_entry = by_action[sampled]
                baseline_entry = by_action[baseline]
                sampled_value = float(sampled_entry["counterfactual_value"])
                baseline_value = float(baseline_entry["counterfactual_value"])
                decisions.append(
                    {
                        "game_index": game["game_index"],
                        "pair_id": game["pair_id"],
                        "pair_index": game["pair_index"],
                        "pair_leg": game["pair_leg"],
                        "candidate_won": game["winner"] == "agent_a",
                        "source_path": str(path),
                        "source_line": line_number,
                        "tag": context["tag"],
                        "battle_turn": context["battle_turn"],
                        "decision_idx": context["decision_idx"],
                        "rqid": context["rqid"],
                        "baseline_action": baseline,
                        "sampled_action": sampled,
                        "policy_argmax_action": top["action"],
                        "executed_action": executed,
                        "baseline_class": _action_class(baseline),
                        "sampled_class": _action_class(sampled),
                        "executed_class": _action_class(executed),
                        "baseline_agreement": baseline == sampled,
                        "sampled_is_policy_argmax": sampled == top["action"],
                        "selection_class": override["selection_class"],
                        "selection_reason": override["reason"],
                        "action_count": len(policy),
                        "entropy": entropy,
                        "normalized_entropy": normalized_entropy,
                        "top_probability": float(top["probability"]),
                        "sampled_probability": float(sampled_entry["probability"]),
                        "sampled_counterfactual_value": sampled_value,
                        "baseline_counterfactual_value": baseline_value,
                        "counterfactual_delta": sampled_value - baseline_value,
                        "solver_seed": override["solver_diagnostics"]["seed"],
                        "particle_count": override["solver_diagnostics"]["input_particle_count"],
                        "canonical_particle_count": override["solver_diagnostics"]["canonical_particle_count"],
                        "payoff_cells": override["solver_diagnostics"]["payoff_cells"],
                        "exploitability": override["solver_diagnostics"]["exploitability"],
                        "nash_conv": override["solver_diagnostics"]["nash_conv"],
                        "mixed_strategy_draw": override["mixed_strategy_draw"],
                    }
                )
        if candidate_rows:
            search_inputs.append({"path": str(path), "sha256": _sha256(path), "rows": len(candidate_rows)})
    if not decisions:
        raise ValueError("no shared-root decisions found")

    decisions.sort(key=lambda row: (row["game_index"], row["decision_idx"]))
    losses = [decision for decision in decisions if not decision["candidate_won"]]
    wins = [decision for decision in decisions if decision["candidate_won"]]
    game_records = []
    for game in games:
        game_decisions = [decision for decision in decisions if decision["game_index"] == game["game_index"]]
        game_records.append(
            {
                "game_index": game["game_index"],
                "pair_id": game["pair_id"],
                "pair_index": game["pair_index"],
                "pair_leg": game["pair_leg"],
                "candidate_won": game["winner"] == "agent_a",
                "metrics": _aggregate(game_decisions),
            }
        )
    examples = {}
    if losses:
        examples = {
            "largest_estimated_improvement": max(losses, key=lambda row: row["counterfactual_delta"]),
            "largest_estimated_disadvantage": min(losses, key=lambda row: row["counterfactual_delta"]),
            "highest_entropy": max(losses, key=lambda row: row["entropy"]),
            "lowest_sampled_probability": min(losses, key=lambda row: row["sampled_probability"]),
        }
    pair_records = []
    for pair_id in sorted({game["pair_id"] for game in games}):
        pair = sorted((game for game in games if game["pair_id"] == pair_id), key=lambda game: game["pair_leg"])
        pair_records.append(
            {
                "pair_id": pair_id,
                "pair_index": pair[0]["pair_index"],
                "candidate_wins": sum(game["winner"] == "agent_a" for game in pair),
                "pair_score": sum(game["winner"] == "agent_a" for game in pair) / 2.0,
                "game_indices": [game["game_index"] for game in pair],
            }
        )
    return {
        "schema_version": 1,
        "metric_definition_version": "stage2-loss-attribution-v1",
        "inputs": {
            "results": {"path": str(results_path), "sha256": _sha256(results_path)},
            "search_logs": search_inputs,
        },
        "definitions": {
            "baseline_action": "choice_override.baseline (frozen R1 direct-policy action)",
            "sampled_action": "choice_override.sampled_action (HMAC-seeded shared-policy draw)",
            "executed_action": "choice_override.final_choice after deterministic safeguards",
            "entropy": "natural-log Shannon entropy of the complete shared policy",
            "counterfactual_delta": "logged sampled-action value minus logged baseline-action value",
        },
        "outcome": {
            "games": len(games),
            "candidate_wins": sum(game["winner"] == "agent_a" for game in games),
            "candidate_losses": sum(game["winner"] == "agent_b" for game in games),
            "pair_score_mean": results["summary"]["pair_score_mean"],
        },
        "pair_records": pair_records,
        "aggregate_metrics": {
            "all_candidate_decisions": _aggregate(decisions),
            "loss_decisions": _aggregate(losses),
            "win_decisions": _aggregate(wins),
        },
        "game_records": game_records,
        "example_decisions": examples,
        "decision_records": decisions,
        "comparison_eligibility": {
            "within_candidate_baseline_comparison": True,
            "same_root_independent_mcts_comparison": False,
            "counterfactual_winner_claim": False,
        },
        "limitations": [
            "The baseline action was not executed on the observed trajectory, so logged value deltas are not causal outcome effects.",
            "Candidate and comparator act on opposite sides within a game, and mirrored legs diverge after the first joint action.",
            "Independent-MCTS visit mass and shared-policy probability are different estimands.",
            "N=20 traces omit serialized sampled states and payoff matrices, preventing exact artifact-only root re-solving.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(args.results.expanduser().resolve(), args.log_dir.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate_metrics"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
