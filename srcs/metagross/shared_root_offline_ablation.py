#!/usr/bin/env python3
"""Run preregistered shared-root ablations on immutable teacher roots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from srcs.metagross import shared_root_replay


@dataclass(frozen=True)
class Configuration:
    name: str
    iterations: int
    continuation_iterations: int
    seed: int
    prior_strength: float


SEEDS = (8675309, 8675310, 8675311)
CONFIGURATIONS = tuple(
    Configuration(f"continuation-{continuation}-seed-{seed}", 10_000, continuation, seed, 1.0)
    for continuation in (8, 32, 128)
    for seed in SEEDS
) + (
    Configuration("rm-1000-continuation-128", 1_000, 128, SEEDS[0], 1.0),
    Configuration("rm-20000-continuation-128", 20_000, 128, SEEDS[0], 1.0),
    Configuration("prior-0-continuation-128", 10_000, 128, SEEDS[0], 0.0),
    Configuration("prior-10-continuation-128", 10_000, 128, SEEDS[0], 10.0),
)
BASELINE = "continuation-8-seed-8675309"
REVISED = "continuation-128-seed-8675309"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(result) -> dict[str, float]:
    return {entry.action: float(entry.probability) for entry in result.policy}


def _argmax(policy: dict[str, float]) -> str:
    return max(policy, key=lambda action: (policy[action], action))


def _tv(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * math.fsum(
        abs(left.get(action, 0.0) - right.get(action, 0.0))
        for action in left.keys() | right.keys()
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values, default=None),
    }


def _teacher_policy(schedule: dict) -> dict[str, float]:
    repeats = schedule.get("aggregate_treatments", {}).get("S-4B")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError("root evaluation lacks S-4B teacher treatments")
    totals: dict[str, float] = {}
    for repeat in repeats:
        for entry in repeat["side_one_policy"]:
            action = str(entry["action"]).lower()
            totals[action] = totals.get(action, 0.0) + float(entry["probability"])
    total = math.fsum(totals.values())
    return {action: value / total for action, value in totals.items()}


def _teacher_alignment(policy: dict[str, float], teacher: dict[str, float]) -> float:
    return math.fsum(policy.get(action, 0.0) * probability for action, probability in teacher.items())


def _seed_stability(root_rows: list[dict], continuation: int) -> dict:
    names = [f"continuation-{continuation}-seed-{seed}" for seed in SEEDS]
    tvs = []
    disagreements = 0
    comparisons = 0
    for row in root_rows:
        by_name = {result["configuration"]: result for result in row["results"]}
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left = by_name[left_name]
                right = by_name[right_name]
                tvs.append(_tv(left["policy"], right["policy"]))
                disagreements += left["selected_action"] != right["selected_action"]
                comparisons += 1
    return {
        "continuation_iterations": continuation,
        "pairwise_policy_tv": _summary(tvs),
        "argmax_disagreements": disagreements,
        "argmax_comparisons": comparisons,
        "argmax_disagreement_fraction": disagreements / comparisons if comparisons else None,
    }


def run(input_path: Path) -> dict:
    import poke_engine

    roots = []
    for record in shared_root_replay._records(input_path):
        schedule = shared_root_replay._schedule(record)
        worlds = schedule["worlds"]
        states = [poke_engine.State.from_string(world["sampled_state"]) for world in worlds]
        weights = [float(world["sample_weight"]) for world in worlds]
        total_weight = math.fsum(weights)
        weights = [weight / total_weight for weight in weights]
        player_prior, opponent_prior = shared_root_replay._priors(record, schedule)
        opponent_priors = [opponent_prior for _state in states]
        teacher = _teacher_policy(schedule)
        teacher_argmax = _argmax(teacher)
        result_rows = []
        for configuration in CONFIGURATIONS:
            started = time.perf_counter()
            result = poke_engine.shared_information_set_root_search(
                states=states,
                particle_weights=weights,
                iterations=configuration.iterations,
                continuation_iterations=configuration.continuation_iterations,
                seed=configuration.seed,
                prior_strength=configuration.prior_strength,
                s1_prior=player_prior,
                s2_priors=opponent_priors,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            policy = _policy(result)
            selected = _argmax(policy)
            probabilities = list(policy.values())
            if not probabilities or not all(math.isfinite(value) and value >= 0 for value in probabilities):
                raise ValueError("solver returned malformed probabilities")
            if abs(math.fsum(probabilities) - 1.0) > 1e-8:
                raise ValueError("solver returned unnormalized policy")
            result_rows.append(
                {
                    "configuration": configuration.name,
                    "policy": policy,
                    "selected_action": selected,
                    "teacher_alignment": _teacher_alignment(policy, teacher),
                    "teacher_argmax_agreement": selected == teacher_argmax,
                    "latency_ms": elapsed_ms,
                    "diagnostics": asdict(result.diagnostics),
                }
            )
        by_name = {row["configuration"]: row for row in result_rows}
        baseline = by_name[BASELINE]
        for row in result_rows:
            row["tv_from_baseline"] = _tv(row["policy"], baseline["policy"])
            row["argmax_changed_from_baseline"] = row["selected_action"] != baseline["selected_action"]
        roots.append(
            {
                "identity": record["identity"],
                "capture_sha256": record.get("evaluation_sha256") or record.get("capture_sha256"),
                "particle_count": len(states),
                "teacher_policy": teacher,
                "teacher_argmax": teacher_argmax,
                "results": result_rows,
            }
        )

    config_summaries = []
    for configuration in CONFIGURATIONS:
        rows = [
            next(result for result in root["results"] if result["configuration"] == configuration.name)
            for root in roots
        ]
        config_summaries.append(
            {
                "configuration": configuration.name,
                "parameters": asdict(configuration),
                "latency_ms": _summary([row["latency_ms"] for row in rows]),
                "tv_from_baseline": _summary([row["tv_from_baseline"] for row in rows]),
                "argmax_changes_from_baseline": sum(row["argmax_changed_from_baseline"] for row in rows),
                "teacher_alignment": _summary([row["teacher_alignment"] for row in rows]),
                "teacher_argmax_agreements": sum(row["teacher_argmax_agreement"] for row in rows),
                "mean_exploitability": math.fsum(row["diagnostics"]["exploitability"] for row in rows) / len(rows),
            }
        )
    summaries_by_name = {summary["configuration"]: summary for summary in config_summaries}
    baseline = summaries_by_name[BASELINE]
    revised = summaries_by_name[REVISED]
    seed_stability = [_seed_stability(roots, continuation) for continuation in (8, 32, 128)]
    stability_by_budget = {row["continuation_iterations"]: row for row in seed_stability}
    gate = {
        "material_policy_change": revised["tv_from_baseline"]["mean"] >= 0.05
        or revised["argmax_changes_from_baseline"] >= 3,
        "teacher_alignment_improved": revised["teacher_alignment"]["mean"]
        > baseline["teacher_alignment"]["mean"],
        "teacher_argmax_agreement_noninferior": revised["teacher_argmax_agreements"]
        >= baseline["teacher_argmax_agreements"],
        "payoff_seed_stability_improved": stability_by_budget[128]["pairwise_policy_tv"]["p95"]
        < stability_by_budget[8]["pairwise_policy_tv"]["p95"],
        "latency_below_safety_limit": revised["latency_ms"]["max"] < 10_000,
    }
    gate["offline_evidence_predicts_improvement"] = all(gate.values())
    return {
        "schema_version": 1,
        "mode": "stage2_shared_root_offline_ablation",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "baseline_configuration": BASELINE,
        "proposed_revised_configuration": REVISED,
        "configurations": [asdict(configuration) for configuration in CONFIGURATIONS],
        "configuration_summaries": config_summaries,
        "seed_stability": seed_stability,
        "offline_prediction_gate": gate,
        "roots": roots,
        "limitations": [
            "S-4B teacher policy is a stronger-search proxy, not a game-outcome ground truth.",
            "The 26-root panel is not the completed N=20 trajectory and cannot estimate a revised win rate.",
            "Continuation-seed sensitivity measures payoff-game stability but does not isolate individual payoff-cell error.",
            "Opponent-model and particle-weight re-solving require payoff-matrix capture not present in current artifacts.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = run(args.input.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["offline_prediction_gate"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
