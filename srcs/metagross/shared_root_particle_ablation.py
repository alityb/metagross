#!/usr/bin/env python3
"""Measure shared-root policy sensitivity to production-sized particle cohorts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from srcs.metagross import shared_root_offline_ablation as metrics
from srcs.metagross import shared_root_replay


ITERATIONS = 10_000
CONTINUATION_ITERATIONS = 8
SEED = 8675309
PRIOR_STRENGTH = 1.0
COHORT_SIZES = (2, 4)


def _solve(poke_engine, states, player_prior, opponent_prior) -> dict:
    started = time.perf_counter()
    count = len(states)
    result = poke_engine.shared_information_set_root_search(
        states=states,
        particle_weights=[1.0 / count] * count,
        iterations=ITERATIONS,
        continuation_iterations=CONTINUATION_ITERATIONS,
        seed=SEED,
        prior_strength=PRIOR_STRENGTH,
        s1_prior=player_prior,
        s2_priors=[opponent_prior for _state in states],
    )
    policy = metrics._policy(result)
    return {
        "policy": policy,
        "selected_action": metrics._argmax(policy),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "diagnostics": asdict(result.diagnostics),
    }


def _average_policy(rows: list[dict]) -> dict[str, float]:
    actions = set().union(*(row["policy"] for row in rows))
    return {
        action: math.fsum(row["policy"].get(action, 0.0) for row in rows) / len(rows)
        for action in actions
    }


def run(input_path: Path) -> dict:
    import poke_engine

    roots = []
    for record in shared_root_replay._records(input_path):
        schedule = shared_root_replay._schedule(record)
        states = [
            poke_engine.State.from_string(world["sampled_state"])
            for world in schedule["worlds"]
        ]
        player_prior, opponent_prior = shared_root_replay._priors(record, schedule)
        teacher = metrics._teacher_policy(schedule)
        full = _solve(poke_engine, states, player_prior, opponent_prior)
        full["teacher_alignment"] = metrics._teacher_alignment(full["policy"], teacher)
        full["teacher_argmax_agreement"] = full["selected_action"] == metrics._argmax(teacher)
        cohorts = {}
        for cohort_size in COHORT_SIZES:
            rows = []
            for start in range(0, len(states), cohort_size):
                cohort = states[start : start + cohort_size]
                if len(cohort) != cohort_size:
                    raise ValueError("particle count is not divisible by cohort size")
                row = _solve(poke_engine, cohort, player_prior, opponent_prior)
                row["cohort_index"] = start // cohort_size
                row["tv_from_full"] = metrics._tv(row["policy"], full["policy"])
                row["argmax_changed_from_full"] = row["selected_action"] != full["selected_action"]
                row["teacher_alignment"] = metrics._teacher_alignment(row["policy"], teacher)
                row["teacher_argmax_agreement"] = row["selected_action"] == metrics._argmax(teacher)
                rows.append(row)
            ensemble = _average_policy(rows)
            cohorts[str(cohort_size)] = {
                "rows": rows,
                "ensemble_policy": ensemble,
                "ensemble_tv_from_full": metrics._tv(ensemble, full["policy"]),
                "ensemble_selected_action": metrics._argmax(ensemble),
                "ensemble_teacher_alignment": metrics._teacher_alignment(ensemble, teacher),
            }
        roots.append(
            {
                "identity": record["identity"],
                "particle_count": len(states),
                "teacher_argmax": metrics._argmax(teacher),
                "full": full,
                "cohorts": cohorts,
            }
        )

    summaries = {}
    for cohort_size in COHORT_SIZES:
        key = str(cohort_size)
        root_mean_tvs = []
        root_p95_tvs = []
        root_argmax_mismatch_fractions = []
        root_teacher_alignments = []
        root_teacher_agreement_fractions = []
        ensemble_tvs = []
        ensemble_alignment_deltas = []
        latencies = []
        for root in roots:
            rows = root["cohorts"][key]["rows"]
            tvs = [row["tv_from_full"] for row in rows]
            root_mean_tvs.append(math.fsum(tvs) / len(tvs))
            root_p95_tvs.append(metrics._percentile(tvs, 0.95))
            root_argmax_mismatch_fractions.append(
                sum(row["argmax_changed_from_full"] for row in rows) / len(rows)
            )
            root_teacher_alignments.append(
                math.fsum(row["teacher_alignment"] for row in rows) / len(rows)
            )
            root_teacher_agreement_fractions.append(
                sum(row["teacher_argmax_agreement"] for row in rows) / len(rows)
            )
            ensemble_tvs.append(root["cohorts"][key]["ensemble_tv_from_full"])
            ensemble_alignment_deltas.append(
                root["cohorts"][key]["ensemble_teacher_alignment"]
                - root["full"]["teacher_alignment"]
            )
            latencies.extend(row["latency_ms"] for row in rows)
        summaries[key] = {
            "roots": len(roots),
            "cohorts": sum(len(root["cohorts"][key]["rows"]) for root in roots),
            "root_mean_tv_from_full": metrics._summary(root_mean_tvs),
            "root_p95_tv_from_full": metrics._summary(root_p95_tvs),
            "root_argmax_mismatch_fraction": metrics._summary(root_argmax_mismatch_fractions),
            "root_mean_teacher_alignment": metrics._summary(root_teacher_alignments),
            "root_teacher_agreement_fraction": metrics._summary(root_teacher_agreement_fractions),
            "ensemble_tv_from_full": metrics._summary(ensemble_tvs),
            "ensemble_teacher_alignment_delta": metrics._summary(ensemble_alignment_deltas),
            "cohort_latency_ms": metrics._summary(latencies),
        }
    diagnostic = {
        "two_particle_instability_detected": summaries["2"]["root_p95_tv_from_full"]["mean"] > 0.25
        or summaries["2"]["root_argmax_mismatch_fraction"]["mean"] > 0.10,
        "four_particle_instability_detected": summaries["4"]["root_p95_tv_from_full"]["mean"] > 0.15
        or summaries["4"]["root_argmax_mismatch_fraction"]["mean"] > 0.05,
    }
    return {
        "schema_version": 1,
        "mode": "stage2_shared_root_particle_cohort_ablation",
        "input": {"path": str(input_path), "sha256": metrics._sha256(input_path)},
        "configuration": {
            "iterations": ITERATIONS,
            "continuation_iterations": CONTINUATION_ITERATIONS,
            "seed": SEED,
            "prior_strength": PRIOR_STRENGTH,
            "cohort_sizes": list(COHORT_SIZES),
            "partition": "consecutive_nonoverlapping_capture_order",
        },
        "summaries": summaries,
        "diagnostic": diagnostic,
        "roots": roots,
        "limitations": [
            "Capture weights are uniform, so this panel cannot identify a particle-weighting effect.",
            "Consecutive cohorts measure sensitivity for one immutable partition, not the full subset distribution.",
            "Teacher alignment is a stronger-search proxy, not game-outcome ground truth.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = run(args.input.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostic"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
