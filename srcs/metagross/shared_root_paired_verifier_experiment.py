#!/usr/bin/env python3
"""Test shared-root proposals behind the production paired holdout verifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace

from srcs.metagross import run_foul_play
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    holdout_result_payload,
)


BASE_SEED = 2026081001
SEVERE_DELTA = -0.20
CANDIDATE_STRATEGIES = (
    "rm_policy_argmax",
    "opponent_prior_expected",
    "worst_case_endpoint",
    "bounded_robust_0.10",
    "bounded_robust_0.25",
    "bounded_robust_0.50",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_key(identity: dict) -> str:
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)


def _seed(identity: dict, selection_schedule: int, world_index: int) -> int:
    payload = json.dumps(
        [BASE_SEED, identity, selection_schedule, world_index],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _runtime_identity() -> dict:
    import poke_engine

    native = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    root = Path(__file__).resolve().parent
    return {
        "python_version": platform.python_version(),
        "engine_contract": ENGINE_CONTRACT,
        "engine_source_sha256": ENGINE_SOURCE_SHA256,
        "engine_native_sha256": _sha256(native),
        "dependencies": {
            "run_foul_play.py": _sha256(root / "run_foul_play.py"),
            "mcts_contract.py": _sha256(root / "mcts_contract.py"),
            "holdout_metrics.py": _sha256(root / "holdout_metrics.py"),
        },
        "distribution_version": getattr(poke_engine, "__version__", "0.0.47"),
    }


def _validate_protocol(
    protocol_path: Path,
    matrix_path: Path,
    evaluation_path: Path,
    enrichment_path: Path,
    candidate_strategy: str,
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("inputs")
        != {
            "evaluation_sha256": _sha256(evaluation_path),
            "matrix_diagnostics_sha256": _sha256(matrix_path),
            "prior_enrichment_sha256": _sha256(enrichment_path),
        }
        or protocol.get("runtime") != _runtime_identity()
        or protocol.get("configuration")
        != {
            "base_seed": BASE_SEED,
            "baseline": "independent_s_b_search_argmax",
            "candidate": candidate_strategy,
            "certification_schedule": "next_schedule_modulo_four",
            "holdout_rollouts": run_foul_play.HOLDOUT_ROLLOUTS,
            "holdout_continuation_iterations": run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS,
            "holdout_continuation_horizons": list(
                run_foul_play.HOLDOUT_CONTINUATION_HORIZONS
            ),
            "severe_regression_delta": SEVERE_DELTA,
        }
    ):
        raise ValueError("paired-verifier experiment differs from its frozen protocol")


def _snapshot_result(snapshot: dict) -> SimpleNamespace:
    def side(name: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                move_choice=row["action"],
                total_score=float(row["total_score"]),
                visits=int(row["visits"]),
            )
            for row in snapshot[name]
        ]

    return SimpleNamespace(
        side_one=side("side_one"),
        side_two=side("side_two"),
        total_visits=int(snapshot["total_visits"]),
    )


def _independent_search_action(schedule: dict) -> str:
    weighted = [
        (
            _snapshot_result(world["treatments"]["S-B"][0]["result"]),
            float(world["sample_weight"]),
            int(world["world_index"]),
        )
        for world in schedule["worlds"]
    ]
    ordered, _mass = run_foul_play._mcts_actions_and_visit_mass(weighted)
    return ordered[0]


def _teacher_mass(schedule: dict, action: str) -> float:
    repeats = schedule["aggregate_treatments"]["S-4B"]
    return math.fsum(
        next(
            (
                float(row["probability"])
                for row in repeat["side_one_policy"]
                if row["action"] == action
            ),
            0.0,
        )
        for repeat in repeats
    ) / len(repeats)


def analyze(
    matrix_path: Path,
    evaluation_path: Path,
    enrichment_path: Path,
    protocol_path: Path,
    candidate_strategy: str,
) -> dict[str, object]:
    if candidate_strategy not in CANDIDATE_STRATEGIES:
        raise ValueError("unsupported paired-verifier candidate strategy")
    _validate_protocol(
        protocol_path,
        matrix_path,
        evaluation_path,
        enrichment_path,
        candidate_strategy,
    )
    import poke_engine

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    matrix_roots = {_identity_key(root["identity"]): root for root in matrix["roots"]}
    evaluation_roots = {_identity_key(root["identity"]): root for root in evaluations}
    enriched_roots = {_identity_key(root["identity"]): root for root in enrichment["roots"]}
    if not matrix_roots.keys() == evaluation_roots.keys() == enriched_roots.keys():
        raise ValueError("paired-verifier experiment root join is incomplete")

    roots = []
    for root_index, (key, matrix_root) in enumerate(matrix_roots.items()):
        evaluation = evaluation_roots[key]
        enriched = enriched_roots[key]
        evaluation_schedules = {
            int(schedule["schedule_id"]): schedule for schedule in evaluation["schedules"]
        }
        matrix_schedules = {
            int(schedule["schedule_id"]): schedule for schedule in matrix_root["schedules"]
        }
        raw_opponent_prior = [
            tuple(row) for row in enriched["source_captured_raw_opponent_priors"]
        ]
        schedule_rows = []
        for selection_schedule_id in sorted(matrix_schedules):
            selection_schedule = evaluation_schedules[selection_schedule_id]
            matrix_schedule = matrix_schedules[selection_schedule_id]
            certification_schedule_id = (selection_schedule_id + 1) % len(matrix_schedules)
            certification_schedule = evaluation_schedules[certification_schedule_id]
            selection_hashes = {
                world["state_sha256"] for world in selection_schedule["worlds"]
            }
            certification_hashes = {
                world["state_sha256"] for world in certification_schedule["worlds"]
            }
            overlapping_state_hashes = sorted(selection_hashes & certification_hashes)
            baseline = _independent_search_action(selection_schedule)
            candidate = matrix_schedule["strategy_aggregates"]["strategies"][
                candidate_strategy
            ]["selected_action"]
            certificates = {}
            qualified = False
            elapsed_ms = 0.0
            if candidate is not None and candidate != baseline:
                worlds = certification_schedule["worlds"]
                weights = [float(world["sample_weight"]) for world in worlds]
                state_hashes = [world["state_sha256"] for world in worlds]
                seeds = [
                    _seed(
                        matrix_root["identity"],
                        selection_schedule_id,
                        int(world["world_index"]),
                    )
                    for world in worlds
                ]
                for horizon_index, continuation_steps in enumerate(
                    run_foul_play.HOLDOUT_CONTINUATION_HORIZONS
                ):
                    started = time.perf_counter_ns()
                    results = [
                        holdout_result_payload(
                            poke_engine.paired_root_policy_evaluation(
                                poke_engine.State.from_string(world["sampled_state"]),
                                baseline_action=baseline,
                                candidate_action=candidate,
                                rollouts=run_foul_play.HOLDOUT_ROLLOUTS,
                                continuation_iterations=run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS,
                                continuation_steps=continuation_steps,
                                seed=seed,
                                opponent_priors=raw_opponent_prior,
                            ),
                            expected_pairs=run_foul_play.HOLDOUT_ROLLOUTS,
                            maximum_executed=(
                                2
                                * run_foul_play.HOLDOUT_ROLLOUTS
                                * run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS
                                * continuation_steps
                            ),
                        )
                        for world, seed in zip(worlds, seeds, strict=True)
                    ]
                    elapsed_ms += (time.perf_counter_ns() - started) / 1_000_000
                    certificate = run_foul_play.robust_holdout_certificate(
                        results,
                        weights,
                        state_hashes,
                        state_hashes,
                        candidate,
                        baseline,
                        root_index * 4 + selection_schedule_id,
                        1,
                        horizon_index,
                    )
                    certificates[continuation_steps] = certificate
                    if not certificate["qualified"]:
                        break
                combined = run_foul_play.combined_robust_holdout_certificate(certificates)
                qualified = bool(combined["qualified"])
            else:
                combined = None
            final_action = candidate if qualified else baseline
            baseline_mass = _teacher_mass(selection_schedule, baseline)
            final_mass = _teacher_mass(selection_schedule, final_action)
            schedule_rows.append(
                {
                    "selection_schedule_id": selection_schedule_id,
                    "certification_schedule_id": certification_schedule_id,
                    "overlapping_unique_states": len(overlapping_state_hashes),
                    "overlapping_state_hashes": overlapping_state_hashes,
                    "baseline_action": baseline,
                    "candidate_action": candidate,
                    "candidate_differs": candidate is not None and candidate != baseline,
                    "qualified": qualified,
                    "final_action": final_action,
                    "baseline_teacher_mass": baseline_mass,
                    "final_teacher_mass": final_mass,
                    "teacher_mass_delta": final_mass - baseline_mass,
                    "holdout_ms": elapsed_ms,
                    "combined_certificate": combined,
                }
            )
        root_delta = math.fsum(row["teacher_mass_delta"] for row in schedule_rows) / len(
            schedule_rows
        )
        roots.append(
            {
                "identity": matrix_root["identity"],
                "poststratification_weight": float(
                    matrix_root["sampling"]["poststratification_weight"]
                ),
                "teacher_mass_delta": root_delta,
                "severe_regression": root_delta <= SEVERE_DELTA,
                "schedules": schedule_rows,
            }
        )
    total_weight = math.fsum(root["poststratification_weight"] for root in roots)
    poststratified_delta = math.fsum(
        root["teacher_mass_delta"] * root["poststratification_weight"] for root in roots
    ) / total_weight
    qualified_schedules = sum(
        schedule["qualified"] for root in roots for schedule in root["schedules"]
    )
    return {
        "schema_version": 1,
        "mode": "shared_root_proposal_paired_verifier_experiment",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "runtime": _runtime_identity(),
        "inputs": {
            "matrix_diagnostics": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
            "prior_enrichment": {
                "path": str(enrichment_path),
                "sha256": _sha256(enrichment_path),
            },
        },
        "configuration": {
            "base_seed": BASE_SEED,
            "baseline": "independent_s_b_search_argmax",
            "candidate": candidate_strategy,
            "certification_schedule": "next_schedule_modulo_four",
            "holdout_rollouts": run_foul_play.HOLDOUT_ROLLOUTS,
            "holdout_continuation_iterations": run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS,
            "holdout_continuation_horizons": list(
                run_foul_play.HOLDOUT_CONTINUATION_HORIZONS
            ),
            "severe_regression_delta": SEVERE_DELTA,
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
            "candidate_differences": sum(
                schedule["candidate_differs"]
                for root in roots
                for schedule in root["schedules"]
            ),
            "qualified_schedules": qualified_schedules,
            "severe_regressions": sum(root["severe_regression"] for root in roots),
            "schedule_pairs_with_state_overlap": sum(
                schedule["overlapping_unique_states"] > 0
                for root in roots
                for schedule in root["schedules"]
            ),
            "overlapping_unique_state_occurrences": sum(
                schedule["overlapping_unique_states"]
                for root in roots
                for schedule in root["schedules"]
            ),
        },
        "metrics": {
            "poststratified_teacher_mass_delta": poststratified_delta,
            "mean_teacher_mass_delta": math.fsum(root["teacher_mass_delta"] for root in roots)
            / len(roots),
            "minimum_root_delta": min(root["teacher_mass_delta"] for root in roots),
            "maximum_root_delta": max(root["teacher_mass_delta"] for root in roots),
            "holdout_ms_total": math.fsum(
                schedule["holdout_ms"] for root in roots for schedule in root["schedules"]
            ),
        },
        "gate": {
            "passed": poststratified_delta > 0
            and qualified_schedules > 0
            and not any(root["severe_regression"] for root in roots),
            "conditions": {
                "positive_poststratified_teacher_mass_delta": poststratified_delta > 0,
                "at_least_one_qualified_override": qualified_schedules > 0,
                "zero_severe_root_regressions": not any(
                    root["severe_regression"] for root in roots
                ),
            },
            "new_candidate_supported": False,
            "new_games_authorized": False,
        },
        "roots": roots,
        "limitations": [
            "Selection and certification use distinct schedule IDs, but their sampled state sets can overlap and are quantified per decision.",
            "The S-4B teacher is a development proxy, not game-outcome ground truth.",
            "A passing result still requires implementation, source-bound local games, and independent review before any ladder canary.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prior-enrichment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-strategy", choices=CANDIDATE_STRATEGIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = analyze(
        args.matrix.expanduser().resolve(),
        args.evaluation.expanduser().resolve(),
        args.prior_enrichment.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
        args.candidate_strategy,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "metrics": report["metrics"], "gate": report["gate"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
