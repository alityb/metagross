#!/usr/bin/env python3
"""Audit the complete shared-root proposal paired-verifier sweep."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from srcs.metagross import run_foul_play
from srcs.metagross.shared_root_paired_verifier_experiment import (
    _independent_search_action,
    _runtime_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_key(identity: dict) -> str:
    return json.dumps(identity, separators=(",", ":"), sort_keys=True)


def _teacher_mass(schedule: dict, action: str) -> float:
    repeats = schedule["aggregate_treatments"]["S-4B"]
    return math.fsum(
        next(
            (float(row["probability"]) for row in repeat["side_one_policy"] if row["action"] == action),
            0.0,
        )
        for repeat in repeats
    ) / len(repeats)


def audit(protocol_path: Path, evaluation_path: Path, report_paths: list[Path]) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_hashes = {_sha256(path) for path in report_paths}
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("runner", {}).get("sha256") != _sha256(Path(__file__).resolve())
        or protocol.get("evaluation_sha256") != _sha256(evaluation_path)
        or set(protocol.get("report_sha256s", [])) != report_hashes
        or len(report_hashes) != len(report_paths)
    ):
        raise ValueError("paired-verifier sweep differs from its frozen audit protocol")
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    evaluation_roots = {_identity_key(root["identity"]): root for root in evaluations}
    strategies = {}
    check_failures = Counter()
    recomputed_certificates = 0
    horizon_one_passes = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report_protocol_path = Path(report["protocol"]["path"])
        if (
            report.get("mode") != "shared_root_proposal_paired_verifier_experiment"
            or report["protocol"]["sha256"] != _sha256(report_protocol_path)
            or report["inputs"]["evaluation"]["sha256"] != _sha256(evaluation_path)
            or report["counts"]["roots"] != 26
            or report["counts"]["schedules"] != 104
            or report["gate"]["passed"] is not False
        ):
            raise ValueError("paired-verifier report has an invalid frozen contract")
        report_protocol = json.loads(report_protocol_path.read_text(encoding="utf-8"))
        if (
            report_protocol["runner"]["sha256"]
            != _sha256(Path(report_protocol["runner"]["path"]))
            or report_protocol.get("runtime") != _runtime_identity()
            or report.get("runtime") != _runtime_identity()
        ):
            raise ValueError("paired-verifier report runner differs from its protocol")
        strategy = report["configuration"]["candidate"]
        if strategy in strategies:
            raise ValueError("paired-verifier sweep contains a duplicate strategy")
        report_roots = {_identity_key(root["identity"]): root for root in report["roots"]}
        if len(report_roots) != len(report["roots"]) or report_roots.keys() != evaluation_roots.keys():
            raise ValueError("paired-verifier report root coverage is invalid")
        matrix_path = Path(report["inputs"]["matrix_diagnostics"]["path"])
        if report["inputs"]["matrix_diagnostics"]["sha256"] != _sha256(matrix_path):
            raise ValueError("paired-verifier matrix input hash is invalid")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix_roots = {_identity_key(root["identity"]): root for root in matrix["roots"]}
        matrix_root_indices = {
            _identity_key(root["identity"]): index for index, root in enumerate(matrix["roots"])
        }
        proposals = []
        candidate_differences = 0
        overlap_pairs = 0
        overlap_occurrences = 0
        qualified_schedules = 0
        severe_regressions = 0
        root_deltas = []
        holdout_latencies = []
        for root in report["roots"]:
            root_key = _identity_key(root["identity"])
            evaluation = evaluation_roots[root_key]
            schedules = {
                schedule["schedule_id"]: schedule for schedule in evaluation["schedules"]
            }
            matrix_schedules = {
                schedule["schedule_id"]: schedule
                for schedule in matrix_roots[root_key]["schedules"]
            }
            if {row["selection_schedule_id"] for row in root["schedules"]} != {0, 1, 2, 3}:
                raise ValueError("paired-verifier schedule coverage is invalid")
            schedule_deltas = []
            for schedule in root["schedules"]:
                selection_id = schedule["selection_schedule_id"]
                certification_id = schedule["certification_schedule_id"]
                if certification_id != (selection_id + 1) % 4:
                    raise ValueError("paired-verifier certification schedule is invalid")
                source = schedules[selection_id]
                certification = schedules[certification_id]
                baseline = _independent_search_action(source)
                candidate = matrix_schedules[selection_id]["strategy_aggregates"]["strategies"][
                    strategy
                ]["selected_action"]
                if schedule["baseline_action"] != baseline or schedule["candidate_action"] != candidate:
                    raise ValueError("paired-verifier actions differ from frozen inputs")
                candidate_differs = candidate is not None and candidate != baseline
                if schedule["candidate_differs"] != candidate_differs:
                    raise ValueError("paired-verifier candidate-difference flag is invalid")
                candidate_differences += candidate_differs
                expected_overlap = sorted(
                    {world["state_sha256"] for world in source["worlds"]}
                    & {world["state_sha256"] for world in certification["worlds"]}
                )
                if schedule["overlapping_state_hashes"] != expected_overlap:
                    raise ValueError("paired-verifier schedule overlap is invalid")
                overlap_pairs += bool(expected_overlap)
                overlap_occurrences += len(expected_overlap)
                combined = schedule["combined_certificate"]
                if combined is None:
                    if candidate_differs:
                        raise ValueError("paired-verifier omitted a differing proposal")
                    baseline_mass = _teacher_mass(source, baseline)
                    if (
                        schedule["qualified"] is not False
                        or schedule["final_action"] != baseline
                        or schedule["baseline_teacher_mass"] != baseline_mass
                        or schedule["final_teacher_mass"] != baseline_mass
                        or schedule["teacher_mass_delta"] != 0.0
                    ):
                        raise ValueError("paired-verifier same-action outcome is invalid")
                    schedule_deltas.append(0.0)
                    holdout_latencies.append(float(schedule["holdout_ms"]))
                    continue
                expected_weights = [float(world["sample_weight"]) for world in certification["worlds"]]
                expected_hashes = [world["state_sha256"] for world in certification["worlds"]]
                certificates = {}
                for horizon_text, certificate in combined["certificates"].items():
                    if (
                        certificate["world_weights"] != expected_weights
                        or certificate["state_hashes"] != expected_hashes
                        or certificate["cluster_hashes"] != expected_hashes
                        or certificate["baseline"] != baseline
                        or certificate["candidate"] != candidate
                        or certificate["alpha_sequence_index"]
                        != matrix_root_indices[root_key] * 4 + selection_id
                    ):
                        raise ValueError("paired-verifier certificate provenance is invalid")
                    recomputed = run_foul_play.recompute_robust_holdout_certificate(certificate)
                    if recomputed != certificate:
                        raise ValueError("paired-verifier certificate does not recompute exactly")
                    certificates[int(horizon_text)] = certificate
                    recomputed_certificates += 1
                    for name, passed in certificate["checks"].items():
                        if not passed:
                            check_failures[name] += 1
                if run_foul_play.combined_robust_holdout_certificate(certificates) != combined:
                    raise ValueError("combined paired-verifier certificate is invalid")
                if schedule["qualified"] != combined["qualified"]:
                    raise ValueError("paired-verifier schedule qualification is invalid")
                qualified_schedules += schedule["qualified"]
                candidate_delta = _teacher_mass(source, schedule["candidate_action"]) - _teacher_mass(
                    source, schedule["baseline_action"]
                )
                proposal = {
                    "identity": root["identity"],
                    "selection_schedule_id": schedule["selection_schedule_id"],
                    "candidate_teacher_mass_delta": candidate_delta,
                    "executed_horizons": combined["executed_horizons"],
                    "qualified": combined["qualified"],
                }
                proposals.append(proposal)
                first = certificates[run_foul_play.HOLDOUT_CONTINUATION_HORIZONS[0]]
                if first["qualified"]:
                    horizon_one_passes.append({"strategy": strategy, **proposal})
                expected_final = candidate if schedule["qualified"] else baseline
                baseline_mass = _teacher_mass(source, baseline)
                final_mass = _teacher_mass(source, expected_final)
                if (
                    schedule["final_action"] != expected_final
                    or schedule["baseline_teacher_mass"] != baseline_mass
                    or schedule["final_teacher_mass"] != final_mass
                    or schedule["teacher_mass_delta"] != final_mass - baseline_mass
                ):
                    raise ValueError("paired-verifier schedule outcome is invalid")
                schedule_deltas.append(schedule["teacher_mass_delta"])
                holdout_latencies.append(float(schedule["holdout_ms"]))
            if len(schedule_deltas) != 4:
                raise ValueError("paired-verifier root schedule outcomes are incomplete")
            root_delta = math.fsum(schedule_deltas) / len(schedule_deltas)
            if root["teacher_mass_delta"] != root_delta or root["severe_regression"] != (
                root_delta <= -0.20
            ):
                raise ValueError("paired-verifier root outcome is invalid")
            root_deltas.append(
                (root_delta, float(root["poststratification_weight"]))
            )
            severe_regressions += root["severe_regression"]
        total_weight = math.fsum(weight for _delta, weight in root_deltas)
        expected_metrics = {
            "poststratified_teacher_mass_delta": math.fsum(
                delta * weight for delta, weight in root_deltas
            )
            / total_weight,
            "mean_teacher_mass_delta": math.fsum(delta for delta, _weight in root_deltas)
            / len(root_deltas),
            "minimum_root_delta": min(delta for delta, _weight in root_deltas),
            "maximum_root_delta": max(delta for delta, _weight in root_deltas),
            "holdout_ms_total": math.fsum(holdout_latencies),
        }
        if report["metrics"] != expected_metrics:
            raise ValueError("paired-verifier report metrics do not recompute")
        expected_counts = {
            "roots": len(report["roots"]),
            "schedules": sum(len(root["schedules"]) for root in report["roots"]),
            "candidate_differences": candidate_differences,
            "qualified_schedules": qualified_schedules,
            "severe_regressions": severe_regressions,
            "schedule_pairs_with_state_overlap": overlap_pairs,
            "overlapping_unique_state_occurrences": overlap_occurrences,
        }
        if report["counts"] != expected_counts:
            raise ValueError("paired-verifier report counts do not recompute")
        strategies[strategy] = {
            "report": {"path": str(path), "sha256": _sha256(path)},
            "proposals": len(proposals),
            "positive_teacher_proposals": sum(
                row["candidate_teacher_mass_delta"] > 0 for row in proposals
            ),
            "negative_teacher_proposals": sum(
                row["candidate_teacher_mass_delta"] < 0 for row in proposals
            ),
            "horizon_one_qualified": sum(len(row["executed_horizons"]) == 2 for row in proposals),
            "all_horizons_qualified": sum(row["qualified"] for row in proposals),
            "candidate_teacher_mass_delta": {
                "mean": math.fsum(row["candidate_teacher_mass_delta"] for row in proposals)
                / len(proposals),
                "minimum": min(row["candidate_teacher_mass_delta"] for row in proposals),
                "maximum": max(row["candidate_teacher_mass_delta"] for row in proposals),
            },
        }
    expected_strategies = set(protocol["strategies"])
    if set(strategies) != expected_strategies:
        raise ValueError("paired-verifier sweep strategy coverage is incomplete")
    return {
        "schema_version": 1,
        "mode": "paired_verifier_strategy_sweep_audit",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        "counts": {
            "strategies": len(strategies),
            "proposals": sum(row["proposals"] for row in strategies.values()),
            "recomputed_certificates": recomputed_certificates,
            "horizon_one_qualified": len(horizon_one_passes),
            "all_horizons_qualified": sum(
                row["all_horizons_qualified"] for row in strategies.values()
            ),
        },
        "strategies": strategies,
        "failed_check_counts": dict(sorted(check_failures.items())),
        "horizon_one_passes": horizon_one_passes,
        "decision": "no_shared_root_proposal_strategy_supported",
        "authorization": {
            "implementation_authorized": False,
            "local_games_authorized": False,
            "public_ladder_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(
        args.protocol.expanduser().resolve(),
        args.evaluation.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.report],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "decision": result["decision"]}, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
