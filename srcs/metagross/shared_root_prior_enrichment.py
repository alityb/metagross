#!/usr/bin/env python3
"""Join captured opponent priors onto frozen shared-root evaluation worlds."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    shared_root_result_payload,
    validate_priors,
)
from srcs.metagross.shared_root_replay import _records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _identity_key(identity: object) -> str:
    if not isinstance(identity, dict):
        raise ValueError("root identity is invalid")
    return _canonical_json(identity)


def _validate_content_hash(record: dict, field: str, label: str) -> str:
    unhashed = dict(record)
    claimed = unhashed.pop(field, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unhashed):
        raise ValueError(f"{label} content hash is invalid")
    return claimed


def _condition_prior(
    prior: list[tuple[str, float]], support: list[str]
) -> tuple[list[list[object]] | None, float]:
    by_action = {action.lower(): probability for action, probability in prior}
    values = [by_action.get(action, 0.0) for action in support]
    matched_mass = math.fsum(values)
    if matched_mass <= 0:
        return None, 0.0
    return [
        [action, probability / matched_mass]
        for action, probability in zip(support, values, strict=True)
    ], matched_mass


def _validate_source_join(evaluation: dict, capture: dict) -> None:
    evaluation_hash = _validate_content_hash(
        evaluation, "evaluation_sha256", "teacher evaluation"
    )
    capture_hash = _validate_content_hash(capture, "capture_sha256", "teacher capture")
    if (
        evaluation.get("record_type") != "teacher_root_evaluation"
        or capture.get("record_type") != "teacher_root_capture"
        or evaluation.get("identity") != capture.get("identity")
        or evaluation.get("configuration", {}).get("source_capture_sha256")
        != capture_hash
    ):
        raise ValueError("teacher evaluation and source capture do not join")
    if not evaluation_hash:
        raise ValueError("teacher evaluation hash is missing")

    capture_schedules = {
        schedule.get("schedule_id"): schedule for schedule in capture.get("schedules", [])
    }
    evaluation_schedules = evaluation.get("schedules")
    if (
        not isinstance(evaluation_schedules, list)
        or len(capture_schedules) != len(capture.get("schedules", []))
        or len({schedule.get("schedule_id") for schedule in evaluation_schedules})
        != len(evaluation_schedules)
        or len(capture_schedules) != len(evaluation_schedules)
        or set(capture_schedules)
        != {schedule.get("schedule_id") for schedule in evaluation_schedules}
    ):
        raise ValueError("teacher evaluation schedule join is incomplete")
    for schedule in evaluation_schedules:
        source = capture_schedules[schedule["schedule_id"]]
        source_worlds = {world.get("world_index"): world for world in source["worlds"]}
        worlds = schedule.get("worlds")
        if (
            not isinstance(worlds, list)
            or len(source_worlds) != len(source["worlds"])
            or len({world.get("world_index") for world in worlds}) != len(worlds)
            or len(source_worlds) != len(worlds)
            or set(source_worlds) != {world.get("world_index") for world in worlds}
        ):
            raise ValueError("teacher evaluation world join is incomplete")
        for world in worlds:
            captured = source_worlds[world["world_index"]]
            if (
                world.get("sampled_state") != captured.get("sampled_state")
                or world.get("state_sha256") != captured.get("state_sha256")
                or float(world.get("sample_weight"))
                != float(captured.get("sample_weight"))
            ):
                raise ValueError("teacher evaluation world differs from source capture")


def _derive_schedule(
    worlds: list[dict], raw_prior: list[tuple[str, float]], *, seed: int
) -> tuple[list[dict], str]:
    import poke_engine

    weights = [float(world["sample_weight"]) for world in worlds]
    total = math.fsum(weights)
    normalized_weights = [weight / total for weight in weights]
    result = poke_engine.shared_information_set_root_search(
        states=[poke_engine.State.from_string(world["sampled_state"]) for world in worlds],
        particle_weights=normalized_weights,
        iterations=1,
        continuation_iterations=1,
        seed=seed,
        prior_strength=0.0,
        s1_prior=None,
        s2_priors=[raw_prior for _world in worlds],
    )
    capture = shared_root_result_payload(result)["replay_capture"]
    source_particles = {}
    for particle in capture["canonical_particles"]:
        support = particle["opponent_action_support"]
        effective = particle["normalized_opponent_prior"]
        expected, matched_mass = _condition_prior(raw_prior, support)
        expected_vector = None if expected is None else [row[1] for row in expected]
        if effective != expected_vector:
            raise ValueError("engine-conditioned opponent prior differs from reconstruction")
        for source in particle["source_particles"]:
            source_particles[source["input_index"]] = {
                "opponent_action_support": support,
                "effective_opponent_priors": expected,
                "matched_raw_prior_mass": matched_mass,
            }
    if set(source_particles) != set(range(len(worlds))):
        raise ValueError("engine replay capture has incomplete source membership")
    enriched_worlds = []
    for source_index, world in enumerate(worlds):
        derived = source_particles[source_index]
        enriched_worlds.append(
            {
                "world_index": world["world_index"],
                "state_sha256": world["state_sha256"],
                **derived,
            }
        )
    return enriched_worlds, _canonical_sha256(capture)


def validate_report(
    report: object, evaluation_path: Path, capture_path: Path
) -> dict[str, object]:
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("mode") != "source_captured_opponent_prior_enrichment"
    ):
        raise ValueError("opponent-prior enrichment has an invalid contract")
    if report.get("inputs") != {
        "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        "capture_panel": {"path": str(capture_path), "sha256": _sha256(capture_path)},
    }:
        raise ValueError("opponent-prior enrichment inputs are invalid")
    native_path = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    if report.get("engine") != {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "native_sha256": _sha256(native_path),
    }:
        raise ValueError("opponent-prior enrichment engine is invalid")

    evaluation_records = _records(evaluation_path)
    capture_records = _records(capture_path)
    evaluations = {
        _identity_key(record.get("identity")): record for record in evaluation_records
    }
    captures = {_identity_key(record.get("identity")): record for record in capture_records}
    roots = report.get("roots")
    if (
        len(evaluations) != len(evaluation_records)
        or len(captures) != len(capture_records)
        or len(evaluations) != len(captures)
        or not isinstance(roots, list)
        or len(roots) != len(evaluations)
        or len({_identity_key(root.get("identity")) for root in roots}) != len(roots)
    ):
        raise ValueError("opponent-prior enrichment root panel is incomplete")

    schedule_count = 0
    world_count = 0
    complete_world_count = 0
    complete_schedule_count = 0
    for root in roots:
        key = _identity_key(root.get("identity"))
        evaluation = evaluations.pop(key, None)
        capture = captures.pop(key, None)
        if evaluation is None or capture is None:
            raise ValueError("opponent-prior enrichment root does not join")
        _validate_source_join(evaluation, capture)
        raw_prior = validate_priors(
            root.get("source_captured_raw_opponent_priors"), "enriched raw opponent prior"
        )
        captured_prior = validate_priors(
            capture.get("recorded_opponent_priors"), "recorded opponent priors"
        )
        if (
            raw_prior is None
            or raw_prior != captured_prior
            or root.get("evaluation_sha256") != evaluation["evaluation_sha256"]
            or root.get("source_capture_sha256") != capture["capture_sha256"]
            or root.get("raw_opponent_prior_sha256")
            != _canonical_sha256([list(row) for row in raw_prior])
        ):
            raise ValueError("opponent-prior enrichment root provenance is invalid")
        source_schedules = {
            schedule["schedule_id"]: schedule for schedule in evaluation["schedules"]
        }
        schedules = root.get("schedules")
        if (
            not isinstance(schedules, list)
            or len(schedules) != len(source_schedules)
            or set(source_schedules) != {schedule.get("schedule_id") for schedule in schedules}
        ):
            raise ValueError("opponent-prior enrichment schedules are incomplete")
        for schedule in schedules:
            source_worlds = {
                world["world_index"]: world
                for world in source_schedules[schedule["schedule_id"]]["worlds"]
            }
            worlds = schedule.get("worlds")
            if (
                not isinstance(schedule.get("replay_capture_sha256"), str)
                or len(schedule["replay_capture_sha256"]) != 64
                or not isinstance(worlds, list)
                or len(worlds) != len(source_worlds)
                or set(source_worlds) != {world.get("world_index") for world in worlds}
            ):
                raise ValueError("opponent-prior enrichment worlds are incomplete")
            schedule_complete = True
            for world in worlds:
                source = source_worlds[world["world_index"]]
                support = world.get("opponent_action_support")
                if (
                    world.get("state_sha256") != source["state_sha256"]
                    or not isinstance(support, list)
                    or not support
                    or support != sorted(set(support))
                    or any(not isinstance(action, str) or action != action.lower() for action in support)
                ):
                    raise ValueError("opponent-prior enrichment world identity is invalid")
                expected, matched_mass = _condition_prior(raw_prior, support)
                if (
                    world.get("effective_opponent_priors") != expected
                    or not math.isclose(
                        float(world.get("matched_raw_prior_mass")),
                        matched_mass,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise ValueError("opponent-prior enrichment conditioning is invalid")
                available = expected is not None
                schedule_complete &= available
                complete_world_count += available
                world_count += 1
            complete_schedule_count += schedule_complete
            schedule_count += 1
    if evaluations or captures:
        raise ValueError("opponent-prior enrichment omitted source roots")
    expected_counts = {
        "roots": len(roots),
        "schedules": schedule_count,
        "world_occurrences": world_count,
        "worlds_with_effective_opponent_priors": complete_world_count,
        "worlds_without_effective_opponent_priors": world_count - complete_world_count,
        "schedules_with_complete_opponent_priors": complete_schedule_count,
    }
    if report.get("counts") != expected_counts:
        raise ValueError("opponent-prior enrichment counts are invalid")
    return report


def load_and_validate(
    report_path: Path, evaluation_path: Path, capture_path: Path
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return validate_report(report, evaluation_path, capture_path)


def run(evaluation_path: Path, capture_path: Path, *, seed: int = 8675309) -> dict:
    evaluation_records = _records(evaluation_path)
    capture_records = _records(capture_path)
    captures = {_identity_key(record.get("identity")): record for record in capture_records}
    if len(captures) != len(capture_records):
        raise ValueError("source capture identities are not unique")

    native_path = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    roots = []
    for root_index, evaluation in enumerate(evaluation_records):
        identity_key = _identity_key(evaluation.get("identity"))
        capture = captures.pop(identity_key, None)
        if capture is None:
            raise ValueError("teacher evaluation has no source capture")
        _validate_source_join(evaluation, capture)
        raw_prior = validate_priors(
            capture.get("recorded_opponent_priors"), "recorded opponent priors"
        )
        if raw_prior is None:
            raise ValueError("source capture has no opponent prior")
        schedules = []
        for schedule in evaluation["schedules"]:
            schedule_seed = seed ^ (root_index << 8) ^ int(schedule["schedule_id"])
            worlds, replay_capture_sha256 = _derive_schedule(
                schedule["worlds"], raw_prior, seed=schedule_seed
            )
            schedules.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "replay_capture_sha256": replay_capture_sha256,
                    "worlds": worlds,
                }
            )
        roots.append(
            {
                "identity": evaluation["identity"],
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "source_capture_sha256": capture["capture_sha256"],
                "source_captured_raw_opponent_priors": [list(row) for row in raw_prior],
                "raw_opponent_prior_sha256": _canonical_sha256([list(row) for row in raw_prior]),
                "schedules": schedules,
            }
        )
    if captures:
        raise ValueError("source capture panel has unmatched roots")

    worlds = [
        world
        for root in roots
        for schedule in root["schedules"]
        for world in schedule["worlds"]
    ]
    complete_worlds = sum(world["effective_opponent_priors"] is not None for world in worlds)
    report = {
        "schema_version": 1,
        "mode": "source_captured_opponent_prior_enrichment",
        "inputs": {
            "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
            "capture_panel": {"path": str(capture_path), "sha256": _sha256(capture_path)},
        },
        "engine": {
            "contract": ENGINE_CONTRACT,
            "source_sha256": ENGINE_SOURCE_SHA256,
            "native_sha256": _sha256(native_path),
        },
        "configuration": {
            "seed": seed,
            "support_derivation_iterations": 1,
            "support_derivation_continuation_iterations": 1,
            "support_derivation_prior_strength": 0.0,
        },
        "provenance": {
            "raw_prior": "source-captured root-level opponent prior",
            "effective_prior": "derived by conditioning the raw prior on each frozen world's engine-legal side-two support",
            "source_identical_effective_prior": False,
        },
        "counts": {
            "roots": len(roots),
            "schedules": sum(len(root["schedules"]) for root in roots),
            "world_occurrences": len(worlds),
            "worlds_with_effective_opponent_priors": complete_worlds,
            "worlds_without_effective_opponent_priors": len(worlds) - complete_worlds,
            "schedules_with_complete_opponent_priors": sum(
                all(world["effective_opponent_priors"] is not None for world in schedule["worlds"])
                for root in roots
                for schedule in root["schedules"]
            ),
        },
        "roots": roots,
        "limitations": [
            "The raw root prior is source-captured, but per-world conditioning is reconstructed offline.",
            "Frozen engine states do not retain the original ordered protocol or Metamon observation state.",
            "This enrichment cannot estimate win rate or authorize games.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--capture-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = run(
        args.evaluation.expanduser().resolve(),
        args.capture_panel.expanduser().resolve(),
        seed=args.seed,
    )
    validate_report(
        report,
        args.evaluation.expanduser().resolve(),
        args.capture_panel.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
