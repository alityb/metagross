#!/usr/bin/env python3
"""Replay deterministic teacher treatments offline on frozen sampled roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from scripts.teacher_root_bundle import (  # noqa: E402
    RootBundleConfig,
    RootBundleError,
    _canonical_json,
    append_root_bundle,
    build_root_bundle,
    derive_scheduled_tree_seed,
    equal_priors,
    match_priors_to_options,
    run_world_treatments,
    snapshot_result,
    validate_root_capture,
    validate_root_bundle,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVALUATION_SCHEMA_VERSION = 2


def _snapshot_result(snapshot: Mapping[str, Any]) -> SimpleNamespace:
    def side(name: str) -> list[SimpleNamespace]:
        entries = snapshot.get(name)
        if not isinstance(entries, list) or not entries:
            raise RootBundleError(f"source live_result.{name} is invalid")
        return [
            SimpleNamespace(
                move_choice=str(entry["action"]),
                visits=int(entry["visits"]),
                total_score=float(entry["total_score"]),
            )
            for entry in entries
        ]

    total_visits = int(snapshot.get("total_visits", 0))
    if total_visits <= 0:
        raise RootBundleError("source live_result has no visits")
    return SimpleNamespace(
        side_one=side("side_one"),
        side_two=side("side_two"),
        total_visits=total_visits,
    )


def _evaluate_world(payload: Mapping[str, Any]) -> tuple[SimpleNamespace, float, int]:
    source_world = payload["world"]
    source_capture = source_world["capture"]
    state_string = source_capture.get("sampled_state")
    if not isinstance(state_string, str) or not state_string:
        raise RootBundleError("offline evaluation requires every sampled private state")
    state_hash = hashlib.sha256(state_string.encode("utf-8")).hexdigest()
    if state_hash != source_capture.get("state_sha256"):
        raise RootBundleError("sampled state hash does not match source bundle")

    config = RootBundleConfig(
        output_path=Path("unused"),
        iterations=int(payload["iterations"]),
        repeats=int(payload["repeats"]),
        deep_multiplier=int(payload["deep_multiplier"]),
        base_seed=int(payload["base_seed"]),
        c_puct=float(payload["c_puct"]),
        include_state=True,
        manifest_sha256=str(payload["manifest_sha256"]),
    )
    live_result = _snapshot_result(source_capture["live_result"])
    world_index = int(source_world["world_index"])
    capture = run_world_treatments(
        state_string=state_string,
        live_result=live_result,
        world_index=world_index,
        identity=payload["identity"],
        player_priors=[tuple(value) for value in source_capture["recorded_player_priors"]],
        opponent_priors=[tuple(value) for value in source_capture.get("recorded_opponent_priors", [])],
        config=config,
    )
    live_result._teacher_root_capture = {
        "identity": dict(payload["identity"]),
        "configuration": {
            "iterations": config.iterations,
            "repeats": config.repeats,
            "deep_multiplier": config.deep_multiplier,
            "base_seed": config.base_seed,
            "c_puct": config.c_puct,
            "input_manifest_sha256": config.manifest_sha256,
            "source_bundle_sha256": payload["source_bundle_sha256"],
            "source_input_manifest_sha256": payload["source_input_manifest_sha256"],
            "primary_side_two_treatment": "equal_legal_priors",
            "threads": 1,
            "execution": "offline",
        },
        "world_index": world_index,
        "world": capture,
    }
    weight = float(source_world["sample_weight"])
    if not math.isfinite(weight) or weight < 0:
        raise RootBundleError("source world weight is invalid")
    return live_result, weight, world_index


def evaluate_bundle(
    source_bundle: Mapping[str, Any],
    *,
    iterations: int,
    repeats: int,
    deep_multiplier: int,
    base_seed: int,
    c_puct: float,
    manifest_sha256: str,
    workers: int = 1,
) -> dict[str, Any]:
    validate_root_bundle(source_bundle)
    if iterations <= 0 or repeats <= 0 or deep_multiplier < 1 or workers <= 0:
        raise RootBundleError("iterations, repeats, deep multiplier, and workers must be positive")
    if not 0 <= base_seed < 2**64:
        raise RootBundleError("base seed must fit an unsigned 64-bit integer")
    if not math.isfinite(c_puct) or c_puct <= 0:
        raise RootBundleError("c_puct must be finite and positive")
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise RootBundleError("manifest SHA-256 must be 64 lowercase hexadecimal characters")

    source_configuration = source_bundle.get("configuration")
    if not isinstance(source_configuration, dict):
        raise RootBundleError("source bundle configuration is invalid")
    source_manifest = source_configuration.get("input_manifest_sha256")
    if not isinstance(source_manifest, str) or not SHA256_RE.fullmatch(source_manifest):
        raise RootBundleError("source bundle has no valid input-manifest linkage")
    common = {
        "identity": source_bundle["identity"],
        "iterations": iterations,
        "repeats": repeats,
        "deep_multiplier": deep_multiplier,
        "base_seed": base_seed,
        "c_puct": c_puct,
        "manifest_sha256": manifest_sha256,
        "source_bundle_sha256": source_bundle["bundle_sha256"],
        "source_input_manifest_sha256": source_manifest,
    }
    payloads = [{**common, "world": world} for world in source_bundle["worlds"]]
    if workers == 1:
        results = [_evaluate_world(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_evaluate_world, payloads))
    return build_root_bundle(results)


def _evaluate_capture_world(payload: Mapping[str, Any]) -> dict[str, Any]:
    import poke_engine

    world = payload["world"]
    state_string = world.get("sampled_state")
    state_sha256 = hashlib.sha256(state_string.encode("utf-8")).hexdigest()
    if state_sha256 != world.get("state_sha256"):
        raise RootBundleError("sampled state hash does not match root capture")
    state = poke_engine.State.from_string(state_string)
    side_one_actions, side_two_actions = poke_engine.root_options(state)
    side_one_options = [SimpleNamespace(move_choice=action) for action in side_one_actions]
    side_two_options = [SimpleNamespace(move_choice=action) for action in side_two_actions]
    side_one_equal = equal_priors(side_one_options)
    side_two_equal = equal_priors(side_two_options)
    player_priors = [tuple(value) for value in payload["recorded_player_priors"]]
    effective_player = match_priors_to_options(player_priors, side_one_options)
    specs = [
        ("U-B", payload["iterations"], side_one_equal, side_two_equal),
        ("S-B", payload["iterations"], effective_player, side_two_equal),
    ]
    if payload["deep_multiplier"] > 1:
        specs.append(
            (
                f"S-{payload['deep_multiplier']}B",
                payload["iterations"] * payload["deep_multiplier"],
                effective_player,
                side_two_equal,
            )
        )
    treatments = {}
    for treatment, budget, side_one_priors, side_two_priors in specs:
        treatment_repeats = []
        for repeat in range(payload["repeats"]):
            seed = derive_scheduled_tree_seed(
                payload["base_seed"],
                payload["identity"],
                payload["schedule_id"],
                state_sha256,
                world["world_index"],
                treatment,
                repeat,
            )
            result = poke_engine.monte_carlo_tree_search(
                poke_engine.State.from_string(state_string),
                duration_ms=0,
                iterations=budget,
                threads=1,
                s1_priors=side_one_priors,
                s2_priors=side_two_priors,
                c_puct=payload["c_puct"],
                seed=seed,
            )
            treatment_repeats.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "iterations": budget,
                    "result": snapshot_result(result),
                }
            )
        treatments[treatment] = treatment_repeats
    return {
        "world_index": world["world_index"],
        "sample_weight": world["sample_weight"],
        "state_sha256": state_sha256,
        "sampled_state": state_string,
        "effective_player_priors": [[action, mass] for action, mass in effective_player],
        "equal_side_one_priors": [[action, mass] for action, mass in side_one_equal],
        "equal_side_two_priors": [[action, mass] for action, mass in side_two_equal],
        "treatments": treatments,
    }


def _aggregate_schedule_policy(
    worlds: Sequence[Mapping[str, Any]], treatment: str, repeat: int
) -> list[dict[str, Any]]:
    masses: dict[str, float] = {}
    for world in worlds:
        weight = float(world["sample_weight"])
        result = world["treatments"][treatment][repeat]["result"]
        total = result["total_visits"]
        for option in result["side_one"]:
            masses[option["action"]] = masses.get(option["action"], 0.0) + (
                weight * option["visits"] / total
            )
    total_mass = math.fsum(masses.values())
    if total_mass <= 0:
        raise RootBundleError("schedule aggregate has no policy mass")
    return [
        {"action": action, "probability": mass / total_mass}
        for action, mass in sorted(masses.items())
    ]


def validate_root_evaluation(evaluation: Mapping[str, Any]) -> None:
    if (
        evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or evaluation.get("record_type") != "teacher_root_evaluation"
    ):
        raise RootBundleError("invalid teacher root-evaluation record")
    unhashed = dict(evaluation)
    claimed = unhashed.pop("evaluation_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
    if claimed != actual:
        raise RootBundleError("teacher root-evaluation hash does not match content")
    schedules = evaluation.get("schedules")
    if not isinstance(schedules, list) or not schedules:
        raise RootBundleError("teacher root evaluation has no schedules")
    if [schedule.get("schedule_id") for schedule in schedules] != list(range(len(schedules))):
        raise RootBundleError("teacher root-evaluation schedule IDs are invalid")


def evaluate_capture(
    source_capture: Mapping[str, Any],
    *,
    iterations: int,
    repeats: int,
    deep_multiplier: int,
    base_seed: int,
    c_puct: float,
    manifest_sha256: str,
    workers: int = 1,
) -> dict[str, Any]:
    validate_root_capture(source_capture)
    if iterations <= 0 or repeats <= 0 or deep_multiplier < 1 or workers <= 0:
        raise RootBundleError("iterations, repeats, deep multiplier, and workers must be positive")
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise RootBundleError("manifest SHA-256 must be 64 lowercase hexadecimal characters")
    common = {
        "identity": source_capture["identity"],
        "recorded_player_priors": source_capture["recorded_player_priors"],
        "iterations": iterations,
        "repeats": repeats,
        "deep_multiplier": deep_multiplier,
        "base_seed": base_seed,
        "c_puct": c_puct,
    }
    evaluated_schedules = []
    for source_schedule in source_capture["schedules"]:
        payloads = [
            {
                **common,
                "schedule_id": source_schedule["schedule_id"],
                "world": world,
            }
            for world in source_schedule["worlds"]
        ]
        if workers == 1:
            worlds = [_evaluate_capture_world(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                worlds = list(executor.map(_evaluate_capture_world, payloads))
        treatment_names = sorted(worlds[0]["treatments"])
        aggregate = {
            treatment: [
                {
                    "repeat": repeat,
                    "side_one_policy": _aggregate_schedule_policy(
                        worlds, treatment, repeat
                    ),
                }
                for repeat in range(repeats)
            ]
            for treatment in treatment_names
        }
        evaluated_schedules.append(
            {
                "schedule_id": source_schedule["schedule_id"],
                "sampling_seed": source_schedule["sampling_seed"],
                "world_count": len(worlds),
                "world_weight_sum": math.fsum(
                    float(world["sample_weight"]) for world in worlds
                ),
                "worlds": worlds,
                "aggregate_treatments": aggregate,
            }
        )
    evaluation = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "record_type": "teacher_root_evaluation",
        "identity": dict(source_capture["identity"]),
        "configuration": {
            "iterations": iterations,
            "repeats": repeats,
            "deep_multiplier": deep_multiplier,
            "base_seed": base_seed,
            "c_puct": c_puct,
            "threads": 1,
            "execution": "offline",
            "input_manifest_sha256": manifest_sha256,
            "source_capture_sha256": source_capture["capture_sha256"],
            "source_input_manifest_sha256": source_capture["configuration"]["input_manifest_sha256"],
            "primary_side_two_treatment": "equal_legal_priors",
        },
        "schedules": evaluated_schedules,
    }
    if "sampling" in source_capture:
        evaluation["sampling"] = dict(source_capture["sampling"])
    evaluation["evaluation_sha256"] = hashlib.sha256(
        _canonical_json(evaluation).encode("ascii")
    ).hexdigest()
    validate_root_evaluation(evaluation)
    return evaluation


def _append_evaluation(path: Path, evaluation: Mapping[str, Any]) -> None:
    validate_root_evaluation(evaluation)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (_canonical_json(dict(evaluation)) + "\n").encode("ascii")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("teacher root-evaluation append made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_records(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        RootBundleError(f"non-finite JSON value {value}")
                    ),
                )
            except (json.JSONDecodeError, RootBundleError) as exc:
                raise RootBundleError(f"invalid source record at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise RootBundleError(f"source record at line {line_number} is not an object")
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise RootBundleError("source root-bundle file is empty")
    return records


def evaluate_file(
    input_path: Path,
    output_path: Path,
    *,
    iterations: int,
    repeats: int,
    deep_multiplier: int,
    base_seed: int,
    c_puct: float,
    manifest_sha256: str,
    workers: int = 1,
    max_records: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise RootBundleError("offline output must differ from the source bundle")
    if max_records is not None and max_records <= 0:
        raise RootBundleError("max records must be positive")
    if not force and os.path.lexists(output_path):
        raise RootBundleError(f"offline output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    record_count = 0
    world_count = 0
    try:
        for source in _load_records(input_path, max_records):
            if source.get("record_type") == "teacher_root_capture":
                evaluated = evaluate_capture(
                    source,
                    iterations=iterations,
                    repeats=repeats,
                    deep_multiplier=deep_multiplier,
                    base_seed=base_seed,
                    c_puct=c_puct,
                    manifest_sha256=manifest_sha256,
                    workers=workers,
                )
                _append_evaluation(temporary_path, evaluated)
                world_count += sum(
                    int(schedule["world_count"]) for schedule in evaluated["schedules"]
                )
            else:
                evaluated = evaluate_bundle(
                    source,
                    iterations=iterations,
                    repeats=repeats,
                    deep_multiplier=deep_multiplier,
                    base_seed=base_seed,
                    c_puct=c_puct,
                    manifest_sha256=manifest_sha256,
                    workers=workers,
                )
                append_root_bundle(temporary_path, evaluated)
                world_count += int(evaluated["world_count"])
            record_count += 1
        if force:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise RootBundleError(f"offline output already exists: {output_path}") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_path": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "records": record_count,
        "worlds": world_count,
        "iterations": iterations,
        "repeats": repeats,
        "deep_multiplier": deep_multiplier,
        "base_seed": base_seed,
        "c_puct": c_puct,
        "workers": workers,
        "manifest_sha256": manifest_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay deterministic teacher treatments on private frozen roots."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--deep-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_file(
        args.input,
        args.output,
        iterations=args.iterations,
        repeats=args.repeats,
        deep_multiplier=args.deep_multiplier,
        base_seed=args.seed,
        c_puct=args.cpuct,
        manifest_sha256=args.manifest_sha256,
        workers=args.workers,
        max_records=args.max_records,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
