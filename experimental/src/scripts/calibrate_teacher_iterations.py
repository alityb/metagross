#!/usr/bin/env python3
"""Calibrate an exact teacher budget from deployment-duration root searches."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import pstdev
from typing import Any, Callable, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from eval.experiment_manifest import validate_manifest  # noqa: E402
from scripts.teacher_root_bundle import (  # noqa: E402
    RootBundleError,
    validate_root_bundle,
    validate_root_capture,
)


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TREATMENTS = ("accepted-live", "S-B", "U-B")
MASS_TOLERANCE = 1e-9


class CalibrationError(ValueError):
    """Raised when calibration cannot satisfy its fail-closed contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _number(value: Any, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise CalibrationError(f"{description} must be {qualifier}")
    return result


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{description} must be a positive integer")
    return value


def _pairs(
    value: Any,
    description: str,
    *,
    allow_empty: bool = False,
    normalized: bool = True,
) -> list[tuple[str, float]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise CalibrationError(f"{description} must be {qualifier}")
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 2:
            raise CalibrationError(f"{description} entries must be [action, mass] pairs")
        action, raw_mass = entry
        if not isinstance(action, str) or not action or action in seen:
            raise CalibrationError(f"{description} contains an invalid or duplicate action")
        mass = _number(raw_mass, f"{description} mass")
        if mass < 0.0:
            raise CalibrationError(f"{description} mass must be nonnegative")
        seen.add(action)
        result.append((action, mass))
    if result:
        total = math.fsum(mass for _, mass in result)
        if total <= 0.0:
            raise CalibrationError(f"{description} has no positive mass")
        if normalized and not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
            raise CalibrationError(f"{description} mass is {total!r}, expected 1")
    return result


def _equal_pairs(value: Any, description: str) -> list[tuple[str, float]]:
    result = _pairs(value, description)
    masses = [mass for _, mass in result]
    if min(masses) <= 0.0 or max(masses) - min(masses) > MASS_TOLERANCE:
        raise CalibrationError(f"{description} must assign equal positive legal-action mass")
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CalibrationError(f"non-finite manifest constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CalibrationError) as exc:
        raise CalibrationError(f"cannot read frozen input manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CalibrationError("frozen input manifest must be an object")
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        raise CalibrationError(f"invalid frozen input manifest: {exc}") from exc
    if manifest.get("manifest_type") != "experiment_input":
        raise CalibrationError("calibration requires an experiment_input manifest")
    return manifest


def _validate_bundle(
    bundle: Mapping[str, Any],
    *,
    manifest_sha256: str,
    root_index: int,
    seen_identities: set[tuple[str, str, int]],
) -> dict[str, Any]:
    if bundle.get("record_type") == "teacher_root_capture":
        try:
            validate_root_capture(bundle)
        except RootBundleError as exc:
            raise CalibrationError(str(exc)) from exc
        identity = bundle["identity"]
        identity_key = (
            identity.get("battle_tag"),
            identity.get("username"),
            identity.get("decision_idx"),
        )
        if identity_key in seen_identities:
            raise CalibrationError("duplicate root identity")
        seen_identities.add(identity_key)
        configuration = bundle["configuration"]
        if configuration.get("input_manifest_sha256") != manifest_sha256:
            raise CalibrationError("root capture does not link to the supplied frozen input manifest")
        behavior_id = bundle.get("behavior_schedule_id")
        schedule = bundle["schedules"][behavior_id]
        worlds = []
        for world in schedule["worlds"]:
            state = world["sampled_state"]
            if hashlib.sha256(state.encode()).hexdigest() != world["state_sha256"]:
                raise CalibrationError("sampled private state hash does not match root capture")
            worlds.append(
                {
                    "state": state,
                    "state_sha256": world["state_sha256"],
                    "recorded_player_priors": [
                        tuple(value) for value in bundle["recorded_player_priors"]
                    ],
                    "recorded_opponent_priors": [
                        tuple(value) for value in bundle["recorded_opponent_priors"]
                    ],
                }
            )
        return {
            "root_index": root_index,
            "c_puct": float(configuration.get("c_puct", 2.0)),
            "worlds": worlds,
        }
    try:
        validate_root_bundle(bundle)
    except RootBundleError as exc:
        raise CalibrationError(str(exc)) from exc
    if bundle.get("contains_sampled_private_states") is not True:
        raise CalibrationError("root bundle must declare sampled private states")

    identity = bundle.get("identity")
    configuration = bundle.get("configuration")
    if not isinstance(identity, dict) or not isinstance(configuration, dict):
        raise CalibrationError("root-bundle identity and configuration must be objects")
    battle_tag = identity.get("battle_tag")
    username = identity.get("username")
    decision_idx = identity.get("decision_idx")
    if not isinstance(battle_tag, str) or not battle_tag:
        raise CalibrationError("root bundle has an invalid battle identity")
    if not isinstance(username, str) or not username:
        raise CalibrationError("root bundle has an invalid player identity")
    if isinstance(decision_idx, bool) or not isinstance(decision_idx, int) or decision_idx < 0:
        raise CalibrationError("root bundle has an invalid decision index")
    identity_key = (battle_tag, username, decision_idx)
    if identity_key in seen_identities:
        raise CalibrationError("duplicate root identity")
    seen_identities.add(identity_key)

    if configuration.get("input_manifest_sha256") != manifest_sha256:
        raise CalibrationError("root bundle does not link to the supplied frozen input manifest")
    if configuration.get("threads") != 1:
        raise CalibrationError("source root bundle must declare one-thread searches")
    c_puct = _number(configuration.get("c_puct"), "root-bundle c_puct", positive=True)

    worlds = bundle.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise CalibrationError("root bundle has no worlds")
    declared_weight = _number(bundle.get("world_weight_sum"), "world weight sum", positive=True)
    weight_sum = 0.0
    validated_worlds = []
    for expected_index, world in enumerate(worlds):
        if not isinstance(world, dict) or world.get("world_index") != expected_index:
            raise CalibrationError("root-bundle world indices are invalid")
        weight = _number(world.get("sample_weight"), "world sample weight")
        if weight < 0.0:
            raise CalibrationError("world sample weight must be nonnegative")
        weight_sum += weight
        capture = world.get("capture")
        if not isinstance(capture, dict) or capture.get("world_index") != expected_index:
            raise CalibrationError("world capture index is invalid")
        state_string = capture.get("sampled_state")
        state_sha256 = capture.get("state_sha256")
        if not isinstance(state_string, str) or not state_string:
            raise CalibrationError("every world requires a sampled private state")
        if not isinstance(state_sha256, str) or not SHA256_RE.fullmatch(state_sha256):
            raise CalibrationError("world state SHA-256 is invalid")
        actual_hash = hashlib.sha256(state_string.encode("utf-8")).hexdigest()
        if actual_hash != state_sha256:
            raise CalibrationError("sampled private state hash does not match source bundle")

        recorded_player = _pairs(
            capture.get("recorded_player_priors"),
            "recorded player priors",
            normalized=False,
        )
        recorded_opponent = _pairs(
            capture.get("recorded_opponent_priors"),
            "recorded opponent priors",
            allow_empty=True,
            normalized=False,
        )
        effective_player = _pairs(capture.get("effective_player_priors"), "effective player priors")
        equal_side_one = _equal_pairs(capture.get("equal_side_one_priors"), "equal side-one priors")
        equal_side_two = _equal_pairs(capture.get("equal_side_two_priors"), "equal side-two priors")
        if {action for action, _ in effective_player} != {action for action, _ in equal_side_one}:
            raise CalibrationError("effective and equal side-one priors disagree on legal actions")
        validated_worlds.append(
            {
                "state": state_string,
                "state_sha256": state_sha256,
                "recorded_player_priors": recorded_player,
                "recorded_opponent_priors": recorded_opponent,
                "effective_player_priors": effective_player,
                "equal_side_one_priors": equal_side_one,
                "equal_side_two_priors": equal_side_two,
            }
        )
    if not math.isclose(weight_sum, declared_weight, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise CalibrationError("world weights do not match the declared sum")
    return {"root_index": root_index, "c_puct": c_puct, "worlds": validated_worlds}


def _load_roots(
    path: Path, *, manifest_sha256: str, max_records: int | None
) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CalibrationError(f"cannot read root-bundle input {path}: {exc}") from exc
    input_sha256 = hashlib.sha256(payload).hexdigest()
    roots = []
    seen_identities: set[tuple[str, str, int]] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if max_records is not None and len(roots) >= max_records:
            break
        try:
            bundle = json.loads(
                raw_line.decode("ascii"),
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    CalibrationError(f"non-finite JSON constant {constant}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError, CalibrationError) as exc:
            raise CalibrationError(f"{path}:{line_number}: invalid root-bundle JSON: {exc}") from exc
        if not isinstance(bundle, dict):
            raise CalibrationError(f"{path}:{line_number}: root-bundle record must be an object")
        try:
            roots.append(
                _validate_bundle(
                    bundle,
                    manifest_sha256=manifest_sha256,
                    root_index=len(roots),
                    seen_identities=seen_identities,
                )
            )
        except CalibrationError as exc:
            raise CalibrationError(f"{path}:{line_number}: {exc}") from exc
    if not roots:
        raise CalibrationError("root-bundle input contains no records")
    reference_c_puct = roots[0]["c_puct"]
    if any(root["c_puct"] != reference_c_puct for root in roots[1:]):
        raise CalibrationError("root-bundle input mixes c_puct configurations")
    return roots, input_sha256


def _total_visits(result: Any, description: str) -> int:
    total = getattr(result, "total_visits", None)
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise CalibrationError(f"{description} returned no positive integer visit count")
    return total


def preflight_seeded_exact_api(state_string: str) -> None:
    """Fail before calibration unless the loaded engine supports exact seeded MCTS."""
    try:
        import poke_engine

        state = poke_engine.State.from_string(state_string)
        result = poke_engine.monte_carlo_tree_search(
            state,
            duration_ms=0,
            iterations=1,
            threads=1,
            seed=0,
        )
    except Exception as exc:
        raise CalibrationError(f"seeded exact-MCTS preflight failed: {exc}") from exc
    if _total_visits(result, "seeded exact-MCTS preflight") != 1:
        raise CalibrationError("seeded exact-MCTS preflight did not complete exactly one iteration")


def _run_duration_world(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Process worker for one frozen world; intentionally passes no seed."""
    import poke_engine

    state = poke_engine.State.from_string(payload["state"])
    if "effective_player_priors" not in payload:
        side_one, side_two = poke_engine.root_options(state)
        side_one_mass = 1.0 / len(side_one)
        side_two_mass = 1.0 / len(side_two)
        payload = {
            **payload,
            "effective_player_priors": _match_actions(
                payload["recorded_player_priors"], side_one
            ),
            "equal_side_one_priors": [
                (action, side_one_mass) for action in side_one
            ],
            "equal_side_two_priors": [
                (action, side_two_mass) for action in side_two
            ],
        }
    kwargs: dict[str, Any] = {
        "duration_ms": payload["duration_ms"],
        "iterations": 0,
        "threads": 1,
        "c_puct": payload["c_puct"],
    }
    treatment = payload["treatment"]
    if treatment == "accepted-live":
        kwargs["s1_priors"] = payload["recorded_player_priors"]
        if payload["recorded_opponent_priors"]:
            kwargs["s2_priors"] = payload["recorded_opponent_priors"]
    elif treatment == "S-B":
        kwargs["s1_priors"] = payload["effective_player_priors"]
        kwargs["s2_priors"] = payload["equal_side_two_priors"]
    elif treatment == "U-B":
        kwargs["s1_priors"] = payload["equal_side_one_priors"]
        kwargs["s2_priors"] = payload["equal_side_two_priors"]
    else:
        raise CalibrationError(f"unsupported treatment: {treatment}")
    result = poke_engine.monte_carlo_tree_search(state, **kwargs)
    return {
        "state_sha256": payload["state_sha256"],
        "iterations": _total_visits(result, "duration-mode MCTS"),
    }


def _match_actions(
    priors: Sequence[tuple[str, float]], actions: Sequence[str]
) -> list[tuple[str, float]]:
    by_action = dict(priors)
    matched = [(action, float(by_action.get(action, 0.0))) for action in actions]
    total = math.fsum(mass for _, mass in matched)
    if total <= 0:
        raise CalibrationError("recorded player priors have no mass on legal actions")
    return [(action, mass / total) for action, mass in matched]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CalibrationError("cannot summarize no iteration measurements")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _round_half_up(value: float, round_to: int) -> int:
    units = (Decimal(str(value)) / Decimal(round_to)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(units * round_to)


def summarize_iterations(values: Sequence[float], *, round_to: int) -> dict[str, float | int]:
    """Summarize equally weighted root/repeat mean iteration counts."""
    _positive_integer(round_to, "round-to increment")
    if not values:
        raise CalibrationError("cannot summarize no iteration measurements")
    checked = [_number(value, "iteration measurement", positive=True) for value in values]
    mean = math.fsum(checked) / len(checked)
    median = _percentile(checked, 0.5)
    return {
        "mean": mean,
        "median": median,
        "p10": _percentile(checked, 0.1),
        "p25": _percentile(checked, 0.25),
        "p75": _percentile(checked, 0.75),
        "p90": _percentile(checked, 0.9),
        "min": min(checked),
        "max": max(checked),
        "cv": pstdev(checked) / mean,
        "recommendation": _round_half_up(median, round_to),
        "recommendation_statistic": "median",
        "round_to": round_to,
    }


def _world_payload(world: Mapping[str, Any], root: Mapping[str, Any], treatment: str, duration_ms: int) -> dict[str, Any]:
    return {
        **world,
        "c_puct": root["c_puct"],
        "treatment": treatment,
        "duration_ms": duration_ms,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("calibration_schema_version") != SCHEMA_VERSION:
        raise CalibrationError("invalid calibration report schema")
    if report.get("record_type") != "teacher_iteration_calibration":
        raise CalibrationError("invalid calibration report type")
    unhashed = dict(report)
    claimed = unhashed.pop("calibration_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
    if claimed != actual:
        raise CalibrationError("calibration report hash does not match content")
    serialized = _canonical_json(report)
    for forbidden_key in ('"sampled_state"', '"identity"', '"battle_tag"', '"username"'):
        if forbidden_key in serialized:
            raise CalibrationError("calibration report contains a private identity or sampled state")


def calibrate(
    input_path: Path,
    input_manifest_path: Path,
    *,
    duration_ms: int,
    parallelism: int = 8,
    repeats: int = 3,
    treatment: str = "accepted-live",
    max_records: int | None = None,
    round_to: int = 1000,
    executor_factory: Callable[..., Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    duration_ms = _positive_integer(duration_ms, "duration-ms")
    parallelism = _positive_integer(parallelism, "parallelism")
    repeats = _positive_integer(repeats, "repeats")
    round_to = _positive_integer(round_to, "round-to increment")
    if max_records is not None:
        _positive_integer(max_records, "max-records")
    if treatment not in TREATMENTS:
        raise CalibrationError(f"treatment must be one of {', '.join(TREATMENTS)}")

    manifest = _load_manifest(input_manifest_path)
    manifest_sha256 = str(manifest["manifest_sha256"])
    roots, input_sha256 = _load_roots(
        input_path,
        manifest_sha256=manifest_sha256,
        max_records=max_records,
    )
    timer = time.perf_counter if clock is None else clock
    executor_type = ProcessPoolExecutor if executor_factory is None else executor_factory

    total_started = timer()
    preflight_seeded_exact_api(roots[0]["worlds"][0]["state"])
    measurements = []
    root_repeat_means = []
    total_waves = 0
    total_world_searches = 0
    for root in roots:
        for repeat in range(repeats):
            payloads = [
                _world_payload(world, root, treatment, duration_ms)
                for world in root["worlds"]
            ]
            started = timer()
            with executor_type(max_workers=parallelism) as executor:
                world_results = list(executor.map(_run_duration_world, payloads))
            wall_seconds = timer() - started
            if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
                raise CalibrationError("wall clock returned an invalid duration")
            iterations = [result["iterations"] for result in world_results]
            mean_iterations = math.fsum(iterations) / len(iterations)
            waves = math.ceil(len(world_results) / parallelism)
            root_repeat_means.append(mean_iterations)
            total_waves += waves
            total_world_searches += len(world_results)
            measurements.append(
                {
                    "root_index": root["root_index"],
                    "repeat": repeat,
                    "world_count": len(world_results),
                    "executor_waves": waves,
                    "wall_seconds": wall_seconds,
                    "mean_iterations": mean_iterations,
                    "worlds": world_results,
                }
            )
    total_wall_seconds = timer() - total_started
    if not math.isfinite(total_wall_seconds) or total_wall_seconds < 0.0:
        raise CalibrationError("wall clock returned an invalid total duration")

    report: dict[str, Any] = {
        "calibration_schema_version": SCHEMA_VERSION,
        "record_type": "teacher_iteration_calibration",
        "calibration_mode": "duration_to_exact_iterations",
        "claim_status": "hardware_specific_descriptive_calibration",
        "inputs": {
            "root_bundle_sha256": input_sha256,
            "frozen_input_manifest_sha256": manifest_sha256,
        },
        "privacy": {
            "sampled_private_states_read": True,
            "sampled_private_states_emitted": False,
            "battle_or_player_identities_emitted": False,
            "state_sha256_emitted": True,
        },
        "configuration": {
            "duration_ms": duration_ms,
            "parallelism": parallelism,
            "repeats": repeats,
            "treatment": treatment,
            "threads": 1,
            "seed": None,
            "c_puct": roots[0]["c_puct"],
            "round_to": round_to,
        },
        "counts": {
            "roots": len(roots),
            "root_repeats": len(measurements),
            "executor_instances": len(measurements),
            "world_searches": total_world_searches,
            "executor_waves": total_waves,
        },
        "timing": {
            "total_wall_seconds": total_wall_seconds,
            "executor_wall_seconds_sum": math.fsum(
                measurement["wall_seconds"] for measurement in measurements
            ),
        },
        "metric_definitions": {
            "observation": "arithmetic mean completed iterations across worlds in one root/repeat",
            "root_balance": "every root has the same repeat count and therefore equal weight in all statistics",
            "cv": "population standard deviation divided by the root-balanced mean",
            "recommendation": "root-balanced median rounded to round_to with ties rounded upward",
            "executor_wave": "ceil(world_count / parallelism) for one root/repeat executor",
        },
        "statistics": summarize_iterations(root_repeat_means, round_to=round_to),
        "measurements": measurements,
    }
    report["calibration_sha256"] = hashlib.sha256(
        _canonical_json(report).encode("ascii")
    ).hexdigest()
    validate_report(report)
    return report


def _write_private_atomic(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise CalibrationError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path: Path | None = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise CalibrationError(f"output already exists: {path}") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_report(report: Mapping[str, Any], output_path: Path, *, force: bool = False) -> None:
    validate_report(report)
    payload = (_canonical_json(dict(report)) + "\n").encode("ascii")
    _write_private_atomic(output_path, payload, force=force)


def calibrate_file(
    input_path: Path,
    input_manifest_path: Path,
    output_path: Path,
    *,
    duration_ms: int,
    parallelism: int = 8,
    repeats: int = 3,
    treatment: str = "accepted-live",
    max_records: int | None = None,
    round_to: int = 1000,
    force: bool = False,
    executor_factory: Callable[..., Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    output_resolved = output_path.resolve()
    if output_resolved in {input_path.resolve(), input_manifest_path.resolve()}:
        raise CalibrationError("output path must differ from both input paths")
    if not force and os.path.lexists(output_path):
        raise CalibrationError(f"output already exists: {output_path}")
    report = calibrate(
        input_path,
        input_manifest_path,
        duration_ms=duration_ms,
        parallelism=parallelism,
        repeats=repeats,
        treatment=treatment,
        max_records=max_records,
        round_to=round_to,
        executor_factory=executor_factory,
        clock=clock,
    )
    write_report(report, output_path, force=force)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-ms", required=True, type=int)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--treatment", choices=TREATMENTS, default="accepted-live")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--round-to", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = calibrate_file(
        args.input,
        args.input_manifest,
        args.output,
        duration_ms=args.duration_ms,
        parallelism=args.parallelism,
        repeats=args.repeats,
        treatment=args.treatment,
        max_records=args.max_records,
        round_to=args.round_to,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "calibration_sha256": report["calibration_sha256"],
                "roots": report["counts"]["roots"],
                "recommended_iterations": report["statistics"]["recommendation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
