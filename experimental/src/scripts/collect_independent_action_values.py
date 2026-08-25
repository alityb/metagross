#!/usr/bin/env python3
"""Collect development-only action values under uniform-legal continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from eval.experiment_manifest import validate_manifest  # noqa: E402
from scripts.teacher_root_bundle import (  # noqa: E402
    RootBundleError,
    validate_root_capture,
)


SCHEMA_VERSION = 1
POLICY_ID = "uniform_legal_v1"
TAPE_ID = "sha256_counter_common_tape_v1"
TAPE_CHANNELS = ("side_one_action", "side_two_action", "chance")


class IndependentActionValueError(ValueError):
    """Raised when the independent-value estimand cannot be collected exactly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IndependentActionValueError(f"{description} must be a positive integer")
    return value


def _base_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise IndependentActionValueError("seed must fit an unsigned 64-bit integer")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def counter_uniform(
    *,
    base_seed: int,
    source_capture_identity: Mapping[str, Any],
    source_capture_sha256: str,
    schedule_id: int,
    world_index: int,
    rollout: int,
    decision: int,
    channel: str,
) -> float:
    """Map a candidate-independent SHA-256 counter to an exact 53-bit uniform."""
    _base_seed(base_seed)
    if channel not in TAPE_CHANNELS:
        raise IndependentActionValueError(f"invalid common-tape channel: {channel}")
    material = {
        "base_seed": base_seed,
        "source_capture_identity": dict(source_capture_identity),
        "source_capture_sha256": source_capture_sha256,
        "schedule_id": schedule_id,
        "world_index": world_index,
        "rollout": rollout,
        "decision": decision,
        "channel": channel,
    }
    digest = hashlib.sha256(_canonical_json(material).encode("ascii")).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def _options(engine: Any, state: Any) -> tuple[list[str], list[str]]:
    try:
        raw_side_one, raw_side_two = engine.root_options(state)
    except Exception as exc:
        raise IndependentActionValueError(f"cannot enumerate legal actions: {exc}") from exc
    sides: list[list[str]] = []
    for side_name, raw in (("side one", raw_side_one), ("side two", raw_side_two)):
        if not isinstance(raw, (list, tuple)) or not raw:
            raise IndependentActionValueError(f"{side_name} has no legal actions")
        actions = list(raw)
        if any(not isinstance(action, str) or not action for action in actions):
            raise IndependentActionValueError(f"{side_name} legal actions are invalid")
        if len(set(actions)) != len(actions):
            raise IndependentActionValueError(f"{side_name} legal actions contain duplicates")
        sides.append(actions)
    return sides[0], sides[1]


def _uniform_action(actions: Sequence[str], uniform: float) -> str:
    return actions[int(uniform * len(actions))]


def _side_alive(state: Any, side_name: str) -> bool:
    side = getattr(state, side_name, None)
    pokemon = getattr(side, "pokemon", None)
    if pokemon is None:
        raise IndependentActionValueError(
            f"state {side_name} does not expose Pokemon HP for terminal detection"
        )
    try:
        members = list(pokemon)
    except TypeError as exc:
        raise IndependentActionValueError(
            f"state {side_name} Pokemon collection is invalid"
        ) from exc
    for member in members:
        hp = getattr(member, "hp", None)
        if isinstance(hp, bool) or not isinstance(hp, (int, float)):
            raise IndependentActionValueError(
                f"state {side_name} Pokemon does not expose numeric HP"
            )
        if not math.isfinite(float(hp)):
            raise IndependentActionValueError(f"state {side_name} Pokemon HP is non-finite")
        if hp > 0:
            return True
    return False


def _is_terminal(state: Any) -> bool:
    side_one_alive = _side_alive(state, "side_one")
    side_two_alive = _side_alive(state, "side_two")
    return not side_one_alive or not side_two_alive


def _terminal_score(engine: Any, state: Any) -> float:
    try:
        value = float(engine.terminal_value(state))
    except Exception as exc:
        raise IndependentActionValueError(f"cannot evaluate terminal state: {exc}") from exc
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise IndependentActionValueError("terminal_value must be finite and in [-1, 1]")
    return (value + 1.0) / 2.0


def _rollout(
    engine: Any,
    state_string: str,
    candidate_action: str,
    *,
    tape: Mapping[str, Any],
    max_decisions: int,
) -> float:
    try:
        state = engine.State.from_string(state_string)
    except Exception as exc:
        raise IndependentActionValueError(f"cannot parse sampled root state: {exc}") from exc
    if _is_terminal(state):
        raise IndependentActionValueError("sampled root state is already terminal")

    for decision in range(max_decisions):
        side_one, side_two = _options(engine, state)
        if decision == 0:
            if candidate_action not in side_one:
                raise IndependentActionValueError(
                    f"candidate action is not legal in sampled world: {candidate_action}"
                )
            side_one_action = candidate_action
        else:
            side_one_uniform = counter_uniform(
                **tape, decision=decision, channel="side_one_action"
            )
            side_one_action = _uniform_action(side_one, side_one_uniform)
        side_two_uniform = counter_uniform(
            **tape, decision=decision, channel="side_two_action"
        )
        chance_uniform = counter_uniform(**tape, decision=decision, channel="chance")
        side_two_action = _uniform_action(side_two, side_two_uniform)
        try:
            stepped = engine.step_with_uniform(
                state, side_one_action, side_two_action, chance_uniform
            )
            state = stepped[0]
        except Exception as exc:
            raise IndependentActionValueError(
                f"cannot step rollout at decision {decision}: {exc}"
            ) from exc
        if _is_terminal(state):
            return _terminal_score(engine, state)

    raise IndependentActionValueError(
        f"rollout reached max decisions ({max_decisions}) while nonterminal"
    )


def _common_tape_definition() -> dict[str, Any]:
    return {
        "id": TAPE_ID,
        "hash": "sha256",
        "counter_encoding": "canonical_json_ascii",
        "counter_fields": [
            "base_seed",
            "source_capture_identity",
            "source_capture_sha256",
            "schedule_id",
            "world_index",
            "rollout",
            "decision",
            "channel",
        ],
        "channels": list(TAPE_CHANNELS),
        "uniform_mapping": "first_53_sha256_bits_divided_by_2^53",
        "candidate_player_action_in_counter": False,
        "root_side_one_action": "candidate",
        "root_side_two_action": POLICY_ID,
        "continuation_actions": POLICY_ID,
        "chance": "step_with_uniform",
    }


def validate_action_value_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("record_type") != "independent_action_values"
    ):
        raise IndependentActionValueError("invalid independent action-value record")
    unhashed = dict(record)
    claimed = unhashed.pop("record_sha256", None)
    actual = _sha256(_canonical_json(unhashed).encode("ascii"))
    if claimed != actual:
        raise IndependentActionValueError("independent action-value hash does not match content")
    if record.get("oracle") is not False or record.get("r1_continuation_value") is not False:
        raise IndependentActionValueError("independent action values must be non-oracle and non-r1")
    if record.get("opponent_policy_id") != POLICY_ID:
        raise IndependentActionValueError("invalid opponent policy ID")
    if record.get("continuation_policy_id") != POLICY_ID:
        raise IndependentActionValueError("invalid continuation policy ID")
    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        raise IndependentActionValueError("independent action-value record has no actions")
    seen: set[str] = set()
    for result in actions:
        if not isinstance(result, dict):
            raise IndependentActionValueError("action result must be an object")
        action = result.get("action")
        estimate = result.get("q")
        count = result.get("sample_count")
        if not isinstance(action, str) or not action or action in seen:
            raise IndependentActionValueError("action results contain invalid or duplicate actions")
        if (
            isinstance(estimate, bool)
            or not isinstance(estimate, (int, float))
            or not math.isfinite(float(estimate))
            or not 0.0 <= estimate <= 1.0
        ):
            raise IndependentActionValueError("action Q estimate must be in [0, 1]")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise IndependentActionValueError("action sample count must be positive")
        seen.add(action)
    serialized = _canonical_json(record)
    if "sampled_state" in serialized:
        raise IndependentActionValueError("output must not contain sampled states")


def collect_capture(
    capture: Mapping[str, Any],
    *,
    rollouts: int,
    max_decisions: int,
    base_seed: int,
    input_sha256: str,
    manifest_sha256: str,
    manifest_file_sha256: str,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Evaluate every root action with paired common-tape rollouts."""
    rollout_count = _positive_integer(rollouts, "rollouts")
    horizon = _positive_integer(max_decisions, "max decisions")
    seed = _base_seed(base_seed)
    try:
        validate_root_capture(capture)
    except (RootBundleError, TypeError, ValueError, OverflowError) as exc:
        raise IndependentActionValueError(f"invalid source root capture: {exc}") from exc
    configuration = capture.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("input_manifest_sha256") != manifest_sha256
    ):
        raise IndependentActionValueError(
            "source capture does not link to the supplied frozen input manifest"
        )
    if engine is None:
        import poke_engine as engine

    identity = capture["identity"]
    capture_sha256 = capture["capture_sha256"]
    expected_actions: list[str] | None = None
    action_schedule_values: dict[str, list[float]] = {}
    action_sample_counts: dict[str, int] = {}
    total_worlds = 0

    for schedule in capture["schedules"]:
        schedule_id = int(schedule["schedule_id"])
        weighted_values: dict[str, list[float]] = {}
        weight_sum = float(schedule["world_weight_sum"])
        for world in schedule["worlds"]:
            total_worlds += 1
            state_string = world["sampled_state"]
            if _sha256(state_string.encode("utf-8")) != world["state_sha256"]:
                raise IndependentActionValueError("sampled root state hash does not match capture")
            try:
                root_state = engine.State.from_string(state_string)
            except Exception as exc:
                raise IndependentActionValueError(f"cannot parse sampled root state: {exc}") from exc
            if _is_terminal(root_state):
                raise IndependentActionValueError("sampled root state is already terminal")
            side_one_actions, _ = _options(engine, root_state)
            canonical_actions = sorted(side_one_actions)
            if expected_actions is None:
                expected_actions = canonical_actions
                action_schedule_values = {action: [] for action in expected_actions}
                action_sample_counts = {action: 0 for action in expected_actions}
            elif canonical_actions != expected_actions:
                raise IndependentActionValueError(
                    "side-one root legal actions differ across schedules or worlds"
                )

            weight = float(world["sample_weight"])
            for action in expected_actions:
                outcomes = []
                for rollout in range(rollout_count):
                    tape = {
                        "base_seed": seed,
                        "source_capture_identity": identity,
                        "source_capture_sha256": capture_sha256,
                        "schedule_id": schedule_id,
                        "world_index": int(world["world_index"]),
                        "rollout": rollout,
                    }
                    outcomes.append(
                        _rollout(
                            engine,
                            state_string,
                            action,
                            tape=tape,
                            max_decisions=horizon,
                        )
                    )
                world_mean = math.fsum(outcomes) / rollout_count
                weighted_values.setdefault(action, []).append(weight * world_mean)
                action_sample_counts[action] += rollout_count
        for action in expected_actions or ():
            action_schedule_values[action].append(
                math.fsum(weighted_values[action]) / weight_sum
            )

    if expected_actions is None:
        raise IndependentActionValueError("source capture has no sampled worlds")
    schedule_count = len(capture["schedules"])
    sampling = capture.get("sampling")
    original_capture_sha256 = None
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise IndependentActionValueError("capture sampling linkage must be an object")
        original_capture_sha256 = sampling.get("source_capture_sha256")
        if (
            not isinstance(original_capture_sha256, str)
            or len(original_capture_sha256) != 64
            or any(character not in "0123456789abcdef" for character in original_capture_sha256)
        ):
            raise IndependentActionValueError("panel-selected capture source hash is invalid")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "independent_action_values",
        "root_identity": dict(identity),
        "source_linkage": {
            "capture_sha256": capture_sha256,
            "panel_source_capture_sha256": original_capture_sha256,
            "input_sha256": input_sha256,
            "input_manifest_sha256": manifest_sha256,
            "input_manifest_file_sha256": manifest_file_sha256,
        },
        "estimand": "development independent action value under uniform-legal continuations; not r1-continuation value",
        "oracle": False,
        "r1_continuation_value": False,
        "opponent_policy_id": POLICY_ID,
        "continuation_policy_id": POLICY_ID,
        "value_scale": "terminal_value [-1,1] mapped affinely to [0,1]",
        "configuration": {
            "rollouts": rollout_count,
            "max_decisions": horizon,
            "base_seed": seed,
            "schedule_aggregation": "equal",
            "world_aggregation": "source_sample_weight",
            "rollout_aggregation": "equal",
            "terminal_detection": "either State side has no Pokemon with hp > 0",
            "nonterminal_horizon": "error",
        },
        "common_tape": _common_tape_definition(),
        "schedule_count": schedule_count,
        "world_count": total_worlds,
        "actions": [
            {
                "action": action,
                "q": math.fsum(action_schedule_values[action]) / schedule_count,
                "sample_count": action_sample_counts[action],
            }
            for action in expected_actions
        ],
    }
    record["record_sha256"] = _sha256(_canonical_json(record).encode("ascii"))
    validate_action_value_record(record)
    return record


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                IndependentActionValueError(f"non-finite manifest constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, IndependentActionValueError) as exc:
        raise IndependentActionValueError(f"cannot read frozen input manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise IndependentActionValueError("frozen input manifest must be an object")
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        raise IndependentActionValueError(f"invalid frozen input manifest: {exc}") from exc
    if manifest.get("manifest_type") != "experiment_input":
        raise IndependentActionValueError(
            "independent action-value collection requires an experiment_input manifest"
        )
    return manifest, _sha256(payload)


def _load_captures(path: Path, max_records: int | None) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IndependentActionValueError(f"cannot read teacher root captures {path}: {exc}") from exc
    captures = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            capture = json.loads(
                raw_line.decode("utf-8"),
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    IndependentActionValueError(f"non-finite JSON constant {constant}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError, IndependentActionValueError) as exc:
            raise IndependentActionValueError(f"{path}:{line_number}: invalid capture JSON: {exc}") from exc
        if not isinstance(capture, dict):
            raise IndependentActionValueError(f"{path}:{line_number}: capture must be an object")
        captures.append(capture)
        if max_records is not None and len(captures) >= max_records:
            break
    if not captures:
        raise IndependentActionValueError("teacher root-capture input contains no records")
    return captures, _sha256(payload)


def _paths_collide(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _install_private(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    installed = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
            installed = True
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise IndependentActionValueError(f"output already exists: {path}") from exc
            installed = True
            temporary_path.unlink()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed or temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def collect_file(
    input_path: Path,
    input_manifest_path: Path,
    output_path: Path,
    *,
    rollouts: int,
    max_decisions: int,
    base_seed: int,
    max_records: int | None = None,
    force: bool = False,
    engine: Any | None = None,
) -> dict[str, Any]:
    for left, right in (
        (input_path, input_manifest_path),
        (input_path, output_path),
        (input_manifest_path, output_path),
    ):
        if _paths_collide(left, right):
            raise IndependentActionValueError("input, input-manifest, and output paths must be distinct")
    rollout_count = _positive_integer(rollouts, "rollouts")
    horizon = _positive_integer(max_decisions, "max decisions")
    seed = _base_seed(base_seed)
    if max_records is not None:
        _positive_integer(max_records, "max records")
    if not force and os.path.lexists(output_path):
        raise IndependentActionValueError(f"output already exists: {output_path}")

    manifest, manifest_file_sha256 = _load_manifest(input_manifest_path)
    captures, input_sha256 = _load_captures(input_path, max_records)
    manifest_sha256 = str(manifest["manifest_sha256"])
    seen_capture_hashes: set[str] = set()
    records = []
    for index, capture in enumerate(captures, start=1):
        capture_sha256 = capture.get("capture_sha256")
        if capture_sha256 in seen_capture_hashes:
            raise IndependentActionValueError(f"duplicate source capture at selected record {index}")
        if isinstance(capture_sha256, str):
            seen_capture_hashes.add(capture_sha256)
        records.append(
            collect_capture(
                capture,
                rollouts=rollout_count,
                max_decisions=horizon,
                base_seed=seed,
                input_sha256=input_sha256,
                manifest_sha256=manifest_sha256,
                manifest_file_sha256=manifest_file_sha256,
                engine=engine,
            )
        )
    payload = "".join(_canonical_json(record) + "\n" for record in records).encode("ascii")
    _install_private(output_path, payload, force=force)
    return {
        "input_sha256": input_sha256,
        "input_manifest_sha256": manifest_sha256,
        "input_manifest_file_sha256": manifest_file_sha256,
        "output_sha256": _sha256(payload),
        "records": len(records),
        "rollouts": rollout_count,
        "max_decisions": horizon,
        "base_seed": seed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=int)
    parser.add_argument("--max-decisions", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = collect_file(
        args.input,
        args.input_manifest,
        args.output,
        rollouts=args.rollouts,
        max_decisions=args.max_decisions,
        base_seed=args.seed,
        max_records=args.max_records,
        force=args.force,
    )
    print(_canonical_json(summary))


if __name__ == "__main__":
    main()
