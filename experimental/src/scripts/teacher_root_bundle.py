"""Deterministic same-root treatment capture for teacher qualification."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 3
LEGACY_CAPTURE_SCHEMA_VERSION = 2


class RootBundleError(ValueError):
    """Raised when deterministic root capture cannot satisfy its contract."""


@dataclass(frozen=True)
class RootBundleConfig:
    output_path: Path
    iterations: int
    repeats: int
    deep_multiplier: int
    base_seed: int
    c_puct: float
    include_state: bool
    manifest_sha256: str | None = None


@dataclass(frozen=True)
class RootCaptureConfig:
    output_path: Path
    schedule_count: int
    base_seed: int
    manifest_sha256: str
    c_puct: float = 2.0


def config_from_environment(
    environment: Mapping[str, str] | None = None,
) -> RootBundleConfig | None:
    source = os.environ if environment is None else environment
    output = source.get("METAGROSS_TEACHER_ROOT_BUNDLE")
    if not output:
        return None
    try:
        iterations = int(source["METAGROSS_TEACHER_ITERATIONS"])
        repeats = int(source.get("METAGROSS_TEACHER_REPEATS", "1"))
        deep_multiplier = int(source.get("METAGROSS_TEACHER_DEEP_MULTIPLIER", "4"))
        base_seed = int(source.get("METAGROSS_TEACHER_SEED", "0"))
        c_puct = float(source.get("METAGROSS_CPUCT", "2.0"))
    except (KeyError, ValueError) as exc:
        raise RootBundleError(f"invalid teacher root-bundle configuration: {exc}") from exc
    if iterations <= 0:
        raise RootBundleError("METAGROSS_TEACHER_ITERATIONS must be positive")
    if repeats <= 0:
        raise RootBundleError("METAGROSS_TEACHER_REPEATS must be positive")
    if deep_multiplier < 1:
        raise RootBundleError("METAGROSS_TEACHER_DEEP_MULTIPLIER must be at least 1")
    if not 0 <= base_seed < 2**64:
        raise RootBundleError("METAGROSS_TEACHER_SEED must fit an unsigned 64-bit integer")
    if not math.isfinite(c_puct) or c_puct <= 0:
        raise RootBundleError("METAGROSS_CPUCT must be finite and positive")
    manifest_sha256 = source.get("METAGROSS_TEACHER_MANIFEST_SHA256")
    if manifest_sha256 is not None and (
        len(manifest_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in manifest_sha256)
    ):
        raise RootBundleError(
            "METAGROSS_TEACHER_MANIFEST_SHA256 must be 64 hexadecimal characters"
        )
    return RootBundleConfig(
        output_path=Path(output),
        iterations=iterations,
        repeats=repeats,
        deep_multiplier=deep_multiplier,
        base_seed=base_seed,
        c_puct=c_puct,
        include_state=source.get("METAGROSS_TEACHER_INCLUDE_STATE", "1") == "1",
        manifest_sha256=manifest_sha256.lower() if manifest_sha256 else None,
    )


def capture_config_from_environment(
    environment: Mapping[str, str] | None = None,
) -> RootCaptureConfig | None:
    source = os.environ if environment is None else environment
    output = source.get("METAGROSS_TEACHER_ROOT_BUNDLE")
    if not output:
        return None
    try:
        schedule_count = int(source.get("METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES", "1"))
        base_seed = int(source.get("METAGROSS_TEACHER_DETERMINIZATION_SEED", "0"))
        c_puct = float(source.get("METAGROSS_CPUCT", "2.0"))
    except ValueError as exc:
        raise RootBundleError(f"invalid teacher root-capture configuration: {exc}") from exc
    manifest_sha256 = source.get("METAGROSS_TEACHER_MANIFEST_SHA256", "").lower()
    if schedule_count <= 0:
        raise RootBundleError("METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES must be positive")
    if not 0 <= base_seed < 2**64:
        raise RootBundleError("METAGROSS_TEACHER_DETERMINIZATION_SEED must fit an unsigned 64-bit integer")
    if not math.isfinite(c_puct) or c_puct <= 0:
        raise RootBundleError("METAGROSS_CPUCT must be finite and positive")
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise RootBundleError("teacher root capture requires a lowercase 64-character manifest SHA-256")
    return RootCaptureConfig(Path(output), schedule_count, base_seed, manifest_sha256, c_puct)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_seed(
    base_seed: int,
    identity: Mapping[str, Any],
    state_sha256: str,
    world_index: int,
    treatment: str,
    repeat: int,
) -> int:
    material = {
        "base_seed": base_seed,
        "identity": dict(identity),
        "state_sha256": state_sha256,
        "world_index": world_index,
        "treatment": treatment,
        "repeat": repeat,
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(material).encode("ascii")).digest()[:8], "big")


def derive_schedule_seed(
    base_seed: int, identity: Mapping[str, Any], schedule_id: int
) -> int:
    material = {
        "base_seed": base_seed,
        "identity": dict(identity),
        "schedule_id": schedule_id,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json(material).encode("ascii")).digest()[:8], "big"
    )


def derive_scheduled_tree_seed(
    base_seed: int,
    identity: Mapping[str, Any],
    schedule_id: int,
    state_sha256: str,
    world_index: int,
    treatment: str,
    repeat: int,
) -> int:
    material = {
        "base_seed": base_seed,
        "identity": dict(identity),
        "schedule_id": schedule_id,
        "state_sha256": state_sha256,
        "world_index": world_index,
        "treatment": treatment,
        "repeat": repeat,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json(material).encode("ascii")).digest()[:8], "big"
    )


def _validate_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity.get("namespace"), str) or any(
        identity.get(field) in (None, "")
        for field in ("battle_tag", "username", "decision_idx")
    ):
        raise RootBundleError(
            "teacher capture requires namespace, battle tag, username, and decision index"
        )


def _validate_r1_policy_snapshot(
    snapshot: Any,
    identity: Mapping[str, Any],
    player_priors: Sequence[tuple[str, float]],
) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != 3:
        raise RootBundleError("teacher capture requires an r1 schema-v3 policy snapshot")
    expected_identity = {
        "tag": identity.get("battle_tag"),
        "namespace": identity.get("namespace"),
        "username": identity.get("username"),
        "decision_idx": identity.get("decision_idx"),
        "battle_turn": identity.get("battle_turn"),
    }
    if any(snapshot.get(field) != value for field, value in expected_identity.items()):
        raise RootBundleError("r1 policy snapshot identity does not match teacher capture")
    text = snapshot.get("text_tokens")
    if not isinstance(text, list) or not text or any(
        isinstance(value, bool) or not isinstance(value, int) for value in text
    ):
        raise RootBundleError("r1 policy snapshot text tokens are invalid")
    numbers = snapshot.get("numbers")
    if not isinstance(numbers, list) or not numbers or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numbers
    ):
        raise RootBundleError("r1 policy snapshot numbers are invalid")
    illegal = snapshot.get("illegal_actions")
    if (
        not isinstance(illegal, list)
        or len(illegal) != 13
        or any(not isinstance(value, bool) for value in illegal)
        or all(illegal)
    ):
        raise RootBundleError("r1 policy snapshot legality mask is invalid")
    if snapshot.get("mask_fallback") is not False or snapshot.get("mask_fallback_error") is not None:
        raise RootBundleError("r1 policy snapshot used a fallback legality mask")
    name_table = snapshot.get("name_table")
    if not isinstance(name_table, dict) or not name_table:
        raise RootBundleError("r1 policy snapshot name table is invalid")
    indices: list[int] = []
    for action, index in name_table.items():
        if (
            not isinstance(action, str)
            or not action
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < 13
        ):
            raise RootBundleError("r1 policy snapshot name table is invalid")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise RootBundleError("r1 policy snapshot name table indices are duplicated")
    probabilities = snapshot.get("probs")
    if not isinstance(probabilities, list) or len(probabilities) != 13 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in probabilities
    ):
        raise RootBundleError("r1 policy snapshot probabilities are invalid")
    if not math.isclose(math.fsum(float(value) for value in probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RootBundleError("r1 policy snapshot probabilities are not normalized")
    if any(float(probabilities[index]) > 1e-12 for index, flag in enumerate(illegal) if flag):
        raise RootBundleError("r1 policy snapshot assigns probability to an illegal action")
    expected_priors = [(action, float(probabilities[index])) for action, index in name_table.items()]
    if len(player_priors) != len(expected_priors) or dict(player_priors) != dict(expected_priors):
        raise RootBundleError("recorded player priors do not match the r1 policy snapshot")
    protocol_prefix = snapshot.get("protocol_prefix")
    if (
        not isinstance(protocol_prefix, list)
        or not protocol_prefix
        or any(not isinstance(line, str) for line in protocol_prefix)
        or not any(line.startswith("|request|") for line in protocol_prefix)
    ):
        raise RootBundleError("r1 policy snapshot protocol prefix is invalid")
    information_state = snapshot.get("player_information_state")
    if (
        not isinstance(information_state, dict)
        or information_state.get("schema_version") != 1
        or not isinstance(information_state.get("universal_state"), dict)
        or not isinstance(information_state.get("player_team"), list)
        or not isinstance(information_state.get("opponent_public_team"), list)
    ):
        raise RootBundleError("r1 policy snapshot player-information state is invalid")
    for field in ("player_observation_history", "continuation_observation_history"):
        history = snapshot.get(field)
        if (
            not isinstance(history, dict)
            or not isinstance(history.get("any_opponent_asleep"), bool)
            or not isinstance(history.get("any_opponent_frozen"), bool)
            or not isinstance(history.get("revealed_opponents"), list)
            or any(
                not isinstance(species, str) or not species
                for species in history.get("revealed_opponents", [])
            )
        ):
            raise RootBundleError(f"r1 policy snapshot {field} is invalid")


def build_root_capture(
    *,
    identity: Mapping[str, Any],
    player_priors: Sequence[tuple[str, float]] | None,
    opponent_priors: Sequence[tuple[str, float]] | None,
    r1_policy_snapshot: Mapping[str, Any] | None,
    schedules: Sequence[Sequence[tuple[str, float]]],
    config: RootCaptureConfig,
) -> dict[str, Any]:
    """Build an immutable capture without running or storing shadow treatments."""
    _validate_identity(identity)
    player = _validated_priors(player_priors, "player priors")
    _validate_r1_policy_snapshot(r1_policy_snapshot, identity, player)
    opponent = (
        _validated_priors(opponent_priors, "recorded opponent priors")
        if opponent_priors
        else []
    )
    if len(schedules) != config.schedule_count:
        raise RootBundleError("captured schedule count does not match configuration")
    captured_schedules = []
    for schedule_id, worlds in enumerate(schedules):
        if not worlds:
            raise RootBundleError("captured schedules must contain at least one world")
        captured_worlds = []
        weight_sum = 0.0
        for world_index, (state_string, raw_weight) in enumerate(worlds):
            if not isinstance(state_string, str) or not state_string:
                raise RootBundleError("captured worlds require a sampled state string")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise RootBundleError("captured world weights must be finite and nonnegative")
            weight_sum += weight
            captured_worlds.append(
                {
                    "world_index": world_index,
                    "sample_weight": weight,
                    "state_sha256": _sha256_text(state_string),
                    "sampled_state": state_string,
                }
            )
        if weight_sum <= 0:
            raise RootBundleError("captured schedule has no positive world weight")
        captured_schedules.append(
            {
                "schedule_id": schedule_id,
                "sampling_seed": (
                    None
                    if schedule_id == 0
                    else derive_schedule_seed(config.base_seed, identity, schedule_id)
                ),
                "world_count": len(captured_worlds),
                "world_weight_sum": weight_sum,
                "worlds": captured_worlds,
            }
        )
    capture = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "record_type": "teacher_root_capture",
        "identity": dict(identity),
        "configuration": {
            "schedule_count": config.schedule_count,
            "schedule_base_seed": config.base_seed,
            "input_manifest_sha256": config.manifest_sha256,
            "c_puct": config.c_puct,
        },
        "behavior_schedule_id": 0,
        "recorded_player_priors": [[action, mass] for action, mass in player],
        "recorded_opponent_priors": [[action, mass] for action, mass in opponent],
        "r1_policy_snapshot": dict(r1_policy_snapshot),
        "schedules": captured_schedules,
    }
    capture["capture_sha256"] = hashlib.sha256(
        _canonical_json(capture).encode("ascii")
    ).hexdigest()
    validate_root_capture(capture)
    return capture


def validate_root_capture(capture: Mapping[str, Any]) -> None:
    schema_version = capture.get("schema_version")
    if schema_version not in (LEGACY_CAPTURE_SCHEMA_VERSION, CAPTURE_SCHEMA_VERSION) or capture.get("record_type") != "teacher_root_capture":
        raise RootBundleError("invalid teacher root-capture record")
    unhashed = dict(capture)
    claimed = unhashed.pop("capture_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
    if claimed != actual:
        raise RootBundleError("teacher root-capture hash does not match content")
    identity = capture.get("identity")
    configuration = capture.get("configuration")
    if not isinstance(identity, dict) or not isinstance(configuration, dict):
        raise RootBundleError("teacher root-capture identity and configuration are invalid")
    _validate_identity(identity)
    schedule_count = configuration.get("schedule_count")
    if isinstance(schedule_count, bool) or not isinstance(schedule_count, int) or schedule_count <= 0:
        raise RootBundleError("teacher root-capture schedule count is invalid")
    schedule_base_seed = configuration.get("schedule_base_seed")
    if (
        isinstance(schedule_base_seed, bool)
        or not isinstance(schedule_base_seed, int)
        or not 0 <= schedule_base_seed < 2**64
    ):
        raise RootBundleError("teacher root-capture schedule base seed is invalid")
    manifest_sha256 = configuration.get("input_manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise RootBundleError("teacher root-capture manifest hash is invalid")
    _validated_priors(
        [tuple(value) for value in capture.get("recorded_player_priors", [])],
        "player priors",
    )
    if schema_version == CAPTURE_SCHEMA_VERSION:
        _validate_r1_policy_snapshot(
            capture.get("r1_policy_snapshot"),
            identity,
            [tuple(value) for value in capture.get("recorded_player_priors", [])],
        )
    opponent = capture.get("recorded_opponent_priors")
    if not isinstance(opponent, list):
        raise RootBundleError("recorded opponent priors must be a list")
    if opponent:
        _validated_priors([tuple(value) for value in opponent], "recorded opponent priors")
    schedules = capture.get("schedules")
    if not isinstance(schedules, list) or len(schedules) != schedule_count:
        raise RootBundleError("teacher root-capture schedules are incomplete")
    if capture.get("behavior_schedule_id") != 0:
        raise RootBundleError("teacher root-capture behavior schedule must be zero")
    for schedule_id, schedule in enumerate(schedules):
        if not isinstance(schedule, dict) or schedule.get("schedule_id") != schedule_id:
            raise RootBundleError("teacher root-capture schedule IDs must be contiguous")
        expected_seed = (
            None
            if schedule_id == 0
            else derive_schedule_seed(
                int(configuration["schedule_base_seed"]), identity, schedule_id
            )
        )
        if schedule.get("sampling_seed") != expected_seed:
            raise RootBundleError("teacher root-capture schedule seed is invalid")
        worlds = schedule.get("worlds")
        if (
            not isinstance(worlds, list)
            or not worlds
            or schedule.get("world_count") != len(worlds)
        ):
            raise RootBundleError("teacher root-capture world count is invalid")
        weight_sum = 0.0
        for world_index, world in enumerate(worlds):
            if not isinstance(world, dict) or world.get("world_index") != world_index:
                raise RootBundleError("teacher root-capture world indices are invalid")
            state = world.get("sampled_state")
            if (
                not isinstance(state, str)
                or not state
                or world.get("state_sha256") != _sha256_text(state)
            ):
                raise RootBundleError("teacher root-capture sampled state hash is invalid")
            weight = float(world.get("sample_weight", float("nan")))
            if not math.isfinite(weight) or weight < 0:
                raise RootBundleError("teacher root-capture world weight is invalid")
            weight_sum += weight
        if weight_sum <= 0 or not math.isclose(
            weight_sum,
            float(schedule.get("world_weight_sum", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RootBundleError("teacher root-capture world weights are invalid")


def append_root_capture(path: Path, capture: Mapping[str, Any]) -> None:
    validate_root_capture(capture)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (_canonical_json(dict(capture)) + "\n").encode("ascii")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("teacher root-capture append made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_priors(
    priors: Sequence[tuple[str, float]] | None,
    name: str,
) -> list[tuple[str, float]]:
    if not priors:
        raise RootBundleError(f"{name} are required for teacher capture")
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for action, raw_mass in priors:
        if not isinstance(action, str) or not action or action in seen:
            raise RootBundleError(f"{name} contain an invalid or duplicate action")
        mass = float(raw_mass)
        if not math.isfinite(mass) or mass < 0:
            raise RootBundleError(f"{name} contain non-finite or negative mass")
        seen.add(action)
        result.append((action, mass))
    if math.fsum(mass for _, mass in result) <= 0:
        raise RootBundleError(f"{name} have no positive mass")
    return result


def equal_priors(options: Sequence[Any]) -> list[tuple[str, float]]:
    actions = [str(option.move_choice) for option in options]
    if not actions or len(actions) != len(set(actions)):
        raise RootBundleError("MCTS root options must be nonempty and unique")
    mass = 1.0 / len(actions)
    return [(action, mass) for action in actions]


def match_priors_to_options(
    priors: Sequence[tuple[str, float]], options: Sequence[Any]
) -> list[tuple[str, float]]:
    by_action = dict(priors)
    actions = [str(option.move_choice) for option in options]
    matched = [(action, by_action.get(action, 0.0)) for action in actions]
    total = math.fsum(mass for _, mass in matched)
    if total <= 0:
        raise RootBundleError("player priors have no mass on this world's legal actions")
    return [(action, mass / total) for action, mass in matched]


def snapshot_result(result: Any) -> dict[str, Any]:
    def side(entries: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "action": str(entry.move_choice),
                "visits": int(entry.visits),
                "total_score": float(entry.total_score),
            }
            for entry in entries
        ]

    total = int(result.total_visits)
    if total <= 0:
        raise RootBundleError("deterministic treatment returned no visits")
    return {"total_visits": total, "side_one": side(result.side_one), "side_two": side(result.side_two)}


def run_world_treatments(
    *,
    state_string: str,
    live_result: Any,
    world_index: int,
    identity: Mapping[str, Any],
    player_priors: Sequence[tuple[str, float]] | None,
    opponent_priors: Sequence[tuple[str, float]] | None,
    config: RootBundleConfig,
) -> dict[str, Any]:
    """Run matched shadow searches without mutating the live MCTS result."""
    import poke_engine

    if not isinstance(identity.get("namespace"), str) or any(
        identity.get(field) in (None, "")
        for field in ("battle_tag", "username", "decision_idx")
    ):
        raise RootBundleError("teacher capture requires namespace, battle tag, username, and decision index")
    player = _validated_priors(player_priors, "player priors")
    opponent = _validated_priors(opponent_priors, "recorded opponent priors") if opponent_priors else []
    side_one_equal = equal_priors(live_result.side_one)
    side_two_equal = equal_priors(live_result.side_two)
    effective_player = match_priors_to_options(player, live_result.side_one)
    state_sha256 = _sha256_text(state_string)
    treatment_specs = [
        ("U-B", config.iterations, side_one_equal, side_two_equal),
        ("S-B", config.iterations, effective_player, side_two_equal),
    ]
    if config.deep_multiplier > 1:
        deep_treatment = f"S-{config.deep_multiplier}B"
        treatment_specs.append(
            (
                deep_treatment,
                config.iterations * config.deep_multiplier,
                effective_player,
                side_two_equal,
            )
        )

    results: dict[str, list[dict[str, Any]]] = {}
    for treatment, iterations, side_one_priors, side_two_priors in treatment_specs:
        repeats = []
        for repeat in range(config.repeats):
            seed = derive_seed(
                config.base_seed,
                identity,
                state_sha256,
                world_index,
                treatment,
                repeat,
            )
            state = poke_engine.State.from_string(state_string)
            result = poke_engine.monte_carlo_tree_search(
                state,
                duration_ms=0,
                iterations=iterations,
                threads=1,
                s1_priors=side_one_priors,
                s2_priors=side_two_priors,
                c_puct=config.c_puct,
                seed=seed,
            )
            repeats.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "iterations": iterations,
                    "result": snapshot_result(result),
                }
            )
        results[treatment] = repeats

    world = {
        "world_index": world_index,
        "state_sha256": state_sha256,
        "live_result": snapshot_result(live_result),
        "recorded_player_priors": [[action, mass] for action, mass in player],
        "effective_player_priors": [[action, mass] for action, mass in effective_player],
        "recorded_opponent_priors": [[action, mass] for action, mass in opponent],
        "equal_side_one_priors": [[action, mass] for action, mass in side_one_equal],
        "equal_side_two_priors": [[action, mass] for action, mass in side_two_equal],
        "treatments": results,
    }
    if config.include_state:
        world["sampled_state"] = state_string
    return world


def _aggregate_policy(
    worlds: Sequence[dict[str, Any]], treatment: str, repeat: int
) -> list[dict[str, Any]]:
    mass_by_action: dict[str, float] = {}
    weight_total = 0.0
    for world in worlds:
        weight = float(world["sample_weight"])
        result = world["capture"]["treatments"][treatment][repeat]["result"]
        total_visits = result["total_visits"]
        weight_total += weight
        for option in result["side_one"]:
            mass_by_action[option["action"]] = mass_by_action.get(option["action"], 0.0) + (
                weight * option["visits"] / total_visits
            )
    total = math.fsum(mass_by_action.values())
    if weight_total <= 0 or total <= 0:
        raise RootBundleError("cannot aggregate treatment with nonpositive world or policy mass")
    return [
        {"action": action, "probability": mass / total}
        for action, mass in sorted(mass_by_action.items())
    ]


def build_root_bundle(mcts_results: Sequence[tuple[Any, float, int]]) -> dict[str, Any]:
    worlds: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for live_result, raw_weight, index in mcts_results:
        capture = getattr(live_result, "_teacher_root_capture", None)
        if not isinstance(capture, dict):
            raise RootBundleError("MCTS result is missing deterministic teacher capture")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise RootBundleError("world sample weight must be finite and nonnegative")
        if int(capture["world_index"]) != int(index):
            raise RootBundleError("world index changed between worker and aggregator")
        identities.append(dict(capture["identity"]))
        configs.append(dict(capture["configuration"]))
        worlds.append(
            {
                "world_index": int(index),
                "sample_weight": weight,
                "capture": capture["world"],
            }
        )
    if not worlds:
        raise RootBundleError("root bundle has no worlds")
    if any(identity != identities[0] for identity in identities[1:]):
        raise RootBundleError("world captures disagree on decision identity")
    if any(configuration != configs[0] for configuration in configs[1:]):
        raise RootBundleError("world captures disagree on treatment configuration")
    worlds.sort(key=lambda world: world["world_index"])
    expected = list(range(len(worlds)))
    if [world["world_index"] for world in worlds] != expected:
        raise RootBundleError("world indices must be contiguous from zero")

    treatment_names = sorted(worlds[0]["capture"]["treatments"])
    repeat_count = int(configs[0]["repeats"])
    aggregate = {
        treatment: [
            {"repeat": repeat, "side_one_policy": _aggregate_policy(worlds, treatment, repeat)}
            for repeat in range(repeat_count)
        ]
        for treatment in treatment_names
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "teacher_root_bundle",
        "identity": identities[0],
        "configuration": configs[0],
        "contains_sampled_private_states": any(
            "sampled_state" in world["capture"] for world in worlds
        ),
        "world_count": len(worlds),
        "world_weight_sum": math.fsum(world["sample_weight"] for world in worlds),
        "worlds": worlds,
        "aggregate_treatments": aggregate,
    }
    bundle["bundle_sha256"] = hashlib.sha256(_canonical_json(bundle).encode("ascii")).hexdigest()
    return bundle


def validate_root_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("record_type") != "teacher_root_bundle":
        raise RootBundleError("invalid root-bundle record")
    unhashed = dict(bundle)
    claimed = unhashed.pop("bundle_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
    if claimed != actual:
        raise RootBundleError("root-bundle hash does not match content")
    worlds = bundle.get("worlds")
    if not isinstance(worlds, list) or not worlds or bundle.get("world_count") != len(worlds):
        raise RootBundleError("root-bundle world count is invalid")
    if [world.get("world_index") for world in worlds] != list(range(len(worlds))):
        raise RootBundleError("root-bundle world indices are invalid")


def append_root_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    validate_root_bundle(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (_canonical_json(dict(bundle)) + "\n").encode("ascii")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("root-bundle append made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
