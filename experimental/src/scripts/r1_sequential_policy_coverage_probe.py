#!/usr/bin/env python3
"""One-sided private aggregate sequential coverage probe for frozen r1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import scripts.r1_public_events as r1_public_events_module
from eval.experiment_manifest import validate_manifest
from scripts.r1_public_events import (
    R1_SEMANTIC_CONTRACT,
    R1SwitchTracker,
    _canonical_action,
    _norm,
    project_information_set_observations,
)
from scripts.teacher_root_bundle import validate_root_capture
from scripts.verify_r1_policy_snapshots import infer_snapshots


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPORT_SCHEMA_VERSION = 1
ESTIMAND = "one_sided_player_r1_vs_uniform_common_legal_sequential_certificate_coverage"
ROOT_POLICY_TOLERANCE = 1e-7
CHECKPOINT_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
PLAYER_POLICY_ID = f"frozen_r1_epoch_5_stateless_v1:{CHECKPOINT_SHA256}"
OPPONENT_POLICY_ID = "uniform_common_legal_v1"
FAILURE_CODES = (
    "ENGINE_TERMINAL_INVALID",
    "TERMINAL_DISAGREEMENT",
    "COMMON_LEGAL_SUPPORT_REJECTED",
    "POLICY_INVALID",
    "POLICY_SUPPORT_REJECTED",
    "PROJECTION_REJECTED",
    "PROJECTION_LINEAGE_REJECTED",
    "TRACKER_REJECTED",
)
INPUT_HASH_FIELDS = {
    "capture_file_sha256",
    "source_manifest_sha256",
    "source_manifest_file_sha256",
    "analysis_manifest_sha256",
    "analysis_manifest_file_sha256",
    "engine_binding_sha256",
    "probe_script_sha256",
    "public_events_module_sha256",
    "checkpoint_sha256",
}


class SequentialCoverageProbeError(ValueError):
    pass


@dataclass(frozen=True)
class WeightedWorld:
    state: Any
    mass: float


@dataclass
class ProbeNode:
    worlds: tuple[WeightedWorld, ...]
    tracker: Any
    observation: Mapping[str, Any]
    mass: float


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise SequentialCoverageProbeError("cannot hash a pinned input artifact") from exc


def counter_tape_uniform(
    base_seed: int,
    capture_file_sha256: str,
    schedule_id: int,
    rollout: int,
    depth: int,
    channel: str,
) -> float:
    """Return one deterministic SHA-256 counter-tape draw in [0, 1)."""
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed < 2**64
        or not isinstance(capture_file_sha256, str)
        or len(capture_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in capture_file_sha256)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (schedule_id, rollout, depth))
        or channel not in {"player", "opponent", "chance"}
    ):
        raise SequentialCoverageProbeError("invalid counter tape coordinate")
    payload = _canonical_json(
        [base_seed, capture_file_sha256, schedule_id, rollout, depth, channel]
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def _inverse_cdf(items: Sequence[tuple[Any, float]], uniform: float) -> Any:
    if any(not math.isfinite(weight) or weight < 0 for _, weight in items):
        raise SequentialCoverageProbeError("invalid sampling distribution")
    total = math.fsum(weight for _, weight in items)
    if not items or not math.isfinite(total) or total <= 0:
        raise SequentialCoverageProbeError("invalid sampling distribution")
    threshold = uniform * total
    cumulative = 0.0
    for item, weight in items:
        cumulative += weight
        if threshold < cumulative:
            return item
    return items[-1][0]


def _policy_distribution(
    observation: Mapping[str, Any],
    policy_infer: Callable[[Mapping[str, Any]], Sequence[float]],
) -> list[tuple[str, float]]:
    illegal = observation.get("illegal_actions")
    names = observation.get("name_table")
    try:
        raw = policy_infer(observation)
        weights = [float(value) for value in raw]
    except Exception:
        raise SequentialCoverageProbeError("POLICY_INVALID") from None
    if (
        not isinstance(illegal, list)
        or not isinstance(names, Mapping)
        or len(weights) != len(illegal)
        or any(not isinstance(value, bool) for value in illegal)
        or any(not math.isfinite(value) or value < 0 for value in weights)
    ):
        raise SequentialCoverageProbeError("POLICY_INVALID")
    by_index: dict[int, str] = {}
    canonical_indices: dict[str, int] = {}
    for name, raw_index in names.items():
        if (
            not isinstance(name, str)
            or isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or not 0 <= raw_index < len(illegal)
        ):
            raise SequentialCoverageProbeError("POLICY_INVALID")
        canonical = _canonical_action(name)
        if (
            raw_index in by_index
            and by_index[raw_index] != canonical
            or canonical in canonical_indices
            and canonical_indices[canonical] != raw_index
        ):
            raise SequentialCoverageProbeError("POLICY_INVALID")
        by_index[raw_index] = canonical
        canonical_indices[canonical] = raw_index
    support: list[tuple[str, float]] = []
    for index, weight in enumerate(weights):
        if weight <= 0:
            continue
        canonical = by_index.get(index)
        if illegal[index] or canonical is None:
            raise SequentialCoverageProbeError("POLICY_INVALID")
        support.append((canonical, weight))
    if not support or not math.isclose(math.fsum(weight for _, weight in support), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SequentialCoverageProbeError("POLICY_INVALID")
    return support


def _common_options(engine: Any, worlds: Sequence[WeightedWorld]) -> tuple[dict[str, str], list[str]]:
    side_one_sets: list[set[str]] = []
    side_two_sets: list[set[str]] = []
    side_one_names: dict[str, str] = {}
    side_two_names: dict[str, str] = {}
    try:
        for world in worlds:
            first, second = engine.root_options(state=world.state)
            first, second = list(first), list(second)
            if not first or not second:
                raise ValueError
            canonical_first = [_canonical_action(action) for action in first]
            canonical_second = [_canonical_action(action) for action in second]
            if len(set(canonical_first)) != len(first) or len(set(canonical_second)) != len(second):
                raise ValueError
            side_one_sets.append(set(canonical_first))
            side_two_sets.append(set(canonical_second))
            side_one_names.update(zip(canonical_first, first))
            side_two_names.update(zip(canonical_second, second))
    except Exception:
        raise SequentialCoverageProbeError("COMMON_LEGAL_SUPPORT_REJECTED") from None
    common_first = set.intersection(*side_one_sets)
    common_second = set.intersection(*side_two_sets)
    if not common_first or not common_second:
        raise SequentialCoverageProbeError("COMMON_LEGAL_SUPPORT_REJECTED")
    return (
        {name: side_one_names[name] for name in sorted(common_first)},
        [side_two_names[name] for name in sorted(common_second)],
    )


def _engine_terminal(state: Any) -> bool:
    marker = getattr(state, "terminal", None)
    if isinstance(marker, bool):
        return marker
    try:
        return any(
            all(float(pokemon.hp) <= 0 for pokemon in side.pokemon)
            for side in (state.side_one, state.side_two)
        )
    except Exception:
        raise SequentialCoverageProbeError("ENGINE_TERMINAL_INVALID") from None


def _tracker_terminal(tracker: Any, observation: Mapping[str, Any] | None = None) -> bool:
    if observation is not None and isinstance(observation.get("terminal"), bool):
        return bool(observation["terminal"])
    marker = getattr(tracker, "terminal", None)
    if isinstance(marker, bool):
        return marker
    state = getattr(tracker, "state", None)
    won, lost = getattr(state, "battle_won", None), getattr(state, "battle_lost", None)
    if isinstance(won, bool) and isinstance(lost, bool):
        return won or lost
    return False


def _terminal_consensus(worlds: Sequence[WeightedWorld], tracker: Any, observation: Mapping[str, Any] | None = None) -> bool:
    statuses = [_engine_terminal(world.state) for world in worlds]
    if not statuses or any(status != statuses[0] for status in statuses[1:]) or statuses[0] != _tracker_terminal(tracker, observation):
        raise SequentialCoverageProbeError("TERMINAL_DISAGREEMENT")
    return statuses[0]


def _projection_children(projection: Any, node: ProbeNode) -> list[tuple[tuple[WeightedWorld, ...], Any, Mapping[str, Any]]]:
    classes = getattr(projection, "observation_classes", None)
    if classes is None:
        classes = (projection,)
        indices_by_class = (tuple(range(len(node.worlds))),)
    else:
        indices_by_class = tuple(tuple(item.source_world_indices) for item in classes)
    flattened = [index for indices in indices_by_class for index in indices]
    if sorted(flattened) != list(range(len(node.worlds))):
        raise SequentialCoverageProbeError("PROJECTION_LINEAGE_REJECTED")
    children = []
    for item, indices in zip(classes, indices_by_class):
        next_states = tuple(item.next_states)
        if len(next_states) != len(indices):
            raise SequentialCoverageProbeError("PROJECTION_LINEAGE_REJECTED")
        worlds = tuple(
            WeightedWorld(state, node.worlds[source_index].mass)
            for state, source_index in zip(next_states, indices)
        )
        try:
            if hasattr(item, "observation") and hasattr(item, "tracker"):
                tracker = item.tracker
                observation = item.observation.policy_payload()
            else:
                # Dependency-injected legacy shape retained for isolated tests.
                tracker = node.tracker.fork()
                if hasattr(projection, "observation_classes"):
                    observation = tracker.apply_basic_move_class(item)
                else:
                    observation = tracker.apply_switch_projection(projection)
        except Exception:
            raise SequentialCoverageProbeError("TRACKER_REJECTED") from None
        children.append((worlds, tracker, observation))
    if not math.isclose(
        math.fsum(world.mass for worlds, _, _ in children for world in worlds),
        math.fsum(world.mass for world in node.worlds),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SequentialCoverageProbeError("PROJECTION_LINEAGE_REJECTED")
    return children


def _blank_depth(depth: int) -> dict[str, Any]:
    return {
        "depth": depth,
        "entering": {"count": 0, "mass": 0.0},
        "certified_continuation": {"count": 0, "mass": 0.0},
        "certified_terminal": {"count": 0, "mass": 0.0},
        "horizon_censored": {"count": 0, "mass": 0.0},
        "fixed_failures": {code: {"count": 0, "mass": 0.0} for code in FAILURE_CODES},
    }


def _add(bucket: dict[str, Any], mass: float) -> None:
    bucket["count"] += 1
    bucket["mass"] += mass


def _root_weight(capture: Mapping[str, Any]) -> float:
    sampling = capture.get("sampling")
    weight = float(sampling["poststratification_weight"]) if isinstance(sampling, Mapping) else 1.0
    if not math.isfinite(weight) or weight <= 0:
        raise SequentialCoverageProbeError("root weights must be finite and positive")
    return weight


def player_tracker_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(dict(snapshot))
    cleaned["continuation_observation_history"] = copy.deepcopy(
        snapshot["player_observation_history"]
    )
    return cleaned


def probe_captures(
    captures: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
    tracker_factory: Callable[[Mapping[str, Any]], Any],
    policy_infer: Callable[[Mapping[str, Any]], Sequence[float]],
    horizon: int,
    rollouts: int,
    base_seed: int,
    capture_file_sha256: str,
    input_hashes: Mapping[str, str] | None = None,
    root_policy_tolerance: float = ROOT_POLICY_TOLERANCE,
) -> dict[str, Any]:
    """Run the dependency-injected aggregate probe over loaded schema-v3 captures."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise SequentialCoverageProbeError("horizon must be a nonnegative integer")
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts <= 0:
        raise SequentialCoverageProbeError("rollouts must be a positive integer")
    if not math.isfinite(root_policy_tolerance) or root_policy_tolerance < 0:
        raise SequentialCoverageProbeError("root policy tolerance is invalid")
    counter_tape_uniform(base_seed, capture_file_sha256, 0, 0, 0, "player")
    if not captures:
        raise SequentialCoverageProbeError("no captures to probe")
    hashes = dict(input_hashes or {"capture_file_sha256": capture_file_sha256})
    if (
        hashes.get("capture_file_sha256") != capture_file_sha256
        or not set(hashes).issubset(INPUT_HASH_FIELDS)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise SequentialCoverageProbeError("invalid input artifact hashes")
    try:
        if engine.r1_semantic_contract() != R1_SEMANTIC_CONTRACT:
            raise SequentialCoverageProbeError("engine semantic contract does not match")
        for capture in captures:
            validate_root_capture(capture)
            if capture.get("schema_version") != 3:
                raise SequentialCoverageProbeError("sequential probe requires schema-v3 captures")
    except SequentialCoverageProbeError:
        raise
    except Exception as exc:
        raise SequentialCoverageProbeError("invalid probe input") from exc

    root_weights = [_root_weight(capture) for capture in captures]
    root_total = math.fsum(root_weights)
    depths = [_blank_depth(depth) for depth in range(horizon + 1)]
    schedule_count = sampled_world_count = 0
    for capture, root_weight in zip(captures, root_weights):
        snapshot = capture["r1_policy_snapshot"]
        try:
            inferred_root = [float(value) for value in policy_infer(snapshot)]
            captured_root = [float(value) for value in snapshot["probs"]]
        except Exception:
            raise SequentialCoverageProbeError("root policy validation failed") from None
        if (
            len(inferred_root) != len(captured_root)
            or any(not math.isfinite(value) for value in inferred_root + captured_root)
            or any(abs(actual - expected) > root_policy_tolerance for actual, expected in zip(inferred_root, captured_root))
        ):
            raise SequentialCoverageProbeError("root policy validation failed")
        schedules = capture["schedules"]
        root_mass = root_weight / root_total
        for schedule in schedules:
            schedule_ordinal = schedule_count
            schedule_count += 1
            total = float(schedule["world_weight_sum"])
            worlds = []
            for world in schedule["worlds"]:
                try:
                    state = engine.State.from_string(world["sampled_state"])
                except Exception:
                    raise SequentialCoverageProbeError("cannot deserialize captured world") from None
                mass = root_mass / len(schedules) * float(world["sample_weight"]) / total / rollouts
                worlds.append(WeightedWorld(state, mass))
                sampled_world_count += 1
            for rollout in range(rollouts):
                try:
                    tracker = tracker_factory(snapshot)
                except Exception:
                    raise SequentialCoverageProbeError("cannot initialize root tracker") from None
                nodes = [ProbeNode(tuple(worlds), tracker, snapshot, math.fsum(world.mass for world in worlds))]
                for depth in range(horizon + 1):
                    next_nodes: list[ProbeNode] = []
                    for node in nodes:
                        _add(depths[depth]["entering"], node.mass)
                        if depth == horizon:
                            _add(depths[depth]["horizon_censored"], node.mass)
                            continue
                        try:
                            terminal = _terminal_consensus(node.worlds, node.tracker)
                        except SequentialCoverageProbeError as exc:
                            _add(depths[depth]["fixed_failures"][str(exc)], node.mass)
                            continue
                        if terminal:
                            _add(depths[depth]["certified_terminal"], node.mass)
                            continue
                        try:
                            common_player, common_opponent = _common_options(engine, node.worlds)
                            policy = _policy_distribution(node.observation, policy_infer)
                        except SequentialCoverageProbeError as exc:
                            _add(depths[depth]["fixed_failures"][str(exc)], node.mass)
                            continue
                        sampled_player_action = _inverse_cdf(policy, counter_tape_uniform(base_seed, capture_file_sha256, schedule_ordinal, rollout, depth, "player"))
                        if sampled_player_action not in common_player:
                            _add(
                                depths[depth]["fixed_failures"]["POLICY_SUPPORT_REJECTED"],
                                node.mass,
                            )
                            continue
                        player_action = common_player[sampled_player_action]
                        opponent_action = _inverse_cdf(
                            [(action, 1.0) for action in common_opponent],
                            counter_tape_uniform(base_seed, capture_file_sha256, schedule_ordinal, rollout, depth, "opponent"),
                        )
                        chance = counter_tape_uniform(base_seed, capture_file_sha256, schedule_ordinal, rollout, depth, "chance")
                        try:
                            projection = project_information_set_observations(
                                engine,
                                [world.state for world in node.worlds],
                                node.tracker,
                                player_action,
                                opponent_action,
                                chance,
                                public_opponent=node.tracker.public_opponent_registry(),
                            )
                        except Exception:
                            _add(depths[depth]["fixed_failures"]["PROJECTION_REJECTED"], node.mass)
                            continue
                        try:
                            children = _projection_children(projection, node)
                        except SequentialCoverageProbeError as exc:
                            _add(depths[depth]["fixed_failures"][str(exc)], node.mass)
                            continue
                        for child_worlds, child_tracker, observation in children:
                            child_mass = math.fsum(world.mass for world in child_worlds)
                            try:
                                child_terminal = _terminal_consensus(child_worlds, child_tracker, observation)
                            except SequentialCoverageProbeError as exc:
                                _add(depths[depth]["fixed_failures"][str(exc)], child_mass)
                                continue
                            if child_terminal:
                                _add(depths[depth]["certified_terminal"], child_mass)
                            else:
                                _add(depths[depth]["certified_continuation"], child_mass)
                                next_nodes.append(ProbeNode(child_worlds, child_tracker, observation, child_mass))
                    nodes = next_nodes

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "record_type": "r1_sequential_policy_coverage_probe",
        "claim_status": "private_descriptive_coverage_only",
        "estimand": ESTIMAND,
        "r1_continuation_value_allowed": False,
        "inputs": hashes,
        "configuration": {
            "horizon": horizon,
            "rollouts": rollouts,
            "base_seed": base_seed,
            "root_policy_tolerance": root_policy_tolerance,
            "player_policy": PLAYER_POLICY_ID,
            "opponent_policy": OPPONENT_POLICY_ID,
            "horizon_unit": "certified_transitions",
            "continuation_history_boundary": "player_observation_history_excluding_legacy_opponent_query",
        },
        "common_tape": {
            "algorithm": "sha256_counter_v1",
            "channels": ["chance", "opponent", "player"],
            "coordinate_fields": ["base_seed", "capture_file_sha256", "schedule_ordinal", "rollout", "depth", "channel"],
            "uniform_mapping": "first_53_sha256_bits_divided_by_2^53",
        },
        "counts": {
            "roots": len(captures),
            "schedules": schedule_count,
            "sampled_worlds": sampled_world_count,
            "rollouts_per_schedule": rollouts,
        },
        "depths": depths,
        "continuation_readiness": {
            "status": "blocked",
            "r1_continuation_value_allowed": False,
            "blockers": ["finite_horizon_coverage_only", "one_sided_certificate_coverage_only"],
        },
        "privacy": {
            "aggregate_only": True,
            "private_state_emitted": False,
            "identifier_emitted": False,
            "action_detail_emitted": False,
            "event_or_observation_emitted": False,
            "exception_text_emitted": False,
            "per_root_rows_emitted": False,
        },
    }
    report["report_sha256"] = _sha256_bytes(_canonical_json(report).encode("ascii"))
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != {
        "report_schema_version",
        "record_type",
        "claim_status",
        "estimand",
        "r1_continuation_value_allowed",
        "inputs",
        "configuration",
        "common_tape",
        "counts",
        "depths",
        "continuation_readiness",
        "privacy",
        "report_sha256",
    } or report.get("report_schema_version") != REPORT_SCHEMA_VERSION or report.get("record_type") != "r1_sequential_policy_coverage_probe" or report.get("claim_status") != "private_descriptive_coverage_only" or report.get("estimand") != ESTIMAND or report.get("r1_continuation_value_allowed") is not False:
        raise SequentialCoverageProbeError("invalid sequential coverage report")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("capture_file_sha256") is None
        or not set(inputs).issubset(INPUT_HASH_FIELDS)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in inputs.values()
        )
    ):
        raise SequentialCoverageProbeError("invalid report inputs")
    if report.get("common_tape") != {
        "algorithm": "sha256_counter_v1",
        "channels": ["chance", "opponent", "player"],
        "coordinate_fields": ["base_seed", "capture_file_sha256", "schedule_ordinal", "rollout", "depth", "channel"],
        "uniform_mapping": "first_53_sha256_bits_divided_by_2^53",
    }:
        raise SequentialCoverageProbeError("invalid common tape declaration")
    configuration = report.get("configuration")
    counts = report.get("counts")
    privacy = report.get("privacy")
    if not isinstance(configuration, Mapping) or set(configuration) != {
        "horizon", "rollouts", "base_seed", "root_policy_tolerance",
        "player_policy", "opponent_policy", "horizon_unit",
        "continuation_history_boundary",
    } or not isinstance(counts, Mapping) or set(counts) != {
        "roots", "schedules", "sampled_worlds", "rollouts_per_schedule",
    } or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise SequentialCoverageProbeError("invalid report configuration or counts")
    if (
        configuration.get("player_policy") != PLAYER_POLICY_ID
        or configuration.get("opponent_policy") != OPPONENT_POLICY_ID
        or configuration.get("horizon_unit") != "certified_transitions"
        or configuration.get("continuation_history_boundary")
        != "player_observation_history_excluding_legacy_opponent_query"
        or isinstance(configuration.get("horizon"), bool)
        or not isinstance(configuration.get("horizon"), int)
        or configuration["horizon"] < 0
        or isinstance(configuration.get("rollouts"), bool)
        or not isinstance(configuration.get("rollouts"), int)
        or configuration["rollouts"] <= 0
    ):
        raise SequentialCoverageProbeError("invalid report configuration or counts")
    if privacy != {
        "aggregate_only": True,
        "private_state_emitted": False,
        "identifier_emitted": False,
        "action_detail_emitted": False,
        "event_or_observation_emitted": False,
        "exception_text_emitted": False,
        "per_root_rows_emitted": False,
    }:
        raise SequentialCoverageProbeError("invalid privacy declaration")
    readiness = report.get("continuation_readiness")
    if readiness != {
        "status": "blocked",
        "r1_continuation_value_allowed": False,
        "blockers": ["finite_horizon_coverage_only", "one_sided_certificate_coverage_only"],
    }:
        raise SequentialCoverageProbeError("invalid continuation readiness")
    unhashed = dict(report)
    claimed = unhashed.pop("report_sha256", None)
    if claimed != _sha256_bytes(_canonical_json(unhashed).encode("ascii")):
        raise SequentialCoverageProbeError("sequential coverage report hash does not match")
    depths = report.get("depths")
    if not isinstance(depths, list) or not depths:
        raise SequentialCoverageProbeError("invalid depth aggregates")
    for index, row in enumerate(depths):
        if set(row) != {
            "depth", "entering", "certified_continuation", "certified_terminal",
            "horizon_censored", "fixed_failures",
        } or row.get("depth") != index or set(row.get("fixed_failures", {})) != set(FAILURE_CODES):
            raise SequentialCoverageProbeError("invalid depth aggregates")
        cells = [
            row[name]
            for name in (
                "entering",
                "certified_continuation",
                "certified_terminal",
                "horizon_censored",
            )
        ] + list(row["fixed_failures"].values())
        if any(
            not isinstance(cell, Mapping)
            or isinstance(cell.get("count"), bool)
            or not isinstance(cell.get("count"), int)
            or cell["count"] < 0
            or not isinstance(cell.get("mass"), (int, float))
            or not math.isfinite(float(cell["mass"]))
            or float(cell["mass"]) < 0
            for cell in cells
        ):
            raise SequentialCoverageProbeError("invalid depth aggregates")
        entering = float(row["entering"]["mass"])
        accounted = math.fsum(
            [float(row[name]["mass"]) for name in ("certified_continuation", "certified_terminal", "horizon_censored")]
            + [float(item["mass"]) for item in row["fixed_failures"].values()]
        )
        if not math.isclose(entering, accounted, rel_tol=0.0, abs_tol=1e-10):
            raise SequentialCoverageProbeError("depth mass is not conserved")
        if index and not math.isclose(entering, float(depths[index - 1]["certified_continuation"]["mass"]), rel_tol=0.0, abs_tol=1e-10):
            raise SequentialCoverageProbeError("inter-depth mass is not conserved")
    serialized = _canonical_json(report)
    for forbidden in ("sampled_state", "state_sha256", "battle_tag", "username", "decision_idx", "name_table", "protocol_prefix"):
        if forbidden in serialized:
            raise SequentialCoverageProbeError("sequential coverage report contains private fields")


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        validate_manifest(manifest)
    except Exception as exc:
        raise SequentialCoverageProbeError("invalid frozen input manifest") from exc
    if manifest.get("manifest_type") != "experiment_input":
        raise SequentialCoverageProbeError("frozen manifest must be an experiment input")
    return manifest, _sha256_bytes(payload)


def _load_captures(path: Path, manifest_hash: str) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SequentialCoverageProbeError("cannot read root captures") from exc
    captures, seen, identities = [], set(), set()
    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue
        try:
            capture = json.loads(raw_line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            validate_root_capture(capture)
        except Exception as exc:
            raise SequentialCoverageProbeError("invalid root capture") from exc
        identity = capture["identity"]
        identity_key = (
            identity.get("namespace"),
            identity.get("battle_tag"),
            identity.get("username"),
            identity.get("decision_idx"),
        )
        if capture.get("schema_version") != 3 or capture.get("configuration", {}).get("input_manifest_sha256") != manifest_hash or capture["capture_sha256"] in seen or identity_key in identities:
            raise SequentialCoverageProbeError("root capture linkage is invalid")
        seen.add(capture["capture_sha256"])
        identities.add(identity_key)
        captures.append(capture)
    if not captures:
        raise SequentialCoverageProbeError("root-capture input contains no records")
    captures.sort(key=lambda item: item["capture_sha256"])
    return captures, _sha256_bytes(payload)


def probe_file(
    input_path: Path,
    source_manifest_path: Path,
    analysis_manifest_path: Path,
    *,
    engine: Any,
    tracker_factory: Callable[[Mapping[str, Any]], Any],
    policy_infer: Callable[[Mapping[str, Any]], Sequence[float]],
    horizon: int,
    rollouts: int,
    base_seed: int,
) -> dict[str, Any]:
    source, source_file_hash = _load_manifest(source_manifest_path)
    analysis, analysis_file_hash = _load_manifest(analysis_manifest_path)
    artifact = analysis.get("artifacts", {}).get("engine_binding")
    if not isinstance(artifact, Mapping):
        raise SequentialCoverageProbeError("frozen manifest does not pin an engine binding")
    engine_path = Path(str(artifact.get("path", "")))
    engine_hash = _sha256_file(engine_path)
    runtime = getattr(getattr(engine, "poke_engine", None), "__file__", None)
    if engine_hash != artifact.get("sha256") or runtime is None or Path(runtime).resolve() != engine_path.resolve():
        raise SequentialCoverageProbeError("runtime engine binding is not the pinned artifact")
    checkpoint_artifact = analysis.get("artifacts", {}).get("checkpoint")
    if not isinstance(checkpoint_artifact, Mapping):
        raise SequentialCoverageProbeError("frozen manifest does not pin the checkpoint")
    checkpoint_hash = _sha256_file(Path(str(checkpoint_artifact.get("path", ""))))
    if checkpoint_hash != CHECKPOINT_SHA256 or checkpoint_hash != checkpoint_artifact.get("sha256"):
        raise SequentialCoverageProbeError("pinned checkpoint hash does not match")
    captures, capture_hash = _load_captures(input_path, str(source["manifest_sha256"]))
    return probe_captures(
        captures,
        engine=engine,
        tracker_factory=tracker_factory,
        policy_infer=policy_infer,
        horizon=horizon,
        rollouts=rollouts,
        base_seed=base_seed,
        capture_file_sha256=capture_hash,
        input_hashes={
            "capture_file_sha256": capture_hash,
            "source_manifest_sha256": str(source["manifest_sha256"]),
            "source_manifest_file_sha256": source_file_hash,
            "analysis_manifest_sha256": str(analysis["manifest_sha256"]),
            "analysis_manifest_file_sha256": analysis_file_hash,
            "engine_binding_sha256": engine_hash,
            "probe_script_sha256": _sha256_file(Path(__file__)),
            "public_events_module_sha256": _sha256_file(Path(r1_public_events_module.__file__)),
            "checkpoint_sha256": checkpoint_hash,
        },
    )


def write_report(report: Mapping[str, Any], path: Path, *, force: bool = False) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise SequentialCoverageProbeError("output already exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path: Path | None = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write((_canonical_json(dict(report)) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise SequentialCoverageProbeError("output already exists") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--analysis-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--rollouts", required=True, type=int)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    import poke_engine
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(WORKSPACE_ROOT / "srcs" / "models"),
        model_name="randbats_exit_r1",
        default_checkpoint=5,
    )
    experiment = model.initialize_agent(checkpoint=5, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device

    def policy(snapshot: Mapping[str, Any]) -> Sequence[float]:
        return infer_snapshots(agent, [dict(snapshot)], device)[0].tolist()

    report = probe_file(
        args.input,
        args.source_manifest,
        args.analysis_manifest,
        engine=poke_engine,
        tracker_factory=lambda snapshot: R1SwitchTracker.from_snapshot(
            player_tracker_snapshot(snapshot),
            model.observation_space,
        ),
        policy_infer=policy,
        horizon=args.horizon,
        rollouts=args.rollouts,
        base_seed=args.base_seed,
    )
    write_report(report, args.output, force=args.force)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
