#!/usr/bin/env python3
"""Aggregate actual-state one-transition dual r1 tracker parity probe.

This estimates coverage of a frozen-r1-versus-frozen-r1 transition check.  It
is deliberately not a joint-belief estimand, value estimate, or strength
estimate: separately sampled hidden opponent teams are never coupled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import scripts.r1_public_events as public_events_module
from eval.experiment_manifest import sha256_file, validate_manifest
from scripts.audit_dual_r1_policy_snapshots import (
    _is_forced_switch,
    _public_prefix,
    _validated_snapshot,
    normalize_battle_tag,
    normalize_username,
)

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None  # type: ignore[assignment]
from scripts.r1_public_events import (
    R1_SEMANTIC_CONTRACT,
    R1SwitchTracker,
    _canonical_action,
    project_information_set_transition,
)
from scripts.r1_sequential_policy_coverage_probe import player_tracker_snapshot
from scripts.teacher_root_bundle import validate_root_capture
from scripts.verify_r1_policy_snapshots import infer_snapshots


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPORT_SCHEMA_VERSION = 1
ROOT_POLICY_TOLERANCE = 1e-7
CHECKPOINT_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
POLICY_ID = f"frozen_r1_epoch_5_stateless_v1:{CHECKPOINT_SHA256}"
ESTIMAND = "fused_actual_sides_mechanical_root_equal_weighted_frozen_r1_vs_frozen_r1_one_transition_dual_tracker_parity_coverage"
ROOT_FAILURES = (
    "ROOT_FUSION_REJECTED",
    "ROOT_TRACKER_REJECTED_P1_ONLY",
    "ROOT_TRACKER_REJECTED_P2_ONLY",
    "ROOT_TRACKER_REJECTED_BOTH",
    "ROOT_POLICY_PARITY_REJECTED_P1_ONLY",
    "ROOT_POLICY_PARITY_REJECTED_P2_ONLY",
    "ROOT_POLICY_PARITY_REJECTED_BOTH",
)
TRANSITION_FAILURES = (
    "ACTION_MAPPING_REJECTED_P1_ONLY",
    "ACTION_MAPPING_REJECTED_P2_ONLY",
    "ACTION_MAPPING_REJECTED_BOTH",
    "PROJECTION_REJECTED_P1_ONLY",
    "PROJECTION_REJECTED_P2_ONLY",
    "PROJECTION_REJECTED_BOTH",
    "PROJECTION_LINEAGE_REJECTED",
    "PUBLIC_OUTCOME_MISMATCH",
    "TRACKER_REJECTED_P1_ONLY",
    "TRACKER_REJECTED_P2_ONLY",
    "TRACKER_REJECTED_BOTH",
    "NEXT_LEGALITY_MISMATCH_P1_ONLY",
    "NEXT_LEGALITY_MISMATCH_P2_ONLY",
    "NEXT_LEGALITY_MISMATCH_BOTH",
    "NEXT_POLICY_INVALID_P1_ONLY",
    "NEXT_POLICY_INVALID_P2_ONLY",
    "NEXT_POLICY_INVALID_BOTH",
    "TERMINAL_DISAGREEMENT",
)
OUTCOMES = ("certified_nonterminal", "certified_terminal", *ROOT_FAILURES, *TRANSITION_FAILURES)
INPUT_HASH_FIELDS = {
    "p1_capture_file_sha256", "p2_capture_file_sha256", "source_manifest_sha256",
    "source_manifest_file_sha256", "analysis_manifest_sha256",
    "analysis_manifest_file_sha256", "engine_binding_sha256", "checkpoint_sha256",
    "probe_script_sha256", "public_events_module_sha256",
}
PRIVACY = {
    "aggregate_only": True,
    "identifiers_emitted": False,
    "actions_or_species_emitted": False,
    "states_or_state_hashes_emitted": False,
    "events_or_event_digests_emitted": False,
    "exceptions_emitted": False,
    "per_root_rows_emitted": False,
}
FORBIDDEN_REPORT_TERMS = (
    "battle_tag", "username", "decision_idx", "sampled_state", "state_sha256",
    "name_table", "protocol_prefix", "action_detail",
)


class DualTrackerProbeError(ValueError):
    """Fail-closed input, execution, or report validation error."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _side_code(prefix: str, p1_failed: bool, p2_failed: bool) -> str | None:
    if not p1_failed and not p2_failed:
        return None
    return f"{prefix}_{'BOTH' if p1_failed and p2_failed else 'P1_ONLY' if p1_failed else 'P2_ONLY'}"


def counter_tape_uniform(base_seed: int, joined_root_ordinal: int, rollout: int, channel: str) -> float:
    """Deterministic common SHA-256 tape draw in [0, 1)."""
    if (
        isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed < 2**64
        or isinstance(joined_root_ordinal, bool) or not isinstance(joined_root_ordinal, int) or joined_root_ordinal < 0
        or isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0
        or channel not in {"p1", "p2", "chance"}
    ):
        raise DualTrackerProbeError("invalid counter tape coordinate")
    digest = hashlib.sha256(_canonical_json([base_seed, joined_root_ordinal, rollout, channel]).encode("ascii")).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def _inverse_cdf(items: Sequence[tuple[Any, float]], u: float) -> Any:
    total = math.fsum(weight for _, weight in items)
    if not items or not math.isfinite(total) or total <= 0 or any(not math.isfinite(w) or w < 0 for _, w in items):
        raise DualTrackerProbeError("invalid captured policy")
    threshold = u * total
    cumulative = 0.0
    for item, weight in items:
        cumulative += weight
        if threshold < cumulative:
            return item
    return items[-1][0]


def _policy_support(snapshot: Mapping[str, Any], probabilities: Sequence[float] | None = None) -> list[tuple[str, float]]:
    illegal = snapshot.get("illegal_actions")
    names = snapshot.get("name_table")
    raw = snapshot.get("probs") if probabilities is None else probabilities
    try:
        probs = [float(value) for value in raw]  # type: ignore[arg-type]
    except Exception:
        raise DualTrackerProbeError("invalid policy") from None
    if (
        not isinstance(illegal, list) or len(illegal) != 13 or any(not isinstance(x, bool) for x in illegal)
        or not isinstance(names, Mapping) or len(probs) != 13
        or any(not math.isfinite(x) or x < 0 for x in probs)
        or not math.isclose(math.fsum(probs), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise DualTrackerProbeError("invalid policy")
    by_index: dict[int, str] = {}
    for name, index in names.items():
        if not isinstance(name, str) or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 13:
            raise DualTrackerProbeError("invalid policy")
        canonical = _canonical_action(name)
        if index in by_index or canonical in by_index.values():
            raise DualTrackerProbeError("invalid policy")
        by_index[index] = canonical
    support = []
    for index, probability in enumerate(probs):
        if illegal[index] and probability != 0:
            raise DualTrackerProbeError("invalid policy")
        if probability > 0:
            if index not in by_index or illegal[index]:
                raise DualTrackerProbeError("invalid policy")
            support.append((by_index[index], probability))
    if not support:
        raise DualTrackerProbeError("invalid policy")
    return support


def _snapshot_boundary(capture: Mapping[str, Any]) -> tuple[object, ...]:
    snapshot = capture["r1_policy_snapshot"]
    return (str(snapshot["namespace"]), normalize_battle_tag(str(snapshot["tag"])), snapshot["battle_turn"])


def _local_identity(capture: Mapping[str, Any]) -> tuple[object, ...]:
    identity = capture["identity"]
    return (identity["namespace"], normalize_battle_tag(str(identity["battle_tag"])), normalize_username(str(identity["username"])), identity["decision_idx"])


def join_captures(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], int]:
    """Validate and fail-closed join two opposite-client capture streams."""
    if not first or not second:
        raise DualTrackerProbeError("both capture inputs must be nonempty")
    all_rows = [*first, *second]
    seen_capture: set[str] = set()
    for rows in (first, second):
        identities: set[tuple[object, ...]] = set()
        for capture in rows:
            try:
                validate_root_capture(capture)
            except Exception as exc:
                raise DualTrackerProbeError("invalid root capture") from exc
            if capture.get("schema_version") != 3:
                raise DualTrackerProbeError("dual probe requires schema-v3 captures")
            if _validated_snapshot(capture) is None:
                raise DualTrackerProbeError("invalid schema-v3 policy snapshot")
            identity = _local_identity(capture)
            if identity in identities or capture["capture_sha256"] in seen_capture:
                raise DualTrackerProbeError("duplicate local identity or capture")
            identities.add(identity)
            seen_capture.add(str(capture["capture_sha256"]))
    manifests = {row["configuration"]["input_manifest_sha256"] for row in all_rows}
    namespaces = {str(row["r1_policy_snapshot"].get("namespace", "")) for row in all_rows}
    if len(manifests) != 1 or len(namespaces) != 1 or not next(iter(namespaces)).strip():
        raise DualTrackerProbeError("capture streams do not share source manifest and namespace")
    grouped: dict[tuple[object, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for source, rows in enumerate((first, second)):
        for row in rows:
            grouped[_snapshot_boundary(row)].append((source, row))
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    forced = 0
    for boundary in sorted(grouped, key=lambda key: _canonical_json(key)):
        by_prefix: dict[tuple[str, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for source, row in grouped[boundary]:
            by_prefix[tuple(_public_prefix(row["r1_policy_snapshot"]))].append(
                (source, row)
            )
        for prefix in sorted(by_prefix, key=_canonical_json):
            rows = by_prefix[prefix]
            forced_rows = [
                row
                for _, row in rows
                if _is_forced_switch(row["r1_policy_snapshot"])
            ]
            if forced_rows:
                if len(forced_rows) != len(rows) or len(rows) not in {1, 2}:
                    raise DualTrackerProbeError("unexplained or duplicate boundary join")
                forced += 1
                continue
            if len(rows) != 2 or {source for source, _ in rows} != {0, 1}:
                raise DualTrackerProbeError("unexplained or duplicate boundary join")
            left, right = rows[0][1], rows[1][1]
            a, b = left["r1_policy_snapshot"], right["r1_policy_snapshot"]
            if {a.get("player_role"), b.get("player_role")} != {"p1", "p2"}:
                raise DualTrackerProbeError("joined roles are not opposite")
            if (
                normalize_username(str(a.get("opponent_username", "")))
                != normalize_username(str(b.get("username", "")))
                or normalize_username(str(b.get("opponent_username", "")))
                != normalize_username(str(a.get("username", "")))
            ):
                raise DualTrackerProbeError("joined usernames are not reciprocal")
            p1, p2 = (left, right) if a["player_role"] == "p1" else (right, left)
            pairs.append((p1, p2))
    if not pairs:
        raise DualTrackerProbeError("no ordinary joined roots")
    return pairs, forced


_GLOBALS = (
    "weather", "weather_turns_remaining", "terrain", "terrain_turns_remaining",
    "trick_room", "trick_room_turns_remaining", "team_preview",
)


def _serialize_state(state: Any) -> str:
    value = state.to_string()
    if not isinstance(value, str):
        raise TypeError
    return value


def _side_bytes(engine: Any, side: Any, globals_: Mapping[str, Any]) -> bytes:
    # Duplicating the side makes the engine's own canonical state serializer a
    # byte-level side serializer without inspecting or pairing the other side.
    state = engine.State(side_one=side, side_two=side, **dict(globals_))
    return _serialize_state(state).encode("utf-8")


def fuse_actual_state(p1_capture: Mapping[str, Any], p2_capture: Mapping[str, Any], engine: Any) -> Any:
    """Fuse invariant local self sides into one globally oriented actual state."""
    parsed: list[list[Any]] = [[], []]
    try:
        for client, capture in enumerate((p1_capture, p2_capture)):
            for schedule in capture["schedules"]:
                for world in schedule["worlds"]:  # includes zero-weight worlds by design
                    parsed[client].append(engine.State.from_string(world["sampled_state"]))
        if not parsed[0] or not parsed[1]:
            raise ValueError
        globals_ = {field: getattr(parsed[0][0], field) for field in _GLOBALS}
        global_signature = tuple(globals_.values())
        if any(tuple(getattr(state, field) for field in _GLOBALS) != global_signature for states in parsed for state in states):
            raise ValueError
        sides = []
        for states in parsed:
            signatures = [_side_bytes(engine, state.side_one, globals_) for state in states]
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError
            sides.append(states[0].side_one)
        fused = engine.State(
            side_one=sides[0],
            side_two=sides[1],
            s1_threat=0.0,
            s2_threat=0.0,
            scout_value=0.0,
            threat_matrix=[0.0] * 36,
            wincon_matrix=[0.0] * 36,
            **globals_,
        )
        _serialize_state(fused)
        return fused
    except Exception:
        raise DualTrackerProbeError("ROOT_FUSION_REJECTED") from None


def _engine_actions(engine: Any, state: Any, side: int) -> dict[str, str]:
    options = list(engine.root_options(state=state)[side])
    canonical = [_canonical_action(value) for value in options]
    if not options or len(set(canonical)) != len(options):
        raise DualTrackerProbeError("action mapping rejected")
    return dict(zip(canonical, options))


def _snapshot_legal(snapshot: Mapping[str, Any]) -> set[str]:
    illegal = snapshot["illegal_actions"]
    return {_canonical_action(name) for name, index in snapshot["name_table"].items() if not illegal[index]}


def _projection_child(projection: Any) -> tuple[Any, Any]:
    classes = getattr(projection, "observation_classes", None)
    if classes is None:
        states = tuple(projection.next_states)
        if len(states) != 1:
            raise DualTrackerProbeError("PROJECTION_LINEAGE_REJECTED")
        return projection, states[0]
    if len(classes) != 1:
        raise DualTrackerProbeError("PROJECTION_LINEAGE_REJECTED")
    item = classes[0]
    if tuple(item.source_world_indices) != (0,) or len(tuple(item.next_states)) != 1:
        raise DualTrackerProbeError("PROJECTION_LINEAGE_REJECTED")
    return item, tuple(item.next_states)[0]


def _display_fraction(value: float, fainted: bool = False) -> float:
    if fainted or value <= 0:
        return 0.0
    displayed = math.ceil(value * 100.0 - 1e-12)
    if displayed >= 100 and value < 1.0:
        displayed = 99
    return displayed / 100.0


def canonical_public_events(projection: Any, observer: str) -> tuple[str, ...]:
    """Canonical global-p1/p2 sorted event multiset, retained only in memory."""
    item = projection.observation_classes[0] if getattr(projection, "observation_classes", None) is not None else projection
    result = []
    for event in item.events:
        row = asdict(event) if is_dataclass(event) else dict(event)
        actor = row.get("actor")
        if actor in {"self", "opponent"}:
            row["actor"] = observer if actor == "self" else ("p2" if observer == "p1" else "p1")
        if "hp_fraction" in row:
            row["hp_fraction"] = _display_fraction(float(row["hp_fraction"]), bool(row.get("fainted", False)))
        result.append(_canonical_json(row))
    return tuple(sorted(result))


def _apply_projection(tracker: Any, projection: Any, item: Any) -> Mapping[str, Any]:
    fork = tracker.fork()
    if getattr(projection, "observation_classes", None) is None:
        return fork.apply_switch_projection(projection), fork
    return fork.apply_basic_move_class(item), fork


def _legal_from_observation(observation: Mapping[str, Any]) -> set[str]:
    illegal, names = observation.get("illegal_actions"), observation.get("name_table")
    if not isinstance(illegal, list) or len(illegal) != 13 or not isinstance(names, Mapping):
        raise DualTrackerProbeError("invalid next legality")
    result = set()
    for name, index in names.items():
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 13:
            raise DualTrackerProbeError("invalid next legality")
        if not illegal[index]:
            result.add(_canonical_action(str(name)))
    return result


def _policy_valid(observation: Mapping[str, Any], policy_infer: Callable[[Mapping[str, Any]], Sequence[float]]) -> bool:
    try:
        _policy_support(observation, policy_infer(observation))
    except Exception:
        return False
    return True


def _terminal(state: Any, engine: Any) -> tuple[bool, int]:
    try:
        value = float(engine.terminal_value(state))
        if value not in {-1.0, 0.0, 1.0}:
            raise ValueError
        return value != 0, int(value)
    except Exception:
        marker = getattr(state, "terminal", None)
        winner = getattr(state, "winner", 0)
        if not isinstance(marker, bool) or winner not in {-1, 0, 1}:
            raise DualTrackerProbeError("terminal state is invalid") from None
        return marker, int(winner)


def _tracker_result(tracker: Any, observation: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    state = getattr(tracker, "state", None)
    won, lost = getattr(state, "battle_won", False), getattr(state, "battle_lost", False)
    terminal = observation.get("terminal", won or lost)
    if not all(isinstance(value, bool) for value in (terminal, won, lost)) or won and lost:
        raise DualTrackerProbeError("tracker terminal invalid")
    return terminal, won, lost


def _add(bucket: dict[str, Any], count: int, mass: float) -> None:
    bucket["count"] += count
    bucket["mass"] += mass


def probe_joined_captures(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
    tracker_factory: Callable[[Mapping[str, Any]], Any],
    policy_infer: Callable[[Mapping[str, Any]], Sequence[float]],
    rollouts: int,
    base_seed: int,
    input_hashes: Mapping[str, str],
    projection_fn: Callable[..., Any] = project_information_set_transition,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the aggregate dependency-injected dual-tracker probe."""
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts <= 0:
        raise DualTrackerProbeError("rollouts must be positive")
    counter_tape_uniform(base_seed, 0, 0, "p1")
    if not input_hashes or any(not _valid_hash(value) for value in input_hashes.values()):
        raise DualTrackerProbeError("invalid artifact hashes")
    try:
        if engine.r1_semantic_contract() != R1_SEMANTIC_CONTRACT:
            raise DualTrackerProbeError("engine semantic contract mismatch")
    except DualTrackerProbeError:
        raise
    except Exception as exc:
        raise DualTrackerProbeError("engine semantic contract unavailable") from exc
    pairs, forced = join_captures(first, second)
    pairs.sort(
        key=lambda pair: _canonical_json(
            [
                _snapshot_boundary(pair[0]),
                _public_prefix(pair[0]["r1_policy_snapshot"]),
            ]
        )
    )
    outcomes = {name: {"count": 0, "mass": 0.0} for name in OUTCOMES}
    start = clock()
    root_mass = 1.0 / len(pairs)
    for ordinal, (p1_capture, p2_capture) in enumerate(pairs):
        snapshots = (p1_capture["r1_policy_snapshot"], p2_capture["r1_policy_snapshot"])
        try:
            fused = fuse_actual_state(p1_capture, p2_capture, engine)
            root_actions = (_engine_actions(engine, fused, 0), _engine_actions(engine, fused, 1))
            if any(_snapshot_legal(snapshot) != set(actions) for snapshot, actions in zip(snapshots, root_actions)):
                raise DualTrackerProbeError("ROOT_FUSION_REJECTED")
        except Exception:
            _add(outcomes["ROOT_FUSION_REJECTED"], rollouts, root_mass)
            continue
        trackers: list[Any | None] = []
        for snapshot in snapshots:
            try:
                trackers.append(tracker_factory(player_tracker_snapshot(snapshot)))
            except Exception:
                trackers.append(None)
        tracker_code = _side_code("ROOT_TRACKER_REJECTED", trackers[0] is None, trackers[1] is None)
        if tracker_code:
            _add(outcomes[tracker_code], rollouts, root_mass)
            continue
        parity_failed = []
        for snapshot in snapshots:
            try:
                actual = [float(x) for x in policy_infer(snapshot)]
                expected = [float(x) for x in snapshot["probs"]]
                parity_failed.append(len(actual) != 13 or any(not math.isfinite(x) for x in actual) or any(abs(a - b) > ROOT_POLICY_TOLERANCE for a, b in zip(actual, expected)))
            except Exception:
                parity_failed.append(True)
        parity_code = _side_code("ROOT_POLICY_PARITY_REJECTED", *parity_failed)
        if parity_code:
            _add(outcomes[parity_code], rollouts, root_mass)
            continue
        policies = []
        policy_support_failed = []
        for snapshot in snapshots:
            try:
                policies.append(_policy_support(snapshot))
                policy_support_failed.append(False)
            except Exception:
                policies.append([])
                policy_support_failed.append(True)
        support_code = _side_code(
            "ROOT_POLICY_PARITY_REJECTED", *policy_support_failed
        )
        if support_code:
            _add(outcomes[support_code], rollouts, root_mass)
            continue
        trial_mass = root_mass / rollouts
        for rollout in range(rollouts):
            sampled = (
                _inverse_cdf(policies[0], counter_tape_uniform(base_seed, ordinal, rollout, "p1")),
                _inverse_cdf(policies[1], counter_tape_uniform(base_seed, ordinal, rollout, "p2")),
            )
            chance = counter_tape_uniform(base_seed, ordinal, rollout, "chance")
            # Sampling intentionally precedes this support query.  A captured
            # action unsupported by the actual fused engine root is a mapping
            # failure, not a conditional resample.
            try:
                raw_options = engine.root_options(state=fused)
            except Exception:
                raw_options = ((), ())
            current_actions: list[dict[str, str]] = []
            mapping_failed_list = []
            for side in range(2):
                try:
                    values = list(raw_options[side])
                    canonical = [_canonical_action(value) for value in values]
                    if not values or len(set(canonical)) != len(values):
                        raise ValueError
                    mapped = dict(zip(canonical, values))
                    current_actions.append(mapped)
                    mapping_failed_list.append(sampled[side] not in mapped)
                except Exception:
                    current_actions.append({})
                    mapping_failed_list.append(True)
            mapping_failed = tuple(mapping_failed_list)
            code = _side_code("ACTION_MAPPING_REJECTED", *mapping_failed)
            if code:
                _add(outcomes[code], 1, trial_mass)
                continue
            global_actions = (current_actions[0][sampled[0]], current_actions[1][sampled[1]])
            projections: list[Any | None] = []
            for side, tracker in zip(("SideOne", "SideTwo"), trackers):
                try:
                    projections.append(projection_fn(engine, [fused], *global_actions, chance, observer_side=side, public_opponent=tracker.public_opponent_registry()))
                except Exception:
                    projections.append(None)
            code = _side_code("PROJECTION_REJECTED", projections[0] is None, projections[1] is None)
            if code:
                _add(outcomes[code], 1, trial_mass)
                continue
            try:
                children = (_projection_child(projections[0]), _projection_child(projections[1]))
            except Exception:
                _add(outcomes["PROJECTION_LINEAGE_REJECTED"], 1, trial_mass)
                continue
            try:
                same_state = _serialize_state(children[0][1]) == _serialize_state(children[1][1])
                same_events = canonical_public_events(projections[0], "p1") == canonical_public_events(projections[1], "p2")
            except Exception:
                same_state = same_events = False
            if not same_state or not same_events:
                _add(outcomes["PUBLIC_OUTCOME_MISMATCH"], 1, trial_mass)
                continue
            observations: list[Mapping[str, Any] | None] = []
            next_trackers: list[Any | None] = []
            for tracker, projection, child in zip(trackers, projections, children):
                try:
                    observation, next_tracker = _apply_projection(tracker, projection, child[0])
                    observations.append(observation)
                    next_trackers.append(next_tracker)
                except Exception:
                    observations.append(None)
                    next_trackers.append(None)
            code = _side_code("TRACKER_REJECTED", observations[0] is None, observations[1] is None)
            if code:
                _add(outcomes[code], 1, trial_mass)
                continue
            terminal, winner = _terminal(children[0][1], engine)
            legality_failed = []
            for side, observation in enumerate(observations):
                try:
                    expected = set() if terminal else set(_engine_actions(engine, children[0][1], side))
                    legality_failed.append(_legal_from_observation(observation) != expected)
                except Exception:
                    legality_failed.append(True)
            code = _side_code("NEXT_LEGALITY_MISMATCH", *legality_failed)
            if code:
                _add(outcomes[code], 1, trial_mass)
                continue
            policy_failed = tuple(False if terminal else not _policy_valid(obs, policy_infer) for obs in observations)
            code = _side_code("NEXT_POLICY_INVALID", *policy_failed)
            if code:
                _add(outcomes[code], 1, trial_mass)
                continue
            try:
                p1_result = _tracker_result(next_trackers[0], observations[0])
                p2_result = _tracker_result(next_trackers[1], observations[1])
                terminal_ok = p1_result[0] == terminal == p2_result[0]
                if terminal and winner:
                    terminal_ok = terminal_ok and p1_result[1:] == ((winner == 1), (winner == -1)) and p2_result[1:] == ((winner == -1), (winner == 1))
                elif terminal:
                    terminal_ok = False
            except Exception:
                terminal_ok = False
            if not terminal_ok:
                _add(outcomes["TERMINAL_DISAGREEMENT"], 1, trial_mass)
            else:
                _add(outcomes["certified_terminal" if terminal else "certified_nonterminal"], 1, trial_mass)
    elapsed = max(0.0, clock() - start)
    try:
        if resource is None:
            raise RuntimeError
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.uname().sysname != "Darwin":
            peak *= 1024
    except Exception:
        peak = 0
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "record_type": "r1_dual_tracker_fused_mechanical_root_parity_probe",
        "claim_status": "private_descriptive_coverage_only",
        "estimand": ESTIMAND,
        "estimand_exclusions": ["joint_belief", "strength", "value"],
        "r1_continuation_value_allowed": False,
        "inputs": dict(sorted(input_hashes.items())),
        "configuration": {
            "rollouts_per_root": rollouts,
            "base_seed": base_seed,
            "root_weighting": "equal",
            "transition_count": 1,
            "p1_policy": POLICY_ID,
            "p2_policy": POLICY_ID,
            "root_policy_absolute_tolerance": ROOT_POLICY_TOLERANCE,
            "state_orientation": "fused_actual_own_sides_global_p1_side_one_p2_side_two_neutral_nonmechanical_annotations",
            "continuation_history_boundary": "player_observation_history_excluding_legacy_opponent_query",
        },
        "common_tape": {
            "algorithm": "sha256_counter_v1",
            "channels": ["chance", "p1", "p2"],
            "coordinate_fields": ["base_seed", "joined_root_ordinal", "rollout", "channel"],
            "uniform_mapping": "first_53_sha256_bits_divided_by_2^53",
            "action_sampling_order": "p1_then_p2_then_engine_support_test",
            "chance_draw_shared": True,
        },
        "counts": {
            "ordinary_joined_roots": len(pairs),
            "private_forced_switch_boundaries_excluded": forced,
            "rollouts_per_root": rollouts,
            "total_trials": len(pairs) * rollouts,
        },
        "outcomes": outcomes,
        "performance": {
            "wall_seconds": elapsed,
            "roots_per_second": len(pairs) / elapsed if elapsed else 0.0,
            "trials_per_second": len(pairs) * rollouts / elapsed if elapsed else 0.0,
            "peak_rss_bytes": peak,
        },
        "continuation_readiness": {
            "status": "blocked",
            "r1_continuation_value_allowed": False,
            "blockers": ["one_transition_coverage_only", "actual_state_parity_only"],
        },
        "privacy": dict(PRIVACY),
    }
    report["report_sha256"] = _sha256_bytes(_canonical_json(report).encode("ascii"))
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    required = {
        "report_schema_version", "record_type", "claim_status", "estimand", "estimand_exclusions",
        "r1_continuation_value_allowed", "inputs", "configuration", "common_tape", "counts",
        "outcomes", "performance", "continuation_readiness", "privacy", "report_sha256",
    }
    if set(report) != required or report.get("report_schema_version") != 1 or report.get("record_type") != "r1_dual_tracker_fused_mechanical_root_parity_probe" or report.get("claim_status") != "private_descriptive_coverage_only" or report.get("estimand") != ESTIMAND:
        raise DualTrackerProbeError("invalid report schema")
    if report.get("r1_continuation_value_allowed") is not False or report.get("privacy") != PRIVACY or report.get("estimand_exclusions") != ["joint_belief", "strength", "value"]:
        raise DualTrackerProbeError("invalid report declarations")
    inputs, counts, outcomes, performance = report.get("inputs"), report.get("counts"), report.get("outcomes"), report.get("performance")
    if not isinstance(inputs, Mapping) or set(inputs) != INPUT_HASH_FIELDS or any(not _valid_hash(value) for value in inputs.values()):
        raise DualTrackerProbeError("invalid report inputs")
    configuration = report.get("configuration")
    if (
        not isinstance(configuration, Mapping)
        or set(configuration) != {
            "rollouts_per_root", "base_seed", "root_weighting", "transition_count",
            "p1_policy", "p2_policy", "root_policy_absolute_tolerance",
            "state_orientation", "continuation_history_boundary",
        }
        or configuration.get("root_weighting") != "equal"
        or configuration.get("transition_count") != 1
        or configuration.get("p1_policy") != POLICY_ID
        or configuration.get("p2_policy") != POLICY_ID
        or configuration.get("root_policy_absolute_tolerance") != ROOT_POLICY_TOLERANCE
        or configuration.get("state_orientation") != "fused_actual_own_sides_global_p1_side_one_p2_side_two_neutral_nonmechanical_annotations"
        or configuration.get("continuation_history_boundary") != "player_observation_history_excluding_legacy_opponent_query"
        or isinstance(configuration.get("rollouts_per_root"), bool)
        or not isinstance(configuration.get("rollouts_per_root"), int)
        or configuration["rollouts_per_root"] <= 0
        or isinstance(configuration.get("base_seed"), bool)
        or not isinstance(configuration.get("base_seed"), int)
        or not 0 <= configuration["base_seed"] < 2**64
    ):
        raise DualTrackerProbeError("invalid report configuration")
    if report.get("common_tape") != {
        "algorithm": "sha256_counter_v1",
        "channels": ["chance", "p1", "p2"],
        "coordinate_fields": ["base_seed", "joined_root_ordinal", "rollout", "channel"],
        "uniform_mapping": "first_53_sha256_bits_divided_by_2^53",
        "action_sampling_order": "p1_then_p2_then_engine_support_test",
        "chance_draw_shared": True,
    }:
        raise DualTrackerProbeError("invalid common tape declaration")
    if not isinstance(counts, Mapping) or set(counts) != {"ordinary_joined_roots", "private_forced_switch_boundaries_excluded", "rollouts_per_root", "total_trials"} or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()) or counts["ordinary_joined_roots"] <= 0 or counts["rollouts_per_root"] <= 0 or counts["rollouts_per_root"] != configuration["rollouts_per_root"] or counts["total_trials"] != counts["ordinary_joined_roots"] * counts["rollouts_per_root"]:
        raise DualTrackerProbeError("invalid report counts")
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(OUTCOMES):
        raise DualTrackerProbeError("invalid outcome categories")
    cells = list(outcomes.values())
    if any(not isinstance(cell, Mapping) or set(cell) != {"count", "mass"} or isinstance(cell["count"], bool) or not isinstance(cell["count"], int) or cell["count"] < 0 or isinstance(cell["mass"], bool) or not isinstance(cell["mass"], (int, float)) or not math.isfinite(float(cell["mass"])) or cell["mass"] < 0 for cell in cells):
        raise DualTrackerProbeError("invalid outcome aggregates")
    if sum(cell["count"] for cell in cells) != counts["total_trials"] or not math.isclose(math.fsum(float(cell["mass"]) for cell in cells), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise DualTrackerProbeError("outcome count or mass is not conserved")
    if any(
        not math.isclose(
            float(cell["mass"]),
            cell["count"] / counts["total_trials"],
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        for cell in cells
    ):
        raise DualTrackerProbeError("outcome count and mass disagree")
    if not isinstance(performance, Mapping) or set(performance) != {"wall_seconds", "roots_per_second", "trials_per_second", "peak_rss_bytes"} or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 for value in performance.values()):
        raise DualTrackerProbeError("invalid performance aggregates")
    readiness = report.get("continuation_readiness")
    if readiness != {"status": "blocked", "r1_continuation_value_allowed": False, "blockers": ["one_transition_coverage_only", "actual_state_parity_only"]}:
        raise DualTrackerProbeError("invalid continuation declaration")
    unhashed = dict(report)
    claimed = unhashed.pop("report_sha256", None)
    if claimed != _sha256_bytes(_canonical_json(unhashed).encode("ascii")):
        raise DualTrackerProbeError("report hash does not match")
    serialized = _canonical_json(report).lower()
    if any(term in serialized for term in FORBIDDEN_REPORT_TERMS):
        raise DualTrackerProbeError("report contains private detail")


def write_report(report: Mapping[str, Any], path: Path, *, force: bool = False) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise DualTrackerProbeError("output already exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary: Path | None = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write((_canonical_json(dict(report)) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise DualTrackerProbeError("output already exists") from exc
            temporary.unlink()
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        validate_manifest(value)
    except Exception as exc:
        raise DualTrackerProbeError("invalid input manifest") from exc
    if value.get("manifest_type") != "experiment_input":
        raise DualTrackerProbeError("manifest is not an experiment input")
    return value, _sha256_bytes(payload)


def _load_capture_file(path: Path, source_hash: str) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DualTrackerProbeError("cannot read capture input") from exc
    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            validate_root_capture(row)
        except Exception as exc:
            raise DualTrackerProbeError("invalid capture input") from exc
        if row.get("schema_version") != 3 or row["configuration"].get("input_manifest_sha256") != source_hash:
            raise DualTrackerProbeError("capture source manifest linkage is invalid")
        rows.append(row)
    if not rows:
        raise DualTrackerProbeError("capture input is empty")
    return rows, _sha256_bytes(payload)


def probe_files(
    p1_path: Path, p2_path: Path, source_manifest_path: Path, analysis_manifest_path: Path,
    *, engine: Any, tracker_factory: Callable[[Mapping[str, Any]], Any],
    policy_infer: Callable[[Mapping[str, Any]], Sequence[float]], rollouts: int, base_seed: int,
) -> dict[str, Any]:
    source, source_file_hash = _load_manifest(source_manifest_path)
    analysis, analysis_file_hash = _load_manifest(analysis_manifest_path)
    first, first_hash = _load_capture_file(p1_path, str(source["manifest_sha256"]))
    second, second_hash = _load_capture_file(p2_path, str(source["manifest_sha256"]))
    if first_hash == second_hash:
        raise DualTrackerProbeError("capture files must be distinct")
    artifacts = analysis.get("artifacts", {})
    pins = {
        "engine_binding": Path(getattr(getattr(engine, "poke_engine", None), "__file__", "")),
        "checkpoint": Path(str(artifacts.get("checkpoint", {}).get("path", ""))),
        "probe_script": Path(__file__),
        "public_events_module": Path(public_events_module.__file__),
    }
    artifact_hashes = {}
    for name, runtime_path in pins.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping) or not runtime_path or Path(str(artifact.get("path", ""))).resolve() != runtime_path.resolve():
            raise DualTrackerProbeError("analysis manifest artifact binding is invalid")
        digest = sha256_file(runtime_path)
        if digest != artifact.get("sha256"):
            raise DualTrackerProbeError("analysis manifest artifact hash is invalid")
        artifact_hashes[f"{name}_sha256"] = digest
    if artifact_hashes["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise DualTrackerProbeError("checkpoint is not frozen r1")
    return probe_joined_captures(
        first, second, engine=engine, tracker_factory=tracker_factory, policy_infer=policy_infer,
        rollouts=rollouts, base_seed=base_seed,
        input_hashes={
            "p1_capture_file_sha256": first_hash, "p2_capture_file_sha256": second_hash,
            "source_manifest_sha256": str(source["manifest_sha256"]),
            "source_manifest_file_sha256": source_file_hash,
            "analysis_manifest_sha256": str(analysis["manifest_sha256"]),
            "analysis_manifest_file_sha256": analysis_file_hash, **artifact_hashes,
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-input", required=True, type=Path)
    parser.add_argument("--p2-input", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--analysis-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        base_model=pretrained.Kakuna, amago_ckpt_dir=str(WORKSPACE_ROOT / "srcs" / "models"),
        model_name="randbats_exit_r1", default_checkpoint=5,
    )
    experiment = model.initialize_agent(checkpoint=5, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device

    def infer(snapshot: Mapping[str, Any]) -> Sequence[float]:
        return infer_snapshots(agent, [dict(snapshot)], device)[0].tolist()

    report = probe_files(
        args.p1_input, args.p2_input, args.source_manifest, args.analysis_manifest,
        engine=poke_engine,
        tracker_factory=lambda snapshot: R1SwitchTracker.from_snapshot(snapshot, model.observation_space),
        policy_infer=infer, rollouts=args.rollouts, base_seed=args.base_seed,
    )
    write_report(report, args.output, force=args.force)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
