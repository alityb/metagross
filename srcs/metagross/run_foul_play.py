#!/usr/bin/env python3
"""Run Foul Play with a verified production root-prior integration."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from srcs.metagross.decision_harness import (
    CallableBelief,
    CallableController,
    CallablePolicy,
    CallableSearch,
    CallableVerifier,
    DecisionHarness,
    PolicySnapshot,
)
from srcs.metagross.holdout_metrics import compute_holdout_metrics
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    MAX_SHARED_ROOT_REPLAY_BYTES,
    MAX_WIRE_BATCH_SIZE,
    MODAL_CONTAINER_BATCH_SIZE,
    MODAL_MAX_CONTAINERS,
    REQUEST_SCHEMA,
    shared_root_result_payload,
    validate_holdout_result_payload,
    validate_loopback_search_url,
    validate_result_payload,
    validate_shared_root_result_payload,
)
from srcs.metagross.world_provenance import (
    RNG_SCHEME,
    append_ledger_row,
    derive_seed,
    deterministic_request_id,
    seeded_global_random,
    state_sha256,
)
from srcs.metagross.terminal_mcts_one_deviation import OneDeviationController
from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedgerError,
    attached_ledger,
    clear_public_protocol_lines,
    convert_battle_with_causal_ledger,
    freeze_and_attach_battle_ledger,
    protocol_lines_for_battle,
    record_public_protocol_lines,
    verify_sampled_ledgers,
    verify_sampled_move_states,
)

_PRIOR_STATE = {
    "priors": None,
    "opp_priors": None,
    "cpuct": 2.0,
    "context": None,
    "remote_search": None,
    "battle": None,
}
_REMOTE_FUNCTIONS: dict[int, object] = {}
_CAPPED_SETUP_STREAKS: dict[tuple[str, str, str], int] = {}
_CHOICE_HISTORY: dict[str, list[tuple[object, str]]] = {}
_REQUEST_CHOICE_CACHE: dict[tuple[str, int], tuple[str, str]] = {}
_POKE_ENGINE_PROVENANCE: dict[str, object] | None = None
_TERMINAL_MCTS_ONE_DEVIATION_CONTROLLER: OneDeviationController | None = None
_HOLDOUT_DECISION_SEQUENCE = 0
_CAUSAL_RECEIPT_CONTEXT: dict[str, object] | None = None
_CAUSAL_RECEIPT_CONTEXT_LOCK = threading.Lock()
_CYCLE25_EXECUTION_BOUNDARIES: dict[str, dict[str, object]] = {}
REMOTE_MCTS_SCHEMA = REQUEST_SCHEMA
REMOTE_ENGINE_CONTRACT = ENGINE_CONTRACT
MAX_REMOTE_RESPONSE_BYTES = 16_000_000
MAX_CONSECUTIVE_CAPPED_SWORDS_DANCE = 1
MIN_PAIRED_POSTERIOR_COVERAGE = 1.0
MIN_PAIRED_ADVANTAGE = 0.10
PAIRED_CONFIDENCE_Z = 1.96
ENABLE_ADAPTIVE_SEARCH_OVERRIDES = False
MAX_TRACKED_BATTLES = 128
HOLDOUT_ROLLOUTS = 64
HOLDOUT_CONTINUATION_ITERATIONS = 64
HOLDOUT_CONTINUATION_STEPS = 1
HOLDOUT_CONTINUATION_HORIZONS = (1, 2)
HOLDOUT_CANDIDATE_COUNT = 3
HOLDOUT_MIN_EFFECTIVE_WORLDS = 12.0
HOLDOUT_MIN_PAIRS = 512
HOLDOUT_MIN_MEAN_ADVANTAGE = 0.08
HOLDOUT_MIN_LOWER_BOUND = 0.02
HOLDOUT_MIN_POSITIVE_WORLD_WEIGHT = 0.75
HOLDOUT_MAX_CATASTROPHIC_RATE = 0.005
HOLDOUT_MIN_SIGN_MARGIN = 0.10
HOLDOUT_ALPHA_BUDGET = 0.001
HOLDOUT_CVAR_TAIL_MASS = 0.10
HOLDOUT_MAX_CATASTROPHE_RATE_GAP = 0.005
HOLDOUT_MAX_CATASTROPHE_SEVERITY_GAP = 0.05
HOLDOUT_MIN_EVALUATOR_DELTA_DIFFERENCE = -0.02
HOLDOUT_ALPHA_CHECKS_PER_LOOK = 4
HOLDOUT_OPPONENT_UNIFORM_MIX = 0.25
CONTROLLER_MODES = ("certified", "search_first")
DEFAULT_CONTROLLER_MODE = "search_first"
ROOT_SEARCH_MODES = ("independent_mcts", "independent_ensemble", "shared_rm_plus")
DEFAULT_ROOT_SEARCH_MODE = "independent_mcts"
DEFAULT_SHARED_ROOT_ITERATIONS = 10_000
DEFAULT_SHARED_ROOT_CONTINUATION_ITERATIONS = 8
DEFAULT_SHARED_ROOT_PRIOR_STRENGTH = 1.0

_PURE_BOOST_MOVES = {
    "acidarmor": ("defense",),
    "agility": ("speed",),
    "amnesia": ("special-defense",),
    "barrier": ("defense",),
    "bellydrum": ("attack",),
    "bulkup": ("attack", "defense"),
    "calmmind": ("special-attack", "special-defense"),
    "coil": ("attack", "defense", "accuracy"),
    "cosmicpower": ("defense", "special-defense"),
    "cottonguard": ("defense",),
    "defendorder": ("defense", "special-defense"),
    "doubleteam": ("evasion",),
    "dragondance": ("attack", "speed"),
    "growth": ("attack", "special-attack"),
    "honeclaws": ("attack", "accuracy"),
    "howl": ("attack",),
    "irondefense": ("defense",),
    "meditate": ("attack",),
    "nastyplot": ("special-attack",),
    "quiverdance": ("special-attack", "special-defense", "speed"),
    "rockpolish": ("speed",),
    "sharpen": ("attack",),
    "shiftgear": ("attack", "speed"),
    "swordsdance": ("attack",),
    "tailglow": ("special-attack",),
    "victorydance": ("attack", "defense", "speed"),
    "workup": ("attack", "special-attack"),
}
_KNOWN_REFLECTABLE_MOVES = {
    "spikes",
    "stealthrock",
    "stickyweb",
    "toxic",
    "toxicspikes",
}
_ABILITY_TYPE_IMMUNITIES = {
    "dryskin": "water",
    "eartheater": "ground",
    "flashfire": "fire",
    "levitate": "ground",
    "lightningrod": "electric",
    "motordrive": "electric",
    "sapsipper": "grass",
    "stormdrain": "water",
    "voltabsorb": "electric",
    "waterabsorb": "water",
    "wellbakedbody": "fire",
}
_ABILITY_FLAG_IMMUNITIES = {
    "bulletproof": "bullet",
    "soundproof": "sound",
    "windrider": "wind",
}
_MOLD_BREAKER_ABILITIES = {"moldbreaker", "teravolt", "turboblaze"}
_CURRENT_TARGETS = {
    "adjacentfoe",
    "all",
    "alladjacent",
    "alladjacentfoes",
    "any",
    "normal",
    "randomnormal",
}
_PROTECT_MOVES = {
    "banefulbunker",
    "burningbulwark",
    "detect",
    "kingsshield",
    "obstruct",
    "protect",
    "silktrap",
    "spikyshield",
}
_RECOVERY_MOVES = {
    "healorder",
    "lifedew",
    "milkdrink",
    "moonlight",
    "morningsun",
    "recover",
    "rest",
    "roost",
    "shoreup",
    "slackoff",
    "softboiled",
    "strengthsap",
    "synthesis",
}
_PIVOT_MOVES = {
    "batonpass",
    "chillyreception",
    "flipturn",
    "partingshot",
    "shedtail",
    "teleport",
    "uturn",
    "voltswitch",
}
_NON_DAMAGE_TERMINAL_PIVOTS = {"batonpass", "shedtail", "teleport"}


def _append_jsonl(environment_variable: str, row: dict) -> None:
    path = os.environ.get(environment_variable)
    if not path:
        return
    payload = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    if environment_variable == "METAGROSS_SEARCH_DUMP":
        os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_shared_root_replay_envelope(
    *,
    states: list[str],
    source_weights: list[float],
    normalized_weights: list[float],
    iterations: int,
    continuation_iterations: int,
    solver_seed: int,
    action_seed: int,
    result: dict[str, object],
    remote_search: dict[str, object],
    request_actions: set[str],
) -> dict[str, object]:
    if not (
        states
        and len(states) == len(source_weights) == len(normalized_weights)
        and isinstance(result.get("replay_capture"), dict)
    ):
        raise RuntimeError("shared-root replay inputs are incomplete")
    source_weight_sum = math.fsum(source_weights)
    if not math.isfinite(source_weight_sum) or source_weight_sum <= 0:
        raise RuntimeError("shared-root replay source weights are invalid")
    opponent_prior = [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])] or None
    provenance = remote_search.get("engine")
    if not isinstance(provenance, dict):
        provenance = _POKE_ENGINE_PROVENANCE
    if not isinstance(provenance, dict):
        raise RuntimeError("shared-root replay has no engine provenance")
    native_sha256 = provenance.get("native_sha256")
    if not isinstance(native_sha256, str) or len(native_sha256) != 64:
        raise RuntimeError("shared-root replay has invalid native provenance")
    engine = {
        "contract": provenance.get("contract", ENGINE_CONTRACT),
        "source_sha256": provenance.get("source_sha256", ENGINE_SOURCE_SHA256),
        "native_sha256": native_sha256,
        "distribution_version": provenance.get("distribution_version"),
    }
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("shared-root replay result has no diagnostics")
    policy = result.get("policy")
    if not isinstance(policy, list) or not request_actions:
        raise RuntimeError("shared-root replay has no action support")
    action_aliases = []
    mapped_actions = set()
    for row in policy:
        native_action = row.get("action") if isinstance(row, dict) else None
        request_action = (
            _authorized_action_name(native_action, request_actions)
            if isinstance(native_action, str)
            else None
        )
        if request_action is None or request_action in mapped_actions:
            raise RuntimeError("shared-root replay action mapping is invalid")
        mapped_actions.add(request_action)
        action_aliases.append(
            {"native_action": native_action, "request_action": request_action}
        )
    envelope = {
        "schema_version": 1,
        "capture_kind": "production_shared_root_replay_v1",
        "source_particles": [
            {
                "source_index": index,
                "serialized_state": state,
                "state_sha256": state_sha256(state),
                "source_weight": float(source_weight),
                "normalized_weight": float(normalized_weight),
            }
            for index, (state, source_weight, normalized_weight) in enumerate(
                zip(states, source_weights, normalized_weights, strict=True)
            )
        ],
        "source_weight_sum": source_weight_sum,
        "solver": {
            "iterations": iterations,
            "continuation_iterations": continuation_iterations,
            "seed": solver_seed,
            "prior_strength": diagnostics["prior_strength"],
            "s1_prior": [list(row) for row in (_PRIOR_STATE["priors"] or [])] or None,
            "s2_priors": [opponent_prior for _state in states],
        },
        "sampling": {
            "world_channel": "selection-worlds",
            "world_seed": remote_search.get("sampling_seed"),
            "action_channel": "shared-root-action",
            "action_seed": action_seed,
        },
        "request_ids": list(remote_search.get("request_ids") or []),
        "request_action_support": sorted(request_actions),
        "action_aliases": action_aliases,
        "engine": engine,
        "native_capture_sha256": _canonical_sha256(result["replay_capture"]),
    }
    if len(
        json.dumps(envelope, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ) > MAX_SHARED_ROOT_REPLAY_BYTES:
        raise RuntimeError("shared-root replay envelope exceeds the encoded size bound")
    envelope["capture_sha256"] = _canonical_sha256(envelope)
    return envelope


def _decision_coordinates(*, required: bool = False) -> tuple[str, int] | None:
    context = _PRIOR_STATE.get("context") or {}
    tag = context.get("tag")
    decision_index = context.get("decision_idx")
    valid = (
        isinstance(tag, str)
        and bool(tag)
        and isinstance(decision_index, int)
        and not isinstance(decision_index, bool)
        and decision_index >= 0
    )
    if not valid:
        if required:
            raise RuntimeError("deterministic remote execution requires decision context")
        return None
    return tag, decision_index


def battle_request_identity(battle) -> tuple[tuple[str, int], str]:
    tag = getattr(battle, "battle_tag", None)
    rqid = getattr(battle, "rqid", None)
    request_json = getattr(battle, "request_json", None)
    if (
        not isinstance(tag, str)
        or not tag
        or isinstance(rqid, bool)
        or not isinstance(rqid, int)
        or rqid < 0
        or not isinstance(request_json, dict)
    ):
        raise RuntimeError("battle has no valid request identity")
    fingerprint = hashlib.sha256(
        json.dumps(request_json, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (tag, rqid), fingerprint


def _run_seed(*, required: bool = False) -> str | None:
    run_seed = os.environ.get("METAGROSS_RUN_SEED")
    rng_scheme = os.environ.get("METAGROSS_RNG_SCHEME")
    if run_seed and rng_scheme == RNG_SCHEME:
        return run_seed
    if required:
        raise RuntimeError("deterministic remote execution requires the frozen run seed")
    return None


def _derived_seed(channel: str, cohort: str | int, *, required: bool = False) -> int | None:
    coordinates = _decision_coordinates(required=required)
    run_seed = _run_seed(required=required)
    if coordinates is None or run_seed is None:
        return None
    tag, decision_index = coordinates
    return derive_seed(run_seed, channel, tag, decision_index, cohort)


def _deterministic_request_id(channel: str, cohort: str | int) -> str:
    coordinates = _decision_coordinates(required=True)
    run_seed = _run_seed(required=True)
    assert coordinates is not None and run_seed is not None
    tag, decision_index = coordinates
    return deterministic_request_id(
        run_seed, tag, decision_index, cohort, channel=channel
    )


def holdout_alpha(sequence_index: int, candidate_rank: int, horizon_index: int) -> float:
    """Allocate the preregistered family-wise budget to one certification look."""
    if (
        isinstance(sequence_index, bool)
        or not isinstance(sequence_index, int)
        or sequence_index < 0
        or isinstance(candidate_rank, bool)
        or not isinstance(candidate_rank, int)
        or not 1 <= candidate_rank <= HOLDOUT_CANDIDATE_COUNT
        or isinstance(horizon_index, bool)
        or not isinstance(horizon_index, int)
        or not 0 <= horizon_index < len(HOLDOUT_CONTINUATION_HORIZONS)
    ):
        raise ValueError("invalid holdout alpha coordinates")
    decision_number = sequence_index + 1
    decision_budget = (
        HOLDOUT_ALPHA_BUDGET
        * 6.0
        / (math.pi * math.pi)
        / (decision_number * decision_number)
    )
    family_size = (
        HOLDOUT_CANDIDATE_COUNT
        * len(HOLDOUT_CONTINUATION_HORIZONS)
        * HOLDOUT_ALPHA_CHECKS_PER_LOOK
    )
    return decision_budget / family_size


def _mcts_result_payload(result) -> dict:
    def side_payload(options):
        return [
            {
                "move_choice": option.move_choice,
                "total_score": float(option.total_score),
                "visits": int(option.visits),
            }
            for option in options
        ]

    return {
        "side_one": side_payload(result.side_one),
        "side_two": side_payload(result.side_two),
        "total_visits": int(result.total_visits),
    }


def _best_mcts_alternative_by_score(mcts_results, blocked_choice: str) -> str | None:
    blocked = blocked_choice.removesuffix("-tera")
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    for result, sample_chance, _index in mcts_results:
        weight = float(sample_chance)
        if not math.isfinite(weight) or weight <= 0:
            continue
        for option in result.side_one:
            if (
                option.move_choice.removesuffix("-tera") == blocked
                or option.visits <= 0
            ):
                continue
            mean_score = float(option.total_score) / option.visits
            if not math.isfinite(mean_score):
                continue
            scores[option.move_choice] = (
                scores.get(option.move_choice, 0.0) + weight * mean_score
            )
            weights[option.move_choice] = weights.get(option.move_choice, 0.0) + weight
    if not scores:
        return None
    return max(scores, key=lambda move: (scores[move] / weights[move], move))


def guard_repeated_capped_swords_dance(
    battle, choice: str, mcts_results, streaks=None
) -> tuple[str, dict[str, object] | None]:
    """Permit one Sucker Punch dodge at +6, then prevent a repeated no-op."""
    if streaks is None:
        streaks = _CAPPED_SETUP_STREAKS
    tag = str(getattr(battle, "battle_tag", ""))
    active = getattr(getattr(battle, "user", None), "active", None)
    active_name = str(getattr(active, "name", ""))
    key = (tag, active_name, "swordsdance")
    try:
        attack_boost = float(
            _mapping_value(getattr(active, "boosts", {}) or {}, "attack", 0)
        )
    except (TypeError, ValueError):
        attack_boost = 0
    is_capped = choice.removesuffix("-tera") == "swordsdance" and attack_boost >= 6
    if not is_capped:
        for existing in [candidate for candidate in streaks if candidate[0] == tag]:
            streaks.pop(existing, None)
        return choice, None

    streaks[key] = streaks.get(key, 0) + 1
    if streaks[key] <= MAX_CONSECUTIVE_CAPPED_SWORDS_DANCE:
        return choice, None

    alternative = _best_mcts_alternative_by_score(mcts_results, choice)
    if alternative is None:
        return choice, None
    streak = streaks.pop(key)
    return alternative, {
        "reason": "repeated_capped_swords_dance",
        "original_choice": choice,
        "replacement_choice": alternative,
        "capped_streak": streak,
    }


def _mcts_actions_and_visit_mass(mcts_results) -> tuple[list[str], dict[str, float]]:
    actions: set[str] = set()
    visit_mass: dict[str, float] = {}
    for row in mcts_results:
        try:
            result, sample_chance, _index = row
            weight = float(sample_chance)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight) or weight <= 0:
            continue
        options = list(getattr(result, "side_one", ()) or ())
        valid_options = []
        for option in options:
            action = getattr(option, "move_choice", None)
            if not isinstance(action, str) or not action:
                continue
            actions.add(action)
            try:
                visits = int(getattr(option, "visits", 0))
            except (TypeError, ValueError):
                visits = 0
            valid_options.append((action, max(0, visits)))
        if not valid_options:
            continue
        try:
            total_visits = int(getattr(result, "total_visits", 0))
        except (TypeError, ValueError):
            total_visits = 0
        if total_visits <= 0:
            total_visits = sum(visits for _action, visits in valid_options)
        for action, visits in valid_options:
            share = (
                visits / total_visits if total_visits > 0 else 1.0 / len(valid_options)
            )
            visit_mass[action] = visit_mass.get(action, 0.0) + weight * share
    ordered = sorted(actions, key=lambda action: (-visit_mass.get(action, 0.0), action))
    return ordered, visit_mass


def _shared_root_actions_and_probability_mass(
    policy: object, request_actions: set[str] | None
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    if not isinstance(policy, list) or not policy:
        raise RuntimeError("shared-root policy is empty")
    probability_mass: dict[str, float] = {}
    counterfactual_values: dict[str, float] = {}
    for row in policy:
        if not isinstance(row, dict):
            raise RuntimeError("shared-root policy entry is invalid")
        action = row.get("action")
        probability = row.get("probability")
        value = row.get("counterfactual_value")
        if (
            not isinstance(action, str)
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError("shared-root policy entry is invalid")
        authorized = (
            action
            if request_actions is None
            else _authorized_action_name(action, request_actions)
        )
        if authorized is None:
            if probability > 0:
                raise RuntimeError(
                    f"shared-root policy assigns mass to request-illegal action: {action}"
                )
            continue
        if authorized in probability_mass:
            raise RuntimeError("shared-root policy aliases duplicate request actions")
        probability_mass[authorized] = float(probability)
        counterfactual_values[authorized] = float(value)
    if not probability_mass or abs(math.fsum(probability_mass.values()) - 1.0) > 1e-8:
        raise RuntimeError("shared-root request-valid policy is not normalized")
    ordered = sorted(
        probability_mass,
        key=lambda action: (
            -probability_mass[action],
            -counterfactual_values[action],
            action,
        ),
    )
    return ordered, probability_mass, counterfactual_values


def _sample_shared_root_action(
    ordered: list[str], probability_mass: dict[str, float], seed: int
) -> tuple[str, float]:
    draw = random.Random(seed).random()
    cumulative = 0.0
    sampled = ordered[-1]
    for action in ordered:
        cumulative += probability_mass[action]
        if draw < cumulative:
            sampled = action
            break
    return sampled, draw


def _authorized_action_name(action: str, allowed: set[str]) -> str | None:
    if action in allowed:
        return action
    if not action.startswith("switch "):
        return None
    requested = "".join(character for character in action[7:] if character.isalnum())

    def switch_family(name: str) -> str:
        # poke-engine serializes every Minior shell colour as ``miniormeteor``;
        # Showdown's private request retains the cosmetic colour.  They are the
        # same switch slot and battle state, not interchangeable species.
        if name.startswith("minior"):
            return "minior"
        return name

    candidates = []
    for candidate in allowed:
        if not candidate.startswith("switch "):
            continue
        authorized = "".join(
            character for character in candidate[7:] if character.isalnum()
        )
        if (
            requested
            and authorized
            and (
                requested.startswith(authorized)
                or authorized.startswith(requested)
                or switch_family(requested) == switch_family(authorized)
            )
        ):
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def derive_policy_baseline(
    mcts_results, priors=None, request_actions: set[str] | None = None
) -> tuple[str, list[str], dict[str, float]]:
    """Choose the highest request-valid player-prior action."""
    ordered, visit_mass = _mcts_actions_and_visit_mass(mcts_results)
    prior_by_action: dict[str, float] = {}
    for row in priors or ():
        try:
            action, probability = row
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if isinstance(action, str) and math.isfinite(probability):
            prior_by_action[action] = probability
    allowed = None if request_actions is None else set(request_actions)
    if allowed == set():
        return "no move", ordered, visit_mass
    authorized_priors: dict[str, float] = {}
    for action, probability in prior_by_action.items():
        authorized = (
            action if allowed is None else _authorized_action_name(action, allowed)
        )
        if authorized is not None:
            authorized_priors[authorized] = max(
                authorized_priors.get(authorized, 0.0), probability
            )
    prior_actions = list(authorized_priors)
    if prior_actions:
        baseline = min(
            prior_actions,
            key=lambda action: (-authorized_priors[action], action),
        )
    else:
        authorized_search = [
            action for action in ordered if allowed is None or action in allowed
        ]
        baseline = authorized_search[0] if authorized_search else "no move"
    return baseline, ordered, visit_mass


def request_player_actions(battle) -> set[str] | None:
    """Build exact own-side actions from the private Showdown request."""
    request = getattr(battle, "request_json", None)
    rqid = getattr(battle, "rqid", None)
    if request is None and rqid is None:
        # Standalone controller fixtures and offline replay callers predate raw
        # request retention. Live play is request-bound in propose() below.
        return _legacy_clone_player_actions(battle)
    if not isinstance(request, dict) or request.get("rqid") != rqid:
        raise RuntimeError("battle has no correlated private request")
    force_switch_rows = request.get("forceSwitch", [False])
    if not isinstance(force_switch_rows, list) or not force_switch_rows:
        raise RuntimeError("battle request has invalid forceSwitch metadata")
    force_switch = force_switch_rows[0]
    if not isinstance(force_switch, bool):
        raise RuntimeError("battle request has invalid forceSwitch metadata")
    active_rows = request.get("active", [])
    if active_rows is None:
        active_rows = []
    if not isinstance(active_rows, list):
        raise RuntimeError("battle request has invalid active metadata")
    active = active_rows[0] if active_rows else {}
    if not isinstance(active, dict):
        raise RuntimeError("battle request has invalid active metadata")
    actions: set[str] = set()
    if not force_switch:
        can_tera = active.get("canTerastallize", False)
        if can_tera is None:
            can_tera = False
        if not isinstance(can_tera, (bool, str)):
            raise RuntimeError("battle request has invalid Tera metadata")
        moves = active.get("moves", [])
        if not isinstance(moves, list):
            raise RuntimeError("battle request has invalid move metadata")
        for move in moves:
            if not isinstance(move, dict):
                raise RuntimeError("battle request has invalid move metadata")
            disabled = move.get("disabled", False)
            pp = move.get("pp")
            if not isinstance(disabled, bool):
                raise RuntimeError("battle request has invalid disabled metadata")
            if isinstance(pp, bool) or (pp is not None and not isinstance(pp, int)):
                raise RuntimeError("battle request has invalid PP metadata")
            if disabled or pp == 0:
                continue
            name = _normalize_identifier(move.get("id", ""))
            if not name:
                continue
            actions.add(name)
            if can_tera:
                actions.add(f"{name}-tera")
    trapped = active.get("trapped", False)
    if not isinstance(trapped, bool):
        raise RuntimeError("battle request has invalid trapped metadata")
    side = request.get("side")
    if not isinstance(side, dict) or not isinstance(side.get("pokemon"), list):
        raise RuntimeError("battle request has invalid side metadata")
    if force_switch or not trapped:
        for pokemon in side["pokemon"]:
            if not isinstance(pokemon, dict):
                raise RuntimeError("battle request has invalid Pokemon metadata")
            if pokemon.get("active") is True:
                continue
            condition = pokemon.get("condition")
            details = pokemon.get("details")
            if not isinstance(condition, str) or not isinstance(details, str):
                raise RuntimeError("battle request has invalid Pokemon metadata")
            hp_text = condition.split(" ", 1)[0].split("/", 1)[0]
            if condition.endswith(" fnt") or hp_text == "0":
                continue
            target = _normalize_identifier(details.split(",", 1)[0])
            if target:
                actions.add(f"switch {target}")
    if not actions:
        raise RuntimeError("battle request contains no legal actions")
    return actions


def _legacy_clone_player_actions(battle) -> set[str] | None:
    user = getattr(battle, "user", None)
    active = getattr(user, "active", None)
    if active is None or not hasattr(active, "moves"):
        return None
    actions: set[str] = set()
    force_switch = bool(getattr(battle, "force_switch", False))
    if not force_switch:
        can_tera = bool(getattr(active, "can_terastallize", False))
        for move in getattr(active, "moves", ()) or ():
            if bool(getattr(move, "disabled", False)):
                continue
            name = str(getattr(move, "name", getattr(move, "id", "")) or "")
            if name:
                actions.add(name)
                if can_tera:
                    actions.add(f"{name}-tera")
    if force_switch or not bool(getattr(user, "trapped", False)):
        for pokemon in getattr(user, "reserve", ()) or ():
            try:
                alive = float(getattr(pokemon, "hp", 0)) > 0
            except (TypeError, ValueError):
                alive = False
            if alive:
                actions.add(f"switch {getattr(pokemon, 'name', '')}")
    return actions


def _request_allows(action: str, request_actions: set[str] | None) -> bool:
    return request_actions is None or action in request_actions


def paired_candidate_evidence(
    mcts_results,
    candidate: str,
    baseline: str,
    minimum_coverage: float = MIN_PAIRED_POSTERIOR_COVERAGE,
) -> dict[str, object]:
    """Measure paired candidate advantage without renormalizing covered worlds."""
    worlds = []
    total_weight = 0.0
    for row in mcts_results:
        try:
            result, sample_chance, index = row
            weight = float(sample_chance)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight) or weight <= 0:
            continue
        total_weight += weight
        by_action = {
            getattr(option, "move_choice", None): option
            for option in (getattr(result, "side_one", ()) or ())
        }
        candidate_option = by_action.get(candidate)
        baseline_option = by_action.get(baseline)
        if candidate_option is None or baseline_option is None:
            continue
        try:
            candidate_visits = int(getattr(candidate_option, "visits", 0))
            baseline_visits = int(getattr(baseline_option, "visits", 0))
            candidate_mean = float(candidate_option.total_score) / candidate_visits
            baseline_mean = float(baseline_option.total_score) / baseline_visits
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            continue
        if (
            candidate_visits <= 0
            or baseline_visits <= 0
            or not math.isfinite(candidate_mean)
            or not math.isfinite(baseline_mean)
        ):
            continue
        try:
            world_index = int(index)
        except (TypeError, ValueError):
            world_index = repr(index)
        delta = candidate_mean - baseline_mean
        standard_error_bound = math.sqrt(1.0 / candidate_visits + 1.0 / baseline_visits)
        worlds.append(
            {
                "index": world_index,
                "weight": weight,
                "delta": delta,
                "lower_bound": delta - PAIRED_CONFIDENCE_Z * standard_error_bound,
                "candidate_visits": candidate_visits,
                "baseline_visits": baseline_visits,
            }
        )
    paired_weight = sum(float(world["weight"]) for world in worlds)
    coverage = paired_weight / total_weight if total_weight > 0 else 0.0
    posterior_delta = (
        sum(float(world["weight"]) * float(world["delta"]) for world in worlds)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    lower_bound = (
        sum(float(world["weight"]) * float(world["lower_bound"]) for world in worlds)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    complete = total_weight > 0 and coverage >= minimum_coverage
    heuristic_qualified = complete and lower_bound >= MIN_PAIRED_ADVANTAGE
    return {
        "candidate": candidate,
        "baseline": baseline,
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "paired_posterior_weight": paired_weight,
        "total_posterior_weight": total_weight,
        "paired_worlds": len(worlds),
        "posterior_delta": posterior_delta,
        "paired_lower_confidence_bound": lower_bound,
        "complete": complete,
        "minimum_advantage": MIN_PAIRED_ADVANTAGE,
        "evidence_kind": "adaptive_mcts_heuristic",
        "heuristic_qualified": heuristic_qualified,
        "qualified": ENABLE_ADAPTIVE_SEARCH_OVERRIDES and heuristic_qualified,
        "worlds": worlds,
    }


_LEGACY_HOLDOUT_FIELDS = {
    "pairs",
    "baseline_sum",
    "candidate_sum",
    "delta_sum",
    "delta_squared_sum",
    "catastrophic_count",
    "candidate_better_count",
    "baseline_better_count",
    "equal_count",
    "baseline_terminal_count",
    "candidate_terminal_count",
    "continuation_iterations_executed",
}


def _validate_legacy_holdout_result_payload(raw: object) -> dict[str, object]:
    """Validate only the aggregate fields available in immutable v4 captures."""
    if not isinstance(raw, dict) or set(raw) != _LEGACY_HOLDOUT_FIELDS:
        raise ValueError("legacy holdout result has invalid fields")
    pairs = raw["pairs"]
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs <= 0:
        raise ValueError("legacy holdout result has invalid pairs")
    numeric = {}
    for name in ("baseline_sum", "candidate_sum", "delta_sum", "delta_squared_sum"):
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"legacy holdout result has invalid {name}")
        numeric[name] = float(value)
    for name in _LEGACY_HOLDOUT_FIELDS - set(numeric):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"legacy holdout result has invalid {name}")
    tolerance = 1e-8 * max(1, pairs)
    if (
        abs(numeric["candidate_sum"] - numeric["baseline_sum"] - numeric["delta_sum"])
        > tolerance
        or numeric["delta_squared_sum"] + tolerance
        < numeric["delta_sum"] ** 2 / pairs
        or raw["candidate_better_count"]
        + raw["baseline_better_count"]
        + raw["equal_count"]
        != pairs
    ):
        raise ValueError("legacy holdout result is inconsistent")
    return {**raw, **numeric}


def independent_holdout_certificate(
    results,
    world_weights,
    candidate: str,
    baseline: str,
    decision_index: int = 0,
) -> dict[str, object]:
    """Apply a one-candidate, fresh-world admission rule to paired aggregates."""
    if len(results) != len(world_weights) or not results:
        raise ValueError("holdout results must exactly cover their fresh worlds")
    worlds = []
    total_weight = 0.0
    for index, (raw, raw_weight) in enumerate(zip(results, world_weights, strict=True)):
        if isinstance(raw, dict) and set(raw) == _LEGACY_HOLDOUT_FIELDS:
            payload = _validate_legacy_holdout_result_payload(raw)
        else:
            payload = validate_holdout_result_payload(raw)
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("holdout world weights must be positive and finite")
        pairs = payload["pairs"]
        delta_mean = payload["delta_sum"] / pairs
        delta_second_moment = payload["delta_squared_sum"] / pairs
        within_variance = max(0.0, delta_second_moment - delta_mean * delta_mean)
        worlds.append(
            {
                "index": index,
                "weight": weight,
                "pairs": pairs,
                "delta_mean": delta_mean,
                "within_variance": within_variance,
                "catastrophic_count": payload["catastrophic_count"],
                "candidate_better_count": payload["candidate_better_count"],
                "baseline_better_count": payload["baseline_better_count"],
                "equal_count": payload["equal_count"],
                "continuation_iterations_executed": payload[
                    "continuation_iterations_executed"
                ],
            }
        )
        total_weight += weight

    normalized = [world["weight"] / total_weight for world in worlds]
    sum_squared_weights = math.fsum(weight * weight for weight in normalized)
    effective_worlds = 1.0 / sum_squared_weights
    mean = math.fsum(
        weight * world["delta_mean"]
        for weight, world in zip(normalized, worlds, strict=True)
    )
    weighted_variance = math.fsum(
        weight * (world["delta_mean"] - mean) ** 2
        for weight, world in zip(normalized, worlds, strict=True)
    )
    correction = max(1e-12, 1.0 - sum_squared_weights)
    between_variance = weighted_variance / correction
    within_mean_variance = math.fsum(
        weight * weight * world["within_variance"] / world["pairs"]
        for weight, world in zip(normalized, worlds, strict=True)
    )
    standard_error = math.sqrt(
        max(0.0, between_variance * sum_squared_weights + within_mean_variance)
    )
    try:
        decision_number = max(1, int(decision_index) + 1)
    except (TypeError, ValueError):
        decision_number = 1
    alpha = (
        HOLDOUT_ALPHA_BUDGET
        * 6.0
        / (math.pi * math.pi)
        / (decision_number * decision_number)
    )
    z_value = statistics.NormalDist().inv_cdf(1.0 - alpha)
    lower_bound = mean - z_value * standard_error
    total_pairs = sum(world["pairs"] for world in worlds)
    catastrophic_rate = (
        sum(world["catastrophic_count"] for world in worlds) / total_pairs
    )
    candidate_better_rate = (
        sum(world["candidate_better_count"] for world in worlds) / total_pairs
    )
    baseline_better_rate = (
        sum(world["baseline_better_count"] for world in worlds) / total_pairs
    )
    positive_world_weight = math.fsum(
        weight
        for weight, world in zip(normalized, worlds, strict=True)
        if world["delta_mean"] > 0
    )
    positive_worlds = sum(world["delta_mean"] > 0 for world in worlds)
    sign_p_value = math.fsum(
        math.comb(len(worlds), successes)
        for successes in range(positive_worlds, len(worlds) + 1)
    ) / (2 ** len(worlds))
    checks = {
        "effective_worlds": effective_worlds >= HOLDOUT_MIN_EFFECTIVE_WORLDS,
        "pairs": total_pairs >= HOLDOUT_MIN_PAIRS,
        "mean_advantage": mean >= HOLDOUT_MIN_MEAN_ADVANTAGE,
        "lower_bound": lower_bound >= HOLDOUT_MIN_LOWER_BOUND,
        "positive_world_weight": (
            positive_world_weight >= HOLDOUT_MIN_POSITIVE_WORLD_WEIGHT
        ),
        "world_sign_test": sign_p_value <= alpha,
        "catastrophic_rate": catastrophic_rate <= HOLDOUT_MAX_CATASTROPHIC_RATE,
        "sign_margin": (
            candidate_better_rate - baseline_better_rate >= HOLDOUT_MIN_SIGN_MARGIN
        ),
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "evidence_kind": "independent_seeded_paired_holdout",
        "fresh_worlds": len(worlds),
        "effective_worlds": effective_worlds,
        "pairs": total_pairs,
        "posterior_delta": mean,
        "standard_error": standard_error,
        "alpha": alpha,
        "z_value": z_value,
        "paired_lower_confidence_bound": lower_bound,
        "positive_world_weight": positive_world_weight,
        "positive_worlds": positive_worlds,
        "sign_p_value": sign_p_value,
        "catastrophic_rate": catastrophic_rate,
        "candidate_better_rate": candidate_better_rate,
        "baseline_better_rate": baseline_better_rate,
        "coverage": 1.0,
        "complete": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "worlds": worlds,
    }


def robust_holdout_certificate(
    results,
    world_weights,
    state_hashes,
    cluster_hashes,
    candidate: str,
    baseline: str,
    alpha_sequence_index: int,
    candidate_rank: int,
    horizon_index: int,
) -> dict[str, object]:
    """Apply the preregistered symmetric, cluster-aware v5 admission rule."""
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("holdout candidate must be a nonempty action")
    if not isinstance(baseline, str) or not baseline or candidate == baseline:
        raise ValueError("holdout baseline must be a different nonempty action")
    validated_results = [validate_holdout_result_payload(row) for row in results]
    alpha = holdout_alpha(alpha_sequence_index, candidate_rank, horizon_index)
    metrics = compute_holdout_metrics(
        validated_results,
        world_weights,
        state_hashes,
        cluster_hashes,
        alpha=alpha,
        cvar_tail_mass=HOLDOUT_CVAR_TAIL_MASS,
    )
    z_value = statistics.NormalDist().inv_cdf(1.0 - alpha)
    lower_bound = float(metrics["weighted_mean_delta"]) - z_value * float(
        metrics["standard_error"]
    )
    active_clusters = [
        cluster
        for cluster in metrics["cluster_aggregates"]
        if float(cluster["normalized_weight"]) > 0.0
    ]
    positive_clusters = sum(
        float(cluster["delta_mean"]) > 0.0 for cluster in active_clusters
    )
    sign_p_value = math.fsum(
        math.comb(len(active_clusters), successes)
        for successes in range(positive_clusters, len(active_clusters) + 1)
    ) / (2 ** len(active_clusters))
    candidate_catastrophe_rate = float(metrics["candidate_catastrophe_rate"])
    baseline_catastrophe_rate = float(metrics["baseline_catastrophe_rate"])
    candidate_severity = float(metrics["candidate_catastrophe_severity_mean"])
    baseline_severity = float(metrics["baseline_catastrophe_severity_mean"])
    evaluator_difference = metrics[
        "weighted_nonterminal_evaluation_delta_mean_difference"
    ]
    checks = {
        "effective_clusters": (
            float(metrics["effective_clusters"]) >= HOLDOUT_MIN_EFFECTIVE_WORLDS
        ),
        "pairs": int(metrics["total_pairs"]) >= HOLDOUT_MIN_PAIRS,
        "mean_advantage": (
            float(metrics["weighted_mean_delta"]) >= HOLDOUT_MIN_MEAN_ADVANTAGE
        ),
        "lower_bound": lower_bound >= HOLDOUT_MIN_LOWER_BOUND,
        "positive_cluster_mass": (
            float(metrics["positive_cluster_mass"])
            >= HOLDOUT_MIN_POSITIVE_WORLD_WEIGHT
        ),
        "cluster_sign_test": sign_p_value <= alpha,
        "candidate_catastrophe_rate": (
            candidate_catastrophe_rate <= HOLDOUT_MAX_CATASTROPHIC_RATE
        ),
        "symmetric_catastrophe_rate_gap": (
            candidate_catastrophe_rate
            <= baseline_catastrophe_rate + HOLDOUT_MAX_CATASTROPHE_RATE_GAP
        ),
        "symmetric_catastrophe_severity": (
            candidate_severity
            <= baseline_severity + HOLDOUT_MAX_CATASTROPHE_SEVERITY_GAP
        ),
        "sign_margin": (
            float(metrics["candidate_better_rate"])
            - float(metrics["baseline_better_rate"])
            >= HOLDOUT_MIN_SIGN_MARGIN
        ),
        "lower_tail_cvar": (
            float(metrics["candidate_lower_tail_cvar"]) >= 0.0
        ),
        "median_of_means": (
            float(metrics["candidate_median_of_means"])
            >= HOLDOUT_MIN_LOWER_BOUND
        ),
        "evaluator_calibration": (
            evaluator_difference is None
            or float(evaluator_difference) >= HOLDOUT_MIN_EVALUATOR_DELTA_DIFFERENCE
        ),
    }
    return {
        "schema_version": 2,
        "candidate": candidate,
        "candidate_rank": candidate_rank,
        "baseline": baseline,
        "alpha_sequence_index": alpha_sequence_index,
        "horizon_index": horizon_index,
        "continuation_steps": HOLDOUT_CONTINUATION_HORIZONS[horizon_index],
        "evidence_kind": "independent_deterministic_robust_paired_holdout_v5",
        "fresh_worlds": int(metrics["world_count"]),
        "effective_worlds": float(metrics["effective_clusters"]),
        "pairs": int(metrics["total_pairs"]),
        "posterior_delta": float(metrics["weighted_mean_delta"]),
        "standard_error": float(metrics["standard_error"]),
        "alpha": alpha,
        "z_value": z_value,
        "paired_lower_confidence_bound": lower_bound,
        "positive_world_weight": float(metrics["positive_cluster_mass"]),
        "positive_worlds": positive_clusters,
        "sign_p_value": sign_p_value,
        "catastrophic_rate": candidate_catastrophe_rate,
        "baseline_catastrophic_rate": baseline_catastrophe_rate,
        "candidate_better_rate": float(metrics["candidate_better_rate"]),
        "baseline_better_rate": float(metrics["baseline_better_rate"]),
        "coverage": 1.0,
        "complete": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "world_weights": [float(weight) for weight in world_weights],
        "state_hashes": list(state_hashes),
        "cluster_hashes": list(cluster_hashes),
        "raw_results": validated_results,
    }


def combined_robust_holdout_certificate(
    certificates: dict[int, dict],
) -> dict[str, object]:
    """Combine an adaptive prefix of the frozen horizon schedule."""
    if not certificates:
        raise ValueError("at least one robust holdout look is required")
    executed = list(certificates)
    planned = list(HOLDOUT_CONTINUATION_HORIZONS)
    if executed != planned[: len(executed)]:
        raise ValueError("robust holdout looks are not a planned horizon prefix")
    candidate = None
    baseline = None
    for horizon, certificate in certificates.items():
        if not isinstance(certificate, dict) or certificate.get("complete") is not True:
            raise ValueError(f"robust holdout horizon {horizon} is incomplete")
        if candidate is None:
            candidate = certificate.get("candidate")
            baseline = certificate.get("baseline")
        if (
            certificate.get("candidate") != candidate
            or certificate.get("baseline") != baseline
            or certificate.get("continuation_steps") != horizon
        ):
            raise ValueError("robust holdout looks evaluate different frozen inputs")
    passed = all(row.get("qualified") is True for row in certificates.values())
    qualified = passed and executed == planned
    if not passed:
        stop_reason = f"rejected_at_horizon_{executed[-1]}"
    elif qualified:
        stop_reason = "all_horizons_qualified"
    else:
        raise ValueError("passing robust holdout prefix cannot stop early")
    return {
        "schema_version": 2,
        "candidate": candidate,
        "candidate_rank": certificates[executed[0]].get("candidate_rank"),
        "baseline": baseline,
        "evidence_kind": "adaptive_multi_horizon_robust_paired_holdout_v5",
        "planned_horizons": planned,
        "executed_horizons": executed,
        "horizons": executed,
        "stop_reason": stop_reason,
        "complete": True,
        "coverage": min(float(row["coverage"]) for row in certificates.values()),
        "pairs": sum(int(row["pairs"]) for row in certificates.values()),
        "posterior_delta": min(
            float(row["posterior_delta"]) for row in certificates.values()
        ),
        "paired_lower_confidence_bound": min(
            float(row["paired_lower_confidence_bound"])
            for row in certificates.values()
        ),
        "catastrophic_rate": max(
            float(row["catastrophic_rate"]) for row in certificates.values()
        ),
        "qualified": qualified,
        "certificates": {str(key): value for key, value in certificates.items()},
    }


def recompute_robust_holdout_certificate(certificate: dict) -> dict[str, object]:
    """Recompute one captured v5 look from its raw aggregate evidence."""
    if not isinstance(certificate, dict) or certificate.get("schema_version") != 2:
        raise ValueError("captured robust holdout certificate has an invalid schema")
    return robust_holdout_certificate(
        certificate["raw_results"],
        certificate["world_weights"],
        certificate["state_hashes"],
        certificate["cluster_hashes"],
        certificate["candidate"],
        certificate["baseline"],
        certificate["alpha_sequence_index"],
        certificate["candidate_rank"],
        certificate["horizon_index"],
    )


def combined_holdout_certificate(certificates: dict[int, dict]) -> dict[str, object]:
    """Require the frozen candidate to pass every configured continuation horizon."""
    if set(certificates) != set(HOLDOUT_CONTINUATION_HORIZONS):
        raise ValueError("holdout certificates do not cover every required horizon")
    candidate = None
    baseline = None
    for horizon, certificate in certificates.items():
        if not isinstance(certificate, dict) or not certificate.get("complete"):
            raise ValueError(f"holdout horizon {horizon} is incomplete")
        if candidate is None:
            candidate = certificate.get("candidate")
            baseline = certificate.get("baseline")
        if (
            certificate.get("candidate") != candidate
            or certificate.get("baseline") != baseline
        ):
            raise ValueError("holdout horizons evaluate different frozen actions")
    qualified = all(
        certificate.get("qualified") is True for certificate in certificates.values()
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "evidence_kind": "independent_seeded_paired_multi_horizon_holdout",
        "horizons": list(HOLDOUT_CONTINUATION_HORIZONS),
        "complete": True,
        "coverage": min(float(row["coverage"]) for row in certificates.values()),
        "pairs": sum(int(row["pairs"]) for row in certificates.values()),
        "posterior_delta": min(
            float(row["posterior_delta"]) for row in certificates.values()
        ),
        "paired_lower_confidence_bound": min(
            float(row["paired_lower_confidence_bound"]) for row in certificates.values()
        ),
        "catastrophic_rate": max(
            float(row["catastrophic_rate"]) for row in certificates.values()
        ),
        "qualified": qualified,
        "certificates": {str(key): value for key, value in certificates.items()},
    }


def _mapping_value(mapping, key, default=0):
    try:
        return mapping.get(key, default)
    except AttributeError:
        try:
            return mapping[key]
        except (KeyError, TypeError):
            return default


def _normalize_identifier(value) -> str:
    return "".join(
        character for character in str(value or "").lower() if character.isalnum()
    )


def showdown_login_status(message: str, username: str) -> str | None:
    expected = _normalize_identifier(username)
    for line in message.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == "nametaken" and _normalize_identifier(parts[2]) == expected:
            return "rejected"
        if (
            len(parts) >= 4
            and parts[1] == "updateuser"
            and _normalize_identifier(parts[2]) == expected
            and parts[3] == "1"
        ):
            return "confirmed"
    return None


def _opponent_tera_used(opponent) -> bool:
    pokemon = [getattr(opponent, "active", None)]
    pokemon.extend(getattr(opponent, "reserve", ()) or ())
    return any(
        member is not None and bool(getattr(member, "terastallized", False))
        for member in pokemon
    )


def sanitize_opponent_priors(battle, priors: object) -> list[tuple[str, float]] | None:
    """Keep only finite priors that match the current public opponent state."""
    if not priors:
        return None
    if not isinstance(priors, dict):
        raise RuntimeError("opponent priors must be an object")
    opponent = getattr(battle, "opponent", None)
    active = getattr(opponent, "active", None)
    if active is None:
        return None
    move_names = {
        _normalize_identifier(getattr(move, "name", move))
        for move in (getattr(active, "moves", ()) or ())
    }
    switch_names = {
        _normalize_identifier(getattr(pokemon, "name", ""))
        for pokemon in (getattr(opponent, "reserve", ()) or ())
        if float(getattr(pokemon, "hp", 0) or 0) > 0
    }
    tera_available = not _opponent_tera_used(opponent)
    expected_actions = set(move_names)
    expected_actions.update(f"switch {name}" for name in switch_names)
    if tera_available:
        expected_actions.update(f"{name}-tera" for name in move_names)
    retained: dict[str, float] = {}
    for raw_name, raw_probability in priors.items():
        if (
            isinstance(raw_probability, bool)
            or not isinstance(raw_probability, (int, float))
            or not math.isfinite(raw_probability)
            or raw_probability < 0
        ):
            raise RuntimeError("opponent priors contain an invalid probability")
        name = str(raw_name).strip().lower()
        if name.startswith("switch "):
            target = _normalize_identifier(name.removeprefix("switch "))
            if not target or target not in switch_names:
                continue
            canonical = f"switch {target}"
        else:
            tera = name.endswith("-tera")
            move = _normalize_identifier(name.removesuffix("-tera"))
            if not move or move not in move_names or (tera and not tera_available):
                continue
            canonical = move + ("-tera" if tera else "")
        retained[canonical] = retained.get(canonical, 0.0) + float(raw_probability)
    total = sum(retained.values())
    if total <= 0:
        return None
    return [(name, probability / total) for name, probability in retained.items()]


def _showdown_move_data(move: str):
    try:
        from data import all_move_json
    except ImportError:
        return None
    return all_move_json.get(move)


def _type_effectiveness_modifier(move_type: str, target_types: list[str]):
    try:
        from fp.helpers import type_effectiveness_modifier
    except ImportError:
        return None
    try:
        return type_effectiveness_modifier(move_type, target_types)
    except (KeyError, TypeError, ValueError):
        return None


def _pending_wish(battle) -> bool:
    wish = getattr(getattr(battle, "user", None), "wish", None)
    try:
        return float(wish[0]) > 0
    except (IndexError, TypeError, ValueError):
        return False


def _encore_failure_reason(battle) -> str | None:
    opponent = getattr(battle, "opponent", None)
    target = getattr(opponent, "active", None)
    if target is None:
        return None

    volatiles = {
        _normalize_identifier(value)
        for value in (getattr(target, "volatile_statuses", ()) or ())
    }
    if "dynamax" in volatiles:
        return "encore_target_incompatible"
    if "encore" in volatiles:
        return "encore_target_already_encored"

    last_used = getattr(opponent, "last_used_move", None)
    last_move = _normalize_identifier(getattr(last_used, "move", ""))
    last_pokemon = _normalize_identifier(getattr(last_used, "pokemon_name", ""))
    target_name = _normalize_identifier(getattr(target, "name", ""))
    if str(getattr(last_used, "move", "") or "").lower().startswith("switch "):
        return "encore_target_just_switched"
    if not last_move or (last_pokemon and target_name and last_pokemon != target_name):
        return "encore_no_eligible_last_move"

    target_moves = getattr(target, "moves", None)
    if target_moves is None:
        return None
    matching_move = next(
        (
            candidate
            for candidate in target_moves
            if _normalize_identifier(
                getattr(candidate, "name", getattr(candidate, "id", candidate))
            )
            == last_move
        ),
        None,
    )
    if matching_move is None:
        return "encore_no_eligible_last_move"
    try:
        if float(getattr(matching_move, "current_pp")) <= 0:
            return "encore_last_move_no_pp"
    except (AttributeError, TypeError, ValueError):
        pass

    last_move_data = _showdown_move_data(last_move)
    if last_move_data is None:
        return None
    if bool((last_move_data.get("flags") or {}).get("failencore")):
        return "encore_last_move_incompatible"
    if bool(last_move_data.get("isZ") or last_move_data.get("isMax")):
        return "encore_last_move_incompatible"
    return None


def _known_noop_reason(battle, choice: str) -> str | None:
    move = choice.removesuffix("-tera").lower()
    if move.startswith("switch "):
        return None
    active = getattr(getattr(battle, "user", None), "active", None)
    if active is None:
        return None
    if choice.endswith("-tera") and not bool(getattr(active, "terastallized", False)):
        return None
    if move == "wish" and _pending_wish(battle):
        return "wish_already_pending"
    if move == "encore":
        encore_reason = _encore_failure_reason(battle)
        if encore_reason:
            return encore_reason
    try:
        hp_value = float(getattr(active, "hp", None))
        max_hp_value = float(getattr(active, "max_hp", None))
    except (TypeError, ValueError):
        hp_value = max_hp_value = float("nan")

    if (
        move == "substitute"
        and math.isfinite(hp_value)
        and math.isfinite(max_hp_value)
        and max_hp_value > 0
        and hp_value <= max_hp_value / 4
    ):
        return "substitute_insufficient_hp"

    status = str(getattr(active, "status", "") or "").lower()
    if move == "rest" and status in {"slp", "sleep"}:
        item = str(getattr(active, "item", "") or "").lower()
        ability = str(getattr(active, "ability", "") or "").lower()
        volatiles = {
            str(value).lower()
            for value in (getattr(active, "volatile_statuses", ()) or ())
        }
        opponent = getattr(getattr(battle, "opponent", None), "active", None)
        opponent_ability = str(getattr(opponent, "ability", "") or "").lower()
        field = str(getattr(battle, "field", "") or "").lower()
        usable_chesto = item == "chestoberry" and not (
            ability == "klutz"
            or "embargo" in volatiles
            or field == "magicroom"
            or opponent_ability == "unnerve"
        )
        if not usable_chesto and item not in {"unknownitem", "unknown"}:
            return "rest_already_asleep_without_usable_chesto"

    boosted_stats = _PURE_BOOST_MOVES.get(move)
    boosts = getattr(active, "boosts", {}) or {}
    try:
        capped = boosted_stats and all(
            float(_mapping_value(boosts, stat, 0)) >= 6 for stat in boosted_stats
        )
    except (TypeError, ValueError):
        capped = False
    if capped:
        return "capped_boost"
    return None


def _opponent_has_known_conditional_priority(battle) -> bool:
    """Status no-ops can be intentional into Sucker Punch-like moves."""
    opponent = getattr(getattr(battle, "opponent", None), "active", None)
    for move in getattr(opponent, "moves", ()) or ():
        name = str(getattr(move, "name", getattr(move, "id", move)) or "")
        normalized = "".join(
            character for character in name.lower() if character.isalnum()
        )
        if normalized in {"suckerpunch", "thunderclap"}:
            return True
    return False


def _effective_defensive_types(pokemon) -> list[str]:
    tera_type = _normalize_identifier(getattr(pokemon, "tera_type", ""))
    if bool(getattr(pokemon, "terastallized", False)) and tera_type not in {
        "",
        "nothing",
        "typeless",
    }:
        return [tera_type]
    return [
        normalized
        for normalized in (
            _normalize_identifier(value)
            for value in (getattr(pokemon, "types", ()) or ())
        )
        if normalized and normalized not in {"nothing", "typeless"}
    ]


def _move_ignores_ability(active, move_data) -> bool:
    ability = _normalize_identifier(getattr(active, "ability", ""))
    if ability in _MOLD_BREAKER_ABILITIES:
        return True
    if move_data is None:
        return False
    if bool(move_data.get("ignoreAbility")):
        return True
    return (
        ability == "myceliummight"
        and str(move_data.get("category", "")).lower() == "status"
    )


def _ability_immunity_reason(battle, move: str, move_data, opponent) -> str | None:
    ability = _normalize_identifier(getattr(opponent, "ability", ""))
    if not ability:
        return None
    active = getattr(getattr(battle, "user", None), "active", None)
    if active is None or _move_ignores_ability(active, move_data):
        return None

    target = _normalize_identifier(move_data.get("target", "normal"))
    if target not in _CURRENT_TARGETS:
        return None
    move_type = _normalize_identifier(move_data.get("type", ""))
    flags = move_data.get("flags") or {}
    immune_type = _ABILITY_TYPE_IMMUNITIES.get(ability)
    if immune_type == move_type:
        if ability == "levitate" and move == "thousandarrows":
            return None
        return f"revealed_{ability}_immunity"
    immune_flag = _ABILITY_FLAG_IMMUNITIES.get(ability)
    if immune_flag and bool(flags.get(immune_flag)):
        return f"revealed_{ability}_immunity"
    if (
        ability == "goodasgold"
        and str(move_data.get("category", "")).lower() == "status"
    ):
        return "revealed_goodasgold_immunity"
    if ability == "overcoat" and bool(flags.get("powder")):
        return "revealed_overcoat_immunity"
    if ability == "wonderguard" and move != "struggle":
        modifier = _type_effectiveness_modifier(
            move_type, _effective_defensive_types(opponent)
        )
        if (
            str(move_data.get("category", "")).lower() != "status"
            and modifier is not None
            and modifier <= 1
        ):
            return "revealed_wonderguard_immunity"
    return None


def _type_immunity_applies(battle, move: str, move_data, opponent) -> bool:
    if str(move_data.get("category", "")).lower() == "status":
        return False
    ignored = move_data.get("ignoreImmunity")
    move_type = _normalize_identifier(move_data.get("type", ""))
    if ignored is True or bool(_mapping_value(ignored, move_type, False)):
        return False
    if _normalize_identifier(getattr(opponent, "item", "")) == "ringtarget":
        return False
    target_types = _effective_defensive_types(opponent)
    attacker = getattr(getattr(battle, "user", None), "active", None)
    attacker_ability = _normalize_identifier(getattr(attacker, "ability", ""))
    if (
        move_type in {"normal", "fighting"}
        and "ghost" in target_types
        and attacker_ability in {"scrappy", "mindseye"}
    ):
        return False
    if move_type == "ground" and "flying" in target_types:
        grounded = (
            bool(getattr(battle, "gravity", False))
            or _normalize_identifier(getattr(opponent, "item", "")) == "ironball"
            or bool(
                {"ingrain", "smackdown"}
                & {
                    _normalize_identifier(value)
                    for value in (getattr(opponent, "volatile_statuses", ()) or ())
                }
            )
            or move == "thousandarrows"
        )
        if grounded:
            target_types = [value for value in target_types if value != "flying"]
            if not target_types:
                return False
    modifier = _type_effectiveness_modifier(move_type, target_types)
    return bool(target_types) and modifier == 0


def _prediction_sensitive_reason(battle, choice: str) -> str | None:
    move = choice.removesuffix("-tera").lower()
    if move.startswith("switch "):
        return None
    opponent = getattr(getattr(battle, "opponent", None), "active", None)
    if opponent is None:
        return None
    move_data = _showdown_move_data(move)
    reflectable = bool((move_data or {}).get("flags", {}).get("reflectable"))
    if move_data is None:
        reflectable = move in _KNOWN_REFLECTABLE_MOVES

    ability = _normalize_identifier(getattr(opponent, "ability", ""))
    if ability == "magicbounce" and reflectable:
        active = getattr(getattr(battle, "user", None), "active", None)
        if active is None or not _move_ignores_ability(active, move_data):
            return "revealed_magic_bounce"
    if move_data:
        ability_reason = _ability_immunity_reason(battle, move, move_data, opponent)
        if ability_reason:
            return ability_reason
        if _type_immunity_applies(battle, move, move_data, opponent):
            return "known_type_immunity"
    return None


def _freeze_public_mapping(value) -> tuple:
    try:
        items = value.items()
    except AttributeError:
        return ()
    return tuple(sorted((str(key), repr(item)) for key, item in items))


def _pokemon_public_state(pokemon) -> tuple | None:
    if pokemon is None:
        return None
    return (
        str(getattr(pokemon, "name", "")),
        repr(getattr(pokemon, "hp", None)),
        repr(getattr(pokemon, "max_hp", None)),
        str(getattr(pokemon, "status", "") or ""),
        str(getattr(pokemon, "item", "") or ""),
        str(getattr(pokemon, "ability", "") or ""),
        bool(getattr(pokemon, "terastallized", False)),
        str(getattr(pokemon, "tera_type", "") or ""),
        repr(getattr(pokemon, "rest_turns", None)),
        repr(getattr(pokemon, "sleep_turns", None)),
        _freeze_public_mapping(getattr(pokemon, "boosts", {})),
        tuple(
            sorted(
                str(value)
                for value in (getattr(pokemon, "volatile_statuses", ()) or ())
            )
        ),
    )


def public_battle_state(battle) -> tuple:
    """Return progress-bearing public state, intentionally excluding turn and PP."""
    sides = []
    for side_name in ("user", "opponent"):
        side = getattr(battle, side_name, None)
        sides.append(
            (
                _pokemon_public_state(getattr(side, "active", None)),
                tuple(
                    _pokemon_public_state(pokemon)
                    for pokemon in (getattr(side, "reserve", ()) or ())
                ),
                _freeze_public_mapping(getattr(side, "side_conditions", {})),
                repr(getattr(side, "wish", None)),
            )
        )
    return (
        *sides,
        str(getattr(battle, "weather", "") or ""),
        repr(getattr(battle, "weather_turns_remaining", None)),
        str(getattr(battle, "field", "") or ""),
        repr(getattr(battle, "field_turns_remaining", None)),
        bool(getattr(battle, "trick_room", False)),
        repr(getattr(battle, "trick_room_turns_remaining", None)),
        bool(getattr(battle, "gravity", False)),
        bool(getattr(battle, "force_switch", False)),
    )


def _repeated_no_progress_period(history, state, choice: str) -> int | None:
    proposed = (state, choice)
    for period in (1, 2, 3):
        if len(history) < 2 * period:
            continue
        first = history[-2 * period : -period]
        second = history[-period:]
        if first == second and proposed == first[0]:
            return period
    return None


def _state_active(state: tuple, side: int) -> tuple | None:
    try:
        active = state[side][0]
    except (IndexError, TypeError):
        return None
    return active if isinstance(active, tuple) else None


def _state_hp_fraction(state: tuple, side: int) -> float | None:
    active = _state_active(state, side)
    if active is None:
        return None
    try:
        hp = float(active[1])
        maximum = float(active[2])
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(hp) or not math.isfinite(maximum) or maximum <= 0:
        return None
    return hp / maximum


def _same_active(state: tuple, other: tuple, side: int) -> bool:
    active = _state_active(state, side)
    other_active = _state_active(other, side)
    return bool(active and other_active and active[0] == other_active[0])


def _opponent_made_no_hp_progress(first: tuple, current: tuple) -> bool:
    if not _same_active(first, current, 1):
        return False
    first_hp = _state_hp_fraction(first, 1)
    current_hp = _state_hp_fraction(current, 1)
    return (
        first_hp is not None
        and current_hp is not None
        and current_hp >= first_hp - 0.02
    )


def _state_boosts(state: tuple, side: int) -> dict[str, float]:
    active = _state_active(state, side)
    if active is None:
        return {}
    try:
        frozen_boosts = active[10]
    except IndexError:
        return {}
    boosts = {}
    for key, value in frozen_boosts:
        try:
            boosts[key] = float(value)
        except (TypeError, ValueError):
            continue
    return boosts


def _state_live_count(state: tuple, side: int) -> int:
    try:
        active = state[side][0]
        reserve = state[side][1]
    except (IndexError, TypeError):
        return 0
    count = 0
    for pokemon in (active, *(reserve or ())):
        if not isinstance(pokemon, tuple):
            continue
        try:
            count += float(pokemon[1]) > 0
        except (IndexError, TypeError, ValueError):
            continue
    return count


def _opponent_offense_increased(first: tuple, current: tuple) -> bool:
    previous = _state_boosts(first, 1)
    present = _state_boosts(current, 1)
    return any(
        present.get(stat, 0) > previous.get(stat, 0)
        for stat in ("attack", "special-attack", "speed", "atk", "spa", "spe")
    )


def _losing_stall(history, state: tuple) -> bool:
    same_matchup = []
    for previous_state, _previous_choice in reversed(history):
        if not _same_active(previous_state, state, 0) or not _same_active(
            previous_state, state, 1
        ):
            break
        same_matchup.append(previous_state)
    if len(same_matchup) < 4 or _state_live_count(state, 0) <= 1:
        return False
    first = same_matchup[-1]
    first_user_hp = _state_hp_fraction(first, 0)
    current_user_hp = _state_hp_fraction(state, 0)
    return (
        first_user_hp is not None
        and current_user_hp is not None
        and current_user_hp <= first_user_hp - 0.10
        and _opponent_made_no_hp_progress(first, state)
        and _opponent_offense_increased(first, state)
    )


def _meaningful_progress_since(first: tuple, current: tuple) -> bool:
    if not _same_active(first, current, 0) or not _same_active(first, current, 1):
        return True
    if not _opponent_made_no_hp_progress(first, current):
        return True

    first_user_boosts = _state_boosts(first, 0)
    current_user_boosts = _state_boosts(current, 0)
    if any(
        current_user_boosts.get(stat, 0) > value
        for stat, value in first_user_boosts.items()
    ):
        return True

    first_opponent_boosts = _state_boosts(first, 1)
    current_opponent_boosts = _state_boosts(current, 1)
    if any(
        current_opponent_boosts.get(stat, 0) < value
        for stat, value in first_opponent_boosts.items()
    ):
        return True

    first_opponent = _state_active(first, 1)
    current_opponent = _state_active(current, 1)
    # Status, item, ability, tera, or volatile changes can make an otherwise
    # repeated action productive. HP and turn counters are handled separately.
    progress_fields = (3, 4, 5, 6, 7, 11)
    return any(
        first_opponent[index] != current_opponent[index] for index in progress_fields
    )


def _semantic_no_progress_reason(history, state: tuple, choice: str) -> str | None:
    base_choice = choice.removesuffix("-tera")
    if not choice.startswith("switch ") and _losing_stall(history, state):
        return "losing_stall"
    if base_choice in _PIVOT_MOVES:
        for previous_state, previous_choice in reversed(history):
            if previous_choice.removesuffix("-tera") != base_choice:
                continue
            if not _same_active(previous_state, state, 0) or not _same_active(
                previous_state, state, 1
            ):
                continue
            if (
                _state_live_count(state, 0) < _state_live_count(previous_state, 0)
                and _opponent_offense_increased(previous_state, state)
            ):
                return "repeated_sacrificial_pivot"
            break
    if history and base_choice in _PROTECT_MOVES:
        previous_state, previous_choice = history[-1]
        if (
            previous_choice.removesuffix("-tera") == base_choice
            and _same_active(previous_state, state, 0)
            and _same_active(previous_state, state, 1)
            and not _meaningful_progress_since(previous_state, state)
        ):
            return "consecutive_protect"

    repeated_history = 5 if base_choice in _RECOVERY_MOVES else 2
    if len(history) >= repeated_history:
        recent = history[-repeated_history:]
        if all(
            previous_choice.removesuffix("-tera") == base_choice
            and _same_active(previous_state, state, 0)
            and _same_active(previous_state, state, 1)
            for previous_state, previous_choice in recent
        ) and not _meaningful_progress_since(recent[0][0], state):
            return "repeated_action"

    if choice.startswith("switch ") and len(history) >= 3:
        recent = history[-3:]
        if all(
            previous_choice.startswith("switch ") for _state, previous_choice in recent
        ):
            first_state = recent[0][0]
            if all(
                _same_active(previous_state, state, 1) for previous_state, _ in recent
            ):
                if _opponent_made_no_hp_progress(first_state, state):
                    return "switch_carousel"
    return None


def _terminal_action_reason(battle, choice: str) -> str | None:
    move = choice.removesuffix("-tera")
    if move not in _PIVOT_MOVES:
        return None
    reserves = getattr(getattr(battle, "user", None), "reserve", ()) or ()
    for pokemon in reserves:
        try:
            if float(getattr(pokemon, "hp", 0)) > 0:
                return None
        except (TypeError, ValueError):
            continue
    move_data = _showdown_move_data(move)
    if move in _NON_DAMAGE_TERMINAL_PIVOTS or str(
        (move_data or {}).get("category", "")
    ).lower() in {"physical", "special"}:
        return "pivot_without_reserve"
    return None


def _credible_terminal_replacement(blocked_choice: str, replacement: str) -> bool:
    blocked = blocked_choice.removesuffix("-tera")
    if blocked in _NON_DAMAGE_TERMINAL_PIVOTS:
        return True
    blocked_data = _showdown_move_data(blocked) or {}
    replacement_data = _showdown_move_data(replacement.removesuffix("-tera")) or {}
    if str(replacement_data.get("category", "")).lower() not in {
        "physical",
        "special",
    }:
        return False
    try:
        blocked_power = float(blocked_data.get("basePower", 0))
        replacement_power = float(replacement_data.get("basePower", 0))
    except (TypeError, ValueError):
        return False
    return blocked_power > 0 and replacement_power >= blocked_power * 0.75


def freeze_holdout_candidate_panel(
    battle, provisional: dict[str, object]
) -> list[dict[str, object]]:
    """Freeze the eligible search-ranked candidates before certification starts."""
    baseline = provisional.get("baseline")
    request_actions = set(provisional.get("request_actions") or ())
    visit_mass = provisional.get("visit_mass") or {}
    panel = []
    for action in provisional.get("search_actions") or ():
        if (
            action == baseline
            or action not in request_actions
            or _known_noop_reason(battle, action) is not None
            or _prediction_sensitive_reason(battle, action) is not None
        ):
            continue
        panel.append(
            {
                "rank": len(panel) + 1,
                "action": action,
                "visit_mass": float(visit_mass.get(action, 0.0)),
            }
        )
        if len(panel) == HOLDOUT_CANDIDATE_COUNT:
            break
    return panel


def select_final_choice(
    battle,
    mcts_results,
    priors=None,
    histories=None,
    independent_evidence=None,
    record_history: bool = True,
    selection_mode: str = "certified",
    search_policy: list[dict[str, object]] | None = None,
    sampled_search_action: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Apply one declared controller and public-state deterministic safeguards."""
    if selection_mode not in CONTROLLER_MODES:
        raise ValueError(f"unsupported controller mode: {selection_mode}")
    search_first = selection_mode == "search_first"
    request_actions = request_player_actions(battle)
    if search_policy is None:
        baseline, ordered, visit_mass = derive_policy_baseline(
            mcts_results, priors, request_actions
        )
        counterfactual_values = {}
    else:
        ordered, visit_mass, counterfactual_values = (
            _shared_root_actions_and_probability_mass(search_policy, request_actions)
        )
        baseline, _unused_order, _unused_mass = derive_policy_baseline(
            (), priors, request_actions
        )
        if baseline == "no move":
            baseline = ordered[0]
        if sampled_search_action is None or sampled_search_action not in ordered:
            raise RuntimeError("shared-root sampled action is not in its policy")
        ordered = [sampled_search_action] + [
            action for action in ordered if action != sampled_search_action
        ]
    raw_choice = ordered[0] if ordered else baseline
    baseline_missing_from_search = baseline not in ordered
    if search_policy is None:
        evidence_by_action = {
            action: paired_candidate_evidence(mcts_results, action, baseline)
            for action in ordered
            if action != baseline
        }
    else:
        evidence_by_action = {
            action: {
                "candidate": action,
                "baseline": baseline,
                "coverage": 1.0,
                "complete": True,
                "qualified": False,
                "evidence_kind": "shared_rm_plus_policy",
                "policy_probability": visit_mass[action],
                "counterfactual_value": counterfactual_values[action],
            }
            for action in ordered
            if action != baseline
        }
    for action, holdout in (independent_evidence or {}).items():
        if action in evidence_by_action and isinstance(holdout, dict):
            evidence_by_action[action] = {
                **holdout,
                "adaptive_heuristic": evidence_by_action[action],
            }
    baseline_evidence = {
        "candidate": baseline,
        "baseline": baseline,
        "coverage": 1.0,
        "complete": True,
        "qualified": True,
        "paired_lower_confidence_bound": 0.0,
        "posterior_delta": 0.0,
        "minimum_advantage": MIN_PAIRED_ADVANTAGE,
        "evidence_kind": "policy_baseline",
        "heuristic_qualified": True,
    }

    valid_search_actions = [
        action for action in ordered if _request_allows(action, request_actions)
    ]
    if search_first and valid_search_actions:
        choice = valid_search_actions[0]
        evidence = evidence_by_action.get(choice, baseline_evidence)
        reason = (
            "search_first_policy_agreement"
            if choice == baseline
            else "search_first_search_selection"
        )
    elif search_first:
        choice = baseline
        evidence = baseline_evidence
        reason = "search_infrastructure_policy_fallback"
    elif baseline_missing_from_search:
        choice = baseline
        reason = "policy_baseline_missing_from_search"
        evidence = baseline_evidence
    elif raw_choice == baseline:
        choice = baseline
        reason = "policy_baseline"
        evidence = baseline_evidence
    else:
        qualified_candidates = [
            action
            for action in ordered
            if action != baseline
            and evidence_by_action.get(action, {}).get("qualified") is True
        ]
        if qualified_candidates:
            choice = qualified_candidates[0]
            evidence = evidence_by_action[choice]
            reason = "independent_holdout_qualified_search_override"
        else:
            choice = baseline
            evidence = evidence_by_action[raw_choice]
            reason = "incomplete_or_nonpositive_evidence"

    def qualified(action: str) -> bool:
        return action == baseline or bool(
            evidence_by_action.get(action, {}).get("qualified")
        )

    if histories is None:
        histories = _CHOICE_HISTORY
    tag = str(getattr(battle, "battle_tag", "") or id(battle))
    if record_history:
        if tag not in histories and len(histories) >= MAX_TRACKED_BATTLES:
            histories.pop(next(iter(histories)))
        history = histories.setdefault(tag, [])
    else:
        history = histories.get(tag, [])
    state = public_battle_state(battle)
    blocked_safeguard = None
    shadow_risks = []

    noop_reason = _known_noop_reason(battle, choice)
    if noop_reason:
        repeated_noop = any(
            previous_choice == choice for _state, previous_choice in history[-3:]
        )
        tactical_dodge = (
            _opponent_has_known_conditional_priority(battle) and not repeated_noop
        )
        if not tactical_dodge:
            safe = [
                action
                for action in ordered
                if _request_allows(action, request_actions)
                and _known_noop_reason(battle, action) is None
            ]
            qualified_safe = [action for action in safe if qualified(action)]
            replacement = (
                (safe or [choice])[0]
                if search_first
                else (qualified_safe or safe or [choice])[0]
            )
            if replacement != choice:
                choice = replacement
                reason = f"guaranteed_noop_{noop_reason}"
                evidence = evidence_by_action.get(choice, baseline_evidence)

    prediction_reason = _prediction_sensitive_reason(battle, choice)
    if prediction_reason and search_first:
        shadow_risks.append(
            {
                "reason": prediction_reason,
                "action": choice,
                "selection_eligible": False,
            }
        )
    elif prediction_reason:
        repeated_prediction = any(
            previous_choice == choice for _state, previous_choice in history[-3:]
        )
        safe = [
            action
            for action in ordered
            if action != choice
            and _request_allows(action, request_actions)
            and _known_noop_reason(battle, action) is None
            and _prediction_sensitive_reason(battle, action) is None
        ]
        prior_by_action = {
            action: float(probability)
            for action, probability in (priors or ())
            if isinstance(action, str)
            and isinstance(probability, (int, float))
            and not isinstance(probability, bool)
            and math.isfinite(probability)
        }
        safe.sort(key=lambda action: (-prior_by_action.get(action, 0.0), action))
        comparison = None
        if safe:
            comparison = paired_candidate_evidence(mcts_results, choice, safe[0])
        if repeated_prediction or not comparison or not comparison["qualified"]:
            qualified_safe = [action for action in safe if qualified(action)]
            replacement = (qualified_safe or [choice])[0]
            if replacement != choice:
                choice = replacement
                reason = (
                    f"repeated_{prediction_reason}"
                    if repeated_prediction
                    else f"unqualified_{prediction_reason}"
                )
                evidence = evidence_by_action.get(choice, comparison or baseline_evidence)
            elif safe:
                blocked_safeguard = {
                    "reason": prediction_reason,
                    "cause": "no_qualified_replacement",
                    "candidates": safe,
                }
    period = _repeated_no_progress_period(history, state, choice)
    if period is not None:
        safe = [
            action
            for action in ordered
            if action != choice
            and _request_allows(action, request_actions)
            and _known_noop_reason(battle, action) is None
            and _repeated_no_progress_period(history, state, action) is None
        ]
        qualified_safe = [action for action in safe if qualified(action)]
        replacement = (
            (safe or [choice])[0]
            if search_first
            else (qualified_safe or [choice])[0]
        )
        if replacement == choice and safe and blocked_safeguard is None:
            blocked_safeguard = {
                "reason": f"repeated_no_progress_period_{period}",
                "cause": "no_qualified_replacement",
                "candidates": safe,
            }
        if replacement != choice:
            choice = replacement
            reason = f"repeated_no_progress_period_{period}"
            evidence = evidence_by_action.get(choice, baseline_evidence)
    terminal_reason = _terminal_action_reason(battle, choice)
    semantic_reason = _semantic_no_progress_reason(history, state, choice)
    if terminal_reason or semantic_reason:
        blocked_base = choice.removesuffix("-tera")
        safe = [
            action
            for action in ordered
            if action.removesuffix("-tera") != blocked_base
            and _request_allows(action, request_actions)
            and _known_noop_reason(battle, action) is None
            and _prediction_sensitive_reason(battle, action) is None
            and not (
                semantic_reason == "switch_carousel" and action.startswith("switch ")
            )
            and not (
                semantic_reason == "repeated_sacrificial_pivot"
                and action.removesuffix("-tera") in _PIVOT_MOVES
            )
            and not (terminal_reason and _terminal_action_reason(battle, action))
            and not (
                terminal_reason and not _credible_terminal_replacement(choice, action)
            )
        ]
        qualified_safe = [action for action in safe if qualified(action)]
        if terminal_reason:
            replacement = (qualified_safe or safe or [choice])[0]
        elif semantic_reason == "losing_stall":
            switches = [action for action in safe if action.startswith("switch ")]
            qualified_switches = [action for action in switches if qualified(action)]
            replacement = (
                (switches or safe or [choice])[0]
                if search_first
                else (qualified_switches or qualified_safe or [choice])[0]
            )
        else:
            replacement = (
                (safe or [choice])[0]
                if search_first
                else (qualified_safe or [choice])[0]
            )
        if (
            semantic_reason
            and replacement == choice
            and safe
            and blocked_safeguard is None
        ):
            blocked_safeguard = {
                "reason": semantic_reason,
                "cause": "no_qualified_replacement",
                "candidates": safe,
            }
        if replacement != choice:
            choice = replacement
            reason = (
                f"terminal_{terminal_reason}"
                if terminal_reason
                else f"semantic_no_progress_{semantic_reason}"
            )
            evidence = evidence_by_action.get(choice, baseline_evidence)
    if record_history:
        history.append((state, choice))
        del history[:-12]

    qualified_shadow_actions = [
        action
        for action in ordered
        if action != baseline and evidence_by_action.get(action, {}).get("qualified") is True
    ]
    deterministic_correction = reason.startswith(
        ("guaranteed_noop_", "repeated_no_progress_", "semantic_no_progress_", "terminal_")
    )
    if deterministic_correction:
        selection_class = "deterministic_correction"
    elif reason == "search_infrastructure_policy_fallback":
        selection_class = "infrastructure_fallback"
    elif search_first:
        selection_class = "search_selection"
    else:
        selection_class = "certified_selection"
    telemetry = {
        "controller_mode": selection_mode,
        "selection_class": selection_class,
        "baseline": baseline,
        "raw_choice": raw_choice,
        "final_choice": choice,
        "reason": reason,
        "coverage": evidence.get("coverage", 0.0),
        "evidence": evidence,
        "overridden": choice != raw_choice,
        "search_override_admitted": (
            reason == "independent_holdout_qualified_search_override"
            and choice != baseline
        ),
        "blocked_safeguard": blocked_safeguard,
        "shadow_risks": shadow_risks,
        "verifier_shadow": {
            "selection_eligible": not search_first,
            "evidence_provided": bool(independent_evidence),
            "qualified_actions": qualified_shadow_actions,
        },
        "visit_mass": visit_mass,
        "search_mass_kind": (
            "shared_policy_probability" if search_policy is not None else "weighted_visits"
        ),
        "request_actions": sorted(request_actions or ()),
        "search_actions": ordered,
        "missing_request_actions": sorted((request_actions or set()) - set(ordered)),
    }
    return choice, telemetry


def select_search_first_choice(
    battle,
    mcts_results,
    priors=None,
    histories=None,
    independent_evidence=None,
    record_history: bool = True,
) -> tuple[str, dict[str, object]]:
    """Select valid search output while keeping verifier evidence shadow-only."""
    return select_final_choice(
        battle,
        mcts_results,
        priors,
        histories,
        independent_evidence,
        record_history,
        selection_mode="search_first",
    )


def select_shared_root_choice(
    battle,
    shared_result: dict[str, object],
    priors=None,
    *,
    seed: int,
    histories=None,
    record_history: bool = True,
) -> tuple[str, dict[str, object]]:
    """Sample one mixed shared-root action, then apply public-state safeguards."""
    try:
        validated = validate_shared_root_result_payload(shared_result)
    except ValueError as exc:
        raise RuntimeError(f"shared-root result is invalid: {exc}") from exc
    request_actions = request_player_actions(battle)
    ordered, probability_mass, _values = _shared_root_actions_and_probability_mass(
        validated["policy"], request_actions
    )
    sampled, draw = _sample_shared_root_action(ordered, probability_mass, seed)
    choice, telemetry = select_final_choice(
        battle,
        (),
        priors,
        histories,
        None,
        record_history,
        selection_mode="search_first",
        search_policy=validated["policy"],
        sampled_search_action=sampled,
    )
    telemetry["root_search_mode"] = "shared_rm_plus"
    telemetry["mixed_strategy"] = validated["policy"]
    telemetry["mixed_strategy_seed"] = seed
    telemetry["mixed_strategy_draw"] = draw
    telemetry["sampled_action"] = sampled
    telemetry["solver_diagnostics"] = validated["diagnostics"]
    telemetry["opponent_policies"] = validated["opponent_policies"]
    return choice, telemetry


def controller_select_fn(mode: str | None = None):
    mode = mode or os.environ.get("METAGROSS_CONTROLLER_MODE", DEFAULT_CONTROLLER_MODE)
    if mode == "search_first":
        return select_search_first_choice
    if mode == "certified":
        return select_final_choice
    raise RuntimeError(f"unsupported METAGROSS_CONTROLLER_MODE: {mode}")


def _mcts_result_from_payload(payload: object, engine_module=None):
    try:
        payload = validate_result_payload(payload)
    except ValueError as exc:
        raise RuntimeError(f"remote MCTS result is invalid: {exc}") from exc
    if engine_module is None:
        import poke_engine as engine_module

    def side(value: object, label: str):
        if not isinstance(value, list) or (label == "side_one" and not value):
            raise RuntimeError(f"remote MCTS {label} is invalid")
        options = []
        for row in value:
            if not isinstance(row, dict):
                raise RuntimeError(f"remote MCTS {label} entry is invalid")
            move = row.get("move_choice")
            score = row.get("total_score")
            visits = row.get("visits")
            if not isinstance(move, str) or not move:
                raise RuntimeError(f"remote MCTS {label} move is invalid")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RuntimeError(f"remote MCTS {label} score is invalid")
            if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
                raise RuntimeError(f"remote MCTS {label} visits are invalid")
            options.append(
                engine_module.MctsSideResult(
                    move_choice=move,
                    total_score=float(score),
                    visits=visits,
                )
            )
        return options

    total_visits = payload.get("total_visits")
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
    ):
        raise RuntimeError("remote MCTS total visits are invalid")
    return engine_module.MctsResult(
        side_one=side(payload.get("side_one"), "side_one"),
        side_two=side(payload.get("side_two"), "side_two"),
        total_visits=total_visits,
    )


def _remote_mcts_function():
    pid = os.getpid()
    if pid not in _REMOTE_FUNCTIONS:
        import modal

        app_name = os.environ["METAGROSS_REMOTE_MCTS_APP"]
        function_name = os.environ["METAGROSS_REMOTE_MCTS_FUNCTION"]
        _REMOTE_FUNCTIONS.clear()
        _REMOTE_FUNCTIONS[pid] = modal.Function.from_name(app_name, function_name)
    return _REMOTE_FUNCTIONS[pid]


def _http_mcts_call(payload: object) -> object:
    import urllib.request

    url = validate_loopback_search_url(os.environ.get("METAGROSS_REMOTE_MCTS_URL", ""))
    token = os.environ.get("METAGROSS_REMOTE_MCTS_TOKEN")
    if not token:
        raise RuntimeError("METAGROSS_REMOTE_MCTS_TOKEN is required")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=positive_environment_seconds(
                "METAGROSS_REMOTE_MCTS_TIMEOUT_SECONDS", 10.0
            ),
        ) as response:
            body = response.read(MAX_REMOTE_RESPONSE_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(
            f"remote HTTP MCTS request failed: {type(exc).__name__}"
        ) from exc
    if len(body) > MAX_REMOTE_RESPONSE_BYTES:
        raise RuntimeError("remote HTTP MCTS response is too large")
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("remote HTTP MCTS returned invalid JSON") from exc


MODAL_BATCH_SIZE = MODAL_CONTAINER_BATCH_SIZE
MODAL_MAX_CONCURRENT_BATCHES = MODAL_MAX_CONTAINERS


def _modal_mcts_call(payload: object) -> object:
    function = _remote_mcts_function()
    timeout = positive_environment_seconds("METAGROSS_REMOTE_MCTS_TIMEOUT_SECONDS", 10.0)
    batches = (
        [payload]
        if not isinstance(payload, list) or len(payload) <= MODAL_BATCH_SIZE
        else [
            payload[start : start + MODAL_BATCH_SIZE]
            for start in range(0, len(payload), MODAL_BATCH_SIZE)
        ]
    )
    if len(batches) > MODAL_MAX_CONCURRENT_BATCHES:
        raise RuntimeError("remote Modal MCTS request exceeds shard concurrency")
    outcomes = [queue.Queue(maxsize=1) for _batch in batches]

    def invoke(index: int, batch: object) -> None:
        try:
            outcomes[index].put((True, function.remote(batch)))
        except BaseException as exc:
            outcomes[index].put((False, exc))

    for index, batch in enumerate(batches):
        threading.Thread(
            target=invoke,
            args=(index, batch),
            name=f"modal-mcts-{index}",
            daemon=True,
        ).start()
    deadline = time.monotonic() + timeout
    responses = []
    for outcome in outcomes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"remote Modal MCTS exceeded {timeout:g}s")
        try:
            succeeded, value = outcome.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(f"remote Modal MCTS exceeded {timeout:g}s") from exc
        if not succeeded:
            raise value
        responses.append(value)
    if len(batches) == 1:
        return responses[0]
    if any(not isinstance(response, list) for response in responses):
        raise RuntimeError("remote Modal MCTS shard returned an invalid response")
    return [row for response in responses for row in response]


def _remote_mcts_call(payload: object) -> object:
    transport = os.environ.get("METAGROSS_REMOTE_MCTS_TRANSPORT", "modal")
    if transport == "modal":
        return _modal_mcts_call(payload)
    if transport == "http":
        return _http_mcts_call(payload)
    raise RuntimeError("unsupported remote MCTS transport")


def _validate_remote_response(response: object, request_id: str, index: int) -> dict:
    if not isinstance(response, dict) or response.get("schema") != REMOTE_MCTS_SCHEMA:
        raise RuntimeError("remote MCTS returned an invalid schema")
    if response.get("request_id") != request_id or response.get("index") != index:
        raise RuntimeError("remote MCTS response correlation mismatch")
    engine = response.get("engine")
    expected_sha = os.environ.get("METAGROSS_REMOTE_ENGINE_SHA256")
    if not isinstance(engine, dict) or engine.get("contract") != REMOTE_ENGINE_CONTRACT:
        raise RuntimeError("remote MCTS engine contract mismatch")
    if engine.get("source_sha256") != ENGINE_SOURCE_SHA256:
        raise RuntimeError("remote MCTS engine source SHA-256 mismatch")
    if not expected_sha or engine.get("native_sha256") != expected_sha:
        raise RuntimeError("remote MCTS engine SHA-256 mismatch")
    if os.environ.get("METAGROSS_REMOTE_MCTS_TRANSPORT", "modal") == "http":
        resources = engine.get("resources")
        expected_instance_type = os.environ.get("METAGROSS_REMOTE_MCTS_INSTANCE_TYPE")
        if (
            not isinstance(resources, dict)
            or resources.get("provider") != "aws_ec2"
            or not expected_instance_type
            or resources.get("instance_type") != expected_instance_type
            or resources.get("logical_cpus") != 32
            or not isinstance(resources.get("memory_mib"), int)
            or resources.get("memory_mib") < 60_000
        ):
            raise RuntimeError("remote HTTP MCTS resource identity mismatch")
        timing = response.get("timing")
        required_timings = {
            "queue_ms",
            "validation_ms",
            "search_ms",
            "worker_ms",
            "batch_ms",
        }
        if not isinstance(timing, dict) or required_timings - set(timing):
            raise RuntimeError("remote HTTP MCTS timing telemetry is incomplete")
        if any(
            isinstance(timing[name], bool)
            or not isinstance(timing[name], (int, float))
            or not math.isfinite(timing[name])
            or timing[name] < 0
            for name in required_timings
        ):
            raise RuntimeError("remote HTTP MCTS timing telemetry is invalid")
        batch_size = timing.get("batch_size")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 64
        ):
            raise RuntimeError("remote HTTP MCTS batch telemetry is invalid")
        if resources.get("worker_processes") != 16:
            raise RuntimeError("remote HTTP MCTS worker identity mismatch")
    if response.get("ok") is not True:
        error = response.get("error") or {}
        raise RuntimeError(f"remote MCTS failed: {error.get('kind', 'unknown error')}")
    return response


def _remote_mcts_batch(state_strings: list[str], search_time_ms: int, threads: int):
    requests = []
    for index, state_string in enumerate(state_strings):
        requests.append(
            {
                "schema": REMOTE_MCTS_SCHEMA,
                "operation": "search",
                "request_id": _deterministic_request_id(
                    "selection-search-request", index
                ),
                "index": index,
                "state": state_string,
                "duration_ms": int(search_time_ms),
                "threads": int(threads),
                "s1_priors": [list(row) for row in (_PRIOR_STATE["priors"] or [])]
                or None,
                "s2_priors": [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])]
                or None,
                "c_puct": float(_PRIOR_STATE["cpuct"]),
            }
        )
    started = time.monotonic()
    responses = _remote_mcts_call(requests)
    rpc_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote MCTS returned the wrong batch size")
    import poke_engine

    results = []
    timings = []
    for request, response in zip(requests, responses, strict=True):
        validated = _validate_remote_response(
            response, request["request_id"], request["index"]
        )
        results.append(_mcts_result_from_payload(validated.get("result"), poke_engine))
        timings.append(validated.get("timing"))
    _PRIOR_STATE["remote_search"] = {
        "rpc_ms": rpc_ms,
        "worlds": len(requests),
        "engine": responses[0].get("engine"),
        "state_hashes": [state_sha256(state) for state in state_strings],
        "request_ids": [request["request_id"] for request in requests],
        "timings": timings,
    }
    return results


def _remote_mcts_ensemble_batch(
    state_strings: list[str], search_time_ms: int, threads: int, repeat_count: int
):
    if repeat_count <= 0 or repeat_count > 3:
        raise RuntimeError("independent ensemble repeat count is outside its contract")
    if not state_strings or len(state_strings) * repeat_count > MAX_WIRE_BATCH_SIZE:
        raise RuntimeError("independent ensemble exceeds the remote batch bound")
    requests = []
    for repeat in range(repeat_count):
        for world_index, state_string in enumerate(state_strings):
            index = repeat * len(state_strings) + world_index
            requests.append(
                {
                    "schema": REMOTE_MCTS_SCHEMA,
                    "operation": "search",
                    "request_id": _deterministic_request_id(
                        "selection-ensemble-request", index
                    ),
                    "index": index,
                    "state": state_string,
                    "duration_ms": int(search_time_ms),
                    "threads": int(threads),
                    "s1_priors": [list(row) for row in (_PRIOR_STATE["priors"] or [])]
                    or None,
                    "s2_priors": [
                        list(row) for row in (_PRIOR_STATE["opp_priors"] or [])
                    ]
                    or None,
                    "c_puct": float(_PRIOR_STATE["cpuct"]),
                }
            )
    started = time.monotonic()
    responses = _remote_mcts_call(requests)
    rpc_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote ensemble MCTS returned the wrong batch size")
    import poke_engine

    results = []
    timings = []
    for request, response in zip(requests, responses, strict=True):
        validated = _validate_remote_response(
            response, request["request_id"], request["index"]
        )
        results.append(_mcts_result_from_payload(validated.get("result"), poke_engine))
        timings.append(validated.get("timing"))
    _PRIOR_STATE["remote_search"] = {
        "operation": "independent_ensemble",
        "rpc_ms": rpc_ms,
        "worlds": len(state_strings),
        "repeat_count": repeat_count,
        "searches": len(requests),
        "engine": responses[0].get("engine"),
        "state_hashes": [state_sha256(state) for state in state_strings],
        "request_ids": [request["request_id"] for request in requests],
        "timings": timings,
    }
    return results


def _ensemble_weighted_results(results, battles, repeat_count: int):
    if repeat_count <= 0 or len(results) != len(battles) * repeat_count:
        raise ValueError("ensemble results do not exactly cover every world and repeat")
    weights = []
    for _sampled, chance in battles:
        try:
            weight = float(chance)
        except (TypeError, ValueError) as exc:
            raise ValueError("ensemble world weight is invalid") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("ensemble world weight is invalid")
        weights.append(weight)
    if math.fsum(weights) <= 0:
        raise ValueError("ensemble world weights have no positive mass")
    return [
        (
            results[repeat * len(battles) + world_index],
            weights[world_index] / repeat_count,
            repeat * len(battles) + world_index,
        )
        for repeat in range(repeat_count)
        for world_index in range(len(battles))
    ]


def _adaptive_ensemble_repeat_count(world_count: int, maximum_repeats: int = 3) -> int:
    if world_count <= 0 or world_count > MAX_WIRE_BATCH_SIZE:
        raise RuntimeError("adaptive ensemble world count exceeds the wire batch bound")
    if maximum_repeats <= 0:
        raise RuntimeError("adaptive ensemble maximum repeats must be positive")
    return min(maximum_repeats, MAX_WIRE_BATCH_SIZE // world_count)


def _remote_shared_root_batch(
    state_strings: list[str],
    particle_weights: list[float],
    iterations: int,
    continuation_iterations: int,
    seed: int,
) -> dict[str, object]:
    try:
        prior_strength = float(
            os.environ.get(
                "METAGROSS_SHARED_ROOT_PRIOR_STRENGTH",
                str(DEFAULT_SHARED_ROOT_PRIOR_STRENGTH),
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "METAGROSS_SHARED_ROOT_PRIOR_STRENGTH must be numeric"
        ) from exc
    if not math.isfinite(prior_strength) or not 0 <= prior_strength <= 1_000:
        raise RuntimeError(
            "METAGROSS_SHARED_ROOT_PRIOR_STRENGTH must be finite and in [0, 1000]"
        )
    if os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") != "1":
        import poke_engine

        states = [poke_engine.State.from_string(state) for state in state_strings]
        opponent_prior = list(_PRIOR_STATE["opp_priors"] or ()) or None
        started = time.monotonic()
        native_result = poke_engine.shared_information_set_root_search(
            states,
            particle_weights,
            iterations,
            continuation_iterations,
            seed,
            prior_strength,
            list(_PRIOR_STATE["priors"] or ()) or None,
            [opponent_prior for _state in states],
        )
        result = shared_root_result_payload(
            native_result,
            expected_particles=len(states),
            expected_iterations=iterations,
            expected_continuation_iterations=continuation_iterations,
            expected_seed=seed,
            expected_prior_strength=prior_strength,
        )
        _PRIOR_STATE["remote_search"] = {
            "operation": "shared_root",
            "transport": "local",
            "rpc_ms": round((time.monotonic() - started) * 1000, 3),
            "worlds": len(state_strings),
            "state_hashes": [state_sha256(state) for state in state_strings],
            "particle_weights": particle_weights,
            "request_ids": [],
            "timings": [],
            "solver_diagnostics": result["diagnostics"],
        }
        return result
    request_id = _deterministic_request_id("shared-root-request", 0)
    opponent_prior = [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])] or None
    request = {
        "schema": REMOTE_MCTS_SCHEMA,
        "operation": "shared_root",
        "request_id": request_id,
        "index": 0,
        "states": state_strings,
        "particle_weights": particle_weights,
        "iterations": iterations,
        "continuation_iterations": continuation_iterations,
        "seed": seed,
        "prior_strength": prior_strength,
        "s1_prior": [list(row) for row in (_PRIOR_STATE["priors"] or [])] or None,
        "s2_priors": [opponent_prior for _state in state_strings],
    }
    started = time.monotonic()
    responses = _remote_mcts_call([request])
    rpc_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != 1:
        raise RuntimeError("remote shared-root solve returned the wrong batch size")
    validated = _validate_remote_response(responses[0], request_id, 0)
    try:
        result = validate_shared_root_result_payload(
            validated.get("result"),
            expected_particles=len(state_strings),
            expected_iterations=iterations,
            expected_continuation_iterations=continuation_iterations,
            expected_seed=seed,
            expected_prior_strength=prior_strength,
            require_replay_capture=True,
        )
    except ValueError as exc:
        raise RuntimeError(f"remote shared-root result is invalid: {exc}") from exc
    _PRIOR_STATE["remote_search"] = {
        "operation": "shared_root",
        "rpc_ms": rpc_ms,
        "worlds": len(state_strings),
        "engine": responses[0].get("engine"),
        "state_hashes": [state_sha256(state) for state in state_strings],
        "particle_weights": particle_weights,
        "request_ids": [request_id],
        "timings": [validated.get("timing")],
        "solver_diagnostics": result["diagnostics"],
    }
    return result


def _remote_holdout_batch(
    state_strings: list[str],
    baseline: str,
    candidate: str,
    continuation_steps: int,
    seeds: list[int],
    candidate_rank: int,
    *,
    request_channel: str = "certification-request",
    telemetry_key: str = "holdout",
) -> list[dict[str, object]]:
    if len(seeds) != len(state_strings):
        raise ValueError("holdout seeds must exactly cover their worlds")
    requests = [
        {
            "schema": REMOTE_MCTS_SCHEMA,
            "operation": "paired_holdout",
            "request_id": _deterministic_request_id(
                request_channel,
                f"{candidate_rank}:{continuation_steps}:{index}",
            ),
            "index": index,
            "state": state_string,
            "baseline_action": baseline,
            "candidate_action": candidate,
            "rollouts": HOLDOUT_ROLLOUTS,
            "continuation_iterations": HOLDOUT_CONTINUATION_ITERATIONS,
            "continuation_steps": continuation_steps,
            "seed": seeds[index],
            "opponent_priors": [
                list(row) for row in (_PRIOR_STATE["opp_priors"] or [])
            ]
            or None,
        }
        for index, state_string in enumerate(state_strings)
    ]
    started = time.monotonic()
    responses = _remote_mcts_call(requests)
    rpc_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote holdout returned the wrong batch size")
    results = []
    timings = []
    for request, response in zip(requests, responses, strict=True):
        validated = _validate_remote_response(
            response, request["request_id"], request["index"]
        )
        try:
            results.append(
                validate_holdout_result_payload(
                    validated.get("result"),
                    expected_pairs=request["rollouts"],
                    maximum_executed=(
                        2
                        * request["rollouts"]
                        * request["continuation_iterations"]
                        * request["continuation_steps"]
                    ),
                )
            )
        except ValueError as exc:
            raise RuntimeError(f"remote holdout result is invalid: {exc}") from exc
        timings.append(validated.get("timing"))
    remote = _PRIOR_STATE.get("remote_search") or {}
    holdout_timings = remote.setdefault(telemetry_key, [])
    holdout_timings.append(
        {
            "continuation_steps": continuation_steps,
            "rpc_ms": rpc_ms,
            "worlds": len(requests),
            "candidate": candidate,
            "candidate_rank": candidate_rank,
            "seeds": seeds,
            "state_hashes": [state_sha256(state) for state in state_strings],
            "request_ids": [request["request_id"] for request in requests],
            "opponent_priors": requests[0]["opponent_priors"] if requests else None,
            "timings": timings,
        }
    )
    _PRIOR_STATE["remote_search"] = remote
    return results


def _prepare_search_battles(battle, search_main, sampling_channel: str | None = None):
    # The decision-boundary ledger must already be frozen before any belief
    # completion. Deepcopy/sampling may populate raw hidden fields, but those
    # fields are never allowed to create reveal authorization.
    freeze_and_attach_battle_ledger(battle)
    sampled_battle = search_main.deepcopy(battle)
    if sampled_battle.team_preview:
        sampled_battle.user.active = sampled_battle.user.reserve.pop(0)
        sampled_battle.opponent.active = sampled_battle.opponent.reserve.pop(0)
    def sample(rng=None):
        if sampled_battle.battle_type == search_main.BattleType.RANDOM_BATTLE:
            count, duration = search_main.search_time_num_battles_randombattles(
                sampled_battle
            )
            worlds = search_main.prepare_random_battles(
                sampled_battle, count, rng=rng
            )
        elif sampled_battle.battle_type == search_main.BattleType.BATTLE_FACTORY:
            count, duration = search_main.search_time_num_battles_standard_battle(
                sampled_battle
            )
            worlds = search_main.prepare_random_battles(
                sampled_battle, count, rng=rng
            )
        elif sampled_battle.battle_type == search_main.BattleType.STANDARD_BATTLE:
            count, duration = search_main.search_time_num_battles_standard_battle(
                sampled_battle
            )
            worlds = search_main.prepare_battles(sampled_battle, count)
        else:
            raise ValueError("Unsupported battle type")
        return worlds, count, duration

    if sampling_channel is None:
        battles, num_battles, search_time_ms = sample()
    else:
        sampling_seed = _derived_seed(sampling_channel, 0, required=True)
        assert sampling_seed is not None
        if sampled_battle.battle_type in {
            search_main.BattleType.RANDOM_BATTLE,
            search_main.BattleType.BATTLE_FACTORY,
        }:
            battles, num_battles, search_time_ms = sample(random.Random(sampling_seed))
        else:
            with seeded_global_random(sampling_seed):
                battles, num_battles, search_time_ms = sample()
    verify_sampled_ledgers(sampled_battle, battles)
    verify_sampled_move_states(sampled_battle, battles)
    return battles, num_battles, search_time_ms


def prepare_production_random_battles_with_causal_move_receipts(
    sampler, battle, num_battles: int, rng=None
):
    """Attach the frozen causal move contract to vendor production worlds.

    The vendor production path samples internally in ``find_best_move`` rather
    than through ``_prepare_search_battles``. This wrapper is deliberately
    post-sampling and pre-conversion: it may verify and attach audit sidecars,
    but it cannot change the sampler's worlds, order, weights, or mechanics.
    """
    worlds = sampler(battle, num_battles, rng=rng)
    original_ids = [id(world) for world, _weight in worlds]
    original_weights = [weight for _world, weight in worlds]
    verify_sampled_ledgers(battle, worlds)
    verify_sampled_move_states(battle, worlds)
    if (
        [id(world) for world, _weight in worlds] != original_ids
        or [weight for _world, weight in worlds] != original_weights
    ):
        raise RuntimeError("causal production sampler wrapper changed worlds or weights")
    return worlds


def root_search_mode() -> str:
    mode = os.environ.get("METAGROSS_ROOT_SEARCH_MODE", DEFAULT_ROOT_SEARCH_MODE)
    if mode not in ROOT_SEARCH_MODES:
        raise RuntimeError(f"unsupported METAGROSS_ROOT_SEARCH_MODE: {mode}")
    if mode == "shared_rm_plus":
        if os.environ.get("METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT") != "1":
            raise RuntimeError(
                "shared RM+ requires METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT=1"
            )
        controller_mode = os.environ.get(
            "METAGROSS_CONTROLLER_MODE", DEFAULT_CONTROLLER_MODE
        )
        if controller_mode != "search_first":
            raise RuntimeError("shared RM+ requires the search_first controller")
    if mode == "independent_ensemble":
        if os.environ.get("METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE") != "1":
            raise RuntimeError(
                "independent ensemble requires METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE=1"
            )
        if os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") != "1":
            raise RuntimeError("independent ensemble requires remote MCTS")
        if os.environ.get("METAGROSS_ALLOW_INSECURE_LOOPBACK") != "1":
            raise RuntimeError("independent ensemble is restricted to loopback experiments")
        if positive_environment_int("METAGROSS_INDEPENDENT_ENSEMBLE_REPEATS", 3) != 3:
            raise RuntimeError("independent ensemble requires exactly three repeats")
        controller_mode = os.environ.get(
            "METAGROSS_CONTROLLER_MODE", DEFAULT_CONTROLLER_MODE
        )
        if controller_mode != "search_first":
            raise RuntimeError("independent ensemble requires the search_first controller")
    return mode


def _remote_find_best_move(battle, search_main, harness: DecisionHarness):
    battles, num_battles, search_time_ms = harness.belief.expand(
        battle, search_main, "selection-worlds"
    )

    search_main.logger.info("Searching for a move using remote MCTS...")
    search_main.logger.info(
        "Sampling %s battles at %sms each", num_battles, search_time_ms
    )
    states = [
        search_main.battle_to_poke_engine_state(sampled).to_string()
        for sampled, _chance in battles
    ]
    from config import FoulPlayConfig

    mode = root_search_mode()
    if mode == "shared_rm_plus":
        weights = []
        for _sampled, chance in battles:
            try:
                weight = float(chance)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("belief expansion returned an invalid weight") from exc
            if not math.isfinite(weight) or weight < 0:
                raise RuntimeError("belief expansion returned an invalid weight")
            weights.append(weight)
        total_weight = math.fsum(weights)
        if total_weight <= 0:
            raise RuntimeError("belief expansion returned no positive particle mass")
        normalized_weights = [weight / total_weight for weight in weights]
        iterations = positive_environment_int(
            "METAGROSS_SHARED_ROOT_ITERATIONS", DEFAULT_SHARED_ROOT_ITERATIONS
        )
        continuation_iterations = positive_environment_int(
            "METAGROSS_SHARED_ROOT_CONTINUATION_ITERATIONS",
            DEFAULT_SHARED_ROOT_CONTINUATION_ITERATIONS,
        )
        solver_seed = _derived_seed("shared-root-solver", 0, required=True)
        action_seed = _derived_seed("shared-root-action", 0, required=True)
        if solver_seed is None or action_seed is None:
            raise RuntimeError("shared-root deterministic seed derivation failed")
        result = harness.search.solve_shared_root(
            states,
            normalized_weights,
            iterations,
            continuation_iterations,
            solver_seed,
        )
        _PRIOR_STATE["remote_search"]["sampling_seed"] = _derived_seed(
            "selection-worlds", 0, required=True
        )
        _PRIOR_STATE["remote_search"]["action_seed"] = action_seed
        choice, telemetry = harness.controller.select_shared(
            battle,
            result,
            _PRIOR_STATE["priors"],
            seed=action_seed,
        )
        if os.environ.get("METAGROSS_SEARCH_DUMP"):
            replay_envelope = build_shared_root_replay_envelope(
                states=states,
                source_weights=weights,
                normalized_weights=normalized_weights,
                iterations=iterations,
                continuation_iterations=continuation_iterations,
                solver_seed=solver_seed,
                action_seed=action_seed,
                result=result,
                remote_search=_PRIOR_STATE["remote_search"],
                request_actions=request_player_actions(battle),
            )
            _append_jsonl(
                "METAGROSS_SEARCH_DUMP",
                {
                    "schema": 4,
                    "time_ns": time.time_ns(),
                    "context": _PRIOR_STATE["context"],
                    "choice": choice,
                    "original_choice": telemetry["raw_choice"],
                    "choice_override": telemetry,
                    "player_priors": _PRIOR_STATE["priors"],
                    "opponent_priors": _PRIOR_STATE["opp_priors"],
                    "remote_search": _PRIOR_STATE["remote_search"],
                    "shared_root": result,
                    "shared_root_replay": replay_envelope,
                },
            )
        return choice

    if mode == "independent_ensemble":
        maximum_repeat_count = positive_environment_int(
            "METAGROSS_INDEPENDENT_ENSEMBLE_REPEATS", 3
        )
        repeat_count = _adaptive_ensemble_repeat_count(
            len(states), maximum_repeat_count
        )
        results = _remote_mcts_ensemble_batch(
            states, search_time_ms, FoulPlayConfig.search_threads, repeat_count
        )
        _PRIOR_STATE["remote_search"]["maximum_repeat_count"] = maximum_repeat_count
        _PRIOR_STATE["remote_search"]["effective_repeat_count"] = repeat_count
        _PRIOR_STATE["remote_search"]["wire_batch_limit"] = MAX_WIRE_BATCH_SIZE
    else:
        repeat_count = 1
        results = harness.search.evaluate(
            states, search_time_ms, FoulPlayConfig.search_threads
        )
    _PRIOR_STATE["remote_search"]["sampling_seed"] = _derived_seed(
        "selection-worlds", 0, required=True
    )
    weighted = _ensemble_weighted_results(results, battles, repeat_count)
    action_seed = _derived_seed("selection-action", 0, required=True)
    _PRIOR_STATE["remote_search"]["action_seed"] = action_seed
    return _select_seeded_mcts_action(search_main, weighted, action_seed)


def _select_seeded_mcts_action(search_main, weighted: list, action_seed: int) -> str:
    with seeded_global_random(action_seed):
        return search_main.select_move_from_mcts_results(weighted)


def validate_poke_engine_provenance(provenance: dict, expected_source: Path) -> None:
    source = Path(provenance["source_path"]).resolve()
    expected_source = expected_source.resolve()
    if not source.is_relative_to(expected_source):
        raise RuntimeError(
            f"poke-engine source mismatch: expected {expected_source}, found {source}"
        )
    if provenance["editable"]:
        raise RuntimeError("production poke-engine must not be an editable install")
    parameters = provenance["mcts_parameters"]
    required = {"state", "duration_ms", "threads", "s1_priors", "s2_priors", "c_puct"}
    missing = required - set(parameters)
    if missing:
        raise RuntimeError(
            f"production poke-engine missing MCTS parameters: {sorted(missing)}"
        )
    if "seed" in parameters:
        raise RuntimeError("experimental poke-engine MCTS signature detected")
    if provenance.get("shared_root_parameters") != [
        "states",
        "particle_weights",
        "iterations",
        "continuation_iterations",
        "seed",
        "prior_strength",
        "s1_prior",
        "s2_priors",
    ]:
        raise RuntimeError("production poke-engine shared-root signature mismatch")


def inspect_poke_engine() -> dict:
    root = Path(__file__).resolve().parents[2]
    expected_source = root / "srcs" / "vendor" / "poke-engine"
    import poke_engine

    native_module = importlib.import_module("poke_engine.poke_engine")
    native_path = Path(native_module.__file__).resolve()
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    pinned_root_raw = os.environ.get("METAGROSS_PINNED_ENGINE_IMPORT_ROOT")
    pinned_sha256 = os.environ.get("METAGROSS_PINNED_ENGINE_SHA256")
    if bool(pinned_root_raw) != bool(pinned_sha256):
        raise RuntimeError("pinned engine provenance contract is incomplete")
    if pinned_root_raw and pinned_sha256:
        pinned_root = Path(pinned_root_raw).expanduser().resolve()
        module_path = Path(poke_engine.__file__).resolve()
        if not module_path.is_relative_to(pinned_root) or not native_path.is_relative_to(
            pinned_root
        ):
            raise RuntimeError("pinned engine escaped isolated import root")
        if native_sha256 != pinned_sha256.lower():
            raise RuntimeError("pinned engine native SHA-256 mismatch")
        state = poke_engine.State()
        if not hasattr(state, "s1_public_reveals") or not hasattr(
            state, "s2_public_reveals"
        ):
            raise RuntimeError("pinned engine lacks native reveal masks")
        root_parameters = list(
            inspect.signature(
                poke_engine.monte_carlo_tree_search_with_s1_request
            ).parameters
        )
        if root_parameters != [
            "state",
            "request_actions",
            "duration_ms",
            "iterations",
            "threads",
            "s1_priors",
            "s2_priors",
            "c_puct",
            "seed",
        ]:
            raise RuntimeError("pinned engine request-authoritative ABI mismatch")
        return {
            "distribution_version": importlib.metadata.version("poke_engine"),
            "editable": False,
            "module_path": str(module_path),
            "native_path": str(native_path),
            "native_sha256": native_sha256,
            "source_path": None,
            "isolated_import_root": str(pinned_root),
            "mcts_parameters": list(
                inspect.signature(poke_engine.monte_carlo_tree_search).parameters
            ),
            "request_root_parameters": root_parameters,
            "native_reveal_masks": True,
            "mode": "exact_pinned_experimental_runtime",
        }

    distribution = importlib.metadata.distribution("poke_engine")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_url = direct_url.get("url")
    if not source_url or urlparse(source_url).scheme != "file":
        raise RuntimeError("poke-engine install has no local source provenance")
    source_path = Path(unquote(urlparse(source_url).path)).resolve()
    provenance = {
        "distribution_version": distribution.version,
        "editable": bool((direct_url.get("dir_info") or {}).get("editable")),
        "module_path": str(Path(poke_engine.__file__).resolve()),
        "native_path": str(native_path),
        "native_sha256": native_sha256,
        "source_path": str(source_path),
        "mcts_parameters": list(
            inspect.signature(poke_engine.monte_carlo_tree_search).parameters
        ),
        "shared_root_parameters": list(
            inspect.signature(
                poke_engine.shared_information_set_root_search
            ).parameters
        ),
    }
    validate_poke_engine_provenance(provenance, expected_source)
    return provenance


def is_loopback_websocket_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"ws", "wss"} and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


def require_local_ensemble_websocket(uri: str) -> None:
    if root_search_mode() == "independent_ensemble" and not is_loopback_websocket_uri(uri):
        raise RuntimeError("independent ensemble requires a loopback Showdown websocket")


def positive_environment_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


def positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0 or (raw is not None and str(value) != raw):
        raise RuntimeError(f"{name} must be a positive integer")
    return value


class ShowdownLivenessError(TimeoutError):
    pass


def websocket_connect_kwargs(source: dict[str, object]) -> dict[str, object]:
    kwargs = dict(source)
    if os.environ.get("METAGROSS_WEBSOCKET_KEEPALIVE") == "1":
        kwargs.setdefault(
            "ping_interval",
            positive_environment_seconds(
                "METAGROSS_WEBSOCKET_PING_INTERVAL_SECONDS", 20.0
            ),
        )
        kwargs.setdefault(
            "ping_timeout",
            positive_environment_seconds(
                "METAGROSS_WEBSOCKET_PING_TIMEOUT_SECONDS", 60.0
            ),
        )
        kwargs.setdefault("close_timeout", 10)
    else:
        kwargs.setdefault("ping_interval", None)
    return kwargs


async def receive_websocket_message(receive) -> str:
    receive_timeout = positive_environment_seconds(
        "METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS", 120.0
    )
    try:
        return await asyncio.wait_for(receive(), timeout=receive_timeout)
    except asyncio.TimeoutError as exc:
        raise ShowdownLivenessError(
            f"no Showdown websocket message within {receive_timeout:g}s"
        ) from exc


def actionable_showdown_request(message: str) -> bool:
    for line in message.splitlines():
        if not line.startswith("|request|"):
            continue
        try:
            request = json.loads(line.removeprefix("|request|"))
        except json.JSONDecodeError:
            return False
        return isinstance(request, dict) and request.get("wait") is not True and any(
            key in request for key in ("active", "forceSwitch", "teamPreview")
        )
    return False


def showdown_request_payload(message: str) -> dict | None:
    for line in reversed(message.splitlines()):
        if not line.startswith("|request|"):
            continue
        try:
            payload = json.loads(line.removeprefix("|request|"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Showdown sent malformed request JSON") from exc
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise RuntimeError("Showdown request payload is not an object")
        return payload
    return None


def has_showdown_request(message: str) -> bool:
    return any(line.startswith("|request|") for line in message.splitlines())


def showdown_sent_choice(message: str) -> str | None:
    for line in reversed(message.splitlines()):
        if line.startswith("|sentchoice|"):
            return line.removeprefix("|sentchoice|")
    return None


def showdown_choice_error(messages: list[str]) -> str | None:
    for message in messages:
        for line in message.splitlines():
            if line.startswith(
                ("|error|[Invalid choice]", "|error|[Unavailable choice]")
            ):
                return line
    return None


def outbound_choice_identity(message_list: list[str]) -> tuple[str, int]:
    if len(message_list) != 2:
        raise RuntimeError("battle command has no exact rqid")
    command, rqid_text = message_list
    if command.startswith("/choose "):
        choice = command.removeprefix("/choose ")
    elif command.startswith("/switch "):
        choice = command.removeprefix("/")
    elif command == "/choose default":
        choice = "default"
    else:
        raise RuntimeError("unsupported battle command for recovery")
    try:
        rqid = int(rqid_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("battle command has an invalid rqid") from exc
    if rqid < 0 or str(rqid) != rqid_text:
        raise RuntimeError("battle command has an invalid rqid")
    return choice, rqid


def send_recovery_action(
    request: dict | None,
    sent_choice: str | None,
    expected_choice: str,
    expected_rqid: int,
    *,
    terminal: bool = False,
) -> str:
    if terminal:
        return "terminal"
    if request is None:
        raise RuntimeError("nonterminal reconnect has no request object")
    recovered_rqid = request.get("rqid")
    if (
        isinstance(recovered_rqid, bool)
        or not isinstance(recovered_rqid, int)
        or recovered_rqid < 0
    ):
        raise RuntimeError("reconnect request has an invalid rqid")
    if recovered_rqid < expected_rqid:
        raise RuntimeError("reconnect request regressed to an older rqid")
    if request.get("wait") is True:
        if sent_choice is not None:
            raise RuntimeError("wait request unexpectedly has a pending choice")
        return "wait"
    if recovered_rqid > expected_rqid:
        if sent_choice is not None:
            raise RuntimeError("newer reconnect request already has a pending choice")
        return "advanced"
    if sent_choice is None:
        return "resend"
    if sent_choice != expected_choice:
        raise RuntimeError("reconnect found a different pending choice for the same rqid")
    return "confirmed"


def terminal_showdown_message(message: str) -> bool:
    return any(line.startswith(("|win|", "|tie|")) for line in message.splitlines())


def canonical_showdown_public_lines(message: str, room: str) -> list[str]:
    lines = message.splitlines()
    if not lines or lines[0] != f">{room}":
        return []
    ignored_prefixes = (
        "|request|",
        "|sentchoice|",
        "|inactive|",
        "|inactiveoff|",
        "|html|",
        "|raw|",
        "|j|",
        "|J|",
        "|l|",
        "|L|",
        "|n|",
        "|player|",
        "|title|",
        "|init|",
        "|c|",
        "|c:|",
    )
    return [
        line
        for line in lines[1:]
        if line not in ("", "|") and not line.startswith(ignored_prefixes)
    ]


def reconnect_delta_chunks(
    messages: list[str], room: str, previously_seen: list[str]
) -> tuple[list[str], list[str]]:
    replayed_lines = []
    latest_request = None
    for message in messages:
        replayed_lines.extend(canonical_showdown_public_lines(message, room))
        for line in message.splitlines()[1:]:
            if line.startswith("|request|"):
                latest_request = f">{room}\n{line}"
    terminal = any(terminal_showdown_message(message) for message in messages)
    if latest_request is None and not terminal:
        raise RuntimeError("reconnect replay has no current request")
    if replayed_lines[: len(previously_seen)] != previously_seen:
        raise RuntimeError("reconnect replay does not extend prior public history")
    delta = replayed_lines[len(previously_seen) :]
    chunks = []
    if delta:
        chunks.append(f">{room}\n" + "\n".join(delta))
    if latest_request is not None and not terminal:
        chunks.append(latest_request)
    return chunks, replayed_lines


def format_private_request_switch(battle, decision: str) -> list[str] | None:
    """Format a switch using its authoritative private-request team slot."""
    if not decision.startswith("switch "):
        return None
    request = getattr(battle, "request_json", None)
    rqid = getattr(battle, "rqid", None)
    if not isinstance(request, dict) or request.get("rqid") != rqid:
        return None
    allowed = request_player_actions(battle)
    authorized = _authorized_action_name(decision, allowed)
    if authorized is None:
        raise ValueError(f"Request does not allow switch: {decision}")
    target = _normalize_identifier(authorized.removeprefix("switch "))
    side = request.get("side", {})
    for index, pokemon in enumerate(side.get("pokemon", ()), start=1):
        details = pokemon.get("details") if isinstance(pokemon, dict) else None
        if not isinstance(details, str):
            continue
        species = _normalize_identifier(details.split(",", 1)[0])
        if species == target:
            return [f"/switch {index}", str(rqid)]
    raise ValueError(f"Request switch has no team slot: {authorized}")


def patch_foul_play_protocol() -> None:
    """Apply the protocol safeguards used by the accepted deployment."""
    import fp.run_battle as run_battle
    import fp.websocket_client as websocket_client
    import websockets

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        if isinstance(decision, str):
            # MCTS runs on sampled battle copies whose reserve can differ from
            # our actual hidden team.  The correlated private request is the
            # authority for both legality and Showdown's one-based team slot.
            formatted_switch = format_private_request_switch(battle, decision)
            if formatted_switch is not None:
                return formatted_switch
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default

    original_connect = websockets.connect

    def connect_with_safe_ping(address, *args, **kwargs):
        return original_connect(
            address, *args, **websocket_connect_kwargs(kwargs)
        )

    websocket_client.websockets.connect = connect_with_safe_ping

    async def login_without_idle_delay(self):
        """Authenticate without the upstream three-second proxy-idle window."""
        websocket_client.logger.info("Logging in...")
        client_id, challstr = await self.get_id_and_challstr()
        local_no_security = (
            os.environ.get("METAGROSS_ALLOW_INSECURE_LOOPBACK") == "1"
            and is_loopback_websocket_uri(self.address)
        )
        if local_no_security:
            await self.send_message("", [f"/trn {self.username},0,"])
            user_id = self.username
        else:
            guest_login = self.password is None
            if guest_login:
                response = websocket_client.requests.post(
                    self.login_uri,
                    data={
                        "act": "getassertion",
                        "userid": self.username,
                        "challstr": "|".join([client_id, challstr]),
                    },
                    timeout=positive_environment_seconds(
                        "METAGROSS_SHOWDOWN_LOGIN_TIMEOUT_SECONDS", 10.0
                    ),
                )
            else:
                response = websocket_client.requests.post(
                    self.login_uri,
                    data={
                        "name": self.username,
                        "pass": self.password,
                        "challstr": "|".join([client_id, challstr]),
                    },
                    timeout=positive_environment_seconds(
                        "METAGROSS_SHOWDOWN_LOGIN_TIMEOUT_SECONDS", 10.0
                    ),
                )
            if response.status_code != 200:
                raise websocket_client.LoginError("Could not get assertion")
            if guest_login:
                assertion = response.text
                user_id = self.username
            else:
                response_json = websocket_client.json.loads(response.text[1:])
                if "actionsuccess" not in response_json:
                    raise websocket_client.LoginError(f"Could not log-in: {response_json}")
                assertion = response_json.get("assertion")
                user_id = response_json["curuser"]["userid"]
            await self.send_message("", [f"/trn {self.username},0,{assertion}"])

        timeout = positive_environment_seconds(
            "METAGROSS_SHOWDOWN_LOGIN_TIMEOUT_SECONDS", 10.0
        )
        while True:
            try:
                message = await asyncio.wait_for(self.receive_message(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise websocket_client.LoginError(
                    f"Showdown did not confirm username {self.username}"
                ) from exc
            status = showdown_login_status(message, self.username)
            if status == "confirmed":
                return user_id
            if status == "rejected":
                raise websocket_client.LoginError(
                    f"Showdown rejected username {self.username}"
                )

    websocket_client.PSWebsocketClient.login = login_without_idle_delay

    original_receive = websocket_client.PSWebsocketClient.receive_message

    def record_protocol(message: str, direction: str = "received") -> None:
        if message.startswith(">battle-"):
            _append_jsonl(
                "METAGROSS_PROTOCOL_DUMP",
                {
                    "schema": 1,
                    "time_ns": time.time_ns(),
                    "direction": direction,
                    "message": message,
                },
            )

    async def recover_and_capture(
        self, rooms: list[str], cause: BaseException, *, stop_on_any_request: bool
    ) -> tuple[list[str], list[str]]:
        reconnects = getattr(self, "metagross_reconnect_count", 0)
        max_reconnects = positive_environment_int(
            "METAGROSS_WEBSOCKET_MAX_RECONNECTS", 1
        )
        if len(rooms) != 1 or reconnects >= max_reconnects:
            raise RuntimeError(
                f"Showdown websocket recovery exhausted after {reconnects} reconnects"
            ) from cause
        self.metagross_reconnect_count = reconnects + 1
        reconnect_timeout = positive_environment_seconds(
            "METAGROSS_WEBSOCKET_RECONNECT_TIMEOUT_SECONDS", 30.0
        )
        try:
            await asyncio.wait_for(reconnect(self, rooms), timeout=reconnect_timeout)
        except asyncio.TimeoutError as reconnect_exc:
            raise RuntimeError(
                f"Showdown websocket reconnect exceeded {reconnect_timeout:g}s"
            ) from reconnect_exc
        captured = []
        while True:
            captured_message = await receive_websocket_message(
                lambda: original_receive(self)
            )
            record_protocol(captured_message, "reconnect_received")
            captured.append(captured_message)
            if terminal_showdown_message(captured_message) or (
                stop_on_any_request and has_showdown_request(captured_message)
            ) or (
                not stop_on_any_request
                and actionable_showdown_request(captured_message)
            ):
                break
        histories = getattr(self, "metagross_public_history", {})
        chunks, replayed_lines = reconnect_delta_chunks(
            captured, rooms[0], histories.get(rooms[0], [])
        )
        histories[rooms[0]] = replayed_lines
        self.metagross_public_history = histories
        return captured, chunks

    async def receive_with_rating_log(self):
        queued = getattr(self, "metagross_reconnect_message_queue", [])
        if queued:
            message = queued.pop(0)
            record_protocol(message)
            return message
        try:
            if getattr(self, "metagross_fault_stall_next_receive", False):
                self.metagross_fault_stall_next_receive = False

                async def injected_stall():
                    await asyncio.Event().wait()

                message = await receive_websocket_message(injected_stall)
            else:
                message = await receive_websocket_message(
                    lambda: original_receive(self)
                )
        except (ShowdownLivenessError, websockets.exceptions.ConnectionClosed) as exc:
            rooms = sorted(getattr(self, "metagross_active_battle_rooms", set()))
            _captured, chunks = await recover_and_capture(
                self, rooms, exc, stop_on_any_request=False
            )
            self.metagross_reconnect_message_queue = chunks[1:]
            record_protocol(chunks[0])
            return chunks[0]
        record_protocol(message)
        lines = message.splitlines()
        if lines and lines[0].startswith(">battle-"):
            room = lines[0].removeprefix(">")
            histories = getattr(self, "metagross_public_history", {})
            histories.setdefault(room, []).extend(
                canonical_showdown_public_lines(message, room)
            )
            self.metagross_public_history = histories
        for line in message.splitlines():
            if line.startswith("|raw|") and (
                "<strong>" in line or "rating:" in line.lower()
            ):
                print(f"RATING_LINE {line}", flush=True)
        return message

    websocket_client.PSWebsocketClient.receive_message = receive_with_rating_log

    original_send = websocket_client.PSWebsocketClient.send_message

    async def send_with_battle_log(self, room, message_list):
        battle_command = room.startswith("battle-") and bool(message_list) and (
            message_list[0].startswith("/choose move ")
            or message_list[0].startswith("/switch ")
            or message_list[0] == "/choose default"
        )
        if room.startswith("battle-"):
            rooms = getattr(self, "metagross_active_battle_rooms", set())
            rooms.add(room)
            self.metagross_active_battle_rooms = rooms
        send_timeout = positive_environment_seconds(
            "METAGROSS_WEBSOCKET_SEND_TIMEOUT_SECONDS", 10.0
        )
        try:
            prospective_count = (
                getattr(self, "metagross_battle_command_count", 0) + 1
            )
            unacknowledged_fault = os.environ.get(
                "METAGROSS_FAULT_SEND_UNACKNOWLEDGED_AFTER_BATTLE_COMMANDS"
            )
            acknowledged_fault = os.environ.get(
                "METAGROSS_FAULT_SEND_ACKNOWLEDGED_AFTER_BATTLE_COMMANDS"
            )
            if battle_command and (unacknowledged_fault or acknowledged_fault):
                if not is_loopback_websocket_uri(self.address):
                    raise RuntimeError("websocket fault injection is loopback-only")
            if (
                battle_command
                and unacknowledged_fault
                and prospective_count
                == positive_environment_int(
                    "METAGROSS_FAULT_SEND_UNACKNOWLEDGED_AFTER_BATTLE_COMMANDS", 1
                )
            ):
                await self.websocket.close(code=1012, reason="send fault injection")
                raise asyncio.TimeoutError
            result = await asyncio.wait_for(
                original_send(self, room, message_list), timeout=send_timeout
            )
            if (
                battle_command
                and acknowledged_fault
                and prospective_count
                == positive_environment_int(
                    "METAGROSS_FAULT_SEND_ACKNOWLEDGED_AFTER_BATTLE_COMMANDS", 1
                )
            ):
                await self.websocket.close(code=1012, reason="send fault injection")
                raise asyncio.TimeoutError
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as exc:
            if room.startswith("battle-"):
                _append_jsonl(
                    "METAGROSS_PROTOCOL_DUMP",
                    {
                        "schema": 1,
                        "time_ns": time.time_ns(),
                        "direction": "send_failure",
                        "room": room,
                        "messages": list(message_list),
                        "error": type(exc).__name__,
                    },
                )
            if not battle_command:
                raise RuntimeError(
                    "ambiguous Showdown command delivery outside a recoverable choice"
                ) from exc
            rooms = sorted(getattr(self, "metagross_active_battle_rooms", set()))
            captured, chunks = await recover_and_capture(
                self, rooms, exc, stop_on_any_request=True
            )
            terminal = any(terminal_showdown_message(message) for message in captured)
            request = None if terminal else next(
                (
                    showdown_request_payload(message)
                    for message in reversed(captured)
                    if has_showdown_request(message)
                ),
                None,
            )
            sent_choice = next(
                (
                    showdown_sent_choice(message)
                    for message in reversed(captured)
                    if showdown_sent_choice(message) is not None
                ),
                None,
            )
            expected_choice, expected_rqid = outbound_choice_identity(message_list)
            recovered_rqid = request.get("rqid") if isinstance(request, dict) else None
            recovery_action = send_recovery_action(
                request,
                sent_choice,
                expected_choice,
                expected_rqid,
                terminal=terminal,
            )
            choice_error = showdown_choice_error(captured)
            if choice_error is not None and recovery_action in {
                "resend",
                "confirmed",
            }:
                raise RuntimeError(
                    "same-rqid reconnect contained a choice rejection"
                ) from exc
            public_chunks = [
                chunk for chunk in chunks if not has_showdown_request(chunk)
            ]
            self.metagross_reconnect_message_queue = public_chunks
            if recovery_action == "wait":
                self.metagross_reconnect_message_queue = chunks
                result = None
                direction = "sent" if choice_error is None else "send_rejected"
                _append_jsonl(
                    "METAGROSS_PROTOCOL_DUMP",
                    {
                        "schema": 1,
                        "time_ns": time.time_ns(),
                        "direction": direction,
                        "room": room,
                        "messages": list(message_list),
                        "recovery": "advanced_to_wait_request",
                        "choice_error": choice_error,
                    },
                )
            elif recovery_action in {"resend", "confirmed"}:
                if recovery_action == "resend":
                    try:
                        result = await asyncio.wait_for(
                            original_send(self, room, message_list), timeout=send_timeout
                        )
                    except (
                        asyncio.TimeoutError,
                        websockets.exceptions.ConnectionClosed,
                    ) as retry_exc:
                        raise RuntimeError(
                            "exact Showdown command resend failed"
                        ) from retry_exc
                    recovery = "resent_after_missing_sentchoice"
                else:
                    result = None
                    recovery = "confirmed_by_sentchoice"
                _append_jsonl(
                    "METAGROSS_PROTOCOL_DUMP",
                    {
                        "schema": 1,
                        "time_ns": time.time_ns(),
                        "direction": "sent",
                        "room": room,
                        "messages": list(message_list),
                        "recovery": recovery,
                    },
                )
            else:
                self.metagross_reconnect_message_queue = chunks
                result = None
                direction = "sent" if choice_error is None else "send_rejected"
                _append_jsonl(
                    "METAGROSS_PROTOCOL_DUMP",
                    {
                        "schema": 1,
                        "time_ns": time.time_ns(),
                        "direction": direction,
                        "room": room,
                        "messages": list(message_list),
                        "recovery": f"{recovery_action}_after_reconnect",
                        "recovered_rqid": recovered_rqid,
                        "choice_error": choice_error,
                    },
                )
        else:
            if room.startswith("battle-"):
                _append_jsonl(
                    "METAGROSS_PROTOCOL_DUMP",
                    {
                        "schema": 1,
                        "time_ns": time.time_ns(),
                        "direction": "sent",
                        "room": room,
                        "messages": list(message_list),
                    },
                )
        if room.startswith("battle-"):
            if battle_command:
                count = getattr(self, "metagross_battle_command_count", 0) + 1
                self.metagross_battle_command_count = count
                disconnect_after = os.environ.get(
                    "METAGROSS_FAULT_DISCONNECT_AFTER_BATTLE_COMMANDS"
                )
                stall_after = os.environ.get(
                    "METAGROSS_FAULT_STALL_AFTER_BATTLE_COMMANDS"
                )
                if disconnect_after or stall_after:
                    if not is_loopback_websocket_uri(self.address):
                        raise RuntimeError("websocket fault injection is loopback-only")
                if disconnect_after and count == positive_environment_int(
                    "METAGROSS_FAULT_DISCONNECT_AFTER_BATTLE_COMMANDS", 1
                ):
                    await self.websocket.close(code=1012, reason="fault injection")
                if stall_after and count == positive_environment_int(
                    "METAGROSS_FAULT_STALL_AFTER_BATTLE_COMMANDS", 1
                ):
                    self.metagross_fault_stall_next_receive = True
        return result

    websocket_client.PSWebsocketClient.send_message = send_with_battle_log

    original_pokemon_battle = run_battle.pokemon_battle

    async def reconnect(self, rooms: list[str]) -> None:
        websocket_client.logger.warning(
            "Reconnecting Showdown websocket and rebuilding rooms: %s", rooms
        )
        try:
            await asyncio.wait_for(self.websocket.close(), timeout=5)
        except Exception:
            pass
        replacement = await websocket_client.PSWebsocketClient.create(
            self.username, self.password, self.address
        )
        self.websocket = replacement.websocket
        self.login_uri = replacement.login_uri
        await self.login()
        for room in rooms:
            await self.send_message("", [f"/join {room}"])
        _append_jsonl(
            "METAGROSS_PROTOCOL_DUMP",
            {
                "schema": 1,
                "time_ns": time.time_ns(),
                "direction": "reconnect",
                "rooms": rooms,
            },
        )

    async def pokemon_battle_with_reconnect(
        ps_websocket_client, pokemon_battle_type, team_dict
    ):
        try:
            return await original_pokemon_battle(
                ps_websocket_client, pokemon_battle_type, team_dict
            )
        except RuntimeError as exc:
            # Two per-battle failures should abandon just this battle (forfeit +
            # leave, count a loss, keep laddering) instead of crashing the whole
            # block — matching how the deployed causal stack was operated
            # (auto-resume). (1) A relayed (Tailscale/DERP) websocket drops
            # mid-battle and the reconnect integrity guard refuses the replay.
            # (2) A causal-ledger edge case (e.g. ability-changing Transform)
            # the ledger cannot represent. Both guards are CORRECT to refuse the
            # unverifiable/unsupported battle; the causal-history guarantee is
            # preserved (we never play on history we could not build/verify).
            # ABANDON_REASON is logged so the forfeit rate is measurable, and
            # per-arm so the causal-only ledger forfeits are not conflated with
            # the relay forfeits both arms share.
            is_reconnect = "reconnect replay" in str(exc)
            is_ledger = isinstance(exc, CausalRevealLedgerError)
            if not (is_reconnect or is_ledger):
                raise
            reason = "reconnect" if is_reconnect else "ledger"
            logger.warning(
                "ABANDON_BATTLE reason=%s (%s); forfeiting and continuing the ladder",
                reason, exc,
            )
            rooms = set(getattr(
                ps_websocket_client, "metagross_active_battle_rooms", set()))
            for room in rooms:
                for command in ("/forfeit", "/leave"):
                    try:
                        await ps_websocket_client.send_message(room, [command])
                    except Exception:
                        pass
            return None
        finally:
            ps_websocket_client.metagross_active_battle_rooms = set()
            ps_websocket_client.metagross_reconnect_count = 0

    run_battle.pokemon_battle = pokemon_battle_with_reconnect


def _mcts_with_root_priors(state_str, search_time_ms, index, threads=1):
    """Run the patched engine with player and opponent root priors."""
    import poke_engine
    from config import FoulPlayConfig

    if os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") == "1":
        request_id = _deterministic_request_id("search-request", int(index))
        request = {
            "schema": REMOTE_MCTS_SCHEMA,
            "operation": "search",
            "request_id": request_id,
            "index": int(index),
            "state": state_str,
            "duration_ms": int(search_time_ms),
            "threads": int(FoulPlayConfig.search_threads),
            "s1_priors": [list(row) for row in (_PRIOR_STATE["priors"] or [])] or None,
            "s2_priors": [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])]
            or None,
            "c_puct": float(_PRIOR_STATE["cpuct"]),
        }
        if os.environ.get("METAGROSS_REMOTE_MCTS_TRANSPORT", "modal") == "http":
            responses = _remote_mcts_call([request])
            if not isinstance(responses, list) or len(responses) != 1:
                raise RuntimeError("remote MCTS returned the wrong batch size")
            response = responses[0]
        else:
            response = _remote_mcts_call(request)
        validated = _validate_remote_response(response, request_id, int(index))
        return _mcts_result_from_payload(validated.get("result"), poke_engine)

    state = poke_engine.State.from_string(state_str)
    kwargs = {}
    if _PRIOR_STATE["priors"]:
        kwargs["s1_priors"] = _PRIOR_STATE["priors"]
        kwargs["c_puct"] = _PRIOR_STATE["cpuct"]
    if _PRIOR_STATE["opp_priors"]:
        kwargs["s2_priors"] = _PRIOR_STATE["opp_priors"]
    # Deterministic cross-platform depth: when set, convert the wall-clock
    # budget to an exact iteration budget (scaled by the scheduler's own
    # 250/500 ms tier ratio). Absent env = unchanged production behavior.
    iterations_per_500 = os.environ.get("METAGROSS_SEARCH_ITERATIONS_PER_500MS")
    if iterations_per_500:
        return poke_engine.monte_carlo_tree_search(
            state,
            0,
            iterations=max(1, int(int(iterations_per_500) * search_time_ms / 500)),
            threads=FoulPlayConfig.search_threads,
            **kwargs,
        )
    return poke_engine.monte_carlo_tree_search(
        state,
        search_time_ms,
        threads=FoulPlayConfig.search_threads,
        **kwargs,
    )


def build_decision_harness() -> DecisionHarness:
    """Compose the production adapters without changing their algorithms."""
    import urllib.request
    from urllib.parse import quote

    server_url = os.environ.get("METAGROSS_PRIOR_SERVER")
    if not server_url:
        raise RuntimeError("METAGROSS_PRIOR_SERVER is required")
    namespace = os.environ.get("METAGROSS_PRIOR_NAMESPACE", "")

    def observe(tag: str, lines: list[str]) -> None:
        request = urllib.request.Request(
            f"{server_url}/lines",
            data=json.dumps(
                {"tag": tag, "namespace": namespace, "lines": lines}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5.0):
            pass

    def propose(battle) -> PolicySnapshot:
        tag = getattr(battle, "battle_tag", None)
        if not tag:
            raise RuntimeError("battle has no tag")
        full_tag = tag if tag.startswith("battle-") else f"battle-{tag}"
        from config import FoulPlayConfig

        username = quote(str(getattr(FoulPlayConfig, "username", "") or ""))
        (_identity_tag, rqid), request_sha256 = battle_request_identity(battle)
        with urllib.request.urlopen(
            f"{server_url}/priors?tag={full_tag}"
            f"&username={username}&namespace={quote(namespace)}&rqid={rqid}"
            f"&request_sha256={request_sha256}",
            timeout=30,
        ) as response:
            payload = json.loads(response.read())
        priors = payload.get("priors") or {}
        if not priors:
            raise RuntimeError("policy server returned no player priors")
        if payload.get("rqid") != rqid:
            raise RuntimeError("policy server returned stale request priors")
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("policy server returned priors for a different request")
        opponent_priors = sanitize_opponent_priors(
            battle, payload.get("opp_priors") or {}
        )
        return PolicySnapshot(
            priors=tuple(
                (name, float(probability)) for name, probability in priors.items()
            ),
            opponent_priors=(
                tuple(opponent_priors) if opponent_priors is not None else None
            ),
            context={
                "tag": full_tag,
                "decision_idx": payload.get("decision_idx"),
                "battle_turn": payload.get("battle_turn"),
                "rqid": payload.get("rqid"),
                "request_sha256": payload.get("request_sha256"),
                "own_legality": payload.get("own_legality"),
                "opponent_support": payload.get("opponent_support"),
                "r1_policy_snapshot": payload.get("r1_policy_snapshot"),
            },
        )

    def acknowledge(snapshot: PolicySnapshot, action: str) -> None:
        context = snapshot.context
        request = urllib.request.Request(
            f"{server_url}/action",
            data=json.dumps(
                {
                    "tag": context.get("tag"),
                    "namespace": namespace,
                    "rqid": context.get("rqid"),
                    "request_sha256": context.get("request_sha256"),
                    "decision_idx": context.get("decision_idx"),
                    "action": action,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            result = json.loads(response.read())
        if result.get("ok") is not True:
            raise RuntimeError("policy server did not acknowledge selected action")

    return DecisionHarness(
        policy=CallablePolicy(observe, propose, acknowledge),
        belief=CallableBelief(_prepare_search_battles),
        search=CallableSearch(
            _remote_mcts_batch,
            _remote_holdout_batch,
            _remote_shared_root_batch,
        ),
        controller=CallableController(
            controller_select_fn(),
            select_shared_root_choice,
        ),
        verifier=CallableVerifier(
            robust_holdout_certificate, combined_robust_holdout_certificate
        ),
    )


def _terminal_mcts_teacher_enabled() -> bool:
    slot = os.environ.get("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT")
    namespace = os.environ.get("METAGROSS_PRIOR_NAMESPACE")
    return bool(slot and slot == namespace)


def _terminal_mcts_one_deviation_enabled() -> bool:
    slot = os.environ.get("METAGROSS_TERMINAL_MCTS_ONE_DEVIATION_SLOT")
    namespace = os.environ.get("METAGROSS_PRIOR_NAMESPACE")
    return bool(slot and slot == namespace)


def _terminal_mcts_one_deviation_controller() -> OneDeviationController:
    global _TERMINAL_MCTS_ONE_DEVIATION_CONTROLLER
    if _TERMINAL_MCTS_ONE_DEVIATION_CONTROLLER is None:
        seed = os.environ.get("METAGROSS_TERMINAL_MCTS_ONE_DEVIATION_SEED")
        prefix = os.environ.get("METAGROSS_TERMINAL_MCTS_ONE_DEVIATION_PREFIX")
        if not seed or not prefix:
            raise RuntimeError("one-deviation randomization configuration is incomplete")
        _TERMINAL_MCTS_ONE_DEVIATION_CONTROLLER = OneDeviationController(
            seed=seed,
            username_prefix=prefix,
            teacher_contract=os.environ.get(
                "METAGROSS_TERMINAL_MCTS_ONE_DEVIATION_CONTRACT",
                "legacy_terminal_mcts",
            ),
        )
    return _TERMINAL_MCTS_ONE_DEVIATION_CONTROLLER


def _terminal_mcts_one_deviation_identity() -> tuple[str, str, int]:
    context = _PRIOR_STATE.get("context") or {}
    tag = str(context.get("tag") or "")
    decision_index = int(context.get("decision_idx") or 0)
    snapshot = context.get("r1_policy_snapshot")
    username = str(snapshot.get("username") or "") if isinstance(snapshot, dict) else ""
    if not username:
        from config import FoulPlayConfig

        username = str(getattr(FoulPlayConfig, "username", "") or "")
    if not tag or not username:
        raise RuntimeError("one-deviation identity is incomplete")
    return tag, username, decision_index


def _terminal_mcts_teacher_decision(
    harness: DecisionHarness,
    search_main,
    battle,
    production_choice: str,
) -> dict[str, object]:
    """Call the isolated Python-3.11 exact engine without exporting worlds."""
    python = os.environ.get("METAGROSS_TERMINAL_MCTS_PYTHON")
    script = os.environ.get("METAGROSS_TERMINAL_MCTS_SCRIPT")
    pythonpath = os.environ.get("METAGROSS_TERMINAL_MCTS_PYTHONPATH")
    if not python or not script or not pythonpath:
        raise RuntimeError("terminal-MCTS teacher command is incomplete")
    global _CAUSAL_RECEIPT_CONTEXT
    context = _PRIOR_STATE.get("context") or {}
    tag = str(context.get("tag") or getattr(battle, "battle_tag", ""))
    decision_index = int(context.get("decision_idx") or 0)
    rqid = getattr(battle, "rqid", None)
    root_id = hashlib.sha256(
        f"terminal-mcts-live\0{tag}\0{decision_index}".encode("utf-8")
    ).hexdigest()
    schedules = []
    for schedule_id in (0, 1):
        sampled, _count, _duration = harness.belief.expand(
            battle, search_main, f"terminal-mcts-schedule-{schedule_id}"
        )
        if len(sampled) < 8:
            raise RuntimeError("terminal-MCTS teacher belief has fewer than eight worlds")
        selected = sampled[:8]
        masses = [float(weight) for _world, weight in selected]
        total = math.fsum(masses)
        if total <= 0 or any(not math.isfinite(weight) or weight < 0 for weight in masses):
            raise RuntimeError("terminal-MCTS teacher belief weights are invalid")
        worlds = []
        for world_index, ((world, _raw_weight), weight) in enumerate(
            zip(selected, masses, strict=True)
        ):
            previous_context = _CAUSAL_RECEIPT_CONTEXT
            _CAUSAL_RECEIPT_CONTEXT = {
                "phase": "equal8192_candidate",
                "cohort": "fixed_two_by_eight",
                "battle_tag": tag,
                "rqid": rqid,
                "decision_index": decision_index,
                "root_id": root_id,
                "declared_world_count": 16,
                "conversion_index": schedule_id * 8 + world_index,
                "schedule_index": schedule_id,
                "world_index": world_index,
            }
            try:
                state = search_main.battle_to_poke_engine_state(world).to_string()
            finally:
                _CAUSAL_RECEIPT_CONTEXT = previous_context
            worlds.append({"state": state, "weight": weight / total})
        schedules.append({"worlds": worlds})
    payload = {
        "root_id": root_id,
        "battle_id": tag,
        "production_choice": production_choice,
        "request_actions": sorted(request_player_actions(battle)),
        "seed": _derived_seed("terminal-mcts-outcome", 0, required=True),
        "schedules": schedules,
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = pythonpath
    timeout = positive_environment_seconds("METAGROSS_TERMINAL_MCTS_TIMEOUT_SECONDS", 60.0)
    completed = subprocess.run(
        [python, script, "--workers", os.environ.get("METAGROSS_TERMINAL_MCTS_WORKERS", "4")],
        input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("terminal-MCTS teacher subprocess failed")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("terminal-MCTS teacher returned invalid output framing")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("terminal-MCTS teacher returned invalid JSON") from exc
    if (
        not isinstance(result, dict)
        or result.get("schema") != "metagross-terminal-mcts-live-decision/v1"
        or result.get("selected_action") not in request_player_actions(battle)
        or result.get("decision") not in {"override", "abstain"}
    ):
        raise RuntimeError("terminal-MCTS teacher returned an invalid decision")
    return result


def automatic_request_action(own_legality: object) -> str | None:
    """Identify Showdown-only forced commands that are not policy decisions."""
    if not isinstance(own_legality, dict):
        return None
    actions = own_legality.get("actions")
    if not isinstance(actions, (list, tuple)) or len(actions) != 1:
        return None
    action = actions[0]
    return action if action in {"recharge", "struggle"} else None


def cycle30_dynamic_boundary_evidence(
    battle: object,
    own_legality: object,
    decision_index: object,
    *,
    max_decision_index: int,
    max_battle_turn: int,
) -> dict[str, object]:
    """Classify the smoke-only Cycle30 causal boundary without policy values.

    This function is deliberately read-only.  The exact private request decides
    whether a root is ordinary, while the attached causal ledger decides whether
    an intrinsic opponent move has become public.  Search output, sampled worlds
    and hidden completion fields are not consulted.
    """
    if (
        isinstance(decision_index, bool)
        or not isinstance(decision_index, int)
        or decision_index < 0
    ):
        raise RuntimeError("Cycle30 boundary has invalid decision index")
    request = getattr(battle, "request_json", None)
    if not isinstance(request, dict):
        raise RuntimeError("Cycle30 boundary has no exact private request")
    wait = request.get("wait", False)
    if not isinstance(wait, bool):
        raise RuntimeError("Cycle30 boundary has invalid wait metadata")
    force_switch_rows = request.get("forceSwitch", [False])
    if not isinstance(force_switch_rows, list) or not force_switch_rows:
        raise RuntimeError("Cycle30 boundary has invalid forceSwitch metadata")
    if any(not isinstance(value, bool) for value in force_switch_rows):
        raise RuntimeError("Cycle30 boundary has invalid forceSwitch metadata")
    force_switch = any(force_switch_rows)
    automatic = automatic_request_action(own_legality)
    battle_turn = getattr(battle, "turn", None)
    if isinstance(battle_turn, bool) or not isinstance(battle_turn, int):
        raise RuntimeError("Cycle30 boundary has invalid public battle turn")
    ledger = attached_ledger(battle)
    intrinsic_events = [
        event
        for fact in ledger.facts
        for event in fact.move_events
        if event.authority == "intrinsic_public_execution"
    ]
    ordinary = not wait and not force_switch and automatic is None
    within_bounds = (
        decision_index <= max_decision_index and battle_turn <= max_battle_turn
    )
    return {
        "eligible": ordinary and bool(intrinsic_events) and within_bounds,
        "ordinary": ordinary,
        "wait": wait,
        "force_switch": force_switch,
        "automatic_action": automatic,
        "intrinsic_opponent_move_events": len(intrinsic_events),
        "decision_index": decision_index,
        "battle_turn": battle_turn,
        "max_decision_index": max_decision_index,
        "max_battle_turn": max_battle_turn,
        "within_bounds": within_bounds,
        "protocol_sha256": ledger.protocol_sha256,
    }


def cycle31_candidate_boundary_receipt(
    battle: object,
    own_legality: object,
    decision_index: object,
    teacher: object,
    selected_action: str,
    *,
    max_decision_index: int,
    max_battle_turn: int,
) -> dict[str, object]:
    """Bind Cycle30 eligibility to a completed agent-A equal8192 result."""
    if (
        os.environ.get("METAGROSS_PRIOR_NAMESPACE") != "agent_a"
        or os.environ.get("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT") != "agent_a"
    ):
        raise RuntimeError("Cycle31 boundary is not in the agent-A candidate stream")
    if not isinstance(teacher, dict) or (
        teacher.get("controller_schema")
        != "metagross-cycle19-equal8192-production-selector/v1"
    ):
        raise RuntimeError("Cycle31 boundary lacks the frozen equal8192 controller")
    if (
        teacher.get("iterations_per_world") != 8192
        or teacher.get("schedule_count") != 2
        or teacher.get("world_count") != 16
        or str(teacher.get("reason", "")).startswith("fail_closed")
    ):
        raise RuntimeError("Cycle31 equal8192 candidate result is incomplete")
    receipts = teacher.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != 16:
        raise RuntimeError("Cycle31 equal8192 receipt count changed")
    cells = set()
    for row in receipts:
        if not isinstance(row, dict) or row.get("total_visits") != 8192:
            raise RuntimeError("Cycle31 candidate world lacks exactly 8192 visits")
        schedule = row.get("schedule_index")
        world = row.get("world_index")
        if (
            isinstance(schedule, bool)
            or not isinstance(schedule, int)
            or isinstance(world, bool)
            or not isinstance(world, int)
        ):
            raise RuntimeError("Cycle31 candidate receipt cell is invalid")
        cells.add((schedule, world))
    if cells != {(schedule, world) for schedule in range(2) for world in range(8)}:
        raise RuntimeError("Cycle31 candidate receipt cells are incomplete")
    evidence = cycle30_dynamic_boundary_evidence(
        battle,
        own_legality,
        decision_index,
        max_decision_index=max_decision_index,
        max_battle_turn=max_battle_turn,
    )
    context = _PRIOR_STATE.get("context") or {}
    tag = str(context.get("tag") or getattr(battle, "battle_tag", ""))
    rqid = getattr(battle, "rqid", None)
    root_id = hashlib.sha256(
        f"terminal-mcts-live\0{tag}\0{decision_index}".encode("utf-8")
    ).hexdigest()
    ledger = attached_ledger(battle)
    username = str(getattr(getattr(battle, "user", None), "name", "") or "")
    if (
        not tag
        or isinstance(rqid, bool)
        or not isinstance(rqid, int)
        or not username
        or ledger.battle_tag not in {tag, tag.removeprefix("battle-")}
        or ledger.protocol_sha256 != evidence["protocol_sha256"]
    ):
        raise RuntimeError("Cycle31 candidate identity is incomplete")
    return {
        "battle_tag": tag,
        "rqid": rqid,
        "decision_index": decision_index,
        "root_id": root_id,
        "selected_action": selected_action,
        "active_name": getattr(
            getattr(getattr(battle, "user", None), "active", None), "name", ""
        ),
        "decision_complete_time_ns": time.time_ns(),
        "cycle30_dynamic_boundary": evidence,
        "cycle31_candidate_attribution": {
            "namespace": "agent_a",
            "username": username,
            "observer_role": ledger.observer_role,
            "battle_tag": tag,
            "rqid": rqid,
            "decision_index": decision_index,
            "root_id": root_id,
            "protocol_sha256": ledger.protocol_sha256,
            "controller_schema": teacher["controller_schema"],
            "candidate_cells": [list(cell) for cell in sorted(cells)],
            "iterations_per_world": 8192,
        },
    }


def cycle32_authenticated_candidate_boundary_receipt(
    battle: object,
    own_legality: object,
    decision_index: object,
    teacher: object,
    selected_action: str,
    *,
    max_decision_index: int,
    max_battle_turn: int,
) -> dict[str, object]:
    """Keep Cycle31 mechanics while separating role from public username."""
    receipt = cycle31_candidate_boundary_receipt(
        battle,
        own_legality,
        decision_index,
        teacher,
        selected_action,
        max_decision_index=max_decision_index,
        max_battle_turn=max_battle_turn,
    )
    attribution = receipt["cycle31_candidate_attribution"]
    ledger = attached_ledger(battle)
    internal_role = str(
        getattr(getattr(battle, "user", None), "name", "") or ""
    )
    if internal_role not in {"p1", "p2"} or internal_role != ledger.observer_role:
        raise RuntimeError("Cycle32 internal battle role disagrees with causal ledger")
    from config import FoulPlayConfig

    configured_username = str(getattr(FoulPlayConfig, "username", "") or "")
    if not configured_username:
        raise RuntimeError("Cycle32 spawned runtime has no configured username")
    public_names = set()
    for line in protocol_lines_for_battle(ledger.battle_tag):
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "player" and parts[2] == internal_role:
            public_names.add(parts[3])
    if (
        len(public_names) != 1
        or _normalize_identifier(next(iter(public_names)))
        != _normalize_identifier(configured_username)
    ):
        raise RuntimeError("Cycle32 public player mapping is missing or mismatched")
    attribution["internal_battle_role"] = internal_role
    attribution["external_authenticated_username"] = configured_username
    attribution["external_username_authority"] = (
        "spawned_runtime_plus_causal_public_player_line"
    )
    attribution.pop("username", None)
    return receipt


def patch_root_priors(harness: DecisionHarness | None = None) -> None:
    """Connect Foul Play's search roots to the local r1 policy server."""
    server_url = os.environ.get("METAGROSS_PRIOR_SERVER")
    if not server_url:
        raise RuntimeError("METAGROSS_PRIOR_SERVER is required")
    if harness is None:
        harness = build_decision_harness()

    import logging

    import fp.run_battle as run_battle
    import fp.search.main as search_main
    from fp.websocket_client import PSWebsocketClient

    logger = logging.getLogger("fp.root_priors")
    _PRIOR_STATE["cpuct"] = float(os.environ.get("METAGROSS_CPUCT", "2.0"))

    original_receive = PSWebsocketClient.receive_message

    async def receive_with_tee(self):
        message = await original_receive(self)
        if message.startswith(">battle-"):
            lines = message.split("\n")
            tag = lines[0].lstrip(">").strip()
            record_public_protocol_lines(tag, lines[1:])
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    harness.policy.observe,
                    tag,
                    lines[1:],
                )
            except Exception as exc:
                if os.environ.get("METAGROSS_REQUIRE_PRIORS") == "1":
                    raise RuntimeError(
                        f"required protocol tee failed: {exc!r}"
                    ) from exc
                logger.warning("prior protocol tee failed: %r", exc)
            boundary_path = (
                os.environ.get("METAGROSS_CYCLE32_BOUNDARY_RECEIPT")
                or os.environ.get("METAGROSS_CYCLE31_BOUNDARY_RECEIPT")
                or os.environ.get("METAGROSS_CYCLE30_BOUNDARY_RECEIPT")
                or os.environ.get("METAGROSS_CYCLE27_BOUNDARY_RECEIPT")
                or os.environ.get("METAGROSS_CYCLE25_BOUNDARY_RECEIPT")
            )
            boundary = _CYCLE25_EXECUTION_BOUNDARIES.get(tag)
            if boundary_path and boundary and not Path(boundary_path).exists():
                selected = str(boundary["selected_action"])
                target = _normalize_identifier(
                    selected.removeprefix("switch ").removesuffix("-tera")
                )
                active = _normalize_identifier(boundary.get("active_name"))
                matched_line = None
                for line in lines[1:]:
                    parts = line.split("|")
                    if selected.startswith("switch "):
                        if (
                            len(parts) >= 4
                            and parts[1] in {"switch", "drag", "replace"}
                            and _normalize_identifier(parts[3].split(",", 1)[0]) == target
                        ):
                            matched_line = line
                            break
                    elif (
                        len(parts) >= 4
                        and parts[1] == "move"
                        and _normalize_identifier(parts[2].split(":", 1)[-1]) == active
                        and _normalize_identifier(parts[3]) == target
                    ):
                        matched_line = line
                        break
                if matched_line is not None:
                    receipt = {
                        "schema": "metagross-cycle25-public-execution-boundary/v1",
                        **boundary,
                        "public_line": matched_line,
                        "public_execution_time_ns": time.time_ns(),
                    }
                    destination = Path(boundary_path).resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("x", encoding="utf-8") as handle:
                        json.dump(receipt, handle, sort_keys=True)
                        handle.write("\n")
                    # Smoke-only latch: the external monitor terminates this
                    # process before the newly received next request is parsed.
                    while (
                        os.environ.get("METAGROSS_CYCLE32_BOUNDARY_RECEIPT")
                        or os.environ.get("METAGROSS_CYCLE31_BOUNDARY_RECEIPT")
                        or os.environ.get("METAGROSS_CYCLE30_BOUNDARY_RECEIPT")
                        or os.environ.get("METAGROSS_CYCLE27_BOUNDARY_RECEIPT")
                        or os.environ.get("METAGROSS_CYCLE25_BOUNDARY_RECEIPT")
                    ):
                        time.sleep(0.05)
            if terminal_showdown_message(message):
                clear_public_protocol_lines(tag)
        return message

    PSWebsocketClient.receive_message = receive_with_tee
    original_battle_to_engine = search_main.battle_to_poke_engine_state
    original_prepare_random_battles = search_main.prepare_random_battles
    engine_module = importlib.import_module("poke_engine")

    def prepare_random_battles_with_causal_move_receipts(
        battle, num_battles, rng=None
    ):
        return prepare_production_random_battles_with_causal_move_receipts(
            original_prepare_random_battles, battle, num_battles, rng=rng
        )

    search_main.prepare_random_battles = (
        prepare_random_battles_with_causal_move_receipts
    )

    def battle_to_engine_with_causal_ledger(battle, swap=False):
        with _CAUSAL_RECEIPT_CONTEXT_LOCK:
            receipt_context = (
                dict(_CAUSAL_RECEIPT_CONTEXT)
                if _CAUSAL_RECEIPT_CONTEXT is not None
                else None
            )
            if receipt_context is not None and receipt_context["phase"] == "production_control":
                receipt_context["conversion_index"] = int(
                    _CAUSAL_RECEIPT_CONTEXT["conversion_index"]
                )
                _CAUSAL_RECEIPT_CONTEXT["conversion_index"] = int(
                    _CAUSAL_RECEIPT_CONTEXT["conversion_index"]
                ) + 1
        return convert_battle_with_causal_ledger(
            battle,
            original_battle_to_engine,
            engine_module,
            swap=bool(swap),
            receipt_context=receipt_context,
        )

    search_main.battle_to_poke_engine_state = battle_to_engine_with_causal_ledger
    search_main.get_result_from_mcts = _mcts_with_root_priors
    original_find_best_move = search_main.find_best_move

    def select_move_with_dump(mcts_results):
        global _HOLDOUT_DECISION_SEQUENCE
        alpha_sequence_index = _HOLDOUT_DECISION_SEQUENCE
        _HOLDOUT_DECISION_SEQUENCE += 1
        controller_mode = os.environ.get(
            "METAGROSS_CONTROLLER_MODE", DEFAULT_CONTROLLER_MODE
        )
        verifier_shadow_enabled = os.environ.get("METAGROSS_VERIFIER_SHADOW") == "1"
        verifier_required = controller_mode == "certified"
        _provisional, provisional = harness.controller.select(
            _PRIOR_STATE["battle"],
            mcts_results,
            _PRIOR_STATE["priors"],
            record_history=False,
        )
        baseline = provisional["baseline"]
        candidate_panel = freeze_holdout_candidate_panel(
            _PRIOR_STATE["battle"], provisional
        )
        holdout_by_action = {}
        holdout_panel = None
        if (
            candidate_panel
            and os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") == "1"
            and (verifier_required or verifier_shadow_enabled)
        ):
            try:
                fresh_battles, _count, _duration = harness.belief.expand(
                    _PRIOR_STATE["battle"], search_main, "certification-worlds"
                )
                fresh_states = [
                    search_main.battle_to_poke_engine_state(sampled).to_string()
                    for sampled, _chance in fresh_battles
                ]
                fresh_weights = [
                    float(chance) for _sampled, chance in fresh_battles
                ]
                state_hashes = [state_sha256(state) for state in fresh_states]
                cluster_hashes = list(state_hashes)
                holdout_seeds = [
                    _derived_seed("holdout-tape", index, required=True)
                    for index in range(len(fresh_states))
                ]
                if any(seed is None for seed in holdout_seeds):
                    raise RuntimeError("holdout tape derivation failed")
                for panel_row in candidate_panel:
                    candidate = str(panel_row["action"])
                    candidate_rank = int(panel_row["rank"])
                    certificates = {}
                    for horizon_index, continuation_steps in enumerate(
                        HOLDOUT_CONTINUATION_HORIZONS
                    ):
                        holdout_results = harness.search.holdout(
                            fresh_states,
                            baseline,
                            candidate,
                            continuation_steps,
                            holdout_seeds,
                            candidate_rank,
                        )
                        certificate = harness.verifier.certify(
                            holdout_results,
                            fresh_weights,
                            state_hashes,
                            cluster_hashes,
                            candidate,
                            baseline,
                            alpha_sequence_index,
                            candidate_rank,
                            horizon_index,
                        )
                        certificates[continuation_steps] = certificate
                        if certificate["qualified"] is not True:
                            break
                    holdout_by_action[candidate] = (
                        harness.verifier.combine(certificates)
                    )
                holdout_panel = {
                    "schema_version": 1,
                    "evidence_kind": "frozen_top_k_adaptive_holdout_panel_v5",
                    "baseline": baseline,
                    "alpha_sequence_index": alpha_sequence_index,
                    "candidate_panel": candidate_panel,
                    "selection_cohort": {
                        "sampling_seed": (_PRIOR_STATE["remote_search"] or {}).get(
                            "sampling_seed"
                        ),
                        "state_hashes": (_PRIOR_STATE["remote_search"] or {}).get(
                            "state_hashes"
                        ),
                        "request_ids": (_PRIOR_STATE["remote_search"] or {}).get(
                            "request_ids"
                        ),
                    },
                    "certification_cohort": {
                        "sampling_seed": _derived_seed(
                            "certification-worlds", 0, required=True
                        ),
                        "state_hashes": state_hashes,
                        "cluster_hashes": cluster_hashes,
                        "weights": fresh_weights,
                        "tape_seeds": holdout_seeds,
                    },
                    "opponent_priors": _PRIOR_STATE["opp_priors"],
                    "opponent_uniform_mix": HOLDOUT_OPPONENT_UNIFORM_MIX,
                    "certificates_by_action": holdout_by_action,
                    "complete": True,
                    "qualified_actions": [
                        row["action"]
                        for row in candidate_panel
                        if holdout_by_action[row["action"]]["qualified"] is True
                    ],
                }
            except Exception as exc:
                logger.error("required independent holdout failed: %s", type(exc).__name__)
                raise RuntimeError("required independent holdout failed") from exc
        choice, choice_override = harness.controller.select(
            _PRIOR_STATE["battle"],
            mcts_results,
            _PRIOR_STATE["priors"],
            independent_evidence=holdout_by_action or None,
        )
        terminal_teacher_enabled = _terminal_mcts_teacher_enabled()
        one_deviation_enabled = _terminal_mcts_one_deviation_enabled()
        if terminal_teacher_enabled and one_deviation_enabled:
            raise RuntimeError(
                "legacy terminal teacher and one-deviation experiment cannot both be enabled"
            )
        if terminal_teacher_enabled:
            try:
                terminal_teacher = _terminal_mcts_teacher_decision(
                    harness,
                    search_main,
                    _PRIOR_STATE["battle"],
                    choice,
                )
            except Exception as exc:
                logger.error(
                    "terminal-MCTS teacher failed closed: %s", type(exc).__name__
                )
                terminal_teacher = {
                    "schema": "metagross-terminal-mcts-live-decision/v1",
                    "decision": "abstain",
                    "selected_action": choice,
                    "reason": f"fail_closed:{type(exc).__name__}",
                }
            previous_choice = choice
            if terminal_teacher["decision"] == "override":
                choice = str(terminal_teacher["selected_action"])
                if os.environ.get("METAGROSS_TERMINAL_MCTS_MODE") in {
                    "cycle18_equal8192",
                    "cycle19_equal8192",
                }:
                    cycle = os.environ.get("METAGROSS_TERMINAL_MCTS_MODE").split("_")[0]
                    choice_override["reason"] = f"{cycle}_equal8192_override"
                    choice_override["selection_class"] = "prospective_fixed_search_candidate"
                else:
                    choice_override["reason"] = "certified_terminal_mcts_override"
                    choice_override["selection_class"] = "certified_terminal_teacher"
            choice_override["terminal_mcts_teacher"] = terminal_teacher
            choice_override["terminal_mcts_production_choice"] = previous_choice
            choice_override["final_choice"] = choice
            choice_override["overridden"] = choice != choice_override["raw_choice"]
        elif one_deviation_enabled:
            one_deviation = _terminal_mcts_one_deviation_controller()
            tag, username, decision_index = _terminal_mcts_one_deviation_identity()
            if one_deviation.should_query(tag, username):
                previous_choice = choice
                try:
                    terminal_teacher = _terminal_mcts_teacher_decision(
                        harness,
                        search_main,
                        _PRIOR_STATE["battle"],
                        choice,
                    )
                except Exception as exc:
                    logger.error(
                        "one-deviation terminal-MCTS teacher failed closed: %s",
                        type(exc).__name__,
                    )
                    terminal_teacher = {
                        "schema": "metagross-terminal-mcts-live-decision/v1",
                        "decision": "abstain",
                        "selected_action": choice,
                        "reason": f"fail_closed:{type(exc).__name__}",
                    }
                choice, one_deviation_row = one_deviation.observe(
                    battle_tag=tag,
                    username=username,
                    decision_index=decision_index,
                    production_choice=previous_choice,
                    teacher=terminal_teacher,
                )
                if one_deviation_row["eligible"] and one_deviation_row[
                    "intervention_applied"
                ]:
                    choice_override["reason"] = "randomized_one_deviation_teacher"
                    choice_override["selection_class"] = (
                        "randomized_terminal_teacher_intervention"
                    )
                choice_override["terminal_mcts_teacher"] = terminal_teacher
                choice_override["terminal_mcts_one_deviation"] = one_deviation_row
                choice_override["terminal_mcts_production_choice"] = previous_choice
                choice_override["final_choice"] = choice
                choice_override["overridden"] = choice != choice_override["raw_choice"]
        if (
            os.environ.get("METAGROSS_CYCLE32_BOUNDARY_RECEIPT")
            or os.environ.get("METAGROSS_CYCLE31_BOUNDARY_RECEIPT")
        ):
            if terminal_teacher_enabled:
                cycle32 = bool(os.environ.get("METAGROSS_CYCLE32_BOUNDARY_RECEIPT"))
                boundary_builder = (
                    cycle32_authenticated_candidate_boundary_receipt
                    if cycle32
                    else cycle31_candidate_boundary_receipt
                )
                cycle31_boundary = boundary_builder(
                    _PRIOR_STATE["battle"],
                    (_PRIOR_STATE.get("context") or {}).get("own_legality"),
                    (_PRIOR_STATE.get("context") or {}).get("decision_idx"),
                    choice_override.get("terminal_mcts_teacher"),
                    choice,
                    max_decision_index=int(
                        os.environ.get(
                            "METAGROSS_CYCLE32_MAX_DECISION_INDEX"
                            if cycle32
                            else "METAGROSS_CYCLE31_MAX_DECISION_INDEX",
                            "5",
                        )
                    ),
                    max_battle_turn=int(
                        os.environ.get(
                            "METAGROSS_CYCLE32_MAX_BATTLE_TURN"
                            if cycle32
                            else "METAGROSS_CYCLE31_MAX_BATTLE_TURN",
                            "6",
                        )
                    ),
                )
                if (
                    cycle31_boundary["cycle30_dynamic_boundary"]["within_bounds"]
                    is not True
                ):
                    raise RuntimeError(
                        "candidate smoke reached its decision/turn bound without an eligible root"
                    )
                if (
                    cycle31_boundary["cycle30_dynamic_boundary"]["eligible"]
                    is True
                ):
                    _CYCLE25_EXECUTION_BOUNDARIES[
                        str(cycle31_boundary["battle_tag"])
                    ] = cycle31_boundary
        selected_holdout = holdout_by_action.get(choice)
        if selected_holdout is None and candidate_panel:
            selected_holdout = holdout_by_action.get(candidate_panel[0]["action"])
        choice_override["holdout"] = selected_holdout
        choice_override["holdout_panel"] = holdout_panel
        choice_override["candidate_panel"] = candidate_panel
        choice_override["verifier_execution"] = {
            "mode": (
                "admission"
                if verifier_required
                else "shadow"
                if verifier_shadow_enabled
                else "disabled"
            ),
            "executed": holdout_panel is not None,
            "selection_eligible": verifier_required,
        }
        if choice_override["overridden"]:
            logger.warning(
                "final-choice override (%s): %s -> %s",
                choice_override["reason"],
                choice_override["raw_choice"],
                choice,
            )
        ledger_path = os.environ.get("METAGROSS_HOLDOUT_LEDGER")
        if ledger_path:
            append_ledger_row(
                Path(ledger_path),
                {
                    "schema_version": 1,
                    "sequence_index": alpha_sequence_index,
                    "context": _PRIOR_STATE["context"],
                    "baseline": baseline,
                    "candidate_panel": candidate_panel,
                    "selection_cohort": {
                        "sampling_seed": (_PRIOR_STATE["remote_search"] or {}).get(
                            "sampling_seed"
                        ),
                        "state_hashes": (_PRIOR_STATE["remote_search"] or {}).get(
                            "state_hashes"
                        ),
                        "request_ids": (_PRIOR_STATE["remote_search"] or {}).get(
                            "request_ids"
                        ),
                        "weights": [
                            float(sample_chance)
                            for _result, sample_chance, _index in mcts_results
                        ],
                    },
                    "certification": holdout_panel,
                    "final_choice": choice,
                    "final_reason": choice_override["reason"],
                },
            )
        _append_jsonl(
            "METAGROSS_SEARCH_DUMP",
            {
                "schema": 2,
                "time_ns": time.time_ns(),
                "context": _PRIOR_STATE["context"],
                "choice": choice,
                "original_choice": choice_override["raw_choice"],
                "choice_override": choice_override,
                "player_priors": _PRIOR_STATE["priors"],
                "opponent_priors": _PRIOR_STATE["opp_priors"],
                "remote_search": _PRIOR_STATE["remote_search"],
                "samples": [
                    {
                        "sample_chance": float(sample_chance),
                        "index": int(index),
                        "result": _mcts_result_payload(result),
                    }
                    for result, sample_chance, index in mcts_results
                ],
            },
        )
        return choice

    search_main.select_move_from_mcts_results = select_move_with_dump

    def capture_dual_r1_root(battle, snapshot: PolicySnapshot) -> None:
        if os.environ.get("METAGROSS_DUAL_R1_CAPTURE") != "1":
            return
        # Recharge/Struggle requests are automatic Showdown commands outside
        # R1's learned 13-action policy. The prior server intentionally does
        # not advance or snapshot the policy trajectory for them.
        if automatic_request_action(snapshot.context.get("own_legality")) is not None:
            return
        decision_log = os.environ.get("METAGROSS_DECISION_LOG")
        policy_snapshot = snapshot.context.get("r1_policy_snapshot")
        if not decision_log or not isinstance(policy_snapshot, dict):
            raise RuntimeError("dual-R1 capture requires a decision log and schema-6 snapshot")
        if policy_snapshot.get("schema") != 6:
            raise RuntimeError("dual-R1 capture requires a schema-6 policy snapshot")
        state = search_main.battle_to_poke_engine_state(battle).to_string()
        row = {
            "schema": "metagross-causal-dual-r1-root/v1",
            "identity": {
                "namespace": os.environ.get("METAGROSS_PRIOR_NAMESPACE", ""),
                "battle_tag": snapshot.context.get("tag"),
                "username": str(policy_snapshot.get("username") or ""),
                "decision_idx": snapshot.context.get("decision_idx"),
                "battle_turn": snapshot.context.get("battle_turn"),
            },
            "state": state,
            "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
            "r1_policy_snapshot": policy_snapshot,
        }
        row["capture_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        path = Path(f"{decision_log}.dual-r1-roots.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            )
            while payload:
                written = os.write(descriptor, payload)
                if written <= 0:
                    raise OSError("dual-R1 root capture made no progress")
                payload = payload[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def find_best_move_with_priors(battle):
        global _CAUSAL_RECEIPT_CONTEXT
        request_key, request_fingerprint = battle_request_identity(battle)
        cached = _REQUEST_CHOICE_CACHE.get(request_key)
        if cached is not None:
            cached_fingerprint, cached_choice = cached
            if cached_fingerprint != request_fingerprint:
                raise RuntimeError("Showdown replay changed an existing rqid payload")
            logger.warning("replaying cached choice for duplicate rqid %s", request_key)
            return cached_choice
        freeze_and_attach_battle_ledger(battle)
        _PRIOR_STATE["priors"] = None
        _PRIOR_STATE["opp_priors"] = None
        _PRIOR_STATE["context"] = None
        _PRIOR_STATE["remote_search"] = None
        _PRIOR_STATE["battle"] = battle
        snapshot = None
        try:
            snapshot = harness.policy.propose(battle)
            from srcs.metagross.prior_temperature import flatten_priors
            _PRIOR_STATE["priors"] = flatten_priors(
                list(snapshot.priors), getattr(battle, "turn", None))
            _PRIOR_STATE["opp_priors"] = (
                list(snapshot.opponent_priors)
                if snapshot.opponent_priors is not None
                else None
            )
            _PRIOR_STATE["context"] = dict(snapshot.context)
            capture_dual_r1_root(battle, snapshot)
            logger.info(
                f"loaded {len(snapshot.priors)} player and "
                f"{len(_PRIOR_STATE['opp_priors'] or ())} effective opponent priors"
            )
        except Exception as exc:
            if os.environ.get("METAGROSS_REQUIRE_PRIORS", "1") == "1":
                raise RuntimeError(f"required prior fetch failed: {exc!r}") from exc
            logger.warning("prior fetch failed; using unguided search: %r", exc)
        if (
            os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") == "1"
            or root_search_mode() == "shared_rm_plus"
        ):
            choice = _remote_find_best_move(battle, search_main, harness)
        else:
            context = _PRIOR_STATE.get("context") or {}
            tag = str(context.get("tag") or request_key[0])
            decision_index = int(context.get("decision_idx") or 0)
            root_id = hashlib.sha256(
                f"terminal-mcts-live\0{tag}\0{decision_index}".encode("utf-8")
            ).hexdigest()
            cycle30_boundary = None
            if os.environ.get("METAGROSS_CYCLE30_BOUNDARY_RECEIPT"):
                cycle30_boundary = cycle30_dynamic_boundary_evidence(
                    battle,
                    context.get("own_legality"),
                    decision_index,
                    max_decision_index=int(
                        os.environ.get("METAGROSS_CYCLE30_MAX_DECISION_INDEX", "5")
                    ),
                    max_battle_turn=int(
                        os.environ.get("METAGROSS_CYCLE30_MAX_BATTLE_TURN", "6")
                    ),
                )
                if cycle30_boundary["within_bounds"] is not True:
                    raise RuntimeError(
                        "Cycle30 reached its decision/turn bound without an eligible root"
                    )
            declared_world_count, _search_time = (
                search_main.search_time_num_battles_randombattles(battle)
            )
            _CAUSAL_RECEIPT_CONTEXT = {
                "phase": "production_control",
                "cohort": "adaptive_root_search",
                "battle_tag": tag,
                "rqid": request_key[1],
                "decision_index": decision_index,
                "root_id": root_id,
                "declared_world_count": declared_world_count,
                "conversion_index": 0,
                "schedule_index": None,
                "world_index": None,
            }
            try:
                choice = original_find_best_move(battle)
            finally:
                _CAUSAL_RECEIPT_CONTEXT = None
            boundary_target = int(
                os.environ.get("METAGROSS_SMOKE_BOUNDARY_DECISION_INDEX", "0")
            )
            legacy_boundary = (
                os.environ.get("METAGROSS_CYCLE27_BOUNDARY_RECEIPT")
                or os.environ.get("METAGROSS_CYCLE25_BOUNDARY_RECEIPT")
            ) and decision_index >= boundary_target
            if (
                cycle30_boundary is not None
                and cycle30_boundary["eligible"] is True
            ) or legacy_boundary:
                _CYCLE25_EXECUTION_BOUNDARIES[tag] = {
                    "battle_tag": tag,
                    "rqid": request_key[1],
                    "decision_index": decision_index,
                    "root_id": root_id,
                    "selected_action": choice,
                    "active_name": getattr(
                        getattr(getattr(battle, "user", None), "active", None),
                        "name",
                        "",
                    ),
                    "decision_complete_time_ns": time.time_ns(),
                    "cycle30_dynamic_boundary": cycle30_boundary,
                }
        if snapshot is not None:
            # This is the causal action boundary: persist the exact final
            # choice before the caller is allowed to send it to Showdown.
            harness.policy.acknowledge(snapshot, choice)
        _REQUEST_CHOICE_CACHE[request_key] = (request_fingerprint, choice)
        while len(_REQUEST_CHOICE_CACHE) > MAX_TRACKED_BATTLES:
            _REQUEST_CHOICE_CACHE.pop(next(iter(_REQUEST_CHOICE_CACHE)))
        return choice

    search_main.find_best_move = find_best_move_with_priors
    run_battle.find_best_move = find_best_move_with_priors
    logger.info(
        f"root-prior patch active (server={server_url}, c_puct={_PRIOR_STATE['cpuct']})"
    )


def _install_websocket_socks_egress() -> None:
    """Route the Showdown websocket through a SOCKS5 proxy when
    METAGROSS_WEBSOCKET_SOCKS is set (e.g. a Tailscale userspace proxy whose
    exit node is a residential connection). Fail-closed: if the env is set but
    the proxy machinery is unusable, raise rather than fall back to a direct
    (datacenter) connection that Showdown would proxy-lock. No-op when unset,
    so local/loopback runs are byte-identical."""
    proxy_url = os.environ.get("METAGROSS_WEBSOCKET_SOCKS")
    if not proxy_url:
        return
    import ssl as _ssl
    import websockets
    from python_socks.async_.asyncio import Proxy

    original_connect = websockets.connect

    async def _connect_via_socks(uri, *args, **kwargs):
        parsed = urlparse(uri)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        proxy = Proxy.from_url(proxy_url)
        sock = await proxy.connect(dest_host=host, dest_port=port)
        if parsed.scheme == "wss":
            kwargs.setdefault("ssl", _ssl.create_default_context())
            kwargs.setdefault("server_hostname", host)
        kwargs["sock"] = sock
        return await original_connect(uri, *args, **kwargs)

    websockets.connect = _connect_via_socks
    print(
        f"WEBSOCKET_SOCKS_EGRESS active via {urlparse(proxy_url).scheme}://"
        f"{urlparse(proxy_url).hostname}:{urlparse(proxy_url).port}",
        flush=True,
    )


def main() -> None:
    global _POKE_ENGINE_PROVENANCE
    _install_websocket_socks_egress()
    root = Path(__file__).resolve().parents[2]
    provenance = inspect_poke_engine()
    _POKE_ENGINE_PROVENANCE = {
        **provenance,
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
    }
    print(
        f"POKE_ENGINE_PROVENANCE {json.dumps(provenance, sort_keys=True)}", flush=True
    )
    receipt_dir = os.environ.get("METAGROSS_PINNED_ENGINE_RECEIPT_DIR")
    if receipt_dir:
        namespace = os.environ.get("METAGROSS_PRIOR_NAMESPACE")
        if namespace not in {"agent_a", "agent_b"}:
            raise RuntimeError("pinned engine receipt requires an isolated agent namespace")
        receipt_path = Path(receipt_dir).expanduser().resolve() / (
            f"{namespace}-engine-provenance.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": "metagross-spawned-engine-provenance/v1",
                    "namespace": namespace,
                    "python_executable": str(Path(sys.executable).resolve()),
                    "provenance": provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    foul_play_dir = (
        Path(os.environ.get("FOUL_PLAY_DIR", root / "srcs" / "vendor" / "foul-play"))
        .expanduser()
        .resolve()
    )
    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    os.chdir(foul_play_dir)
    sys.path.insert(0, str(foul_play_dir))
    patch_foul_play_protocol()
    patch_root_priors(build_decision_harness())

    from run import run_foul_play
    from config import FoulPlayConfig

    original_configure = FoulPlayConfig.configure

    def configure_with_environment_password():
        original_configure()
        require_local_ensemble_websocket(FoulPlayConfig.websocket_uri)
        password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
        local_no_security = (
            os.environ.get("METAGROSS_ALLOW_INSECURE_LOOPBACK") == "1"
            and is_loopback_websocket_uri(FoulPlayConfig.websocket_uri)
        )
        if not password and not local_no_security:
            raise RuntimeError("METAGROSS_SHOWDOWN_PASSWORD is required")
        FoulPlayConfig.password = password

    # Keep credentials out of argv and process listings. Foul Play receives the
    # password only after parsing its non-secret command line.
    FoulPlayConfig.configure = configure_with_environment_password

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
