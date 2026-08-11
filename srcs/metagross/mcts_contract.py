"""Shared wire contract for production remote MCTS providers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import ipaddress
import json
import math
from pathlib import Path
import struct
from urllib.parse import urlparse


ENGINE_CONTRACT = "poke-engine-0.0.47-shared-root-v7"
ENGINE_SOURCE_SHA256 = (
    "639982daced7abb3ebad4fed8bc6b5408dc82c7386241b3732a7481a7aacae73"
)
REQUEST_SCHEMA = 4
MAX_WIRE_BATCH_SIZE = 64
MAX_SHARED_ROOT_REPLAY_BYTES = 16_000_000
MODAL_CONTAINER_BATCH_SIZE = 16
MODAL_MAX_CONTAINERS = 4
MODAL_MAX_WORLD_CONCURRENCY = MODAL_CONTAINER_BATCH_SIZE * MODAL_MAX_CONTAINERS
HOLDOUT_RESULT_FIELDS = {
    "pairs",
    "baseline_sum",
    "candidate_sum",
    "delta_sum",
    "delta_squared_sum",
    "catastrophic_count",
    "candidate_catastrophic_count",
    "baseline_catastrophic_count",
    "candidate_catastrophic_severity_sum",
    "baseline_catastrophic_severity_sum",
    "candidate_better_count",
    "baseline_better_count",
    "equal_count",
    "baseline_terminal_count",
    "candidate_terminal_count",
    "baseline_nonterminal_evaluation_delta_sum",
    "candidate_nonterminal_evaluation_delta_sum",
    "baseline_nonterminal_count",
    "candidate_nonterminal_count",
    "continuation_iterations_executed",
}
SHARED_ROOT_DIAGNOSTIC_FIELDS = {
    "solver_contract",
    "iterations",
    "continuation_iterations",
    "seed",
    "prior_strength",
    "expected_value",
    "player_best_response_value",
    "opponent_best_response_value",
    "player_best_response_gain",
    "opponent_best_response_gain",
    "nash_conv",
    "exploitability",
    "player_regret_bound",
    "opponent_regret_bound",
    "total_regret_bound",
    "payoff_cells",
    "total_forced_continuation_iterations",
    "input_particle_count",
    "positive_particle_count",
    "canonical_particle_count",
    "normalized_weight_sum",
    "action_support_digest",
    "particle_digest",
    "payoff_digest",
    "player_prior_digest",
    "opponent_prior_digest",
}
SHARED_ROOT_REPLAY_FIELDS = {
    "schema_version",
    "solver_contract",
    "configuration",
    "own_action_support",
    "normalized_player_prior",
    "canonical_particles",
}
SHARED_ROOT_REPLAY_PARTICLE_FIELDS = {
    "canonical_index",
    "state",
    "normalized_weight",
    "source_particles",
    "opponent_action_support",
    "normalized_opponent_prior",
    "payoff_matrix",
    "continuations",
    "opponent_policy",
}
SHARED_ROOT_CONTINUATION_FIELDS = {
    "seed",
    "requested_iterations",
    "executed_iterations",
    "visits",
    "total_score",
    "total_score_bits",
    "payoff",
    "payoff_bits",
}


def compute_engine_source_sha256(root: Path) -> str:
    """Hash the production Rust/Python source contract deterministically."""
    files = {
        root / "Cargo.lock",
        root / "Cargo.toml",
        root / "pyproject.toml",
        root / "poke-engine-py" / "Cargo.toml",
    }
    files.update((root / "src").rglob("*.rs"))
    files.update((root / "poke-engine-py" / "src").rglob("*.rs"))
    for package in (
        root / "python" / "poke_engine",
        root / "poke-engine-py" / "python" / "poke_engine",
    ):
        files.update(
            path for path in package.iterdir() if path.suffix in {".py", ".pyi"}
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def engine_identity(resources: dict[str, object]) -> dict[str, object]:
    import poke_engine

    native = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    parameters = list(inspect.signature(poke_engine.monte_carlo_tree_search).parameters)
    if parameters != [
        "state",
        "duration_ms",
        "iterations",
        "threads",
        "s1_priors",
        "s2_priors",
        "c_puct",
    ]:
        raise RuntimeError("remote poke-engine has an invalid MCTS contract")
    holdout_parameters = list(
        inspect.signature(poke_engine.paired_root_policy_evaluation).parameters
    )
    if holdout_parameters != [
        "state",
        "baseline_action",
        "candidate_action",
        "rollouts",
        "continuation_iterations",
        "continuation_steps",
        "seed",
        "opponent_priors",
    ]:
        raise RuntimeError("remote poke-engine has an invalid holdout contract")
    shared_root_parameters = list(
        inspect.signature(poke_engine.shared_information_set_root_search).parameters
    )
    if shared_root_parameters != [
        "states",
        "particle_weights",
        "iterations",
        "continuation_iterations",
        "seed",
        "prior_strength",
        "s1_prior",
        "s2_priors",
    ]:
        raise RuntimeError("remote poke-engine has an invalid shared-root contract")
    return {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "distribution_version": importlib.metadata.version("poke_engine"),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "mcts_parameters": parameters,
        "holdout_parameters": holdout_parameters,
        "shared_root_parameters": shared_root_parameters,
        "resources": resources,
    }


def validate_priors(value: object, label: str) -> list[tuple[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"{label} must be a bounded list")
    priors: list[tuple[str, float]] = []
    moves: set[str] = set()
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{label} entries must contain a move and probability")
        move, probability = row
        if (
            not isinstance(move, str)
            or not move
            or move in moves
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"{label} contains an invalid entry")
        moves.add(move)
        priors.append((move, float(probability)))
    total = math.fsum(probability for _move, probability in priors)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label} must contain positive finite probability mass")
    return priors


def validate_request(request: dict[str, object]) -> dict[str, object]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported request schema")
    request_id = request.get("request_id")
    index = request.get("index")
    operation = request.get("operation")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("invalid request ID")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("invalid world index")
    common = {"operation": operation}
    if operation == "shared_root":
        allowed_fields = {
            "schema",
            "request_id",
            "index",
            "operation",
            "states",
            "particle_weights",
            "iterations",
            "continuation_iterations",
            "seed",
            "prior_strength",
            "s1_prior",
            "s2_priors",
        }
        if set(request) - allowed_fields or set(request) != allowed_fields:
            raise ValueError("shared-root request has missing or unknown fields")
        states = request.get("states")
        weights = request.get("particle_weights")
        if not isinstance(states, list) or not 1 <= len(states) <= 64:
            raise ValueError("shared-root states must contain between 1 and 64 particles")
        if any(not isinstance(state, str) or not 0 < len(state) <= 1_000_000 for state in states):
            raise ValueError("shared-root request contains an invalid state")
        if sum(len(state) for state in states) > 64_000_000:
            raise ValueError("shared-root states exceed the serialized bound")
        if not isinstance(weights, list) or len(weights) != len(states):
            raise ValueError("shared-root particle weights must match states")
        normalized_weights: list[float] = []
        for weight in weights:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("shared-root particle weights are invalid")
            normalized_weights.append(float(weight))
        if abs(math.fsum(normalized_weights) - 1.0) > 1e-6:
            raise ValueError("shared-root particle weights must sum to one")
        iterations = request.get("iterations")
        continuation_iterations = request.get("continuation_iterations")
        seed = request.get("seed")
        prior_strength = request.get("prior_strength")
        for name, value in (
            ("iterations", iterations),
            ("continuation_iterations", continuation_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000:
                raise ValueError(f"invalid shared-root {name}")
        if len(states) * 16 * 16 * continuation_iterations > 100_000_000:
            raise ValueError("shared-root continuation work exceeds the resource bound")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 64) - 1:
            raise ValueError("invalid shared-root seed")
        if (
            isinstance(prior_strength, bool)
            or not isinstance(prior_strength, (int, float))
            or not math.isfinite(prior_strength)
            or not 0 <= prior_strength <= 1_000
        ):
            raise ValueError("invalid shared-root prior strength")
        s2_priors = request.get("s2_priors")
        if s2_priors is None:
            validated_s2_priors = None
        else:
            if not isinstance(s2_priors, list) or len(s2_priors) != len(states):
                raise ValueError("shared-root opponent priors must match states")
            validated_s2_priors = [
                validate_priors(prior, f"s2_priors[{world}]")
                for world, prior in enumerate(s2_priors)
            ]
        return {
            **common,
            "states": states,
            "particle_weights": normalized_weights,
            "iterations": iterations,
            "continuation_iterations": continuation_iterations,
            "seed": seed,
            "prior_strength": float(prior_strength),
            "s1_prior": validate_priors(request.get("s1_prior"), "s1_prior"),
            "s2_priors": validated_s2_priors,
        }
    state_string = request.get("state")
    if not isinstance(state_string, str) or not 0 < len(state_string) <= 1_000_000:
        raise ValueError("invalid state string")
    common["state"] = state_string
    if operation == "paired_holdout":
        allowed_fields = {
            "schema",
            "request_id",
            "index",
            "operation",
            "state",
            "baseline_action",
            "candidate_action",
            "rollouts",
            "continuation_iterations",
            "continuation_steps",
            "seed",
            "opponent_priors",
        }
        if set(request) - allowed_fields:
            raise ValueError("paired holdout request has unknown fields")
        if "opponent_priors" not in request:
            raise ValueError("paired holdout request requires opponent_priors")
        baseline_action = request.get("baseline_action")
        candidate_action = request.get("candidate_action")
        rollouts = request.get("rollouts")
        continuation_iterations = request.get("continuation_iterations")
        continuation_steps = request.get("continuation_steps")
        seed = request.get("seed")
        if not isinstance(baseline_action, str) or not 0 < len(baseline_action) <= 128:
            raise ValueError("invalid baseline action")
        if (
            not isinstance(candidate_action, str)
            or not 0 < len(candidate_action) <= 128
        ):
            raise ValueError("invalid candidate action")
        for name, value, upper in (
            ("rollouts", rollouts, 100_000),
            ("continuation_iterations", continuation_iterations, 10_000_000),
            ("continuation_steps", continuation_steps, 100),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= upper
            ):
                raise ValueError(f"invalid {name}")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= (1 << 64) - 1
        ):
            raise ValueError("invalid holdout seed")
        if 2 * rollouts * continuation_iterations * continuation_steps > 100_000_000:
            raise ValueError("holdout work exceeds the resource bound")
        return {
            **common,
            "baseline_action": baseline_action,
            "candidate_action": candidate_action,
            "rollouts": rollouts,
            "continuation_iterations": continuation_iterations,
            "continuation_steps": continuation_steps,
            "seed": seed,
            "opponent_priors": validate_priors(
                request["opponent_priors"], "opponent_priors"
            ),
        }
    if operation != "search":
        raise ValueError("unsupported operation")
    allowed_fields = {
        "schema",
        "request_id",
        "index",
        "operation",
        "state",
        "duration_ms",
        "threads",
        "s1_priors",
        "s2_priors",
        "c_puct",
    }
    if set(request) - allowed_fields:
        raise ValueError("search request has unknown fields")
    duration_ms = request.get("duration_ms")
    threads = request.get("threads")
    c_puct = request.get("c_puct")
    if duration_ms not in {250, 500}:
        raise ValueError("duration must be 250 or 500 ms")
    if threads != 1:
        raise ValueError("remote search requires one thread")
    if float(c_puct) != 2.0:
        raise ValueError("remote search requires c_puct=2.0")
    return {
        **common,
        "duration_ms": duration_ms,
        "threads": threads,
        "s1_priors": validate_priors(request.get("s1_priors"), "s1_priors"),
        "s2_priors": validate_priors(request.get("s2_priors"), "s2_priors"),
        "c_puct": float(c_puct),
    }


def holdout_result_payload(
    result,
    *,
    expected_pairs: int | None = None,
    maximum_executed: int | None = None,
) -> dict[str, object]:
    payload = {name: getattr(result, name) for name in HOLDOUT_RESULT_FIELDS}
    return validate_holdout_result_payload(
        payload,
        expected_pairs=expected_pairs,
        maximum_executed=maximum_executed,
    )


def validate_holdout_result_payload(
    value: object,
    *,
    expected_pairs: int | None = None,
    maximum_executed: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != HOLDOUT_RESULT_FIELDS:
        raise ValueError("holdout result has invalid fields")
    pairs = value["pairs"]
    if (
        isinstance(pairs, bool)
        or not isinstance(pairs, int)
        or not 1 <= pairs <= 100_000
    ):
        raise ValueError("holdout result has invalid pair count")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError("holdout result pair count differs from the request")
    count_fields = HOLDOUT_RESULT_FIELDS - {
        "baseline_sum",
        "candidate_sum",
        "delta_sum",
        "delta_squared_sum",
        "candidate_catastrophic_severity_sum",
        "baseline_catastrophic_severity_sum",
        "baseline_nonterminal_evaluation_delta_sum",
        "candidate_nonterminal_evaluation_delta_sum",
        "continuation_iterations_executed",
    }
    for name in count_fields:
        field = value[name]
        if (
            isinstance(field, bool)
            or not isinstance(field, int)
            or not 0 <= field <= pairs
        ):
            raise ValueError(f"holdout result has invalid {name}")
    executed = value["continuation_iterations_executed"]
    if isinstance(executed, bool) or not isinstance(executed, int) or executed < 0:
        raise ValueError("holdout result has invalid executed iterations")
    execution_bound = 100_000_000 if maximum_executed is None else maximum_executed
    if executed > execution_bound:
        raise ValueError("holdout result exceeds its iteration bound")
    numeric = {}
    for name in (
        "baseline_sum",
        "candidate_sum",
        "delta_sum",
        "delta_squared_sum",
        "candidate_catastrophic_severity_sum",
        "baseline_catastrophic_severity_sum",
        "baseline_nonterminal_evaluation_delta_sum",
        "candidate_nonterminal_evaluation_delta_sum",
    ):
        field = value[name]
        if (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or not math.isfinite(field)
        ):
            raise ValueError(f"holdout result has invalid {name}")
        numeric[name] = float(field)
    tolerance = 1e-8 * max(1, pairs)
    if not -tolerance <= numeric["baseline_sum"] <= pairs + tolerance:
        raise ValueError("holdout baseline sum is out of range")
    if not -tolerance <= numeric["candidate_sum"] <= pairs + tolerance:
        raise ValueError("holdout candidate sum is out of range")
    if not -pairs - tolerance <= numeric["delta_sum"] <= pairs + tolerance:
        raise ValueError("holdout delta sum is out of range")
    if not -tolerance <= numeric["delta_squared_sum"] <= pairs + tolerance:
        raise ValueError("holdout squared delta sum is out of range")
    if (
        numeric["delta_squared_sum"] + tolerance
        < numeric["delta_sum"] * numeric["delta_sum"] / pairs
    ):
        raise ValueError("holdout delta moments are inconsistent")
    if (
        abs(numeric["candidate_sum"] - numeric["baseline_sum"] - numeric["delta_sum"])
        > tolerance
    ):
        raise ValueError("holdout sums are inconsistent")
    if (
        value["candidate_better_count"]
        + value["baseline_better_count"]
        + value["equal_count"]
        != pairs
    ):
        raise ValueError("holdout comparison counts are inconsistent")
    if value["catastrophic_count"] != value["candidate_catastrophic_count"]:
        raise ValueError("holdout catastrophic alias is inconsistent")
    if value["candidate_catastrophic_count"] > value["baseline_better_count"]:
        raise ValueError("holdout candidate catastrophe count is inconsistent")
    if value["baseline_catastrophic_count"] > value["candidate_better_count"]:
        raise ValueError("holdout baseline catastrophe count is inconsistent")
    for arm in ("baseline", "candidate"):
        terminal_count = value[f"{arm}_terminal_count"]
        nonterminal_count = value[f"{arm}_nonterminal_count"]
        if terminal_count + nonterminal_count != pairs:
            raise ValueError(f"holdout {arm} terminal partition is inconsistent")
        count = value[f"{arm}_catastrophic_count"]
        severity = numeric[f"{arm}_catastrophic_severity_sum"]
        if not 0.5 * count - tolerance <= severity <= count + tolerance:
            raise ValueError(f"holdout {arm} catastrophe severity is out of range")
    return {**value, **numeric}


def _shared_root_replay_capture_payload(capture) -> dict[str, object]:
    configuration = capture.configuration
    return {
        "schema_version": capture.schema_version,
        "solver_contract": capture.solver_contract,
        "configuration": {
            "iterations": configuration.iterations,
            "continuation_iterations": configuration.continuation_iterations,
            "seed": configuration.seed,
            "prior_strength": configuration.prior_strength,
        },
        "own_action_support": list(capture.own_action_support),
        "normalized_player_prior": (
            list(capture.normalized_player_prior)
            if capture.normalized_player_prior is not None
            else None
        ),
        "canonical_particles": [
            {
                "canonical_index": particle.canonical_index,
                "state": particle.state,
                "normalized_weight": particle.normalized_weight,
                "source_particles": [
                    {
                        "input_index": source.input_index,
                        "input_weight": source.input_weight,
                    }
                    for source in particle.source_particles
                ],
                "opponent_action_support": list(particle.opponent_action_support),
                "normalized_opponent_prior": (
                    list(particle.normalized_opponent_prior)
                    if particle.normalized_opponent_prior is not None
                    else None
                ),
                "payoff_matrix": [list(row) for row in particle.payoff_matrix],
                "continuations": [
                    [
                        {
                            name: getattr(cell, name)
                            for name in SHARED_ROOT_CONTINUATION_FIELDS
                        }
                        for cell in row
                    ]
                    for row in particle.continuations
                ],
                "opponent_policy": list(particle.opponent_policy),
            }
            for particle in capture.canonical_particles
        ],
    }


def _fnv1a64_digest(parts) -> str:
    value = 0xCBF29CE484222325
    for part in parts:
        for byte in part:
            value ^= byte
            value = (value * 0x00000100000001B3) & ((1 << 64) - 1)
        value ^= 0xFF
        value = (value * 0x00000100000001B3) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}"


def _normalized_replay_vector(value: object, length: int, label: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} has invalid dimensions")
    normalized = []
    for entry in value:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(entry)
            or entry < 0
        ):
            raise ValueError(f"{label} contains an invalid probability")
        normalized.append(float(entry))
    if abs(math.fsum(normalized) - 1.0) > 1e-8:
        raise ValueError(f"{label} is not normalized")
    return normalized


def _validate_shared_root_replay_capture(
    value: object,
    *,
    policy: list[dict[str, object]],
    opponent_policies: list[list[dict[str, object]]],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != SHARED_ROOT_REPLAY_FIELDS:
        raise ValueError("shared-root replay capture has invalid fields")
    try:
        encoded_size = len(
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("shared-root replay capture is not JSON serializable") from exc
    if encoded_size > MAX_SHARED_ROOT_REPLAY_BYTES:
        raise ValueError("shared-root replay capture exceeds the encoded size bound")
    if value["schema_version"] != 1 or value["solver_contract"] != diagnostics["solver_contract"]:
        raise ValueError("shared-root replay capture has an invalid contract")
    configuration = value["configuration"]
    if not isinstance(configuration, dict) or set(configuration) != {
        "iterations",
        "continuation_iterations",
        "seed",
        "prior_strength",
    }:
        raise ValueError("shared-root replay configuration is invalid")
    for name in ("iterations", "continuation_iterations", "seed"):
        if configuration[name] != diagnostics[name]:
            raise ValueError("shared-root replay configuration differs from diagnostics")
    prior_strength = configuration["prior_strength"]
    if (
        isinstance(prior_strength, bool)
        or not isinstance(prior_strength, (int, float))
        or not math.isfinite(prior_strength)
        or abs(float(prior_strength) - diagnostics["prior_strength"]) > 1e-8
    ):
        raise ValueError("shared-root replay prior strength differs from diagnostics")
    normalized_configuration = {**configuration, "prior_strength": float(prior_strength)}

    own_support = value["own_action_support"]
    if (
        not isinstance(own_support, list)
        or own_support != sorted(own_support)
        or own_support != [row["action"] for row in sorted(policy, key=lambda row: row["action"])]
        or len(set(own_support)) != len(own_support)
    ):
        raise ValueError("shared-root replay own action support is invalid")
    player_prior = _normalized_replay_vector(
        value["normalized_player_prior"], len(own_support), "shared-root player prior"
    )
    particles = value["canonical_particles"]
    if (
        not isinstance(particles, list)
        or len(particles) != diagnostics["canonical_particle_count"]
        or len(particles) != len(opponent_policies)
    ):
        raise ValueError("shared-root replay particles are invalid")

    normalized_particles = []
    all_source_indices = set()
    total_state_bytes = 0
    payoff_parts = []
    opponent_prior_parts = []
    particle_digest_parts = []
    payoff_cells = 0
    counterfactual_values = [0.0] * len(own_support)
    for canonical_index, (particle, top_level_opponent) in enumerate(
        zip(particles, opponent_policies, strict=True)
    ):
        if not isinstance(particle, dict) or set(particle) != SHARED_ROOT_REPLAY_PARTICLE_FIELDS:
            raise ValueError("shared-root replay particle has invalid fields")
        if particle["canonical_index"] != canonical_index:
            raise ValueError("shared-root replay canonical indices are invalid")
        state = particle["state"]
        if not isinstance(state, str) or not state or len(state) > 1_000_000:
            raise ValueError("shared-root replay state is invalid")
        total_state_bytes += len(state.encode("utf-8"))
        if total_state_bytes > 64_000_000:
            raise ValueError("shared-root replay states exceed the size bound")
        weight = particle["normalized_weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("shared-root replay weight is invalid")
        weight = float(weight)

        source_particles = particle["source_particles"]
        if not isinstance(source_particles, list) or not source_particles:
            raise ValueError("shared-root replay source particles are invalid")
        normalized_sources = []
        previous_source_index = -1
        for source in source_particles:
            if not isinstance(source, dict) or set(source) != {"input_index", "input_weight"}:
                raise ValueError("shared-root replay source particle has invalid fields")
            input_index = source["input_index"]
            input_weight = source["input_weight"]
            if (
                isinstance(input_index, bool)
                or not isinstance(input_index, int)
                or not previous_source_index < input_index < diagnostics["input_particle_count"]
                or input_index in all_source_indices
                or isinstance(input_weight, bool)
                or not isinstance(input_weight, (int, float))
                or not math.isfinite(input_weight)
                or input_weight <= 0
            ):
                raise ValueError("shared-root replay source particle is invalid")
            previous_source_index = input_index
            all_source_indices.add(input_index)
            normalized_sources.append(
                {"input_index": input_index, "input_weight": float(input_weight)}
            )

        opponent_support = particle["opponent_action_support"]
        if (
            not isinstance(opponent_support, list)
            or not opponent_support
            or len(opponent_support) > 16
            or opponent_support != sorted(opponent_support)
            or len(set(opponent_support)) != len(opponent_support)
            or opponent_support != [row["action"] for row in top_level_opponent]
        ):
            raise ValueError("shared-root replay opponent support is invalid")
        opponent_prior = _normalized_replay_vector(
            particle["normalized_opponent_prior"],
            len(opponent_support),
            "shared-root opponent prior",
        )
        opponent_policy = _normalized_replay_vector(
            particle["opponent_policy"],
            len(opponent_support),
            "shared-root captured opponent policy",
        )
        if any(
            abs(probability - top_level["probability"]) > 1e-8
            for probability, top_level in zip(opponent_policy, top_level_opponent, strict=True)
        ):
            raise ValueError("shared-root captured opponent policy differs from the result")

        matrix = particle["payoff_matrix"]
        continuations = particle["continuations"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != len(own_support)
            or not isinstance(continuations, list)
            or len(continuations) != len(own_support)
        ):
            raise ValueError("shared-root replay payoff matrix has invalid dimensions")
        normalized_matrix = []
        normalized_continuations = []
        for own_index, (matrix_row, continuation_row) in enumerate(
            zip(matrix, continuations, strict=True)
        ):
            if (
                not isinstance(matrix_row, list)
                or len(matrix_row) != len(opponent_support)
                or not isinstance(continuation_row, list)
                or len(continuation_row) != len(opponent_support)
            ):
                raise ValueError("shared-root replay payoff row has invalid dimensions")
            normalized_matrix_row = []
            normalized_continuation_row = []
            for payoff, continuation in zip(matrix_row, continuation_row, strict=True):
                if (
                    isinstance(payoff, bool)
                    or not isinstance(payoff, (int, float))
                    or not math.isfinite(payoff)
                    or not -1e-8 <= payoff <= 1 + 1e-8
                    or not isinstance(continuation, dict)
                    or set(continuation) != SHARED_ROOT_CONTINUATION_FIELDS
                ):
                    raise ValueError("shared-root replay continuation is invalid")
                payoff = float(payoff)
                integer_bounds = {
                    "seed": (0, (1 << 64) - 1),
                    "requested_iterations": (1, 1_000_000),
                    "executed_iterations": (1, 1_000_000),
                    "visits": (1, 1_000_000),
                    "total_score_bits": (0, (1 << 32) - 1),
                    "payoff_bits": (0, (1 << 64) - 1),
                }
                for name, (lower, upper) in integer_bounds.items():
                    item = continuation[name]
                    if isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper:
                        raise ValueError("shared-root replay continuation integer is invalid")
                total_score = continuation["total_score"]
                captured_payoff = continuation["payoff"]
                if (
                    isinstance(total_score, bool)
                    or not isinstance(total_score, (int, float))
                    or not math.isfinite(total_score)
                    or not 0 <= total_score <= continuation["visits"]
                    or isinstance(captured_payoff, bool)
                    or not isinstance(captured_payoff, (int, float))
                    or not math.isfinite(captured_payoff)
                    or continuation["requested_iterations"] != diagnostics["continuation_iterations"]
                    or continuation["executed_iterations"] != continuation["requested_iterations"]
                    or continuation["visits"] != continuation["executed_iterations"]
                    or struct.unpack("<I", struct.pack("<f", float(total_score)))[0]
                    != continuation["total_score_bits"]
                    or struct.unpack("<Q", struct.pack("<d", float(captured_payoff)))[0]
                    != continuation["payoff_bits"]
                    or abs(float(captured_payoff) - payoff) > 1e-12
                    or abs(float(captured_payoff) - float(total_score) / continuation["visits"]) > 1e-12
                ):
                    raise ValueError("shared-root replay continuation is inconsistent")
                normalized_matrix_row.append(payoff)
                normalized_continuation_row.append(
                    {
                        **continuation,
                        "total_score": float(total_score),
                        "payoff": float(captured_payoff),
                    }
                )
                payoff_parts.append(struct.pack("<d", payoff))
                payoff_cells += 1
            normalized_matrix.append(normalized_matrix_row)
            normalized_continuations.append(normalized_continuation_row)
            counterfactual_values[own_index] += weight * math.fsum(
                payoff * probability
                for payoff, probability in zip(
                    normalized_matrix_row, opponent_policy, strict=True
                )
            )
        particle_digest_parts.extend(
            (state.encode("utf-8"), struct.pack("<d", weight))
        )
        opponent_prior_parts.extend(
            [struct.pack("<d", probability) for probability in opponent_prior]
            if opponent_prior is not None
            else [b"none"]
        )
        normalized_particles.append(
            {
                "canonical_index": canonical_index,
                "state": state,
                "normalized_weight": weight,
                "source_particles": normalized_sources,
                "opponent_action_support": opponent_support,
                "normalized_opponent_prior": opponent_prior,
                "payoff_matrix": normalized_matrix,
                "continuations": normalized_continuations,
                "opponent_policy": opponent_policy,
            }
        )

    if len(all_source_indices) != diagnostics["positive_particle_count"]:
        raise ValueError("shared-root replay source count is inconsistent")
    weights = [particle["normalized_weight"] for particle in normalized_particles]
    if abs(math.fsum(weights) - 1.0) > 1e-8:
        raise ValueError("shared-root replay canonical weights are not normalized")
    source_weight_sum = math.fsum(
        source["input_weight"]
        for particle in normalized_particles
        for source in particle["source_particles"]
    )
    if abs(source_weight_sum - 1.0) > 1e-8:
        raise ValueError("shared-root replay source weights are not normalized")
    for particle in normalized_particles:
        source_sum = math.fsum(source["input_weight"] for source in particle["source_particles"])
        if abs(source_sum / source_weight_sum - particle["normalized_weight"]) > 1e-8:
            raise ValueError("shared-root replay canonical weight differs from its sources")
    if payoff_cells != diagnostics["payoff_cells"]:
        raise ValueError("shared-root replay payoff count is inconsistent")
    if any(
        abs(counterfactual - policy_row["counterfactual_value"]) > 1e-8
        for counterfactual, policy_row in zip(counterfactual_values, sorted(policy, key=lambda row: row["action"]), strict=True)
    ):
        raise ValueError("shared-root replay counterfactual values differ from the result")
    digest_expectations = {
        "action_support_digest": _fnv1a64_digest(action.encode("utf-8") for action in own_support),
        "particle_digest": _fnv1a64_digest(particle_digest_parts),
        "payoff_digest": _fnv1a64_digest(payoff_parts),
        "player_prior_digest": _fnv1a64_digest(
            [struct.pack("<d", probability) for probability in player_prior]
            if player_prior is not None
            else [b"none"]
        ),
        "opponent_prior_digest": _fnv1a64_digest(opponent_prior_parts),
    }
    if any(diagnostics[name] != expected for name, expected in digest_expectations.items()):
        raise ValueError("shared-root replay capture differs from a diagnostic digest")
    return {
        "schema_version": 1,
        "solver_contract": value["solver_contract"],
        "configuration": normalized_configuration,
        "own_action_support": own_support,
        "normalized_player_prior": player_prior,
        "canonical_particles": normalized_particles,
    }


def shared_root_result_payload(
    result,
    *,
    expected_particles: int | None = None,
    expected_iterations: int | None = None,
    expected_continuation_iterations: int | None = None,
    expected_seed: int | None = None,
    expected_prior_strength: float | None = None,
) -> dict[str, object]:
    payload = {
        "policy": [
            {
                "action": entry.action,
                "probability": entry.probability,
                "counterfactual_value": entry.counterfactual_value,
            }
            for entry in result.policy
        ],
        "opponent_policies": [
            [
                {"action": action, "probability": probability}
                for action, probability in policy
            ]
            for policy in result.opponent_policies
        ],
        "diagnostics": {
            name: getattr(result.diagnostics, name)
            for name in SHARED_ROOT_DIAGNOSTIC_FIELDS
        },
        "replay_capture": _shared_root_replay_capture_payload(result.replay_capture),
    }
    return validate_shared_root_result_payload(
        payload,
        expected_particles=expected_particles,
        expected_iterations=expected_iterations,
        expected_continuation_iterations=expected_continuation_iterations,
        expected_seed=expected_seed,
        expected_prior_strength=expected_prior_strength,
    )


def validate_shared_root_result_payload(
    value: object,
    *,
    expected_particles: int | None = None,
    expected_iterations: int | None = None,
    expected_continuation_iterations: int | None = None,
    expected_seed: int | None = None,
    expected_prior_strength: float | None = None,
    require_replay_capture: bool = False,
) -> dict[str, object]:
    legacy_fields = {"policy", "opponent_policies", "diagnostics"}
    fields = frozenset(value) if isinstance(value, dict) else frozenset()
    if not isinstance(value, dict) or fields not in (
        frozenset(legacy_fields),
        frozenset(legacy_fields | {"replay_capture"}),
    ):
        raise ValueError("shared-root result has invalid fields")
    if require_replay_capture and "replay_capture" not in value:
        raise ValueError("shared-root result is missing its replay capture")

    policy = value["policy"]
    if not isinstance(policy, list) or not 1 <= len(policy) <= 16:
        raise ValueError("shared-root policy is invalid")
    normalized_policy = []
    actions = set()
    for row in policy:
        if not isinstance(row, dict) or set(row) != {
            "action",
            "probability",
            "counterfactual_value",
        }:
            raise ValueError("shared-root policy contains an invalid entry")
        action = row["action"]
        probability = row["probability"]
        counterfactual_value = row["counterfactual_value"]
        if (
            not isinstance(action, str)
            or not 0 < len(action) <= 128
            or action in actions
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or isinstance(counterfactual_value, bool)
            or not isinstance(counterfactual_value, (int, float))
            or not math.isfinite(counterfactual_value)
            or not -1e-8 <= counterfactual_value <= 1 + 1e-8
        ):
            raise ValueError("shared-root policy contains an invalid entry")
        actions.add(action)
        normalized_policy.append(
            {
                "action": action,
                "probability": float(probability),
                "counterfactual_value": float(counterfactual_value),
            }
        )
    if abs(math.fsum(row["probability"] for row in normalized_policy) - 1.0) > 1e-8:
        raise ValueError("shared-root policy is not normalized")

    opponent_policies = value["opponent_policies"]
    if not isinstance(opponent_policies, list) or len(opponent_policies) > 64:
        raise ValueError("shared-root opponent policies are invalid")
    normalized_opponents = []
    for policy_index, opponent_policy in enumerate(opponent_policies):
        if not isinstance(opponent_policy, list) or not 1 <= len(opponent_policy) <= 16:
            raise ValueError("shared-root opponent policy is invalid")
        opponent_actions = set()
        normalized_opponent = []
        for row in opponent_policy:
            if not isinstance(row, dict) or set(row) != {"action", "probability"}:
                raise ValueError("shared-root opponent policy contains an invalid entry")
            action = row["action"]
            probability = row["probability"]
            if (
                not isinstance(action, str)
                or not 0 < len(action) <= 128
                or action in opponent_actions
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("shared-root opponent policy contains an invalid entry")
            opponent_actions.add(action)
            normalized_opponent.append(
                {"action": action, "probability": float(probability)}
            )
        if abs(math.fsum(row["probability"] for row in normalized_opponent) - 1.0) > 1e-8:
            raise ValueError(
                f"shared-root opponent policy {policy_index} is not normalized"
            )
        normalized_opponents.append(normalized_opponent)

    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != SHARED_ROOT_DIAGNOSTIC_FIELDS:
        raise ValueError("shared-root diagnostics have invalid fields")
    if diagnostics["solver_contract"] != "weighted-shared-rm-plus-v1":
        raise ValueError("shared-root solver contract is invalid")
    integer_bounds = {
        "iterations": (1, 1_000_000),
        "continuation_iterations": (1, 1_000_000),
        "seed": (0, (1 << 64) - 1),
        "payoff_cells": (1, 16_384),
        "total_forced_continuation_iterations": (1, 100_000_000),
        "input_particle_count": (1, 64),
        "positive_particle_count": (1, 64),
        "canonical_particle_count": (1, 64),
    }
    normalized_diagnostics = dict(diagnostics)
    for name, (lower, upper) in integer_bounds.items():
        field = diagnostics[name]
        if isinstance(field, bool) or not isinstance(field, int) or not lower <= field <= upper:
            raise ValueError(f"shared-root diagnostics contain invalid {name}")
    bounded_values = (
        "expected_value",
        "player_best_response_value",
        "opponent_best_response_value",
    )
    nonnegative_values = (
        "player_best_response_gain",
        "opponent_best_response_gain",
        "nash_conv",
        "exploitability",
        "player_regret_bound",
        "opponent_regret_bound",
        "total_regret_bound",
    )
    for name in bounded_values + nonnegative_values + (
        "normalized_weight_sum",
        "prior_strength",
    ):
        field = diagnostics[name]
        if isinstance(field, bool) or not isinstance(field, (int, float)) or not math.isfinite(field):
            raise ValueError(f"shared-root diagnostics contain invalid {name}")
        normalized_diagnostics[name] = float(field)
    if any(not -1e-8 <= normalized_diagnostics[name] <= 1 + 1e-8 for name in bounded_values):
        raise ValueError("shared-root values are outside their payoff range")
    if any(normalized_diagnostics[name] < -1e-12 for name in nonnegative_values):
        raise ValueError("shared-root diagnostics contain a negative gap or regret")
    if not 0 <= normalized_diagnostics["prior_strength"] <= 1_000:
        raise ValueError("shared-root prior strength is invalid")
    for name in (
        "action_support_digest",
        "particle_digest",
        "payoff_digest",
        "player_prior_digest",
        "opponent_prior_digest",
    ):
        field = diagnostics[name]
        if (
            not isinstance(field, str)
            or len(field) != 24
            or not field.startswith("fnv1a64:")
            or any(character not in "0123456789abcdef" for character in field[8:])
        ):
            raise ValueError(f"shared-root diagnostics contain invalid {name}")

    tolerance = 1e-8
    if diagnostics["positive_particle_count"] > diagnostics["input_particle_count"]:
        raise ValueError("shared-root particle counts are inconsistent")
    if diagnostics["canonical_particle_count"] > diagnostics["positive_particle_count"]:
        raise ValueError("shared-root canonical particle count is inconsistent")
    if len(normalized_opponents) != diagnostics["canonical_particle_count"]:
        raise ValueError("shared-root opponent policy count is inconsistent")
    if abs(normalized_diagnostics["normalized_weight_sum"] - 1.0) > tolerance:
        raise ValueError("shared-root canonical weights are not normalized")
    if diagnostics["total_forced_continuation_iterations"] != diagnostics["payoff_cells"] * diagnostics["continuation_iterations"]:
        raise ValueError("shared-root continuation work is inconsistent")
    expected_value = math.fsum(
        row["probability"] * row["counterfactual_value"]
        for row in normalized_policy
    )
    if abs(expected_value - normalized_diagnostics["expected_value"]) > tolerance:
        raise ValueError("shared-root expected value is inconsistent")
    if abs(
        normalized_diagnostics["player_best_response_value"]
        - normalized_diagnostics["expected_value"]
        - normalized_diagnostics["player_best_response_gain"]
    ) > tolerance:
        raise ValueError("shared-root player best-response gain is inconsistent")
    if abs(
        normalized_diagnostics["expected_value"]
        - normalized_diagnostics["opponent_best_response_value"]
        - normalized_diagnostics["opponent_best_response_gain"]
    ) > tolerance:
        raise ValueError("shared-root opponent best-response gain is inconsistent")
    if abs(normalized_diagnostics["nash_conv"] - normalized_diagnostics["player_best_response_gain"] - normalized_diagnostics["opponent_best_response_gain"]) > tolerance:
        raise ValueError("shared-root NashConv is inconsistent")
    if abs(normalized_diagnostics["exploitability"] * 2 - normalized_diagnostics["nash_conv"]) > tolerance:
        raise ValueError("shared-root exploitability is inconsistent")
    if abs(normalized_diagnostics["total_regret_bound"] - normalized_diagnostics["player_regret_bound"] - normalized_diagnostics["opponent_regret_bound"]) > tolerance:
        raise ValueError("shared-root regret bounds are inconsistent")
    if expected_particles is not None and diagnostics["input_particle_count"] != expected_particles:
        raise ValueError("shared-root particle count differs from the request")
    if expected_iterations is not None and diagnostics["iterations"] != expected_iterations:
        raise ValueError("shared-root iterations differ from the request")
    if expected_continuation_iterations is not None and diagnostics["continuation_iterations"] != expected_continuation_iterations:
        raise ValueError("shared-root continuation iterations differ from the request")
    if expected_seed is not None and diagnostics["seed"] != expected_seed:
        raise ValueError("shared-root seed differs from the request")
    if (
        expected_prior_strength is not None
        and abs(
            normalized_diagnostics["prior_strength"] - expected_prior_strength
        )
        > tolerance
    ):
        raise ValueError("shared-root prior strength differs from the request")
    normalized_result = {
        "policy": normalized_policy,
        "opponent_policies": normalized_opponents,
        "diagnostics": normalized_diagnostics,
    }
    if "replay_capture" in value:
        normalized_result["replay_capture"] = _validate_shared_root_replay_capture(
            value["replay_capture"],
            policy=normalized_policy,
            opponent_policies=normalized_opponents,
            diagnostics=normalized_diagnostics,
        )
    return normalized_result


def result_payload(result) -> dict[str, object]:
    def side(options, label: str) -> tuple[list[dict[str, object]], int]:
        payload = []
        moves: set[str] = set()
        visit_sum = 0
        for option in options:
            move = option.move_choice
            score = option.total_score
            visits = option.visits
            if not isinstance(move, str) or not move or move in moves:
                raise ValueError(f"MCTS {label} contains an invalid action")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"MCTS {label} contains an invalid score")
            if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
                raise ValueError(f"MCTS {label} contains invalid visits")
            if not 0 <= score <= visits:
                raise ValueError(f"MCTS {label} contains a score outside its visits")
            moves.add(move)
            visit_sum += visits
            payload.append(
                {
                    "move_choice": move,
                    "total_score": float(score),
                    "visits": visits,
                }
            )
        return payload, visit_sum

    total_visits = result.total_visits
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
    ):
        raise ValueError("MCTS result contains invalid total visits")
    side_one, side_one_visits = side(result.side_one, "side_one")
    side_two, side_two_visits = side(result.side_two, "side_two")
    if side_one_visits != side_two_visits or side_one_visits > total_visits:
        raise ValueError("MCTS result contains inconsistent visit totals")

    return {
        "side_one": side_one,
        "side_two": side_two,
        "total_visits": total_visits,
    }


def validate_result_payload(value: object) -> dict[str, object]:
    """Validate an untrusted wire result using the producer's invariants."""
    if not isinstance(value, dict):
        raise ValueError("MCTS result must be an object")

    def side(name: str) -> tuple[list[dict[str, object]], int]:
        rows = value.get(name)
        if (
            not isinstance(rows, list)
            or len(rows) > 64
            or (name == "side_one" and not rows)
        ):
            raise ValueError(f"MCTS {name} is invalid")
        payload = []
        moves = set()
        visit_sum = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"MCTS {name} contains an invalid option")
            move = row.get("move_choice")
            score = row.get("total_score")
            visits = row.get("visits")
            if (
                not isinstance(move, str)
                or not 0 < len(move) <= 128
                or move in moves
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or isinstance(visits, bool)
                or not isinstance(visits, int)
                or visits < 0
                or not 0 <= score <= visits
            ):
                raise ValueError(f"MCTS {name} contains an invalid option")
            moves.add(move)
            visit_sum += visits
            payload.append(
                {"move_choice": move, "total_score": float(score), "visits": visits}
            )
        return payload, visit_sum

    total_visits = value.get("total_visits")
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
    ):
        raise ValueError("MCTS result contains invalid total visits")
    side_one, side_one_visits = side("side_one")
    side_two, side_two_visits = side("side_two")
    if side_one_visits != side_two_visits or side_one_visits > total_visits:
        raise ValueError("MCTS result contains inconsistent visit totals")
    return {
        "side_one": side_one,
        "side_two": side_two,
        "total_visits": total_visits,
    }


def validate_loopback_search_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        parsed.port
    except ValueError as exc:
        raise ValueError("remote HTTP MCTS URL must use a loopback IP address") from exc
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/search"
    ):
        raise ValueError(
            "remote HTTP MCTS URL must be a loopback http://.../search URL"
        )
    return value
