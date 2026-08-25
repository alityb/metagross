#!/usr/bin/env python3
"""Validate and exactly replay production shared-root search captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    MAX_SHARED_ROOT_REPLAY_BYTES,
    shared_root_result_payload,
    validate_priors,
    validate_shared_root_result_payload,
)
from srcs.metagross.world_provenance import state_sha256


ENVELOPE_FIELDS = {
    "schema_version",
    "capture_kind",
    "source_particles",
    "source_weight_sum",
    "solver",
    "sampling",
    "request_ids",
    "request_action_support",
    "action_aliases",
    "engine",
    "native_capture_sha256",
    "capture_sha256",
}
SOURCE_FIELDS = {
    "source_index",
    "serialized_state",
    "state_sha256",
    "source_weight",
    "normalized_weight",
}
MAX_SEARCH_ROW_BYTES = 2 * MAX_SHARED_ROOT_REPLAY_BYTES + 2_000_000


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _bounded_float(value: object, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _positive_integer(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _sampled_action(policy: list[dict], seed: int) -> tuple[str, float]:
    ordered = sorted(
        policy,
        key=lambda row: (
            -row["probability"],
            -row["counterfactual_value"],
            row["action"],
        ),
    )
    draw = random.Random(seed).random()
    cumulative = 0.0
    selected = ordered[-1]["action"]
    for row in ordered:
        cumulative += row["probability"]
        if draw < cumulative:
            selected = row["action"]
            break
    return selected, draw


def _condition_prior(prior, support: list[str]) -> list[float] | None:
    if prior is None:
        return None
    by_action = {action.lower(): probability for action, probability in prior}
    values = [float(by_action.get(action, 0.0)) for action in support]
    total = math.fsum(values)
    if total <= 0:
        return None
    return [value / total for value in values]


def _vectors_equal(left, right, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return len(left) == len(right) and all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )


def validate_search_row(row: object, *, rerun: bool = False) -> dict[str, object]:
    if not isinstance(row, dict) or row.get("schema") != 4:
        raise ValueError("shared-root search row must use schema 4")
    if not isinstance(row.get("shared_root_replay"), dict):
        raise ValueError("shared-root search row has no replay envelope")
    result = validate_shared_root_result_payload(
        row.get("shared_root"), require_replay_capture=True
    )
    envelope = row["shared_root_replay"]
    if set(envelope) != ENVELOPE_FIELDS:
        raise ValueError("shared-root replay envelope has invalid fields")
    if envelope["schema_version"] != 1 or envelope["capture_kind"] != "production_shared_root_replay_v1":
        raise ValueError("shared-root replay envelope has an invalid contract")
    unsigned = {**envelope}
    claimed_capture_sha256 = unsigned.pop("capture_sha256")
    if claimed_capture_sha256 != canonical_sha256(unsigned):
        raise ValueError("shared-root replay envelope hash is invalid")
    if envelope["native_capture_sha256"] != canonical_sha256(result["replay_capture"]):
        raise ValueError("shared-root native replay capture hash is invalid")

    source_particles = envelope["source_particles"]
    if not isinstance(source_particles, list) or not 1 <= len(source_particles) <= 64:
        raise ValueError("shared-root replay source particles are invalid")
    normalized_sources = []
    total_state_bytes = 0
    for source_index, source in enumerate(source_particles):
        if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
            raise ValueError("shared-root replay source particle has invalid fields")
        state = source["serialized_state"]
        if (
            source["source_index"] != source_index
            or not isinstance(state, str)
            or not state
            or len(state) > 1_000_000
            or source["state_sha256"] != state_sha256(state)
        ):
            raise ValueError("shared-root replay source identity is invalid")
        total_state_bytes += len(state.encode("utf-8"))
        if total_state_bytes > 64_000_000:
            raise ValueError("shared-root replay source states exceed the size bound")
        normalized_sources.append(
            {
                **source,
                "source_weight": _bounded_float(
                    source["source_weight"], "shared-root source weight"
                ),
                "normalized_weight": _bounded_float(
                    source["normalized_weight"], "shared-root normalized weight"
                ),
            }
        )
    source_weight_sum = _bounded_float(
        envelope["source_weight_sum"], "shared-root source weight sum", minimum=1e-300
    )
    if abs(math.fsum(source["source_weight"] for source in normalized_sources) - source_weight_sum) > 1e-8:
        raise ValueError("shared-root source weight sum is inconsistent")
    if abs(math.fsum(source["normalized_weight"] for source in normalized_sources) - 1.0) > 1e-8:
        raise ValueError("shared-root normalized source weights do not sum to one")
    for source in normalized_sources:
        if abs(source["source_weight"] / source_weight_sum - source["normalized_weight"]) > 1e-8:
            raise ValueError("shared-root normalized source weight is inconsistent")

    solver = envelope["solver"]
    if not isinstance(solver, dict) or set(solver) != {
        "iterations",
        "continuation_iterations",
        "seed",
        "prior_strength",
        "s1_prior",
        "s2_priors",
    }:
        raise ValueError("shared-root replay solver configuration is invalid")
    iterations = _positive_integer(solver["iterations"], "shared-root iterations", 1_000_000)
    continuation_iterations = _positive_integer(
        solver["continuation_iterations"], "shared-root continuation iterations", 1_000_000
    )
    seed = solver["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError("shared-root solver seed is invalid")
    prior_strength = _bounded_float(solver["prior_strength"], "shared-root prior strength")
    if prior_strength > 1_000:
        raise ValueError("shared-root prior strength is invalid")
    s1_prior = validate_priors(solver["s1_prior"], "shared-root s1_prior")
    s2_priors = solver["s2_priors"]
    if not isinstance(s2_priors, list) or len(s2_priors) != len(source_particles):
        raise ValueError("shared-root s2 priors are invalid")
    normalized_s2 = [
        validate_priors(prior, f"shared-root s2_priors[{index}]")
        for index, prior in enumerate(s2_priors)
    ]
    diagnostics = result["diagnostics"]
    if (
        diagnostics["input_particle_count"] != len(source_particles)
        or diagnostics["iterations"] != iterations
        or diagnostics["continuation_iterations"] != continuation_iterations
        or diagnostics["seed"] != seed
        or abs(diagnostics["prior_strength"] - prior_strength) > 1e-8
    ):
        raise ValueError("shared-root replay solver configuration differs from the result")
    native_capture = result["replay_capture"]
    if not _vectors_equal(
        _condition_prior(s1_prior, native_capture["own_action_support"]),
        native_capture["normalized_player_prior"],
    ):
        raise ValueError("shared-root replay player prior differs from the native capture")
    row_player_priors = validate_priors(row.get("player_priors"), "row player_priors")
    row_opponent_priors = validate_priors(
        row.get("opponent_priors"), "row opponent_priors"
    )
    if row_player_priors != s1_prior or any(prior != row_opponent_priors for prior in normalized_s2):
        raise ValueError("shared-root replay priors differ from the search row")

    native_sources = {}
    for canonical_particle in native_capture["canonical_particles"]:
        for source in canonical_particle["source_particles"]:
            native_sources[source["input_index"]] = (
                canonical_particle["state"],
                source["input_weight"],
                canonical_particle["opponent_action_support"],
                canonical_particle["normalized_opponent_prior"],
            )
    positive_sources = [
        source for source in normalized_sources if source["normalized_weight"] > 0
    ]
    if len(native_sources) != len(positive_sources):
        raise ValueError("shared-root native source membership is incomplete")
    for source in positive_sources:
        native_state, native_weight, opponent_support, native_opponent_prior = native_sources[
            source["source_index"]
        ]
        source_state = source["serialized_state"]
        if native_state != source_state:
            import poke_engine

            source_state = poke_engine.State.from_string(source_state).to_string()
        if native_state != source_state or abs(native_weight - source["normalized_weight"]) > 1e-8:
            raise ValueError("shared-root native source membership differs from the envelope")
        if not _vectors_equal(
            _condition_prior(normalized_s2[source["source_index"]], opponent_support),
            native_opponent_prior,
        ):
            raise ValueError("shared-root opponent prior differs from the native capture")

    sampling = envelope["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != {
        "world_channel",
        "world_seed",
        "action_channel",
        "action_seed",
    }:
        raise ValueError("shared-root replay sampling metadata is invalid")
    if sampling["world_channel"] != "selection-worlds" or sampling["action_channel"] != "shared-root-action":
        raise ValueError("shared-root replay sampling channel is invalid")
    for name in ("world_seed", "action_seed"):
        value = sampling[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
            raise ValueError("shared-root replay sampling seed is invalid")
    request_support = envelope["request_action_support"]
    aliases = envelope["action_aliases"]
    if (
        not isinstance(request_support, list)
        or request_support != sorted(request_support)
        or not request_support
        or len(set(request_support)) != len(request_support)
        or not isinstance(aliases, list)
        or len(aliases) != len(result["policy"])
    ):
        raise ValueError("shared-root replay action mapping is invalid")
    mapped_policy = []
    mapped_actions = set()
    from srcs.metagross.run_foul_play import _authorized_action_name

    for policy_row, alias in zip(result["policy"], aliases, strict=True):
        if (
            not isinstance(alias, dict)
            or set(alias) != {"native_action", "request_action"}
            or alias["native_action"] != policy_row["action"]
            or alias["request_action"] not in request_support
            or alias["request_action"] in mapped_actions
            or _authorized_action_name(
                alias["native_action"], set(request_support)
            )
            != alias["request_action"]
        ):
            raise ValueError("shared-root replay action alias is invalid")
        mapped_actions.add(alias["request_action"])
        mapped_policy.append({**policy_row, "action": alias["request_action"]})
    sampled_action, draw = _sampled_action(mapped_policy, sampling["action_seed"])
    override = row.get("choice_override")
    captured_draw = (
        override.get("mixed_strategy_draw") if isinstance(override, dict) else None
    )
    if (
        not isinstance(override, dict)
        or override.get("sampled_action") != sampled_action
        or isinstance(captured_draw, bool)
        or not isinstance(captured_draw, (int, float))
        or not math.isfinite(captured_draw)
        or abs(float(captured_draw) - draw) > 1e-15
    ):
        raise ValueError("shared-root replay sampled action is inconsistent")

    engine = envelope["engine"]
    if not isinstance(engine, dict) or set(engine) != {
        "contract",
        "source_sha256",
        "native_sha256",
        "distribution_version",
    }:
        raise ValueError("shared-root replay engine identity is invalid")
    if (
        not isinstance(engine["distribution_version"], str)
        or not 0 < len(engine["distribution_version"]) <= 64
    ):
        raise ValueError("shared-root replay engine version is invalid")
    if engine["contract"] != ENGINE_CONTRACT or engine["source_sha256"] != ENGINE_SOURCE_SHA256:
        raise ValueError("shared-root replay engine contract is invalid")
    for name in ("source_sha256", "native_sha256"):
        digest = engine[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("shared-root replay engine digest is invalid")
    request_ids = envelope["request_ids"]
    if not isinstance(request_ids, list) or len(request_ids) > 64 or any(
        not isinstance(request_id, str) or not request_id or len(request_id) > 128
        for request_id in request_ids
    ):
        raise ValueError("shared-root replay request IDs are invalid")
    remote_search = row.get("remote_search")
    if not isinstance(remote_search, dict):
        raise ValueError("shared-root replay search telemetry is missing")
    if (
        remote_search.get("sampling_seed") != sampling["world_seed"]
        or remote_search.get("action_seed") != sampling["action_seed"]
        or list(remote_search.get("request_ids") or []) != request_ids
    ):
        raise ValueError("shared-root replay sampling metadata differs from telemetry")
    remote_engine = remote_search.get("engine")
    if remote_search.get("transport") == "local":
        import importlib

        native_path = Path(importlib.import_module("poke_engine.poke_engine").__file__)
        if hashlib.sha256(native_path.read_bytes()).hexdigest() != engine["native_sha256"]:
            raise ValueError("shared-root local native identity differs from the capture")
    elif not isinstance(remote_engine, dict) or any(
        remote_engine.get(name) != engine[name]
        for name in (
            "contract",
            "source_sha256",
            "native_sha256",
            "distribution_version",
        )
    ):
        raise ValueError("shared-root replay engine identity differs from telemetry")

    replayed = None
    if rerun:
        import poke_engine

        native_result = poke_engine.shared_information_set_root_search(
            states=[poke_engine.State.from_string(source["serialized_state"]) for source in normalized_sources],
            particle_weights=[source["normalized_weight"] for source in normalized_sources],
            iterations=iterations,
            continuation_iterations=continuation_iterations,
            seed=seed,
            prior_strength=prior_strength,
            s1_prior=s1_prior,
            s2_priors=normalized_s2,
        )
        replayed = shared_root_result_payload(
            native_result,
            expected_particles=len(source_particles),
            expected_iterations=iterations,
            expected_continuation_iterations=continuation_iterations,
            expected_seed=seed,
            expected_prior_strength=prior_strength,
        )
        if canonical_sha256(replayed) != canonical_sha256(result):
            raise ValueError("shared-root exact replay differs from the captured result")
    return {
        "capture_sha256": claimed_capture_sha256,
        "particles": len(source_particles),
        "canonical_particles": diagnostics["canonical_particle_count"],
        "payoff_cells": diagnostics["payoff_cells"],
        "sampled_action": sampled_action,
        "exact_replay": rerun
        and canonical_sha256(replayed) == canonical_sha256(result),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    records = []
    with args.input.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_SEARCH_ROW_BYTES:
                raise ValueError(f"capture line {line_number}: encoded row exceeds the size bound")
            try:
                summary = validate_search_row(json.loads(line), rerun=args.rerun)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"capture line {line_number}: {exc}") from exc
            records.append({"line_number": line_number, **summary})
    report = {
        "schema_version": 1,
        "input": str(args.input.expanduser().resolve()),
        "captures": len(records),
        "all_exact": bool(records) and all(record["exact_replay"] for record in records),
        "records": records,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
