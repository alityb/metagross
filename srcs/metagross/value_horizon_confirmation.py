#!/usr/bin/env python3
"""Confirm the selected horizon-2 value intervention on 384 disjoint roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from srcs.metagross.known_team_decision_v2 import canonical_json, sha256_path
from srcs.metagross.known_team_search_failure_attribution import selected_particles
from srcs.metagross.run_foul_play import _authorized_action_name
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    REQUEST_SCHEMA,
    validate_holdout_result_payload,
)
from srcs.metagross.value_horizon_tournament import _call_remote, _select_action


PROTOCOL_SCHEMA = "metagross-value-horizon-confirmation-protocol/v1"
RESULT_SCHEMA = "metagross-value-horizon-confirmation/v1"
HORIZON = 2
REPEATS = 2
ROLLOUTS = 64
CONTINUATION_ITERATIONS = 64
BENEFICIAL_DELTA = 0.02
HARMFUL_DELTA = -0.02
CATASTROPHIC_DELTA = -0.10
MASTER_SEED = 2026081302
BOOTSTRAPS = 10_000


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    identity = row.get("identity")
    if isinstance(identity, list):
        return str(identity[0]), str(identity[1]), int(identity[2])
    if isinstance(identity, dict):
        return str(identity["corpus_uid"]), str(identity["observer"]), int(identity["decision_idx"])
    return str(row["corpus_uid"]), str(row["observer"]), int(row["decision_idx"])


def _seed(identity: Sequence[object], repeat: int, particle: int) -> int:
    material = canonical_json(
        [MASTER_SEED, *identity, "horizon", HORIZON, "repeat", repeat, "particle", particle]
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _request_id(identity: Sequence[object], action: str, repeat: int, particle: int) -> str:
    return _sha256_bytes(
        canonical_json([RESULT_SCHEMA, *identity, action, repeat, particle]).encode("ascii")
    )


def _protocol_payload(
    phase1_path: Path,
    bank_path: Path,
    attribution_path: Path,
    tournament_path: Path,
) -> dict[str, Any]:
    dependency = Path(__file__).with_name("value_horizon_tournament.py")
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen_before_execution",
        "inputs": {
            "phase1": {"path": str(phase1_path), "sha256": sha256_path(phase1_path)},
            "world_bank": {"path": str(bank_path), "sha256": sha256_path(bank_path)},
            "failure_attribution": {
                "path": str(attribution_path),
                "sha256": sha256_path(attribution_path),
            },
            "development_tournament": {
                "path": str(tournament_path),
                "sha256": sha256_path(tournament_path),
                "required_winner": HORIZON,
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(Path(__file__).resolve()),
            "tournament_dependency_sha256": sha256_path(dependency),
        },
        "remote": {
            "request_schema": REQUEST_SCHEMA,
            "engine_contract": ENGINE_CONTRACT,
            "engine_source_sha256": ENGINE_SOURCE_SHA256,
            "expected_physical_cores": 64,
        },
        "configuration": {
            "cohort": "all_411_world_bank_roots_excluding_27_development_roots_then_truth_blind_native_support_filter",
            "source_roots": 384,
            "minimum_supported_roots": 300,
            "native_support_filter": "baseline_and_at_least_one_candidate_must_map_uniquely_in_all_16_particles; engine-only actions ignored",
            "particle_panel": "failure_attribution_seeded_systematic_16",
            "particle_aggregation": "equal_over_16_occurrences",
            "horizon": HORIZON,
            "repeats": REPEATS,
            "rollouts": ROLLOUTS,
            "continuation_iterations": CONTINUATION_ITERATIONS,
            "master_seed": MASTER_SEED,
            "opponent_priors": None,
            "common_random_numbers_exclude_candidate_action": True,
            "action_selection": "maximum_belief_mean_paired_delta_vs_baseline_with_lexical_tie_break",
            "teacher_role": "post_selection_diagnostic_only",
            "bootstrap_count": BOOTSTRAPS,
            "beneficial_teacher_delta": BENEFICIAL_DELTA,
            "harmful_teacher_delta": HARMFUL_DELTA,
            "catastrophic_teacher_delta": CATASTROPHIC_DELTA,
            "promotion_gate": {
                "minimum_seed_agreement_fraction": 0.80,
                "minimum_changed_roots": 20,
                "minimum_beneficial_to_harmful_ratio": 1.5,
                "minimum_mean_teacher_delta_ci95_lower": 0.0,
                "maximum_catastrophic_roots": 1,
            },
            "public_ladder_authorized": False,
            "model_training_authorized_by_protocol": False,
        },
    }


def freeze_protocol(*paths: Path) -> dict[str, Any]:
    protocol = _protocol_payload(*paths)
    protocol["protocol_sha256"] = _sha256_bytes(canonical_json(protocol).encode("ascii"))
    return protocol


def _validate_protocol(protocol: Mapping[str, Any], *paths: Path) -> None:
    unhashed = dict(protocol)
    claimed = unhashed.pop("protocol_sha256", None)
    if claimed != _sha256_bytes(canonical_json(unhashed).encode("ascii")):
        raise ValueError("confirmation protocol hash does not match content")
    if unhashed != _protocol_payload(*paths):
        raise ValueError("confirmation differs from its frozen protocol")


def _bootstrap_mean(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("confirmation bootstrap requires at least two roots")
    rng = random.Random(MASTER_SEED)
    count = len(values)
    estimates = []
    for _ in range(BOOTSTRAPS):
        estimates.append(math.fsum(values[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    return estimates[int(0.025 * BOOTSTRAPS)], estimates[int(0.975 * BOOTSTRAPS) - 1]


def _load_inputs(
    phase1_path: Path, bank_path: Path, attribution_path: Path, tournament_path: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    tournament = json.loads(tournament_path.read_text(encoding="utf-8"))
    if tournament.get("decision", {}).get("winner") != HORIZON:
        raise ValueError("development tournament did not select confirmation horizon")
    excluded = {_identity(root) for root in attribution["roots"]}
    phase1_by_identity = {_identity(root): root for root in phase1["roots"]}
    roots = [root for root in bank["roots"] if _identity(root) not in excluded]
    if len(excluded) != 27 or len(roots) != 384 or len(phase1_by_identity) != 411:
        raise ValueError("confirmation cohort is not the disjoint frozen 384-root set")
    if any(_identity(root) not in phase1_by_identity for root in roots):
        raise ValueError("confirmation input join is incomplete")
    return roots, phase1_by_identity


def _build_requests(
    roots: Sequence[Mapping[str, Any]],
    phase1: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[tuple[str, str, int], str, int, int]],
    list[dict[str, Any]],
    dict[tuple[str, str, int], list[str]],
]:
    requests = []
    cells = {}
    native_action_cache: dict[str, dict[str, str]] = {}
    import poke_engine

    eligible_roots = []
    common_supports: dict[tuple[str, str, int], list[str]] = {}
    for bank_root in roots:
        identity = _identity(bank_root)
        phase1_root = phase1[identity]
        baseline = str(phase1_root["baseline_action"])
        particles = selected_particles(bank_root)
        if len(particles) != 16:
            raise ValueError("confirmation root does not have 16 particles")
        particle_maps = []
        allowed = set(phase1_root["legal_actions"])
        for particle in particles:
            state_hash = str(particle["state_sha256"])
            native_actions = native_action_cache.get(state_hash)
            if native_actions is None:
                result = poke_engine.monte_carlo_tree_search(
                    poke_engine.State.from_string(particle["state"]),
                    duration_ms=0,
                    iterations=1,
                    threads=1,
                )
                native_actions = {}
                for row in result.side_one:
                    request_action = _authorized_action_name(str(row.move_choice), allowed)
                    if request_action is None:
                        # Engine-level actions that the private Showdown request
                        # forbids are not part of the public decision support.
                        continue
                    if request_action in native_actions:
                        raise ValueError(
                            "confirmation native/request action mapping is ambiguous"
                        )
                    native_actions[request_action] = str(row.move_choice)
                native_action_cache[state_hash] = native_actions
            particle_maps.append(native_actions)
        common_support = set.intersection(*(set(row) for row in particle_maps))
        candidates = sorted(common_support - {baseline})
        if baseline not in common_support or not candidates:
            continue
        eligible_roots.append(bank_root)
        common_supports[identity] = sorted(common_support)
        for repeat in range(REPEATS):
            for action in candidates:
                for particle_index, particle in enumerate(particles):
                    state_hash = str(particle["state_sha256"])
                    native_actions = native_action_cache[state_hash]
                    request_id = _request_id(identity, action, repeat, particle_index)
                    requests.append(
                        {
                            "schema": REQUEST_SCHEMA,
                            "request_id": request_id,
                            "index": len(requests),
                            "operation": "paired_holdout",
                            "state": particle["state"],
                            "baseline_action": native_actions[baseline],
                            "candidate_action": native_actions[action],
                            "rollouts": ROLLOUTS,
                            "continuation_iterations": CONTINUATION_ITERATIONS,
                            "continuation_steps": HORIZON,
                            "seed": _seed(identity, repeat, particle_index),
                            "opponent_priors": None,
                        }
                    )
                    cells[request_id] = (identity, action, repeat, particle_index)
    if len(cells) != len(requests):
        raise ValueError("confirmation request IDs are not unique")
    if len(eligible_roots) < 300:
        raise ValueError("confirmation native-support cohort is undersized")
    return requests, cells, eligible_roots, common_supports


def _summarize(
    roots: Sequence[Mapping[str, Any]],
    phase1: Mapping[tuple[str, str, int], Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    request_cells: Mapping[str, tuple[tuple[str, str, int], str, int, int]],
    common_supports: Mapping[tuple[str, str, int], Sequence[str]],
    wall_seconds: float,
) -> dict[str, Any]:
    cells: dict[tuple[tuple[str, str, int], str, int], list[float]] = {}
    seen = set()
    canonical_engine = None
    executed = 0
    for response in responses:
        request_id = response.get("request_id")
        if request_id not in request_cells or request_id in seen or response.get("ok") is not True:
            raise ValueError(f"invalid or failed confirmation response: {request_id}")
        seen.add(request_id)
        engine = response.get("engine")
        if not isinstance(engine, dict) or engine.get("contract") != ENGINE_CONTRACT or engine.get("source_sha256") != ENGINE_SOURCE_SHA256:
            raise ValueError("confirmation engine identity mismatch")
        if canonical_engine is None:
            canonical_engine = engine
        elif engine != canonical_engine:
            raise ValueError("confirmation used multiple engine identities")
        result = validate_holdout_result_payload(response.get("result"), expected_pairs=ROLLOUTS)
        identity, action, repeat, _particle = request_cells[str(request_id)]
        cells.setdefault((identity, action, repeat), []).append(
            float(result["delta_sum"]) / int(result["pairs"])
        )
        executed += int(result["continuation_iterations_executed"])
    if seen != set(request_cells):
        raise ValueError("confirmation response coverage is incomplete")

    root_rows = []
    for bank_root in roots:
        identity = _identity(bank_root)
        source = phase1[identity]
        baseline = str(source["baseline_action"])
        candidates = sorted(set(common_supports[identity]) - {baseline})
        repeats = []
        for repeat in range(REPEATS):
            deltas = {}
            for action in candidates:
                values = cells.get((identity, action, repeat))
                if values is None or len(values) != 16:
                    raise ValueError("confirmation action cell is incomplete")
                deltas[action] = math.fsum(values) / len(values)
            selected = _select_action(deltas, baseline)
            repeats.append(
                {
                    "repeat": repeat,
                    "selected_action": selected,
                    "selected_delta": 0.0 if selected == baseline else deltas[selected],
                }
            )
        actions = [row["selected_action"] for row in repeats]
        agreement = len(set(actions)) == 1
        selected_action = actions[0] if agreement else None
        teacher_delta = (
            float(source["teacher_mean_q"][selected_action])
            - float(source["teacher_mean_q"][baseline])
            if selected_action is not None
            else None
        )
        root_rows.append(
            {
                "identity": list(identity),
                "battle_id": source["battle_id"],
                "panel": source["panel"],
                "turn": int(source["turn"]),
                "baseline_action": baseline,
                "selected_action": selected_action,
                "seed_agreement": agreement,
                "changed": selected_action is not None and selected_action != baseline,
                "teacher_delta": teacher_delta,
                "beneficial": teacher_delta is not None and teacher_delta >= BENEFICIAL_DELTA,
                "harmful": teacher_delta is not None and teacher_delta <= HARMFUL_DELTA,
                "catastrophic": teacher_delta is not None and teacher_delta <= CATASTROPHIC_DELTA,
                "repeats": repeats,
            }
        )
    agreed = [row for row in root_rows if row["seed_agreement"]]
    deltas = [float(row["teacher_delta"]) for row in agreed]
    beneficial = sum(row["beneficial"] for row in root_rows)
    harmful = sum(row["harmful"] for row in root_rows)
    catastrophic = sum(row["catastrophic"] for row in root_rows)
    changed = sum(row["changed"] for row in root_rows)
    ci = _bootstrap_mean(deltas)
    ratio = math.inf if harmful == 0 and beneficial > 0 else beneficial / max(1, harmful)
    metrics = {
        "roots": len(root_rows),
        "seed_agreement_roots": len(agreed),
        "seed_agreement_fraction": len(agreed) / len(root_rows),
        "changed_roots": changed,
        "beneficial_roots": beneficial,
        "harmful_roots": harmful,
        "catastrophic_roots": catastrophic,
        "beneficial_to_harmful_ratio": ratio,
        "mean_teacher_delta": math.fsum(deltas) / len(deltas),
        "mean_teacher_delta_ci95": list(ci),
    }
    gate = {
        "seed_agreement": metrics["seed_agreement_fraction"] >= 0.80,
        "changed_roots": changed >= 20,
        "benefit_harm_ratio": ratio >= 1.5,
        "positive_mean_ci": ci[0] > 0.0,
        "catastrophic_safety": catastrophic <= 1,
    }
    passed = all(gate.values())
    return {
        "engine": canonical_engine,
        "execution": {
            "source_roots": 384,
            "supported_roots": len(roots),
            "mechanically_excluded_roots": 384 - len(roots),
            "requests": len(responses),
            "wall_seconds": wall_seconds,
            "continuation_iterations_executed": executed,
        },
        "metrics": metrics,
        "gate": gate,
        "decision": {
            "passed": passed,
            "next": "authorize_small_value_pilot" if passed else "do_not_train_value_model",
            "public_ladder_authorized": False,
            "model_training_authorized": passed,
        },
        "roots": root_rows,
    }


def run(protocol_path: Path, *paths: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol, *paths)
    roots, phase1 = _load_inputs(*paths)
    requests, cells, eligible_roots, common_supports = _build_requests(roots, phase1)
    responses, wall_seconds = _call_remote(requests)
    summary = _summarize(
        eligible_roots, phase1, responses, cells, common_supports, wall_seconds
    )
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_path(protocol_path),
        **summary,
    }
    result["result_sha256"] = _sha256_bytes(canonical_json(result).encode("ascii"))
    return result


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--world-bank", type=Path, required=True)
    parser.add_argument("--failure-attribution", type=Path, required=True)
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    paths = [
        args.phase1.expanduser().resolve(),
        args.world_bank.expanduser().resolve(),
        args.failure_attribution.expanduser().resolve(),
        args.tournament.expanduser().resolve(),
    ]
    protocol_path = args.protocol.expanduser().resolve()
    if args.freeze:
        if args.output is not None:
            parser.error("--output is not used with --freeze")
        _write_new(protocol_path, freeze_protocol(*paths))
        return 0
    if args.output is None:
        parser.error("--output is required unless --freeze is used")
    _write_new(args.output.expanduser().resolve(), run(protocol_path, *paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
