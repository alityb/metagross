#!/usr/bin/env python3
"""Run a frozen CPU tournament for action-conditioned continuation depth.

The candidate action at every root is selected without consulting the frozen
known-team teacher.  The teacher is used only after selection to score whether
the public-information intervention rescued or harmed the baseline decision.
Sampled hidden worlds are never written to the output artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from srcs.metagross.known_team_decision_v2 import canonical_json, sha256_path
from srcs.metagross.known_team_search_failure_attribution import selected_particles
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    REQUEST_SCHEMA,
    validate_holdout_result_payload,
)


PROTOCOL_SCHEMA = "metagross-value-horizon-tournament-protocol/v1"
RESULT_SCHEMA = "metagross-value-horizon-tournament/v1"
APP = "metagross-mcts-adaptive-ensemble-p16"
FUNCTION = "search_batch"
HORIZONS = (1, 2, 4, 8)
REPEATS = 2
ROLLOUTS = 64
CONTINUATION_ITERATIONS = 64
BENEFICIAL_DELTA = 0.02
HARMFUL_DELTA = -0.02
MASTER_SEED = 2026081301
MAX_IN_FLIGHT_BATCHES = 4
BATCH_SIZE = 64


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    identity = row.get("identity")
    if isinstance(identity, list) and len(identity) == 3:
        return str(identity[0]), str(identity[1]), int(identity[2])
    if isinstance(identity, dict):
        return (
            str(identity["corpus_uid"]),
            str(identity["observer"]),
            int(identity["decision_idx"]),
        )
    return str(row["corpus_uid"]), str(row["observer"]), int(row["decision_idx"])


def _seed(identity: Sequence[object], horizon: int, repeat: int, particle: int) -> int:
    # Candidate action is deliberately absent: every action receives the same
    # common-random-number tape within a root/horizon/repeat/particle cell.
    material = canonical_json(
        [MASTER_SEED, *identity, "horizon", horizon, "repeat", repeat, "particle", particle]
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _request_id(
    identity: Sequence[object], action: str, horizon: int, repeat: int, particle: int
) -> str:
    return _sha256_bytes(
        canonical_json(
            [
                RESULT_SCHEMA,
                *identity,
                action,
                horizon,
                repeat,
                particle,
            ]
        ).encode("ascii")
    )


def _protocol_payload(
    phase1_path: Path, bank_path: Path, attribution_path: Path
) -> dict[str, Any]:
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
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "remote": {
            "app": APP,
            "function": FUNCTION,
            "request_schema": REQUEST_SCHEMA,
            "engine_contract": ENGINE_CONTRACT,
            "engine_source_sha256": ENGINE_SOURCE_SHA256,
            "expected_physical_cores": 64,
        },
        "configuration": {
            "cohort": "all_27_stable_meaningful_phase1_headroom_roots",
            "particle_panel": "exact_failure_attribution_seeded_systematic_16",
            "particle_aggregation": "equal_over_16_occurrences",
            "horizons": list(HORIZONS),
            "repeats": REPEATS,
            "rollouts": ROLLOUTS,
            "continuation_iterations": CONTINUATION_ITERATIONS,
            "master_seed": MASTER_SEED,
            "opponent_priors": None,
            "action_selection": "maximum_belief_mean_paired_delta_vs_baseline_with_lexical_tie_break",
            "common_random_numbers_exclude_candidate_action": True,
            "teacher_role": "post_selection_diagnostic_only",
            "beneficial_teacher_delta": BENEFICIAL_DELTA,
            "harmful_teacher_delta": HARMFUL_DELTA,
            "promotion_gate": {
                "minimum_unresolved_rescues": 3,
                "maximum_all_cohort_harms": 0,
                "minimum_all_cohort_seed_agreement": 0.80,
                "minimum_mean_teacher_delta": 0.0,
            },
            "public_ladder_authorized": False,
            "model_training_authorized": False,
        },
    }


def freeze_protocol(
    phase1_path: Path, bank_path: Path, attribution_path: Path
) -> dict[str, Any]:
    protocol = _protocol_payload(phase1_path, bank_path, attribution_path)
    protocol["protocol_sha256"] = _sha256_bytes(canonical_json(protocol).encode("ascii"))
    return protocol


def _validate_protocol(
    protocol: Mapping[str, Any], phase1_path: Path, bank_path: Path, attribution_path: Path
) -> None:
    unhashed = dict(protocol)
    claimed = unhashed.pop("protocol_sha256", None)
    if claimed != _sha256_bytes(canonical_json(unhashed).encode("ascii")):
        raise ValueError("tournament protocol hash does not match content")
    if unhashed != _protocol_payload(phase1_path, bank_path, attribution_path):
        raise ValueError("tournament inputs, runner, or configuration differ from protocol")


def _load_inputs(
    phase1_path: Path, bank_path: Path, attribution_path: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    if not attribution.get("complete") or attribution.get("completed_roots") != 27:
        raise ValueError("failure-attribution cohort is not the complete frozen N=27 panel")
    phase1_by_identity = {_identity(root): root for root in phase1["roots"]}
    bank_by_identity = {_identity(root): root for root in bank["roots"]}
    cohort = list(attribution["roots"])
    if len({_identity(root) for root in cohort}) != 27:
        raise ValueError("failure-attribution cohort identities are not unique")
    if any(_identity(root) not in phase1_by_identity or _identity(root) not in bank_by_identity for root in cohort):
        raise ValueError("tournament input join is incomplete")
    return cohort, phase1_by_identity, bank_by_identity


def _build_requests(
    cohort: Sequence[Mapping[str, Any]],
    phase1: Mapping[tuple[str, str, int], Mapping[str, Any]],
    bank: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[tuple[str, str, int], str, int, int, int]]]:
    requests: list[dict[str, Any]] = []
    request_cells: dict[str, tuple[tuple[str, str, int], str, int, int, int]] = {}
    for attribution_root in cohort:
        identity = _identity(attribution_root)
        phase1_root = phase1[identity]
        bank_root = bank[identity]
        particles = selected_particles(bank_root)
        expected_hashes = attribution_root["particle_state_hashes"]
        actual_hashes = [particle["state_sha256"] for particle in particles]
        if actual_hashes != expected_hashes or len(particles) != 16:
            raise ValueError("tournament particle panel differs from failure attribution")
        baseline = str(phase1_root["baseline_action"])
        candidates = sorted(set(phase1_root["legal_actions"]) - {baseline})
        for horizon in HORIZONS:
            for repeat in range(REPEATS):
                for action in candidates:
                    for particle_index, particle in enumerate(particles):
                        request_id = _request_id(
                            identity, action, horizon, repeat, particle_index
                        )
                        request = {
                            "schema": REQUEST_SCHEMA,
                            "request_id": request_id,
                            "index": len(requests),
                            "operation": "paired_holdout",
                            "state": particle["state"],
                            "baseline_action": baseline,
                            "candidate_action": action,
                            "rollouts": ROLLOUTS,
                            "continuation_iterations": CONTINUATION_ITERATIONS,
                            "continuation_steps": horizon,
                            "seed": _seed(identity, horizon, repeat, particle_index),
                            "opponent_priors": None,
                        }
                        requests.append(request)
                        request_cells[request_id] = (
                            identity,
                            action,
                            horizon,
                            repeat,
                            particle_index,
                        )
    if len(request_cells) != len(requests):
        raise ValueError("tournament request IDs are not unique")
    return requests, request_cells


def _call_remote(requests: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    import modal

    function = modal.Function.from_name(APP, FUNCTION)
    batches = [list(requests[start : start + BATCH_SIZE]) for start in range(0, len(requests), BATCH_SIZE)]

    def call(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = function.remote(batch)
                if not isinstance(response, list) or len(response) != len(batch):
                    raise RuntimeError("remote tournament returned the wrong response count")
                return response
            except Exception as exc:  # pragma: no cover - cloud transport only
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("remote tournament batch failed after retries") from last_error

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT_BATCHES) as executor:
        nested = list(executor.map(call, batches))
    return [row for batch in nested for row in batch], time.perf_counter() - started


def _select_action(action_deltas: Mapping[str, float], baseline: str) -> str:
    values = {baseline: 0.0, **{str(action): float(delta) for action, delta in action_deltas.items()}}
    return sorted(values, key=lambda action: (-values[action], action))[0]


def _summarize(
    cohort: Sequence[Mapping[str, Any]],
    phase1: Mapping[tuple[str, str, int], Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    request_cells: Mapping[str, tuple[tuple[str, str, int], str, int, int, int]],
    wall_seconds: float,
) -> dict[str, Any]:
    cells: dict[tuple[tuple[str, str, int], str, int, int], list[float]] = {}
    search_ms: list[float] = []
    canonical_engine: dict[str, Any] | None = None
    executed = 0
    seen: set[str] = set()
    for response in responses:
        request_id = response.get("request_id")
        if request_id not in request_cells or request_id in seen or response.get("ok") is not True:
            raise ValueError(f"invalid or failed remote tournament response: {request_id}")
        seen.add(str(request_id))
        engine = response.get("engine")
        if not isinstance(engine, dict) or engine.get("contract") != ENGINE_CONTRACT or engine.get("source_sha256") != ENGINE_SOURCE_SHA256:
            raise ValueError("remote tournament engine identity mismatch")
        if canonical_engine is None:
            canonical_engine = engine
        elif engine != canonical_engine:
            raise ValueError("remote tournament used multiple engine identities")
        result = validate_holdout_result_payload(response.get("result"), expected_pairs=ROLLOUTS)
        identity, action, horizon, repeat, _particle = request_cells[str(request_id)]
        cells.setdefault((identity, action, horizon, repeat), []).append(
            float(result["delta_sum"]) / int(result["pairs"])
        )
        executed += int(result["continuation_iterations_executed"])
        search_ms.append(float((response.get("timing") or {})["search_ms"]))
    if seen != set(request_cells):
        raise ValueError("remote tournament response coverage is incomplete")

    roots = []
    for attribution_root in cohort:
        identity = _identity(attribution_root)
        phase1_root = phase1[identity]
        baseline = str(phase1_root["baseline_action"])
        legal = sorted(phase1_root["legal_actions"])
        horizon_rows = {}
        for horizon in HORIZONS:
            repeat_rows = []
            for repeat in range(REPEATS):
                deltas = {}
                for action in legal:
                    if action == baseline:
                        continue
                    values = cells.get((identity, action, horizon, repeat))
                    if values is None or len(values) != 16:
                        raise ValueError("tournament cell does not contain 16 particles")
                    deltas[action] = math.fsum(values) / len(values)
                selected = _select_action(deltas, baseline)
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "selected_action": selected,
                        "selected_delta": 0.0 if selected == baseline else deltas[selected],
                        "action_deltas": dict(sorted(deltas.items())),
                    }
                )
            selections = [row["selected_action"] for row in repeat_rows]
            seed_agreement = len(set(selections)) == 1
            selected_action = selections[0] if seed_agreement else None
            teacher_delta = (
                float(phase1_root["teacher_mean_q"][selected_action])
                - float(phase1_root["teacher_mean_q"][baseline])
                if selected_action is not None
                else None
            )
            horizon_rows[str(horizon)] = {
                "repeats": repeat_rows,
                "seed_agreement": seed_agreement,
                "selected_action": selected_action,
                "teacher_delta": teacher_delta,
                "beneficial": teacher_delta is not None and teacher_delta >= BENEFICIAL_DELTA,
                "harmful": teacher_delta is not None and teacher_delta <= HARMFUL_DELTA,
            }
        roots.append(
            {
                "identity": list(identity),
                "battle_id": phase1_root["battle_id"],
                "turn": int(phase1_root["turn"]),
                "prior_attribution": attribution_root["attribution"],
                "baseline_action": baseline,
                "teacher_best_action": phase1_root["teacher_best_action"],
                "baseline_headroom": float(phase1_root["baseline_headroom"]),
                "horizons": horizon_rows,
            }
        )

    summaries = {}
    unresolved_name = "unresolved_information_value_or_opponent_model"
    gate = {
        "minimum_unresolved_rescues": 3,
        "maximum_all_cohort_harms": 0,
        "minimum_all_cohort_seed_agreement": 0.80,
        "minimum_mean_teacher_delta": 0.0,
    }
    passing_horizons = []
    for horizon in HORIZONS:
        rows = [root["horizons"][str(horizon)] for root in roots]
        unresolved = [
            root["horizons"][str(horizon)]
            for root in roots
            if root["prior_attribution"] == unresolved_name
        ]
        agreed_deltas = [float(row["teacher_delta"]) for row in rows if row["teacher_delta"] is not None]
        summary = {
            "seed_agreement_roots": sum(row["seed_agreement"] for row in rows),
            "seed_agreement_fraction": sum(row["seed_agreement"] for row in rows) / len(rows),
            "beneficial_roots": sum(row["beneficial"] for row in rows),
            "harmful_roots": sum(row["harmful"] for row in rows),
            "unresolved_beneficial_roots": sum(row["beneficial"] for row in unresolved),
            "unresolved_harmful_roots": sum(row["harmful"] for row in unresolved),
            "mean_teacher_delta_on_agreed_roots": math.fsum(agreed_deltas) / len(agreed_deltas),
        }
        summary["promotion_gate_passed"] = (
            summary["unresolved_beneficial_roots"] >= gate["minimum_unresolved_rescues"]
            and summary["harmful_roots"] <= gate["maximum_all_cohort_harms"]
            and summary["seed_agreement_fraction"] >= gate["minimum_all_cohort_seed_agreement"]
            and summary["mean_teacher_delta_on_agreed_roots"] > gate["minimum_mean_teacher_delta"]
        )
        if summary["promotion_gate_passed"]:
            passing_horizons.append(horizon)
        summaries[str(horizon)] = summary
    winner = (
        sorted(
            passing_horizons,
            key=lambda horizon: (
                -summaries[str(horizon)]["unresolved_beneficial_roots"],
                -summaries[str(horizon)]["mean_teacher_delta_on_agreed_roots"],
                horizon,
            ),
        )[0]
        if passing_horizons
        else None
    )
    ordered_ms = sorted(search_ms)
    return {
        "engine": canonical_engine,
        "execution": {
            "requests": len(responses),
            "wall_seconds": wall_seconds,
            "continuation_iterations_executed": executed,
            "remote_search_ms_p50": statistics.median(search_ms),
            "remote_search_ms_p95": ordered_ms[int(0.95 * (len(ordered_ms) - 1))],
            "remote_search_ms_max": max(search_ms),
        },
        "horizon_summaries": summaries,
        "decision": {
            "passing_horizons": passing_horizons,
            "winner": winner,
            "next": "run_411_root_confirmation" if winner is not None else "do_not_train_value_model",
            "public_ladder_authorized": False,
            "model_training_authorized": False,
        },
        "roots": roots,
    }


def run(
    protocol_path: Path,
    phase1_path: Path,
    bank_path: Path,
    attribution_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol, phase1_path, bank_path, attribution_path)
    cohort, phase1, bank = _load_inputs(phase1_path, bank_path, attribution_path)
    requests, request_cells = _build_requests(cohort, phase1, bank)
    responses, wall_seconds = _call_remote(requests)
    summary = _summarize(cohort, phase1, responses, request_cells, wall_seconds)
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_path(protocol_path),
        "cohort_roots": len(cohort),
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    paths = [
        args.phase1.expanduser().resolve(),
        args.world_bank.expanduser().resolve(),
        args.failure_attribution.expanduser().resolve(),
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
