#!/usr/bin/env python3
"""Benchmark independent-MCTS ensemble batches on the production HTTP worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

from srcs.metagross.aws_http_mcts import MctsService
from srcs.metagross.mcts_contract import REQUEST_SCHEMA


DURATION_MS = 500
TRIALS = 20
REPEAT_COUNT = 3
SCENARIOS = (
    {"name": "later_8x500", "world_count": 8, "duration_ms": 500, "repeat_count": 3},
    {"name": "early_32x250", "world_count": 32, "duration_ms": 250, "repeat_count": 2},
)
P95_LIMIT_MS = 1554.293
MAX_LIMIT_MS = 10_000.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _requests(
    states: list[str], repeat_count: int, duration_ms: int = DURATION_MS
) -> list[dict]:
    return [
        {
            "schema": REQUEST_SCHEMA,
            "operation": "search",
            "request_id": f"latency-{repeat}-{world}",
            "index": repeat * len(states) + world,
            "state": state,
            "duration_ms": duration_ms,
            "threads": 1,
            "s1_priors": None,
            "s2_priors": None,
            "c_puct": 2.0,
        }
        for repeat in range(repeat_count)
        for world, state in enumerate(states)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_protocol(protocol_path: Path, evaluation_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_scenarios = [
        {**scenario, "requests": scenario["world_count"] * scenario["repeat_count"]}
        for scenario in SCENARIOS
    ]
    identity = protocol.get("source_identity", {})
    if (
        protocol.get("status") != "frozen_before_execution"
        or protocol.get("configuration")
        != {
            "trials_per_arm": TRIALS,
            "worker_processes": 16,
            "alternating_arm_order": True,
            "maximum_repeat_count": REPEAT_COUNT,
            "scenarios": expected_scenarios,
        }
        or identity.get("independent_ensemble_latency.py")
        != _sha256(Path(__file__).resolve())
        or identity.get("run_foul_play.py")
        != _sha256(Path(__file__).with_name("run_foul_play.py").resolve())
        or identity.get("aws_http_mcts.py")
        != _sha256(Path(__file__).with_name("aws_http_mcts.py").resolve())
        or protocol.get("input", {}).get("sha256") != _sha256(evaluation_path)
    ):
        raise ValueError("adaptive ensemble latency differs from its frozen protocol")


def _run(service: MctsService, requests: list[dict]) -> tuple[float, float, list[dict]]:
    body = json.dumps(requests, separators=(",", ":")).encode("utf-8")
    started = time.perf_counter_ns()
    responses = service.search(body)
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000
    if len(responses) != len(requests) or any(response.get("ok") is not True for response in responses):
        errors = [response.get("error") for response in responses if response.get("ok") is not True]
        raise RuntimeError(f"latency benchmark worker failed: {errors}")
    timings = [dict(response["timing"]) for response in responses]
    required_timings = {
        "queue_ms",
        "validation_ms",
        "search_ms",
        "worker_ms",
        "batch_size",
        "batch_ms",
    }
    if any(set(timing) != required_timings for timing in timings):
        raise RuntimeError("latency benchmark worker returned incomplete timings")
    return wall_ms, float(responses[0]["timing"]["batch_ms"]), timings


def benchmark(evaluation_path: Path, protocol_path: Path) -> dict[str, object]:
    _validate_protocol(protocol_path, evaluation_path)
    records = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    worlds = next(
        schedule["worlds"]
        for record in records
        for schedule in record["schedules"]
        if len(schedule["worlds"]) >= max(row["world_count"] for row in SCENARIOS)
    )
    selected_states = [
        world["sampled_state"]
        for world in worlds[: max(row["world_count"] for row in SCENARIOS)]
    ]
    input_sha256 = hashlib.sha256(
        json.dumps(selected_states, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    service = MctsService("x" * 32, "local-benchmark")
    cohorts = {}
    raw_trials = {}
    try:
        _run(service, _requests([worlds[0]["sampled_state"]], 1))
        for scenario in SCENARIOS:
            cohort_size = scenario["world_count"]
            duration_ms = scenario["duration_ms"]
            candidate_repeats = scenario["repeat_count"]
            states = [world["sampled_state"] for world in worlds[:cohort_size]]
            modes = {"single": 1, "ensemble": candidate_repeats}
            rows = {mode: [] for mode in modes}
            for trial in range(TRIALS):
                order = list(modes) if trial % 2 == 0 else list(reversed(modes))
                for mode in order:
                    wall_ms, batch_ms, timings = _run(
                        service, _requests(states, modes[mode], duration_ms)
                    )
                    rows[mode].append(
                        {
                            "trial": trial,
                            "order": order,
                            "wall_ms": wall_ms,
                            "batch_ms": batch_ms,
                            "request_timings": timings,
                        }
                    )
            key = scenario["name"]
            raw_trials[key] = rows
            cohorts[key] = {
                "world_count": cohort_size,
                "duration_ms": duration_ms,
                "effective_repeat_count": candidate_repeats,
                **{
                    mode: {
                        "requests": cohort_size * modes[mode],
                        "trials": TRIALS,
                        "wall_ms_p50": _percentile(
                            [row["wall_ms"] for row in values], 0.50
                        ),
                        "wall_ms_p95": _percentile(
                            [row["wall_ms"] for row in values], 0.95
                        ),
                        "wall_ms_max": max(row["wall_ms"] for row in values),
                        "batch_ms_p95": _percentile(
                            [row["batch_ms"] for row in values], 0.95
                        ),
                    }
                    for mode, values in rows.items()
                },
            }
    finally:
        service.pool.shutdown(wait=True, cancel_futures=True)
    ensemble_p95 = max(row["ensemble"]["wall_ms_p95"] for row in cohorts.values())
    ensemble_max = max(row["ensemble"]["wall_ms_max"] for row in cohorts.values())
    complete_trials = all(
        len(raw_trials[scenario["name"]][mode]) == TRIALS
        for scenario in SCENARIOS
        for mode in ("single", "ensemble")
    )
    complete_timings = all(
        len(row["request_timings"]) == cohort_size * repeat_count
        for scenario in SCENARIOS
        for cohort_size in (scenario["world_count"],)
        for mode, repeat_count in (
            ("single", 1),
            ("ensemble", scenario["repeat_count"]),
        )
        for row in raw_trials[scenario["name"]][mode]
    )
    conditions = {
        "ensemble_p95_within_limit": ensemble_p95 <= P95_LIMIT_MS,
        "ensemble_max_within_limit": ensemble_max <= MAX_LIMIT_MS,
        "complete_trial_count": complete_trials,
        "complete_request_timings": complete_timings,
    }
    passed = all(conditions.values())
    return {
        "schema_version": 1,
        "mode": "independent_mcts_ensemble_local_worker_latency",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "configuration": {
            "trials": TRIALS,
            "maximum_repeat_count": REPEAT_COUNT,
            "scenarios": list(SCENARIOS),
            "worker_processes": 16,
            "p95_limit_ms": P95_LIMIT_MS,
            "max_limit_ms": MAX_LIMIT_MS,
        },
        "identity": {
            "python": sys.version,
            "platform": platform.platform(),
            "worker": service.identity,
            "input_states_sha256": input_sha256,
            "evaluation_sha256": _sha256(evaluation_path),
            "benchmark_source_sha256": _sha256(Path(__file__).resolve()),
            "worker_source_sha256": _sha256(
                Path(__file__).with_name("aws_http_mcts.py").resolve()
            ),
            "runtime_source_sha256": _sha256(
                Path(__file__).with_name("run_foul_play.py").resolve()
            ),
        },
        "cohorts": cohorts,
        "raw_trials": raw_trials,
        "gate": {
            "passed": passed,
            "conditions": conditions,
            "ensemble_p95_ms": ensemble_p95,
            "ensemble_max_ms": ensemble_max,
            "local_four_game_execution_smoke_authorized": passed,
            "public_ladder_authorized": False,
        },
        "limitations": [
            "This measures the local 16-process HTTP worker without network transport.",
            "The benchmark uses fixed captured states and does not measure full battle-decision latency.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = benchmark(
        args.evaluation.expanduser().resolve(), args.protocol.expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
