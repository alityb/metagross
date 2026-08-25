#!/usr/bin/env python3
"""Validate full schema-4 replay envelopes on immutable teacher roots."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path

from srcs.metagross import run_foul_play, shared_root_replay
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    shared_root_result_payload,
)
from srcs.metagross.shared_root_capture import validate_search_row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_roots(
    input_path: Path,
    *,
    iterations: int,
    continuation_iterations: int,
    seed: int,
    prior_strength: float,
) -> dict[str, object]:
    import poke_engine

    native_path = Path(importlib.import_module("poke_engine.poke_engine").__file__).resolve()
    engine = {
        "contract": ENGINE_CONTRACT,
        "source_sha256": ENGINE_SOURCE_SHA256,
        "native_sha256": _sha256(native_path),
        "distribution_version": "0.0.47",
    }
    summaries = []
    previous_prior_state = dict(run_foul_play._PRIOR_STATE)
    try:
        for root_index, record in enumerate(shared_root_replay._records(input_path)):
            schedule = shared_root_replay._schedule(record)
            worlds = schedule["worlds"]
            state_strings = [world["sampled_state"] for world in worlds]
            source_weights = [float(world["sample_weight"]) for world in worlds]
            source_weight_sum = math.fsum(source_weights)
            normalized_weights = [weight / source_weight_sum for weight in source_weights]
            player_prior, opponent_prior = shared_root_replay._priors(record, schedule)
            native = poke_engine.shared_information_set_root_search(
                states=[poke_engine.State.from_string(state) for state in state_strings],
                particle_weights=normalized_weights,
                iterations=iterations,
                continuation_iterations=continuation_iterations,
                seed=seed,
                prior_strength=prior_strength,
                s1_prior=player_prior,
                s2_priors=[opponent_prior for _state in state_strings],
            )
            result = shared_root_result_payload(native)
            action_seed = seed ^ root_index
            remote_search = {
                "sampling_seed": seed ^ (1 << 32) ^ root_index,
                "action_seed": action_seed,
                "request_ids": [],
                "engine": engine,
            }
            run_foul_play._PRIOR_STATE.update(
                {"priors": player_prior, "opp_priors": opponent_prior}
            )
            request_actions = {row["action"] for row in result["policy"]}
            envelope = run_foul_play.build_shared_root_replay_envelope(
                states=state_strings,
                source_weights=source_weights,
                normalized_weights=normalized_weights,
                iterations=iterations,
                continuation_iterations=continuation_iterations,
                solver_seed=seed,
                action_seed=action_seed,
                result=result,
                remote_search=remote_search,
                request_actions=request_actions,
            )
            ordered = sorted(
                result["policy"],
                key=lambda row: (
                    -row["probability"],
                    -row["counterfactual_value"],
                    row["action"],
                ),
            )
            sampled_action, draw = run_foul_play._sample_shared_root_action(
                [row["action"] for row in ordered],
                {row["action"]: row["probability"] for row in ordered},
                action_seed,
            )
            row = {
                "schema": 4,
                "context": record["identity"],
                "choice": sampled_action,
                "choice_override": {
                    "sampled_action": sampled_action,
                    "mixed_strategy_draw": draw,
                },
                "player_priors": player_prior,
                "opponent_priors": opponent_prior,
                "remote_search": remote_search,
                "shared_root": result,
                "shared_root_replay": envelope,
            }
            try:
                summary = validate_search_row(row, rerun=True)
            except ValueError as exc:
                raise ValueError(
                    f"root {root_index} {record['identity']}: {exc}"
                ) from exc
            summaries.append(
                {
                    "identity": record["identity"],
                    **summary,
                    "native_capture_sha256": envelope["native_capture_sha256"],
                }
            )
    finally:
        run_foul_play._PRIOR_STATE.clear()
        run_foul_play._PRIOR_STATE.update(previous_prior_state)
    return {
        "schema_version": 1,
        "mode": "shared_root_schema4_exact_replay",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "engine": engine,
        "configuration": {
            "iterations": iterations,
            "continuation_iterations": continuation_iterations,
            "seed": seed,
            "prior_strength": prior_strength,
        },
        "counts": {
            "roots": len(summaries),
            "exact_replays": sum(summary["exact_replay"] for summary in summaries),
            "unique_envelopes": len(
                {summary["capture_sha256"] for summary in summaries}
            ),
            "unique_native_captures": len(
                {summary["native_capture_sha256"] for summary in summaries}
            ),
        },
        "passed": bool(summaries) and all(summary["exact_replay"] for summary in summaries),
        "roots": summaries,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--continuation-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = validate_roots(
        args.input.expanduser().resolve(),
        iterations=args.iterations,
        continuation_iterations=args.continuation_iterations,
        seed=args.seed,
        prior_strength=args.prior_strength,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
