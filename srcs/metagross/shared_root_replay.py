#!/usr/bin/env python3
"""Evaluate weighted shared-root RM+ on immutable particle-cohort captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") not in {
                "teacher_root_capture",
                "teacher_root_evaluation",
            }:
                raise ValueError(f"record {line_number} is not a root capture")
            records.append(record)
    if not records:
        raise ValueError("root capture file is empty")
    return records


def _schedule(record: dict) -> dict:
    schedule_id = int(record.get("behavior_schedule_id", 0))
    try:
        return next(
            schedule
            for schedule in record["schedules"]
            if int(schedule["schedule_id"]) == schedule_id
        )
    except (KeyError, StopIteration) as exc:
        raise ValueError("root capture has no behavior schedule") from exc


def _priors(record: dict, schedule: dict):
    if record["record_type"] == "teacher_root_evaluation":
        first_world = schedule["worlds"][0]
        player = first_world.get("effective_player_priors")
        opponent = first_world.get("effective_opponent_priors")
    else:
        player = record.get("recorded_player_priors")
        opponent = record.get("recorded_opponent_priors")
    return player or None, opponent or None


def _policy(result) -> dict[str, float]:
    return {entry.action: float(entry.probability) for entry in result.policy}


def _total_variation(left, right) -> float:
    left_policy = _policy(left)
    right_policy = _policy(right)
    return 0.5 * math.fsum(
        abs(left_policy.get(action, 0.0) - right_policy.get(action, 0.0))
        for action in left_policy.keys() | right_policy.keys()
    )


def _result_payload(result) -> dict:
    from srcs.metagross.mcts_contract import shared_root_result_payload

    return shared_root_result_payload(result)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_files(
    paths: list[Path],
    *,
    iterations: int,
    continuation_iterations: int,
    seed: int,
    prior_strength: float,
    tv_threshold: float,
) -> dict:
    import poke_engine
    from srcs.metagross import run_foul_play

    roots = []
    for path in paths:
        for record in _records(path):
            schedule = _schedule(record)
            worlds = schedule["worlds"]
            states = [
                poke_engine.State.from_string(world["sampled_state"])
                for world in worlds
            ]
            weights = [float(world["sample_weight"]) for world in worlds]
            total = math.fsum(weights)
            weights = [weight / total for weight in weights]
            player_prior, opponent_prior = _priors(record, schedule)
            opponent_priors = [opponent_prior for _state in states]
            arguments = {
                "states": states,
                "particle_weights": weights,
                "iterations": iterations,
                "continuation_iterations": continuation_iterations,
                "seed": seed,
                "prior_strength": prior_strength,
                "s1_prior": player_prior,
                "s2_priors": opponent_priors,
            }
            first = poke_engine.shared_information_set_root_search(**arguments)
            repeated = poke_engine.shared_information_set_root_search(**arguments)
            doubled = poke_engine.shared_information_set_root_search(
                **{**arguments, "iterations": iterations * 2}
            )
            first_payload = _result_payload(first)
            repeated_payload = _result_payload(repeated)
            doubled_payload = _result_payload(doubled)
            probabilities = [entry.probability for entry in first.policy]
            legal_actions = {str(action).lower() for action, _value in player_prior or ()}
            output_actions = {entry.action for entry in first.policy}
            canonical_mappings = {
                action: (
                    action
                    if not legal_actions
                    else run_foul_play._authorized_action_name(action, legal_actions)
                )
                for action in output_actions
            }
            illegal_positive_actions = sorted(
                entry.action
                for entry in first.policy
                if entry.probability > 1e-12
                and legal_actions
                and canonical_mappings[entry.action] is None
            )
            legal = not illegal_positive_actions
            normalized = (
                all(math.isfinite(value) and value >= 0 for value in probabilities)
                and abs(math.fsum(probabilities) - 1.0) <= 1e-8
            )
            reproducible = first_payload == repeated_payload
            total_variation = _total_variation(first, doubled)
            stable = total_variation <= tv_threshold
            selected = max(
                first.policy,
                key=lambda entry: (
                    entry.probability,
                    entry.counterfactual_value,
                    entry.action,
                ),
            )
            direct_argmax = (
                max(player_prior, key=lambda row: (float(row[1]), str(row[0])))[0].lower()
                if player_prior
                else None
            )
            collapsed_to_direct = (
                direct_argmax == selected.action and selected.probability >= 0.999
            )
            collapse_explained = (
                not collapsed_to_direct
                or (
                    first.diagnostics.nash_conv <= 0.05
                    and selected.counterfactual_value
                    >= max(entry.counterfactual_value for entry in first.policy) - 1e-8
                )
            )
            diagnostics_complete = all(
                math.isfinite(value) and value >= -1e-12
                for value in (
                    first.diagnostics.player_best_response_gain,
                    first.diagnostics.opponent_best_response_gain,
                    first.diagnostics.nash_conv,
                    first.diagnostics.exploitability,
                    first.diagnostics.total_regret_bound,
                )
            )
            passed = (
                legal
                and normalized
                and reproducible
                and stable
                and collapse_explained
                and diagnostics_complete
            )
            roots.append(
                {
                    "input_path": str(path),
                    "input_sha256": _sha256(path),
                    "identity": record["identity"],
                    "capture_sha256": record.get("capture_sha256")
                    or record.get("evaluation_sha256"),
                    "particle_count": len(states),
                    "selected_action": selected.action,
                    "direct_policy_argmax": direct_argmax,
                    "collapsed_to_direct_argmax": collapsed_to_direct,
                    "collapse_explained": collapse_explained,
                    "legal": legal,
                    "canonical_action_mappings": canonical_mappings,
                    "illegal_positive_actions": illegal_positive_actions,
                    "normalized": normalized,
                    "reproducible": reproducible,
                    "replay_capture_schema_version": first_payload["replay_capture"][
                        "schema_version"
                    ],
                    "replay_capture_sha256": _canonical_sha256(
                        first_payload["replay_capture"]
                    ),
                    "replay_capture_exactly_reproduced": first_payload[
                        "replay_capture"
                    ]
                    == repeated_payload["replay_capture"],
                    "total_variation_1x_2x": total_variation,
                    "stable": stable,
                    "diagnostics_complete": diagnostics_complete,
                    "nash_conv": first.diagnostics.nash_conv,
                    "exploitability": first.diagnostics.exploitability,
                    "total_regret_bound": first.diagnostics.total_regret_bound,
                    "policy_1x": first_payload["policy"],
                    "policy_2x": doubled_payload["policy"],
                    "provenance": {
                        key: first_payload["diagnostics"][key]
                        for key in (
                            "solver_contract",
                            "iterations",
                            "continuation_iterations",
                            "seed",
                            "prior_strength",
                            "input_particle_count",
                            "positive_particle_count",
                            "canonical_particle_count",
                            "action_support_digest",
                            "particle_digest",
                            "payoff_digest",
                            "player_prior_digest",
                            "opponent_prior_digest",
                        )
                    },
                    "passed": passed,
                }
            )
    stable_count = sum(root["stable"] for root in roots)
    required_stable = math.ceil(0.95 * len(roots)) if roots else 0
    return {
        "schema_version": 1,
        "mode": "particle_weighted_shared_rm_plus_fixed_root",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "configuration": {
            "iterations_1x": iterations,
            "iterations_2x": iterations * 2,
            "continuation_iterations": continuation_iterations,
            "seed": seed,
            "prior_strength": prior_strength,
            "total_variation_threshold": tv_threshold,
            "required_stable_fraction": 0.95,
        },
        "counts": {
            "roots": len(roots),
            "passed": sum(root["passed"] for root in roots),
            "failures": sum(not root["passed"] for root in roots),
            "stable": stable_count,
            "required_stable": required_stable,
            "reproducible": sum(root["reproducible"] for root in roots),
            "legal": sum(root["legal"] for root in roots),
            "normalized": sum(root["normalized"] for root in roots),
            "direct_argmax_collapses": sum(
                root["collapsed_to_direct_argmax"] for root in roots
            ),
            "unexplained_direct_argmax_collapses": sum(
                root["collapsed_to_direct_argmax"]
                and not root["collapse_explained"]
                for root in roots
            ),
        },
        "roots": roots,
        "gate": {
            "passed": bool(roots)
            and all(
                root["legal"]
                and root["normalized"]
                and root["reproducible"]
                and root["collapse_explained"]
                and root["diagnostics_complete"]
                for root in roots
            )
            and stable_count >= required_stable,
            "all_legal": all(root["legal"] for root in roots),
            "all_normalized": all(root["normalized"] for root in roots),
            "all_reproducible": all(root["reproducible"] for root in roots),
            "stable_fraction": stable_count / len(roots) if roots else 0.0,
            "no_unexplained_direct_argmax_collapse": all(
                root["collapse_explained"] for root in roots
            ),
            "all_diagnostics_complete": all(
                root["diagnostics_complete"] for root in roots
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--continuation-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--tv-threshold", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.iterations <= 0 or args.iterations * 2 > 1_000_000:
        raise ValueError("iterations must be in [1, 500000]")
    if args.continuation_iterations <= 0:
        raise ValueError("continuation iterations must be positive")
    if not 0 <= args.seed <= (1 << 64) - 1:
        raise ValueError("seed must fit in u64")
    if not math.isfinite(args.prior_strength) or not 0 <= args.prior_strength <= 1_000:
        raise ValueError("prior strength must be finite and in [0, 1000]")
    if not math.isfinite(args.tv_threshold) or not 0 <= args.tv_threshold <= 1:
        raise ValueError("TV threshold must be finite and in [0, 1]")
    report = verify_files(
        [path.expanduser().resolve() for path in args.input],
        iterations=args.iterations,
        continuation_iterations=args.continuation_iterations,
        seed=args.seed,
        prior_strength=args.prior_strength,
        tv_threshold=args.tv_threshold,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], sort_keys=True))
    print(output)
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
