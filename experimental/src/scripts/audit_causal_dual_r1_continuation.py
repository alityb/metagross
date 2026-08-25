#!/usr/bin/env python3
"""Certify root parity and terminating sequential coverage for causal dual R1."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scripts.audit_dual_r1_policy_snapshots import _public_prefix, normalize_username
from scripts.r1_dual_tracker_parity_probe import _canonical_action, counter_tape_uniform
from scripts.r1_public_events import (
    PublicEventProjectionError,
    project_information_set_observations,
)
from train.causal_dual_r1 import CausalR1PolicyState, SCHEMA


REPORT_SCHEMA = "metagross-causal-dual-r1-continuation-certificate/v1"
ROOT_TOLERANCE = 1e-7
TERMINAL_COVERAGE_GATE = 0.95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("causal dual-R1 capture is empty")
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise ValueError("invalid causal dual-R1 capture schema")
        state = row.get("state")
        claimed = row.get("capture_sha256")
        unhashed = dict(row)
        unhashed.pop("capture_sha256", None)
        if (
            not isinstance(state, str)
            or hashlib.sha256(state.encode()).hexdigest() != row.get("state_sha256")
            or hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
            != claimed
            or row.get("r1_policy_snapshot", {}).get("schema") != 6
        ):
            raise ValueError("causal dual-R1 capture hash or payload is invalid")
    return rows


def join_captures(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], int]:
    grouped: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = (
        collections.defaultdict(list)
    )
    for source, rows in enumerate((first, second)):
        for row in rows:
            snapshot = row["r1_policy_snapshot"]
            key = (
                snapshot["tag"],
                snapshot["battle_turn"],
                tuple(_public_prefix(snapshot)),
            )
            grouped[key].append((source, row))
    pairs = []
    unmatched = 0
    for rows in grouped.values():
        if len(rows) != 2 or {source for source, _ in rows} != {0, 1}:
            unmatched += len(rows)
            continue
        left, right = rows[0][1], rows[1][1]
        a, b = left["r1_policy_snapshot"], right["r1_policy_snapshot"]
        if {a.get("player_role"), b.get("player_role")} != {"p1", "p2"}:
            raise ValueError("joined causal snapshots do not have opposite roles")
        if (
            normalize_username(str(a.get("opponent_username", "")))
            != normalize_username(str(b.get("username", "")))
            or normalize_username(str(b.get("opponent_username", "")))
            != normalize_username(str(a.get("username", "")))
        ):
            raise ValueError("joined causal snapshots are not reciprocal")
        pairs.append((left, right) if a["player_role"] == "p1" else (right, left))
    if not pairs:
        raise ValueError("no causal dual-R1 roots joined")
    pairs.sort(
        key=lambda pair: (
            pair[0]["r1_policy_snapshot"]["tag"],
            pair[0]["r1_policy_snapshot"]["battle_turn"],
            pair[0]["r1_policy_snapshot"]["decision_idx"],
        )
    )
    return pairs, unmatched


def fuse_actual_state(pair: tuple[Mapping[str, Any], Mapping[str, Any]], engine: Any):
    first = engine.State.from_string(pair[0]["state"])
    second = engine.State.from_string(pair[1]["state"])
    globals_ = {
        field: getattr(first, field)
        for field in (
            "weather", "weather_turns_remaining", "terrain",
            "terrain_turns_remaining", "trick_room",
            "trick_room_turns_remaining", "team_preview",
        )
    }
    if any(getattr(second, field) != value for field, value in globals_.items()):
        raise ValueError("opposite-POV roots disagree on public mechanical state")
    return engine.State(
        side_one=first.side_one,
        side_two=second.side_one,
        s1_threat=0.0,
        s2_threat=0.0,
        scout_value=0.0,
        threat_matrix=[0.0] * 36,
        wincon_matrix=[0.0] * 36,
        **globals_,
    )


def _inverse_cdf(items: Sequence[tuple[str, float]], u: float) -> str:
    total = math.fsum(weight for _, weight in items)
    threshold = u * total
    cumulative = 0.0
    for item, weight in items:
        cumulative += weight
        if threshold < cumulative:
            return item
    return items[-1][0]


def _terminal(engine: Any, state: Any) -> bool:
    return float(engine.terminal_value(state)) != 0.0


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine
    import metamon.rl.pretrained as pretrained

    first, second = _load(args.first), _load(args.second)
    pairs, unmatched = join_captures(first, second)
    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(args.checkpoint_root),
        model_name="randbats_exit_r1",
        default_checkpoint=5,
    )
    experiment = model.initialize_agent(checkpoint=5, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    failures: collections.Counter[str] = collections.Counter()
    survived_depth: collections.Counter[int] = collections.Counter()
    terminal = 0
    root_parity = 0
    total = len(pairs) * args.rollouts
    for ordinal, pair in enumerate(pairs):
        try:
            root_state = fuse_actual_state(pair, poke_engine)
            root_policies = [
                CausalR1PolicyState.from_snapshot(
                    row["r1_policy_snapshot"], model.observation_space
                )
                for row in pair
            ]
            root_probabilities = [policy.probabilities(agent, device) for policy in root_policies]
            if any(
                max(abs(actual - expected) for actual, expected in zip(probabilities, row["r1_policy_snapshot"]["probs"]))
                > ROOT_TOLERANCE
                for probabilities, row in zip(root_probabilities, pair)
            ):
                raise ValueError("root causal policy parity mismatch")
            root_parity += 1
        except Exception:
            failures["root_reconstruction_or_policy_parity"] += args.rollouts
            continue
        for rollout in range(args.rollouts):
            state = poke_engine.State.from_string(root_state.to_string())
            policies = [policy.fork() for policy in root_policies]
            probabilities = [list(values) for values in root_probabilities]
            for depth in range(args.max_decisions):
                if _terminal(poke_engine, state):
                    terminal += 1
                    survived_depth[depth] += 1
                    break
                try:
                    supports = [
                        policy.action_support(values)
                        for policy, values in zip(policies, probabilities)
                    ]
                    actions = (
                        _inverse_cdf(
                            supports[0],
                            counter_tape_uniform(args.seed + depth, ordinal, rollout, "p1"),
                        ),
                        _inverse_cdf(
                            supports[1],
                            counter_tape_uniform(args.seed + depth, ordinal, rollout, "p2"),
                        ),
                    )
                    options = poke_engine.root_options(state)
                    mappings = [
                        {_canonical_action(value): str(value) for value in side}
                        for side in options
                    ]
                    global_actions = (
                        mappings[0][actions[0]], mappings[1][actions[1]]
                    )
                except Exception:
                    failures["policy_or_action_mapping"] += 1
                    survived_depth[depth] += 1
                    break
                chance = counter_tape_uniform(
                    args.seed + depth, ordinal, rollout, "chance"
                )
                projections = []
                projection_failures = []
                for side, policy in zip(("SideOne", "SideTwo"), policies):
                    try:
                        projections.append(
                            project_information_set_observations(
                                poke_engine,
                                [state],
                                policy.tracker,
                                *global_actions,
                                chance,
                                observer_side=side,
                                public_opponent=policy.tracker.public_opponent_registry(),
                            )
                        )
                        projection_failures.append(None)
                    except PublicEventProjectionError as exc:
                        projections.append(None)
                        projection_failures.append(exc.code.lower())
                    except Exception:
                        projections.append(None)
                        projection_failures.append("engine_or_binding_error")
                if any(projection_failures):
                    failed_codes = "+".join(
                        sorted({code for code in projection_failures if code})
                    )
                    sides = (
                        "both"
                        if all(projection_failures)
                        else "p1" if projection_failures[0] else "p2"
                    )
                    failures[f"semantic_projection_{sides}:{failed_codes}"] += 1
                    survived_depth[depth] += 1
                    break
                if any(len(projection.observation_classes) != 1 for projection in projections):
                    failures["information_set_lineage"] += 1
                    survived_depth[depth] += 1
                    break
                classes = [projection.observation_classes[0] for projection in projections]
                if classes[0].next_states[0].to_string() != classes[1].next_states[0].to_string():
                    failures["opposite_pov_mechanical_mismatch"] += 1
                    survived_depth[depth] += 1
                    break
                try:
                    for policy, item, action in zip(policies, classes, global_actions):
                        policy.advance(
                            item.tracker,
                            item.observation,
                            action,
                            model.reward_function,
                        )
                    state = classes[0].next_states[0]
                    if not _terminal(poke_engine, state):
                        probabilities = [
                            None
                            if policy.current_observation.get("automatic_action")
                            else policy.probabilities(agent, device)
                            for policy in policies
                        ]
                except Exception:
                    failures["tracker_reward_or_next_policy"] += 1
                    survived_depth[depth + 1] += 1
                    break
            else:
                failures["decision_horizon"] += 1
                survived_depth[args.max_decisions] += 1
    terminal_rate = terminal / total
    admitted = root_parity == len(pairs) and terminal_rate >= TERMINAL_COVERAGE_GATE
    report = {
        "schema": REPORT_SCHEMA,
        "claim": "causal_history_dual_r1_terminating_continuation_certificate",
        "inputs": {
            "first_sha256": sha256(args.first),
            "second_sha256": sha256(args.second),
            "engine_binary_sha256": sha256(Path(poke_engine.poke_engine.__file__)),
            "checkpoint_sha256": sha256(
                args.checkpoint_root
                / "randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
            ),
        },
        "configuration": {
            "rollouts_per_root": args.rollouts,
            "max_decisions": args.max_decisions,
            "seed": args.seed,
            "policy": "corrected_causal_history_r1_epoch_5_both_sides",
            "chance_tape": "sha256_counter_v1",
            "root_policy_tolerance": ROOT_TOLERANCE,
            "terminal_coverage_gate": TERMINAL_COVERAGE_GATE,
        },
        "counts": {
            "first_capture_rows": len(first),
            "second_capture_rows": len(second),
            "joined_roots": len(pairs),
            "unmatched_boundaries": unmatched,
            "root_policy_parity_passes": root_parity,
            "rollouts": total,
            "terminal_rollouts": terminal,
        },
        "terminal_rate": terminal_rate,
        "failure_counts": dict(sorted(failures.items())),
        "survival_depth_counts": {
            str(depth): count for depth, count in sorted(survived_depth.items())
        },
        "continuation_readiness": {
            "status": "admitted" if admitted else "blocked",
            "r1_continuation_value_allowed": admitted,
            "next_panel_allowed": admitted,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    print(json.dumps(audit(args), sort_keys=True))


if __name__ == "__main__":
    main()
