#!/usr/bin/env python3
"""Force root actions and collect matched terminal outcomes under MCTS continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import Any

from train.outcome_grounded import PANEL_SCHEMA, RESULT_SCHEMA, stable_u64, stable_uniform, weighted_choice


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_panel(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != PANEL_SCHEMA for row in rows):
        raise ValueError("invalid outcome-grounded panel")
    return rows


def _selected(options: list[Any]) -> str:
    if not options:
        raise RuntimeError("continuation search returned no options")
    return str(max(options, key=lambda option: (int(option.visits), str(option.move_choice))).move_choice)


def _terminal(engine: Any, state: Any) -> float | None:
    value = float(engine.terminal_value(state))
    return None if value == 0.0 else (value + 1.0) / 2.0


def _task(payload: tuple[dict[str, Any], dict[str, Any], argparse.Namespace]) -> dict[str, Any]:
    root, schedule, args = payload
    import poke_engine

    action_outcomes = {action: [] for action in root["candidate_actions"]}
    continuation_searches = 0
    for world in schedule["worlds"]:
        source = poke_engine.State.from_string(world["state"])
        root_search = poke_engine.monte_carlo_tree_search(
            source,
            duration_ms=0,
            iterations=args.root_iterations,
            threads=1,
            seed=stable_u64(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], "root") % (2**32),
        )
        opponent_policy = [(str(option.move_choice), float(option.visits)) for option in root_search.side_two]
        for rollout in range(args.rollout_start, args.rollout_start + args.rollouts):
            opponent_action = weighted_choice(
                opponent_policy,
                stable_uniform(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], rollout, "root-opponent"),
            )
            for action in root["candidate_actions"]:
                state = poke_engine.State.from_string(world["state"])
                chance = stable_uniform(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], rollout, 0, "chance")
                state = poke_engine.step_with_uniform(state, action, opponent_action, chance)[0]
                outcome = _terminal(poke_engine, state)
                decisions = 1
                while outcome is None and decisions < args.max_decisions:
                    result = poke_engine.monte_carlo_tree_search(
                        state,
                        duration_ms=0,
                        iterations=args.continuation_iterations,
                        threads=1,
                        seed=stable_u64(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], rollout, decisions, "continuation") % (2**32),
                    )
                    continuation_searches += 1
                    side_one_action = _selected(result.side_one)
                    side_two_action = _selected(result.side_two)
                    chance = stable_uniform(args.seed, root["root_id"], schedule["schedule_id"], world["world_index"], rollout, decisions, "chance")
                    state = poke_engine.step_with_uniform(state, side_one_action, side_two_action, chance)[0]
                    decisions += 1
                    outcome = _terminal(poke_engine, state)
                action_outcomes[action].append({
                    "world_index": int(world["world_index"]),
                    "rollout": rollout,
                    "outcome": outcome,
                    "decisions": decisions,
                })
    return {
        "schema": RESULT_SCHEMA,
        "battle_id": root["battle_id"],
        "root_id": root["root_id"],
        "schedule_id": int(schedule["schedule_id"]),
        "baseline_action": root["baseline_action"],
        "candidate_actions": root["candidate_actions"],
        "action_outcomes": action_outcomes,
        "configuration": {
            "root_iterations": args.root_iterations,
            "continuation_iterations": args.continuation_iterations,
            "rollouts": args.rollouts,
            "rollout_start": args.rollout_start,
            "max_decisions": args.max_decisions,
            "seed": args.seed,
            "root_opponent_policy": "20k_mcts_visit_distribution",
            "continuation_policy": "seeded_exact_mcts_argmax_both_sides",
        },
        "continuation_searches": continuation_searches,
    }


def write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_progress_row(
    row: dict[str, Any],
    expected: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str, int]:
    key = (str(row.get("root_id")), int(row.get("schedule_id", -1)))
    root = expected.get(key)
    if (
        row.get("schema") != RESULT_SCHEMA
        or root is None
        or row.get("battle_id") != root["battle_id"]
        or row.get("baseline_action") != root["baseline_action"]
        or row.get("candidate_actions") != root["candidate_actions"]
    ):
        raise ValueError(f"progress row does not belong to this panel: {key}")
    configuration = row.get("configuration", {})
    required = {
        "root_iterations": args.root_iterations,
        "continuation_iterations": args.continuation_iterations,
        "rollouts": args.rollouts,
        "rollout_start": getattr(args, "rollout_start", 0),
        "max_decisions": args.max_decisions,
        "seed": args.seed,
    }
    if any(configuration.get(name, 0 if name == "rollout_start" else None) != value for name, value in required.items()):
        raise ValueError(f"progress row configuration mismatch: {key}")
    return key


def load_progress(
    path: Path | None,
    expected: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = _validate_progress_row(row, expected, args)
        if key in seen:
            raise ValueError(f"duplicate progress row: {key}")
        seen.add(key)
    return rows


def append_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="ascii") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def collect(args: argparse.Namespace) -> dict[str, Any]:
    frozen_configurations = {
        "termination_probe_v1": (20_000, 2_048, 1, 128, 0),
        "pilot_v1": (20_000, 256, 8, 128, 0),
        "strong_confirmation_v2": (20_000, 2_048, 4, 128, 0),
        "staged_screen_4_v1": (20_000, 2_048, 4, 128, 0),
        "staged_extension_12_to_16_v1": (20_000, 2_048, 12, 128, 4),
        "strong_power_v3": (20_000, 2_048, 8, 128, 0),
        "strong_scale_horizon_v4": (20_000, 2_048, 8, 192, 0),
        "diagnostic_rollouts_16_v1": (20_000, 2_048, 16, 128, 0),
        "targeted_termination_192_v1": (20_000, 2_048, 1, 192, 0),
        "targeted_screen_4_192_v1": (20_000, 2_048, 4, 192, 0),
        "targeted_extension_12_to_16_192_v1": (20_000, 2_048, 12, 192, 4),
    }
    observed = (
        args.root_iterations,
        args.continuation_iterations,
        args.rollouts,
        args.max_decisions,
        args.rollout_start,
    )
    if observed != frozen_configurations[args.protocol]:
        raise ValueError(
            f"outcome-grounded {args.protocol} configuration differs from frozen protocol: "
            f"expected {frozen_configurations[args.protocol]}, got {observed}"
        )
    panel = read_panel(args.panel)
    import poke_engine
    if not hasattr(poke_engine.State, "with_side_one_public_reveals"):
        raise RuntimeError("outcome continuation requires the causal public-reveal engine")
    if any(action.endswith("-tera") for root in panel for action in root["candidate_actions"]):
        audited_states = [
            poke_engine.State.from_string(root["schedules"][0]["worlds"][0]["state"])
            for root in panel
        ]
        if not any(any(action.endswith("-tera") for action in poke_engine.root_options(state)[0]) for state in audited_states):
            raise RuntimeError("outcome panel requires a Gen 9 engine with Tera root options")
    expected = {
        (root["root_id"], int(schedule["schedule_id"])): root
        for root in panel
        for schedule in root["schedules"]
    }
    rows = load_progress(args.progress, expected, args)
    completed_keys = {(row["root_id"], int(row["schedule_id"])) for row in rows}
    tasks = [
        (root, schedule, args)
        for root in panel
        for schedule in root["schedules"]
        if (root["root_id"], int(schedule["schedule_id"])) not in completed_keys
    ]
    if tasks:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for row in pool.imap_unordered(_task, tasks):
                if args.progress is not None:
                    append_progress(args.progress, row)
                rows.append(row)
    if len(rows) != len(expected):
        raise RuntimeError(f"collection incomplete: {len(rows)} of {len(expected)} schedule rows")
    rows.sort(key=lambda row: (row["root_id"], row["schedule_id"]))
    write_private(args.output, rows)
    outcomes = [sample["outcome"] for row in rows for samples in row["action_outcomes"].values() for sample in samples]
    completed_samples = sum(value is not None for value in outcomes)
    report = {
        "schema": "metagross-outcome-grounded-collection-report/v1",
        "panel_sha256": sha256(args.panel),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "samples": len(outcomes),
        "terminal_samples": completed_samples,
        "terminal_rate": completed_samples / len(outcomes),
        "continuation_searches": sum(row["continuation_searches"] for row in rows),
        "configuration": rows[0]["configuration"],
        "protocol": args.protocol,
        "resumed_rows": len(completed_keys),
        "progress_sha256": sha256(args.progress) if args.progress is not None else None,
        "engine_binary_sha256": sha256(Path(poke_engine.poke_engine.__file__)),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root-iterations", type=int, default=20_000)
    parser.add_argument("--continuation-iterations", type=int, default=256)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--rollout-start", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--protocol",
        choices=(
            "termination_probe_v1",
            "pilot_v1",
            "strong_confirmation_v2",
            "staged_screen_4_v1",
            "staged_extension_12_to_16_v1",
            "strong_power_v3",
            "strong_scale_horizon_v4",
            "diagnostic_rollouts_16_v1",
            "targeted_termination_192_v1",
            "targeted_screen_4_192_v1",
            "targeted_extension_12_to_16_192_v1",
        ),
        default="pilot_v1",
    )
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args), sort_keys=True))


if __name__ == "__main__":
    main()
