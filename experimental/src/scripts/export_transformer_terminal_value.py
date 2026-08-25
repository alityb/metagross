#!/usr/bin/env python3
"""Export historical Metamon POV trajectories for terminal-value training."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NUM_ACTIONS = 13


def trajectory_identity(path: Path) -> tuple[str, str, int] | None:
    name = path.name
    for suffix in (".json.lz4", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        return None
    outcome_match = re.search(r"_(WIN|LOSS)$", name)
    if outcome_match is None:
        return None
    outcome = 1 if outcome_match.group(1) == "WIN" else 0
    try:
        battle_tag, _, replay_fields = name.split("_", 2)
        pov = replay_fields.split("_vs_", 1)[0]
    except (ValueError, IndexError):
        return None
    if not battle_tag or not pov:
        return None
    return battle_tag, pov, outcome


def load_raw(path: Path) -> dict[str, Any]:
    if path.name.endswith(".lz4"):
        import lz4.frame

        with lz4.frame.open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    return json.loads(path.read_text())


def trajectory_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*.json*")
        if path.is_file() and path.name.endswith((".json", ".json.lz4"))
    )


def _legal_mask(state: Any, action_space: Any) -> list[bool]:
    from metamon.interface import UniversalAction

    illegal = [True] * NUM_ACTIONS
    for action in UniversalAction.maybe_valid_actions(state):
        mapped = action_space.action_to_agent_output(state, action)
        if isinstance(mapped, bool) or not isinstance(mapped, (int, np.integer)):
            raise ValueError("action space produced a non-discrete action")
        mapped = int(mapped)
        if not 0 <= mapped < NUM_ACTIONS:
            raise ValueError("action space produced an out-of-range action")
        illegal[mapped] = False
    if all(illegal):
        raise ValueError("nonterminal state has no maybe-valid action")
    return illegal


def export_row(
    path: Path,
    source_kind: str,
    observation_space_template: Any,
    action_space: Any,
    reward_function: Any,
) -> dict[str, Any]:
    from metamon.interface import UniversalAction, UniversalState

    identity = trajectory_identity(path)
    if identity is None:
        raise ValueError("filename does not carry a terminal outcome")
    battle_tag, pov, outcome = identity
    raw = load_raw(path)
    raw_states, raw_actions = raw.get("states"), raw.get("actions")
    if (
        not isinstance(raw_states, list)
        or len(raw_states) < 2
        or not isinstance(raw_actions, list)
        or len(raw_actions) != len(raw_states)
    ):
        raise ValueError("trajectory state/action sequence is malformed")
    states = [UniversalState.from_dict(value) for value in raw_states]
    decision_states = states[:-1]
    observation_space = copy.deepcopy(observation_space_template)
    observation_space.reset()
    observations = [observation_space.state_to_obs(state) for state in decision_states]

    text_tokens: list[list[int]] = []
    numbers: list[list[float]] = []
    illegal_actions: list[list[bool]] = []
    for state, observation in zip(decision_states, observations, strict=True):
        text_tokens.append(np.asarray(observation["text_tokens"], dtype=np.int32).tolist())
        numeric = np.nan_to_num(
            np.asarray(observation["numbers"], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).tolist()
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("observation contains non-finite numbers")
        numbers.append(numeric)
        illegal_actions.append(_legal_mask(state, action_space))

    rl2 = [[0.0] * (NUM_ACTIONS + 1) for _ in decision_states]
    for timestep in range(1, len(decision_states)):
        raw_action = raw_actions[timestep - 1]
        # Metamon's official offline wrapper maps missing replay actions (-1)
        # to action zero before building AMAGO's RL2 context.
        if isinstance(raw_action, bool) or not isinstance(raw_action, int):
            raise ValueError("trajectory action is not an integer")
        universal_action = UniversalAction(action_idx=max(0, raw_action))
        mapped = int(action_space.action_to_agent_output(decision_states[timestep - 1], universal_action))
        if not 0 <= mapped < NUM_ACTIONS:
            raise ValueError("trajectory action is out of range")
        reward = float(reward_function(states[timestep - 1], states[timestep]))
        if not math.isfinite(reward):
            raise ValueError("trajectory reward is non-finite")
        rl2[timestep][0] = reward
        rl2[timestep][1 + mapped] = 1.0

    return {
        "schema": 1,
        "battle_tag": battle_tag,
        "pov": pov,
        "source_kind": source_kind,
        "outcome": outcome,
        "text_tokens": text_tokens,
        "numbers": numbers,
        "illegal_actions": illegal_actions,
        "rl2": rl2,
    }


def parse_source(value: str) -> tuple[str, Path]:
    try:
        kind, raw_path = value.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError("source must be KIND=PATH") from None
    if kind not in {"human", "selfplay", "league"}:
        raise argparse.ArgumentTypeError("source kind must be human, selfplay, or league")
    path = Path(raw_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"source path does not exist: {path}")
    return kind, path


def initialize_model_interfaces(base_root: Path, run_name: str, checkpoint: int):
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(base_root),
        model_name=run_name,
        default_checkpoint=checkpoint,
    )
    return model.observation_space, model.action_space, model.reward_function


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, default=Path("srcs/models"))
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--limit-per-source", type=int, default=0)
    args = parser.parse_args()
    if args.limit_per_source < 0:
        parser.error("--limit-per-source cannot be negative")

    observation_space, action_space, reward_function = initialize_model_interfaces(
        args.base_root, args.base_run, args.base_checkpoint
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    ) as temporary_handle:
        temporary = Path(temporary_handle.name)
    try:
        if args.output.name.endswith(".lz4"):
            import lz4.frame

            opened = lz4.frame.open(temporary, mode="wt", encoding="utf-8")
        else:
            opened = temporary.open(mode="w", encoding="utf-8")
        with opened as handle:
            for source_kind, source_root in args.source:
                exported_for_source = 0
                for path in trajectory_paths(source_root):
                    if args.limit_per_source and exported_for_source >= args.limit_per_source:
                        break
                    identity = trajectory_identity(path)
                    if identity is None:
                        counters["rejected_filename"] += 1
                        continue
                    key = identity[:2]
                    if key in seen:
                        counters["deduplicated_pov"] += 1
                        continue
                    try:
                        row = export_row(
                            path,
                            source_kind,
                            observation_space,
                            action_space,
                            reward_function,
                        )
                    except Exception:
                        counters["rejected_trajectory"] += 1
                        continue
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    seen.add(key)
                    counters[f"exported_{source_kind}"] += 1
                    counters["exported_total"] += 1
                    exported_for_source += 1
            handle.flush()
        if not counters["exported_total"]:
            raise RuntimeError("no terminal trajectories were exported")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(sorted(counters.items())), sort_keys=True))


if __name__ == "__main__":
    main()
