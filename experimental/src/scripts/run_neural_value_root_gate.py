#!/usr/bin/env python3
"""Run paired local 500 ms root search versus one-step transformer-value search."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eval.neural_value_root_gate import SCHEMA
from inference.transformer_value_oracle import (
    TransformerValueOracle,
    append_branch_observations,
)
from belief.public_reveal_mask import from_transformer_tracker
from scripts.evaluate_teacher_root_bundles import validate_root_evaluation
from scripts.r1_public_events import (
    R1SwitchTracker,
    _canonical_action,
    project_information_set_observations,
)
from scripts.r1_sequential_policy_coverage_probe import player_tracker_snapshot
from scripts.teacher_root_bundle import validate_root_capture


BASE_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
HEAD_SHA256 = "f1a7dce678395f0e2552285e2f2faeaef2611a50fb10a9c4ea03f12795008834"


class NeuralRootRunnerError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in map(Path, paths):
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _battle_id(identity: Mapping[str, Any]) -> str:
    material = {
        "namespace": str(identity.get("namespace", "")),
        "battle_tag": str(identity["battle_tag"]),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    item = row["identity"]
    return (
        str(item.get("namespace", "")),
        str(item["battle_tag"]),
        str(item["username"]),
        int(item["decision_idx"]),
    )


def _load_jsonl(paths: Sequence[Path], validator) -> list[dict[str, Any]]:
    rows = []
    for path in map(Path, paths):
        with path.open(encoding="ascii") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    validator(row)
                except Exception as exc:
                    raise NeuralRootRunnerError(f"invalid pinned row {path}:{line_number}") from exc
                rows.append(row)
    if not rows:
        raise NeuralRootRunnerError("no pinned rows")
    return rows


def _common_options(engine: Any, states: Sequence[Any]) -> tuple[dict[str, str], list[str]]:
    first_sets, second_sets = [], []
    first_names: dict[str, str] = {}
    second_names: dict[str, str] = {}
    for state in states:
        first, second = engine.root_options(state=state)
        canonical_first = {_canonical_action(value): value for value in first}
        canonical_second = {_canonical_action(value): value for value in second}
        if not canonical_first or not canonical_second:
            raise NeuralRootRunnerError("root has no common legal support")
        first_sets.append(set(canonical_first))
        second_sets.append(set(canonical_second))
        first_names.update(canonical_first)
        second_names.update(canonical_second)
    common_first = set.intersection(*first_sets)
    common_second = set.intersection(*second_sets)
    if not common_first or not common_second:
        raise NeuralRootRunnerError("root has no common legal support")
    return (
        {name: first_names[name] for name in sorted(common_first)},
        [second_names[name] for name in sorted(common_second)],
    )


def _matched_priors(raw: Sequence[Sequence[Any]], actions: Sequence[str]) -> list[tuple[str, float]]:
    by_action = {_canonical_action(str(action)): float(weight) for action, weight in raw}
    weights = [max(0.0, by_action.get(_canonical_action(action), 0.0)) for action in actions]
    total = math.fsum(weights)
    if total <= 0:
        weights = [1.0] * len(actions)
        total = float(len(actions))
    return [(action, weight / total) for action, weight in zip(actions, weights, strict=True)]


def _oracle_values(evaluation: Mapping[str, Any], schedule_id: int) -> dict[str, float]:
    schedule = evaluation["schedules"][schedule_id]
    total_weight = math.fsum(float(world["sample_weight"]) for world in schedule["worlds"])
    sums: dict[str, float] = {}
    masses: dict[str, float] = {}
    for world in schedule["worlds"]:
        weight = float(world["sample_weight"]) / total_weight
        result = world["treatments"]["S-B"][0]["result"]
        for option in result["side_one"]:
            visits = int(option["visits"])
            if visits <= 0:
                continue
            action = _canonical_action(option["action"])
            sums[action] = sums.get(action, 0.0) + weight * float(option["total_score"]) / visits
            masses[action] = masses.get(action, 0.0) + weight
    values = {action: sums[action] / masses[action] for action in sums if masses[action] > 0}
    if not values or any(not 0 <= value <= 1 for value in values.values()):
        raise NeuralRootRunnerError("oracle has invalid action values")
    return values


def _baseline(
    engine: Any,
    schedule: Mapping[str, Any],
    player_priors: Sequence[Sequence[Any]],
    *,
    budget_ms: int,
) -> tuple[str, float, int]:
    start = time.perf_counter()
    worlds = schedule["worlds"]
    # Reserve wall time for Python aggregation and serialization.
    per_world_ms = max(1, int((budget_ms * 0.72) / len(worlds)))
    visits_by_action: dict[str, float] = {}
    total_visits = 0
    total_weight = math.fsum(float(world["sample_weight"]) for world in worlds)
    for world in worlds:
        state = engine.State.from_string(world["sampled_state"])
        first, second = engine.root_options(state=state)
        first, second = list(first), list(second)
        result = engine.monte_carlo_tree_search(
            state,
            duration_ms=per_world_ms,
            iterations=0,
            threads=1,
            s1_priors=_matched_priors(player_priors, first),
            s2_priors=[(action, 1.0 / len(second)) for action in second],
            c_puct=2.0,
            seed=None,
        )
        weight = float(world["sample_weight"]) / total_weight
        denominator = max(1, int(result.total_visits))
        total_visits += denominator
        for option in result.side_one:
            action = _canonical_action(option.move_choice)
            visits_by_action[action] = visits_by_action.get(action, 0.0) + (
                weight * int(option.visits) / denominator
            )
    elapsed = (time.perf_counter() - start) * 1000.0
    if not visits_by_action:
        raise NeuralRootRunnerError("baseline search produced no root policy")
    selected = max(visits_by_action, key=lambda action: (visits_by_action[action], action))
    return selected, elapsed, total_visits


def _prefix(snapshot: Mapping[str, Any], device: torch.device):
    obs = {
        "text_tokens": torch.tensor([snapshot["text_tokens"]], dtype=torch.int32, device=device),
        "numbers": torch.tensor([snapshot["numbers"]], dtype=torch.float32, device=device),
        "illegal_actions": torch.tensor([snapshot["illegal_actions"]], dtype=torch.bool, device=device),
    }
    return obs, torch.zeros((1, 14), dtype=torch.float32, device=device)


def _candidate(
    engine: Any,
    capture: Mapping[str, Any],
    schedule: Mapping[str, Any],
    tracker: R1SwitchTracker,
    value_oracle: TransformerValueOracle,
    *,
    budget_ms: int,
    seed: int,
) -> tuple[str, float, int, int]:
    start = time.perf_counter()
    states = [engine.State.from_string(world["sampled_state"]) for world in schedule["worlds"]]
    # Reinstall the exact player-information root used by the transformer.
    # Archived sampled states predate the serialized reveal-mask fields, so
    # reading their completed opponents here would leak the sampled worlds.
    states = [
        state.with_side_one_public_reveals(from_transformer_tracker(state, tracker))
        for state in states
    ]
    world_weights = [float(world["sample_weight"]) for world in schedule["worlds"]]
    total_world_weight = math.fsum(world_weights)
    player, opponent = _common_options(engine, states)
    snapshot = capture["r1_policy_snapshot"]
    priors_by_index = [float(value) for value in snapshot["probs"]]
    name_table = {_canonical_action(name): int(index) for name, index in snapshot["name_table"].items()}
    player_actions = [action for action in player if action in name_table]
    if not player_actions:
        raise NeuralRootRunnerError("candidate has no model-mapped common action")
    player_weights = [max(0.0, priors_by_index[name_table[action]]) for action in player_actions]
    if math.fsum(player_weights) <= 0:
        player_weights = [1.0] * len(player_actions)
    prior_total = math.fsum(player_weights)
    prior = {action: weight / prior_total for action, weight in zip(player_actions, player_weights, strict=True)}
    rng = random.Random(seed)
    observations, action_indices, metadata = [], [], []
    terminal_values: list[tuple[str, float, float]] = []
    attempted = certified = 0
    projection_deadline = start + budget_ms * 0.45 / 1000.0
    max_observations = 16
    # Match the preregistered coverage census: eight policy-weighted,
    # opponent-uniform information-set trials per root/schedule unit.
    for _trial in range(8):
        if time.perf_counter() >= projection_deadline:
            break
        action = rng.choices(player_actions, weights=player_weights, k=1)[0]
        opponent_action = opponent[rng.randrange(len(opponent))]
        attempted += len(states)
        try:
            projection = project_information_set_observations(
                engine,
                states,
                tracker,
                player[action],
                opponent_action,
                rng.random(),
                public_opponent=tracker.public_opponent_registry(),
            )
        except Exception:
            continue
        for observation_class in projection.observation_classes:
            indices = observation_class.source_world_indices
            mass = math.fsum(world_weights[index] for index in indices) / total_world_weight
            if observation_class.observation.terminal:
                state = observation_class.tracker.state
                value = 1.0 if state.battle_won else 0.0
                terminal_values.append((action, mass, value))
                certified += len(indices)
            elif len(observations) < max_observations:
                observations.append(observation_class.observation)
                action_indices.append(name_table[action])
                metadata.append((action, mass))
                certified += len(indices)
    weighted_sum: dict[str, float] = {}
    value_mass: dict[str, float] = {}
    for action, mass, value in terminal_values:
        weighted_sum[action] = weighted_sum.get(action, 0.0) + mass * value
        value_mass[action] = value_mass.get(action, 0.0) + mass
    if observations:
        prefix_obs, prefix_rl2 = _prefix(snapshot, value_oracle.device)
        obs, rl2, time_idxs = append_branch_observations(
            prefix_obs,
            prefix_rl2,
            observations,
            action_indices,
            [0.0] * len(observations),
            device=value_oracle.device,
        )
        predictions = value_oracle.predict(obs, rl2, time_idxs).detach().cpu().tolist()
        for (action, mass), value in zip(metadata, predictions, strict=True):
            weighted_sum[action] = weighted_sum.get(action, 0.0) + mass * float(value)
            value_mass[action] = value_mass.get(action, 0.0) + mass
    scores = {}
    for action in player_actions:
        neural = weighted_sum[action] / value_mass[action] if value_mass.get(action, 0.0) > 0 else 0.5
        scores[action] = 0.8 * neural + 0.2 * prior[action]
    selected = max(scores, key=lambda action: (scores[action], action))
    elapsed = (time.perf_counter() - start) * 1000.0
    return selected, elapsed, certified, attempted


def run(
    capture_paths: Sequence[Path],
    oracle_paths: Sequence[Path],
    output: Path,
    *,
    head_path: Path,
    budget_ms: int = 500,
    max_roots: int | None = None,
) -> dict[str, Any]:
    if budget_ms != 500:
        raise NeuralRootRunnerError("promotion experiment is pinned to 500 ms")
    if _sha256(head_path) != HEAD_SHA256:
        raise NeuralRootRunnerError("value-head hash mismatch")
    captures = _load_jsonl(capture_paths, validate_root_capture)
    if max_roots is not None:
        if max_roots <= 0:
            raise NeuralRootRunnerError("max_roots must be positive")
        captures = captures[:max_roots]
    evaluations = _load_jsonl(oracle_paths, validate_root_evaluation)
    by_identity = {_identity(row): row for row in evaluations}
    if len(by_identity) != len(evaluations):
        raise NeuralRootRunnerError("duplicate oracle identity")
    oracle_hash = _artifact_hash(oracle_paths)

    import poke_engine
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(Path(__file__).resolve().parents[3] / "srcs" / "models"),
        model_name="randbats_exit_r1",
        default_checkpoint=5,
    )
    experiment = model.initialize_agent(checkpoint=5, log=False)
    agent = experiment.policy
    agent.eval()
    value_oracle = TransformerValueOracle(
        agent,
        head_path,
        expected_base_sha256=BASE_SHA256,
        expected_head_sha256=HEAD_SHA256,
    )
    # Exclude one-time PyTorch kernel initialization from every 500 ms arm.
    warm_snapshot = captures[0]["r1_policy_snapshot"]
    warm_obs, warm_rl2 = _prefix(warm_snapshot, value_oracle.device)
    warm_time = torch.zeros((1, 1, 1), dtype=torch.long, device=value_oracle.device)
    value_oracle.predict(
        {key: value.unsqueeze(0) for key, value in warm_obs.items()},
        warm_rl2.unsqueeze(0),
        warm_time,
    )
    rows = []
    for capture in captures:
        evaluation = by_identity.get(_identity(capture))
        if evaluation is None:
            raise NeuralRootRunnerError("capture has no matching oracle")
        root_id = capture["capture_sha256"]
        battle_id = _battle_id(capture["identity"])
        tracker = R1SwitchTracker.from_snapshot(
            player_tracker_snapshot(capture["r1_policy_snapshot"]),
            model.observation_space,
        )
        for schedule in capture["schedules"]:
            schedule_id = int(schedule["schedule_id"])
            oracle_values = _oracle_values(evaluation, schedule_id)
            oracle_action = max(oracle_values, key=lambda action: (oracle_values[action], action))
            pair_id = f"{root_id}:{schedule_id}"
            baseline_action, baseline_elapsed, baseline_leaves = _baseline(
                poke_engine,
                schedule,
                capture["recorded_player_priors"],
                budget_ms=budget_ms,
            )
            seed = int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:8], "big")
            candidate_action, candidate_elapsed, certified, attempted = _candidate(
                poke_engine,
                capture,
                schedule,
                tracker,
                value_oracle,
                budget_ms=budget_ms,
                seed=seed,
            )
            for arm, selected, elapsed, head, cert, leaves in (
                ("baseline", baseline_action, baseline_elapsed, None, 0, baseline_leaves),
                ("candidate", candidate_action, candidate_elapsed, HEAD_SHA256, certified, attempted),
            ):
                if selected not in oracle_values:
                    raise NeuralRootRunnerError("selected action is absent from oracle")
                rows.append(
                    {
                        "schema": SCHEMA,
                        "root_id": root_id,
                        "battle_id": battle_id,
                        "pair_id": pair_id,
                        "arm": arm,
                        "budget_ms": budget_ms,
                        "elapsed_ms": elapsed,
                        "selected_action": selected,
                        "oracle_action": oracle_action,
                        "oracle_best_value": oracle_values[oracle_action],
                        "selected_oracle_value": oracle_values[selected],
                        "oracle_artifact_sha256": oracle_hash,
                        "value_head_sha256": head,
                        "certified_neural_leaves": cert,
                        "total_leaf_evaluations": leaves,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "pairs": len(rows) // 2,
        "roots": len({row["root_id"] for row in rows}),
        "output": str(output),
        "output_sha256": _sha256(output),
        "oracle_artifact_sha256": oracle_hash,
        "head_sha256": HEAD_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--oracle", type=Path, nargs="+", required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        args.capture,
        args.oracle,
        args.output,
        head_path=args.head,
        max_roots=args.max_roots,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
