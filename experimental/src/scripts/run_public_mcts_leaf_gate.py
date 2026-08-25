#!/usr/bin/env python3
"""Run a same-MCTS public-value leaf swap against a frozen held-out root panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from eval.neural_value_root_gate import SCHEMA as RESULT_SCHEMA


PANEL_SCHEMA = "metagross-public-mcts-root-panel/v1"
ORACLE_SCHEMA = "metagross-public-mcts-root-oracle/v1"
REVEAL_SIDECAR_SCHEMA = "metagross-resource-reveal-sidecar/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_panel(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(path)
    if len(rows) < 50:
        raise ValueError("panel requires at least 50 roots")
    if any(row.get("schema") != PANEL_SCHEMA for row in rows):
        raise ValueError("invalid panel schema")
    if len({row["root_id"] for row in rows}) != len(rows):
        raise ValueError("panel root IDs are not unique")
    if len({row["battle_id"] for row in rows}) != len(rows):
        raise ValueError("panel must contain exactly one root per source battle")
    for row in rows:
        if len(row.get("public_features", [])) != 18 or len(row.get("schedules", [])) != 2:
            raise ValueError("panel feature or schedule contract changed")
        for schedule in row["schedules"]:
            if len(schedule.get("worlds", [])) != 8:
                raise ValueError("panel world-count contract changed")
            for world in schedule["worlds"]:
                if hashlib.sha256(world["state"].encode()).hexdigest() != world["state_sha256"]:
                    raise ValueError("panel world hash mismatch")
    return rows, _sha256(path)


def _seed(pair_id: str, world_index: int, purpose: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{pair_id}:{world_index}:{purpose}".encode()).digest()[:8], "big"
    )


def _load_reveal_sidecar(path: Path, panel_hash: str) -> tuple[dict[tuple[str, int, str], int], str]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != REVEAL_SIDECAR_SCHEMA or payload.get("panel_sha256") != panel_hash:
        raise ValueError("reveal sidecar does not belong to this panel")
    mapping = {}
    for row in payload.get("rows", []):
        key = (str(row["pair_id"]), int(row["world_index"]), str(row["state_sha256"]))
        bits = row.get("bits")
        if key in mapping or isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits < (1 << 42):
            raise ValueError("invalid reveal sidecar entry")
        mapping[key] = bits
    return mapping, _sha256(path)


def _oracle_task(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    schedule, iterations = task
    import poke_engine

    sums: dict[str, float] = {}
    masses: dict[str, float] = {}
    total_iterations = 0
    pair_id = schedule["pair_id"]
    for world in schedule["worlds"]:
        state = poke_engine.State.from_string(world["state"])
        if "public_reveal_bits" in world:
            state = state.with_side_one_public_reveals(int(world["public_reveal_bits"]))
        result = poke_engine.monte_carlo_tree_search(
            state,
            iterations=iterations,
            threads=1,
            seed=_seed(pair_id, int(world["world_index"]), "tree"),
        )
        weight = float(world["weight"])
        total_iterations += result.total_visits
        for option in result.side_one:
            if option.visits <= 0:
                continue
            value = float(option.total_score) / int(option.visits)
            sums[option.move_choice] = sums.get(option.move_choice, 0.0) + weight * value
            masses[option.move_choice] = masses.get(option.move_choice, 0.0) + weight
    values = {action: sums[action] / masses[action] for action in sums if masses[action] > 0}
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("oracle produced invalid values")
    action = max(values, key=lambda name: (values[name], name))
    return {
        "schema": ORACLE_SCHEMA,
        "panel_sha256": schedule["panel_sha256"],
        "battle_id": schedule["battle_id"],
        "root_id": schedule["root_id"],
        "pair_id": pair_id,
        "iterations_per_world": iterations,
        "total_iterations": total_iterations,
        "oracle_action": action,
        "oracle_best_value": values[action],
        "action_values": values,
        "reveal_sidecar_sha256": schedule.get("reveal_sidecar_sha256"),
    }


def run_oracle(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("METAGROSS_VALUE_MODEL"):
        raise ValueError("oracle must run without a learned value model")
    panel, panel_hash = _load_panel(args.panel)
    reveal_mapping = {}
    reveal_hash = None
    if args.reveal_sidecar is not None:
        reveal_mapping, reveal_hash = _load_reveal_sidecar(args.reveal_sidecar, panel_hash)
    tasks = []
    for row in panel:
        for schedule in row["schedules"]:
            pair_id = f"{row['root_id']}:{schedule['schedule_id']}"
            worlds = []
            for world in schedule["worlds"]:
                copied = dict(world)
                if reveal_hash is not None:
                    key = (pair_id, int(world["world_index"]), str(world["state_sha256"]))
                    if key not in reveal_mapping:
                        raise ValueError("reveal sidecar is incomplete for the panel")
                    copied["public_reveal_bits"] = reveal_mapping[key]
                worlds.append(copied)
            tasks.append(
                (
                    {
                        **schedule,
                        "worlds": worlds,
                        "panel_sha256": panel_hash,
                        "battle_id": row["battle_id"],
                        "root_id": row["root_id"],
                        "pair_id": pair_id,
                        "reveal_sidecar_sha256": reveal_hash,
                    },
                    args.iterations,
                )
            )
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_oracle_task, tasks))
    rows.sort(key=lambda row: row["pair_id"])
    _write_jsonl(args.output, rows)
    return {
        "mode": "oracle",
        "pairs": len(rows),
        "battles": len({row["battle_id"] for row in rows}),
        "panel_sha256": panel_hash,
        "output_sha256": _sha256(args.output),
        "iterations_per_world": args.iterations,
        "reveal_sidecar_sha256": reveal_hash,
    }


def _load_oracle(path: Path, panel_hash: str) -> tuple[dict[str, dict[str, Any]], str]:
    rows = _read_jsonl(path)
    by_pair = {}
    for row in rows:
        if row.get("schema") != ORACLE_SCHEMA or row.get("panel_sha256") != panel_hash:
            raise ValueError("oracle does not match panel")
        if row["pair_id"] in by_pair:
            raise ValueError("duplicate oracle pair")
        by_pair[row["pair_id"]] = row
    return by_pair, _sha256(path)


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    model_hash = None
    if args.arm == "candidate":
        if args.model is None or not args.model.is_file():
            raise ValueError("candidate requires a model")
        model_hash = _sha256(args.model)
        os.environ["METAGROSS_VALUE_MODEL"] = str(args.model.resolve())
        os.environ["METAGROSS_LEARNED_VALUE_WEIGHT"] = str(args.learned_weight)
    else:
        os.environ.pop("METAGROSS_VALUE_MODEL", None)
        os.environ.pop("METAGROSS_LEARNED_VALUE_WEIGHT", None)
    import poke_engine

    if args.arm == "candidate":
        warm_state = poke_engine.State.from_string(panel[0]["schedules"][0]["worlds"][0]["state"])
        if poke_engine.compute_learned_value(warm_state) is None:
            raise ValueError("candidate model did not load")
    rows = []
    for root in panel:
        for schedule in root["schedules"]:
            pair_id = f"{root['root_id']}:{schedule['schedule_id']}"
            reference = oracle.get(pair_id)
            if reference is None:
                raise ValueError("panel pair missing from oracle")
            started = time.perf_counter()
            per_world_ms = max(1, int(args.budget_ms * 0.72 / len(schedule["worlds"])))
            visits: dict[str, float] = {}
            learned = hand = terminal = total = 0
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                result = poke_engine.monte_carlo_tree_search(
                    state,
                    duration_ms=per_world_ms,
                    threads=1,
                    seed=_seed(pair_id, int(world["world_index"]), "tree"),
                )
                weight = float(world["weight"])
                denominator = max(1, result.total_visits)
                total += result.total_visits
                learned += result.learned_evaluations
                hand += result.hand_evaluations
                terminal += result.terminal_evaluations
                for option in result.side_one:
                    visits[option.move_choice] = visits.get(option.move_choice, 0.0) + (
                        weight * option.visits / denominator
                    )
            elapsed = (time.perf_counter() - started) * 1000.0
            if learned + hand + terminal != total:
                raise ValueError("leaf evaluation census does not equal MCTS iterations")
            selected = max(visits, key=lambda action: (visits[action], action))
            values = reference["action_values"]
            if selected not in values:
                raise ValueError("selected action is absent from oracle")
            rows.append(
                {
                    "schema": RESULT_SCHEMA,
                    "battle_id": root["battle_id"],
                    "root_id": root["root_id"],
                    "pair_id": pair_id,
                    "arm": args.arm,
                    "budget_ms": args.budget_ms,
                    "elapsed_ms": elapsed,
                    "selected_action": selected,
                    "oracle_action": reference["oracle_action"],
                    "oracle_best_value": reference["oracle_best_value"],
                    "selected_oracle_value": values[selected],
                    "oracle_artifact_sha256": oracle_hash,
                    "value_head_sha256": model_hash,
                    "certified_neural_leaves": learned,
                    "total_leaf_evaluations": total,
                    "hand_leaf_evaluations": hand,
                    "terminal_leaf_evaluations": terminal,
                    "learned_value_weight": args.learned_weight if args.arm == "candidate" else 0.0,
                    "panel_sha256": panel_hash,
                }
            )
    _write_jsonl(args.output, rows)
    return {
        "mode": args.arm,
        "pairs": len(rows),
        "battles": len({row["battle_id"] for row in rows}),
        "panel_sha256": panel_hash,
        "oracle_sha256": oracle_hash,
        "model_sha256": model_hash,
        "output_sha256": _sha256(args.output),
        "learned_evaluations": sum(row["certified_neural_leaves"] for row in rows),
        "total_evaluations": sum(row["total_leaf_evaluations"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    oracle = subparsers.add_parser("oracle")
    oracle.add_argument("--panel", type=Path, required=True)
    oracle.add_argument("--output", type=Path, required=True)
    oracle.add_argument("--iterations", type=int, default=187_000)
    oracle.add_argument("--workers", type=int, default=8)
    oracle.add_argument("--reveal-sidecar", type=Path)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--panel", type=Path, required=True)
    arm.add_argument("--oracle", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    arm.add_argument("--model", type=Path)
    arm.add_argument("--learned-weight", type=float, default=0.25)
    arm.add_argument("--budget-ms", type=int, default=500)
    args = parser.parse_args()
    if args.mode == "oracle":
        report = run_oracle(args)
    else:
        if args.budget_ms != 500 or not 0 < args.learned_weight <= 1:
            raise ValueError("gate is pinned to 500 ms and a positive learned weight")
        report = run_arm(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
