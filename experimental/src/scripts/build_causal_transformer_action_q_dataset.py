#!/usr/bin/env python3
"""Join deep counterfactual roots to exact repaired causal R1 trajectories."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from scripts.build_mcts_v3_dataset import GroupRejected, map_move_string
from scripts.run_public_mcts_leaf_gate import _load_oracle, _load_panel


SCHEMA = "metagross-causal-transformer-action-q-dataset/v1"
NUM_ACTIONS = 13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().removeprefix("battle-"))


def rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def terminal_rewards(path: Path, groups: set[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
    by_battle: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for group in groups:
        by_battle[group[0]].add(group)
    if path.name.endswith(".lz4"):
        import lz4.frame

        opened = lz4.frame.open(path, "rt", encoding="utf-8")
    else:
        opened = path.open(encoding="utf-8")
    matched = {}
    with opened as handle:
        for line in handle:
            row = json.loads(line)
            battle = norm(row.get("battle_tag"))
            candidates = by_battle.get(battle, set())
            if not candidates:
                continue
            pov = norm(row.get("pov"))
            possible = [group for group in candidates if pov.endswith(group[1]) or group[1].endswith(pov)]
            if len(possible) != 1:
                continue
            rl2 = row.get("rl2")
            if not isinstance(rl2, list) or not rl2:
                continue
            rewards = []
            for vector in rl2:
                if not isinstance(vector, list) or len(vector) != 14:
                    raise ValueError("terminal trajectory has invalid RL2 shape")
                reward = float(vector[0])
                if not math.isfinite(reward):
                    raise ValueError("terminal trajectory reward is non-finite")
                rewards.append(reward)
            matched[possible[0]] = rewards
    return matched


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel, panel_hash = _load_panel(args.panel)
    oracle, oracle_hash = _load_oracle(args.oracle, panel_hash)
    root_by_group = {}
    for root in panel:
        group = (norm(root.get("battle_tag")), norm(root.get("username")))
        decision = root.get("decision_idx")
        if not all(group) or not isinstance(decision, int) or group in root_by_group:
            raise ValueError("panel has invalid or duplicate causal identity")
        root_by_group[group] = root
    groups = set(root_by_group)

    decisions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in args.decision_log:
        for _, row in rows(path):
            if row.get("record_type") != "decision":
                continue
            group = (norm(row.get("battle_tag")), norm(row.get("username")))
            root = root_by_group.get(group)
            decision = row.get("prior_decision_idx")
            if root is None or not isinstance(decision, int) or decision > root["decision_idx"]:
                continue
            key = (*group, decision)
            if key in decisions:
                raise ValueError("duplicate historical decision identity")
            decisions[key] = row

    snapshots: dict[tuple[str, str, int], dict[str, Any]] = {}
    for _, row in rows(args.prior_snapshot):
        group = (norm(row.get("tag")), norm(row.get("username")))
        root = root_by_group.get(group)
        decision = row.get("decision_idx")
        if root is None or not isinstance(decision, int) or decision > root["decision_idx"]:
            continue
        key = (*group, decision)
        if key in snapshots:
            raise ValueError("duplicate exact R1 snapshot identity")
        snapshots[key] = row

    rewards = terminal_rewards(args.terminal_trajectories, groups)
    records = []
    rejected: collections.Counter[str] = collections.Counter()
    for group, root in root_by_group.items():
        decision_idx = int(root["decision_idx"])
        trajectory_rewards = rewards.get(group)
        if trajectory_rewards is None or len(trajectory_rewards) <= decision_idx:
            rejected["missing_terminal_rewards"] += 1
            continue
        sequence = [snapshots.get((*group, index)) for index in range(decision_idx + 1)]
        history = [decisions.get((*group, index)) for index in range(decision_idx + 1)]
        if any(item is None for item in sequence) or any(item is None for item in history):
            rejected["noncontiguous_history"] += 1
            continue
        if any(item.get("schema") != 3 or item.get("mask_fallback") for item in sequence):
            rejected["invalid_snapshot"] += 1
            continue
        text = [item.get("text_tokens") for item in sequence]
        numbers = [item.get("numbers") for item in sequence]
        illegal = [item.get("illegal_actions") for item in sequence]
        if (
            any(not isinstance(value, list) or not value for value in text)
            or any(not isinstance(value, list) or not value for value in numbers)
            or any(not isinstance(value, list) or len(value) != NUM_ACTIONS for value in illegal)
        ):
            rejected["malformed_snapshot"] += 1
            continue
        rl2 = [[0.0] * (NUM_ACTIONS + 1) for _ in sequence]
        valid = True
        for timestep in range(1, len(sequence)):
            selected = history[timestep - 1].get("canonical_selected_action_index")
            if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < NUM_ACTIONS:
                valid = False
                break
            rl2[timestep][0] = float(trajectory_rewards[timestep])
            rl2[timestep][1 + selected] = 1.0
        if not valid:
            rejected["invalid_action_receipt"] += 1
            continue

        teacher_rows = [oracle.get(f"{root['root_id']}:{schedule['schedule_id']}") for schedule in root["schedules"]]
        if any(row is None for row in teacher_rows):
            raise ValueError("oracle is incomplete for panel")
        action_sets = [set(row["action_values"]) for row in teacher_rows]
        if action_sets[0] != action_sets[1]:
            raise ValueError("teacher support changes across schedules")
        current = sequence[-1]
        names = current.get("name_table")
        current_illegal = illegal[-1]
        if not isinstance(names, dict):
            rejected["missing_name_table"] += 1
            continue
        teacher_q = [0.0] * NUM_ACTIONS
        teacher_support = [False] * NUM_ACTIONS
        mapped_indices = set()
        for action in sorted(action_sets[0]):
            try:
                index, _ = map_move_string(action, names)
            except GroupRejected:
                valid = False
                rejected["ambiguous_teacher_action"] += 1
                break
            if index is None or index in mapped_indices or current_illegal[index]:
                valid = False
                rejected["unmapped_or_illegal_teacher_action"] += 1
                break
            mapped_indices.add(index)
            teacher_support[index] = True
            teacher_q[index] = math.fsum(float(row["action_values"][action]) for row in teacher_rows) / len(teacher_rows)
        if not valid:
            continue
        historical_index = history[-1].get("canonical_selected_action_index")
        if not isinstance(historical_index, int) or not teacher_support[historical_index]:
            rejected["historical_action_outside_teacher_support"] += 1
            continue
        r1_probs = [float(value) for value in current.get("probs", [])]
        if len(r1_probs) != NUM_ACTIONS or any(not math.isfinite(value) or value < 0 for value in r1_probs):
            rejected["invalid_r1_prior"] += 1
            continue
        r1_probs = [value if teacher_support[index] else 0.0 for index, value in enumerate(r1_probs)]
        total = math.fsum(r1_probs)
        if total <= 0:
            rejected["empty_r1_teacher_support"] += 1
            continue
        r1_probs = [value / total for value in r1_probs]
        records.append({
            "battle_id": root["battle_id"],
            "root_id": root["root_id"],
            "battle_tag": root["battle_tag"],
            "username": root["username"],
            "text_tokens": text,
            "numbers": [[float(value) for value in vector] for vector in numbers],
            "illegal_actions": illegal,
            "rl2": rl2,
            "time_indices": list(range(len(sequence))),
            "teacher_support": teacher_support,
            "teacher_q": teacher_q,
            "r1_probs": r1_probs,
            "historical_selected_index": historical_index,
            "public_reveal_fractions": root.get("public_reveal_fractions"),
        })
    if len(records) < 1000 or len({row["battle_id"] for row in records}) != len(records):
        raise ValueError(f"too few unique causal action-Q records: {len(records)}; rejected={dict(rejected)}")
    payload = {
        "schema": SCHEMA,
        "records": records,
        "provenance": {
            "panel_sha256": panel_hash,
            "oracle_sha256": oracle_hash,
            "prior_snapshot_sha256": sha256(args.prior_snapshot),
            "terminal_trajectories_sha256": sha256(args.terminal_trajectories),
            "decision_logs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in args.decision_log],
            "alignment": "reward_first_selected_action_receipts_absolute_time",
            "sampled_state_present": False,
            "rejected": dict(rejected),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "metagross-causal-transformer-action-q-dataset-report/v1",
        "records": len(records),
        "supported_actions": sum(sum(row["teacher_support"]) for row in records),
        "mean_history": math.fsum(len(row["time_indices"]) for row in records) / len(records),
        "maximum_history": max(len(row["time_indices"]) for row in records),
        "rejected": dict(rejected),
        "output_sha256": sha256(args.output),
        "provenance": payload["provenance"],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--prior-snapshot", type=Path, required=True)
    parser.add_argument("--terminal-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
