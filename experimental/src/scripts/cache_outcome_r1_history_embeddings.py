#!/usr/bin/env python3
"""Cache accepted-R1 causal history embeddings for outcome residual roots."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from scripts.train_transformer_terminal_value import initialize_agent
from train.action_semantic_residual import json_dump, sha256


SCHEMA = "metagross-outcome-r1-history-embeddings/v1"
NUM_ACTIONS = 13


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().removeprefix("battle-"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def terminal_rewards(path: Path, groups: set[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
    import lz4.frame
    by_battle: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for group in groups:
        by_battle[group[0]].add(group)
    result = {}
    opened = lz4.frame.open(path, "rt", encoding="utf-8") if path.name.endswith(".lz4") else path.open()
    with opened as handle:
        for line in handle:
            row = json.loads(line)
            candidates = by_battle.get(norm(row.get("battle_tag")), set())
            if not candidates:
                continue
            pov = norm(row.get("pov"))
            matches = [group for group in candidates if pov.endswith(group[1]) or group[1].endswith(pov)]
            if len(matches) != 1:
                continue
            rewards = row.get("rl2")
            if not isinstance(rewards, list):
                continue
            values = []
            for vector in rewards:
                if not isinstance(vector, list) or len(vector) != NUM_ACTIONS + 1:
                    raise ValueError("terminal trajectory has invalid RL2 shape")
                value = float(vector[0])
                if not math.isfinite(value):
                    raise ValueError("terminal trajectory reward is non-finite")
                values.append(value)
            result[matches[0]] = values
    return result


def build_sequences(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    promoted = {row["root_id"]: row for _, row in read_jsonl(args.panel)}
    source = {row["root_id"]: row for _, row in read_jsonl(args.source_panel) if row["root_id"] in promoted}
    if set(source) != set(promoted):
        raise ValueError(f"source-panel join incomplete: {len(source)}/{len(promoted)}")
    identity = {}
    paths = set()
    for root_id, row in source.items():
        group = (norm(row["battle_tag"]), norm(row["username"]))
        identity[group] = (root_id, int(row["decision_idx"]), row)
        paths.add(Path(row["source_path"]))
    if len(identity) != len(source):
        raise ValueError("source panel has duplicate battle/observer identities")

    decisions = {}
    for path in sorted(paths):
        for _, row in read_jsonl(path):
            if row.get("record_type") != "decision":
                continue
            group = (norm(row.get("battle_tag")), norm(row.get("username")))
            target = identity.get(group)
            decision = row.get("prior_decision_idx")
            if target is None or not isinstance(decision, int) or decision > target[1]:
                continue
            key = (*group, decision)
            if key in decisions:
                raise ValueError(f"duplicate decision history row: {key}")
            decisions[key] = row

    snapshots = {}
    for _, row in read_jsonl(args.prior_snapshot):
        group = (norm(row.get("tag")), norm(row.get("username")))
        target = identity.get(group)
        decision = row.get("decision_idx")
        if target is None or not isinstance(decision, int) or decision > target[1]:
            continue
        key = (*group, decision)
        if key in snapshots:
            raise ValueError(f"duplicate R1 snapshot row: {key}")
        snapshots[key] = row

    rewards = terminal_rewards(args.terminal_trajectories, set(identity))
    records = []
    rejected: collections.Counter[str] = collections.Counter()
    for group, (root_id, decision_idx, root) in identity.items():
        sequence = [snapshots.get((*group, index)) for index in range(decision_idx + 1)]
        history = [decisions.get((*group, index)) for index in range(decision_idx + 1)]
        trajectory_rewards = rewards.get(group)
        if any(item is None for item in sequence) or any(item is None for item in history):
            rejected["noncontiguous_history"] += 1; continue
        if trajectory_rewards is None or len(trajectory_rewards) <= decision_idx:
            rejected["missing_terminal_rewards"] += 1; continue
        if len(sequence) > 128:
            rejected["history_over_128"] += 1; continue
        if any(item.get("schema") != 3 or item.get("mask_fallback") for item in sequence):
            rejected["invalid_snapshot"] += 1; continue
        text = [item.get("text_tokens") for item in sequence]
        numbers = [item.get("numbers") for item in sequence]
        illegal = [item.get("illegal_actions") for item in sequence]
        if any(not isinstance(value, list) or not value for value in text + numbers) or any(
            not isinstance(value, list) or len(value) != NUM_ACTIONS for value in illegal
        ):
            rejected["malformed_snapshot"] += 1; continue
        rl2 = [[0.0] * (NUM_ACTIONS + 1) for _ in sequence]
        valid = True
        for timestep in range(1, len(sequence)):
            action = history[timestep - 1].get("canonical_selected_action_index")
            if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < NUM_ACTIONS:
                valid = False; break
            rl2[timestep][0] = float(trajectory_rewards[timestep])
            rl2[timestep][1 + action] = 1.0
        if not valid:
            rejected["invalid_action_receipt"] += 1; continue
        records.append({"root_id": root_id, "battle_id": root["battle_id"],
            "text_tokens": text, "numbers": numbers, "illegal_actions": illegal,
            "rl2": rl2, "time_indices": list(range(len(sequence)))})
    records.sort(key=lambda row: row["root_id"])
    if len(records) != len(promoted):
        raise ValueError(f"causal history join incomplete: {len(records)}/{len(promoted)} rejected={dict(rejected)}")
    return records, dict(rejected)


def cache(agent, records: list[dict[str, Any]], batch_size: int) -> torch.Tensor:
    device = next(agent.parameters()).device
    if device.type != "cpu":
        raise RuntimeError(f"local-only protocol expected CPU, found {device}")
    by_length: dict[int, list[int]] = collections.defaultdict(list)
    for index, record in enumerate(records):
        by_length[len(record["time_indices"])].append(index)
    embeddings = torch.empty((len(records), 900), dtype=torch.float32)
    with torch.no_grad():
        for length in sorted(by_length):
            for indices in torch.tensor(by_length[length]).split(batch_size):
                chosen = [records[int(index)] for index in indices]
                obs = {
                    "text_tokens": torch.tensor([row["text_tokens"] for row in chosen], dtype=torch.int32),
                    "numbers": torch.nan_to_num(torch.tensor([row["numbers"] for row in chosen], dtype=torch.float32)),
                    "illegal_actions": torch.tensor([row["illegal_actions"] for row in chosen], dtype=torch.bool),
                }
                rl2 = torch.tensor([row["rl2"] for row in chosen], dtype=torch.float32)
                times = torch.tensor([[[value] for value in row["time_indices"]] for row in chosen], dtype=torch.int64)
                encoded, _ = agent.get_state_embedding(obs=obs, rl2s=rl2, time_idxs=times, hidden_state=None)
                embeddings[indices] = encoded[:, -1].cpu()
    return embeddings


def atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary); temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--prior-snapshot", type=Path, required=True)
    parser.add_argument("--terminal-trajectories", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, default=Path("srcs/models"))
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.base_root / args.base_run / "ckpts/policy_weights" / f"policy_epoch_{args.base_checkpoint}.pt"
    if sha256(checkpoint) != args.base_sha256:
        raise ValueError("accepted R1 checkpoint hash mismatch")
    records, rejected = build_sequences(args)
    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    # The upstream loader prefers Apple MPS when available.  This experiment is
    # explicitly local-CPU-only, so move the frozen encoder before inference.
    agent.to("cpu")
    agent.eval()
    embeddings = cache(agent, records, args.batch_size)
    payload = {"schema": SCHEMA, "root_ids": [row["root_id"] for row in records],
        "battle_ids": [row["battle_id"] for row in records], "embeddings": embeddings.to(torch.bfloat16),
        "history_lengths": [len(row["time_indices"]) for row in records],
        "provenance": {"panel_sha256": sha256(args.panel), "source_panel_sha256": sha256(args.source_panel),
            "prior_snapshot_sha256": sha256(args.prior_snapshot),
            "terminal_trajectories_sha256": sha256(args.terminal_trajectories),
            "base_checkpoint_sha256": args.base_sha256, "sampled_state_present": False}}
    atomic_save(payload, args.output)
    report = {"schema": "metagross-outcome-r1-history-embedding-report/v1",
        "records": len(records), "embedding_width": embeddings.shape[1],
        "mean_history": sum(payload["history_lengths"]) / len(records),
        "maximum_history": max(payload["history_lengths"]), "rejected": rejected,
        "output_sha256": sha256(args.output), "provenance": payload["provenance"]}
    json_dump(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
