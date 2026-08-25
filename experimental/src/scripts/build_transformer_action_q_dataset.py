#!/usr/bin/env python3
"""Join frozen deep action-Q roots to exact schema-v3 R1 observations."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from scripts.build_mcts_v3_dataset import GroupRejected, map_move_string, normalize_tag


SCHEMA = "metagross-transformer-action-q-dataset/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel = [row for _, row in read_jsonl(args.panel)]
    oracle_by_root: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for _, row in read_jsonl(args.oracle):
        oracle_by_root[str(row["root_id"])].append(row)
    source_paths = {sha256(path): path for path in args.decision_log}
    if len(source_paths) != len(args.decision_log):
        raise ValueError("decision log arguments contain duplicate file content")
    needed_lines: dict[str, set[int]] = collections.defaultdict(set)
    for row in panel:
        source_hash = str(row["source_file_sha256"])
        if source_hash not in source_paths:
            raise ValueError("panel references an undeclared decision log")
        needed_lines[source_hash].add(int(row["source_line"]))
    source_rows = {}
    for source_hash, line_numbers in needed_lines.items():
        for line_number, row in read_jsonl(source_paths[source_hash]):
            if line_number in line_numbers:
                source_rows[(source_hash, line_number)] = row
    identities = {}
    panel_by_identity = {}
    for root in panel:
        source_key = (str(root["source_file_sha256"]), int(root["source_line"]))
        source = source_rows.get(source_key)
        if source is None:
            raise ValueError("panel source line is missing")
        identity = (
            normalize_tag(source.get("battle_tag")),
            str(source.get("username")),
            source.get("prior_decision_idx"),
        )
        if not identity[0] or not isinstance(identity[2], int) or identity in panel_by_identity:
            raise ValueError("invalid or duplicate panel R1 identity")
        identities[root["root_id"]] = identity
        panel_by_identity[identity] = root
    snapshots = {}
    for _, row in read_jsonl(args.prior_snapshot):
        identity = (
            normalize_tag(row.get("tag")),
            str(row.get("username")),
            row.get("decision_idx"),
        )
        if identity not in panel_by_identity:
            continue
        if identity in snapshots:
            raise ValueError("duplicate exact R1 snapshot for a panel root")
        snapshots[identity] = row
    if len(snapshots) != len(panel):
        raise ValueError(f"joined {len(snapshots)} of {len(panel)} exact R1 snapshots")

    records = []
    rejected: collections.Counter[str] = collections.Counter()
    for root in panel:
        identity = identities[root["root_id"]]
        snapshot = snapshots[identity]
        if snapshot.get("schema") != 3 or snapshot.get("mask_fallback"):
            rejected["invalid_or_fallback_snapshot"] += 1
            continue
        text, numbers, illegal, names = (
            snapshot.get("text_tokens"), snapshot.get("numbers"),
            snapshot.get("illegal_actions"), snapshot.get("name_table"),
        )
        if (
            not isinstance(text, list) or not text
            or not isinstance(numbers, list) or not numbers
            or not isinstance(illegal, list) or len(illegal) != 13
            or not all(isinstance(flag, bool) for flag in illegal)
            or not isinstance(names, dict) or not names
        ):
            rejected["malformed_snapshot"] += 1
            continue
        schedules = oracle_by_root.get(root["root_id"], [])
        if len(schedules) != 2:
            raise ValueError("root does not have exactly two teacher schedules")
        action_sets = [set(row["action_values"]) for row in schedules]
        if action_sets[0] != action_sets[1]:
            raise ValueError("teacher support changes across schedules")
        values = [0.0] * 13
        support = [False] * 13
        mapping: dict[str, int] = {}
        failed = False
        for action in sorted(action_sets[0]):
            try:
                index, _ = map_move_string(action, names)
            except GroupRejected:
                rejected["ambiguous_action"] += 1
                failed = True
                break
            if index is None:
                rejected["unmapped_action"] += 1
                failed = True
                break
            if index in mapping.values() or illegal[index]:
                rejected["colliding_or_illegal_action"] += 1
                failed = True
                break
            mapping[action] = index
            support[index] = True
            values[index] = sum(float(row["action_values"][action]) for row in schedules) / 2.0
        if failed:
            continue
        source = source_rows[(str(root["source_file_sha256"]), int(root["source_line"]))]
        historical_action = str(source.get("selected_action", ""))
        historical_index = mapping.get(historical_action)
        if historical_index is None:
            try:
                historical_index, _ = map_move_string(historical_action, names)
            except GroupRejected:
                historical_index = None
        if historical_index is not None and not support[historical_index]:
            historical_index = None
        records.append({
            "battle_id": root["battle_id"],
            "root_id": root["root_id"],
            "text_tokens": text,
            "numbers": [float(value) for value in numbers],
            "illegal_actions": illegal,
            "teacher_support": support,
            "teacher_q": values,
            "historical_selected_index": historical_index,
            "source_identity_sha256": hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    if len(records) < 950 or len({row["battle_id"] for row in records}) != len(records):
        raise ValueError(f"too few unique transformer-Q records: {len(records)}")
    payload = {
        "schema": SCHEMA,
        "records": records,
        "provenance": {
            "panel_sha256": sha256(args.panel),
            "oracle_sha256": sha256(args.oracle),
            "prior_snapshot_sha256": sha256(args.prior_snapshot),
            "decision_logs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.decision_log
            ],
            "rejected": dict(rejected),
            "requested_roots": len(panel),
            "admitted_roots": len(records),
            "sampled_state_present": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "metagross-transformer-action-q-dataset-report/v1",
        "records": len(records),
        "supported_actions": sum(sum(row["teacher_support"]) for row in records),
        "historical_action_coverage": sum(row["historical_selected_index"] is not None for row in records) / len(records),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
