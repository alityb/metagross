#!/usr/bin/env python3
"""Audit fresh mechanical decisions against exact schema-6 causal R1 snapshots."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from scripts.build_causal_action_q_panel import schema6_history_valid


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().removeprefix("battle-"))


def rows(paths: list[Path]):
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                yield path, line_number, json.loads(line)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    decisions: dict[tuple[str, str], dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    results: dict[tuple[str, str], int] = {}
    duplicate_decisions = 0
    for path, line_number, row in rows(args.decision_log):
        if row.get("schema") == "metagross-causal-dual-r1-root/v1":
            identity = row.get("identity") or {}
            group = (norm(identity.get("battle_tag")), norm(identity.get("username")))
            index = identity.get("decision_idx")
            if not all(group) or not isinstance(index, int) or not isinstance(row.get("state"), str):
                raise ValueError(f"invalid causal root at {path}:{line_number}")
            if index in decisions[group]:
                duplicate_decisions += 1
            else:
                decisions[group][index] = row
            continue
        group = (norm(row.get("battle_tag")), norm(row.get("username")))
        if not all(group):
            continue
        if row.get("record_type") == "battle_result":
            label = row.get("label")
            if label not in (0, 1) or group in results:
                raise ValueError(f"invalid or duplicate battle result at {path}:{line_number}")
            results[group] = int(label)
        elif row.get("record_type") == "decision":
            index = row.get("prior_decision_idx")
            if not isinstance(index, int) or not isinstance(row.get("state"), str):
                raise ValueError(f"invalid mechanical decision at {path}:{line_number}")
            if index in decisions[group]:
                duplicate_decisions += 1
            else:
                decisions[group][index] = row

    h2h_result = getattr(args, "h2h_result", None)
    if h2h_result is not None:
        payload = json.loads(h2h_result.read_text())
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError("H2H result must contain a games list")
        winners: dict[str, str] = {}
        for game in games:
            battle = norm(game.get("battle_tag"))
            winner = norm(game.get("winner_username"))
            if not battle or not winner or battle in winners:
                raise ValueError("H2H result contains an invalid or duplicate battle")
            winners[battle] = winner
        for group in decisions:
            battle, username = group
            if battle in winners:
                results[group] = int(username == winners[battle])

    snapshots: dict[tuple[str, str], dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    duplicate_snapshots = 0
    invalid_snapshots = 0
    for path, line_number, row in rows(args.prior_snapshot):
        group = (norm(row.get("tag")), norm(row.get("username")))
        index = row.get("decision_idx")
        if not all(group) or not isinstance(index, int):
            raise ValueError(f"invalid snapshot identity at {path}:{line_number}")
        if index in snapshots[group]:
            duplicate_snapshots += 1
            continue
        if not schema6_history_valid(row):
            invalid_snapshots += 1
            continue
        snapshots[group][index] = row

    groups = sorted(set(decisions) | set(results))
    failures: collections.Counter[str] = collections.Counter()
    complete = []
    for group in groups:
        indices = sorted(decisions.get(group, {}))
        if not indices:
            failures["no_mechanical_decisions"] += 1
            continue
        if indices != list(range(indices[-1] + 1)):
            failures["noncontiguous_decisions"] += 1
            continue
        if group not in results:
            failures["missing_terminal_result"] += 1
            continue
        missing = [index for index in indices if index not in snapshots.get(group, {})]
        if missing:
            failures["missing_schema6_snapshots"] += 1
            continue
        if any(
            snapshots[group][index]["decision_idx"] != index
            or snapshots[group][index]["trajectory"]["time_indices"][-1] != index
            for index in indices
        ):
            failures["trajectory_index_mismatch"] += 1
            continue
        complete.append(group)
    capture_rate = len(complete) / len(groups) if groups else 0.0
    admitted = (
        len(complete) >= args.minimum_battles
        and capture_rate >= args.minimum_capture_rate
        and duplicate_decisions == 0
        and duplicate_snapshots == 0
        and invalid_snapshots == 0
    )
    report = {
        "schema": "metagross-schema6-capture-audit/v1",
        "groups": len(groups),
        "complete_groups": len(complete),
        "capture_rate": capture_rate,
        "mechanical_decisions": sum(len(value) for value in decisions.values()),
        "valid_snapshots": sum(len(value) for value in snapshots.values()),
        "terminal_results": len(results),
        "duplicate_decisions": duplicate_decisions,
        "duplicate_snapshots": duplicate_snapshots,
        "invalid_snapshots": invalid_snapshots,
        "failures": dict(failures),
        "minimum_battles": args.minimum_battles,
        "minimum_capture_rate": args.minimum_capture_rate,
        "admitted": admitted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    try:
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
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--prior-snapshot", type=Path, action="append", required=True)
    parser.add_argument(
        "--h2h-result",
        type=Path,
        default=None,
        help="Optional eval result.json used to label causal-dual-root captures.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-battles", type=int, default=500)
    parser.add_argument("--minimum-capture-rate", type=float, default=0.95)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
