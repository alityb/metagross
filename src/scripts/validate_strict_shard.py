#!/usr/bin/env python3
"""Validate a strict ExIt generation shard before parsing or training.

Fails closed: no manifest is written unless replay identity and MCTS visit
targets meet the protocol requirements in docs/expert_iteration_protocol.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--min-decisions", type=int, default=1)
    args = parser.parse_args()

    shard = args.shard_dir
    replays = sorted((shard / "replays").glob("*.json"))
    if not replays:
        raise SystemExit("no replay JSONs found")

    bad_usernames: list[str] = []
    replay_battles: set[str] = set()
    for replay_path in replays:
        with replay_path.open() as f:
            replay = json.load(f)
        players = replay.get("players") or []
        if any("_" in str(player) for player in players):
            bad_usernames.append(replay_path.name)
        battle_id = str(replay.get("id") or replay_path.stem)
        replay_battles.add(battle_id)

    decision_paths = [
        shard / "acceptor_decisions.jsonl",
        shard / "challenger_decisions.jsonl",
    ]
    decisions = []
    result_tags: set[str] = set()
    errors: list[str] = []
    for path in decision_paths:
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
                continue
            if row.get("record_type") == "battle_result":
                if row.get("battle_tag"):
                    result_tags.add(str(row["battle_tag"]))
                continue
            if row.get("record_type") != "decision":
                continue
            visits = row.get("mcts_visits")
            if not isinstance(visits, dict) or not visits:
                errors.append(f"{path.name}:{line_no}: missing mcts_visits")
                continue
            total = sum(float(value) for value in visits.values())
            if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
                errors.append(f"{path.name}:{line_no}: visit mass {total}")
                continue
            if not row.get("selected_action"):
                errors.append(f"{path.name}:{line_no}: missing selected_action")
                continue
            decisions.append(row)

    if bad_usernames:
        errors.append(f"{len(bad_usernames)} replays use '_' usernames")
    if len(decisions) < args.min_decisions:
        errors.append(f"only {len(decisions)} valid decisions")

    if errors:
        for error in errors[:20]:
            print(f"INVALID: {error}")
        raise SystemExit(f"strict shard rejected ({len(errors)} errors)")

    by_battle = Counter(str(row.get("battle_tag")) for row in decisions)
    manifest = {
        "schema_version": 1,
        "strict": True,
        "raw_replay_files": len(replays),
        "unique_replay_battles": len(replay_battles),
        "decision_records": len(decisions),
        "decision_battles": len(by_battle),
        "battle_result_records": len(result_tags),
        "minimum_decisions_per_battle": min(by_battle.values()),
        "maximum_decisions_per_battle": max(by_battle.values()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
