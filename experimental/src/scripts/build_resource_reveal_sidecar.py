#!/usr/bin/env python3
"""Attach causal replay reveal masks to a frozen root panel without mutating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from belief.public_reveal_mask import (  # noqa: E402
    from_replay_facts,
    information_fractions,
    replay_reveal_snapshots,
)
from scripts.run_public_mcts_leaf_gate import _load_panel, _sha256  # noqa: E402


SCHEMA = "metagross-resource-reveal-sidecar/v1"


def _source_map(paths: list[Path]) -> dict[str, Path]:
    mapped = {_sha256(path): path for path in paths}
    if len(mapped) != len(paths):
        raise ValueError("decision-log hashes must be unique")
    return mapped


def _source_row(path: Path, line_number: int) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, 1):
            if index == line_number:
                return json.loads(line)
    raise ValueError(f"source line is absent: {path}:{line_number}")


def _replay(path: Path, battle_tag: str) -> dict[str, Any]:
    for replay_path in sorted((path.parent / "replays").glob("*.json")):
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        if payload.get("id") == battle_tag:
            return payload
    raise ValueError(f"replay is absent for {battle_tag}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine

    panel, panel_hash = _load_panel(args.panel)
    sources = _source_map(args.decision_log)
    rows = []
    fraction_sums = [0.0] * 4
    nonzero = 0
    for root in panel:
        source = sources.get(str(root["source_file_sha256"]))
        if source is None:
            raise ValueError("panel source decision log was not supplied")
        decision = _source_row(source, int(root["source_line"]))
        battle_tag = str(decision.get("battle_tag"))
        observer = str(decision.get("username"))
        turn = int(decision.get("turn", 0))
        payload = _replay(source, battle_tag)
        facts = replay_reveal_snapshots(payload.get("log", ""), observer).get(turn)
        if facts is None:
            raise ValueError(f"replay has no start-of-turn snapshot for {battle_tag}:{turn}")
        for schedule in root["schedules"]:
            pair_id = f"{root['root_id']}:{schedule['schedule_id']}"
            for world in schedule["worlds"]:
                state = poke_engine.State.from_string(world["state"])
                bits = from_replay_facts(state, facts)
                fractions = information_fractions(bits)
                nonzero += int(bits != 0)
                for index, value in enumerate(fractions):
                    fraction_sums[index] += value
                rows.append(
                    {
                        "pair_id": pair_id,
                        "world_index": int(world["world_index"]),
                        "state_sha256": str(world["state_sha256"]),
                        "bits": bits,
                        "fractions": list(fractions),
                    }
                )
    result = {
        "schema": SCHEMA,
        "panel_sha256": panel_hash,
        "alignment": "conservative_start_of_turn",
        "claim_limit": "forced-switch decisions within a turn may undercount; no fact is revealed early",
        "source_decision_logs": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.decision_log
        ],
        "entries": len(rows),
        "nonzero_entries": nonzero,
        "mean_fractions": [value / len(rows) for value in fraction_sums],
        "rows": rows,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["content_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
