#!/usr/bin/env python3
"""Training rows for the 1c action-in-context likelihood (Policy Inference).

One row per public MOVE event: the acting side's true active set (the actor
always knows their own set - this is behavior modeling, not label leakage),
minimal public context, and the move chosen. At evaluation the model is
queried per CANDIDATE set; the eval opponent's truth is never an input.

Rows come from TRAINING battles only (eval battles excluded here).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experimental.src.scripts.score_tssr_baseline import (
    MOVE_LINE, SPECIES_LINE, norm,
)

TURN_LINE = __import__("re").compile(r"\|turn\|(\d+)")


def rows_for_battle(record: dict, truth: dict):
    states = record["roles"].get("p1") or record["roles"].get("p2") or []
    active: dict[str, str | None] = {"p1": None, "p2": None}
    entered_turn: dict[str, int] = {}
    turn = 0
    for state in states:
        for line in state["prefix_delta"]:
            m = TURN_LINE.match(line)
            if m:
                turn = int(m.group(1))
                continue
            m = SPECIES_LINE.match(line)
            if m:
                active[m.group(1)] = norm(m.group(2))
                entered_turn[m.group(1)] = turn
                continue
            m = MOVE_LINE.match(line)
            if m:
                side, move = m.group(1), norm(m.group(2))
                species = active[side]
                opponent = active["p2" if side == "p1" else "p1"]
                if not species or not opponent or move == "struggle":
                    continue
                true_set = truth.get(side, {}).get(species)
                if true_set is None or move not in true_set["moves"]:
                    continue  # transformed/copied moves fail closed
                yield {
                    "species": species,
                    "moves": true_set["moves"],
                    "item": true_set["item"],
                    "ability": true_set["ability"],
                    "tera": true_set["tera"],
                    "opp_species": opponent,
                    "turn": min(turn, 40),
                    "turns_in": min(turn - entered_turn.get(side, turn), 10),
                    "chosen": move,
                }


def truth_from_labels(label_row: dict):
    out: dict[str, dict[str, dict]] = {}
    for role, team in label_row["true_teams"].items():
        sets = {}
        for p in team:
            species = norm(p["details"].split(",", 1)[0])
            sets[species] = {
                "moves": sorted(norm(m) for m in p.get("moves", [])),
                "item": norm(p.get("item", "")),
                "ability": norm(p.get("baseAbility") or p.get("ability") or ""),
                "tera": norm(p.get("teraType", "")),
            }
        out[role] = sets
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--exclude-battles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    excluded = {json.loads(line)["battle_id"]
                for line in args.exclude_battles.read_text().splitlines() if line}
    labels = {row["battle_id"]: row for row in
              map(json.loads, args.labels.read_text().splitlines())}
    stats = Counter()
    with args.output.open("x", encoding="utf-8") as out:
        for line in args.records.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "failed" or record["battle_id"] in excluded:
                stats["skipped_battles"] += 1
                continue
            label = labels.get(record["battle_id"])
            if label is None:
                stats["skipped_battles"] += 1
                continue
            truth = truth_from_labels(label)
            for row in rows_for_battle(record, truth):
                row["battle_id"] = record["battle_id"]
                out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                stats["rows"] += 1
            stats["battles"] += 1
    print(json.dumps(dict(stats), sort_keys=True))


if __name__ == "__main__":
    main()
