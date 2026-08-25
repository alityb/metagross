#!/usr/bin/env python3
"""Component-1 evidence tables (Buro/Kermit-style, count-based, v1).

For every set observed in TRAINING battles (both sides' true teams are
training instances), count which of its moves was publicly revealed FIRST.
Table: P(first-revealed move | set-key), Laplace-smoothed at query time.
Set-keys match the pool candidate keys: (sorted moves, item, ability).

Training battles exclude the frozen TSSR evaluation battles; the eval
labels file is never read here. Fail-open at query time: an unseen set-key
contributes a uniform likelihood (no belief change).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experimental.src.scripts.score_tssr_baseline import (
    MOVE_LINE, SPECIES_LINE, norm,
)


def first_revealed_moves(record: dict) -> dict[str, dict[str, str]]:
    """Per side, per species: the first publicly revealed move (from either
    POV's public stream; p1's stream suffices since prefixes are shared)."""
    states = record["roles"].get("p1") or record["roles"].get("p2") or []
    active = {"p1": None, "p2": None}
    first: dict[str, dict[str, str]] = {"p1": {}, "p2": {}}
    for state in states:
        for line in state["prefix_delta"]:
            m = SPECIES_LINE.match(line)
            if m:
                active[m.group(1)] = norm(m.group(2))
                continue
            m = MOVE_LINE.match(line)
            if m:
                side, move = m.group(1), norm(m.group(2))
                species = active[side]
                if species and move != "struggle" and species not in first[side]:
                    first[side][species] = move
    return first


def set_keys_from_labels(label_row: dict) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = {}
    for role, team in label_row["true_teams"].items():
        sets = {}
        for pokemon in team:
            species = norm(pokemon["details"].split(",", 1)[0])
            sets[species] = (tuple(sorted(norm(m) for m in pokemon.get("moves", []))),
                            norm(pokemon.get("item", "")),
                            norm(pokemon.get("baseAbility") or pokemon.get("ability") or ""))
        out[role] = sets
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--exclude-battles", type=Path, required=True,
                        help="jsonl whose battle_id values are excluded (eval set)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    excluded = {json.loads(line)["battle_id"]
                for line in args.exclude_battles.read_text().splitlines() if line}
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[row["battle_id"]] = row

    counts: dict[str, dict[tuple, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    used = skipped = 0
    for line in args.records.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial trailing line of an in-progress extraction
        if record.get("status") == "failed" or record["battle_id"] in excluded:
            skipped += 1
            continue
        label = labels.get(record["battle_id"])
        if label is None:
            skipped += 1
            continue
        truth = set_keys_from_labels(label)
        first = first_revealed_moves(record)
        for side in ("p1", "p2"):
            for species, move in first[side].items():
                key = truth[side].get(species)
                if key is not None and move in key[0]:
                    counts[species][key][move] += 1
                    used += 1

    serializable = {
        species: [
            {"set": {"moves": list(key[0]), "item": key[1], "ability": key[2]},
             "first_move_counts": dict(moves)}
            for key, moves in sets.items()
        ]
        for species, sets in counts.items()
    }
    args.output.write_text(json.dumps({
        "schema": "metagross-evidence-tables/v1",
        "evidence": "first_revealed_move",
        "training_instances": used,
        "skipped_battles": skipped,
        "species": serializable,
    }, sort_keys=True) + "\n")
    print(json.dumps({"species": len(serializable), "instances": used,
                      "skipped": skipped}))


if __name__ == "__main__":
    main()
