#!/usr/bin/env python3
"""Set-level TSSR baselines over extracted belief-eval records.

For each actionable decision of each POV role, form a posterior over the
opponent ACTIVE Pokemon's hidden set using the generator pool:

  arm 'prior'    - pool-frequency prior over the species' distinct sets,
                   no in-battle evidence;
  arm 'revealed' - the same prior filtered to sets consistent with the
                   opponent's publicly revealed moves so far.

Truth (species + exact moves + item) comes from the separate labels file and
is used for scoring only. Metrics follow Solinas et al. (AAAI-19): true-set
posterior mass, TSSR (mass x number of distinct candidate sets, i.e. how
many times likelier than uniform), top-1 recovery, and the truth-eliminated
rate (candidate filter removed the true set - e.g. Illusion or pool gaps).
Battle-level means with across-battle standard errors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def build_pool_index(pool_path: Path):
    pool = json.loads(pool_path.read_text())
    by_species: dict[str, Counter] = defaultdict(Counter)
    for team in pool["teams"]:
        for pokemon in team:
            key = (tuple(sorted(norm(m) for m in pokemon["moves"])),
                   norm(pokemon["item"]), norm(pokemon["ability"]))
            by_species[norm(pokemon["speciesId"])][key] += 1
    return by_species


def true_sets_from_labels(label_row: dict) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = {}
    for role, team in label_row["true_teams"].items():
        sets = {}
        for pokemon in team:
            species = norm(pokemon["details"].split(",", 1)[0])
            moves = tuple(sorted(norm(m) for m in pokemon.get("moves", [])))
            item = norm(pokemon.get("item", ""))
            ability = norm(pokemon.get("baseAbility") or pokemon.get("ability") or "")
            sets[species] = (moves, item, ability)
        out[role] = sets
    return out


SPECIES_LINE = re.compile(r"\|(?:switch|drag|replace)\|(p[12])a: [^|]*\|([^,|]+)")
MOVE_LINE = re.compile(r"\|move\|(p[12])a: [^|]*\|([^|]+)")


def scan_decisions(states: list[dict], own_role: str):
    """Yield (decision_index, opponent_active_species, ordered_revealed_moves)
    at each actionable own decision, from the causal public prefix only."""
    opp = "p2" if own_role == "p1" else "p1"
    active: str | None = None
    revealed: dict[str, list] = defaultdict(list)
    for state in states:
        for line in state["prefix_delta"]:
            m = SPECIES_LINE.match(line)
            if m and m.group(1) == opp:
                active = norm(m.group(2))
                continue
            m = MOVE_LINE.match(line)
            if m and m.group(1) == opp and active:
                move = norm(m.group(2))
                if move != "struggle" and move not in revealed[active]:
                    revealed[active].append(move)
        if state["actionable"] and active:
            yield state["request_index"], active, tuple(revealed[active])


LAPLACE = 1.0


def score_decision(candidates: Counter, truth: tuple, revealed: tuple,
                   species_tables: dict | None):
    """Returns per-arm (true_mass, tssr, top1, eliminated)."""
    distinct = len(candidates)
    revealed_set = set(revealed)
    truth_key = None
    for key in candidates:
        if key[0] == truth[0] and key[1] == truth[1]:
            truth_key = key
            break
    arms = ["prior", "revealed"] + (["tables"] if species_tables is not None else [])
    out = {}
    for arm in arms:
        if arm == "prior":
            support = dict(candidates)
        else:
            support = {k: float(v) for k, v in candidates.items()
                       if revealed_set <= set(k[0])}
        if arm == "tables" and revealed and support:
            first = revealed[0]
            for key in support:
                counts = (species_tables or {}).get(key, {})
                total = sum(counts.values())
                likelihood = (counts.get(first, 0) + LAPLACE) / (total + LAPLACE * 4)
                support[key] *= likelihood
        stotal = sum(support.values())
        if truth_key is None or stotal == 0 or truth_key not in support:
            out[arm] = {"mass": 0.0, "tssr": 0.0, "top1": 0.0, "eliminated": 1.0}
            continue
        mass = support[truth_key] / stotal
        top = max(support.values())
        out[arm] = {
            "mass": mass,
            "tssr": mass * distinct,
            "top1": 1.0 if support[truth_key] == top else 0.0,
            "eliminated": 0.0,
        }
    return out, len(revealed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=None,
                        help="evidence tables from build_evidence_tables.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tables = None
    if args.tables:
        raw = json.loads(args.tables.read_text())
        tables = {}
        for species, rows in raw["species"].items():
            tables[species] = {
                (tuple(sorted(row["set"]["moves"])), row["set"]["item"],
                 row["set"]["ability"]): row["first_move_counts"]
                for row in rows
            }

    pool_index = build_pool_index(args.pool)
    labels = {row["battle_id"]: row for row in
              map(json.loads, args.labels.read_text().splitlines())}

    per_battle: dict[str, dict] = {}
    by_reveal_count: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    for line in args.records.read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "failed":
            skipped["failed_record"] += 1
            continue
        label = labels.get(record["battle_id"])
        if label is None:
            skipped["missing_label"] += 1
            continue
        truth = true_sets_from_labels(label)
        battle_rows: dict[str, list] = defaultdict(list)
        for role, states in record["roles"].items():
            opp_role = "p2" if role == "p1" else "p1"
            for _index, species, revealed in scan_decisions(states, role):
                candidates = pool_index.get(species)
                true_set = truth[opp_role].get(species)
                if not candidates or true_set is None:
                    skipped["species_not_in_pool_or_labels"] += 1
                    continue
                scores, n_revealed = score_decision(
                    candidates, true_set, revealed,
                    tables.get(species) if tables else None)
                for arm, row in scores.items():
                    battle_rows[arm].append(row)
                    by_reveal_count[n_revealed][arm].append(row["mass"])
        if battle_rows:
            per_battle[record["battle_id"]] = {
                arm: {metric: statistics.fmean(r[metric] for r in rows)
                      for metric in ("mass", "tssr", "top1", "eliminated")}
                for arm, rows in battle_rows.items()
            }

    def summarize(arm: str, metric: str):
        values = [b[arm][metric] for b in per_battle.values()]
        mean = statistics.fmean(values)
        se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return mean, se

    report = {
        "schema": "metagross-tssr-baseline/v1",
        "battles": len(per_battle),
        "decisions": int(sum(len(v) for c in by_reveal_count.values()
                             for v in [c["prior"]])),
        "skipped": dict(skipped),
        "arms": {},
        "true_mass_by_revealed_moves": {},
        "notes": [
            "truth used as evaluation labels only",
            "evidence = publicly revealed moves of the active species (v1)",
            "eliminated = candidate filter removed the true set (Illusion/pool gap)",
        ],
    }
    arm_names = ["prior", "revealed"] + (["tables"] if tables else [])
    for arm in arm_names:
        report["arms"][arm] = {}
        for metric in ("mass", "tssr", "top1", "eliminated"):
            mean, se = summarize(arm, metric)
            report["arms"][arm][metric] = {"mean": round(mean, 5),
                                           "se": round(se, 5)}
    for count in sorted(by_reveal_count):
        cell = by_reveal_count[count]
        report["true_mass_by_revealed_moves"][str(count)] = {
            arm: {"mean": round(statistics.fmean(cell[arm]), 5),
                  "n": len(cell[arm])}
            for arm in arm_names
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
