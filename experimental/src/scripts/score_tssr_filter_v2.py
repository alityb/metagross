#!/usr/bin/env python3
"""Filter-v2 TSSR: extend reveal-consistency to items, abilities, and tera.

Arms (same held-out records/labels/pool as the frozen baseline):
  prior          - pool-frequency prior, no evidence;
  revealed_moves - moves-only filter (v1 evidence, for continuity);
  revealed_full  - moves + revealed item + revealed ability + tera type.

Candidate keys now include teraType. Truth matches on species + moves +
item + teraType. Reveal misattributions (Illusion, ability-changing
effects, item swaps) surface honestly in the truth-eliminated metric.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experimental.src.scripts.score_tssr_baseline import (
    MOVE_LINE, SPECIES_LINE, norm,
)

SLOT = re.compile(r"\|(p[12])a: ")
ITEM_LINE = re.compile(r"\|-(?:item|enditem)\|(p[12])a: [^|]*\|([^|]+)")
ABILITY_LINE = re.compile(r"\|-ability\|(p[12])a: [^|]*\|([^|]+)")
TERA_LINE = re.compile(r"\|-terastallize\|(p[12])a: [^|]*\|([^|]+)")
FROM_ITEM = re.compile(r"\[from\] item: ([^|]+)")
FROM_ABILITY = re.compile(r"\[from\] ability: ([^|]+)")
OF_SLOT = re.compile(r"\[of\] (p[12])a: ")


def build_pool_index_v2(pool_path: Path):
    pool = json.loads(pool_path.read_text())
    by_species: dict[str, Counter] = defaultdict(Counter)
    for team in pool["teams"]:
        for p in team:
            key = (tuple(sorted(norm(m) for m in p["moves"])), norm(p["item"]),
                   norm(p["ability"]), norm(p["teraType"]))
            by_species[norm(p["speciesId"])][key] += 1
    return by_species


def true_sets_v2(label_row: dict):
    out: dict[str, dict[str, tuple]] = {}
    for role, team in label_row["true_teams"].items():
        sets = {}
        for p in team:
            species = norm(p["details"].split(",", 1)[0])
            sets[species] = (tuple(sorted(norm(m) for m in p.get("moves", []))),
                            norm(p.get("item", "")),
                            norm(p.get("teraType", "")))
        out[role] = sets
    return out


class Evidence:
    __slots__ = ("moves", "item", "ability", "tera")

    def __init__(self):
        self.moves: set = set()
        self.item = None
        self.ability = None
        self.tera = None


def line_side_and_payload(line: str, pattern):
    m = pattern.match(line)
    if m:
        return m.group(1), norm(m.group(2))
    return None, None


def from_side(line: str) -> str | None:
    m = OF_SLOT.search(line)
    if m:
        return m.group(1)
    m = SLOT.search(line)
    return m.group(1) if m else None


def scan_decisions_v2(states: list[dict], own_role: str):
    opp = "p2" if own_role == "p1" else "p1"
    active: dict[str, str | None] = {"p1": None, "p2": None}
    evidence: dict[str, Evidence] = defaultdict(Evidence)

    def opp_evidence(side: str):
        species = active.get(side)
        if side == opp and species:
            return evidence[species]
        return None

    for state in states:
        for line in state["prefix_delta"]:
            m = SPECIES_LINE.match(line)
            if m:
                active[m.group(1)] = norm(m.group(2))
                continue
            m = MOVE_LINE.match(line)
            if m:
                ev = opp_evidence(m.group(1))
                move = norm(m.group(2))
                if ev is not None and move != "struggle":
                    ev.moves.add(move)
                continue
            for pattern, field in ((ITEM_LINE, "item"), (ABILITY_LINE, "ability"),
                                   (TERA_LINE, "tera")):
                side, payload = line_side_and_payload(line, pattern)
                if side:
                    ev = opp_evidence(side)
                    if ev is not None and getattr(ev, field) is None:
                        setattr(ev, field, payload)
                    break
            else:
                m = FROM_ITEM.search(line)
                if m:
                    ev = opp_evidence(from_side(line))
                    if ev is not None and ev.item is None:
                        ev.item = norm(m.group(1))
                m = FROM_ABILITY.search(line)
                if m:
                    ev = opp_evidence(from_side(line))
                    if ev is not None and ev.ability is None:
                        ev.ability = norm(m.group(1))
        if state["actionable"] and active[opp]:
            species = active[opp]
            ev = evidence[species]
            yield species, frozenset(ev.moves), ev.item, ev.ability, ev.tera


def consistent(key: tuple, moves: frozenset, item, ability, tera, full: bool) -> bool:
    if not moves <= set(key[0]):
        return False
    if not full:
        return True
    if item is not None and key[1] != item:
        return False
    if ability is not None and key[2] != ability:
        return False
    if tera is not None and key[3] != tera:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--marginal-rows", type=Path, default=None,
                        help="policy-inference rows; adds a marginal-likelihood arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    marginal = None
    if args.marginal_rows:
        marginal = defaultdict(Counter)
        for line in args.marginal_rows.read_text().splitlines():
            r = json.loads(line)
            marginal[(r["species"], tuple(r["moves"]))][r["chosen"]] += 1

    pool_index = build_pool_index_v2(args.pool)
    labels = {row["battle_id"]: row for row in
              map(json.loads, args.labels.read_text().splitlines())}

    per_battle: dict[str, dict] = {}
    by_reveal: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
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
        truth = true_sets_v2(label)
        battle_rows: dict[str, list] = defaultdict(list)
        for role, states in record["roles"].items():
            opp_role = "p2" if role == "p1" else "p1"
            for species, moves, item, ability, tera in scan_decisions_v2(states, role):
                candidates = pool_index.get(species)
                true_set = truth[opp_role].get(species)
                if not candidates or true_set is None:
                    skipped["species_not_in_pool_or_labels"] += 1
                    continue
                truth_key = next((k for k in candidates
                                  if k[0] == true_set[0] and k[1] == true_set[1]
                                  and k[3] == true_set[2]), None)
                distinct = len(candidates)
                arm_specs = [("prior", None), ("revealed_moves", False),
                             ("revealed_full", True)]
                if marginal is not None:
                    arm_specs.append(("marginal_lh", True))
                for arm, full in arm_specs:
                    if arm == "prior":
                        support = dict(candidates)
                    else:
                        support = {k: float(v) for k, v in candidates.items()
                                   if consistent(k, moves, item, ability, tera, full)}
                    if arm == "marginal_lh" and moves and support:
                        for k in support:
                            counts = marginal.get((species, k[0]), Counter())
                            total = sum(counts.values())
                            for m in moves:
                                support[k] *= (counts.get(m, 0) + 1.0) / (total + 4.0)
                    stotal = sum(support.values())
                    if truth_key is None or stotal == 0 or truth_key not in support:
                        row = {"mass": 0.0, "tssr": 0.0, "top1": 0.0,
                               "eliminated": 1.0}
                    else:
                        mass = support[truth_key] / stotal
                        row = {"mass": mass, "tssr": mass * distinct,
                               "top1": 1.0 if support[truth_key] == max(support.values()) else 0.0,
                               "eliminated": 0.0}
                    battle_rows[arm].append(row)
                    by_reveal[len(moves)][arm].append(row["mass"])
        if battle_rows:
            per_battle[record["battle_id"]] = {
                arm: {metric: statistics.fmean(r[metric] for r in rows)
                      for metric in ("mass", "tssr", "top1", "eliminated")}
                for arm, rows in battle_rows.items()
            }

    def summarize(arm, metric):
        values = [b[arm][metric] for b in per_battle.values()]
        return (statistics.fmean(values),
                statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)

    report = {"schema": "metagross-tssr-filter-v2/v1",
              "battles": len(per_battle), "skipped": dict(skipped),
              "arms": {}, "true_mass_by_revealed_moves": {}}
    arm_names = ["prior", "revealed_moves", "revealed_full"] + (
        ["marginal_lh"] if marginal is not None else [])
    for arm in arm_names:
        report["arms"][arm] = {}
        for metric in ("mass", "tssr", "top1", "eliminated"):
            mean, se = summarize(arm, metric)
            report["arms"][arm][metric] = {"mean": round(mean, 5), "se": round(se, 5)}
    for count in sorted(by_reveal):
        cell = by_reveal[count]
        report["true_mass_by_revealed_moves"][str(count)] = {
            arm: {"mean": round(statistics.fmean(cell[arm]), 5), "n": len(cell[arm])}
            for arm in arm_names}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["arms"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
