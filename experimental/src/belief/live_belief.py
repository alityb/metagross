#!/usr/bin/env python3
"""Live belief tracker for gen9randombattle opponent sets.

Maintains a real-time JSON of what each opponent Pokémon could have,
narrowing with every reveal (move used, ability announced, item shown, etc.).
Uses the public generator data (sets.json) as the exact prior.

This is the "memory" FP lacks: instead of sampling a random consistent set
each turn and pretending to know everything, this tracks what's STILL
uncertain and exposes it for uncertainty-aware evaluation.

Usage: instantiated once per battle, fed protocol lines, queried each turn.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Gen 6+ type chart: attacking type -> {defending type: multiplier} (non-1x only)
TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0,
             "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0,
              "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0,
                 "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5,
              "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0,
              "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0,
            "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5,
                 "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0,
                 "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5,
               "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0,
               "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0,
               "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0,
                "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5,
            "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0,
            "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5,
             "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5,
             "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0,
              "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0,
              "dark": 2.0, "steel": 0.5},
}

# Minimum base power for an unrevealed damaging move to count as a "threat"
THREAT_MIN_BASE_POWER = 60


def effectiveness(move_type: str, defender_types) -> float:
    """Type effectiveness multiplier of move_type vs a (mono or dual) defender."""
    chart = TYPE_CHART.get(move_type)
    if chart is None:
        return 1.0
    mult = 1.0
    for t in defender_types:
        mult *= chart.get(t, 1.0)
    return mult


class OpponentBelief:
    """Tracks the belief over one opponent Pokémon's full set."""

    def __init__(self, species: str, level: int, pool_sets: list[dict]):
        self.species = species
        self.level = level
        self.revealed_moves: set[str] = set()
        self.revealed_ability: Optional[str] = None
        self.revealed_item: Optional[str] = None
        self.revealed_tera: Optional[str] = None
        self.terastallized: bool = False
        # threat caches; invalidated whenever _consistent narrows
        self._masks_cache: Optional[dict[frozenset, int]] = None
        self._threat_cache: dict[tuple, float] = {}
        self._consistent: list[dict] = self._filter(pool_sets)

    def _filter(self, candidates: list[dict]) -> list[dict]:
        """Filter candidate sets by everything revealed so far."""
        self._masks_cache = None
        self._threat_cache = {}
        result = []
        for c in candidates:
            if self.revealed_ability and norm(c["ability"]) != norm(self.revealed_ability):
                continue
            if self.revealed_item and norm(c["item"]) != norm(self.revealed_item):
                continue
            if self.revealed_tera and norm(c["tera"]) != norm(self.revealed_tera):
                continue
            c_moves = {norm(m) for m in c["moves_display"]}
            if not self.revealed_moves <= c_moves:
                continue
            result.append(c)
        return result

    def reveal_move(self, move: str):
        move = norm(move)
        if move and move not in self.revealed_moves:
            self.revealed_moves.add(move)
            self._consistent = self._filter(self._consistent)

    def reveal_ability(self, ability: str):
        ability = norm(ability)
        if ability and not self.revealed_ability:
            self.revealed_ability = ability
            self._consistent = self._filter(self._consistent)

    def reveal_item(self, item: str):
        item = norm(item)
        if item and not self.revealed_item:
            self.revealed_item = item
            self._consistent = self._filter(self._consistent)

    def reveal_tera(self, tera: str):
        tera = norm(tera)
        if tera and not self.revealed_tera:
            self.revealed_tera = tera
            self._consistent = self._filter(self._consistent)

    def mark_terastallized(self):
        self.terastallized = True

    def _unrevealed_type_masks(self, move_info: dict) -> dict[frozenset, int]:
        """Count consistent sets grouped by the frozenset of attack types among
        their UNREVEALED damaging moves (bp >= THREAT_MIN_BASE_POWER).

        move_info: {normalized_move_id: (type, category, base_power)}
        """
        if self._masks_cache is None:
            masks: dict[frozenset, int] = defaultdict(int)
            for c in self._consistent:
                types = set()
                for m in c["moves_display"]:
                    mn = norm(m)
                    if mn in self.revealed_moves:
                        continue
                    info = move_info.get(mn)
                    if not info:
                        continue
                    mtype, category, bp = info
                    if category.lower() == "status" or (bp or 0) < THREAT_MIN_BASE_POWER:
                        continue
                    types.add(mtype)
                masks[frozenset(types)] += 1
            self._masks_cache = dict(masks)
            self._threat_cache = {}
        return self._masks_cache

    def unrevealed_threat_prob(self, defender_types, move_info: dict) -> float:
        """P(this mon's set contains an UNREVEALED damaging move that is
        super-effective vs defender_types), exact under the pool prior."""
        key = tuple(sorted(defender_types))
        masks = self._unrevealed_type_masks(move_info)
        cached = self._threat_cache.get(key)
        if cached is not None:
            return cached
        total = sum(masks.values())
        if total == 0:
            return 0.0
        hit = 0
        for mask, count in masks.items():
            for t in mask:
                if effectiveness(t, key) >= 2.0:
                    hit += count
                    break
        prob = hit / total
        self._threat_cache[key] = prob
        return prob

    def possible_remaining_moves(self) -> dict[str, float]:
        """Probability distribution over moves NOT yet revealed."""
        if not self._consistent:
            return {}
        move_counts: dict[str, int] = defaultdict(int)
        for c in self._consistent:
            for m in c["moves_display"]:
                mn = norm(m)
                if mn not in self.revealed_moves:
                    move_counts[mn] += 1
        total = sum(move_counts.values())
        if total == 0:
            return {}
        return {m: c / total for m, c in sorted(move_counts.items(), key=lambda kv: -kv[1])}

    def possible_items(self) -> dict[str, float]:
        if not self._consistent:
            return {}
        item_counts: dict[str, int] = defaultdict(int)
        for c in self._consistent:
            item_counts[norm(c["item"])] += 1
        total = len(self._consistent)
        return {i: c / total for i, c in sorted(item_counts.items(), key=lambda kv: -kv[1])}

    def possible_tera_types(self) -> dict[str, float]:
        if not self._consistent or self.terastallized:
            return {}
        tera_counts: dict[str, int] = defaultdict(int)
        for c in self._consistent:
            tera_counts[norm(c["tera"])] += 1
        total = len(self._consistent)
        return {t: c / total for t, c in sorted(tera_counts.items(), key=lambda kv: -kv[1])}

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "level": self.level,
            "revealed_moves": sorted(self.revealed_moves),
            "revealed_ability": self.revealed_ability,
            "revealed_item": self.revealed_item,
            "revealed_tera": self.revealed_tera,
            "terastallized": self.terastallized,
            "consistent_sets": len(self._consistent),
            "possible_remaining_moves": self.possible_remaining_moves(),
            "possible_items": self.possible_items(),
            "possible_tera_types": self.possible_tera_types(),
        }


class BeliefTracker:
    """Tracks belief over the entire opponent team for one battle."""

    def __init__(self, pool_path: str = None):
        if pool_path is None:
            pool_path = str(Path(__file__).resolve().parents[1] / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json")
        with open(pool_path) as f:
            pool = json.load(f)
        # index: species_key -> list of candidate set dicts
        self._sets_by_species: dict[str, list[dict]] = defaultdict(list)
        for team in pool["teams"]:
            for p in team:
                key = norm(p["speciesId"])
                entry = {
                    "species": p["species"],
                    "species_key": key,
                    "level": p["level"],
                    "moves_display": p["moves"],
                    "ability": p.get("ability", ""),
                    "item": p.get("item", ""),
                    "tera": p.get("teraType", ""),
                }
                self._sets_by_species[key].append(entry)
        # per-battle state
        self._opponent_mons: dict[str, OpponentBelief] = {}  # key = species_key
        self._seen_species_order: list[str] = []
        self._pool_path = pool_path

    def reset(self):
        """Clear all per-battle state (call when a new battle starts)."""
        self._opponent_mons = {}
        self._seen_species_order = []

    def on_opponent_switch_in(self, species: str, level: int):
        key = norm(species)
        if key not in self._opponent_mons:
            candidates = self._sets_by_species.get(key, [])
            # filter by level
            candidates = [c for c in candidates if c["level"] == level]
            self._opponent_mons[key] = OpponentBelief(species, level, candidates)
            self._seen_species_order.append(key)

    def on_opponent_move(self, species: str, move: str):
        key = norm(species)
        if key in self._opponent_mons:
            self._opponent_mons[key].reveal_move(move)

    def on_opponent_ability(self, species: str, ability: str):
        key = norm(species)
        if key in self._opponent_mons:
            self._opponent_mons[key].reveal_ability(ability)

    def on_opponent_item(self, species: str, item: str):
        key = norm(species)
        if key in self._opponent_mons:
            self._opponent_mons[key].reveal_item(item)

    def on_opponent_tera(self, species: str, tera_type: str):
        key = norm(species)
        if key in self._opponent_mons:
            self._opponent_mons[key].reveal_tera(tera_type)
            self._opponent_mons[key].mark_terastallized()

    def get_belief(self, species: str) -> Optional[dict]:
        key = norm(species)
        if key in self._opponent_mons:
            return self._opponent_mons[key].to_dict()
        return None

    def get_all_beliefs(self) -> dict:
        return {k: self._opponent_mons[k].to_dict() for k in self._seen_species_order}

    def to_json(self) -> str:
        return json.dumps(self.get_all_beliefs(), indent=2, sort_keys=True)
