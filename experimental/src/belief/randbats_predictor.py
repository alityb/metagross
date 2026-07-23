"""Randbats team prediction from the public generator distribution.

gen9randombattle is the one format where the hidden-information prior is *known*:
teams are drawn by public Showdown code. We pre-sampled 50k teams (300k sets)
with that code (data/randbats_pools/gen9randombattle_pool_50000.json). This
predictor completes partially-revealed teams for the metamon replay parser by
matching revealed attributes against those ground-truth generated sets —
replacing the usage-stats predictor, which has no data for this format.

Native contribution: exact-generator belief, used here for BC dataset
reconstruction (Phase 1 of the ladder plan).
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import Optional

from metamon.backend.team_prediction.predictor import TeamPredictor
from metamon.backend.team_prediction.team import PokemonSet, TeamSet

DEFAULT_POOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "randbats_pools", "gen9randombattle_pool_50000.json",
)
UNKNOWN_ITEM = "unknown_item"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class RandbatsPoolPredictor(TeamPredictor):
    """Completes revealed teams by sampling consistent sets from the generator pool."""

    def __init__(self, pool_path: str = DEFAULT_POOL, seed: Optional[int] = None):
        super().__init__()
        self.rng = random.Random(seed)
        with open(pool_path) as f:
            pool = json.load(f)
        # id -> display for moves (via poke-env's gen 9 data)
        from poke_env.data import GenData

        gen_moves = GenData.from_gen(9).moves
        self.move_display = {mid: m.get("name", mid) for mid, m in gen_moves.items()}

        self.sets_by_species: dict[str, list[dict]] = defaultdict(list)
        self.all_sets: list[dict] = []
        for team in pool["teams"]:
            for p in team:
                entry = {
                    "species": p["species"],
                    "species_key": norm(p["speciesId"]),
                    "moves_display": [self.move_display.get(m, m) for m in p["moves"]],
                    "moves_key": frozenset(norm(m) for m in p["moves"]),
                    "ability": p.get("ability") or PokemonSet.NO_ABILITY,
                    "item": p.get("item") or PokemonSet.NO_ITEM,
                    "tera": p.get("teraType") or PokemonSet.NO_TERA_TYPE,
                    "evs": p.get("evs") or {},
                }
                self.sets_by_species[entry["species_key"]].append(entry)
                self.all_sets.append(entry)

    # ----- matching -----

    def _candidates(self, revealed: PokemonSet) -> list[dict]:
        species_key = norm(revealed.base_species if revealed.base_species != PokemonSet.MISSING_NAME else revealed.name)
        cands = self.sets_by_species.get(species_key) or self.sets_by_species.get(norm(revealed.name))
        if not cands:
            return []
        rev_moves = frozenset(
            norm(m) for m in revealed.moves
            if m not in (PokemonSet.MISSING_MOVE, PokemonSet.NO_MOVE)
        )
        rev_ability = None if revealed.ability in (PokemonSet.MISSING_ABILITY,) else norm(revealed.ability)
        rev_item = None if revealed.item in (PokemonSet.MISSING_ITEM,) else norm(revealed.item)
        rev_tera = None if revealed.tera_type in (PokemonSet.MISSING_TERA_TYPE,) else norm(revealed.tera_type)

        def matches(c, use_ability=True, use_item=True, use_tera=True, use_moves=True):
            if use_moves and not rev_moves <= c["moves_key"]:
                return False
            if use_ability and rev_ability and norm(c["ability"]) != rev_ability:
                return False
            if use_item and rev_item and norm(c["item"]) != rev_item:
                return False
            if use_tera and rev_tera and norm(c["tera"]) != rev_tera:
                return False
            return True

        # progressively relax constraints so we always return something
        for kwargs in (
            {},
            {"use_tera": False},
            {"use_item": False, "use_tera": False},
            {"use_ability": False, "use_item": False, "use_tera": False},
            {"use_ability": False, "use_item": False, "use_tera": False, "use_moves": False},
        ):
            hits = [c for c in cands if matches(c, **kwargs)]
            if hits:
                return hits
        return cands

    def _to_pokemon_set(self, c: dict) -> PokemonSet:
        evs = c["evs"]
        return PokemonSet.from_dict(
            {
                "name": c["species"],
                "gen": 9,
                "moves": list(c["moves_display"])[:4],
                "ability": c["ability"],
                "item": c["item"],
                "nature": "Hardy",  # randbats sets have neutral natures
                "evs": [evs.get(k, 85) for k in ("hp", "atk", "def", "spa", "spd", "spe")],
                "ivs": [31] * 6,
                "tera_type": c["tera"],
            }
        )

    # ----- TeamPredictor API -----

    def fill_team(
        self,
        team: TeamSet,
        date,
        rating=None,
        gameid=None,
    ) -> TeamSet:
        used_species = {
            norm(p.name) for p in [team.lead] + list(team.reserve)
            if p.name != PokemonSet.MISSING_NAME
        }
        merged = []
        for p in [team.lead] + list(team.reserve):
            if p.name == PokemonSet.MISSING_NAME:
                # unrevealed teammate: draw from the generator marginal,
                # respecting species clause
                for _ in range(200):
                    c = self.rng.choice(self.all_sets)
                    if c["species_key"] not in used_species:
                        break
                used_species.add(c["species_key"])
                merged.append(self._to_pokemon_set(c))
            else:
                cands = self._candidates(p)
                filled = copy.deepcopy(p)
                item_was_unrevealed = filled.item == PokemonSet.MISSING_ITEM
                if cands:
                    choice = self._to_pokemon_set(self.rng.choice(cands))
                    # names must match for the merge helper; keep revealed name
                    choice.name = filled.name
                    choice.base_species = (
                        filled.base_species
                        if filled.base_species != PokemonSet.MISSING_NAME
                        else choice.base_species
                    )
                    try:
                        filled.fill_from_PokemonSet(choice)
                    except ValueError:
                        pass  # keep whatever was revealed
                if item_was_unrevealed:
                    # A pool match can complete a moveset without revealing which
                    # item it sampled. Keep the replay's hidden-item observation.
                    filled.item = UNKNOWN_ITEM
                # any survivors of matching failure: fill neutral defaults
                filled.moves = [
                    m if m != PokemonSet.MISSING_MOVE else PokemonSet.NO_MOVE
                    for m in filled.moves
                ]
                if filled.ability == PokemonSet.MISSING_ABILITY:
                    filled.ability = PokemonSet.NO_ABILITY
                if filled.item == PokemonSet.MISSING_ITEM:
                    filled.item = PokemonSet.NO_ITEM
                if filled.nature == PokemonSet.MISSING_NATURE:
                    filled.nature = "Hardy"
                if filled.tera_type == PokemonSet.MISSING_TERA_TYPE:
                    filled.tera_type = PokemonSet.NO_TERA_TYPE
                filled.evs = [85 if ev == PokemonSet.MISSING_EV else ev for ev in filled.evs]
                filled.ivs = [31 if iv == PokemonSet.MISSING_IV else iv for iv in filled.ivs]
                merged.append(filled)
        return TeamSet(lead=merged[0], reserve=merged[1:], format=team.format)
