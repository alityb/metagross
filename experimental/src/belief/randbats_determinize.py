"""Strict Random Battle posterior completions for offline counterfactual labels."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


class RandbatsDeterminizer:
    """Sample complete pool teams consistent with the visible opponent state."""

    def __init__(self, pool_path: Path, seed: int = 0):
        self.rng = random.Random(seed)
        self.teams = json.loads(pool_path.read_text(encoding="utf-8"))["teams"]
        self.by_species: dict[str, list[list[dict]]] = defaultdict(list)
        for team in self.teams:
            for pokemon in team:
                self.by_species[_norm(pokemon["speciesId"])].append(team)

    @staticmethod
    def _matches(revealed, candidate: dict) -> bool:
        if _norm(revealed.id) != _norm(candidate["speciesId"]):
            return False
        if revealed.level != candidate["level"]:
            return False
        moves = {_norm(move.id) for move in revealed.moves if _norm(move.id) != "none"}
        if not moves.issubset({_norm(move) for move in candidate["moves"]}):
            return False
        if _norm(revealed.ability) not in {"", "none"} and _norm(revealed.ability) != _norm(candidate["ability"]):
            return False
        if _norm(revealed.item) not in {"", "none", "unknownitem"} and _norm(revealed.item) != _norm(candidate["item"]):
            return False
        if _norm(revealed.tera_type) not in {"", "typeless"} and _norm(revealed.tera_type) != _norm(candidate["teraType"]):
            return False
        return True

    def sample_team(self, revealed: list) -> list[dict] | None:
        if not revealed:
            return self.rng.choice(self.teams)
        candidates = self.by_species.get(_norm(revealed[0].id), [])
        matching = [
            team
            for team in candidates
            if all(any(self._matches(pokemon, set_) for set_ in team) for pokemon in revealed)
        ]
        return self.rng.choice(matching) if matching else None
