from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constraints import (
    filter_by_damage_range,
    filter_by_ability,
    filter_by_item,
    filter_by_seen_moves,
    filter_by_speed_bounds,
    filter_having_item,
    filter_status_move_assault_vest,
    filter_weather_extension,
    filter_without_item,
    normalize_name,
    uniform_posterior,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class SlotBelief:
    species: str
    prior: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    seen_moves: set[str] = field(default_factory=set)
    eliminated_by: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.candidates = list(self.prior)
        self.eliminated_by.append("reset_uniform_prior")

    def update_candidates(self, candidates: list[dict[str, Any]], reason: str) -> None:
        if candidates:
            if len(candidates) != len(self.candidates):
                self.eliminated_by.append(reason)
            self.candidates = candidates
        else:
            LOGGER.warning("All sets eliminated for %s by %s; resetting to uniform prior", self.species, reason)
            self.reset()

    def posterior(self) -> list[dict[str, Any]]:
        return uniform_posterior(self.candidates)


class BeliefStateModule:
    """Rule-based posterior over opponent random-battle sets.

    The tracker is deliberately conservative: if a rule eliminates every set it
    restores the species prior, matching AGENTS.md's warning behavior.
    """

    def __init__(self, pool_path: str | Path = "data/gen9_random_pool.json", rng: random.Random | None = None):
        self.pool_path = Path(pool_path)
        self.pool = self._load_pool(self.pool_path)
        self.rng = rng or random.Random()
        self.slots: dict[str, SlotBelief] = {}

    @staticmethod
    def _load_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
        if not path.exists():
            return {}
        with path.open() as handle:
            data = json.load(handle)
        return {str(species): list(sets or []) for species, sets in data.items()}

    def reveal(self, slot: str, species: str) -> None:
        if slot in self.slots:
            return
        candidates = list(self.pool.get(species) or self.pool.get(self._lookup_species(species)) or [])
        self.slots[slot] = SlotBelief(species=species, prior=candidates, candidates=list(candidates))

    def _lookup_species(self, species: str) -> str:
        target = normalize_name(species)
        for known in self.pool:
            if normalize_name(known) == target:
                return known
        return species

    def observe_move(self, slot: str, move: str) -> None:
        belief = self.slots.get(slot)
        if belief is None:
            return
        belief.seen_moves.add(move)
        belief.update_candidates(filter_by_seen_moves(belief.candidates, belief.seen_moves), f"move_seen:{move}")

    def observe_item(self, slot: str, item: str) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_by_item(belief.candidates, item), f"item_seen:{item}")

    def observe_ability(self, slot: str, ability: str) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_by_ability(belief.candidates, ability), f"ability_seen:{ability}")

    def observe_speed_bounds(self, slot: str, lower: int | None = None, upper: int | None = None) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_by_speed_bounds(belief.candidates, lower, upper), "speed_constraint")

    def observe_status_move(self, slot: str, move: str | None = None) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_status_move_assault_vest(belief.candidates, move), f"status_move:{move or 'unknown'}")

    def observe_hazard_switch(self, slot: str, hazards_present: bool, took_hazard_damage: bool) -> None:
        belief = self.slots.get(slot)
        if belief is None or not hazards_present:
            return
        if took_hazard_damage:
            belief.update_candidates(filter_without_item(belief.candidates, "Heavy-Duty Boots"), "hazard_damage_no_boots")
        else:
            belief.update_candidates(filter_having_item(belief.candidates, "Heavy-Duty Boots"), "hazards_no_damage_boots")

    def observe_weather_duration(self, slot: str, weather: str, turns: int) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_weather_extension(belief.candidates, weather, turns), f"weather_duration:{weather}:{turns}")

    def observe_damage_dealt(
        self,
        slot: str,
        observed_damage: float,
        damage_range_fn: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(
                filter_by_damage_range(belief.candidates, observed_damage, damage_range_fn, *args, **kwargs),
                "damage_dealt_range",
            )

    def observe_damage_received(
        self,
        slot: str,
        observed_damage: float,
        damage_range_fn: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(
                filter_by_damage_range(belief.candidates, observed_damage, damage_range_fn, *args, **kwargs),
                "damage_received_range",
            )

    def observe_we_moved_first(self, slot: str, our_speed: int) -> None:
        self.observe_speed_bounds(slot, lower=our_speed)

    def observe_opponent_moved_first(self, slot: str, our_speed: int) -> None:
        self.observe_speed_bounds(slot, upper=our_speed)

    def refine(self, refined_belief: dict[str, Any]) -> None:
        for slot, posterior in refined_belief.items():
            if slot not in self.slots or not isinstance(posterior, list):
                continue
            candidates: list[dict[str, Any]] = []
            for entry in posterior:
                if isinstance(entry, dict):
                    candidates.append(entry)
            if candidates:
                self.slots[slot].update_candidates(candidates, "rlm_refinement")

    def posterior(self) -> dict[str, list[dict[str, Any]]]:
        return {slot: belief.posterior() for slot, belief in self.slots.items()}

    def sample(self, k: int = 4) -> list[dict[str, dict[str, Any]]]:
        configs: list[dict[str, dict[str, Any]]] = []
        for _ in range(max(1, k)):
            config: dict[str, dict[str, Any]] = {}
            for slot, belief in self.slots.items():
                posterior = belief.posterior()
                if not posterior:
                    continue
                weights = [float(entry.get("probability", 0.0)) for entry in posterior]
                total = sum(weights)
                if total <= 0:
                    chosen = self.rng.choice(posterior)
                else:
                    chosen = self.rng.choices(posterior, weights=weights, k=1)[0]
                config[slot] = chosen
            configs.append(config)
        return configs

    def update(self, battle: Any) -> None:
        """Best-effort poke-env Battle ingestion.

        This keeps the production path safe before the exact event stream parser
        is wired in: use visible opponent team data, and rely on request-derived
        available moves elsewhere for action masking.
        """
        opponent_team = getattr(battle, "opponent_team", None) or {}
        for key, pokemon in getattr(opponent_team, "items", lambda: [])():
            species = getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None)
            if species:
                self.reveal(str(key), str(species))
            moves = getattr(pokemon, "moves", None) or {}
            iterable = moves.values() if isinstance(moves, dict) else moves
            for move in iterable:
                move_name = getattr(move, "id", None) or getattr(move, "name", None) or str(move)
                self.observe_move(str(key), str(move_name))
                category = normalize_name(getattr(getattr(move, "category", None), "name", None) or getattr(move, "category", None))
                if category == "status":
                    self.observe_status_move(str(key), str(move_name))
            item = getattr(pokemon, "item", None)
            if item:
                self.observe_item(str(key), str(item))
            ability = getattr(pokemon, "ability", None)
            if ability:
                self.observe_ability(str(key), str(ability))
