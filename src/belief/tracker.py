from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constraints import (
    CONTACT_MOVES,
    EVIOLITE_CANDIDATES,
    PRANKSTER_ELIGIBLES,
    SELF_RECOIL_MOVES,
    SPEED_MODIFYING_ABILITIES,
    STATUS_MOVES,
    TRAPPING_ABILITIES,
    filter_by_ability,
    filter_by_damage_range,
    filter_by_item,
    filter_by_seen_moves,
    filter_by_speed_bounds,
    filter_having_item,
    filter_status_move_assault_vest,
    filter_weather_extension,
    filter_without_ability,
    filter_without_item,
    filter_without_items,
    normalize_name,
    uniform_posterior,
)


LOGGER = logging.getLogger(__name__)

# Tolerance for HP fraction comparisons (accounts for rounding in PS display).
HP_TOL = 0.02

# Leftovers/Black Sludge recovery fraction per turn.
LEFTOVERS_FRAC = 1 / 16
# Life Orb recoil fraction.
LIFE_ORB_FRAC = 1 / 10
# Regenerator recovery fraction.
REGEN_FRAC = 1 / 3
# Rocky Helmet chip fraction.
ROCKY_HELMET_FRAC = 1 / 6

CHOICE_ITEMS = {"choiceband", "choicescarf", "choicespecs"}


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
            # Graceful fallback: keep current candidates rather than hard reset.
            # Foul Play does this — if strict filter yields nothing, return all sets.
            LOGGER.warning(
                "All sets eliminated for %s by %s; keeping current candidates (graceful fallback)",
                self.species, reason,
            )
            self.eliminated_by.append(f"graceful_fallback:{reason}")

    def posterior(self) -> list[dict[str, Any]]:
        return uniform_posterior(self.candidates)


class BeliefStateModule:
    """
    Rule-based posterior over opponent random-battle sets.

    Tracks per-slot beliefs and updates them as observable game events
    constrain possible sets. Mirrors Foul Play's set-elimination system
    but is more comprehensive: includes Life Orb, Choice lock, Regenerator,
    Leftovers, Focus Sash, Rocky Helmet, Speed Boost, Magic Guard, Prankster,
    and Shed Shell in addition to the base rules.
    """

    def __init__(self, pool_path: str | Path = "data/all_gen_pool.json", rng: random.Random | None = None):
        path = Path(pool_path)
        if not path.exists():
            path = Path("data/gen9_random_pool.json")
        self.pool_path = path
        self.pool = self._load_pool(path)
        self.rng = rng or random.Random()
        self.slots: dict[str, SlotBelief] = {}

        # Inter-turn state tracking for passive-effect inferences.
        # Maps slot_key → last observed HP fraction (at start of turn).
        self._hp_prev: dict[str, float] = {}
        # Maps slot_key → list of moves used in order (for Choice lock).
        self._move_history: dict[str, list[str]] = {}
        # Maps slot_key → set of items definitely not held.
        self._impossible_items: dict[str, set[str]] = {}
        # Maps slot_key → True if this slot has confirmed no Choice item.
        self._no_choice: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Pool loading                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
        if not path.exists():
            return {}
        with path.open() as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "species" in data:
            result: dict[str, list[dict[str, Any]]] = {}
            for species_name, entry in (data.get("species") or {}).items():
                if isinstance(entry, dict):
                    all_sets: list[dict[str, Any]] = []
                    for gen_sets in (entry.get("sets") or {}).values():
                        all_sets.extend(gen_sets or [])
                    result[str(species_name)] = all_sets
                elif isinstance(entry, list):
                    result[str(species_name)] = list(entry)
            return result
        return {str(s): list(sets or []) for s, sets in data.items()}

    # ------------------------------------------------------------------ #
    # Slot management                                                      #
    # ------------------------------------------------------------------ #

    def reveal(self, slot: str, species: str) -> None:
        if slot in self.slots:
            return
        candidates = list(self.pool.get(species) or self.pool.get(self._lookup_species(species)) or [])
        self.slots[slot] = SlotBelief(species=species, prior=list(candidates), candidates=list(candidates))

    def _lookup_species(self, species: str) -> str:
        target = normalize_name(species)
        for known in self.pool:
            if normalize_name(known) == target:
                return known
        return species

    # ------------------------------------------------------------------ #
    # Core observation methods (original)                                  #
    # ------------------------------------------------------------------ #

    def observe_move(self, slot: str, move: str) -> None:
        belief = self.slots.get(slot)
        if belief is None:
            return
        belief.seen_moves.add(move)
        belief.update_candidates(
            filter_by_seen_moves(belief.candidates, belief.seen_moves),
            f"move_seen:{move}",
        )

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
            belief.update_candidates(
                filter_status_move_assault_vest(belief.candidates, move),
                f"status_move:{move or 'unknown'}",
            )

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
            belief.update_candidates(
                filter_weather_extension(belief.candidates, weather, turns),
                f"weather_duration:{weather}:{turns}",
            )

    def observe_damage_dealt(self, slot: str, observed_damage: float, damage_range_fn: Any | None = None, *args: Any, **kwargs: Any) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_by_damage_range(belief.candidates, observed_damage, damage_range_fn, *args, **kwargs), "damage_dealt_range")

    def observe_damage_received(self, slot: str, observed_damage: float, damage_range_fn: Any | None = None, *args: Any, **kwargs: Any) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            belief.update_candidates(filter_by_damage_range(belief.candidates, observed_damage, damage_range_fn, *args, **kwargs), "damage_received_range")

    def observe_we_moved_first(self, slot: str, our_speed: int) -> None:
        self.observe_speed_bounds(slot, lower=our_speed)

    def observe_opponent_moved_first(self, slot: str, our_speed: int) -> None:
        self.observe_speed_bounds(slot, upper=our_speed)

    def observe_impossible_item(self, slot: str, item: str) -> None:
        belief = self.slots.get(slot)
        if belief is not None:
            self._impossible_items.setdefault(slot, set()).add(normalize_name(item))
            belief.update_candidates(filter_without_item(belief.candidates, item), f"impossible_item:{item}")

    # ------------------------------------------------------------------ #
    # New inference methods                                                #
    # ------------------------------------------------------------------ #

    def observe_life_orb_recoil(self, slot: str, move_used: str) -> None:
        """Opponent took ~10% HP recoil after attacking with a non-recoil move → Life Orb."""
        move_norm = normalize_name(move_used)
        if move_norm in SELF_RECOIL_MOVES:
            return  # Natural recoil move — can't infer Life Orb.
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Life Orb inferred for %s (move: %s)", slot, move_used)
            belief.update_candidates(filter_by_item(belief.candidates, "Life Orb"), "life_orb_recoil")

    def observe_no_life_orb_recoil(self, slot: str, move_used: str) -> None:
        """Opponent attacked with a typically Life-Orb move but took no recoil → not Life Orb."""
        move_norm = normalize_name(move_used)
        if move_norm in SELF_RECOIL_MOVES:
            return
        self.observe_impossible_item(slot, "Life Orb")

    def observe_leftovers_recovery(self, slot: str, pokemon_types: list[str] | None = None) -> None:
        """~6.25% end-of-turn HP recovery with no healing move → Leftovers or Black Sludge."""
        belief = self.slots.get(slot)
        if belief is None:
            return
        is_poison = any(normalize_name(t) == "poison" for t in (pokemon_types or []))
        if is_poison:
            # Poison types recover from Black Sludge, take damage from Leftovers — but
            # in practice Leftovers is still possible. Prioritize Black Sludge.
            belief.update_candidates(filter_by_item(belief.candidates, "Black Sludge"), "black_sludge_recovery")
        else:
            belief.update_candidates(filter_by_item(belief.candidates, "Leftovers"), "leftovers_recovery")

    def observe_regenerator_recovery(self, slot: str, hp_before_switch: float, hp_after_switch_in: float) -> None:
        """Pokémon regained ~33% HP on switch → Regenerator ability."""
        recovery = hp_after_switch_in - hp_before_switch
        if recovery >= REGEN_FRAC - HP_TOL:
            belief = self.slots.get(slot)
            if belief is not None:
                LOGGER.info("Regenerator inferred for %s (recovery %.0f%%)", slot, recovery * 100)
                belief.update_candidates(filter_by_ability(belief.candidates, "Regenerator"), "regenerator_recovery")

    def observe_focus_sash_survival(self, slot: str) -> None:
        """Pokémon survived at exactly 1 HP from what should be an OHKO → Focus Sash."""
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Focus Sash inferred for %s", slot)
            belief.update_candidates(filter_by_item(belief.candidates, "Focus Sash"), "focus_sash_survival")

    def observe_rocky_helmet_chip(self, slot: str) -> None:
        """We took ~16.7% HP chip from a contact move → opponent has Rocky Helmet."""
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Rocky Helmet inferred for %s", slot)
            belief.update_candidates(filter_by_item(belief.candidates, "Rocky Helmet"), "rocky_helmet_chip")

    def observe_choice_move(self, slot: str, move_used: str) -> None:
        """Track move history. Same move 3 turns → could be Choice. Two different moves → no Choice."""
        history = self._move_history.setdefault(slot, [])
        move_norm = normalize_name(move_used)
        history.append(move_norm)
        # Two different moves used → definitely no Choice item.
        if len(history) >= 2 and len(set(history[-2:])) > 1 and not self._no_choice.get(slot):
            self._no_choice[slot] = True
            LOGGER.info("No Choice item for %s (used multiple moves)", slot)
            for item in CHOICE_ITEMS:
                self.observe_impossible_item(slot, item)
        # Only one unique move used 2+ times + previously had choice possibility → could be Choice.
        # We don't set item yet (might be Encore/sleep) but don't eliminate it either.

    def observe_choice_item_confirmed(self, slot: str, pokemon_category: str | None = None) -> None:
        """Opponent is locked into a move (Choice confirmed). Narrow by category if known."""
        belief = self.slots.get(slot)
        if belief is None or self._no_choice.get(slot):
            return
        # Narrow to Choice items. If we know attacker category, can narrow further.
        cat = normalize_name(pokemon_category or "")
        if cat == "physical":
            belief.update_candidates(filter_by_item(belief.candidates, "Choice Band"), "choice_band_confirmed")
        elif cat == "special":
            belief.update_candidates(filter_by_item(belief.candidates, "Choice Specs"), "choice_specs_confirmed")
        # Can't distinguish Band/Specs/Scarf without speed info — leave as-is.

    def observe_speed_boost_ability(self, slot: str) -> None:
        """Opponent is faster this turn than last with no weather/item explanation → Speed Boost."""
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Speed Boost inferred for %s", slot)
            belief.update_candidates(filter_by_ability(belief.candidates, "Speed Boost"), "speed_boost_ability")

    def observe_magic_guard_passive(self, slot: str, expected_chip_source: str) -> None:
        """Opponent should have taken passive chip (burn/poison/weather) but didn't → Magic Guard.

        expected_chip_source: 'burn', 'poison', 'weather', 'lifeorb'
        """
        belief = self.slots.get(slot)
        if belief is None:
            return
        source = normalize_name(expected_chip_source)
        LOGGER.info("Magic Guard inferred for %s (no %s chip)", slot, source)
        belief.update_candidates(filter_by_ability(belief.candidates, "Magic Guard"), f"magic_guard_{source}")

    def observe_prankster_priority(self, slot: str, move_used: str) -> None:
        """Status move went before our non-priority move → Prankster."""
        move_norm = normalize_name(move_used)
        if move_norm not in PRANKSTER_ELIGIBLES:
            return
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Prankster inferred for %s (move: %s)", slot, move_used)
            belief.update_candidates(filter_by_ability(belief.candidates, "Prankster"), "prankster_priority")

    def observe_shed_shell_escape(self, slot: str) -> None:
        """Opponent switched out of a trapping ability → Shed Shell."""
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Shed Shell inferred for %s", slot)
            belief.update_candidates(filter_by_item(belief.candidates, "Shed Shell"), "shed_shell_escape")

    def observe_eviolite_suspect(self, slot: str, species: str) -> None:
        """NFE Pokémon took less damage than expected → likely Eviolite."""
        if normalize_name(species) not in EVIOLITE_CANDIDATES:
            return
        belief = self.slots.get(slot)
        if belief is not None:
            LOGGER.info("Eviolite suspected for %s (%s)", slot, species)
            belief.update_candidates(filter_by_item(belief.candidates, "Eviolite"), "eviolite_suspect")

    def infer_choice_scarf(
        self,
        slot: str,
        opponent_speed: int,
        our_speed: int,
        trick_room: bool = False,
        can_have_choice_item: bool = True,
    ) -> None:
        """Infer Choice Scarf from speed-order observation (mirrors Foul Play's check_choicescarf).

        In random battles spread is always serious/85 EVs so speed is deterministic.
        If opponent moved first but their base speed × 1.5 > our speed → scarf.
        """
        belief = self.slots.get(slot)
        if belief is None or not can_have_choice_item or self._no_choice.get(slot):
            return
        if trick_room:
            return  # Trick room reverses speed; scarf logic doesn't apply cleanly.
        # Check if scarf would explain opponent going first.
        scarf_speed = int(opponent_speed * 1.5)
        if our_speed > opponent_speed and our_speed <= scarf_speed:
            LOGGER.info("Choice Scarf inferred for %s: our_spd=%d opp_spd=%d", slot, our_speed, opponent_speed)
            belief.update_candidates(filter_by_item(belief.candidates, "Choice Scarf"), "inferred_choice_scarf")

    def record_hp(self, slot: str, hp_fraction: float) -> None:
        """Record current HP for this slot so next turn's delta can be computed."""
        self._hp_prev[slot] = float(hp_fraction)

    def check_end_of_turn_hp_change(self, slot: str, hp_current: float, pokemon_types: list[str] | None = None) -> None:
        """Compare current HP to previous to detect Leftovers/Black Sludge or Life Orb.

        Call this at the start of each new turn with the Pokémon's HP before the turn.
        """
        prev = self._hp_prev.get(slot)
        if prev is None:
            self._hp_prev[slot] = hp_current
            return
        delta = hp_current - prev
        if delta > HP_TOL:
            # HP increased passively → Leftovers or Black Sludge
            if abs(delta - LEFTOVERS_FRAC) < HP_TOL:
                self.observe_leftovers_recovery(slot, pokemon_types)
        self._hp_prev[slot] = hp_current

    # ------------------------------------------------------------------ #
    # RLM posterior merging                                                #
    # ------------------------------------------------------------------ #

    def refine(self, refined_belief: dict[str, Any]) -> None:
        for slot, posterior in refined_belief.items():
            if slot not in self.slots or not isinstance(posterior, list):
                continue
            candidates: list[dict[str, Any]] = [entry for entry in posterior if isinstance(entry, dict)]
            if candidates:
                self.slots[slot].update_candidates(candidates, "rlm_refinement")

    # ------------------------------------------------------------------ #
    # Posterior / sampling                                                 #
    # ------------------------------------------------------------------ #

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
                chosen = self.rng.choices(posterior, weights=weights, k=1)[0] if total > 0 else self.rng.choice(posterior)
                config[slot] = chosen
            configs.append(config)
        return configs

    # ------------------------------------------------------------------ #
    # poke-env Battle ingestion                                            #
    # ------------------------------------------------------------------ #

    def update(self, battle: Any) -> None:
        """Update belief state from a poke-env Battle object.

        Processes all visible opponent information: revealed moves, items,
        abilities, and triggers passive-effect inferences where possible.
        """
        opponent_team = getattr(battle, "opponent_team", None) or {}
        items_fn = getattr(opponent_team, "items", None)
        if callable(items_fn):
            team_items = items_fn()
        elif isinstance(opponent_team, dict):
            team_items = opponent_team.items()
        else:
            team_items = []

        for key, pokemon in team_items:
            slot = str(key)
            species = getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None)
            if species:
                self.reveal(slot, str(species))

            # Track HP for end-of-turn passive effect detection.
            hp_frac = getattr(pokemon, "current_hp_fraction", None)
            if hp_frac is not None:
                self.check_end_of_turn_hp_change(slot, float(hp_frac))

            # Observed moves.
            moves = getattr(pokemon, "moves", None) or {}
            iterable = moves.values() if isinstance(moves, dict) else (moves or [])
            for move in iterable:
                move_name = getattr(move, "id", None) or getattr(move, "name", None) or str(move)
                move_str = str(move_name)
                self.observe_move(slot, move_str)
                self.observe_choice_move(slot, move_str)
                category = normalize_name(getattr(getattr(move, "category", None), "name", None) or getattr(move, "category", None))
                if category == "status":
                    self.observe_status_move(slot, move_str)

            # Revealed item / ability.
            item = getattr(pokemon, "item", None)
            if item:
                self.observe_item(slot, str(item))
            ability = getattr(pokemon, "ability", None)
            if ability:
                self.observe_ability(slot, str(ability))
