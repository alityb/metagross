"""Compile causal public history into the engine's packed reveal-mask contract."""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any


SPECIES_OFFSET = 0
MOVES_OFFSET = 6
ITEMS_OFFSET = 30
ABILITIES_OFFSET = 36
VALID_MASK = (1 << 42) - 1


@dataclass(frozen=True)
class ReplayRevealFacts:
    """Public opponent facts visible at one Showdown decision boundary."""

    species: frozenset[str]
    moves: tuple[tuple[str, tuple[str, ...]], ...]
    items: frozenset[str]
    abilities: frozenset[str]


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _species_slots(state: Any) -> dict[str, int]:
    slots: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, pokemon in enumerate(state.side_two.pokemon):
        species = _norm(getattr(pokemon, "id", ""))
        if not species or species == "none":
            continue
        if species in slots:
            ambiguous.add(species)
        slots[species] = index
    for species in ambiguous:
        slots.pop(species, None)
    return slots


def _species_bit(index: int) -> int:
    return 1 << (SPECIES_OFFSET + index)


def _move_bit(index: int, move_index: int) -> int:
    return 1 << (MOVES_OFFSET + index * 4 + move_index)


def _item_bit(index: int) -> int:
    return 1 << (ITEMS_OFFSET + index)


def _ability_bit(index: int) -> int:
    return 1 << (ABILITIES_OFFSET + index)


def _matching_move_bits(state: Any, index: int, move_ids: Iterable[str]) -> int:
    revealed = {_norm(move) for move in move_ids}
    bits = 0
    for move_index, move in enumerate(state.side_two.pokemon[index].moves):
        if _norm(getattr(move, "id", "")) in revealed:
            bits |= _move_bit(index, move_index)
    return bits


def from_live_belief_tracker(state: Any, tracker: Any) -> int:
    """Compile the live opponent tracker without inspecting unrevealed fields."""
    slots = _species_slots(state)
    bits = 0
    for species, belief in getattr(tracker, "_opponent_mons", {}).items():
        index = slots.get(_norm(species))
        if index is None:
            continue
        bits |= _species_bit(index)
        bits |= _matching_move_bits(state, index, belief.revealed_moves)
        if belief.revealed_item:
            bits |= _item_bit(index)
        if belief.revealed_ability:
            bits |= _ability_bit(index)
    return bits & VALID_MASK


def from_transformer_tracker(state: Any, tracker: Any) -> int:
    """Compile the exact public team used to construct r1's root observation."""
    slots = _species_slots(state)
    bits = 0
    for record in getattr(tracker, "opponent_team", ()):
        pokemon = record.pokemon
        index = slots.get(_norm(getattr(pokemon, "name", "")))
        if index is None:
            continue
        bits |= _species_bit(index)
        moves = (
            getattr(move, "name", "")
            for move in getattr(pokemon, "moves", ())
            if _norm(getattr(move, "name", "")) not in {"", "nomove", "none"}
        )
        bits |= _matching_move_bits(state, index, moves)
        if getattr(record, "revealed_item", None):
            bits |= _item_bit(index)
        ability = _norm(getattr(pokemon, "ability", ""))
        if ability not in {"", "none", "unknownability"}:
            bits |= _ability_bit(index)
    return bits & VALID_MASK


def from_public_events(state: Any, events: Iterable[Any]) -> int:
    """Compile typed `PublicEventLedger` events against one completed world.

    Completed move/item/ability values are used only to locate the concrete
    slot named by an already-public event; their mere presence never sets a bit.
    Ambiguous identities and events that do not match the sampled world are
    conservatively ignored.
    """
    slots = _species_slots(state)
    bits = 0
    for event in events:
        if getattr(event, "actor", None) != "opponent":
            continue
        species = _norm(getattr(event, "species", ""))
        index = slots.get(species)
        if index is None:
            continue
        kind = getattr(event, "kind", "")
        if kind == "switch":
            bits |= _species_bit(index)
        elif kind == "move":
            bits |= _matching_move_bits(state, index, (getattr(event, "move_id", ""),))
        elif kind == "item" and getattr(event, "stable_identity_evidence", False):
            bits |= _item_bit(index)
        elif kind == "ability" and getattr(event, "stable_identity_evidence", False):
            bits |= _ability_bit(index)
    return bits & VALID_MASK


def _side_from_ident(value: str) -> str:
    match = re.match(r"^(p[12])", value or "")
    return match.group(1) if match else ""


def _species_from_details(value: str) -> str:
    return _norm((value or "").split(",", 1)[0])


def replay_reveal_snapshots(log: str, observer_name: str) -> dict[int, ReplayRevealFacts]:
    """Reconstruct conservative start-of-turn opponent reveals from a public log.

    The replay contains no private requests, so forced-switch decisions within a
    turn reuse that turn's start snapshot. This can undercount newly revealed
    facts, but it cannot reveal a hidden field early.
    """
    player_sides: dict[str, str] = {}
    active: dict[str, str] = {"p1": "", "p2": ""}
    species: dict[str, set[str]] = {"p1": set(), "p2": set()}
    moves: dict[str, dict[str, set[str]]] = {"p1": {}, "p2": {}}
    items: dict[str, set[str]] = {"p1": set(), "p2": set()}
    abilities: dict[str, set[str]] = {"p1": set(), "p2": set()}
    snapshots: dict[int, ReplayRevealFacts] = {}

    def actor_species(ident: str) -> tuple[str, str]:
        side = _side_from_ident(ident)
        return side, active.get(side, "")

    def reveal_tagged_source(parts: list[str]) -> None:
        if len(parts) < 3:
            return
        side, mon = actor_species(parts[2])
        if not side or not mon:
            return
        for value in parts[3:]:
            normalized = value.strip().lower()
            if normalized.startswith("[from] item:") or normalized.startswith("item:"):
                items[side].add(mon)
            elif normalized.startswith("[from] ability:") or normalized.startswith("ability:"):
                abilities[side].add(mon)

    for line in str(log or "").splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "player" and len(parts) >= 4 and parts[2] in {"p1", "p2"}:
            player_sides[parts[3]] = parts[2]
        elif tag in {"switch", "drag", "replace", "detailschange"} and len(parts) >= 4:
            side = _side_from_ident(parts[2])
            mon = _species_from_details(parts[3])
            if side and mon:
                active[side] = mon
                species[side].add(mon)
        elif tag == "move" and len(parts) >= 4:
            side, mon = actor_species(parts[2])
            move = _norm(parts[3])
            if side and mon and move:
                moves[side].setdefault(mon, set()).add(move)
        elif tag in {"-item", "-enditem"} and len(parts) >= 3:
            side, mon = actor_species(parts[2])
            if side and mon:
                items[side].add(mon)
        elif tag == "-ability" and len(parts) >= 3:
            side, mon = actor_species(parts[2])
            if side and mon:
                abilities[side].add(mon)

        reveal_tagged_source(parts)
        if tag == "turn" and len(parts) >= 3:
            try:
                turn = int(parts[2])
            except ValueError:
                continue
            observer_side = player_sides.get(observer_name, "")
            opponent_side = "p2" if observer_side == "p1" else "p1" if observer_side == "p2" else ""
            if not opponent_side:
                continue
            snapshots[turn] = ReplayRevealFacts(
                species=frozenset(species[opponent_side]),
                moves=tuple(
                    (mon, tuple(sorted(revealed)))
                    for mon, revealed in sorted(moves[opponent_side].items())
                ),
                items=frozenset(items[opponent_side]),
                abilities=frozenset(abilities[opponent_side]),
            )
    return snapshots


def from_replay_facts(state: Any, facts: ReplayRevealFacts) -> int:
    """Locate public replay facts in a completed world without reading extras."""
    slots = _species_slots(state)
    bits = 0
    moves = dict(facts.moves)
    for species in facts.species:
        index = slots.get(_norm(species))
        if index is not None:
            bits |= _species_bit(index)
    for species, move_ids in moves.items():
        index = slots.get(_norm(species))
        if index is not None:
            bits |= _matching_move_bits(state, index, move_ids)
    for species in facts.items:
        index = slots.get(_norm(species))
        if index is not None:
            bits |= _item_bit(index)
    for species in facts.abilities:
        index = slots.get(_norm(species))
        if index is not None:
            bits |= _ability_bit(index)
    return bits & VALID_MASK


def information_fractions(bits: int) -> tuple[float, float, float, float]:
    bits &= VALID_MASK
    return (
        (bits & 0x3F).bit_count() / 6.0,
        ((bits >> MOVES_OFFSET) & 0xFFFFFF).bit_count() / 24.0,
        ((bits >> ITEMS_OFFSET) & 0x3F).bit_count() / 6.0,
        ((bits >> ABILITIES_OFFSET) & 0x3F).bit_count() / 6.0,
    )
