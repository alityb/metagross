"""Pure deterministic compilation of public events into Randbats set weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from srcs.metagross.public_history import (
    AbilityEvent,
    HazardAvoidanceEvent,
    ItemEvent,
    MoveEvent,
    PublicEvent,
    SwitchEvent,
    TeraEvent,
    normalize_id,
)


@dataclass(frozen=True)
class CandidateWeight:
    candidate_id: str
    weight: float
    count: int
    level: int | None
    ability: str
    item: str
    moves: tuple[str, ...]
    tera_type: str | None


@dataclass(frozen=True)
class SpeciesBelief:
    species: str
    status: str
    reason: str | None
    candidates: tuple[CandidateWeight, ...]


@dataclass(frozen=True)
class CompiledBelief:
    schema: str
    species: tuple[SpeciesBelief, ...]


def candidate_fields(candidate: Any) -> dict[str, Any]:
    pokemon_set = candidate.pkmn_set
    return {
        "level": pokemon_set.level,
        "ability": normalize_id(pokemon_set.ability),
        "item": normalize_id(pokemon_set.item),
        "moves": tuple(sorted(normalize_id(move) for move in candidate.pkmn_moveset.moves)),
        "tera_type": normalize_id(pokemon_set.tera_type) or None,
        "count": pokemon_set.count,
    }


def candidate_id(species: str, fields: Mapping[str, Any]) -> str:
    identity = {"species": species, **{key: fields[key] for key in sorted(fields) if key != "count"}}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def compile_public_belief(
    events: Sequence[PublicEvent],
    candidate_sets: Mapping[str, Sequence[Any]],
) -> CompiledBelief:
    """Filter immutable candidate data using only direct opponent public facts."""
    if any(event.sequence != index for index, event in enumerate(events)):
        raise ValueError("public event sequence is not contiguous")
    facts: dict[str, dict[str, Any]] = {}
    unsupported: set[str] = set()
    for event in events:
        if event.actor != "opponent":
            continue
        species = event.species
        if isinstance(event, SwitchEvent):
            species = event.species
            row = facts.setdefault(
                species,
                {"level": None, "moves": set(), "abilities": set(), "items": set(), "teras": set(), "avoided_stealth_rock": False, "rock_damage": []},
            )
            if event.replaced_identity:
                unsupported.add(species)
                if event.previous_species is not None:
                    unsupported.add(event.previous_species)
            if event.level is not None:
                if row["level"] not in {None, event.level}:
                    unsupported.add(species)
                row["level"] = event.level
        if species is None:
            continue
        row = facts.setdefault(
            species,
            {"level": None, "moves": set(), "abilities": set(), "items": set(), "teras": set(), "avoided_stealth_rock": False, "rock_damage": []},
        )
        if isinstance(event, MoveEvent):
            row["moves"].add(event.move_id)
        elif isinstance(event, AbilityEvent):
            if event.stable_identity_evidence:
                row["abilities"].add(event.ability_id)
        elif isinstance(event, ItemEvent):
            if event.stable_identity_evidence:
                row["items"].add(event.item_id)
        elif isinstance(event, TeraEvent):
            row["teras"].add(event.tera_type)
        elif isinstance(event, HazardAvoidanceEvent):
            if event.hazard_id == "stealthrock" and event.avoided:
                row["avoided_stealth_rock"] = True
            elif event.hazard_id == "stealthrock":
                row["rock_damage"].append(event)

    compiled = []
    normalized_sets = {normalize_id(species): sets for species, sets in candidate_sets.items()}
    for species in sorted(facts):
        fact = facts[species]
        if species in unsupported or any(
            len(fact[field]) > 1 for field in ("abilities", "items", "teras")
        ):
            compiled.append(SpeciesBelief(species, "unsupported", "ambiguous_public_identity", ()))
            continue
        matches = []
        for candidate in normalized_sets.get(species, ()):
            fields = candidate_fields(candidate)
            if fact["level"] is not None and fields["level"] != fact["level"]:
                continue
            if not fact["moves"].issubset(fields["moves"]):
                continue
            if fact["abilities"] and fields["ability"] not in fact["abilities"]:
                continue
            if fact["items"] and fields["item"] not in fact["items"]:
                continue
            if fact["teras"] and fields["tera_type"] not in fact["teras"]:
                continue
            if fact["avoided_stealth_rock"] and not (
                fields["item"] == "heavydutyboots" or fields["ability"] == "magicguard"
            ):
                continue
            if any(
                (not event.item_effects_suppressed and fields["item"] == "heavydutyboots" and fields["ability"] != "klutz")
                or (not event.ability_effects_suppressed and fields["ability"] == "magicguard")
                for event in fact["rock_damage"]
            ):
                continue
            count = fields["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("candidate count must be a positive integer")
            matches.append((candidate, fields))
        if not matches:
            compiled.append(SpeciesBelief(species, "inconsistent", "empty_candidate_support", ()))
            continue
        total = math.fsum(fields["count"] for _candidate, fields in matches)
        weights = tuple(
            CandidateWeight(
                candidate_id=candidate_id(species, fields),
                weight=fields["count"] / total,
                count=fields["count"],
                level=fields["level"],
                ability=fields["ability"],
                item=fields["item"],
                moves=fields["moves"],
                tera_type=fields["tera_type"],
            )
            for _candidate, fields in sorted(
                matches, key=lambda pair: candidate_id(species, pair[1])
            )
        )
        compiled.append(SpeciesBelief(species, "compiled", None, weights))
    return CompiledBelief("metagross-history-belief/v1", tuple(compiled))
