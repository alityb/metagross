"""Leak-free, stateless player-information projection for interior search.

The exact engine state remains private mechanical authority.  This module emits
only own-private fields, shared public mechanics, and opponent fields authorized
by the engine's causal reveal mask.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "metagross-public-search-state/v1"
VALID_REVEAL_BITS = (1 << 42) - 1


class PublicSearchStateError(ValueError):
    pass


def _norm(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _enum(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _known(value: Any, *unknown: str) -> bool:
    return _norm(value) not in {"", "none", *unknown}


def _reveal_bit(slot: int, field: str, move_slot: int | None = None) -> int:
    if not 0 <= slot < 6:
        raise PublicSearchStateError("invalid Pokemon slot")
    if field == "species":
        offset = slot
    elif field == "move" and move_slot is not None and 0 <= move_slot < 4:
        offset = 6 + slot * 4 + move_slot
    elif field == "item":
        offset = 30 + slot
    elif field == "ability":
        offset = 36 + slot
    else:
        raise PublicSearchStateError("invalid reveal field")
    return 1 << offset


def compile_side_one_reveal_mask(
    state: Any, player_information_state: Mapping[str, Any]
) -> int:
    """Compile the causal snapshot ledger into engine slot mask bits.

    Slot lookup is permitted only to attach already-public facts to simulator
    slots.  No fact is inferred from the completed state.
    """
    rows = player_information_state.get("opponent_public_team")
    if not isinstance(rows, list) or not rows:
        raise PublicSearchStateError("missing causal opponent-public ledger")
    actual = list(state.side_two.pokemon)
    bits = 0
    claimed: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("pokemon"), Mapping):
            raise PublicSearchStateError("invalid public opponent row")
        public = row["pokemon"]
        species = _norm(public.get("name"))
        if not species or species == "none":
            raise PublicSearchStateError("public row lacks species")
        matches = [
            index for index, pokemon in enumerate(actual) if _norm(pokemon.id) == species
        ]
        if len(matches) != 1 or matches[0] in claimed:
            raise PublicSearchStateError("ambiguous public species-to-slot mapping")
        slot = matches[0]
        claimed.add(slot)
        bits |= _reveal_bit(slot, "species")

        moves = public.get("moves", [])
        if not isinstance(moves, list):
            raise PublicSearchStateError("invalid public move ledger")
        actual_moves = list(actual[slot].moves)
        for public_move in moves:
            if not isinstance(public_move, Mapping):
                raise PublicSearchStateError("invalid public move row")
            name = _norm(public_move.get("name"))
            matches = [
                index for index, move in enumerate(actual_moves) if _norm(move.id) == name
            ]
            if not name or len(matches) != 1:
                raise PublicSearchStateError("public move cannot be mapped uniquely")
            bits |= _reveal_bit(slot, "move", matches[0])

        item = public.get("item")
        if _known(item, "unknownitem"):
            if _norm(actual[slot].item) != _norm(item):
                raise PublicSearchStateError("public item disagrees with simulator")
            bits |= _reveal_bit(slot, "item")
        ability = public.get("ability")
        if _known(ability, "unknownability"):
            if _norm(actual[slot].ability) != _norm(ability):
                raise PublicSearchStateError("public ability disagrees with simulator")
            bits |= _reveal_bit(slot, "ability")
    if bits == 0 or bits & ~VALID_REVEAL_BITS:
        raise PublicSearchStateError("invalid compiled reveal mask")
    return bits


def install_side_one_reveal_mask(state: Any, bits: int) -> Any:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0 or bits & ~VALID_REVEAL_BITS:
        raise PublicSearchStateError("causal reveal mask is required")
    return state.with_side_one_public_reveals(bits)


def _conditions(side: Any) -> dict[str, int]:
    names = (
        "aurora_veil", "light_screen", "reflect", "safeguard", "mist",
        "spikes", "stealth_rock", "sticky_web", "tailwind", "toxic_spikes",
    )
    return {name: int(getattr(side.side_conditions, name, 0)) for name in names}


def _public_side(side: Any) -> dict[str, Any]:
    return {
        "active_slot": int(side.active_index),
        "boosts": [
            int(getattr(side, name))
            for name in (
                "attack_boost", "defense_boost", "special_attack_boost",
                "special_defense_boost", "speed_boost", "accuracy_boost",
                "evasion_boost",
            )
        ],
        "conditions": _conditions(side),
        "force_switch": bool(side.force_switch),
        "force_trapped": bool(side.force_trapped),
        "substitute_health": int(side.substitute_health),
        "volatile_statuses": sorted(_enum(item) for item in side.volatile_statuses),
    }


def _own_pokemon(pokemon: Any) -> dict[str, Any]:
    return {
        "ability": _enum(pokemon.ability),
        "base_ability": _enum(pokemon.base_ability),
        "base_types": [_enum(item) for item in pokemon.base_types],
        "evs": [int(item) for item in pokemon.evs],
        "hp": int(pokemon.hp),
        "id": _enum(pokemon.id),
        "item": _enum(pokemon.item),
        "level": int(pokemon.level),
        "maxhp": int(pokemon.maxhp),
        "moves": [
            {"disabled": bool(move.disabled), "id": _enum(move.id), "pp": int(move.pp)}
            for move in pokemon.moves
        ],
        "nature": _enum(pokemon.nature),
        "stats": [
            int(pokemon.attack), int(pokemon.defense), int(pokemon.special_attack),
            int(pokemon.special_defense), int(pokemon.speed),
        ],
        "status": _enum(pokemon.status),
        "tera_type": _enum(pokemon.tera_type),
        "terastallized": bool(pokemon.terastallized),
        "types": [_enum(item) for item in pokemon.types],
    }


def _opponent_pokemon(pokemon: Any, slot: int, bits: int) -> dict[str, Any] | None:
    if not bits & _reveal_bit(slot, "species"):
        return None
    moves = [
        _enum(move.id)
        for move_slot, move in enumerate(pokemon.moves)
        if bits & _reveal_bit(slot, "move", move_slot)
    ]
    return {
        "ability": _enum(pokemon.ability) if bits & _reveal_bit(slot, "ability") else None,
        "hp": int(pokemon.hp),
        "id": _enum(pokemon.id),
        "item": _enum(pokemon.item) if bits & _reveal_bit(slot, "item") else None,
        "level": int(pokemon.level),
        "maxhp": int(pokemon.maxhp),
        "moves": moves,
        "status": _enum(pokemon.status),
        "tera_type": _enum(pokemon.tera_type) if pokemon.terastallized else None,
        "terastallized": bool(pokemon.terastallized),
        "types": [_enum(item) for item in pokemon.types],
    }


def canonical_action_table(actions: Sequence[str]) -> dict[str, Any]:
    canonical = sorted({_norm(action): str(action).lower() for action in actions}.values())
    if len(canonical) != len(actions):
        raise PublicSearchStateError("duplicate canonical action")
    automatic = len(actions) == 1 and _norm(actions[0]) == "nomove"
    if automatic:
        return {
            "automatic_action": "nomove",
            "entries": [None] * 13,
            "illegal_actions": [True] * 13,
            "name_table": {},
        }
    move_actions = [action for action in actions if not _norm(action).startswith("switch")]
    switch_actions = [action for action in actions if _norm(action).startswith("switch")]
    base_moves = sorted(
        {
            action[:-5] if action.endswith("-tera") else action
            for action in move_actions
        }
    )
    switches = sorted(switch_actions, key=_norm)
    if len(base_moves) > 4 or len(switches) > 5:
        raise PublicSearchStateError("action set exceeds canonical 13 slots")
    legal = set(actions)
    entries: list[dict[str, Any] | None] = [None] * 13
    name_table: dict[str, int] = {}
    for index, move in enumerate(base_moves):
        for name, target in ((move, index), (move + "-tera", index + 9)):
            if name in legal:
                entries[target] = {
                    "kind": "move", "name": name, "switch_target": None,
                    "tera": name.endswith("-tera"),
                }
                name_table[name] = target
    for offset, name in enumerate(switches):
        target = 4 + offset
        entries[target] = {
            "kind": "switch", "name": name,
            "switch_target": name.removeprefix("switch "), "tera": False,
        }
        name_table[name] = target
    if set(name_table) != legal:
        raise PublicSearchStateError("canonical action mapping lost a legal action")
    return {
        "automatic_action": None,
        "entries": entries,
        "illegal_actions": [entry is None for entry in entries],
        "name_table": name_table,
    }


def extract_public_search_state(
    state: Any,
    engine: Any,
    *,
    observer: str = "side_one",
) -> dict[str, Any]:
    if observer not in {"side_one", "side_two"}:
        raise PublicSearchStateError("observer must be side_one or side_two")
    own = state.side_one if observer == "side_one" else state.side_two
    opponent = state.side_two if observer == "side_one" else state.side_one
    bits = state.s1_public_reveals if observer == "side_one" else state.s2_public_reveals
    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
        raise PublicSearchStateError("legacy/empty reveal metadata fails closed")
    first, second = engine.root_options(state)
    actions = list(first if observer == "side_one" else second)
    table = canonical_action_table(actions)
    fractions = [
        (bits & 0x3F).bit_count() / 6.0,
        ((bits >> 6) & 0xFFFFFF).bit_count() / 24.0,
        ((bits >> 30) & 0x3F).bit_count() / 6.0,
        ((bits >> 36) & 0x3F).bit_count() / 6.0,
    ]
    result = {
        "action_table": table,
        "belief_summary": {"causal_reveal_fractions": fractions},
        "field": {
            "terrain": _enum(state.terrain),
            "terrain_turns": int(state.terrain_turns_remaining),
            "team_preview": bool(state.team_preview),
            "trick_room": bool(state.trick_room),
            "trick_room_turns": int(state.trick_room_turns_remaining),
            "weather": _enum(state.weather),
            "weather_turns": int(state.weather_turns_remaining),
        },
        "opponent": {
            "pokemon": [
                _opponent_pokemon(pokemon, slot, bits)
                for slot, pokemon in enumerate(opponent.pokemon)
            ],
            "side": _public_side(opponent),
        },
        "own": {
            "pokemon": [_own_pokemon(pokemon) for pokemon in own.pokemon],
            "side": _public_side(own),
        },
        "perspective": "acting_player",
        "schema": SCHEMA,
    }
    canonical_bytes(result)
    return result


def state_fingerprint(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(state)).hexdigest()


def validate_finite_policy(probabilities: Sequence[float], illegal: Sequence[bool]) -> None:
    if (
        len(probabilities) != 13
        or len(illegal) != 13
        or any(not math.isfinite(float(value)) or float(value) < 0 for value in probabilities)
        or any(bool(blocked) and float(value) != 0.0 for value, blocked in zip(probabilities, illegal))
        or not math.isclose(math.fsum(float(value) for value in probabilities), 1.0, abs_tol=1e-6)
    ):
        raise PublicSearchStateError("invalid legal policy distribution")
