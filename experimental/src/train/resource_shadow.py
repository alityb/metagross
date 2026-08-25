"""Leak-free, interpretable resource features for a long-horizon expert.

The feature vector is deliberately small and oriented: except for the final
two context terms, larger values always describe a state that should be no
worse for side one.  A trainer can therefore constrain the corresponding
shadow prices to be non-negative instead of learning an opaque scalar value.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from belief.public_reveal_mask import information_fractions


SCHEMA = "metagross-resource-shadow/v1"
FEATURE_NAMES = (
    "own_team_hp",
    "own_alive",
    "own_active_hp",
    "own_bench_hp",
    "own_switch_depth",
    "own_tera_available",
    "own_pp_reserve",
    "own_move_availability",
    "opponent_active_hp_deficit",
    "opponent_fainted",
    "opponent_status_pressure",
    "opponent_tera_spent",
    "boost_advantage",
    "screen_advantage",
    "hazard_advantage",
    "substitute_advantage",
    "opponent_team_revealed",
    "opponent_moves_revealed",
    "opponent_items_revealed",
    "opponent_abilities_revealed",
    "own_item_reserve",
    "turn_progress_context",
    "trick_room_context",
)
RESOURCE_FEATURE_COUNT = 21
FEATURE_COUNT = len(FEATURE_NAMES)


def _name(value: Any) -> str:
    return str(value).upper().replace("_", "")


def _known_pokemon(pokemon: Any) -> bool:
    return _name(getattr(pokemon, "id", "NONE")) not in {"", "NONE"}


def _known_move(move: Any) -> bool:
    return _name(getattr(move, "id", "NONE")) not in {"", "NONE"}


def _known_item(pokemon: Any) -> bool:
    return _name(getattr(pokemon, "item", "NONE")) not in {
        "",
        "NONE",
        "UNKNOWNITEM",
    }


def _known_ability(pokemon: Any) -> bool:
    return _name(getattr(pokemon, "ability", "NONE")) not in {"", "NONE"}


def _fraction(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if denominator <= 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def _pokemon(side: Any) -> list[Any]:
    try:
        return list(side.pokemon)
    except (AttributeError, TypeError) as exc:
        raise ValueError("state side does not expose a Pokemon collection") from exc


def _active(side: Any, pokemon: Sequence[Any]) -> Any:
    try:
        return pokemon[int(side.active_index)]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("state side does not expose a valid active Pokemon") from exc


def _hp(pokemon: Any) -> float:
    if not _known_pokemon(pokemon):
        return 0.0
    return _fraction(getattr(pokemon, "hp", 0), getattr(pokemon, "maxhp", 0))


def _used_tera(side: Any) -> bool:
    return any(
        _known_pokemon(pokemon) and bool(getattr(pokemon, "terastallized", False))
        for pokemon in _pokemon(side)
    )


def _tera_available(state: Any, side_name: str, side: Any) -> float:
    explicit = getattr(state, "s1_can_tera" if side_name == "side_one" else "s2_can_tera", None)
    allowed = True if explicit is None else bool(explicit)
    return float(allowed and not _used_tera(side))


def _condition(side: Any, name: str) -> float:
    conditions = getattr(side, "side_conditions", None)
    return float(getattr(conditions, name, 0) or 0)


def _screen(side: Any) -> float:
    return min(
        1.0,
        (
            _condition(side, "reflect")
            + _condition(side, "light_screen")
            + 2.0 * _condition(side, "aurora_veil")
        )
        / 8.0,
    )


def _hazard(side: Any) -> float:
    return min(
        1.0,
        (
            _condition(side, "stealth_rock")
            + _condition(side, "spikes")
            + _condition(side, "toxic_spikes")
            + 2.0 * _condition(side, "sticky_web")
        )
        / 8.0,
    )


def _boost(side: Any) -> float:
    names = (
        "attack_boost",
        "defense_boost",
        "special_attack_boost",
        "special_defense_boost",
        "speed_boost",
    )
    total = sum(float(getattr(side, name, 0) or 0) for name in names)
    return min(1.0, max(-1.0, total / 30.0))


def _statused(pokemon: Any) -> bool:
    return _name(getattr(pokemon, "status", "NONE")) not in {"", "NONE"}


def _pp_features(own: Sequence[Any]) -> tuple[float, float]:
    moves = [
        move
        for pokemon in own
        if _known_pokemon(pokemon) and float(getattr(pokemon, "hp", 0) or 0) > 0
        for move in list(getattr(pokemon, "moves", ()))
        if _known_move(move)
    ]
    if not moves:
        return 0.0, 0.0
    # Max PP is not exposed by the Python state contract. log1p(pp)/log(65)
    # is a bounded monotone reserve proxy that does not invent move metadata.
    reserve = sum(
        math.log1p(min(64.0, max(0.0, float(getattr(move, "pp", 0) or 0))))
        / math.log(65.0)
        for move in moves
    ) / len(moves)
    available = sum(
        float(
            float(getattr(move, "pp", 0) or 0) > 0
            and not bool(getattr(move, "disabled", False))
        )
        for move in moves
    ) / len(moves)
    return reserve, available


def extract_resource_features(
    state: Any,
    *,
    turn: int | float = 0,
    include_public_information: bool = False,
) -> list[float]:
    """Extract player-information resource features from a side-one state.

    Opponent reserve HP, stats, PP, items, and abilities are never valued.
    Information features default to zero for backwards-compatible calibration.
    When enabled, they are read only from the causal packed reveal mask. The
    completed opponent fields are never counted as evidence of their own
    visibility.
    """
    own_side = state.side_one
    opponent_side = state.side_two
    own = _pokemon(own_side)
    opponent = _pokemon(opponent_side)
    own_active = _active(own_side, own)
    opponent_active = _active(opponent_side, opponent)
    known_own = [pokemon for pokemon in own if _known_pokemon(pokemon)]
    known_opponent = [pokemon for pokemon in opponent if _known_pokemon(pokemon)]
    alive_own = [pokemon for pokemon in known_own if float(getattr(pokemon, "hp", 0) or 0) > 0]
    active_index = int(own_side.active_index)
    bench = [pokemon for index, pokemon in enumerate(own) if index != active_index]
    live_bench = [pokemon for pokemon in bench if _known_pokemon(pokemon) and float(getattr(pokemon, "hp", 0) or 0) > 0]
    pp_reserve, move_availability = _pp_features(own)

    public_information = (0.0, 0.0, 0.0, 0.0)
    if include_public_information:
        bits = getattr(state, "s1_public_reveals", None)
        if isinstance(bits, bool) or not isinstance(bits, int):
            raise ValueError("public-information features require a causal reveal mask")
        public_information = information_fractions(bits)
    values = [
        sum(_hp(pokemon) for pokemon in known_own) / 6.0,
        len(alive_own) / 6.0,
        _hp(own_active),
        sum(_hp(pokemon) for pokemon in live_bench) / 5.0,
        len(live_bench) / 5.0,
        _tera_available(state, "side_one", own_side),
        pp_reserve,
        move_availability,
        1.0 - _hp(opponent_active),
        sum(
            _known_pokemon(pokemon) and float(getattr(pokemon, "hp", 0) or 0) <= 0
            for pokemon in opponent
        )
        / 6.0,
        float(_statused(opponent_active)),
        1.0 - _tera_available(state, "side_two", opponent_side),
        (_boost(own_side) - _boost(opponent_side) + 2.0) / 4.0,
        (_screen(own_side) - _screen(opponent_side) + 1.0) / 2.0,
        (_hazard(opponent_side) - _hazard(own_side) + 1.0) / 2.0,
        (
            float(float(getattr(own_side, "substitute_health", 0) or 0) > 0)
            - float(float(getattr(opponent_side, "substitute_health", 0) or 0) > 0)
            + 1.0
        )
        / 2.0,
        *public_information,
        sum(_known_item(pokemon) and float(getattr(pokemon, "hp", 0) or 0) > 0 for pokemon in known_own)
        / 6.0,
        min(1.0, max(0.0, float(turn) / 100.0)),
        float(bool(getattr(getattr(state, "trick_room", None), "active", False))),
    ]
    if len(values) != FEATURE_COUNT or any(not math.isfinite(value) for value in values):
        raise ValueError("resource feature extraction produced an invalid vector")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("resource feature contract requires values in [0, 1]")
    return values


def shadow_utility(features: Sequence[float], coefficients: Sequence[float]) -> float:
    if len(features) != FEATURE_COUNT or len(coefficients) != FEATURE_COUNT:
        raise ValueError("resource shadow vector shape mismatch")
    if any(coefficient < 0 for coefficient in coefficients[:RESOURCE_FEATURE_COUNT]):
        raise ValueError("resource shadow prices must be non-negative")
    value = math.fsum(float(feature) * float(coefficient) for feature, coefficient in zip(features, coefficients, strict=True))
    if not math.isfinite(value):
        raise ValueError("resource shadow utility is non-finite")
    return value
