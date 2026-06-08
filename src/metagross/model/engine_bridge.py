from __future__ import annotations

import logging
import math
import re
from typing import Any

from .state import EncodedState, encode_state


LOGGER = logging.getLogger(__name__)

DEFAULT_EVS = (85, 85, 85, 85, 85, 85)
DEFAULT_STATS = {
    "hp": 100,
    "atk": 100,
    "def": 100,
    "spa": 100,
    "spd": 100,
    "spe": 100,
}


def normalize_engine_id(value: Any, default: str = "pikachu") -> str:
    if value is None:
        return default
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return normalized or default


def _load_poke_engine() -> Any | None:
    try:
        import poke_engine  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional Rust wheel
        LOGGER.warning("poke_engine unavailable; MCTS will use root-prior fallback: %s", exc)
        return None
    return poke_engine


def _status(value: Any) -> str:
    normalized = normalize_engine_id(getattr(value, "name", value), default="none")
    return {
        "brn": "Burn",
        "burn": "Burn",
        "frz": "Freeze",
        "freeze": "Freeze",
        "par": "Paralyze",
        "paralyze": "Paralyze",
        "psn": "Poison",
        "poison": "Poison",
        "tox": "Toxic",
        "toxic": "Toxic",
        "slp": "Sleep",
        "sleep": "Sleep",
    }.get(normalized, "None")


def _types(pokemon: Any) -> tuple[str, str]:
    raw_types = None
    if isinstance(pokemon, dict):
        raw_types = pokemon.get("types") or pokemon.get("base_types")
    else:
        raw_types = getattr(pokemon, "types", None) or getattr(pokemon, "base_types", None)
    values: list[str] = []
    for value in list(raw_types or []):
        values.append(normalize_engine_id(getattr(value, "name", value), default="typeless"))
    if not values:
        values = ["normal"]
    while len(values) < 2:
        values.append("typeless")
    return values[0], values[1]


def _hp_values(pokemon: Any, level: int) -> tuple[int, int]:
    current = None
    maximum = None
    if isinstance(pokemon, dict):
        current = pokemon.get("hp") or pokemon.get("current_hp") or pokemon.get("current_hp_fraction") or pokemon.get("hp_fraction")
        maximum = pokemon.get("maxhp") or pokemon.get("max_hp")
    else:
        current = getattr(pokemon, "current_hp", None)
        maximum = getattr(pokemon, "max_hp", None)
        if current is None:
            current = getattr(pokemon, "current_hp_fraction", None)
    try:
        maxhp = int(maximum) if maximum is not None else int(DEFAULT_STATS["hp"] + level + 10)
    except (TypeError, ValueError):
        maxhp = int(DEFAULT_STATS["hp"] + level + 10)
    if isinstance(current, str):
        if "fnt" in current.lower():
            return 0, maxhp
        if "/" in current:
            left, right = current.split("/", 1)
            try:
                return int(float(left)), int(float(right.split()[0]))
            except (TypeError, ValueError):
                pass
    try:
        current_float = float(current) if current is not None else 1.0
    except (TypeError, ValueError):
        current_float = 1.0
    if 0.0 <= current_float <= 1.0:
        hp = int(round(maxhp * current_float))
    elif 1.0 < current_float <= 100.0 and maximum is None:
        hp = int(round(maxhp * (current_float / 100.0)))
    else:
        hp = int(round(current_float))
    return max(0, min(maxhp, hp)), maxhp


def _level(pokemon: Any, fallback: int = 80) -> int:
    value = pokemon.get("level") if isinstance(pokemon, dict) else getattr(pokemon, "level", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _stats(pokemon: Any, level: int, config: dict[str, Any] | None = None) -> dict[str, int]:
    stats = dict(DEFAULT_STATS)
    source = {}
    if isinstance(pokemon, dict):
        source = pokemon.get("stats") or {}
    else:
        source = getattr(pokemon, "stats", None) or {}
    if config:
        source = {**source, **(config.get("stats") or {})}
    for short, aliases in {
        "hp": ("hp", "maxhp"),
        "atk": ("atk", "attack"),
        "def": ("def", "defense"),
        "spa": ("spa", "special_attack"),
        "spd": ("spd", "special_defense"),
        "spe": ("spe", "speed"),
    }.items():
        for alias in aliases:
            if isinstance(source, dict) and alias in source:
                try:
                    stats[short] = int(source[alias])
                    break
                except (TypeError, ValueError):
                    pass
    if stats["hp"] == DEFAULT_STATS["hp"]:
        stats["hp"] = DEFAULT_STATS["hp"] + level + 10
    return stats


def _move_id(move: Any) -> str:
    if isinstance(move, dict):
        move = move.get("id") or move.get("move") or move.get("name")
    return normalize_engine_id(getattr(move, "id", None) or getattr(move, "name", None) or move, default="tackle")


def _move_pp(move: Any) -> int:
    if isinstance(move, dict):
        value = move.get("pp") or move.get("current_pp")
    else:
        value = getattr(move, "current_pp", None) or getattr(move, "pp", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 16


def _move_disabled(move: Any) -> bool:
    return bool(move.get("disabled")) if isinstance(move, dict) else bool(getattr(move, "disabled", False))


def _moves(pokemon: Any, config: dict[str, Any] | None, poke_engine: Any) -> list[Any]:
    raw_moves = None
    if isinstance(pokemon, dict):
        raw_moves = pokemon.get("moves")
    else:
        raw_moves = getattr(pokemon, "moves", None)
    if isinstance(raw_moves, dict):
        raw_moves = list(raw_moves.values())
    if (not raw_moves) and config:
        raw_moves = config.get("moves")
    raw_moves = list(raw_moves or ["tackle"])
    moves = [poke_engine.Move(id=_move_id(move), pp=_move_pp(move), disabled=_move_disabled(move)) for move in raw_moves[:4]]
    while len(moves) < 4:
        moves.append(poke_engine.Move(id="none", pp=0, disabled=True))
    return moves


def _pokemon_to_engine(pokemon: Any, poke_engine: Any, config: dict[str, Any] | None = None) -> Any:
    data = pokemon if isinstance(pokemon, dict) else {}
    species = data.get("species") if data else getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None)
    if not species and config:
        species = config.get("species") or config.get("name")
    level = _level(pokemon, int(config.get("level", 80)) if config else 80)
    stats = _stats(pokemon, level, config)
    hp, maxhp = _hp_values(pokemon, level)
    if config and hp > 0 and maxhp == DEFAULT_STATS["hp"] + level + 10:
        maxhp = int(stats.get("hp", maxhp))
        hp = min(hp, maxhp)
    types = _types(pokemon)
    ability = data.get("ability") if data else getattr(pokemon, "ability", None)
    item = data.get("item") if data else getattr(pokemon, "item", None)
    if config:
        ability = ability or config.get("ability")
        item = item or config.get("item")
    return poke_engine.Pokemon(
        id=normalize_engine_id(species),
        level=level,
        types=types,
        base_types=types,
        hp=hp,
        maxhp=maxhp,
        ability=normalize_engine_id(ability, default="none"),
        base_ability=normalize_engine_id(ability, default=""),
        item=normalize_engine_id(item, default="None"),
        nature="serious",
        evs=DEFAULT_EVS,
        attack=int(stats["atk"]),
        defense=int(stats["def"]),
        special_attack=int(stats["spa"]),
        special_defense=int(stats["spd"]),
        speed=int(stats["spe"]),
        status=_status(data.get("status") if data else getattr(pokemon, "status", None)),
        rest_turns=0,
        sleep_turns=0,
        weight_kg=float(data.get("weight_kg", 1.0)) if data else 1.0,
        moves=_moves(pokemon, config, poke_engine),
        terastallized=bool(data.get("tera_used", False)) if data else bool(getattr(pokemon, "terastallized", False)),
        tera_type=normalize_engine_id(data.get("tera_type") if data else getattr(pokemon, "tera_type", None), default="typeless"),
    )


def _dummy_pokemon(poke_engine: Any) -> Any:
    return poke_engine.Pokemon(id="pikachu", level=1, hp=0)


def _team_from_battle(battle: Any, opponent: bool, opponent_config: dict[str, Any] | None = None) -> list[Any]:
    if isinstance(battle, EncodedState):
        return []
    if isinstance(battle, dict):
        key = "opponent_team" if opponent else "own_team"
        team = battle.get(key) or battle.get("opponent" if opponent else "team") or []
    else:
        team = getattr(battle, "opponent_team" if opponent else "team", {}) or {}
    if isinstance(team, dict):
        return list(team.values())
    return list(team or [])


def _config_for_slot(opponent_config: dict[str, Any] | None, slot: int, pokemon: Any | None = None) -> dict[str, Any] | None:
    if not opponent_config:
        return None
    keys = [f"p2{chr(ord('a') + slot)}", f"p2{slot}", str(slot)]
    species = None
    if pokemon is not None:
        species = pokemon.get("species") if isinstance(pokemon, dict) else getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None)
    if species:
        keys.extend([str(species), normalize_engine_id(species)])
    for key in keys:
        value = opponent_config.get(key)
        if isinstance(value, dict):
            return value
    if slot == 0:
        for value in opponent_config.values():
            if isinstance(value, dict):
                return value
    return None


def _side_from_team(team: list[Any], poke_engine: Any, opponent_config: dict[str, Any] | None = None) -> Any:
    pokemon = []
    for idx, mon in enumerate(team[:6]):
        pokemon.append(_pokemon_to_engine(mon, poke_engine, _config_for_slot(opponent_config, idx, mon)))
    if not pokemon and opponent_config:
        pokemon.append(_pokemon_to_engine({}, poke_engine, _config_for_slot(opponent_config, 0)))
    if not pokemon:
        pokemon.append(_pokemon_to_engine({"species": "Pikachu", "moves": ["Tackle"], "hp_fraction": 1.0}, poke_engine))
    while len(pokemon) < 6:
        pokemon.append(_dummy_pokemon(poke_engine))
    active_index = 0
    for idx, mon in enumerate(team[:6]):
        is_active = mon.get("is_active") if isinstance(mon, dict) else getattr(mon, "active", False)
        if is_active:
            active_index = idx
            break
    active = team[active_index] if active_index < len(team) else {}
    boosts = active.get("boosts") or active.get("stat_boosts") if isinstance(active, dict) else getattr(active, "boosts", {}) or {}
    side_conditions = active.get("side_conditions", {}) if isinstance(active, dict) else {}
    return poke_engine.Side(
        pokemon=pokemon,
        side_conditions=poke_engine.SideConditions(
            spikes=int(side_conditions.get("spikes", 0)) if isinstance(side_conditions, dict) else 0,
            toxic_spikes=int(side_conditions.get("toxic_spikes", 0)) if isinstance(side_conditions, dict) else 0,
            stealth_rock=int(side_conditions.get("stealth_rock", 0)) if isinstance(side_conditions, dict) else 0,
            sticky_web=int(side_conditions.get("sticky_web", 0)) if isinstance(side_conditions, dict) else 0,
            tailwind=int(side_conditions.get("tailwind", 0)) if isinstance(side_conditions, dict) else 0,
            reflect=int(side_conditions.get("reflect", 0)) if isinstance(side_conditions, dict) else 0,
            light_screen=int(side_conditions.get("light_screen", 0)) if isinstance(side_conditions, dict) else 0,
            aurora_veil=int(side_conditions.get("aurora_veil", 0)) if isinstance(side_conditions, dict) else 0,
        ),
        active_index=str(active_index),
        volatile_status_durations=poke_engine.VolatileStatusDurations(),
        wish=(0, 0),
        future_sight=(0, "0"),
        volatile_statuses=set(),
        attack_boost=int(boosts.get("atk", 0)) if isinstance(boosts, dict) else 0,
        defense_boost=int(boosts.get("def", 0)) if isinstance(boosts, dict) else 0,
        special_attack_boost=int(boosts.get("spa", 0)) if isinstance(boosts, dict) else 0,
        special_defense_boost=int(boosts.get("spd", 0)) if isinstance(boosts, dict) else 0,
        speed_boost=int(boosts.get("spe", 0)) if isinstance(boosts, dict) else 0,
        accuracy_boost=int(boosts.get("accuracy", 0)) if isinstance(boosts, dict) else 0,
        evasion_boost=int(boosts.get("evasion", 0)) if isinstance(boosts, dict) else 0,
        last_used_move="move:none",
        switch_out_move_second_saved_move="NONE",
    )


def _weather(value: Any) -> str:
    normalized = normalize_engine_id(getattr(value, "name", value), default="none")
    return {
        "raindance": "rain",
        "rain": "rain",
        "sunnyday": "sun",
        "sun": "sun",
        "sandstorm": "sand",
        "sand": "sand",
        "snow": "snow",
        "hail": "hail",
        "desolateland": "harshsun",
        "primordialsea": "heavyrain",
    }.get(normalized, "none")


def _terrain(value: Any) -> str:
    normalized = normalize_engine_id(getattr(value, "name", value), default="none")
    return {
        "electricterrain": "electricterrain",
        "grassyterrain": "grassyterrain",
        "mistyterrain": "mistyterrain",
        "psychicterrain": "psychicterrain",
    }.get(normalized, "none")


def battle_to_poke_engine_state(battle: Any, opponent_config: dict[str, Any] | None = None) -> Any:
    poke_engine = _load_poke_engine()
    if poke_engine is None:
        raise RuntimeError("poke_engine is not importable")
    if _is_poke_engine_state(battle):
        return battle
    own_team = _team_from_battle(battle, opponent=False)
    opponent_team = _team_from_battle(battle, opponent=True, opponent_config=opponent_config)
    if isinstance(battle, EncodedState):
        own_team = [{"species": "Pikachu", "moves": ["Tackle"], "hp_fraction": 1.0, "is_active": True}]
        opponent_team = [{"species": "Eevee", "moves": ["Tackle"], "hp_fraction": 1.0, "is_active": True}]
    field = battle if isinstance(battle, dict) else {}
    return poke_engine.State(
        side_one=_side_from_team(own_team, poke_engine),
        side_two=_side_from_team(opponent_team, poke_engine, opponent_config),
        weather=_weather(field.get("weather") if isinstance(field, dict) else getattr(battle, "weather", None)),
        weather_turns_remaining=int(field.get("weather_turns_remaining", 0)) if isinstance(field, dict) else 0,
        terrain=_terrain(field.get("terrain") if isinstance(field, dict) else getattr(battle, "field", None)),
        terrain_turns_remaining=int(field.get("terrain_turns_remaining", 0)) if isinstance(field, dict) else 0,
        trick_room=bool(field.get("trick_room", False)) if isinstance(field, dict) else bool(getattr(battle, "trick_room", False)),
        trick_room_turns_remaining=int(field.get("trick_room_turns_remaining", 0)) if isinstance(field, dict) else int(getattr(battle, "trick_room_turns_remaining", 0) or 0),
        team_preview=False,
    )


def _is_poke_engine_state(value: Any) -> bool:
    return hasattr(value, "side_one") and hasattr(value, "side_two") and hasattr(value, "to_string")


def _side_to_team(side: Any) -> list[dict[str, Any]]:
    team = []
    active = int(getattr(side, "active_index", "0") or 0)
    for idx, mon in enumerate(list(getattr(side, "pokemon", []) or [])[:6]):
        moves = [
            {"move": getattr(move, "id", "tackle"), "pp": getattr(move, "pp", 16), "disabled": getattr(move, "disabled", False)}
            for move in list(getattr(mon, "moves", []) or [])[:4]
            if getattr(move, "id", "none") != "none"
        ]
        types_raw = getattr(mon, "types", None)
        types_list = list(types_raw) if types_raw else None
        team.append(
            {
                "species": getattr(mon, "id", "pikachu"),
                "moves": moves,
                "item": getattr(mon, "item", None),
                "ability": getattr(mon, "ability", None),
                "hp": getattr(mon, "hp", 0),
                "maxhp": getattr(mon, "maxhp", 100),
                "max_hp": getattr(mon, "maxhp", 100),
                "hp_fraction": (float(getattr(mon, "hp", 0)) / max(1.0, float(getattr(mon, "maxhp", 100)))),
                "status": getattr(mon, "status", "None"),
                "is_active": idx == active,
                "tera_type": getattr(mon, "tera_type", "typeless"),
                "tera_used": getattr(mon, "terastallized", False),
                "types": types_list,
                "sleep_turns": int(getattr(mon, "sleep_turns", 0) or 0),
                "rest_turns": int(getattr(mon, "rest_turns", 0) or 0),
            }
        )
    return team


def _extract_side_conditions(side: Any) -> dict[str, int]:
    sc = getattr(side, "side_conditions", None)
    if sc is None:
        return {}
    return {
        "stealth_rock": int(getattr(sc, "stealth_rock", 0) or 0),
        "spikes": int(getattr(sc, "spikes", 0) or 0),
        "toxic_spikes": int(getattr(sc, "toxic_spikes", 0) or 0),
        "sticky_web": int(getattr(sc, "sticky_web", 0) or 0),
        "tailwind": int(getattr(sc, "tailwind", 0) or 0),
        "reflect": int(getattr(sc, "reflect", 0) or 0),
        "light_screen": int(getattr(sc, "light_screen", 0) or 0),
        "aurora_veil": int(getattr(sc, "aurora_veil", 0) or 0),
        "mist": int(getattr(sc, "mist", 0) or 0),
        "safeguard": int(getattr(sc, "safeguard", 0) or 0),
        "toxic_count": int(getattr(sc, "toxic_count", 0) or 0),
    }


def _fainted_count(side: Any) -> int:
    return sum(1 for mon in list(getattr(side, "pokemon", []) or []) if getattr(mon, "hp", 0) <= 0)


def poke_engine_state_to_dict(state: Any, mirror: bool = False, generation: int = 9) -> dict[str, Any]:
    side_one = state.side_two if mirror else state.side_one
    side_two = state.side_one if mirror else state.side_two
    active = list(getattr(side_one.pokemon[int(side_one.active_index)], "moves", []) or [])
    available_moves = [
        {"move": getattr(move, "id", "tackle"), "disabled": bool(getattr(move, "disabled", False)), "pp": getattr(move, "pp", 16)}
        for move in active[:4]
        if getattr(move, "id", "none") != "none"
    ]
    available_switches = [
        {"species": getattr(mon, "id", "pikachu")}
        for idx, mon in enumerate(list(getattr(side_one, "pokemon", []) or [])[:6])
        if idx != int(side_one.active_index) and getattr(mon, "hp", 0) > 0
    ][:5]
    sc_p1 = _extract_side_conditions(side_one)
    sc_p2 = _extract_side_conditions(side_two)
    wish_p1 = getattr(side_one, "wish", (0, 0)) or (0, 0)
    wish_p2 = getattr(side_two, "wish", (0, 0)) or (0, 0)
    fs_p1 = getattr(side_one, "future_sight", (0, "0")) or (0, "0")
    fs_p2 = getattr(side_two, "future_sight", (0, "0")) or (0, "0")
    return {
        "turn": 1,
        "generation": generation,
        "own_team": _side_to_team(side_one),
        "opponent_team": _side_to_team(side_two),
        "available_moves": available_moves or [{"move": "tackle", "disabled": False}],
        "available_switches": available_switches,
        "weather": getattr(state, "weather", "none"),
        "weather_turns_remaining": int(getattr(state, "weather_turns_remaining", 0) or 0),
        "terrain": getattr(state, "terrain", "none"),
        "terrain_turns_remaining": int(getattr(state, "terrain_turns_remaining", 0) or 0),
        "trick_room": bool(getattr(state, "trick_room", False)),
        "trick_room_turns_remaining": int(getattr(state, "trick_room_turns_remaining", 0) or 0),
        "side_conditions_p1": sc_p1,
        "side_conditions_p2": sc_p2,
        "fainted_p1": _fainted_count(side_one),
        "fainted_p2": _fainted_count(side_two),
        "wish_p1": wish_p1,
        "wish_p2": wish_p2,
        "future_sight_p1": fs_p1,
        "future_sight_p2": fs_p2,
        "field": {
            "turn": 1,
            "weather": getattr(state, "weather", "none"),
            "weather_turns_remaining": int(getattr(state, "weather_turns_remaining", 0) or 0),
            "terrain": getattr(state, "terrain", "none"),
            "terrain_turns_remaining": int(getattr(state, "terrain_turns_remaining", 0) or 0),
            "trick_room": bool(getattr(state, "trick_room", False)),
            "trick_room_turns": int(getattr(state, "trick_room_turns_remaining", 0) or 0),
        },
    }


def encode_poke_engine_state(state: Any, vocab: Any | None = None, mirror: bool = False, generation: int = 9) -> EncodedState:
    return encode_state(poke_engine_state_to_dict(state, mirror=mirror, generation=generation), vocab=vocab, generation=generation)


def clone_poke_engine_state(state: Any) -> Any:
    return type(state).from_string(state.to_string())
