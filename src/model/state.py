from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


N_POKEMON = 12
TEAM_SIZE = 6
N_ACTIONS = 14
BASE_FIELD_FEATURES = 57
FIELD_FEATURES = BASE_FIELD_FEATURES + 9  # 9-dim generation one-hot
POKEMON_DENSE_FEATURES = 223
UNKNOWN = "<UNKNOWN>"

STATUS_ORDER = ["none", "brn", "frz", "par", "psn", "tox", "slp"]
VOLATILE_EFFECTS = [
    "confusion", "encore", "taunt", "leechseed", "substitute", "yawn",
    "disable", "attract", "curse", "aquaring", "ingrain", "powertrick",
    "healblock", "embargo", "perishsong", "magnetrise", "telekinesis",
    "flashfire", "waterabsorb", "voltabsorb", "dryskin", "stormdrain",
    "lightningrod", "wonderguard", "truant", "slowstart", "zenmode",
    "forecast", "multitype", "rkssystem", "schoolingup", "disguised",
    "battlearmor", "shellarmor", "stickyhold", "contrary", "simple", "unaware",
]
VOLATILE_INDEX = {name: index for index, name in enumerate(VOLATILE_EFFECTS)}

WEATHER_TYPES = ["none", "sunnyday", "raindance", "sandstorm", "hail", "desolateland", "primordialsea"]
TERRAIN_TYPES = ["none", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
POKEMON_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
]  # index 18 = NONE/unknown second type

TERA_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy", "not_terastallized", "unknown",
]

TWO_TURN_MOVES = {
    "fly", "dig", "dive", "bounce", "shadowforce", "phantomforce",
    "skydrop", "solarbeam", "solarblade", "meteorbeam", "geomancy",
    "freezeshock", "iceburn", "razorwind", "skullbash",
}

# --- Species type cache ---
_SPECIES_TYPES: dict[str, tuple[int, int]] = {}


def _load_species_types(pokedex_path: str = "data/pokedex.json") -> None:
    global _SPECIES_TYPES
    if _SPECIES_TYPES:
        return
    try:
        dex = json.loads(Path(pokedex_path).read_text())
        for name, entry in dex.items():
            types = entry.get("types") or []
            t1 = POKEMON_TYPES.index(types[0].lower()) if types and types[0].lower() in POKEMON_TYPES else 18
            t2 = POKEMON_TYPES.index(types[1].lower()) if len(types) > 1 and types[1].lower() in POKEMON_TYPES else 18
            _SPECIES_TYPES[normalize_name(name)] = (t1, t2)
    except Exception:
        pass


def _get_species_types(species_name: Any) -> tuple[int, int]:
    _load_species_types()
    return _SPECIES_TYPES.get(normalize_name(species_name), (18, 18))


# --- Core utilities ---

def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@dataclass(frozen=True)
class Vocabulary:
    species: dict[str, int]
    moves: dict[str, int]
    items: dict[str, int]
    abilities: dict[str, int]

    @property
    def species_size(self) -> int:
        return len(self.species)

    @property
    def move_size(self) -> int:
        return len(self.moves)

    @property
    def item_size(self) -> int:
        return len(self.items)

    @property
    def ability_size(self) -> int:
        return len(self.abilities)


@dataclass
class EncodedState:
    species_ids: np.ndarray
    move_ids: np.ndarray
    item_ids: np.ndarray
    ability_ids: np.ndarray
    pokemon_dense: np.ndarray
    field: np.ndarray
    active_indices: np.ndarray
    action_mask: np.ndarray

    def as_batch(self) -> dict[str, np.ndarray]:
        return {
            "species_ids": self.species_ids[None, :],
            "move_ids": self.move_ids[None, :, :],
            "item_ids": self.item_ids[None, :],
            "ability_ids": self.ability_ids[None, :],
            "pokemon_dense": self.pokemon_dense[None, :, :],
            "field": self.field[None, :],
            "active_indices": self.active_indices[None, :],
            "action_mask": self.action_mask[None, :],
        }


# --- Vocabulary builder ---

def _vocab_index(vocab: dict[str, int], value: Any) -> int:
    normalized = normalize_name(value)
    return vocab.get(normalized, 0)


def _register(vocab: dict[str, int], value: Any) -> None:
    normalized = normalize_name(value)
    if normalized and normalized not in vocab:
        vocab[normalized] = len(vocab)


def build_vocabulary(pool: dict[str, list[dict[str, Any]]] | str | Path | None = None) -> Vocabulary:
    species = {UNKNOWN: 0}
    moves = {UNKNOWN: 0}
    items = {UNKNOWN: 0}
    abilities = {UNKNOWN: 0}
    if isinstance(pool, (str, Path)):
        path = Path(pool)
        data = json.loads(path.read_text()) if path.exists() else {}
    else:
        data = pool or {}
    if isinstance(data, dict) and "species" in data:
        for species_name, entry in (data.get("species") or {}).items():
            _register(species, species_name)
            for gen_sets in (entry.get("sets") or {}).values() if isinstance(entry, dict) else []:
                for candidate in gen_sets or []:
                    for move in candidate.get("moves", []) or []:
                        if isinstance(move, dict):
                            move = move.get("move") or move.get("name")
                        _register(moves, move)
                    _register(items, candidate.get("item") or candidate.get("held_item"))
                    _register(abilities, candidate.get("ability"))
        for move_name in (data.get("moves") or {}).keys():
            _register(moves, move_name)
        for item_name in data.get("items") or []:
            _register(items, item_name)
        for ability_name in data.get("abilities") or []:
            _register(abilities, ability_name)
    else:
        for species_name, sets in data.items():
            _register(species, species_name)
            for candidate in sets or []:
                for move in candidate.get("moves", []) or []:
                    if isinstance(move, dict):
                        move = move.get("move") or move.get("name")
                    _register(moves, move)
                _register(items, candidate.get("item") or candidate.get("held_item"))
                _register(abilities, candidate.get("ability"))
    return Vocabulary(species=species, moves=moves, items=items, abilities=abilities)


# --- Helper encoders ---

def _one_hot(index: int | None, size: int, unknown_uniform: bool = True) -> np.ndarray:
    if index is None or index < 0 or index >= size:
        return np.full(size, 1.0 / size, dtype=np.float32) if unknown_uniform else np.zeros(size, dtype=np.float32)
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def _hp_bin(value: Any) -> int | None:
    if value is None:
        return None
    try:
        hp = float(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and "/" in value:
            current, total = value.split("/", 1)
            try:
                hp = float(current) / max(1.0, float(total.split()[0]))
            except (TypeError, ValueError):
                return None
        else:
            return None
    if hp > 1.0:
        hp /= 100.0
    if hp <= 0:
        return 0
    return min(6, max(0, int(math.ceil(hp * 6))))


def _pp_bin(value: Any) -> int | None:
    if value is None:
        return None
    try:
        pp = int(value)
    except (TypeError, ValueError):
        return None
    if pp <= 0:
        return 0
    if pp <= 2:
        return 1
    if pp <= 8:
        return 2
    return 3


def _pokemon_dict(pokemon: Any) -> dict[str, Any]:
    if pokemon is None:
        return {}
    if isinstance(pokemon, dict):
        return pokemon
    moves = getattr(pokemon, "moves", None)
    if isinstance(moves, dict):
        moves = list(moves.keys())
    return {
        "species": getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None) or getattr(pokemon, "id", None),
        "moves": moves or [],
        "item": getattr(pokemon, "item", None),
        "ability": getattr(pokemon, "ability", None),
        "hp_fraction": getattr(pokemon, "current_hp_fraction", None),
        "hp": getattr(pokemon, "hp", None),
        "maxhp": getattr(pokemon, "maxhp", None),
        "boosts": getattr(pokemon, "boosts", None),
        "status": getattr(getattr(pokemon, "status", None), "name", None) or getattr(pokemon, "status", None),
        "tera_type": getattr(getattr(pokemon, "tera_type", None), "name", None) or getattr(pokemon, "tera_type", None),
        "tera_used": getattr(pokemon, "terastallized", False),
        "is_active": getattr(pokemon, "active", False),
        "types": getattr(pokemon, "types", None),
        "sleep_turns": getattr(pokemon, "sleep_turns", 0),
        "rest_turns": getattr(pokemon, "rest_turns", 0),
    }


def _extract_team(state: Any, key: str) -> list[Any]:
    if isinstance(state, dict):
        value = state.get(key) or state.get("team" if key == "own_team" else "opponent") or []
    else:
        attr = "team" if key == "own_team" else "opponent_team"
        value = getattr(state, attr, {}) or {}
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _encode_pokemon(pokemon: Any, vocab: Vocabulary, active_default: bool = False) -> tuple[int, np.ndarray, int, int, np.ndarray]:
    data = _pokemon_dict(pokemon)
    species_id = _vocab_index(vocab.species, data.get("species"))
    item_id = _vocab_index(vocab.items, data.get("item"))
    ability_id = _vocab_index(vocab.abilities, data.get("ability"))
    raw_moves = list(data.get("moves") or [])[:4]
    move_ids = np.zeros(4, dtype=np.int64)
    pp_values = data.get("pp") or []
    for idx in range(4):
        move = raw_moves[idx] if idx < len(raw_moves) else None
        if isinstance(move, dict):
            pp_values = pp_values or [slot.get("pp") for slot in raw_moves if isinstance(slot, dict)]
            move = move.get("move") or move.get("name")
        move_ids[idx] = _vocab_index(vocab.moves, move)

    parts: list[np.ndarray] = []

    # hp_fraction: 7-bin one-hot
    hp_val = data.get("hp_fraction") or data.get("hp")
    if hp_val is None and data.get("maxhp"):
        try:
            hp_val = float(data.get("hp", 0)) / max(1.0, float(data["maxhp"]))
        except (TypeError, ValueError):
            hp_val = None
    parts.append(_one_hot(_hp_bin(hp_val), 7))

    # stat_boosts: 7 stats * 13 bins = 91
    boosts = data.get("boosts") or data.get("stat_boosts") or {}
    for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        value = boosts.get(stat) if isinstance(boosts, dict) else None
        boost_index = int(value) + 6 if value is not None else 6
        parts.append(_one_hot(boost_index, 13, unknown_uniform=False))

    # status: 7-dim one-hot
    status = normalize_name(data.get("status") or "none") or "none"
    status_index = STATUS_ORDER.index(status) if status in STATUS_ORDER else None
    parts.append(_one_hot(status_index, len(STATUS_ORDER)))

    # volatile_effects: 38 binary flags
    volatiles = data.get("volatile_effects") or data.get("volatiles") or []
    volatile_vec = np.zeros(len(VOLATILE_EFFECTS), dtype=np.float32)
    if isinstance(volatiles, (dict, set)):
        volatiles = list(volatiles.keys()) if isinstance(volatiles, dict) else list(volatiles)
    for name in volatiles:
        normalized = normalize_name(name)
        if normalized in VOLATILE_INDEX:
            volatile_vec[VOLATILE_INDEX[normalized]] = 1.0
    parts.append(volatile_vec)

    # pp: 4 moves * 4 bins = 16
    for idx in range(4):
        pp = pp_values[idx] if idx < len(pp_values) else None
        parts.append(_one_hot(_pp_bin(pp), 4))

    # tera_type: 20-dim one-hot
    tera = normalize_name(data.get("tera_type")) or "unknown"
    tera_index = TERA_TYPES.index(tera) if tera in TERA_TYPES else TERA_TYPES.index("unknown")
    parts.append(_one_hot(tera_index, len(TERA_TYPES), unknown_uniform=False))

    # tera_used: 1
    parts.append(np.array([1.0 if data.get("tera_used") else 0.0], dtype=np.float32))

    # is_active: 1
    parts.append(np.array([1.0 if data.get("is_active", active_default) else 0.0], dtype=np.float32))

    # --- NEW FEATURES (42 dims total) ---

    # type1: 19-dim one-hot (18 types + unknown=18)
    species_name = data.get("species")
    raw_types = data.get("types")
    if raw_types and isinstance(raw_types, (list, tuple)) and len(raw_types) >= 1:
        t1_name = normalize_name(raw_types[0])
        t1 = POKEMON_TYPES.index(t1_name) if t1_name in POKEMON_TYPES else 18
        t2_name = normalize_name(raw_types[1]) if len(raw_types) > 1 else ""
        t2 = POKEMON_TYPES.index(t2_name) if t2_name in POKEMON_TYPES else 18
    else:
        t1, t2 = _get_species_types(species_name)
    parts.append(_one_hot(t1, 19, unknown_uniform=False))

    # type2: 19-dim one-hot
    parts.append(_one_hot(t2, 19, unknown_uniform=False))

    # toxic_counter: scalar / 8.0
    toxic_count = 0
    if status == "tox":
        toxic_count = int(data.get("toxic_count", data.get("toxic_counter", 1)))
    parts.append(np.array([min(1.0, toxic_count / 8.0)], dtype=np.float32))

    # sleep_turns: scalar / 3.0
    sleep_turns = int(data.get("sleep_turns", 0) or 0)
    parts.append(np.array([min(1.0, sleep_turns / 3.0)], dtype=np.float32))

    # two_turn_move: binary
    two_turn = 0.0
    last_move = data.get("last_move") or ""
    if normalize_name(last_move) in TWO_TURN_MOVES:
        two_turn = 1.0
    # Also check volatiles for common two-turn indicators
    for vol in volatiles if isinstance(volatiles, list) else []:
        if normalize_name(vol) in TWO_TURN_MOVES or "twoturnmove" in normalize_name(vol):
            two_turn = 1.0
    parts.append(np.array([two_turn], dtype=np.float32))

    # partial_trap: scalar (turns / 5.0)
    partial_trap = float(data.get("partial_trap_turns", 0) or 0)
    parts.append(np.array([min(1.0, partial_trap / 5.0)], dtype=np.float32))

    dense = np.concatenate(parts).astype(np.float32)
    assert dense.shape[0] == POKEMON_DENSE_FEATURES, f"pokemon dense feature size mismatch: {dense.shape[0]} != {POKEMON_DENSE_FEATURES}"
    return species_id, move_ids, item_id, ability_id, dense


# --- Field encoder ---

def generation_from_format(format_str: Any, default: int = 9) -> int:
    match = re.search(r"gen([1-9])", str(format_str or ""))
    if not match:
        return default
    return max(1, min(9, int(match.group(1))))


def _generation_from_state(state: Any, fallback: int = 9) -> int:
    if isinstance(state, dict):
        return int(state.get("generation") or generation_from_format(state.get("format") or state.get("formatid"), fallback))
    return generation_from_format(
        getattr(state, "battle_format", None)
        or getattr(state, "format", None)
        or getattr(state, "formatid", None)
        or getattr(getattr(state, "format", None), "id", None),
        fallback,
    )


def _weather_index(value: Any) -> int:
    if value is None:
        return 0
    normalized = normalize_name(getattr(value, "name", value))
    WEATHER_MAP = {
        "sunnyday": 1, "sun": 1, "drought": 1,
        "raindance": 2, "rain": 2, "drizzle": 2,
        "sandstorm": 3, "sand": 3,
        "hail": 4, "snow": 4,
        "desolateland": 5, "harshsun": 5,
        "primordialsea": 6, "heavyrain": 6,
    }
    return WEATHER_MAP.get(normalized, 0)


def _terrain_index(value: Any) -> int:
    if value is None:
        return 0
    normalized = normalize_name(getattr(value, "name", value))
    TERRAIN_MAP = {
        "electricterrain": 1, "electric": 1,
        "grassyterrain": 2, "grassy": 2,
        "mistyterrain": 3, "misty": 3,
        "psychicterrain": 4, "psychic": 4,
    }
    return TERRAIN_MAP.get(normalized, 0)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _encode_field(state: Any, generation: int = 9) -> np.ndarray:
    """Encode field features. BASE_FIELD_FEATURES=57, total with gen one-hot=66."""
    base = np.zeros(BASE_FIELD_FEATURES, dtype=np.float32)
    data = state if isinstance(state, dict) else {}
    field = data.get("field", data) if isinstance(data, dict) else {}
    if not isinstance(field, dict):
        field = {}

    # Try to get side conditions from state
    sc_p1 = data.get("side_conditions_p1") or data.get("side_conditions") or {}
    sc_p2 = data.get("side_conditions_p2") or data.get("opponent_side_conditions") or {}

    # Index 0: turn / 100.0
    turn = field.get("turn", data.get("turn", 0))
    base[0] = min(1.0, max(0.0, _safe_int(turn) / 100.0))

    # Index 1-7: weather type one-hot (7 entries)
    weather = field.get("weather") or data.get("weather")
    w_idx = _weather_index(weather)
    if 0 <= w_idx < 7:
        base[1 + w_idx] = 1.0

    # Index 8: weather turns / 8.0
    w_turns = _safe_int(field.get("weather_turns") or data.get("weather_turns_remaining") or field.get("weather_turns_remaining", 0))
    base[8] = min(1.0, w_turns / 8.0)

    # Index 9-13: terrain type one-hot (5 entries)
    terrain = field.get("terrain") or data.get("terrain")
    t_idx = _terrain_index(terrain)
    if 0 <= t_idx < 5:
        base[9 + t_idx] = 1.0

    # Index 14: terrain turns / 5.0
    t_turns = _safe_int(field.get("terrain_turns") or data.get("terrain_turns_remaining") or field.get("terrain_turns_remaining", 0))
    base[14] = min(1.0, t_turns / 5.0)

    # Index 15-16: trick_room
    trick_room = bool(field.get("trick_room") or data.get("trick_room", False))
    base[15] = 1.0 if trick_room else 0.0
    tr_turns = _safe_int(field.get("trick_room_turns") or data.get("trick_room_turns_remaining", 0))
    base[16] = min(1.0, tr_turns / 5.0)

    # Index 17-18: tailwind_p1
    tw_p1 = _safe_int(sc_p1.get("tailwind", 0)) if isinstance(sc_p1, dict) else 0
    base[17] = 1.0 if tw_p1 > 0 else 0.0
    base[18] = min(1.0, tw_p1 / 3.0)

    # Index 19-20: tailwind_p2
    tw_p2 = _safe_int(sc_p2.get("tailwind", 0)) if isinstance(sc_p2, dict) else 0
    base[19] = 1.0 if tw_p2 > 0 else 0.0
    base[20] = min(1.0, tw_p2 / 3.0)

    # Index 21-28: hazards
    base[21] = 1.0 if _safe_int(sc_p1.get("stealth_rock", 0) if isinstance(sc_p1, dict) else 0) else 0.0
    base[22] = min(1.0, _safe_int(sc_p1.get("spikes", 0) if isinstance(sc_p1, dict) else 0) / 3.0)
    base[23] = min(1.0, _safe_int(sc_p1.get("toxic_spikes", 0) if isinstance(sc_p1, dict) else 0) / 2.0)
    base[24] = 1.0 if _safe_int(sc_p1.get("sticky_web", 0) if isinstance(sc_p1, dict) else 0) else 0.0
    base[25] = 1.0 if _safe_int(sc_p2.get("stealth_rock", 0) if isinstance(sc_p2, dict) else 0) else 0.0
    base[26] = min(1.0, _safe_int(sc_p2.get("spikes", 0) if isinstance(sc_p2, dict) else 0) / 3.0)
    base[27] = min(1.0, _safe_int(sc_p2.get("toxic_spikes", 0) if isinstance(sc_p2, dict) else 0) / 2.0)
    base[28] = 1.0 if _safe_int(sc_p2.get("sticky_web", 0) if isinstance(sc_p2, dict) else 0) else 0.0

    # Index 29-40: screens
    for i, key in enumerate(["reflect", "light_screen", "aurora_veil"]):
        val_p1 = _safe_int(sc_p1.get(key, 0) if isinstance(sc_p1, dict) else 0)
        base[29 + i * 2] = 1.0 if val_p1 > 0 else 0.0
        base[30 + i * 2] = min(1.0, val_p1 / 5.0)
    for i, key in enumerate(["reflect", "light_screen", "aurora_veil"]):
        val_p2 = _safe_int(sc_p2.get(key, 0) if isinstance(sc_p2, dict) else 0)
        base[35 + i * 2] = 1.0 if val_p2 > 0 else 0.0
        base[36 + i * 2] = min(1.0, val_p2 / 5.0)

    # Index 41-42: gravity
    gravity = bool(field.get("gravity", False))
    base[41] = 1.0 if gravity else 0.0
    base[42] = min(1.0, _safe_int(field.get("gravity_turns", 0)) / 5.0)

    # Index 43-44: magic_room, wonder_room
    base[43] = 1.0 if field.get("magic_room", False) else 0.0
    base[44] = 1.0 if field.get("wonder_room", False) else 0.0

    # Index 45-48: mist, safeguard
    base[45] = 1.0 if _safe_int(sc_p1.get("mist", 0) if isinstance(sc_p1, dict) else 0) else 0.0
    base[46] = 1.0 if _safe_int(sc_p2.get("mist", 0) if isinstance(sc_p2, dict) else 0) else 0.0
    base[47] = 1.0 if _safe_int(sc_p1.get("safeguard", 0) if isinstance(sc_p1, dict) else 0) else 0.0
    base[48] = 1.0 if _safe_int(sc_p2.get("safeguard", 0) if isinstance(sc_p2, dict) else 0) else 0.0

    # Index 49-50: fainted counts
    fainted_p1 = _safe_int(data.get("fainted_p1", 0))
    fainted_p2 = _safe_int(data.get("fainted_p2", 0))
    base[49] = min(1.0, fainted_p1 / 5.0)
    base[50] = min(1.0, fainted_p2 / 5.0)

    # Index 51-54: wish
    wish_p1 = data.get("wish_p1") or (0, 0)
    wish_p2 = data.get("wish_p2") or (0, 0)
    if isinstance(wish_p1, (list, tuple)) and len(wish_p1) >= 2:
        base[51] = 1.0 if _safe_int(wish_p1[0]) > 0 else 0.0
        base[52] = min(1.0, _safe_int(wish_p1[0]) / 2.0)
    if isinstance(wish_p2, (list, tuple)) and len(wish_p2) >= 2:
        base[53] = 1.0 if _safe_int(wish_p2[0]) > 0 else 0.0
        base[54] = min(1.0, _safe_int(wish_p2[0]) / 2.0)

    # Index 55-56: future sight
    fs_p1 = data.get("future_sight_p1") or (0, "0")
    fs_p2 = data.get("future_sight_p2") or (0, "0")
    if isinstance(fs_p1, (list, tuple)):
        base[55] = min(1.0, _safe_int(fs_p1[0]) / 3.0)
    if isinstance(fs_p2, (list, tuple)):
        base[56] = min(1.0, _safe_int(fs_p2[0]) / 3.0)

    # Generation one-hot (9 dims)
    gen_vec = np.zeros(9, dtype=np.float32)
    gen_vec[max(0, min(8, int(generation) - 1))] = 1.0

    return np.concatenate([base, gen_vec]).astype(np.float32)


# --- Action mask ---

def _action_mask_from_dict(state: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    if "action_mask" in state:
        raw = list(state["action_mask"])
        mask[: min(N_ACTIONS, len(raw))] = raw[:N_ACTIONS]
        return mask
    moves = state.get("available_moves") or []
    for idx, move in enumerate(moves[:4]):
        disabled = bool(move.get("disabled")) if isinstance(move, dict) else False
        mask[idx] = not disabled
    switches = state.get("available_switches") or []
    for idx, _switch in enumerate(switches[:5]):
        mask[4 + idx] = True
    can_tera = bool(state.get("can_tera") or state.get("can_terastallize"))
    if can_tera:
        mask[9:13] = mask[:4]
        if any(mask[4:9]):
            mask[13] = True
    if not mask.any():
        mask[:4] = True
    return mask


def _action_mask_from_battle(battle: Any) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    for idx, _move in enumerate(list(getattr(battle, "available_moves", []) or [])[:4]):
        mask[idx] = True
    for idx, _switch in enumerate(list(getattr(battle, "available_switches", []) or [])[:5]):
        mask[4 + idx] = True
    can_tera = bool(getattr(battle, "can_tera", False) or getattr(battle, "can_terastallize", False))
    if can_tera:
        mask[9:13] = mask[:4]
    if not mask.any():
        mask[:4] = True
    return mask


def _best_set_from_posterior(posterior_entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-probability set from a posterior entry list."""
    if not posterior_entries:
        return None
    return max(posterior_entries, key=lambda e: float(e.get("probability", 0.0)) if isinstance(e, dict) else 0.0)


def _apply_belief_to_opp_team(
    opp_team: list[Any],
    belief_posterior: dict[str, list[dict[str, Any]]] | None,
) -> list[Any]:
    """
    Replace opponent team slots with posterior-informed dicts when belief data is available.

    For each slot in the posterior, use the highest-probability set's moves/item/ability
    to replace UNKNOWN tokens. HP fraction and other observed battle state is preserved
    from the original slot if it contains that data.

    This implements the AGENTS.md requirement:
    "the RLM's cross-turn posterior is written back into the opponent tokens
    as weighted soft assignments over the vocabulary."
    We use mode (argmax) of the posterior rather than full soft weighting since the
    model architecture uses integer embedding lookups, not continuous inputs.
    """
    if not belief_posterior:
        return opp_team
    enriched = list(opp_team)
    # Build a slot-name → slot-index map from the existing team
    slot_species_map: dict[str, int] = {}
    for idx, mon in enumerate(enriched[:TEAM_SIZE]):
        if mon is None:
            continue
        data = _pokemon_dict(mon)
        species = normalize_name(data.get("species") or "")
        if species:
            slot_species_map[species] = idx
    for slot_key, entries in belief_posterior.items():
        if not isinstance(entries, list) or not entries:
            continue
        best = _best_set_from_posterior(entries)
        if not isinstance(best, dict):
            continue
        # Match by species name (normalised) or slot key
        species_key = normalize_name(best.get("species") or slot_key.split("_")[0])
        idx = slot_species_map.get(species_key)
        if idx is None:
            # Try matching the slot key itself as species
            idx = slot_species_map.get(normalize_name(slot_key))
        if idx is None:
            # Append as a new unseen opponent Pokémon (within team limit)
            if len([m for m in enriched[:TEAM_SIZE] if m is not None]) < TEAM_SIZE:
                for empty_idx in range(TEAM_SIZE):
                    if enriched[empty_idx] is None:
                        idx = empty_idx
                        break
        if idx is None:
            continue
        existing = enriched[idx]
        existing_data = _pokemon_dict(existing) if existing is not None else {}
        # Merge: posterior fills in unseen attributes; observed battle state takes priority
        merged: dict[str, Any] = dict(best)
        merged["species"] = existing_data.get("species") or best.get("species") or species_key
        # Preserve observed HP from the battle state
        if existing_data.get("hp_fraction") is not None:
            merged["hp_fraction"] = existing_data["hp_fraction"]
        if existing_data.get("status"):
            merged["status"] = existing_data["status"]
        if existing_data.get("is_active") is not None:
            merged["is_active"] = existing_data["is_active"]
        # Only override moves/item/ability if the posterior has them and battle hasn't revealed them
        if not existing_data.get("moves") and best.get("moves"):
            merged["moves"] = best["moves"]
        elif existing_data.get("moves"):
            merged["moves"] = existing_data["moves"]
        if not existing_data.get("item") and best.get("item"):
            merged["item"] = best["item"]
        elif existing_data.get("item"):
            merged["item"] = existing_data["item"]
        if not existing_data.get("ability") and best.get("ability"):
            merged["ability"] = best["ability"]
        elif existing_data.get("ability"):
            merged["ability"] = existing_data["ability"]
        enriched[idx] = merged
    return enriched


# --- Main encoder ---

def encode_state(
    state: Any,
    vocab: Vocabulary | None = None,
    pool: dict[str, list[dict[str, Any]]] | str | Path | None = None,
    generation: int = 9,
    belief_posterior: dict[str, list[dict[str, Any]]] | None = None,
) -> EncodedState:
    if isinstance(state, EncodedState):
        return state
    default_pool = "data/all_gen_pool.json" if Path("data/all_gen_pool.json").exists() else "data/gen9_random_pool.json"
    vocab = vocab or build_vocabulary(pool or default_pool)
    generation = _generation_from_state(state, generation)
    own_team = _extract_team(state, "own_team")[:TEAM_SIZE]
    opp_team = _extract_team(state, "opponent_team")[:TEAM_SIZE]
    # Apply belief posterior to enrich opponent tokens before encoding
    if belief_posterior is None and isinstance(state, dict):
        belief_posterior = state.get("belief_posterior") or state.get("opponent_posterior")
    opp_team = _apply_belief_to_opp_team(opp_team, belief_posterior)
    own_team.extend([None] * (TEAM_SIZE - len(own_team)))
    opp_team.extend([None] * (TEAM_SIZE - len(opp_team)))
    species_ids = np.zeros(N_POKEMON, dtype=np.int64)
    move_ids = np.zeros((N_POKEMON, 4), dtype=np.int64)
    item_ids = np.zeros(N_POKEMON, dtype=np.int64)
    ability_ids = np.zeros(N_POKEMON, dtype=np.int64)
    dense = np.zeros((N_POKEMON, POKEMON_DENSE_FEATURES), dtype=np.float32)

    active_own = 0
    active_opp = 0
    for idx, pokemon in enumerate(own_team + opp_team):
        species_id, moves, item_id, ability_id, pokemon_dense = _encode_pokemon(pokemon, vocab, active_default=idx in {0, TEAM_SIZE})
        species_ids[idx] = species_id
        move_ids[idx] = moves
        item_ids[idx] = item_id
        ability_ids[idx] = ability_id
        dense[idx] = pokemon_dense
        if pokemon_dense[181] > 0.5:  # is_active is at index 181 (7+91+7+38+16+20+1+1-1=181-1... check)
            # Actually is_active is the last feature before the new 42 features
            # hp(7) + boosts(91) + status(7) + volatiles(38) + pp(16) + tera(20) + tera_used(1) + is_active(1) = 181
            # So is_active is at index 180
            pass
        # Use the is_active position: index = 7+91+7+38+16+20+1 = 180
        if pokemon_dense[180] > 0.5:
            if idx < TEAM_SIZE:
                active_own = idx
            else:
                active_opp = idx - TEAM_SIZE

    field = _encode_field(state, generation=generation)
    if isinstance(state, dict):
        action_mask = _action_mask_from_dict(state)
    else:
        action_mask = _action_mask_from_battle(state)
    return EncodedState(
        species_ids=species_ids,
        move_ids=move_ids,
        item_ids=item_ids,
        ability_ids=ability_ids,
        pokemon_dense=dense,
        field=field,
        active_indices=np.array([active_own, active_opp], dtype=np.int64),
        action_mask=action_mask,
    )


def stack_encoded(states: Iterable[EncodedState]) -> dict[str, np.ndarray]:
    encoded = list(states)
    if not encoded:
        raise ValueError("cannot stack an empty state batch")
    return {
        "species_ids": np.stack([state.species_ids for state in encoded]),
        "move_ids": np.stack([state.move_ids for state in encoded]),
        "item_ids": np.stack([state.item_ids for state in encoded]),
        "ability_ids": np.stack([state.ability_ids for state in encoded]),
        "pokemon_dense": np.stack([state.pokemon_dense for state in encoded]),
        "field": np.stack([state.field for state in encoded]),
        "active_indices": np.stack([state.active_indices for state in encoded]),
        "action_mask": np.stack([state.action_mask for state in encoded]),
    }
