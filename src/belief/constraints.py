from __future__ import annotations

import re
from typing import Any, Iterable


WEATHER_EXTENSION_ITEMS = {
    "sunnyday": "heatrock",
    "desolateland": "heatrock",
    "raindance": "damprock",
    "primordialsea": "damprock",
    "sandstorm": "smoothrock",
    "hail": "icyrock",
    "snow": "icyrock",
}

STATUS_MOVES = {
    "agility", "auroraveil", "bellydrum", "bulkup", "calmmind",
    "chillyreception", "coil", "curse", "defog", "disable",
    "dragondance", "encore", "glare", "haze", "healbell",
    "hypnosis", "irondefense", "leechseed", "nastyplot", "protect",
    "raindance", "rapidspin", "recover", "rest", "roar",
    "rockpolish", "roost", "shellsmash", "slackoff", "sleeppowder",
    "spikes", "stealthrock", "strengthsap", "substitute", "sunnyday",
    "swordsdance", "synthesis", "taunt", "thunderwave", "tidyup",
    "toxic", "toxicspikes", "trick", "trickroom", "willowisp",
    "wish", "yawn",
}

# Moves that cause self-recoil — do NOT infer Life Orb from these.
SELF_RECOIL_MOVES = {
    "flareblitz", "bravebird", "doubleedge", "headsmash", "highjumpkick",
    "jumpkick", "submission", "takedown", "voltcrash", "wavecrash",
    "wildcharge", "woodhammer", "headcharge", "closecombat",
    "dragonascent", "shadowend", "superpower",
}

# Contact moves for Rocky Helmet chip inference.
CONTACT_MOVES = {
    "tackle", "bodyslam", "return", "extremespeed", "quickattack",
    "waterfall", "liquidation", "ironhead", "playrough", "psychicfangs",
    "poisonjab", "drainpunch", "closecombat", "highjumpkick", "uturn",
    "voltswitch", "flipturn", "leechlife", "knockoff", "stoneedge",
    "earthquake", "flareblitz", "wildcharge", "woodhammer", "dragonrush",
    "outrage", "dracometeor", "bravebird", "doubleedge", "extremespeed",
    "facade", "headbutt", "crunch", "bitingmoves", "bite", "firefang",
    "icefang", "thunderfang", "dragonfang", "shadowclaw", "nightslash",
    "slash", "xscissor", "bugbite", "pinmissile", "rapidspin",
    "trailblaze", "icespinner", "dragonpulse", "shadowball", "energyball",
}

# Moves that activate Prankster (status moves that gain +1 priority).
PRANKSTER_ELIGIBLES = {
    "thunderwave", "willowisp", "taunt", "encore", "disable",
    "spikes", "toxicspikes", "stealthrock", "stickyweb", "reflect",
    "lightscreen", "auroraveil", "tailwind", "toxic", "leechseed",
    "substitute", "protect", "destinybond", "recover", "roost",
    "slackoff", "wish", "healbell", "aromatherapy",
}

# Abilities that suppress trapping (exit from Arena Trap, Shadow Tag, Magnet Pull).
ANTI_TRAP_ITEM = "shedshell"
TRAPPING_ABILITIES = {"arenatrap", "shadowtag", "magnetpull"}

# Abilities that have passive speed modification (don't infer scarf from these).
SPEED_MODIFYING_ABILITIES = {
    "speedboost", "swiftswim", "chlorophyll", "sandrush", "slushrush",
    "surgesurfer", "unburden", "quickfeet",
}

# NFE Pokémon that commonly hold Eviolite.
EVIOLITE_CANDIDATES = {
    "chansey", "happiny", "pichu", "pikachu", "raichu", "togepi",
    "togetic", "sneasel", "porygon", "porygon2", "dusclops", "dusknoir",
    "doublade", "golbat", "haunter", "gligar", "scyther", "vigoroth",
    "magmar", "electabuzz", "rhydon", "seadra", "dragonair",
}


def normalize_name(value: Any) -> str:
    """Normalize PS/Smogon names for conservative set matching."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def set_moves(candidate: dict[str, Any]) -> set[str]:
    moves = candidate.get("moves") or candidate.get("move_slots") or []
    normalized: set[str] = set()
    for move in moves:
        if isinstance(move, dict):
            move = move.get("move") or move.get("name")
        normalized.add(normalize_name(move))
    return normalized


def set_item(candidate: dict[str, Any]) -> str:
    return normalize_name(candidate.get("item") or candidate.get("held_item"))


def set_ability(candidate: dict[str, Any]) -> str:
    return normalize_name(candidate.get("ability"))


def set_speed(candidate: dict[str, Any]) -> int | None:
    stats = candidate.get("stats") or candidate.get("evs") or {}
    for key in ("spe", "speed", "Speed"):
        value = stats.get(key) if isinstance(stats, dict) else None
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def filter_by_seen_moves(candidates: Iterable[dict[str, Any]], seen_moves: Iterable[str]) -> list[dict[str, Any]]:
    required = {normalize_name(move) for move in seen_moves if normalize_name(move)}
    if not required:
        return list(candidates)
    return [candidate for candidate in candidates if required.issubset(set_moves(candidate))]


def filter_by_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    normalized = normalize_name(item)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_item(candidate) == normalized]


def filter_by_ability(candidates: Iterable[dict[str, Any]], ability: str) -> list[dict[str, Any]]:
    normalized = normalize_name(ability)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_ability(candidate) == normalized]


def filter_by_speed_bounds(
    candidates: Iterable[dict[str, Any]],
    lower: int | None = None,
    upper: int | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        speed = set_speed(candidate)
        if speed is None:
            filtered.append(candidate)
            continue
        if lower is not None and speed < lower:
            continue
        if upper is not None and speed > upper:
            continue
        filtered.append(candidate)
    return filtered


def filter_without_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    normalized = normalize_name(item)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_item(candidate) != normalized]


def filter_without_items(candidates: Iterable[dict[str, Any]], items: Iterable[str]) -> list[dict[str, Any]]:
    normalized = {normalize_name(i) for i in items if normalize_name(i)}
    return [c for c in candidates if set_item(c) not in normalized]


def filter_having_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    return filter_by_item(candidates, item)


def filter_without_ability(candidates: Iterable[dict[str, Any]], ability: str) -> list[dict[str, Any]]:
    normalized = normalize_name(ability)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_ability(candidate) != normalized]


def filter_status_move_assault_vest(candidates: Iterable[dict[str, Any]], move: str | None = None) -> list[dict[str, Any]]:
    normalized_move = normalize_name(move)
    if normalized_move and normalized_move not in STATUS_MOVES:
        return list(candidates)
    return filter_without_item(candidates, "Assault Vest")


def filter_weather_extension(candidates: Iterable[dict[str, Any]], weather: str, turns: int) -> list[dict[str, Any]]:
    if turns <= 5:
        return list(candidates)
    item = WEATHER_EXTENSION_ITEMS.get(normalize_name(weather))
    if not item:
        return list(candidates)
    return filter_having_item(candidates, item)


def filter_by_damage_range(
    candidates: Iterable[dict[str, Any]],
    observed_damage: float,
    damage_range_fn: Any | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        damage_range = None
        if damage_range_fn is not None:
            damage_range = damage_range_fn(candidate, *args, **kwargs)
        elif "damage_range" in candidate:
            damage_range = candidate.get("damage_range")
        if damage_range is None:
            filtered.append(candidate)
            continue
        low, high = damage_range
        if float(low) <= float(observed_damage) <= float(high):
            filtered.append(candidate)
    return filtered


def uniform_posterior(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    probability = 1.0 / len(candidates)
    posterior: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        entry = dict(candidate)
        entry.setdefault("set_index", index)
        entry["probability"] = probability
        posterior.append(entry)
    return posterior

STATUS_MOVES = {
    "agility",
    "auroraveil",
    "bellydrum",
    "bulkup",
    "calmmind",
    "chillyreception",
    "coil",
    "curse",
    "defog",
    "disable",
    "dragondance",
    "encore",
    "glare",
    "haze",
    "healbell",
    "hypnosis",
    "irondefense",
    "leechseed",
    "nastyplot",
    "protect",
    "raindance",
    "rapidspin",
    "recover",
    "rest",
    "roar",
    "rockpolish",
    "roost",
    "shellsmash",
    "slackoff",
    "sleeppowder",
    "spikes",
    "stealthrock",
    "strengthsap",
    "substitute",
    "sunnyday",
    "swordsdance",
    "synthesis",
    "taunt",
    "thunderwave",
    "tidyup",
    "toxic",
    "toxicspikes",
    "trick",
    "trickroom",
    "willowisp",
    "wish",
    "yawn",
}


def normalize_name(value: Any) -> str:
    """Normalize PS/Smogon names for conservative set matching."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def set_moves(candidate: dict[str, Any]) -> set[str]:
    moves = candidate.get("moves") or candidate.get("move_slots") or []
    normalized: set[str] = set()
    for move in moves:
        if isinstance(move, dict):
            move = move.get("move") or move.get("name")
        normalized.add(normalize_name(move))
    return normalized


def set_item(candidate: dict[str, Any]) -> str:
    return normalize_name(candidate.get("item") or candidate.get("held_item"))


def set_ability(candidate: dict[str, Any]) -> str:
    return normalize_name(candidate.get("ability"))


def set_speed(candidate: dict[str, Any]) -> int | None:
    stats = candidate.get("stats") or candidate.get("evs") or {}
    for key in ("spe", "speed", "Speed"):
        value = stats.get(key) if isinstance(stats, dict) else None
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def filter_by_seen_moves(candidates: Iterable[dict[str, Any]], seen_moves: Iterable[str]) -> list[dict[str, Any]]:
    required = {normalize_name(move) for move in seen_moves if normalize_name(move)}
    if not required:
        return list(candidates)
    return [candidate for candidate in candidates if required.issubset(set_moves(candidate))]


def filter_by_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    normalized = normalize_name(item)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_item(candidate) == normalized]


def filter_by_ability(candidates: Iterable[dict[str, Any]], ability: str) -> list[dict[str, Any]]:
    normalized = normalize_name(ability)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_ability(candidate) == normalized]


def filter_by_speed_bounds(
    candidates: Iterable[dict[str, Any]],
    lower: int | None = None,
    upper: int | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        speed = set_speed(candidate)
        if speed is None:
            filtered.append(candidate)
            continue
        if lower is not None and speed < lower:
            continue
        if upper is not None and speed > upper:
            continue
        filtered.append(candidate)
    return filtered


def filter_without_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    normalized = normalize_name(item)
    if not normalized:
        return list(candidates)
    return [candidate for candidate in candidates if set_item(candidate) != normalized]


def filter_having_item(candidates: Iterable[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    return filter_by_item(candidates, item)


def filter_status_move_assault_vest(candidates: Iterable[dict[str, Any]], move: str | None = None) -> list[dict[str, Any]]:
    normalized_move = normalize_name(move)
    if normalized_move and normalized_move not in STATUS_MOVES:
        return list(candidates)
    return filter_without_item(candidates, "Assault Vest")


def filter_weather_extension(candidates: Iterable[dict[str, Any]], weather: str, turns: int) -> list[dict[str, Any]]:
    if turns <= 5:
        return list(candidates)
    item = WEATHER_EXTENSION_ITEMS.get(normalize_name(weather))
    if not item:
        return list(candidates)
    return filter_having_item(candidates, item)


def filter_by_damage_range(
    candidates: Iterable[dict[str, Any]],
    observed_damage: float,
    damage_range_fn: Any | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        damage_range = None
        if damage_range_fn is not None:
            damage_range = damage_range_fn(candidate, *args, **kwargs)
        elif "damage_range" in candidate:
            damage_range = candidate.get("damage_range")
        if damage_range is None:
            filtered.append(candidate)
            continue
        low, high = damage_range
        if float(low) <= float(observed_damage) <= float(high):
            filtered.append(candidate)
    return filtered


def uniform_posterior(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    probability = 1.0 / len(candidates)
    posterior: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        entry = dict(candidate)
        entry.setdefault("set_index", index)
        entry["probability"] = probability
        posterior.append(entry)
    return posterior
