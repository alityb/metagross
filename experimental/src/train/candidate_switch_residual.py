"""Causal candidate-switch features for selective search correction.

The feature contract deliberately represents a switch target by public/owned
properties and belief-aggregated matchup consequences.  It never serializes a
sampled opponent team, species identity, or an individual world observation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np


SCHEMA = "metagross-candidate-switch-residual/v1"
TYPES = (
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE",
    "FLYING", "GHOST", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
    "PSYCHIC", "ROCK", "STEEL", "WATER",
)
STATIC_SIGNALS = (
    "hp_fraction", "attack", "defense", "special_attack", "special_defense",
    "speed", "physical_bias", "mixed_bulk", "has_pivot", "has_recovery",
)
MATCHUP_SIGNALS = (
    "post_entry_hp_fraction", "entry_hp_preserved", "survives_entry",
    "outgoing_best_fraction", "outgoing_mean_fraction", "outgoing_ko_margin",
    "incoming_best_safety", "incoming_mean_safety", "survival_margin",
    "damage_race_margin", "effective_speed_advantage", "switch_flexibility",
    "next_status_fraction",
)
SUMMARY_STATS = ("mean", "lower_tail")
CANDIDATE_FEATURE_NAMES = (
    *(f"relative_type_{name.lower()}" for name in TYPES),
    *(f"relative_{name}" for name in STATIC_SIGNALS),
    *(f"relative_{name}_{stat}" for name in MATCHUP_SIGNALS for stat in SUMMARY_STATS),
)

PIVOT_MOVES = frozenset({
    "batonpass", "chillyreception", "flipturn", "partingshot", "shedtail",
    "teleport", "uturn", "voltswitch",
})
RECOVERY_MOVES = frozenset({
    "healorder", "junglehealing", "lifedew", "milkdrink", "moonlight",
    "morningsun", "purify", "recover", "rest", "roost", "shoreup", "slackoff", "softboiled",
    "strengthsap", "synthesis", "wish",
})


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def active(side: Any) -> Any:
    try:
        return side.pokemon[int(side.active_index)]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("state does not expose a valid active Pokemon") from exc


def switch_target(state: Any, action: str) -> Any:
    if not action.lower().startswith("switch "):
        raise ValueError(f"candidate action is not a switch: {action!r}")
    query = normalize(action.removeprefix("switch "))
    candidates = [pokemon for pokemon in state.side_one.pokemon
                  if float(pokemon.hp) > 0 and normalize(pokemon.id) == query]
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely resolve switch target {action!r}")
    return candidates[0]


def type_vector(pokemon: Any) -> list[float]:
    present = {str(value).upper() for value in pokemon.types if str(value).upper() != "TYPELESS"}
    return [float(name in present) for name in TYPES]


def static_features(pokemon: Any) -> list[float]:
    moves = {normalize(move.id) for move in pokemon.moves if normalize(move.id) not in {"", "none"}}
    maximum_hp = max(1.0, float(pokemon.maxhp))
    values = [
        float(pokemon.hp) / maximum_hp,
        float(pokemon.attack) / 500.0,
        float(pokemon.defense) / 500.0,
        float(pokemon.special_attack) / 500.0,
        float(pokemon.special_defense) / 500.0,
        float(pokemon.speed) / 500.0,
        (float(pokemon.attack) - float(pokemon.special_attack)) / 500.0,
        min(float(pokemon.defense), float(pokemon.special_defense)) / 500.0,
        float(bool(moves & PIVOT_MOVES)),
        float(bool(moves & RECOVERY_MOVES)),
    ]
    if len(values) != len(STATIC_SIGNALS) or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid candidate static features")
    return values


def canonical_moves(options: Sequence[str]) -> list[str]:
    return sorted({str(option).lower().removesuffix("-tera") for option in options
                   if not str(option).lower().startswith("switch ")
                   and str(option).lower() not in {"no move", "nomove"}})


def matchup_features(before_target: Any, after: Any, *, poke_engine: Any) -> list[float]:
    """Measure an exact post-entry matchup; all outputs are candidate-positive."""
    after_target = next(
        (pokemon for pokemon in after.side_one.pokemon
         if normalize(pokemon.id) == normalize(before_target.id)),
        None,
    )
    before_hp = float(before_target.hp) / max(1.0, float(before_target.maxhp))
    after_hp = (float(after_target.hp) / max(1.0, float(after_target.maxhp))) if after_target else 0.0
    survives = float(bool(after_target and float(after_target.hp) > 0))
    own_options, opponent_options = poke_engine.root_options(after)
    own_moves = canonical_moves(own_options)
    opponent_moves = canonical_moves(opponent_options)
    outgoing: list[float] = []
    incoming: list[float] = []
    status_moves = 0
    opponent = active(after.side_two)
    own_active = active(after.side_one)
    if survives and normalize(own_active.id) == normalize(before_target.id):
        for own_move in own_moves:
            move_outgoing = []
            pairs = opponent_moves or ["struggle"]
            for opponent_move in pairs:
                for own_first in (False, True):
                    try:
                        damage = poke_engine.calculate_damage(
                            after, own_move, opponent_move, own_first,
                        )
                    except ValueError:
                        continue
                    move_outgoing.append(max(map(float, damage[0])) / max(1.0, float(opponent.maxhp)))
                    incoming.append(max(map(float, damage[1])) / max(1.0, float(own_active.maxhp)))
            if move_outgoing:
                outgoing.extend(move_outgoing)
                status_moves += int(max(move_outgoing) <= 1e-12)
        # If every own action is unavailable, still measure an opponent attack
        # against Struggle so that forced/recharge states remain finite.
        if not own_moves:
            for opponent_move in opponent_moves:
                for own_first in (False, True):
                    try:
                        damage = poke_engine.calculate_damage(after, "struggle", opponent_move, own_first)
                    except ValueError:
                        continue
                    incoming.append(max(map(float, damage[1])) / max(1.0, float(own_active.maxhp)))
    outgoing_best = max(outgoing, default=0.0)
    outgoing_mean = float(np.mean(outgoing)) if outgoing else 0.0
    incoming_best = max(incoming, default=0.0)
    incoming_mean = float(np.mean(incoming)) if incoming else 0.0
    opponent_hp = float(opponent.hp) / max(1.0, float(opponent.maxhp))
    speed_delta = (float(own_active.speed) - float(opponent.speed)) / 500.0 if survives else -1.0
    if bool(after.trick_room):
        speed_delta *= -1.0
    switch_count = sum(str(option).lower().startswith("switch ") for option in own_options)
    values = [
        after_hp,
        after_hp - before_hp,
        survives,
        outgoing_best,
        outgoing_mean,
        outgoing_best - opponent_hp,
        -incoming_best,
        -incoming_mean,
        after_hp - incoming_best,
        outgoing_best - incoming_best,
        speed_delta,
        min(1.0, switch_count / 5.0),
        status_moves / max(1, len(own_moves)),
    ]
    if len(values) != len(MATCHUP_SIGNALS) or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid candidate matchup features")
    return values


def summarize_matchups(rows: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(MATCHUP_SIGNALS):
        raise ValueError("invalid candidate matchup matrix")
    result = []
    for column in matrix.T:
        result.extend((float(column.mean()), float(np.quantile(column, 0.10))))
    return result


def residual_features(
    candidate_type: Sequence[float], baseline_type: Sequence[float],
    candidate_static: Sequence[float], baseline_static: Sequence[float],
    candidate_matchup: Sequence[float], baseline_matchup: Sequence[float],
) -> list[float]:
    values = [
        *(float(left) - float(right) for left, right in zip(candidate_type, baseline_type, strict=True)),
        *(float(left) - float(right) for left, right in zip(candidate_static, baseline_static, strict=True)),
        *(float(left) - float(right) for left, right in zip(candidate_matchup, baseline_matchup, strict=True)),
    ]
    if len(values) != len(CANDIDATE_FEATURE_NAMES) or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid relative candidate-switch vector")
    return values
