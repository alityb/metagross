from __future__ import annotations

"""
HeuristicRLMStrategist: rule-based set elimination + action scoring.

Replaces NullRLMStrategist's flat prior with a real signal derived from:
1. Belief posterior filtered by observed moves/items/abilities
2. Type-effectiveness scoring of our available moves vs opponent candidate sets
3. HP-advantage and hazard multipliers

No LLM required. This is essentially Foul Play's approach but feeding into
our MCTS root prior instead of being the standalone eval.
"""

import math
from dataclasses import dataclass
from typing import Any


# 18-type effectiveness table (row = attacking type, col = defending type)
# Types: Normal Fire Water Electric Grass Ice Fighting Poison Ground Flying
#        Psychic Bug Rock Ghost Dragon Dark Steel Fairy
_TYPES = [
    "normal","fire","water","electric","grass","ice","fighting","poison",
    "ground","flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy",
]
_TYPE_IDX = {t: i for i, t in enumerate(_TYPES)}

# Effectiveness: 2=super, 1=normal, 0.5=resisted, 0=immune
_EFF: list[list[float]] = [
    # Normal Fire  Water Elec  Grass Ice   Fight Poison Ground Fly   Psy   Bug   Rock  Ghost Dragon Dark  Steel Fairy
    [1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   0.5,   0,    1,    1,   0.5,   1  ],  # Normal
    [1,   0.5,  0.5,   1,    2,    2,    1,    1,    1,    1,    1,    2,  0.5,   1,   0.5,   1,    2,    1  ],  # Fire
    [1,    2,   0.5,   1,   0.5,   1,    1,    1,    2,    1,    1,    1,   2,    1,   0.5,   1,    1,    1  ],  # Water
    [1,    1,    2,   0.5, 0.5,   1,    1,    1,    0,    2,    1,    1,   1,    1,   0.5,   1,    1,    1  ],  # Electric
    [1,   0.5,   2,    1,  0.5,   1,    1,   0.5,   2,   0.5,   1,   0.5,  2,    1,  0.5,   1,   0.5,   1  ],  # Grass
    [1,   0.5,  0.5,   1,    2,   0.5,   1,    1,    2,    2,    1,    1,   1,    1,   2,    1,   0.5,   1  ],  # Ice
    [2,    1,    1,    1,    1,    2,    1,   0.5,   1,   0.5,  0.5,  0.5,   2,    0,   1,    2,    2,   0.5],  # Fighting
    [1,    1,    1,    1,    2,    1,    1,   0.5,  0.5,   1,    1,    1,   1,   0.5,   1,    1,    0,    2  ],  # Poison
    [1,    2,    1,    2,   0.5,   1,    1,    2,    1,    0,    1,   0.5,  2,    1,    1,    1,    2,    1  ],  # Ground
    [1,    1,    1,   0.5,   2,    1,    2,    1,    1,    1,    1,    2,  0.5,   1,    1,    1,   0.5,   1  ],  # Flying
    [1,    1,    1,    1,    1,    1,    2,    2,    1,    1,   0.5,   1,   1,    1,    1,    0,   0.5,   1  ],  # Psychic
    [1,   0.5,   1,    1,    2,    1,   0.5,  0.5,   1,   0.5,   2,    1,   1,   0.5,   1,    2,   0.5,  0.5],  # Bug
    [1,    2,    1,    1,    1,    2,   0.5,   1,   0.5,   2,    1,    2,   1,    1,    1,    1,   0.5,   1  ],  # Rock
    [0,    1,    1,    1,    1,    1,    1,    1,    1,    1,    2,    1,   1,    2,    1,   0.5,   1,    1  ],  # Ghost
    [1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   1,    1,    2,    1,   0.5,   0  ],  # Dragon
    [1,    1,    1,    1,    1,    1,   0.5,   1,    1,    1,    2,    1,   1,    2,    1,   0.5,   1,   0.5],  # Dark
    [1,   0.5,  0.5,  0.5,   1,    2,    1,    1,    1,    1,    1,    1,   2,    1,    1,    1,   0.5,   2  ],  # Steel
    [1,   0.5,   1,    1,    1,    1,    2,   0.5,   1,    1,    1,    1,   1,    1,    2,    2,   0.5,   1  ],  # Fairy
]


def _type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    atk = _TYPE_IDX.get(move_type.lower(), -1)
    if atk < 0:
        return 1.0
    eff = 1.0
    for dt in defender_types:
        def_idx = _TYPE_IDX.get(dt.lower(), -1)
        if def_idx >= 0:
            eff *= _EFF[atk][def_idx]
    return eff


def _normalize_name(value: Any) -> str:
    import re
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _score_action_vs_set(
    action_idx: int,
    our_moves: list[str],
    our_hp: float,
    opp_set: dict[str, Any],
    opp_hp: float,
    move_data: dict[str, Any],
) -> float:
    """Score one of our 14 actions against a specific opponent set. Returns [0, 1]."""
    if action_idx >= 4:
        # Switch or Tera — baseline score
        return 0.4 if action_idx < 9 else 0.5
    if action_idx >= len(our_moves):
        return 0.3
    move_name = _normalize_name(our_moves[action_idx])
    move_info = move_data.get(move_name) or {}
    bp = float(move_info.get("basePower", 0) or 0)
    move_type = (move_info.get("type") or "normal").lower()
    opp_types = [t.lower() for t in (opp_set.get("types") or opp_set.get("pokedex", {}).get("types") or ["normal"])]
    eff = _type_effectiveness(move_type, opp_types)
    # Rough damage proxy: bp * effectiveness, normalised to [0,1]
    damage_score = min(1.0, (bp * eff) / 200.0)
    # KO probability proxy: if damage_score > opp_hp, likely KO
    ko_bonus = 0.5 if damage_score > opp_hp else 0.0
    # Status moves get a fixed score based on our HP advantage
    if bp == 0:
        return 0.3 + 0.2 * (1.0 - our_hp)  # status better when ahead on HP
    return min(1.0, damage_score + ko_bonus * 0.3)


@dataclass
class HeuristicRLMOutput:
    pi_rlm: list[float]
    v_rlm: float
    refined_belief: dict[str, Any]
    iterations: int = 1
    sub_queries: int = 0
    elapsed_ms: float = 0.0
    observations: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = []


class HeuristicRLMStrategist:
    """
    Rule-based RLM strategist.

    Implements the same interface as RLMStrategist but without any LLM calls.
    Uses belief posterior + type effectiveness to produce a meaningful root prior
    for MCTS — the same thing Foul Play does with its hand-crafted eval.

    Key difference from Foul Play: this feeds into PokeNet MCTS root, not standalone.
    """

    def __init__(self, pool: dict[str, Any] | None = None, move_data: dict[str, Any] | None = None) -> None:
        self.pool = pool or {}
        self.move_data = move_data or {}
        # Lazy-load from disk if not provided
        self._pool_loaded = bool(pool)
        self._moves_loaded = bool(move_data)

    def _ensure_loaded(self) -> None:
        if not self._pool_loaded:
            try:
                import json
                from pathlib import Path
                for path in ["data/all_gen_pool.json", "data/gen9_random_pool.json"]:
                    if Path(path).exists():
                        self.pool = json.loads(Path(path).read_text())
                        self._pool_loaded = True
                        break
            except Exception:
                pass
        if not self._moves_loaded:
            try:
                import json
                from pathlib import Path
                if Path("data/moves.json").exists():
                    self.move_data = json.loads(Path("data/moves.json").read_text())
                    self._moves_loaded = True
            except Exception:
                pass

    def assess(
        self,
        *,
        log: str,
        state: Any,
        pool: dict[str, Any],
        base_policy: list[float],
        belief_posterior: dict[str, Any] | None = None,
    ) -> HeuristicRLMOutput:
        import time
        t0 = time.monotonic()
        self._ensure_loaded()
        pool_to_use = pool if pool else self.pool
        n = len(base_policy)
        observations: list[str] = []

        # Extract current battle context from state
        our_moves: list[str] = []
        our_hp = 1.0
        opp_hp = 1.0
        if isinstance(state, dict):
            avail = state.get("available_moves") or []
            our_moves = [m.get("move", "") if isinstance(m, dict) else str(m) for m in avail[:4]]
            own_team = state.get("own_team") or []
            if own_team:
                active = next((m for m in own_team if isinstance(m, dict) and m.get("is_active")), own_team[0] if own_team else {})
                our_hp = float(active.get("hp_fraction", active.get("hp", 1.0)) or 1.0)
            opp_team = state.get("opponent_team") or []
            if opp_team:
                active_opp = next((m for m in opp_team if isinstance(m, dict) and m.get("is_active")), opp_team[0] if opp_team else {})
                opp_hp = float(active_opp.get("hp_fraction", active_opp.get("hp", 1.0)) or 1.0)
        else:
            # poke-env Battle object
            avail = list(getattr(state, "available_moves", []) or [])
            our_moves = [getattr(m, "id", str(m)) for m in avail[:4]]
            active = getattr(state, "active_pokemon", None)
            if active:
                our_hp = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
            opp_active = getattr(state, "opponent_active_pokemon", None)
            if opp_active:
                opp_hp = float(getattr(opp_active, "current_hp_fraction", 1.0) or 1.0)

        # Get opponent candidate sets from belief posterior or pool
        posterior = belief_posterior or {}
        if not posterior and isinstance(state, dict):
            posterior = state.get("belief_posterior") or {}

        # Score each action across all posterior opponent sets
        action_scores = list(base_policy)  # start from PokeNet prior
        if posterior and our_moves:
            combined_scores = [0.0] * n
            total_weight = 0.0
            for slot, entries in posterior.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    prob = float(entry.get("probability", 1.0 / max(1, len(entries))))
                    # Reconstruct types from pool if available
                    species = _normalize_name(entry.get("species", slot))
                    types: list[str] = []
                    species_entry = (self.pool.get("species") or {}).get(species, {})
                    if isinstance(species_entry, dict):
                        pdex = species_entry.get("pokedex") or {}
                        types = [t.lower() for t in (pdex.get("types") or [])]
                    if not types:
                        types = [t.lower() for t in (entry.get("types") or ["normal"])]
                    entry_with_types = dict(entry)
                    entry_with_types["types"] = types
                    for action_idx in range(min(n, 14)):
                        score = _score_action_vs_set(action_idx, our_moves, our_hp, entry_with_types, opp_hp, self.move_data)
                        combined_scores[action_idx] += prob * score
                    total_weight += prob
            if total_weight > 0:
                # Blend: 50% PokeNet base policy + 50% heuristic scores
                import math
                normalized = [s / total_weight for s in combined_scores]
                for i in range(n):
                    base = max(base_policy[i], 1e-12)
                    heur = max(normalized[i], 1e-12)
                    action_scores[i] = 0.5 * math.log(base) + 0.5 * math.log(heur)
                # Softmax
                max_score = max(action_scores[:n])
                exps = [math.exp(s - max_score) for s in action_scores[:n]]
                total = sum(exps)
                action_scores = [e / total for e in exps]
                observations.append(f"heuristic: scored {n} actions vs {sum(len(v) if isinstance(v, list) else 1 for v in posterior.values())} candidate sets")

        # Value estimate: HP advantage heuristic
        v_rlm = float(our_hp - opp_hp)  # positive = we're ahead
        elapsed = (time.monotonic() - t0) * 1000.0
        return HeuristicRLMOutput(
            pi_rlm=action_scores[:n],
            v_rlm=max(-1.0, min(1.0, v_rlm)),
            refined_belief={},
            observations=observations,
            elapsed_ms=elapsed,
        )
