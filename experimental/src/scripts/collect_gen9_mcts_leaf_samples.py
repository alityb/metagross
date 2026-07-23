#!/usr/bin/env python3
"""Collect exact counterfactual terminal outcomes from deep Gen9 MCTS states."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "external" / "foul-play")]

from belief.randbats_determinize import RandbatsDeterminizer

_DETERMINIZER = None


def determinize_leaf_state(state, poke_engine):
    """Fill only hidden opponents; keep the search leaf and input features public."""
    global _DETERMINIZER
    unknown = [pokemon for pokemon in state.side_two.pokemon if pokemon.id == "NONE"]
    if not unknown:
        return state
    if _DETERMINIZER is None:
        _DETERMINIZER = RandbatsDeterminizer(
            REPO_ROOT / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json"
        )
    revealed = [pokemon for pokemon in state.side_two.pokemon if pokemon.id != "NONE"]
    team = _DETERMINIZER.sample_team(revealed)
    if team is None:
        return None

    from fp.battle import Move, Pokemon
    from fp.search.poke_engine_helpers import pokemon_to_poke_engine_pkmn

    revealed_ids = {pokemon.id.lower() for pokemon in revealed}
    hidden_sets = [set_ for set_ in team if set_["speciesId"].lower() not in revealed_ids]
    if len(hidden_sets) != len(unknown):
        return None
    replacements = iter(hidden_sets)
    pokemon = []
    for current in state.side_two.pokemon:
        if current.id != "NONE":
            pokemon.append(current)
            continue
        set_ = next(replacements)
        generated = Pokemon(
            set_["speciesId"],
            set_["level"],
            evs=tuple(set_["evs"].get(stat, 85) for stat in ("hp", "atk", "def", "spa", "spd", "spe")),
        )
        generated.ability = set_["ability"]
        generated.original_ability = set_["ability"]
        generated.item = set_["item"]
        generated.tera_type = set_["teraType"]
        generated.moves = [Move(move) for move in set_["moves"]]
        pokemon.append(pokemon_to_poke_engine_pkmn(generated))

    side = state.side_two
    completed_side = poke_engine.Side(
        pokemon=pokemon,
        side_conditions=side.side_conditions,
        active_index=side.active_index,
        baton_passing=side.baton_passing,
        shed_tailing=side.shed_tailing,
        volatile_status_durations=side.volatile_status_durations,
        wish=side.wish,
        future_sight=side.future_sight,
        force_switch=side.force_switch,
        force_trapped=side.force_trapped,
        slow_uturn_move=side.slow_uturn_move,
        volatile_statuses=side.volatile_statuses,
        substitute_health=side.substitute_health,
        attack_boost=side.attack_boost,
        defense_boost=side.defense_boost,
        special_attack_boost=side.special_attack_boost,
        special_defense_boost=side.special_defense_boost,
        speed_boost=side.speed_boost,
        accuracy_boost=side.accuracy_boost,
        evasion_boost=side.evasion_boost,
        last_used_move=side.last_used_move,
        switch_out_move_second_saved_move=side.switch_out_move_second_saved_move,
    )
    return poke_engine.State(
        side_one=state.side_one,
        side_two=completed_side,
        weather=state.weather,
        weather_turns_remaining=state.weather_turns_remaining,
        terrain=state.terrain,
        terrain_turns_remaining=state.terrain_turns_remaining,
        trick_room=state.trick_room,
        trick_room_turns_remaining=state.trick_room_turns_remaining,
        team_preview=state.team_preview,
        s1_threat=state.s1_threat,
        s2_threat=state.s2_threat,
        scout_value=state.scout_value,
        threat_matrix=state.threat_matrix,
        wincon_matrix=state.wincon_matrix,
    )


def load_roots(paths: list[Path], limit: int):
    rng = random.Random(0)
    seen = set()
    roots = []
    root_count = 0
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") != "decision" or not isinstance(row.get("state"), str):
                continue
            key = (str(row.get("battle_tag")), str(row.get("username")))
            if key in seen:
                continue
            seen.add(key)
            root_count += 1
            candidate = (key[0], key[1], line_no, row["state"])
            if len(roots) < limit:
                roots.append(candidate)
            else:
                replacement = rng.randrange(root_count)
                if replacement < limit:
                    roots[replacement] = candidate
    return roots


def collect(root):
    import poke_engine

    battle_tag, username, line_no, state_text, iterations, rollout_iterations, max_decisions = root
    state = poke_engine.State.from_string(state_text)
    sample = poke_engine.mcts_leaf_state_sample(
        state,
        iterations=iterations,
    )
    if sample is None:
        return None
    root_features, leaf_features, leaf_state_text, depth, terminal_visits, all_visits = sample
    completed_leaf = determinize_leaf_state(poke_engine.State.from_string(leaf_state_text), poke_engine)
    if completed_leaf is None:
        return None
    target = poke_engine.mcts_rollout_to_terminal(
        completed_leaf,
        rollout_iterations=rollout_iterations,
        max_decisions=max_decisions,
    )
    return {
        "schema": 2,
        "battle_tag": battle_tag,
        "username": username,
        "source_line": line_no,
        "root_features": root_features,
        "leaf_features": leaf_features,
        "target": target,
        "completed": target is not None,
        "depth": depth,
        "terminal_visits": terminal_visits,
        "all_visits": all_visits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=200000)
    parser.add_argument("--rollout-iterations", type=int, default=1000)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    roots = [
        (
            tag,
            user,
            line,
            state,
            args.iterations,
            args.rollout_iterations,
            args.max_decisions,
        )
        for tag, user, line, state in load_roots(args.decision_log, args.max_roots)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with args.output.open("w", encoding="utf-8") as output:
        if args.workers == 1:
            results = map(collect, roots)
        else:
            with mp.Pool(args.workers) as pool:
                results = pool.imap_unordered(collect, roots)
                for sample in results:
                    if sample is not None:
                        output.write(json.dumps(sample, separators=(",", ":")) + "\n")
                        emitted += 1
                results = ()
        for sample in results:
            if sample is not None:
                output.write(json.dumps(sample, separators=(",", ":")) + "\n")
                emitted += 1
    print(json.dumps({
        "roots": len(roots),
        "samples": emitted,
        "iterations": args.iterations,
        "rollout_iterations": args.rollout_iterations,
        "max_decisions": args.max_decisions,
    }))


if __name__ == "__main__":
    main()
