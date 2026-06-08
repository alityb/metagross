from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any

from pipeline.sample_types import TrainingSample
from model.engine_bridge import battle_to_poke_engine_state
from model.state import Vocabulary, build_vocabulary, encode_state, normalize_name


POOL: dict[str, Any] = {}
VOCAB: Vocabulary | None = None
LEARNSETS: dict[str, Any] = {}


def _load_poke_engine() -> Any:
    try:
        import poke_engine  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("poke_engine is required for synthetic generation") from exc
    return poke_engine


def init_worker(pool: dict[str, Any], vocab: Vocabulary, learnsets: dict[str, Any]) -> None:
    global POOL, VOCAB, LEARNSETS
    POOL = pool
    VOCAB = vocab
    LEARNSETS = learnsets


# ------------------------------------------------------------------ #
# Partial observability tracking                                       #
# ------------------------------------------------------------------ #

@dataclass
class BattleObservationState:
    """Tracks what one side has observed about the opponent so far."""
    revealed_slots: set[int]
    revealed_moves: dict[int, set[str]]
    revealed_items: dict[int, str | None]
    revealed_abilities: dict[int, str | None]

    @classmethod
    def initial(cls, active_slot: int) -> "BattleObservationState":
        """Battle starts: only the active slot's species is revealed, no moves seen."""
        return cls(
            revealed_slots={active_slot},
            revealed_moves={active_slot: set()},
            revealed_items={},
            revealed_abilities={},
        )

    def update_after_move(
        self,
        opp_active_slot: int,
        move_used: str | None,
        item_activated: str | None = None,
        ability_triggered: str | None = None,
    ) -> None:
        self.revealed_slots.add(opp_active_slot)
        self.revealed_moves.setdefault(opp_active_slot, set())
        if move_used:
            self.revealed_moves[opp_active_slot].add(move_used)
        if item_activated:
            self.revealed_items[opp_active_slot] = item_activated
        if ability_triggered:
            self.revealed_abilities[opp_active_slot] = ability_triggered

    def update_after_switch(self, new_active_slot: int) -> None:
        """Opponent switches in: new slot species is now revealed."""
        self.revealed_slots.add(new_active_slot)
        self.revealed_moves.setdefault(new_active_slot, set())


def mask_opponent_for_p1(
    full_opp_team: list[dict[str, Any]],
    obs: BattleObservationState,
    active_opp_slot: int,
) -> list[dict[str, Any] | None]:
    """
    Returns the opponent team as the observing player can currently see it.

    - Unrevealed slots → None (encode_state treats None as full UNKNOWN token).
    - Revealed slots → only what has been observed: species known, moves only
      if used, item/ability only if activated/triggered.
    """
    visible: list[dict[str, Any] | None] = []
    for idx, mon in enumerate(full_opp_team):
        if idx not in obs.revealed_slots:
            visible.append(None)  # full UNKNOWN
        else:
            visible.append({
                "species": mon["species"],
                "moves": sorted(obs.revealed_moves.get(idx, set()))[:4],
                "item": obs.revealed_items.get(idx),        # None until activated
                "ability": obs.revealed_abilities.get(idx),  # None until triggered
                "hp_fraction": mon.get("hp_fraction", 1.0),
                "is_active": idx == active_opp_slot,
                "status": mon.get("status"),
                "types": mon.get("types", []),
            })
    return visible


# ------------------------------------------------------------------ #
# Team sampling helpers                                                #
# ------------------------------------------------------------------ #

def stat_calc(base: int, level: int, hp: bool = False, ev: int = 85, iv: int = 31) -> int:
    common = math.floor(((2 * int(base) + iv + math.floor(ev / 4)) * level) / 100)
    return common + level + 10 if hp else common + 5


def stats_from_pokedex(pokedex_entry: dict[str, Any], level: int, set_stats: dict[str, Any] | None = None) -> dict[str, int]:
    if set_stats:
        return {
            "hp": int(set_stats.get("hp", set_stats.get("maxhp", 100))),
            "atk": int(set_stats.get("atk", set_stats.get("attack", 100))),
            "def": int(set_stats.get("def", set_stats.get("defense", 100))),
            "spa": int(set_stats.get("spa", set_stats.get("special_attack", 100))),
            "spd": int(set_stats.get("spd", set_stats.get("special_defense", 100))),
            "spe": int(set_stats.get("spe", set_stats.get("speed", 100))),
        }
    base = pokedex_entry.get("baseStats", {}) if isinstance(pokedex_entry, dict) else {}
    return {
        "hp": stat_calc(base.get("hp", 100), level, hp=True),
        "atk": stat_calc(base.get("atk", 100), level),
        "def": stat_calc(base.get("def", 100), level),
        "spa": stat_calc(base.get("spa", 100), level),
        "spd": stat_calc(base.get("spd", 100), level),
        "spe": stat_calc(base.get("spe", 100), level),
    }


def legal_moves_for_species(species_id: str, gen: int, base_moves: list[str], rng: random.Random) -> list[str]:
    gen_key = f"gen{gen}"
    learnset_moves = list((LEARNSETS.get(species_id, {}) or {}).get(gen_key, []))
    if learnset_moves and rng.random() < 0.35:
        candidates = sorted(set(map(normalize_name, base_moves)) | set(learnset_moves))
    else:
        candidates = [normalize_name(move) for move in base_moves]
    candidates = [move for move in candidates if move and move != "none"]
    return candidates or ["tackle"]


def sample_team(pool: dict[str, Any], gen: int, rng: random.Random) -> list[dict[str, Any]]:
    gen_key = f"gen{gen}"
    eligible = [species_id for species_id, entry in pool["species"].items() if gen_key in (entry.get("sets") or {})]
    if not eligible:
        eligible = list(pool["species"].keys())
    selected = rng.sample(eligible, min(6, len(eligible)))
    team: list[dict[str, Any]] = []
    for species_id in selected:
        entry = pool["species"][species_id]
        gen_sets = (entry.get("sets") or {}).get(gen_key, [])
        if gen_sets:
            base_set = rng.choice(gen_sets)
        else:
            all_sets = [s for sets in (entry.get("sets") or {}).values() for s in sets]
            base_set = rng.choice(all_sets) if all_sets else {}
        all_moves = legal_moves_for_species(species_id, gen, list(base_set.get("moves") or []), rng)
        n_moves = rng.choices([1, 2, 3, 4], weights=[0.05, 0.10, 0.20, 0.65])[0]
        moves = rng.sample(all_moves, min(n_moves, len(all_moves)))
        level = int(base_set.get("level") or (100 if gen >= 6 else 50))
        pokedex_entry = entry.get("pokedex") or {}
        stats = stats_from_pokedex(pokedex_entry, level, base_set.get("stats") or None)
        team.append({
            "species": species_id,
            "moves": moves,
            "item": base_set.get("item", "") if gen >= 2 else "",
            "ability": base_set.get("ability", "") if gen >= 3 else "",
            "level": level,
            "types": pokedex_entry.get("types", ["normal"]),
            "stats": stats,
            "hp": stats["hp"],
            "max_hp": stats["hp"],
            "hp_fraction": 1.0,
            "is_active": False,
        })
    if team:
        team[0]["is_active"] = True
    return team


# ------------------------------------------------------------------ #
# Engine helpers                                                       #
# ------------------------------------------------------------------ #

def teams_to_poke_engine_state(team1: list[dict[str, Any]], team2: list[dict[str, Any]], gen: int) -> Any:
    return battle_to_poke_engine_state({"own_team": team1, "opponent_team": team2, "generation": gen})


def has_healthy_pokemon(side: Any) -> bool:
    return any(getattr(mon, "hp", 0) > 0 for mon in list(getattr(side, "pokemon", []) or []))


def is_terminal(state: Any) -> bool:
    return not has_healthy_pokemon(state.side_one) or not has_healthy_pokemon(state.side_two)


def get_available_engine_moves(side: Any) -> list[str]:
    active = side.pokemon[int(side.active_index)]
    choices = []
    for move in active.moves:
        if not move.disabled and move.id != "none" and move.pp > 0:
            choices.append(str(move.id))
    active_idx = int(side.active_index)
    for idx, mon in enumerate(side.pokemon):
        if idx != active_idx and mon.hp > 0:
            choices.append(f"switch {mon.id}")
    return choices or ["tackle"]


def move_only_choices(side: Any) -> list[str]:
    active = side.pokemon[int(side.active_index)]
    choices = [str(move.id) for move in active.moves if not move.disabled and move.id != "none" and move.pp > 0]
    return choices or ["tackle"]


def engine_move_to_action_index(side: Any, choice: str) -> int:
    active = side.pokemon[int(side.active_index)]
    if choice.startswith("switch "):
        target = choice.split(" ", 1)[1]
        active_idx = int(side.active_index)
        switches = [mon.id for idx, mon in enumerate(side.pokemon) if idx != active_idx and mon.hp > 0]
        return 4 + switches.index(target) if target in switches[:5] else 4
    for idx, move in enumerate(active.moves[:4]):
        if str(move.id) == choice:
            return idx
    return 0


MAX_TURNS = 50


def _hp_fraction_total(side: Any) -> float:
    total_hp = 0.0
    total_max = 0.0
    for mon in list(getattr(side, "pokemon", []) or []):
        hp = float(getattr(mon, "hp", 0))
        maxhp = float(getattr(mon, "maxhp", 1) or 1)
        total_hp += hp
        total_max += maxhp
    return total_hp / max(1.0, total_max)


def _moves_from_choice(choice: str) -> tuple[str | None, bool]:
    """Returns (move_name, is_switch)."""
    if choice.startswith("switch "):
        return None, True
    return choice, False


def _encode_partial_state(
    own_team: list[dict[str, Any]],
    masked_opp: list[dict[str, Any] | None],
    available_choices: list[str],
    turn: int,
    gen: int,
    vocab: Vocabulary,
) -> Any:
    """Encode a state from one player's perspective with masked opponent."""
    move_choices = [c for c in available_choices if not c.startswith("switch ")]
    switch_choices = [c for c in available_choices if c.startswith("switch ")]
    return encode_state(
        {
            "turn": turn,
            "generation": gen,
            "own_team": own_team,
            "opponent_team": masked_opp,
            "available_moves": [{"move": m, "disabled": False} for m in move_choices[:4]],
            "available_switches": [{"species": m.split(" ", 1)[1]} for m in switch_choices[:5]],
        },
        vocab=vocab,
        generation=gen,
    )


# ------------------------------------------------------------------ #
# Main battle runner with partial observability                        #
# ------------------------------------------------------------------ #

def run_random_battle(pool: dict[str, Any], vocab: Vocabulary, gen: int, rng: random.Random) -> list[TrainingSample]:
    poke_engine = _load_poke_engine()
    team1 = sample_team(pool, gen, rng)
    team2 = sample_team(pool, gen, rng)
    state = teams_to_poke_engine_state(team1, team2, gen)

    # Initial active slots
    active_p1 = int(state.side_one.active_index)
    active_p2 = int(state.side_two.active_index)

    # p1 can observe p2's active species; p2 can observe p1's active species.
    # Everything else is UNKNOWN at the start.
    obs_p1 = BattleObservationState.initial(active_p2)
    obs_p2 = BattleObservationState.initial(active_p1)

    samples_p1: list[tuple[Any, int, int]] = []
    samples_p2: list[tuple[Any, int, int]] = []

    for turn in range(MAX_TURNS):
        if is_terminal(state):
            break

        p1_moves = get_available_engine_moves(state.side_one)
        p2_moves = get_available_engine_moves(state.side_two)
        if not p1_moves or not p2_moves:
            break

        p1_choice = rng.choice(p1_moves)
        p2_choice = rng.choice(p2_moves)

        # Current active slots BEFORE applying moves
        active_p2_slot = int(state.side_two.active_index)
        active_p1_slot = int(state.side_one.active_index)

        # Encode with masked opponent view — this is the training signal.
        # Both engine mechanics and the full team are still fully known;
        # only the encoded state seen by the model uses the masked view.
        masked_opp_for_p1 = mask_opponent_for_p1(team2, obs_p1, active_p2_slot)
        masked_opp_for_p2 = mask_opponent_for_p1(team1, obs_p2, active_p1_slot)

        encoded_p1 = _encode_partial_state(team1, masked_opp_for_p1, p1_moves, turn, gen, vocab)
        encoded_p2 = _encode_partial_state(team2, masked_opp_for_p2, p2_moves, turn, gen, vocab)

        action_p1 = engine_move_to_action_index(state.side_one, p1_choice)
        action_p2 = engine_move_to_action_index(state.side_two, p2_choice)

        samples_p1.append((encoded_p1, action_p1, turn))
        samples_p2.append((encoded_p2, action_p2, turn))

        # Apply moves to the engine (uses full information)
        try:
            branches = poke_engine.generate_instructions(state, p1_choice, p2_choice)
        except Exception:
            p1_choice = rng.choice(move_only_choices(state.side_one))
            p2_choice = rng.choice(move_only_choices(state.side_two))
            try:
                branches = poke_engine.generate_instructions(state, p1_choice, p2_choice)
            except Exception:
                break
        if not branches:
            break

        weights = [max(0.0, float(getattr(b, "percentage", 0.0))) for b in branches]
        branch = rng.choices(branches, weights=weights if sum(weights) > 0 else None, k=1)[0]
        state = state.apply_instructions(branch)

        # Update observation state AFTER applying moves.
        # p1 now knows: what move did p2 just use? did p2 switch?
        p2_move_name, p2_is_switch = _moves_from_choice(p2_choice)
        if p2_is_switch:
            new_p2_active = int(state.side_two.active_index)
            obs_p1.update_after_switch(new_p2_active)
        else:
            obs_p1.update_after_move(active_p2_slot, p2_move_name)

        # p2 observes p1 symmetrically
        p1_move_name, p1_is_switch = _moves_from_choice(p1_choice)
        if p1_is_switch:
            new_p1_active = int(state.side_one.active_index)
            obs_p2.update_after_switch(new_p1_active)
        else:
            obs_p2.update_after_move(active_p1_slot, p1_move_name)

    # Outcome determination
    if not has_healthy_pokemon(state.side_two):
        outcome_p1 = 1.0
    elif not has_healthy_pokemon(state.side_one):
        outcome_p1 = -1.0
    else:
        p1_hp = _hp_fraction_total(state.side_one)
        p2_hp = _hp_fraction_total(state.side_two)
        outcome_p1 = 1.0 if p1_hp > p2_hp else -1.0

    battle_id = f"synthetic_gen{gen}_{rng.randint(0, 2**32)}"
    result: list[TrainingSample] = []
    for encoded, action, turn in samples_p1:
        result.append(TrainingSample(battle_id, turn, encoded, action, outcome_p1, {}, gen))
    for encoded, action, turn in samples_p2:
        result.append(TrainingSample(f"{battle_id}_p2", turn, encoded, action, -outcome_p1, {}, gen))
    return result


# ------------------------------------------------------------------ #
# Worker / batch infrastructure                                        #
# ------------------------------------------------------------------ #

def worker(task: tuple[int, int | None, int]) -> list[TrainingSample]:
    index, fixed_gen, seed = task
    rng = random.Random(seed + index * 9973)
    gen = int(fixed_gen or rng.randint(1, 9))
    assert VOCAB is not None
    return run_random_battle(POOL, VOCAB, gen, rng)


def write_batch(output: Path, batch_index: int, samples: list[TrainingSample]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"synthetic_{batch_index:06d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(samples, handle)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random-vs-random synthetic battles with partial observability")
    parser.add_argument("--pool", default="data/all_gen_pool.json")
    parser.add_argument("--learnsets", default="data/learnsets.json")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--n-battles", type=int, default=10_000)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pool_data = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    vocab = build_vocabulary(args.pool)
    learnsets = json.loads(Path(args.learnsets).read_text(encoding="utf-8")) if Path(args.learnsets).exists() else {}
    output = Path(args.output)
    tasks = [(index, args.gen, args.seed) for index in range(args.n_battles)]
    batch_samples: list[TrainingSample] = []
    batch_index = len(list(output.glob("synthetic_*.pkl"))) if output.exists() else 0
    completed = 0
    if args.workers <= 1:
        init_worker(pool_data, vocab, learnsets)
        iterator = map(worker, tasks)
    else:
        pool = mp.Pool(args.workers, initializer=init_worker, initargs=(pool_data, vocab, learnsets))
        iterator = pool.imap_unordered(worker, tasks, chunksize=16)
    try:
        for samples in iterator:
            completed += 1
            batch_samples.extend(samples)
            if completed % 1000 == 0:
                path = write_batch(output, batch_index, batch_samples)
                print(json.dumps({"battles": completed, "samples": len(batch_samples), "path": str(path)}))
                batch_index += 1
                batch_samples = []
        if batch_samples:
            path = write_batch(output, batch_index, batch_samples)
            print(json.dumps({"battles": completed, "samples": len(batch_samples), "path": str(path)}))
    finally:
        if args.workers > 1 and "pool" in locals():
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
