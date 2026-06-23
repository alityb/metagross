#!/usr/bin/env python3
import asyncio
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from pathlib import Path


def patch_foul_play_protocol_bugs() -> None:
    import fp.run_battle as run_battle

    if not hasattr(run_battle, "format_decision") or not callable(run_battle.format_decision):
        raise RuntimeError("Foul Play patch target fp.run_battle.format_decision is missing")

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default


def extract_value_features(state) -> list[float]:
    def hp_fraction(pokemon) -> float:
        if pokemon is None or pokemon.hp <= 0 or pokemon.maxhp <= 0:
            return 0.0
        return max(0.0, min(1.0, pokemon.hp / pokemon.maxhp))

    def active(side):
        return side.pokemon[int(side.active_index)]

    def alive(side) -> list:
        return [pokemon for pokemon in side.pokemon if pokemon.hp > 0]

    def hp_total(side) -> float:
        return sum(hp_fraction(pokemon) for pokemon in side.pokemon) / 6.0

    def alive_fraction(side) -> float:
        return len(alive(side)) / 6.0

    def status_fraction(side) -> float:
        return sum(1 for pokemon in alive(side) if pokemon.status.lower() != "none") / 6.0

    def item_fraction(side) -> float:
        return sum(1 for pokemon in alive(side) if pokemon.item.lower() != "none") / 6.0

    def used_tera(side) -> float:
        return 1.0 if any(pokemon.terastallized for pokemon in side.pokemon) else 0.0

    def screen_score(side) -> float:
        conditions = side.side_conditions
        return (conditions.reflect + conditions.light_screen + conditions.aurora_veil * 2) / 8.0

    def hazard_score(side) -> float:
        conditions = side.side_conditions
        return (
            conditions.stealth_rock
            + conditions.spikes
            + conditions.toxic_spikes
            + conditions.sticky_web * 2
        ) / 8.0

    def active_stat_total(side) -> float:
        pokemon = active(side)
        return (
            pokemon.attack
            + pokemon.defense
            + pokemon.special_attack
            + pokemon.special_defense
            + pokemon.speed
        ) / 1000.0

    def team_stat_total(side) -> float:
        return (
            sum(
                pokemon.attack
                + pokemon.defense
                + pokemon.special_attack
                + pokemon.special_defense
                + pokemon.speed
                for pokemon in alive(side)
            )
            / 6000.0
        )

    side_one = state.side_one
    side_two = state.side_two
    return [
        hp_total(side_one) - hp_total(side_two),
        alive_fraction(side_one) - alive_fraction(side_two),
        hp_fraction(active(side_one)) - hp_fraction(active(side_two)),
        status_fraction(side_two) - status_fraction(side_one),
        item_fraction(side_one) - item_fraction(side_two),
        used_tera(side_two) - used_tera(side_one),
        side_one.attack_boost / 6.0 - side_two.attack_boost / 6.0,
        side_one.defense_boost / 6.0 - side_two.defense_boost / 6.0,
        side_one.special_attack_boost / 6.0 - side_two.special_attack_boost / 6.0,
        side_one.special_defense_boost / 6.0 - side_two.special_defense_boost / 6.0,
        side_one.speed_boost / 6.0 - side_two.speed_boost / 6.0,
        screen_score(side_one) - screen_score(side_two),
        hazard_score(side_two) - hazard_score(side_one),
        active_stat_total(side_one) - active_stat_total(side_two),
        team_stat_total(side_one) - team_stat_total(side_two),
        (1.0 if side_one.substitute_health > 0 else 0.0)
        - (1.0 if side_two.substitute_health > 0 else 0.0),
    ]


def patch_decision_logging() -> None:
    output_path = os.environ.get("METAGROSS_DECISION_LOG")
    if not output_path:
        return

    import config
    import fp.run_battle as run_battle
    import fp.search.poke_engine_helpers as poke_engine_helpers

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pending_rows = []
    original_async_pick_move = run_battle.async_pick_move
    original_pokemon_battle = run_battle.pokemon_battle

    async def async_pick_move_with_logging(battle):
        start_index = len(pending_rows)
        if not battle.team_preview:
            try:
                battle_copy = deepcopy(battle)
                battle_copy.user.update_from_request_json(battle_copy.request_json)
                state = poke_engine_helpers.battle_to_poke_engine_state(battle_copy)
                pending_rows.append(
                    {
                        "battle_tag": battle.battle_tag,
                        "turn": battle.turn,
                        "username": config.FoulPlayConfig.username,
                        "fixed_side": "side_one",
                        "features": extract_value_features(state),
                        "state": state.to_string(),
                    }
                )
            except Exception as exc:
                pending_rows.append(
                    {
                        "battle_tag": getattr(battle, "battle_tag", None),
                        "turn": getattr(battle, "turn", None),
                        "username": config.FoulPlayConfig.username,
                        "fixed_side": "side_one",
                        "error": f"feature_log_failed: {type(exc).__name__}: {exc}",
                    }
                )
        try:
            return await original_async_pick_move(battle)
        except Exception:
            del pending_rows[start_index:]
            raise

    async def pokemon_battle_with_labels(ps_websocket_client, pokemon_battle_type, team_dict):
        start_index = len(pending_rows)
        winner = await original_pokemon_battle(ps_websocket_client, pokemon_battle_type, team_dict)
        label = 1 if winner == config.FoulPlayConfig.username else 0
        with open(output_path, "a", encoding="utf-8") as handle:
            for row in pending_rows[start_index:]:
                if "features" in row:
                    row = dict(row)
                    row["winner"] = winner
                    row["label"] = label
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        del pending_rows[start_index:]
        return winner

    run_battle.async_pick_move = async_pick_move_with_logging
    run_battle.pokemon_battle = pokemon_battle_with_labels


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    foul_play_dir = Path(os.environ.get("FOUL_PLAY_DIR", root_dir / "external" / "foul-play"))

    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    os.chdir(foul_play_dir)
    sys.path.insert(0, str(foul_play_dir))

    patch_foul_play_protocol_bugs()
    patch_decision_logging()

    from run import run_foul_play

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
