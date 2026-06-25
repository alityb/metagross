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
    import fp.websocket_client as ws_client

    if not hasattr(run_battle, "format_decision") or not callable(run_battle.format_decision):
        raise RuntimeError("Foul Play patch target fp.run_battle.format_decision is missing")

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default

    # Disable websocket keepalive pings so the long MCTS subprocess
    # doesn't cause a keepalive timeout during search.
    import websockets
    _orig_connect = websockets.connect

    def connect_no_ping(address, *args, **kwargs):
        kwargs.setdefault("ping_interval", None)
        return _orig_connect(address, *args, **kwargs)

    ws_client.websockets.connect = connect_no_ping


def extract_value_features(state) -> list[float]:
    """Extract 24 enriched features from a poke_engine State object.

    Delegates to the Rust compute_value_features binding so training and
    inference use EXACTLY the same featurization.
    """
    import poke_engine as _pe
    return _pe.compute_value_features(state)


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
