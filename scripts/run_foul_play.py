#!/usr/bin/env python3
import asyncio
import atexit
import json
import multiprocessing as mp
import os
import random
import select
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path


def patch_foul_play_protocol_bugs() -> None:
    import fp.run_battle as run_battle
    import fp.battle as battle_module
    import fp.websocket_client as ws_client
    import constants

    if not hasattr(run_battle, "format_decision") or not callable(run_battle.format_decision):
        raise RuntimeError("Foul Play patch target fp.run_battle.format_decision is missing")

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default

    # Gen1 Struggle patch: when PP is exhausted, Showdown sends "fight" as the
    # move name. FP doesn't know this move and crashes in update_from_request_json.
    # Patch _initialize_user_active_from_request_json to skip unknown moves
    # gracefully so the game can continue (the bot will use /choose default).
    _orig_init_active = battle_module.Battler._initialize_user_active_from_request_json

    def _safe_init_active(self, request_json):
        try:
            _orig_init_active(self, request_json)
        except (IndexError, KeyError, ValueError):
            # PP-exhausted state: Showdown sent "fight" or empty move list.
            # Leave move list unchanged so async_pick_move can fall back to default.
            pass

    battle_module.Battler._initialize_user_active_from_request_json = _safe_init_active

    # Gen1 speed range patch: check_speed_ranges looks up the last selected move
    # in all_move_json, but "nomove" (no move selected at game start / forced switch)
    # is not in the moves database. Patch to skip gracefully.
    import fp.battle_modifier as battle_modifier_module
    _orig_check_speed = battle_modifier_module.check_speed_ranges

    def _safe_check_speed(battle, msg_lines):
        try:
            _orig_check_speed(battle, msg_lines)
        except (KeyError, AttributeError, TypeError):
            pass

    battle_modifier_module.check_speed_ranges = _safe_check_speed

    # Local-server login bypass: FP's login() always calls play.pokemonshowdown.com
    # to get an assertion token, which fails with no network or on local-only machines.
    # For local Showdown servers (--no-security), any assertion string works including
    # the raw challstr itself. Detect local server from sys.argv since FoulPlayConfig
    # may not be initialized yet at patch time.
    import sys as _sys
    _ws_uri = next((a for i, a in enumerate(_sys.argv)
                    if i > 0 and _sys.argv[i-1] == "--websocket-uri"), "")
    if "localhost" in _ws_uri or "127.0.0.1" in _ws_uri:
        import asyncio as _asyncio
        import fp.websocket_client as ws_client2

        _orig_login = ws_client2.PSWebsocketClient.login

        async def _local_login(self):
            import logging as _lg
            _lg.getLogger("fp.local_login").info("Logging in (local server bypass)...")
            _client_id, _challstr = await self.get_id_and_challstr()
            # With --no-security (noguestsecurity), Showdown accepts empty assertion.
            # Sending the raw challstr as assertion fails validation; empty string works.
            await self.send_message("", [f"/trn {self.username},0,"])
            _lg.getLogger("fp.local_login").info("Successfully logged in")
            await _asyncio.sleep(2)
            return self.username

        ws_client2.PSWebsocketClient.login = _local_login

    # Disable websocket keepalive pings so the long MCTS subprocess
    # doesn't cause a keepalive timeout during search.
    import websockets
    _orig_connect = websockets.connect

    def connect_no_ping(address, *args, **kwargs):
        kwargs.setdefault("ping_interval", None)
        return _orig_connect(address, *args, **kwargs)

    ws_client.websockets.connect = connect_no_ping

    # Intercept receive_message to print |raw| rating lines to stdout
    # so the ladder runner can parse ELO without requiring DEBUG logging.
    import logging as _logging
    _logger = _logging.getLogger("fp.rating_intercept")
    original_receive = ws_client.PSWebsocketClient.receive_message

    async def receive_message_with_rating_log(self):
        message = await original_receive(self)
        # The rating line appears as: "|raw|USERNAME's rating: N → <strong>M</strong>..."
        # It's embedded in a multi-line message; scan each line.
        for line in message.splitlines():
            if line.startswith("|raw|") and ("<strong>" in line or "rating:" in line.lower()):
                import sys
                print(f"RATING_LINE {line}", file=sys.stdout, flush=True)
        return message

    ws_client.PSWebsocketClient.receive_message = receive_message_with_rating_log


def extract_value_features(state) -> list[float]:
    """Extract 24 enriched features from a poke_engine State object.

    Delegates to the Rust compute_value_features binding so training and
    inference use EXACTLY the same featurization.
    """
    import poke_engine as _pe
    return _pe.compute_value_features(state)


def patch_randbats_generator_belief() -> None:
    pool_path = os.environ.get("METAGROSS_RANDBATS_POOL")
    conditional_script = os.environ.get("METAGROSS_RANDBATS_CONDITIONAL_SCRIPT")
    if not pool_path and not conditional_script:
        return

    import constants
    import fp.search.main as search_main
    import fp.search.random_battles as random_battles
    from data.pkmn_sets import PokemonMoveset, PokemonSet, PredictedPokemonSet
    from fp.battle import Pokemon
    from fp.helpers import normalize_name
    from fp.search.helpers import populate_pkmn_from_set

    root_dir = Path(__file__).resolve().parents[1]
    format_name = os.environ.get("METAGROSS_RANDBATS_FORMAT")
    if not format_name:
        format_name = next(
            (arg for idx, arg in enumerate(sys.argv) if idx > 0 and sys.argv[idx - 1] == "--pokemon-format"),
            "gen9randombattle",
        )

    def norm(value: object) -> str:
        return normalize_name(str(value or ""))

    def ev_tuple(evs: object) -> tuple[int, int, int, int, int, int]:
        if not isinstance(evs, dict):
            return (85, 85, 85, 85, 85, 85)
        return tuple(int(evs.get(stat, 85)) for stat in ("hp", "atk", "def", "spa", "spd", "spe"))  # type: ignore[return-value]

    def normalize_set(raw_set: dict) -> dict:
        moves = tuple(norm(mv) for mv in raw_set.get("moves", []))
        return {
            "species": norm(raw_set.get("speciesId") or raw_set.get("species")),
            "level": int(raw_set.get("level") or 100),
            "moves": moves,
            "ability": norm(raw_set.get("ability") or "noability"),
            "item": norm(raw_set.get("item") or "none"),
            "tera_type": norm(raw_set.get("teraType") or "typeless"),
            "evs": ev_tuple(raw_set.get("evs")),
        }

    def normalize_teams(raw_teams: list) -> list[dict]:
        pool = []
        for raw_team in raw_teams:
            normalized_team = [normalize_set(raw_set) for raw_set in raw_team]
            by_species = {raw_set["species"]: raw_set for raw_set in normalized_team}
            if len(normalized_team) == 6 and len(by_species) == 6:
                pool.append({"sets": normalized_team, "by_species": by_species})
        return pool

    pool = []
    if pool_path:
        pool_path = str(Path(pool_path).resolve())
        payload = json.loads(Path(pool_path).read_text(encoding="utf-8"))
        raw_teams = payload.get("teams", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_teams, list) or not raw_teams:
            raise RuntimeError(f"Randbats pool is empty or invalid: {pool_path}")
        pool = normalize_teams(raw_teams)
        if not pool:
            raise RuntimeError(f"Randbats pool has no usable six-Pokemon teams: {pool_path}")

    conditional_script_path = Path(conditional_script).resolve() if conditional_script else None
    conditional_samples = int(os.environ.get("METAGROSS_RANDBATS_CONDITIONAL_SAMPLES", "24"))
    conditional_max_teams = int(os.environ.get("METAGROSS_RANDBATS_CONDITIONAL_MAX_TEAMS", "30000"))
    conditional_timeout_s = float(os.environ.get("METAGROSS_RANDBATS_CONDITIONAL_TIMEOUT_S", "8"))
    conditional_cache: dict[str, list[dict]] = {}
    conditional_lock = threading.Lock()
    conditional_proc: subprocess.Popen | None = None

    def stop_conditional_sampler() -> None:
        nonlocal conditional_proc
        if conditional_proc is None:
            return
        proc = conditional_proc
        conditional_proc = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def start_conditional_sampler() -> subprocess.Popen | None:
        nonlocal conditional_proc
        if conditional_script_path is None:
            return None
        if conditional_proc is not None and conditional_proc.poll() is None:
            return conditional_proc
        conditional_proc = subprocess.Popen(
            [
                "node",
                str(conditional_script_path),
                "--format",
                format_name,
                "--server",
                "--samples",
                str(conditional_samples),
                "--max-teams",
                str(conditional_max_teams),
            ],
            cwd=str(root_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return conditional_proc

    atexit.register(stop_conditional_sampler)

    original_prepare_random_battles = random_battles.prepare_random_battles

    def pokemon_species_keys(pkmn: Pokemon) -> set[str]:
        keys = {pkmn.name, pkmn.base_name, pkmn.get_species()}
        return {key for key in keys if key}

    def move_matches(known_move: str, candidate_moves: tuple[str, ...]) -> bool:
        if known_move in candidate_moves:
            return True
        if known_move.startswith(constants.HIDDEN_POWER):
            return any(move.startswith(constants.HIDDEN_POWER) for move in candidate_moves)
        if known_move.startswith("return"):
            return any(move.startswith("return") for move in candidate_moves)
        return False

    def set_matches_pokemon(pkmn: Pokemon, candidate: dict) -> bool:
        if candidate["species"] not in pokemon_species_keys(pkmn):
            return False
        for mv in pkmn.moves:
            if not move_matches(mv.name, candidate["moves"]):
                return False
        known_item = pkmn.removed_item or pkmn.item
        if known_item not in {None, constants.UNKNOWN_ITEM} and known_item != candidate["item"]:
            return False
        if pkmn.ability is not None and pkmn.ability != candidate["ability"]:
            return False
        if pkmn.tera_type and pkmn.tera_type not in {"nothing", "typeless"}:
            if pkmn.tera_type != candidate["tera_type"]:
                return False
        return True

    def revealed_opponent_pokemon(battle) -> list[Pokemon]:
        revealed = []
        if battle.opponent.active is not None:
            revealed.append(battle.opponent.active)
        revealed.extend(battle.opponent.reserve)
        return revealed

    def concrete_item(pkmn: Pokemon) -> str | None:
        item = pkmn.removed_item or pkmn.item
        if item in {None, constants.UNKNOWN_ITEM, "none"}:
            return None
        return item

    def concrete_tera_type(pkmn: Pokemon) -> str | None:
        if pkmn.tera_type and pkmn.tera_type not in {"nothing", "typeless"}:
            return pkmn.tera_type
        return None

    def constraints_for_battle(battle) -> list[dict]:
        constraints = []
        for pkmn in revealed_opponent_pokemon(battle):
            constraints.append(
                {
                    "speciesKeys": sorted(pokemon_species_keys(pkmn)),
                    "moves": sorted(mv.name for mv in pkmn.moves),
                    "item": concrete_item(pkmn),
                    "ability": pkmn.ability,
                    "teraType": concrete_tera_type(pkmn),
                }
            )
        return constraints

    def constraint_signature(battle) -> str:
        return json.dumps(constraints_for_battle(battle), sort_keys=True, separators=(",", ":"))

    def conditional_pool_for_battle(battle, needed: int) -> list[dict]:
        if conditional_script_path is None:
            return []
        signature = constraint_signature(battle)
        cached = conditional_cache.get(signature)
        if cached:
            return cached
        target_samples = max(needed, conditional_samples)
        with conditional_lock:
            proc = start_conditional_sampler()
            if proc is None or proc.stdin is None or proc.stdout is None:
                conditional_cache[signature] = []
                return []
            try:
                proc.stdin.write(
                    json.dumps(
                        {
                            "constraints": constraints_for_battle(battle),
                            "samples": target_samples,
                            "maxTeams": conditional_max_teams,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                proc.stdin.flush()
                readable, _, _ = select.select([proc.stdout], [], [], conditional_timeout_s)
                if not readable:
                    stop_conditional_sampler()
                    conditional_cache[signature] = []
                    return []
                line = proc.stdout.readline()
            except (BrokenPipeError, OSError):
                stop_conditional_sampler()
                conditional_cache[signature] = []
                return []
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            conditional_cache[signature] = []
            return []
        if payload.get("error"):
            conditional_cache[signature] = []
            return []
        raw_teams = payload.get("teams", [])
        teams = normalize_teams(raw_teams) if isinstance(raw_teams, list) else []
        conditional_cache[signature] = teams
        return teams

    def find_candidate_for_pokemon(pkmn: Pokemon, team: dict) -> dict | None:
        for key in pokemon_species_keys(pkmn):
            candidate = team["by_species"].get(key)
            if candidate is not None and set_matches_pokemon(pkmn, candidate):
                return candidate
        return None

    def team_matches_battle(team: dict, battle) -> bool:
        for pkmn in revealed_opponent_pokemon(battle):
            if find_candidate_for_pokemon(pkmn, team) is None:
                return False
        return True

    def sampled_team_for_battle(battle) -> dict | None:
        revealed = revealed_opponent_pokemon(battle)
        conditional_pool = conditional_pool_for_battle(battle, 1)
        if conditional_pool:
            return random.choice(conditional_pool)
        if not pool:
            return None
        if not revealed:
            return random.choice(pool)
        matches = [team for team in pool if team_matches_battle(team, battle)]
        if not matches:
            return None
        return random.choice(matches)

    def predicted_set_from_candidate(candidate: dict) -> PredictedPokemonSet:
        return PredictedPokemonSet(
            pkmn_set=PokemonSet(
                ability=candidate["ability"],
                item=candidate["item"],
                nature="serious",
                evs=candidate["evs"],
                count=1,
                level=candidate["level"],
                tera_type=candidate["tera_type"],
            ),
            pkmn_moveset=PokemonMoveset(moves=candidate["moves"]),
        )

    def populate_battle_from_team(battle, team: dict) -> bool:
        used_species = set()
        for pkmn in revealed_opponent_pokemon(battle):
            candidate = find_candidate_for_pokemon(pkmn, team)
            if candidate is None:
                return False
            used_species.add(candidate["species"])
            populate_pkmn_from_set(
                pkmn,
                predicted_set_from_candidate(candidate),
                source="generator_pool_revealed",
            )
        while len(revealed_opponent_pokemon(battle)) < 6:
            remaining = [candidate for candidate in team["sets"] if candidate["species"] not in used_species]
            if not remaining:
                return False
            candidate = remaining[0]
            used_species.add(candidate["species"])
            pkmn = Pokemon(candidate["species"], candidate["level"])
            populate_pkmn_from_set(
                pkmn,
                predicted_set_from_candidate(candidate),
                source="generator_pool_unrevealed",
            )
            battle.opponent.reserve.append(pkmn)
        return True

    def prepare_generator_pool_random_battles(battle, num_battles: int):
        sampled_battles = []
        fallback_count = 0
        for _index in range(num_battles):
            battle_copy = deepcopy(battle)
            team = sampled_team_for_battle(battle_copy)
            if team is None or not populate_battle_from_team(battle_copy, team):
                fallback_count += 1
                continue
            battle_copy.opponent.lock_moves()
            sampled_battles.append((battle_copy, 1 / num_battles))
        if not sampled_battles:
            return original_prepare_random_battles(battle, num_battles)
        if fallback_count:
            fallback_battles = original_prepare_random_battles(battle, fallback_count)
            sampled_battles.extend((b, 1 / num_battles) for b, _chance in fallback_battles)
        return sampled_battles

    random_battles.prepare_random_battles = prepare_generator_pool_random_battles
    search_main.prepare_random_battles = prepare_generator_pool_random_battles


def patch_decision_logging() -> None:
    output_path = os.environ.get("METAGROSS_DECISION_LOG")
    if not output_path:
        return

    import config
    import fp.run_battle as run_battle
    import fp.search.main as search_main
    import fp.search.poke_engine_helpers as poke_engine_helpers

    # Guard: don't patch twice (would cause recursion in select_and_capture)
    if getattr(search_main, "_metagross_patched", False):
        return
    search_main._metagross_patched = True

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pending_rows = []
    original_async_pick_move = run_battle.async_pick_move
    original_pokemon_battle = run_battle.pokemon_battle
    original_select = search_main.select_move_from_mcts_results

    import threading
    _policy_store: dict = {}
    _store_lock = threading.Lock()

    def select_and_capture(mcts_results):
        # Capture MCTS visit distributions. select_move_from_mcts_results runs in the
        # MAIN process after futures are collected, so no pickling required.
        # Each element: (MctsResult, sample_chance, index)
        try:
            agg: dict[str, float] = {}
            total = 0
            for mcts_result, chance, _idx in mcts_results:
                tv = mcts_result.total_visits
                total += tv
                for opt in mcts_result.side_one:
                    move = str(opt.move_choice)
                    frac = opt.visits / tv if tv > 0 else 0.0
                    agg[move] = agg.get(move, 0.0) + chance * frac
            with _store_lock:
                if agg:
                    _policy_store['__last__'] = {'visits': agg, 'total': total}
        except Exception:
            pass
        return original_select(mcts_results)

    search_main.select_move_from_mcts_results = select_and_capture

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
            result = await original_async_pick_move(battle)
            # Attach MCTS visit distribution if captured
            if pending_rows and len(pending_rows) > start_index:
                row = pending_rows[-1]
                if "features" in row:
                    with _store_lock:
                        captured = _policy_store.pop('__last__', None)
                    if captured:
                        row["mcts_visits"] = captured['visits']
                        row["mcts_total"] = captured['total']
            return result
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

    # run.py imports pokemon_battle by value: `from fp.run_battle import pokemon_battle`
    # run_battle.py calls async_pick_move by local name (not via module attribute).
    # The ONLY place we can intercept is at find_best_move in run_battle.py's namespace,
    # since it's imported by value there too but we can wrap it after import.
    #
    # Wrap run_battle.find_best_move to capture visit distributions from MctsResults.
    # This is safe because find_best_move runs in a ThreadPoolExecutor (same process).
    original_find_best = run_battle.find_best_move

    def find_best_move_capturing(battle):
        """Wrap find_best_move to capture MCTS visit distribution."""
        result = original_find_best(battle)
        # select_and_capture already ran inside find_best_move (via our search_main patch)
        # and stored results in _policy_store. Nothing extra to do here.
        return result

    run_battle.find_best_move = find_best_move_capturing

    # Patch run.py's pokemon_battle reference after it's imported
    import sys as _sys
    import builtins as _builtins
    _orig_builtin_import = _builtins.__import__
    def _import_hook(name, *args, **kwargs):
        mod = _orig_builtin_import(name, *args, **kwargs)
        if name == 'run' and hasattr(mod, 'pokemon_battle') and \
                mod.pokemon_battle is not pokemon_battle_with_labels:
            mod.pokemon_battle = pokemon_battle_with_labels
        return mod
    _builtins.__import__ = _import_hook

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
    patch_randbats_generator_belief()
    patch_decision_logging()

    from run import run_foul_play

    # Patch run.pokemon_battle AFTER import — run.py imports pokemon_battle by value
    # so module-attribute patching in patch_decision_logging() doesn't affect it.
    # We must update run.pokemon_battle directly after import.
    if os.environ.get("METAGROSS_DECISION_LOG"):
        import run as _run_module
        import fp.run_battle as _rb
        # run.py has `from fp.run_battle import pokemon_battle` — imported by value.
        # _rb.pokemon_battle is now our wrapper; run.pokemon_battle is still the original.
        # Force-update run.pokemon_battle to our wrapper.
        _run_module.pokemon_battle = _rb.pokemon_battle
    
    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
