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


SLEEP_MOVES = {"sleeppowder", "hypnosis", "lovelykiss", "sing", "spore"}
PARALYSIS_MOVES = {"thunderwave", "bodyslam", "stunspore", "glare"}
BOOM_MOVES = {"explosion", "selfdestruct"}
RECOVERY_MOVES = {"recover", "softboiled", "rest"}


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


def patch_tauros_action_kind_gate() -> None:
    model_path = os.environ.get("METAGROSS_TAUROS_KIND_MODEL")
    if not model_path:
        return

    import math
    import fp.run_battle as run_battle
    import fp.search.main as search_main

    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    classes = model["classes"]
    vocab = model["vocab"]
    numeric_fields = model["numeric_fields"]
    weights = model["weight"]
    bias = model["bias"]
    threshold = float(os.environ.get("METAGROSS_TAUROS_KIND_THRESHOLD", "0.70"))
    min_policy_frac = float(os.environ.get("METAGROSS_TAUROS_KIND_MIN_POLICY_FRAC", "0.10"))
    allowed_kinds_raw = os.environ.get("METAGROSS_TAUROS_KIND_ALLOWED_KINDS", "attack_or_other,boom,paralysis,recovery,sleep,switch")
    allowed_kinds = {kind.strip() for kind in allowed_kinds_raw.split(",") if kind.strip()}
    gate_log_path = os.environ.get("METAGROSS_TAUROS_KIND_LOG")
    _current_battle = {"battle": None}

    original_async_pick_move = run_battle.async_pick_move

    def hp_bin(value):
        try:
            hp = float(value)
        except (TypeError, ValueError):
            return "unknown"
        if hp <= 0:
            return "0"
        if hp <= 0.25:
            return "1-25"
        if hp <= 0.5:
            return "26-50"
        if hp <= 0.75:
            return "51-75"
        return "76-100"

    def mon_name(mon):
        return getattr(mon, "name", "none") if mon is not None else "none"

    def mon_hp(mon):
        if mon is None or getattr(mon, "max_hp", 0) == 0:
            return 0.0
        return max(0.0, min(1.0, mon.hp / mon.max_hp))

    def mon_status(mon):
        status = getattr(mon, "status", None)
        return str(status).lower() if status else "none"

    def mon_moves(mon):
        return [move.name for move in getattr(mon, "moves", [])] if mon is not None else []

    def alive_count(battler):
        mons = list(getattr(battler, "reserve", []))
        if getattr(battler, "active", None) is not None:
            mons.append(battler.active)
        return sum(1 for mon in mons if getattr(mon, "hp", 0) > 0)

    def pre_action_bucket(battle):
        active = battle.user.active
        opponent = battle.opponent.active
        active_name = mon_name(active)
        opponent_name = mon_name(opponent)
        active_moves = set(mon_moves(active))
        player_alive = alive_count(battle.user)
        opponent_alive = alive_count(battle.opponent)
        if getattr(battle, "force_switch", False):
            return "forced_switch"
        if player_alive <= 2 or opponent_alive <= 2:
            if active_name == "tauros" or opponent_name == "tauros":
                return "tauros_endgame"
            return "low_hp_endgame"
        if active_moves & SLEEP_MOVES:
            return "sleep_pressure"
        if active_moves & PARALYSIS_MOVES:
            return "paralysis_spread"
        if active_moves & BOOM_MOVES:
            return "explosion_opportunity"
        if active_name == "chansey" and opponent_name == "chansey":
            return "chansey_mirror"
        if active_name == "snorlax" or opponent_name == "snorlax":
            return "snorlax_trade"
        if active_moves & RECOVERY_MOVES:
            return "recovery_loop"
        if mon_status(active) != "none":
            return "statused_active"
        return "other"

    def token_features(battle):
        active = battle.user.active
        opponent = battle.opponent.active
        active_hp = mon_hp(active)
        opponent_hp = mon_hp(opponent)
        player_alive = alive_count(battle.user)
        opponent_alive = alive_count(battle.opponent)
        active_moves = mon_moves(active)
        opponent_moves = mon_moves(opponent)
        tokens = [
            f"bucket={pre_action_bucket(battle)}",
            f"active={mon_name(active)}",
            f"opponent={mon_name(opponent)}",
            f"active_status={mon_status(active)}",
            f"opponent_status={mon_status(opponent)}",
            f"player_alive={player_alive}",
            f"opponent_alive={opponent_alive}",
            f"forced_switch={getattr(battle, 'force_switch', False)}",
            f"has_sleep={bool(set(active_moves) & SLEEP_MOVES)}",
            f"has_para={bool(set(active_moves) & PARALYSIS_MOVES)}",
            f"has_boom={bool(set(active_moves) & BOOM_MOVES)}",
            f"has_recovery={bool(set(active_moves) & RECOVERY_MOVES)}",
            f"active_hp_bin={hp_bin(active_hp)}",
            f"opponent_hp_bin={hp_bin(opponent_hp)}",
        ]
        tokens.extend(f"active_move={move}" for move in active_moves)
        tokens.extend(f"opponent_move={move}" for move in opponent_moves)
        numeric = {
            "active_hp": active_hp,
            "opponent_hp": opponent_hp,
            "player_alive": player_alive,
            "opponent_alive": opponent_alive,
            "turn_index": getattr(battle, "turn", 0) / 200.0,
        }
        return tokens, numeric

    def predict_kind(battle):
        if battle is None or battle.user.active is None:
            return None, 0.0
        tokens, numeric = token_features(battle)
        x = [0.0 for _ in range(len(vocab) + len(numeric_fields))]
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                x[idx] = 1.0
        offset = len(vocab)
        for i, field in enumerate(numeric_fields):
            x[offset + i] = float(numeric.get(field, 0.0))
        logits = []
        for class_idx in range(len(classes)):
            logits.append(sum(w * value for w, value in zip(weights[class_idx], x)) + bias[class_idx])
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        denom = sum(exps)
        probs = [value / denom for value in exps]
        best_idx = max(range(len(classes)), key=lambda idx: probs[idx])
        return classes[best_idx], probs[best_idx]

    def choice_kind(choice) -> str:
        value = str(choice).lower()
        if value.startswith("switch "):
            return "switch"
        if value in SLEEP_MOVES:
            return "sleep"
        if value in PARALYSIS_MOVES:
            return "paralysis"
        if value in BOOM_MOVES:
            return "boom"
        if value in RECOVERY_MOVES:
            return "recovery"
        return "attack_or_other"

    def exact_label_to_choice(label: str) -> str | None:
        if label.startswith("move:"):
            return label.split(":", 1)[1]
        if label.startswith("switch:"):
            return "switch " + label.split(":", 1)[1]
        return None

    def final_policy_from_results(mcts_results):
        final_policy = {}
        for mcts_result, sample_chance, _idx in mcts_results:
            total_visits = mcts_result.total_visits
            options = list(mcts_result.side_one)
            if not options:
                continue
            if total_visits <= 0:
                for option in options:
                    final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + sample_chance / len(options)
                continue
            for option in options:
                final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + (
                    sample_chance * (option.visits / total_visits)
                )
        return final_policy

    def choose_from_policy(final_policy):
        ranked = sorted(final_policy.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return "no move"
        highest = ranked[0][1]
        if highest > 0:
            ranked = [item for item in ranked if item[1] >= highest * 0.75]
        weights = [max(item[1], 0.0) for item in ranked]
        if sum(weights) <= 0:
            weights = [1.0 for _ in ranked]
        return random.choices(ranked, weights=weights)[0][0]

    def select_with_tauros_kind_gate(mcts_results):
        final_policy = final_policy_from_results(mcts_results)
        if not final_policy:
            return "no move"
        baseline = choose_from_policy(final_policy)
        predicted_kind, confidence = predict_kind(_current_battle.get("battle"))
        selected = baseline
        used_gate = False
        candidates = {}
        if predicted_kind is not None and confidence >= threshold:
            highest = max(final_policy.values()) if final_policy else 0.0
            exact_choice = exact_label_to_choice(str(predicted_kind))
            if exact_choice is None:
                if predicted_kind in allowed_kinds:
                    candidates = {
                        choice: weight
                        for choice, weight in final_policy.items()
                        if choice_kind(choice) == predicted_kind and (highest <= 0 or weight >= highest * min_policy_frac)
                    }
            else:
                if choice_kind(exact_choice) in allowed_kinds:
                    candidates = {
                        choice: weight
                        for choice, weight in final_policy.items()
                        if str(choice).lower() == exact_choice and (highest <= 0 or weight >= highest * min_policy_frac)
                    }
            if candidates:
                selected = choose_from_policy(candidates)
                used_gate = selected != baseline
        if gate_log_path:
            battle = _current_battle.get("battle")
            row = {
                "turn": getattr(battle, "turn", None),
                "active": mon_name(getattr(getattr(battle, "user", None), "active", None)),
                "opponent_active": mon_name(getattr(getattr(battle, "opponent", None), "active", None)),
                "predicted_kind": predicted_kind,
                "confidence": confidence,
                "baseline": str(baseline),
                "selected": str(selected),
                "used_gate": used_gate,
                "final_policy": {str(choice): weight for choice, weight in final_policy.items()},
                "candidate_policy": {str(choice): weight for choice, weight in candidates.items()},
                "allowed_kinds": sorted(allowed_kinds),
            }
            Path(gate_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(gate_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return selected

    async def async_pick_move_with_tauros_kind_gate(battle):
        _current_battle["battle"] = deepcopy(battle)
        try:
            return await original_async_pick_move(battle)
        finally:
            _current_battle["battle"] = None

    search_main.select_move_from_mcts_results = select_with_tauros_kind_gate
    run_battle.async_pick_move = async_pick_move_with_tauros_kind_gate


_PRIOR_STATE = {"priors": None, "cpuct": 2.0}


def _mcts_with_root_priors(state_str, search_time_ms, index, threads=1):
    """Module-level (picklable/forkable) MCTS runner that applies the current
    turn's root priors. Replaces fp.search.main.get_result_from_mcts.

    Endgame override: when the alive mon count is low (≤3 each side) and
    branching is small (≤50 joint options), switch from MCTS to iterative
    deepening expectiminimax (exact alpha-beta search). This solves stall
    wars and endgame matchups that MCTS can't see within the horizon.
    """
    import poke_engine

    state = poke_engine.State.from_string(state_str)

    # endgame detection: count alive mons and options
    s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
    s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
    use_endgame_solver = False  # disabled: depth 3-4 doesn't reach terminal in randbats endgames

    if use_endgame_solver:
        try:
            id_result = poke_engine.id(state, search_time_ms)
            from poke_engine import MctsResult, MctsSideResult
            s1_moves = id_result.s1
            s2_moves = id_result.s2
            matrix = id_result.matrix
            depth = id_result.depth_searched
            n_s2 = len(s2_moves)
            import sys as _sys
            print(f"ENDGAME_SOLVER: s1={s1_alive} s2={s2_alive} depth={depth} s1_opts={len(s1_moves)} s2_opts={n_s2}", file=_sys.stderr, flush=True)
            side_one = []
            for i, mv in enumerate(s1_moves):
                row = [matrix[i * n_s2 + j] for j in range(n_s2)]
                safest = min(row) if row else 0.0
                side_one.append(MctsSideResult(move_choice=mv, total_score=safest, visits=1))
            side_two = []
            for j, mv in enumerate(s2_moves):
                col = [matrix[i * n_s2 + j] for i in range(len(s1_moves))]
                worst = max(col) if col else 0.0
                side_two.append(MctsSideResult(move_choice=mv, total_score=worst, visits=1))
            return MctsResult(side_one=side_one, side_two=side_two, total_visits=1)
        except Exception as e:
            import sys as _sys
            print(f"ENDGAME_SOLVER: failed, falling back to MCTS: {e}", file=_sys.stderr, flush=True)

    priors = _PRIOR_STATE.get("priors")
    kwargs = {}
    if priors:
        kwargs["s1_priors"] = priors
        kwargs["c_puct"] = _PRIOR_STATE.get("cpuct", 2.0)
    opp_priors = _PRIOR_STATE.get("opp_priors")
    if opp_priors:
        kwargs["s2_priors"] = opp_priors
    from config import FoulPlayConfig

    res = poke_engine.monte_carlo_tree_search(
        state, search_time_ms, threads=FoulPlayConfig.search_threads, **kwargs
    )
    return res


def patch_belief_aware_eval() -> None:
    """METAGROSS_BELIEF_EVAL=1: wire the live belief tracker into FP's eval.

    Computes threat scores from the generator-pool belief over opponent sets
    and injects them into the poke-engine state before each MCTS call.
    The Rust eval uses these for an uncertainty-aware threat penalty.
    """
    if os.environ.get("METAGROSS_BELIEF_EVAL") != "1" and os.environ.get("METAGROSS_WINCON_EVAL") != "1":
        return

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    _wincon_only = os.environ.get("METAGROSS_WINCON_EVAL") == "1" and os.environ.get("METAGROSS_BELIEF_EVAL") != "1"

    if _wincon_only:
        tracker = None
        print("WINCON_EVAL: wincon matrix only (no belief tracker)", file=_sys.stderr, flush=True)
    else:
        from belief.live_belief import BeliefTracker
        pool_path = os.environ.get(
            "METAGROSS_RANDBATS_POOL",
            str(Path(__file__).resolve().parents[2] / "data" / "randbats_pools" / "gen9randombattle_pool_50000.json"),
        )
        tracker = BeliefTracker(pool_path=pool_path)
        print(f"BELIEF_EVAL: tracker initialized from {pool_path}", file=_sys.stderr, flush=True)

    _belief_log = None
    _blog_path = os.environ.get("METAGROSS_BELIEF_LOG")
    if _blog_path:
        _belief_log = open(_blog_path, "w", buffering=1)
        _belief_log.write(f"# belief log started\n# pool={pool_path}\n")

    # Track the current battle's opponent species → species_key mapping
    # (Showdown uses display names; we need normalized keys)
    import re as _re

    def _norm(s):
        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

    # Tee protocol lines to the belief tracker
    from fp.websocket_client import PSWebsocketClient

    original_receive = PSWebsocketClient.receive_message

    _opp_player_id = ["p2"]  # mutable default; updated from |player| line

    async def receive_with_belief(self):
        message = await original_receive(self)
        try:
            if message.startswith(">battle-"):
                lines = message.split("\n")
                for line in lines[1:]:
                    if not line.startswith("|"):
                        continue
                    parts = line.split("|")

                    if len(parts) < 3:
                        continue
                    msg_type = parts[1]
                    # reset tracker on new battle
                    if msg_type == "start":
                        tracker.reset()
                        _opp_player_id[0] = "p2"  # reset to default
                    # detect opponent player ID from |player| lines
                    # |player|p1|Username|rating| or |player|p2|Username|rating|
                    # The opponent is whoever is NOT us
                    if msg_type == "player" and len(parts) >= 4:
                        player_name = parts[3]
                        our_name = config.FoulPlayConfig.username
                        if player_name != our_name:
                            _opp_player_id[0] = parts[2]  # "p1" or "p2"
                        else:
                            # WE are this player; opponent is the other
                            _opp_player_id[0] = "p2" if parts[2] == "p1" else "p1"

                    # opponent switch-in: |switch|pXa: Name|Species, L84
                    if msg_type == "switch" and len(parts) >= 4:
                        ident = parts[2]
                        if ident.startswith(_opp_player_id[0]):
                            species_raw = parts[3].split(",")[0]
                            level = 82  # default; will be corrected if we can parse
                            try:
                                level_str = parts[3].split(",")[1].strip()
                                level = int(level_str.replace("L", "").replace("l", ""))
                            except (IndexError, ValueError):
                                pass
                            tracker.on_opponent_switch_in(species_raw, level)
                    # opponent move: |move|pXa: Name|MoveName|...
                    elif msg_type == "move" and len(parts) >= 4:
                        ident = parts[2]
                        if ident.startswith(_opp_player_id[0]):
                            tracker.on_opponent_move(
                                parts[3].split(",")[0] if "," in parts[3] else parts[3],
                                parts[3],
                            )
                    # ability reveal: |-ability|pXa: Name|AbilityName
                    elif msg_type == "ability" and len(parts) >= 4:
                        ident = parts[2]
                        if ident.startswith(_opp_player_id[0]):
                            tracker.on_opponent_ability(parts[2].split(":")[-1].strip(), parts[3])
                    # item reveal: |-item|pXa: Name|ItemName
                    elif msg_type == "item" and len(parts) >= 4:
                        ident = parts[2]
                        if ident.startswith(_opp_player_id[0]):
                            tracker.on_opponent_item(parts[2].split(":")[-1].strip(), parts[3])
                    # tera: |-terastallize|pXa: Name|Type
                    elif msg_type == "terastallize" and len(parts) >= 4:
                        ident = parts[2]
                        if ident.startswith(_opp_player_id[0]):
                            tracker.on_opponent_tera(parts[2].split(":")[-1].strip(), parts[3])
        except Exception:
            pass
        return message

    PSWebsocketClient.receive_message = receive_with_belief

    # Inject belief scores into state strings in the PARENT process.
    # find_best_move runs in the parent; it calls prepare_random_battles which
    # creates the determinized state strings, then submits them to a
    # ProcessPoolExecutor. We wrap find_best_move to inject belief into the
    # battle object's states before they're serialized.
    #
    # The simplest picklable approach: store belief scores as attributes on
    # the battle object (which is deepcopied into each determinized world),
    # then have the MCTS runner read them. But poke_engine states don't carry
    # Python attributes. Instead, we modify the state STRING after
    # prepare_random_battles returns by intercepting at the
    # battle_to_poke_engine_state level.
    import fp.search.poke_engine_helpers as _peh
    import poke_engine

    _original_btpes = _peh.battle_to_poke_engine_state

    # move id -> (type, category, base_power), from FP's own move data
    from data import all_move_json as _amj
    _move_info = {}
    for _mid, _m in _amj.items():
        _move_info[_norm(_mid)] = (
            (_m.get("type") or "").lower(),
            _m.get("category", "") or "",
            _m.get("basePower", 0) or 0,
        )

    # recovery move ids (same set as Rust has_recovery_move)
    _recovery_moves = {
        "recover", "softboiled", "roost", "synthesis", "milkdrink", "slackoff",
        "moonlight", "rest", "healorder", "wish", "shoreup", "lifedew",
        "junglehealing", "purify",
    }

    # type chart (same as belief/live_belief.py)
    from belief.live_belief import TYPE_CHART, effectiveness

    def _rough_damage_pct(atk_types, atk_stats, atk_moves, def_types, def_stats, def_maxhp, def_level=100):
        """Approximate best damage % our mon can do to their mon in one turn."""
        atk_attack, atk_spattack = atk_stats
        def_defense, def_spdefense = def_stats
        best = 0.0
        for mv_id, mv_type, mv_cat, mv_bp in atk_moves:
            if mv_cat == "Status" or mv_bp <= 0:
                if mv_id in ("seismictoss", "nightshade"):
                    # fixed damage = level
                    mult = effectiveness(mv_type, def_types)
                    if mult == 0:
                        continue
                    pct = atk_stats[4] / max(def_maxhp, 1) * 100.0  # use level from atk_stats
                    best = max(best, pct)
                continue
            mult = effectiveness(mv_type, def_types)
            if mult == 0:
                continue
            is_phys = mv_cat == "Physical"
            atk = atk_attack if is_phys else atk_spattack
            defn = def_defense if is_phys else def_spdefense
            if defn <= 0:
                continue
            stab = 1.5 if mv_type in atk_types else 1.0
            dmg = 42.0 * mv_bp * stab * mult * atk / defn / 50.0
            pct = dmg / max(def_maxhp, 1) * 100.0
            best = max(best, pct)
        return best

    def _has_recovery(moves):
        return any(mv_id in _recovery_moves for mv_id, *_ in moves)

    def _compute_wincon_matrix(state):
        """6x6 matrix: positive = our mon i beats their mon j in damage race, negative = loses."""
        flat = [0.0] * 36
        for i in range(6):
            our = state.side_one.pokemon[i]
            if our.hp <= 0:
                continue
            our_types = tuple(t.lower() for t in our.types if t and t.lower() != "typeless")
            our_stats = (our.attack, our.special_attack, our.defense, our.special_defense, our.level)
            our_moves = []
            for mv in our.moves:
                mn = _norm(mv.id)
                info = _move_info.get(mn)
                if info:
                    our_moves.append((mn, info[0], info[1], info[2]))
            our_heal = 33.0 if _has_recovery(our_moves) else 0.0

            for j in range(6):
                their = state.side_two.pokemon[j]
                if their.hp <= 0:
                    continue
                their_types = tuple(t.lower() for t in their.types if t and t.lower() != "typeless")
                their_stats = (their.attack, their.special_attack, their.defense, their.special_defense, their.level)
                their_moves = []
                for mv in their.moves:
                    mn = _norm(mv.id)
                    info = _move_info.get(mn)
                    if info:
                        their_moves.append((mn, info[0], info[1], info[2]))
                their_heal = 33.0 if _has_recovery(their_moves) else 0.0

                our_dmg = _rough_damage_pct(our_types, our_stats, our_moves, their_types, (their_stats[2], their_stats[3]), their.maxhp)
                their_dmg = _rough_damage_pct(their_types, their_stats, their_moves, our_types, (our_stats[2], our_stats[3]), our.maxhp)
                our_net = our_dmg - their_heal
                their_net = their_dmg - our_heal
                # +1 if we win the race, -1 if we lose, 0 if neutral
                if our_net > 0 and their_net <= 0:
                    flat[i * 6 + j] = 1.0
                elif our_net <= 0 and their_net > 0:
                    flat[i * 6 + j] = -1.0
        return flat

    _dbg = {"injections": 0, "errors": 0, "calls": 0}

    def _btpes_with_belief(battle, *args, **kwargs):
        state = _original_btpes(battle, *args, **kwargs)
        try:
            _dbg["calls"] += 1
            if _dbg["calls"] == 1:
                opp_ids = [state.side_two.pokemon[j].id for j in range(6)]
                s1_ids = [state.side_one.pokemon[i].id for i in range(6)]
                tracker_keys = list(tracker._opponent_mons.keys())
                print(
                    f"BELIEF_EVAL: first btpes call. tracker_mons={tracker_keys} "
                    f"opp_ids={opp_ids} our_ids={s1_ids}",
                    file=_sys.stderr, flush=True,
                )
            # orientation guard: side_one must be us
            if kwargs.get("swap") or (args and args[0]):
                return state
            # win-condition matrix (always computed, doesn't need tracker)
            wincon = _compute_wincon_matrix(state)
            if any(v != 0.0 for v in wincon):
                poke_engine.set_wincon_matrix(state, wincon)
            # belief threat matrix (only if tracker is active)
            if tracker is None:
                return state
            # our mons' current types, straight from the engine state so the
            # matrix indices are guaranteed to match engine indices
            s1_types = []
            for i in range(6):
                p = state.side_one.pokemon[i]
                s1_types.append(tuple(
                    t.lower() for t in p.types
                    if t and t.lower() != "typeless"
                ))
            flat = [0.0] * 36
            any_threat = False
            for j in range(6):
                opp_key = _norm(state.side_two.pokemon[j].id)
                belief_obj = tracker._opponent_mons.get(opp_key)
                if belief_obj is None:
                    continue  # unrevealed/sampled mon: no belief, no term
                for i in range(6):
                    if not s1_types[i]:
                        continue
                    tp = belief_obj.unrevealed_threat_prob(s1_types[i], _move_info)
                    if tp > 0.0:
                        flat[i * 6 + j] = tp
                        any_threat = True
            if any_threat:
                poke_engine.set_threat_matrix(state, flat)
                _dbg["injections"] += 1
                if _dbg["injections"] == 1:
                    print(
                        f"BELIEF_EVAL: first threat matrix injected: {flat}",
                        file=_sys.stderr, flush=True,
                    )
            # win-condition matrix: separate channel from threat matrix
            wincon = _compute_wincon_matrix(state)
            if any(v != 0.0 for v in wincon):
                poke_engine.set_wincon_matrix(state, wincon)
        except Exception as e:
            _dbg["errors"] += 1
            if _dbg["errors"] == 1:
                import traceback
                print(
                    f"BELIEF_EVAL: threat matrix injection FAILED: {e}\n"
                    + traceback.format_exc(),
                    file=_sys.stderr, flush=True,
                )
        return state

    _peh.battle_to_poke_engine_state = _btpes_with_belief
    import fp.search.random_battles as _rb
    import fp.search.standard_battles as _sb
    import fp.search.main as _sm
    for mod in (_rb, _sb, _sm):
        if hasattr(mod, "battle_to_poke_engine_state"):
            mod.battle_to_poke_engine_state = _btpes_with_belief
    print("BELIEF_EVAL: patches active", file=_sys.stderr, flush=True)


def patch_root_priors() -> None:
    """METAGROSS_PRIOR_SERVER=<url>: fetch per-turn root priors from the prior
    server and pass them into the (patched) poke-engine MCTS. Also sends every
    incoming protocol message to the server so it can track battle state.

    Uses synchronous HTTP POST inside the async receive (no background thread)
    to avoid fork-deadlock when FP spawns multiprocessing workers."""
    server_url = os.environ.get("METAGROSS_PRIOR_SERVER")
    if not server_url:
        return
    import logging as _logging
    logger = _logging.getLogger("fp.root_priors")
    import urllib.request as _url

    _PRIOR_STATE["cpuct"] = float(os.environ.get("METAGROSS_CPUCT", "2.0"))

    def _post_sync(path: str, payload: dict, timeout: float = 5.0):
        """Synchronous POST — called from async context via run_in_executor."""
        body = json.dumps(payload).encode()
        req = _url.Request(f"{server_url}{path}", data=body,
                           headers={"Content-Type": "application/json"})
        return _url.urlopen(req, timeout=timeout)

    # 1) send incoming protocol messages to prior server (no background thread)
    from fp.websocket_client import PSWebsocketClient

    original_receive = PSWebsocketClient.receive_message

    async def receive_with_tee(self):
        message = await original_receive(self)
        try:
            if message.startswith(">battle-"):
                lines = message.split("\n")
                tag = lines[0].lstrip(">").strip()
                # synchronous POST in a thread executor — no daemon thread
                # that would deadlock on fork
                import asyncio as _a
                loop = _a.get_event_loop()
                await loop.run_in_executor(
                    None, _post_sync, "/lines",
                    {"tag": tag, "lines": lines[1:]}
                )
        except Exception:
            pass
        return message

    PSWebsocketClient.receive_message = receive_with_tee

    # 2) swap the MCTS runner for the priors-aware one
    import fp.search.main as search_main

    search_main.get_result_from_mcts = _mcts_with_root_priors

    # 3) fetch priors before each search
    import fp.run_battle as run_battle_module

    original_find_best_move = search_main.find_best_move

    def find_best_move_with_priors(battle):
        _PRIOR_STATE["priors"] = None
        _PRIOR_STATE["opp_priors"] = None
        try:
            tag = getattr(battle, "battle_tag", None)
            if tag:
                # messages are sent synchronously in receive_with_tee now,
                # so the server has already seen this turn's data
                full_tag = tag if tag.startswith("battle-") else f"battle-{tag}"
                with _url.urlopen(
                    f"{server_url}/priors?tag={full_tag}", timeout=30
                ) as resp:
                    data = json.loads(resp.read())
                opp_only = os.environ.get("METAGROSS_OPP_PRIORS_ONLY") == "1"
                if opp_only:
                    priors = {}
                else:
                    priors = data.get("priors") or {}
                opp_priors = data.get("opp_priors") or {}
                if opp_only and (data.get("priors") or {}):
                    logger.info("opp-only mode: discarding %d s1 priors", len(data.get("priors") or {}))
                if priors:
                    _PRIOR_STATE["priors"] = [(k, float(v)) for k, v in priors.items()]
                    logger.info("root priors ({}): {}".format(
                        len(priors),
                        {k: round(v, 3) for k, v in sorted(priors.items(), key=lambda kv: -kv[1])[:4]},
                    ))
                if opp_priors:
                    _PRIOR_STATE["opp_priors"] = [(k, float(v)) for k, v in opp_priors.items()]
                    logger.info("opp priors ({}): {}".format(
                        len(opp_priors),
                        {k: round(v, 3) for k, v in sorted(opp_priors.items(), key=lambda kv: -kv[1])[:4]},
                    ))
        except Exception as exc:
            # Generation can require policy-guided games. In that mode a
            # failed fetch must discard the game rather than silently record
            # a fallback-FP trajectory as expert data.
            if os.environ.get("METAGROSS_REQUIRE_PRIORS") == "1":
                raise RuntimeError(f"required prior fetch failed: {exc!r}") from exc
            logger.warning(f"prior fetch failed, searching without priors: {exc!r}")
        return original_find_best_move(battle)

    search_main.find_best_move = find_best_move_with_priors
    run_battle_module.find_best_move = find_best_move_with_priors
    logger.info(f"root-priors patch active (server={server_url}, c_puct={_PRIOR_STATE['cpuct']})")


def patch_state_dump() -> None:
    """METAGROSS_STATE_DUMP=<path>: append every root poke-engine state string
    (one JSON line per decision) — used for hybrid prior development/testing."""
    dump_path = os.environ.get("METAGROSS_STATE_DUMP")
    if not dump_path:
        return
    from fp.search import poke_engine_helpers

    original = poke_engine_helpers.battle_to_poke_engine_state

    def battle_to_state_with_dump(battle, *args, **kwargs):
        state = original(battle, *args, **kwargs)
        try:
            with open(dump_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"state": state.to_string()}) + "\n")
        except Exception:
            pass
        return state

    poke_engine_helpers.battle_to_poke_engine_state = battle_to_state_with_dump
    # standard_battles/random_battles import by value
    import fp.search.standard_battles as _sb
    import fp.search.random_battles as _rb
    import fp.search.main as _sm
    for mod in (_sb, _rb, _sm):
        if hasattr(mod, "battle_to_poke_engine_state"):
            mod.battle_to_poke_engine_state = battle_to_state_with_dump


def patch_foul_play_value_shield() -> None:
    if os.environ.get("METAGROSS_FP_VALUE_SHIELD") != "1":
        return

    import fp.search.main as search_main

    margin = float(os.environ.get("METAGROSS_FP_VALUE_SHIELD_MARGIN", "0.15"))
    min_support = float(os.environ.get("METAGROSS_FP_VALUE_SHIELD_MIN_SUPPORT", "0.10"))
    close_policy_frac = float(os.environ.get("METAGROSS_FP_VALUE_SHIELD_CLOSE_POLICY_FRAC", "0.75"))
    log_path = os.environ.get("METAGROSS_FP_VALUE_SHIELD_LOG")

    def final_policy_from_results(mcts_results):
        final_policy = {}
        for mcts_result, sample_chance, _idx in mcts_results:
            total_visits = mcts_result.total_visits
            options = list(mcts_result.side_one)
            if not options:
                continue
            if total_visits <= 0:
                weight = sample_chance / len(options)
                for option in options:
                    final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + weight
                continue
            for option in options:
                final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + (
                    sample_chance * (option.visits / total_visits)
                )
        return final_policy

    def value_stats_from_results(mcts_results):
        numerators = {}
        denominators = {}
        for mcts_result, sample_chance, _idx in mcts_results:
            for option in mcts_result.side_one:
                if option.visits <= 0:
                    continue
                move = option.move_choice
                weight = sample_chance * option.visits
                numerators[move] = numerators.get(move, 0.0) + weight * (option.total_score / option.visits)
                denominators[move] = denominators.get(move, 0.0) + weight
        return {
            move: {
                "avg_score": numerators[move] / denominators[move],
                "value_weight": denominators[move],
            }
            for move in numerators
            if denominators.get(move, 0.0) > 0
        }

    def choose_from_policy(policy_items):
        if not policy_items:
            return "no move"
        weights = [max(weight, 0.0) for _move, weight in policy_items]
        if sum(weights) <= 0:
            weights = [1.0 for _move, _weight in policy_items]
        return random.choices(policy_items, weights=weights)[0][0]

    def select_with_value_shield(mcts_results):
        final_policy = final_policy_from_results(mcts_results)
        if not final_policy:
            return "no move"

        ranked_policy = sorted(final_policy.items(), key=lambda item: item[1], reverse=True)
        highest_support = ranked_policy[0][1]
        considered = [
            item for item in ranked_policy
            if highest_support <= 0 or item[1] >= highest_support * close_policy_frac
        ]
        baseline = choose_from_policy(considered)

        value_stats = value_stats_from_results(mcts_results)
        baseline_stats = value_stats.get(baseline)
        selected = baseline
        used_shield = False
        best_value_move = None
        best_value_delta = 0.0

        if baseline_stats is not None:
            candidates = []
            for move, support in final_policy.items():
                stats = value_stats.get(move)
                if stats is None or support < min_support:
                    continue
                candidates.append((move, support, stats["avg_score"], stats["value_weight"]))
            if candidates:
                best_value_move, _support, best_value, _weight = max(candidates, key=lambda item: item[2])
                best_value_delta = best_value - baseline_stats["avg_score"]
                if best_value_move != baseline and best_value_delta >= margin:
                    selected = best_value_move
                    used_shield = True

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            row = {
                "baseline": str(baseline),
                "selected": str(selected),
                "used_shield": used_shield,
                "best_value_move": str(best_value_move) if best_value_move is not None else None,
                "best_value_delta": best_value_delta,
                "final_policy": {str(move): support for move, support in final_policy.items()},
                "value_stats": {
                    str(move): stats for move, stats in value_stats.items()
                },
                "margin": margin,
                "min_support": min_support,
                "close_policy_frac": close_policy_frac,
            }
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return selected

    search_main.select_move_from_mcts_results = select_with_value_shield


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

    root_dir = Path(__file__).resolve().parents[2]
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
    conditional_max_ms = int(os.environ.get("METAGROSS_RANDBATS_CONDITIONAL_MAX_MS", "250"))
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
                "--max-ms",
                str(conditional_max_ms),
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
                            "maxMillis": conditional_max_ms,
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
    import threading
    _policy_store: dict = {}
    _store_lock = threading.Lock()

    def write_record(row: dict) -> None:
        """Append immediately: a disconnected game must not erase search data."""
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def safe_select_move_from_mcts_results(mcts_results):
        final_policy = {}
        for mcts_result, sample_chance, _idx in mcts_results:
            total_visits = mcts_result.total_visits
            options = list(mcts_result.side_one)
            if not options:
                continue
            if total_visits <= 0:
                weight = sample_chance / len(options)
                for option in options:
                    final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + weight
                continue
            for option in options:
                final_policy[option.move_choice] = final_policy.get(option.move_choice, 0.0) + (
                    sample_chance * (option.visits / total_visits)
                )

        if not final_policy:
            return "no move"
        ranked_policy = sorted(final_policy.items(), key=lambda item: item[1], reverse=True)
        highest = ranked_policy[0][1]
        if highest > 0:
            ranked_policy = [item for item in ranked_policy if item[1] >= highest * 0.75]
        weights = [max(item[1], 0.0) for item in ranked_policy]
        if sum(weights) <= 0:
            weights = [1.0 for _ in ranked_policy]
        return random.choices(ranked_policy, weights=weights)[0][0]

    def select_and_capture(mcts_results):
        # Capture MCTS visit distributions. select_move_from_mcts_results runs in the
        # MAIN process after futures are collected, so no pickling required.
        # Each element: (MctsResult, sample_chance, index)
        agg: dict[str, float] = {}
        total = 0
        try:
            for mcts_result, chance, _idx in mcts_results:
                tv = mcts_result.total_visits
                total += tv
                for opt in mcts_result.side_one:
                    move = str(opt.move_choice)
                    # Match selection's uniform fallback for a zero-visit
                    # search so logged targets are always distributions.
                    frac = opt.visits / tv if tv > 0 else 1.0 / len(mcts_result.side_one)
                    agg[move] = agg.get(move, 0.0) + chance * frac
        except Exception:
            pass
        selected = safe_select_move_from_mcts_results(mcts_results)
        with _store_lock:
            if agg:
                _policy_store['__last__'] = {
                    'visits': agg,
                    'total': total,
                    'selected_action': str(selected),
                }
        return selected

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
            # Attach and persist the MCTS policy immediately. Game-result rows
            # are optional metadata; they must not gate policy-target writes.
            if pending_rows and len(pending_rows) > start_index:
                row = pending_rows[-1]
                if "features" in row:
                    with _store_lock:
                        captured = _policy_store.pop('__last__', None)
                    if captured:
                        row["mcts_visits"] = captured['visits']
                        row["mcts_total"] = captured['total']
                        row["selected_action"] = captured['selected_action']
                    else:
                        row["mcts_capture_missing"] = True
                    row["record_type"] = "decision"
                    write_record(row)
                # This record is durable now; do not duplicate at battle end.
                del pending_rows[start_index:]
            return result
        except Exception:
            del pending_rows[start_index:]
            raise

    async def pokemon_battle_with_labels(ps_websocket_client, pokemon_battle_type, team_dict):
        winner = await original_pokemon_battle(ps_websocket_client, pokemon_battle_type, team_dict)
        label = 1 if winner == config.FoulPlayConfig.username else 0
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "battle_result",
                "battle_tag": getattr(ps_websocket_client, "battle_tag", None),
                "winner": winner,
                "label": label,
                "username": config.FoulPlayConfig.username,
            }, separators=(",", ":")) + "\n")
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


def patch_replay_capture() -> None:
    """METAGROSS_REPLAY_DIR=<path>: capture full Showdown protocol logs per game
    and save as replay JSONs (compatible with the metamon replay parser).

    Saves maximum information: protocol log, inputlog (if available), player names,
    format, winner, and all raw websocket messages.
    """
    replay_dir = os.environ.get("METAGROSS_REPLAY_DIR")
    if not replay_dir:
        return

    import sys as _sys
    replay_path = Path(replay_dir).resolve()
    replay_path.mkdir(parents=True, exist_ok=True)

    from fp.websocket_client import PSWebsocketClient

    # Store per-battle message logs: tag -> list of protocol lines
    _battle_logs: dict[str, list[str]] = {}
    _battle_players: dict[str, list[str]] = {}
    _battle_inputlog: dict[str, list[str]] = {}
    _our_name = None
    try:
        import config as _cfg
        _our_name = _cfg.FoulPlayConfig.username
    except Exception:
        pass

    original_receive = PSWebsocketClient.receive_message

    async def receive_with_replay(self):
        nonlocal _our_name
        message = await original_receive(self)
        try:
            if message.startswith(">battle-"):
                lines = message.split("\n")
                tag = lines[0].lstrip(">").strip()
                if tag not in _battle_logs:
                    _battle_logs[tag] = []
                    _battle_players[tag] = []
                    _battle_inputlog[tag] = []

                for line in lines[1:]:
                    if not line.startswith("|"):
                        continue
                    parts = line.split("|")
                    if len(parts) < 2:
                        continue
                    msg_type = parts[1]

                    # Capture player names
                    if msg_type == "player" and len(parts) >= 4:
                        pname = parts[3]
                        if pname and pname not in _battle_players[tag]:
                            _battle_players[tag].append(pname)
                        if _our_name is None:
                            import config as _cfg
                            _our_name = _cfg.FoulPlayConfig.username

                    # Capture inputlog lines (>|p1 move ...| etc)
                    if line.startswith(">p1 ") or line.startswith(">p2 ") or line.startswith(">start ") or line.startswith(">player "):
                        _battle_inputlog[tag].append(line)

                    # Capture all protocol lines
                    _battle_logs[tag].append(line)

                    # On game end (win/tie), save the replay
                    if msg_type in ("win", "tie") and len(parts) >= 3:
                        # Build clean log without embedded-JSON protocol lines.
                        # Let json.dump handle all escaping — don't pre-escape.
                        clean_lines = [
                            line for line in _battle_logs.get(tag, [])
                            if not line.startswith("|request|")
                            and not line.startswith("|html|")
                            and not line.startswith("|uhtml|")
                            and not line.startswith("|raw|")
                            and not line.startswith("|c|")
                            and not line.startswith("|chatmsg|")
                        ]
                        log_text = "\n".join(clean_lines)
                        players = _battle_players.get(tag, ["p1", "p2"])

                        # Extract winner
                        winner = None
                        if msg_type == "win":
                            winner = parts[2].strip()

                        replay_json = {
                            "id": tag,
                            "formatid": "gen9randombattle",
                            "format": "[Gen 9] Random Battle",
                            "players": players,
                            "log": log_text,
                            "uploadtime": int(__import__("time").time()),
                            "views": 0,
                            "rating": 0,
                            "private": 0,
                            "password": None,
                            "_winner": winner,
                            "_our_name": _our_name,
                        }

                        out_file = replay_path / f"{tag}_{_our_name or 'agent'}.json"
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(replay_json, f, ensure_ascii=True)

                        print(f"REPLAY_SAVED: {out_file.name} winner={winner}", file=_sys.stderr, flush=True)

                        # Clean up
                        _battle_logs.pop(tag, None)
                        _battle_players.pop(tag, None)
                        _battle_inputlog.pop(tag, None)
        except Exception as e:
            import sys as _sys2
            print(f"REPLAY_CAPTURE_ERROR: {e}", file=_sys2.stderr, flush=True)
        return message

    PSWebsocketClient.receive_message = receive_with_replay
    print(f"REPLAY_CAPTURE: saving to {replay_path}", file=sys.stderr, flush=True)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    foul_play_dir = Path(os.environ.get("FOUL_PLAY_DIR", root_dir / "external" / "foul-play"))

    env_password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
    if env_password and "--ps-password" not in sys.argv:
        sys.argv.extend(["--ps-password", env_password])

    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    os.chdir(foul_play_dir)
    sys.path.insert(0, str(foul_play_dir))

    if os.environ.get("METAGROSS_PRIOR_SERVER"):
        print(f"DEBUG PRIOR_SERVER={os.environ['METAGROSS_PRIOR_SERVER']}", file=sys.stderr, flush=True)
        print(f"DEBUG OPP_PRIORS_ONLY={os.environ.get('METAGROSS_OPP_PRIORS_ONLY', 'not set')}", file=sys.stderr, flush=True)

    patch_foul_play_protocol_bugs()
    patch_tauros_action_kind_gate()
    patch_foul_play_value_shield()
    patch_state_dump()
    patch_belief_aware_eval()
    patch_root_priors()
    patch_randbats_generator_belief()
    patch_decision_logging()
    patch_replay_capture()

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
