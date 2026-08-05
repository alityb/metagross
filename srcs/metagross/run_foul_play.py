#!/usr/bin/env python3
"""Run Foul Play with a verified production root-prior integration."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse


_PRIOR_STATE = {
    "priors": None,
    "opp_priors": None,
    "cpuct": 2.0,
    "context": None,
    "remote_search": None,
}
_REMOTE_FUNCTIONS: dict[int, object] = {}
REMOTE_MCTS_SCHEMA = 1
REMOTE_ENGINE_CONTRACT = "poke-engine-0.0.47-priors-v2"


def _append_jsonl(environment_variable: str, row: dict) -> None:
    path = os.environ.get(environment_variable)
    if not path:
        return
    payload = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


def _mcts_result_payload(result) -> dict:
    def side_payload(options):
        return [
            {
                "move_choice": option.move_choice,
                "total_score": float(option.total_score),
                "visits": int(option.visits),
            }
            for option in options
        ]

    return {
        "side_one": side_payload(result.side_one),
        "side_two": side_payload(result.side_two),
        "total_visits": int(result.total_visits),
    }


def _mcts_result_from_payload(payload: object, engine_module=None):
    if not isinstance(payload, dict):
        raise RuntimeError("remote MCTS result must be an object")
    if engine_module is None:
        import poke_engine as engine_module

    def side(value: object, label: str):
        if not isinstance(value, list) or (label == "side_one" and not value):
            raise RuntimeError(f"remote MCTS {label} is invalid")
        options = []
        for row in value:
            if not isinstance(row, dict):
                raise RuntimeError(f"remote MCTS {label} entry is invalid")
            move = row.get("move_choice")
            score = row.get("total_score")
            visits = row.get("visits")
            if not isinstance(move, str) or not move:
                raise RuntimeError(f"remote MCTS {label} move is invalid")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise RuntimeError(f"remote MCTS {label} score is invalid")
            if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
                raise RuntimeError(f"remote MCTS {label} visits are invalid")
            options.append(
                engine_module.MctsSideResult(
                    move_choice=move,
                    total_score=float(score),
                    visits=visits,
                )
            )
        return options

    total_visits = payload.get("total_visits")
    if isinstance(total_visits, bool) or not isinstance(total_visits, int) or total_visits < 0:
        raise RuntimeError("remote MCTS total visits are invalid")
    return engine_module.MctsResult(
        side_one=side(payload.get("side_one"), "side_one"),
        side_two=side(payload.get("side_two"), "side_two"),
        total_visits=total_visits,
    )


def _remote_mcts_function():
    pid = os.getpid()
    if pid not in _REMOTE_FUNCTIONS:
        import modal

        app_name = os.environ["METAGROSS_REMOTE_MCTS_APP"]
        function_name = os.environ["METAGROSS_REMOTE_MCTS_FUNCTION"]
        _REMOTE_FUNCTIONS.clear()
        _REMOTE_FUNCTIONS[pid] = modal.Function.from_name(app_name, function_name)
    return _REMOTE_FUNCTIONS[pid]


def _validate_remote_response(response: object, request_id: str, index: int) -> dict:
    if not isinstance(response, dict) or response.get("schema") != REMOTE_MCTS_SCHEMA:
        raise RuntimeError("remote MCTS returned an invalid schema")
    if response.get("request_id") != request_id or response.get("index") != index:
        raise RuntimeError("remote MCTS response correlation mismatch")
    engine = response.get("engine")
    expected_sha = os.environ.get("METAGROSS_REMOTE_ENGINE_SHA256")
    if not isinstance(engine, dict) or engine.get("contract") != REMOTE_ENGINE_CONTRACT:
        raise RuntimeError("remote MCTS engine contract mismatch")
    if not expected_sha or engine.get("native_sha256") != expected_sha:
        raise RuntimeError("remote MCTS engine SHA-256 mismatch")
    if response.get("ok") is not True:
        error = response.get("error") or {}
        raise RuntimeError(f"remote MCTS failed: {error.get('kind', 'unknown error')}")
    return response


def _remote_mcts_batch(state_strings: list[str], search_time_ms: int, threads: int):
    requests = []
    for index, state_string in enumerate(state_strings):
        requests.append(
            {
                "schema": REMOTE_MCTS_SCHEMA,
                "request_id": uuid.uuid4().hex,
                "index": index,
                "state": state_string,
                "duration_ms": int(search_time_ms),
                "threads": int(threads),
                "s1_priors": [list(row) for row in (_PRIOR_STATE["priors"] or [])]
                or None,
                "s2_priors": [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])]
                or None,
                "c_puct": float(_PRIOR_STATE["cpuct"]),
            }
        )
    started = time.monotonic()
    responses = _remote_mcts_function().remote(requests)
    rpc_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote MCTS returned the wrong batch size")
    import poke_engine

    results = []
    timings = []
    for request, response in zip(requests, responses, strict=True):
        validated = _validate_remote_response(
            response, request["request_id"], request["index"]
        )
        results.append(_mcts_result_from_payload(validated.get("result"), poke_engine))
        timings.append(validated.get("timing"))
    _PRIOR_STATE["remote_search"] = {
        "rpc_ms": rpc_ms,
        "worlds": len(requests),
        "engine": responses[0].get("engine"),
        "timings": timings,
    }
    return results


def _remote_find_best_move(battle, search_main):
    battle = search_main.deepcopy(battle)
    if battle.team_preview:
        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

    if battle.battle_type == search_main.BattleType.RANDOM_BATTLE:
        num_battles, search_time_ms = search_main.search_time_num_battles_randombattles(
            battle
        )
        battles = search_main.prepare_random_battles(battle, num_battles)
    elif battle.battle_type == search_main.BattleType.BATTLE_FACTORY:
        num_battles, search_time_ms = search_main.search_time_num_battles_standard_battle(
            battle
        )
        battles = search_main.prepare_random_battles(battle, num_battles)
    elif battle.battle_type == search_main.BattleType.STANDARD_BATTLE:
        num_battles, search_time_ms = search_main.search_time_num_battles_standard_battle(
            battle
        )
        battles = search_main.prepare_battles(battle, num_battles)
    else:
        raise ValueError("Unsupported battle type")

    search_main.logger.info("Searching for a move using remote MCTS...")
    search_main.logger.info(
        "Sampling %s battles at %sms each", num_battles, search_time_ms
    )
    states = [
        search_main.battle_to_poke_engine_state(sampled).to_string()
        for sampled, _chance in battles
    ]
    from config import FoulPlayConfig

    results = _remote_mcts_batch(states, search_time_ms, FoulPlayConfig.search_threads)
    weighted = [
        (result, chance, index)
        for index, (result, (_sampled, chance)) in enumerate(zip(results, battles, strict=True))
    ]
    return search_main.select_move_from_mcts_results(weighted)


def validate_poke_engine_provenance(provenance: dict, expected_source: Path) -> None:
    source = Path(provenance["source_path"]).resolve()
    expected_source = expected_source.resolve()
    if not source.is_relative_to(expected_source):
        raise RuntimeError(
            f"poke-engine source mismatch: expected {expected_source}, found {source}"
        )
    if provenance["editable"]:
        raise RuntimeError("production poke-engine must not be an editable install")
    parameters = provenance["mcts_parameters"]
    required = {"state", "duration_ms", "threads", "s1_priors", "s2_priors", "c_puct"}
    missing = required - set(parameters)
    if missing:
        raise RuntimeError(f"production poke-engine missing MCTS parameters: {sorted(missing)}")
    if "seed" in parameters:
        raise RuntimeError("experimental poke-engine MCTS signature detected")


def inspect_poke_engine() -> dict:
    root = Path(__file__).resolve().parents[2]
    expected_source = root / "srcs" / "vendor" / "poke-engine"
    import poke_engine

    distribution = importlib.metadata.distribution("poke_engine")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    source_url = direct_url.get("url")
    if not source_url or urlparse(source_url).scheme != "file":
        raise RuntimeError("poke-engine install has no local source provenance")
    source_path = Path(unquote(urlparse(source_url).path)).resolve()
    native_module = importlib.import_module("poke_engine.poke_engine")
    native_path = Path(native_module.__file__).resolve()
    provenance = {
        "distribution_version": distribution.version,
        "editable": bool((direct_url.get("dir_info") or {}).get("editable")),
        "module_path": str(Path(poke_engine.__file__).resolve()),
        "native_path": str(native_path),
        "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "source_path": str(source_path),
        "mcts_parameters": list(
            inspect.signature(poke_engine.monte_carlo_tree_search).parameters
        ),
    }
    validate_poke_engine_provenance(provenance, expected_source)
    return provenance


def patch_foul_play_protocol() -> None:
    """Apply the protocol safeguards used by the accepted deployment."""
    import fp.run_battle as run_battle
    import fp.websocket_client as websocket_client
    import websockets

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default

    original_connect = websockets.connect

    def connect_with_safe_ping(address, *args, **kwargs):
        if os.environ.get("METAGROSS_WEBSOCKET_KEEPALIVE") == "1":
            # Keep proxy tunnels active, but never let a delayed pong terminate
            # a battle while CPU-bound search is blocking the event loop.
            kwargs.setdefault("ping_interval", 20)
            kwargs.setdefault("ping_timeout", None)
        else:
            kwargs.setdefault("ping_interval", None)
        return original_connect(address, *args, **kwargs)

    websocket_client.websockets.connect = connect_with_safe_ping

    async def login_without_idle_delay(self):
        """Authenticate without the upstream three-second proxy-idle window."""
        websocket_client.logger.info("Logging in...")
        client_id, challstr = await self.get_id_and_challstr()
        guest_login = self.password is None
        if guest_login:
            response = websocket_client.requests.post(
                self.login_uri,
                data={
                    "act": "getassertion",
                    "userid": self.username,
                    "challstr": "|".join([client_id, challstr]),
                },
            )
        else:
            response = websocket_client.requests.post(
                self.login_uri,
                data={
                    "name": self.username,
                    "pass": self.password,
                    "challstr": "|".join([client_id, challstr]),
                },
            )
        if response.status_code != 200:
            raise websocket_client.LoginError("Could not get assertion")
        if guest_login:
            assertion = response.text
            user_id = self.username
        else:
            response_json = websocket_client.json.loads(response.text[1:])
            if "actionsuccess" not in response_json:
                raise websocket_client.LoginError(f"Could not log-in: {response_json}")
            assertion = response_json.get("assertion")
            user_id = response_json["curuser"]["userid"]
        await self.send_message("", [f"/trn {self.username},0,{assertion}"])
        return user_id

    websocket_client.PSWebsocketClient.login = login_without_idle_delay

    original_receive = websocket_client.PSWebsocketClient.receive_message

    async def receive_with_rating_log(self):
        message = await original_receive(self)
        if message.startswith(">battle-"):
            _append_jsonl(
                "METAGROSS_PROTOCOL_DUMP",
                {
                    "schema": 1,
                    "time_ns": time.time_ns(),
                    "direction": "received",
                    "message": message,
                },
            )
        for line in message.splitlines():
            if line.startswith("|raw|") and (
                "<strong>" in line or "rating:" in line.lower()
            ):
                print(f"RATING_LINE {line}", flush=True)
        return message

    websocket_client.PSWebsocketClient.receive_message = receive_with_rating_log

    original_send = websocket_client.PSWebsocketClient.send_message

    async def send_with_battle_log(self, room, message_list):
        if room.startswith("battle-"):
            _append_jsonl(
                "METAGROSS_PROTOCOL_DUMP",
                {
                    "schema": 1,
                    "time_ns": time.time_ns(),
                    "direction": "sent",
                    "room": room,
                    "messages": list(message_list),
                },
            )
        return await original_send(self, room, message_list)

    websocket_client.PSWebsocketClient.send_message = send_with_battle_log


def _mcts_with_root_priors(state_str, search_time_ms, index, threads=1):
    """Run the patched engine with player and opponent root priors."""
    import poke_engine
    from config import FoulPlayConfig

    if os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") == "1":
        request_id = uuid.uuid4().hex
        request = {
            "schema": REMOTE_MCTS_SCHEMA,
            "request_id": request_id,
            "index": int(index),
            "state": state_str,
            "duration_ms": int(search_time_ms),
            "threads": int(FoulPlayConfig.search_threads),
            "s1_priors": [list(row) for row in (_PRIOR_STATE["priors"] or [])] or None,
            "s2_priors": [list(row) for row in (_PRIOR_STATE["opp_priors"] or [])] or None,
            "c_puct": float(_PRIOR_STATE["cpuct"]),
        }
        response = _remote_mcts_function().remote(request)
        validated = _validate_remote_response(response, request_id, int(index))
        return _mcts_result_from_payload(validated.get("result"), poke_engine)

    state = poke_engine.State.from_string(state_str)
    kwargs = {}
    if _PRIOR_STATE["priors"]:
        kwargs["s1_priors"] = _PRIOR_STATE["priors"]
        kwargs["c_puct"] = _PRIOR_STATE["cpuct"]
    if _PRIOR_STATE["opp_priors"]:
        kwargs["s2_priors"] = _PRIOR_STATE["opp_priors"]
    return poke_engine.monte_carlo_tree_search(
        state,
        search_time_ms,
        threads=FoulPlayConfig.search_threads,
        **kwargs,
    )


def patch_root_priors() -> None:
    """Connect Foul Play's search roots to the local r1 policy server."""
    server_url = os.environ.get("METAGROSS_PRIOR_SERVER")
    if not server_url:
        raise RuntimeError("METAGROSS_PRIOR_SERVER is required")

    import logging
    import urllib.request
    from urllib.parse import quote

    import fp.run_battle as run_battle
    import fp.search.main as search_main
    from fp.websocket_client import PSWebsocketClient

    logger = logging.getLogger("fp.root_priors")
    namespace = os.environ.get("METAGROSS_PRIOR_NAMESPACE", "")
    _PRIOR_STATE["cpuct"] = float(os.environ.get("METAGROSS_CPUCT", "2.0"))

    def post(path: str, payload: dict, timeout: float = 5.0):
        request = urllib.request.Request(
            f"{server_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=timeout)

    original_receive = PSWebsocketClient.receive_message

    async def receive_with_tee(self):
        message = await original_receive(self)
        if message.startswith(">battle-"):
            lines = message.split("\n")
            tag = lines[0].lstrip(">").strip()
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    post,
                    "/lines",
                    {"tag": tag, "namespace": namespace, "lines": lines[1:]},
                )
            except Exception as exc:
                if os.environ.get("METAGROSS_REQUIRE_PRIORS") == "1":
                    raise RuntimeError(f"required protocol tee failed: {exc!r}") from exc
                logger.warning("prior protocol tee failed: %r", exc)
        return message

    PSWebsocketClient.receive_message = receive_with_tee
    search_main.get_result_from_mcts = _mcts_with_root_priors
    original_find_best_move = search_main.find_best_move
    original_select_move = search_main.select_move_from_mcts_results

    def select_move_with_dump(mcts_results):
        choice = original_select_move(mcts_results)
        _append_jsonl(
            "METAGROSS_SEARCH_DUMP",
            {
                "schema": 1,
                "time_ns": time.time_ns(),
                "context": _PRIOR_STATE["context"],
                "choice": choice,
                "player_priors": _PRIOR_STATE["priors"],
                "opponent_priors": _PRIOR_STATE["opp_priors"],
                "remote_search": _PRIOR_STATE["remote_search"],
                "samples": [
                    {
                        "sample_chance": float(sample_chance),
                        "index": int(index),
                        "result": _mcts_result_payload(result),
                    }
                    for result, sample_chance, index in mcts_results
                ],
            },
        )
        return choice

    search_main.select_move_from_mcts_results = select_move_with_dump

    def find_best_move_with_priors(battle):
        _PRIOR_STATE["priors"] = None
        _PRIOR_STATE["opp_priors"] = None
        _PRIOR_STATE["context"] = None
        _PRIOR_STATE["remote_search"] = None
        try:
            tag = getattr(battle, "battle_tag", None)
            if not tag:
                raise RuntimeError("battle has no tag")
            full_tag = tag if tag.startswith("battle-") else f"battle-{tag}"
            from config import FoulPlayConfig

            username = quote(str(getattr(FoulPlayConfig, "username", "") or ""))
            with urllib.request.urlopen(
                f"{server_url}/priors?tag={full_tag}"
                f"&username={username}&namespace={quote(namespace)}",
                timeout=30,
            ) as response:
                payload = json.loads(response.read())

            priors = payload.get("priors") or {}
            opponent_priors = payload.get("opp_priors") or {}
            if not priors:
                raise RuntimeError("policy server returned no player priors")
            _PRIOR_STATE["priors"] = [
                (name, float(probability)) for name, probability in priors.items()
            ]
            _PRIOR_STATE["opp_priors"] = [
                (name, float(probability))
                for name, probability in opponent_priors.items()
            ] or None
            _PRIOR_STATE["context"] = {
                "tag": full_tag,
                "decision_idx": payload.get("decision_idx"),
                "battle_turn": payload.get("battle_turn"),
            }
            logger.info(
                f"loaded {len(priors)} player and {len(opponent_priors)} opponent priors"
            )
        except Exception as exc:
            if os.environ.get("METAGROSS_REQUIRE_PRIORS", "1") == "1":
                raise RuntimeError(f"required prior fetch failed: {exc!r}") from exc
            logger.warning("prior fetch failed; using unguided search: %r", exc)
        if os.environ.get("METAGROSS_REQUIRE_REMOTE_MCTS") == "1":
            return _remote_find_best_move(battle, search_main)
        return original_find_best_move(battle)

    search_main.find_best_move = find_best_move_with_priors
    run_battle.find_best_move = find_best_move_with_priors
    logger.info(
        f"root-prior patch active (server={server_url}, c_puct={_PRIOR_STATE['cpuct']})"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    provenance = inspect_poke_engine()
    print(f"POKE_ENGINE_PROVENANCE {json.dumps(provenance, sort_keys=True)}", flush=True)
    foul_play_dir = Path(
        os.environ.get("FOUL_PLAY_DIR", root / "srcs" / "vendor" / "foul-play")
    ).expanduser().resolve()
    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    os.chdir(foul_play_dir)
    sys.path.insert(0, str(foul_play_dir))
    patch_foul_play_protocol()
    patch_root_priors()

    from run import run_foul_play
    from config import FoulPlayConfig

    original_configure = FoulPlayConfig.configure

    def configure_with_environment_password():
        original_configure()
        password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
        if not password:
            raise RuntimeError("METAGROSS_SHOWDOWN_PASSWORD is required")
        FoulPlayConfig.password = password

    # Keep credentials out of argv and process listings. Foul Play receives the
    # password only after parsing its non-secret command line.
    FoulPlayConfig.configure = configure_with_environment_password

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
