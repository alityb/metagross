#!/usr/bin/env python3
"""Run Foul Play with the accepted r1 root-prior integration only."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path


_PRIOR_STATE = {
    "priors": None,
    "opp_priors": None,
    "cpuct": 2.0,
}


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

    def connect_without_ping(address, *args, **kwargs):
        # Search can block the event loop longer than the websocket ping timeout.
        kwargs.setdefault("ping_interval", None)
        return original_connect(address, *args, **kwargs)

    websocket_client.websockets.connect = connect_without_ping

    original_receive = websocket_client.PSWebsocketClient.receive_message

    async def receive_with_rating_log(self):
        message = await original_receive(self)
        for line in message.splitlines():
            if line.startswith("|raw|") and (
                "<strong>" in line or "rating:" in line.lower()
            ):
                print(f"RATING_LINE {line}", flush=True)
        return message

    websocket_client.PSWebsocketClient.receive_message = receive_with_rating_log


def _mcts_with_root_priors(state_str, search_time_ms, index, threads=1):
    """Run the patched engine with player and opponent root priors."""
    import poke_engine
    from config import FoulPlayConfig

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

    def find_best_move_with_priors(battle):
        _PRIOR_STATE["priors"] = None
        _PRIOR_STATE["opp_priors"] = None
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
            logger.info(
                "loaded %d player and %d opponent priors",
                len(priors),
                len(opponent_priors),
            )
        except Exception as exc:
            if os.environ.get("METAGROSS_REQUIRE_PRIORS", "1") == "1":
                raise RuntimeError(f"required prior fetch failed: {exc!r}") from exc
            logger.warning("prior fetch failed; using unguided search: %r", exc)
        return original_find_best_move(battle)

    search_main.find_best_move = find_best_move_with_priors
    run_battle.find_best_move = find_best_move_with_priors
    logger.info(
        "root-prior patch active (server=%s, c_puct=%s)",
        server_url,
        _PRIOR_STATE["cpuct"],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    foul_play_dir = Path(
        os.environ.get("FOUL_PLAY_DIR", root / "srcs" / "vendor" / "foul-play")
    ).expanduser().resolve()
    password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
    if password and "--ps-password" not in sys.argv:
        sys.argv.extend(["--ps-password", password])

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

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
