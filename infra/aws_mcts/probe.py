#!/usr/bin/env python3
"""Exercise a remote HTTP MCTS service with a deterministic synthetic state."""

from __future__ import annotations

import argparse
import os
import statistics
import time
import uuid

from srcs.metagross.mcts_contract import REQUEST_SCHEMA, validate_request


def synthetic_state() -> str:
    import poke_engine

    def pokemon(identifier: str, types: tuple[str, str], moves: list[str]):
        return poke_engine.Pokemon(
            id=identifier,
            level=100,
            types=types,
            hp=100,
            maxhp=100,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=100,
            status="none",
            moves=[poke_engine.Move(id=move, pp=32) for move in moves],
        )

    state = poke_engine.State(
        side_one=poke_engine.Side(
            pokemon=[
                pokemon(
                    "squirtle",
                    ("water", "typeless"),
                    ["watergun", "tackle", "quickattack", "leer"],
                )
            ]
        ),
        side_two=poke_engine.Side(
            pokemon=[
                pokemon(
                    "charmander",
                    ("fire", "typeless"),
                    ["ember", "tackle", "quickattack", "leer"],
                )
            ]
        ),
        weather="none",
        weather_turns_remaining=-1,
        terrain="none",
        terrain_turns_remaining=-1,
        trick_room=False,
        trick_room_turns_remaining=-1,
    )
    return state.to_string()


def build_requests(
    state: str, worlds: int, duration_ms: int
) -> list[dict[str, object]]:
    requests = [
        {
            "schema": REQUEST_SCHEMA,
            "operation": "search",
            "request_id": uuid.uuid4().hex,
            "index": index,
            "state": state,
            "duration_ms": duration_ms,
            "threads": 1,
            "s1_priors": None,
            "s2_priors": None,
            "c_puct": 2.0,
        }
        for index in range(worlds)
    ]
    for request in requests:
        validate_request(request)
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--duration-ms", type=int, choices=(250, 500), default=250)
    return parser.parse_args()


def main() -> None:
    from srcs.metagross import run_foul_play

    args = parse_args()
    state = synthetic_state()
    requests = build_requests(state, args.worlds, args.duration_ms)
    started = time.monotonic()
    responses = run_foul_play._http_mcts_call(requests)
    rpc_ms = (time.monotonic() - started) * 1000
    if not isinstance(responses, list) or len(responses) != args.worlds:
        raise RuntimeError("probe received the wrong response count")
    validated = [
        run_foul_play._validate_remote_response(response, request["request_id"], index)
        for index, (request, response) in enumerate(zip(requests, responses, strict=True))
    ]
    visits = [int(response["result"]["total_visits"]) for response in validated]
    search_ms = [float(response["timing"]["search_ms"]) for response in validated]
    print(
        {
            "worlds": args.worlds,
            "duration_ms": args.duration_ms,
            "rpc_ms": round(rpc_ms, 3),
            "total_visits": sum(visits),
            "median_visits": statistics.median(visits),
            "max_search_ms": max(search_ms),
            "native_sha256": validated[0]["engine"]["native_sha256"],
            "resources": validated[0]["engine"]["resources"],
        }
    )


if __name__ == "__main__":
    if not os.environ.get("METAGROSS_REMOTE_MCTS_TOKEN"):
        raise RuntimeError("METAGROSS_REMOTE_MCTS_TOKEN is required")
    main()
