#!/usr/bin/env python3
"""A poke-env RandomPlayer that plays challenge battles against a named opponent.

Used as the sanity-gate peer for verifying a Metamon policy is actually loaded
and acting (the policy must crush random). Runs in .venv-metamon.
"""
from __future__ import annotations

import argparse
import asyncio

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--opponent-username", required=True)
    parser.add_argument("--role", choices=["challenger", "acceptor"], required=True)
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--n-games", type=int, default=1)
    args = parser.parse_args()

    player = RandomPlayer(
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.battle_format,
        server_configuration=LOCAL,
        max_concurrent_battles=1,
    )
    if args.role == "acceptor":
        await player.accept_challenges(args.opponent_username, args.n_games)
    else:
        for _ in range(args.n_games):
            await player.send_challenges(args.opponent_username, 1)
    print(f"RANDOM_PEER done: W={player.n_won_battles} "
          f"L={player.n_lost_battles} T={player.n_tied_battles}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
