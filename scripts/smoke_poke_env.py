import asyncio
from importlib.metadata import version

from poke_env.player import MaxBasePowerPlayer, RandomPlayer


async def main() -> None:
    random_player = RandomPlayer(
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )
    max_power_player = MaxBasePowerPlayer(
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )

    await random_player.battle_against(max_power_player, n_battles=1)

    print(f"poke-env={version('poke-env')}")
    print(f"finished={random_player.n_finished_battles}")
    print(f"random_wins={random_player.n_won_battles}")
    print(f"max_power_wins={max_power_player.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())
