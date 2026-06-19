from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import re
import secrets
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FORMAT = "gen9randombattle"
LOCAL_WEBSOCKET_URI = "ws://localhost:8000/showdown/websocket"
LIVE_WEBSOCKET_URI = "wss://sim3.psim.us/showdown/websocket"
SHOWDOWN_AUTH_URI = "https://play.pokemonshowdown.com/action.php?"
AGENT_NAMES = ("random", "max_damage", "foul_play")
EXPERIMENT_FIELDS = [
    "run_id",
    "date",
    "phase",
    "format",
    "change (ONE var)",
    "baseline",
    "N_games",
    "winrate",
    "CI95",
    "ladder_elo",
    "gxe",
    "belief_brier",
    "decision(advance/iterate/rollback)",
    "notes",
]


@dataclass
class GameResult:
    game_index: int
    agent_a: str
    agent_b: str
    challenger: str
    acceptor: str
    winner: Optional[str]
    winner_username: Optional[str]
    battle_tag: Optional[str]


@dataclass
class EvalSummary:
    mode: str
    format: str
    server: str
    agent_a: str
    agent_b: str
    n_games: int
    decisive_games: int
    agent_a_wins: int
    agent_a_losses: int
    ties_or_unknown: int
    winrate: float
    ci95_low: float
    ci95_high: float
    paired: bool
    foul_play_search_time_ms: int


class FoulPlayError(RuntimeError):
    pass


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0

    phat = wins / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def normalize_user_id(username: str) -> str:
    return re.sub(r"[^a-z0-9]", "", username.lower())


def make_username(role: str, game_index: int) -> str:
    suffix = secrets.token_hex(2)
    return f"p0{role}{game_index:03d}{suffix}"[:18]


def is_foul_play(agent: str) -> bool:
    return agent == "foul_play"


def agent_for_slot(args: argparse.Namespace, slot: str) -> str:
    if slot == "agent_a":
        return args.agent_a
    if slot == "agent_b":
        return args.agent_b
    raise ValueError(f"Unknown agent slot: {slot}")


def make_server_configuration(args: argparse.Namespace) -> ServerConfiguration:
    if args.websocket_uri:
        websocket_uri = args.websocket_uri
    elif args.server == "live":
        websocket_uri = LIVE_WEBSOCKET_URI
    else:
        websocket_uri = LOCAL_WEBSOCKET_URI

    return ServerConfiguration(websocket_uri, args.authentication_uri or SHOWDOWN_AUTH_URI)


def make_poke_env_player(
    agent: str,
    username: str,
    server_configuration: ServerConfiguration,
    battle_format: str,
) -> Player:
    account_configuration = AccountConfiguration(username, None)
    kwargs = {
        "account_configuration": account_configuration,
        "battle_format": battle_format,
        "max_concurrent_battles": 1,
        "server_configuration": server_configuration,
        "log_level": logging.WARNING,
    }
    if agent == "random":
        return RandomPlayer(**kwargs)
    if agent == "max_damage":
        return MaxBasePowerPlayer(**kwargs)
    raise ValueError(f"Unsupported poke-env agent: {agent}")


async def close_poke_env_player(player: Player) -> None:
    websocket = getattr(player.ps_client, "websocket", None)
    if websocket is not None:
        try:
            await websocket.close()
        except Exception:
            pass


def foul_play_command(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    username: str,
    bot_mode: str,
    user_to_challenge: Optional[str],
) -> list[str]:
    python_bin = Path(args.foul_play_python)
    runner = ROOT_DIR / "scripts" / "run_foul_play.py"
    cmd = [
        str(python_bin),
        str(runner),
        "--websocket-uri",
        server_configuration.websocket_url,
        "--ps-username",
        username,
        "--bot-mode",
        bot_mode,
        "--pokemon-format",
        args.format,
        "--run-count",
        "1",
        "--search-time-ms",
        str(args.foul_play_search_time_ms),
        "--search-parallelism",
        str(args.foul_play_search_parallelism),
        "--search-threads",
        str(args.foul_play_search_threads),
        "--log-level",
        args.foul_play_log_level,
    ]
    if args.password:
        cmd.extend(["--ps-password", args.password])
    if bot_mode == "challenge_user":
        if not user_to_challenge:
            raise ValueError("user_to_challenge is required for challenge_user mode")
        cmd.extend(["--user-to-challenge", user_to_challenge])
    return cmd


async def start_foul_play(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    username: str,
    bot_mode: str,
    user_to_challenge: Optional[str],
    log_dir: Path,
) -> tuple[asyncio.subprocess.Process, Path, object]:
    log_path = log_dir / f"{username}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        *foul_play_command(args, server_configuration, username, bot_mode, user_to_challenge),
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ROOT_DIR,
    )
    return proc, log_path, log_file


async def wait_for_foul_play(
    proc: asyncio.subprocess.Process,
    log_path: Path,
    log_file: object,
    timeout_seconds: int,
) -> str:
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        raise FoulPlayError(f"Foul Play timed out; log={log_path}") from exc
    finally:
        log_file.close()

    output = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FoulPlayError(
            f"Foul Play exited with code {proc.returncode}; log={log_path}\n{output[-4000:]}"
        )
    return output


async def ensure_foul_play_still_running(
    proc: asyncio.subprocess.Process,
    log_path: Path,
    log_file: object,
) -> None:
    if proc.returncode is None:
        return
    await wait_for_foul_play(proc, log_path, log_file, 1)


async def wait_for_external_battle(
    client_task: asyncio.Task,
    proc_task: asyncio.Task,
    timeout_seconds: int,
    client_finish_grace_seconds: int,
) -> str:
    done, _ = await asyncio.wait(
        {client_task, proc_task},
        timeout=timeout_seconds,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not done:
        client_task.cancel()
        raise FoulPlayError("Timed out waiting for external battle to make progress")

    if proc_task in done:
        output = await proc_task
        if not client_task.done():
            try:
                await asyncio.wait_for(client_task, timeout=client_finish_grace_seconds)
            except asyncio.TimeoutError:
                client_task.cancel()
        else:
            await client_task
        return output

    await client_task
    return await proc_task


def parse_foul_play_winner(output: str) -> Optional[str]:
    winner = None
    for line in output.splitlines():
        if "Winner:" in line:
            winner = line.split("Winner:", 1)[1].strip()
    if winner in {"", "None"}:
        return None
    return winner


def parse_foul_play_battle_tag(output: str) -> Optional[str]:
    for line in output.splitlines():
        if "Initialized battle-" in line:
            parts = line.split("Initialized ", 1)[-1].split(" against", 1)
            return parts[0].strip()
    return None


async def play_poke_env_vs_poke_env(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
) -> GameResult:
    challenger_agent = agent_for_slot(args, challenger_slot)
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    challenger_username = make_username("c", game_index)
    acceptor_username = make_username("a", game_index)
    challenger = make_poke_env_player(
        challenger_agent, challenger_username, server_configuration, args.format
    )
    acceptor = make_poke_env_player(
        acceptor_agent, acceptor_username, server_configuration, args.format
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(
                challenger.send_challenges(
                    acceptor_username, 1, to_wait=acceptor.ps_client.logged_in
                ),
                acceptor.accept_challenges(challenger_username, 1),
            ),
            timeout=args.game_timeout_seconds,
        )
        if challenger.n_won_battles == 1:
            winner = challenger_slot
            winner_username = challenger_username
        elif acceptor.n_won_battles == 1:
            winner = acceptor_slot
            winner_username = acceptor_username
        else:
            winner = None
            winner_username = None

        battle_tag = next(iter(challenger.battles.keys()), None)
        return GameResult(
            game_index,
            args.agent_a,
            args.agent_b,
            challenger_slot,
            acceptor_slot,
            winner,
            winner_username,
            battle_tag,
        )
    finally:
        await close_poke_env_player(challenger)
        await close_poke_env_player(acceptor)


async def play_foul_play_accepts_poke_env_challenge(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
) -> GameResult:
    challenger_agent = agent_for_slot(args, challenger_slot)
    fp_username = make_username("f", game_index)
    challenger_username = make_username("c", game_index)
    proc, log_path, log_file = await start_foul_play(
        args, server_configuration, fp_username, "accept_challenge", None, log_dir
    )
    await asyncio.sleep(args.foul_play_startup_delay_seconds)
    await ensure_foul_play_still_running(proc, log_path, log_file)

    challenger = make_poke_env_player(
        challenger_agent, challenger_username, server_configuration, args.format
    )
    try:
        client_task = asyncio.create_task(challenger.send_challenges(fp_username, n_challenges=1))
        proc_task = asyncio.create_task(
            wait_for_foul_play(proc, log_path, log_file, args.game_timeout_seconds)
        )
        output = await wait_for_external_battle(
            client_task,
            proc_task,
            args.game_timeout_seconds,
            args.client_finish_grace_seconds,
        )
        fp_winner = parse_foul_play_winner(output)
        battle_tag = parse_foul_play_battle_tag(output)

        if challenger.n_won_battles == 1:
            winner = challenger_slot
            winner_username = challenger_username
        elif fp_winner == fp_username or challenger.n_lost_battles == 1:
            winner = acceptor_slot
            winner_username = fp_username
        else:
            winner = None
            winner_username = fp_winner

        return GameResult(
            game_index,
            args.agent_a,
            args.agent_b,
            challenger_slot,
            acceptor_slot,
            winner,
            winner_username,
            battle_tag,
        )
    finally:
        await close_poke_env_player(challenger)


async def play_foul_play_challenges_poke_env(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
) -> GameResult:
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    fp_username = make_username("f", game_index)
    acceptor_username = make_username("a", game_index)
    acceptor = make_poke_env_player(
        acceptor_agent, acceptor_username, server_configuration, args.format
    )
    accept_task = asyncio.create_task(acceptor.accept_challenges(fp_username, 1))
    await asyncio.sleep(args.poke_env_startup_delay_seconds)

    proc, log_path, log_file = await start_foul_play(
        args,
        server_configuration,
        fp_username,
        "challenge_user",
        acceptor_username,
        log_dir,
    )
    try:
        proc_task = asyncio.create_task(
            wait_for_foul_play(proc, log_path, log_file, args.game_timeout_seconds)
        )
        output = await wait_for_external_battle(
            accept_task,
            proc_task,
            args.game_timeout_seconds,
            args.client_finish_grace_seconds,
        )
        fp_winner = parse_foul_play_winner(output)
        battle_tag = parse_foul_play_battle_tag(output)

        if acceptor.n_won_battles == 1:
            winner = acceptor_slot
            winner_username = acceptor_username
        elif fp_winner == fp_username or acceptor.n_lost_battles == 1:
            winner = challenger_slot
            winner_username = fp_username
        else:
            winner = None
            winner_username = fp_winner

        return GameResult(
            game_index,
            args.agent_a,
            args.agent_b,
            challenger_slot,
            acceptor_slot,
            winner,
            winner_username,
            battle_tag,
        )
    finally:
        if not accept_task.done():
            accept_task.cancel()
        await close_poke_env_player(acceptor)


async def play_foul_play_vs_foul_play(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
) -> GameResult:
    challenger_username = make_username("x", game_index)
    acceptor_username = make_username("y", game_index)
    acceptor_proc, acceptor_log_path, acceptor_log_file = await start_foul_play(
        args,
        server_configuration,
        acceptor_username,
        "accept_challenge",
        None,
        log_dir,
    )
    await asyncio.sleep(args.foul_play_startup_delay_seconds)
    challenger_proc, challenger_log_path, challenger_log_file = await start_foul_play(
        args,
        server_configuration,
        challenger_username,
        "challenge_user",
        acceptor_username,
        log_dir,
    )

    acceptor_output, challenger_output = await asyncio.gather(
        wait_for_foul_play(
            acceptor_proc,
            acceptor_log_path,
            acceptor_log_file,
            args.game_timeout_seconds,
        ),
        wait_for_foul_play(
            challenger_proc,
            challenger_log_path,
            challenger_log_file,
            args.game_timeout_seconds,
        ),
    )
    fp_winner = parse_foul_play_winner(acceptor_output) or parse_foul_play_winner(
        challenger_output
    )
    battle_tag = parse_foul_play_battle_tag(acceptor_output) or parse_foul_play_battle_tag(
        challenger_output
    )
    if fp_winner == challenger_username:
        winner = challenger_slot
    elif fp_winner == acceptor_username:
        winner = acceptor_slot
    else:
        winner = None

    return GameResult(
        game_index,
        args.agent_a,
        args.agent_b,
        challenger_slot,
        acceptor_slot,
        winner,
        fp_winner,
        battle_tag,
    )


async def play_one_game(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
) -> GameResult:
    challenger_agent = agent_for_slot(args, challenger_slot)
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    challenger_is_fp = is_foul_play(challenger_agent)
    acceptor_is_fp = is_foul_play(acceptor_agent)
    if challenger_is_fp and acceptor_is_fp:
        return await play_foul_play_vs_foul_play(
            args, server_configuration, game_index, challenger_slot, acceptor_slot, log_dir
        )
    if challenger_is_fp:
        return await play_foul_play_challenges_poke_env(
            args, server_configuration, game_index, challenger_slot, acceptor_slot, log_dir
        )
    if acceptor_is_fp:
        return await play_foul_play_accepts_poke_env_challenge(
            args, server_configuration, game_index, challenger_slot, acceptor_slot, log_dir
        )
    return await play_poke_env_vs_poke_env(
        args, server_configuration, game_index, challenger_slot, acceptor_slot
    )


def side_schedule(n_games: int, paired: bool) -> list[tuple[str, str]]:
    if paired:
        if n_games % 2 != 0:
            raise ValueError("--paired requires an even --n-games value")
        schedule = []
        for _ in range(n_games // 2):
            schedule.append(("agent_a", "agent_b"))
            schedule.append(("agent_b", "agent_a"))
        return schedule
    return [("agent_a", "agent_b") for _ in range(n_games)]


async def run_h2h(args: argparse.Namespace) -> tuple[EvalSummary, list[GameResult]]:
    server_configuration = make_server_configuration(args)
    schedule = side_schedule(args.n_games, args.paired)

    results: list[GameResult] = []
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for index, (challenger, acceptor) in enumerate(schedule, start=1):
            print(
                "starting game={} challenger={}({}) acceptor={}({})".format(
                    index,
                    challenger,
                    agent_for_slot(args, challenger),
                    acceptor,
                    agent_for_slot(args, acceptor),
                ),
                flush=True,
            )
            result = await play_one_game(
                args, server_configuration, index, challenger, acceptor, log_dir
            )
            results.append(result)
            print(
                f"game={index} challenger={challenger} acceptor={acceptor} winner={result.winner}",
                flush=True,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="phase0-eval-") as temp_dir_name:
            log_dir = Path(temp_dir_name)
            for index, (challenger, acceptor) in enumerate(schedule, start=1):
                print(
                    "starting game={} challenger={}({}) acceptor={}({})".format(
                        index,
                        challenger,
                        agent_for_slot(args, challenger),
                        acceptor,
                        agent_for_slot(args, acceptor),
                    ),
                    flush=True,
                )
                result = await play_one_game(
                    args, server_configuration, index, challenger, acceptor, log_dir
                )
                results.append(result)
                print(
                    f"game={index} challenger={challenger} acceptor={acceptor} winner={result.winner}",
                    flush=True,
                )

    agent_a_wins = sum(1 for result in results if result.winner == "agent_a")
    agent_a_losses = sum(1 for result in results if result.winner == "agent_b")
    ties_or_unknown = len(results) - agent_a_wins - agent_a_losses
    decisive_games = agent_a_wins + agent_a_losses
    winrate = agent_a_wins / decisive_games if decisive_games else 0.0
    ci_low, ci_high = wilson_ci(agent_a_wins, decisive_games)
    summary = EvalSummary(
        mode="h2h",
        format=args.format,
        server=args.server,
        agent_a=args.agent_a,
        agent_b=args.agent_b,
        n_games=len(results),
        decisive_games=decisive_games,
        agent_a_wins=agent_a_wins,
        agent_a_losses=agent_a_losses,
        ties_or_unknown=ties_or_unknown,
        winrate=winrate,
        ci95_low=ci_low,
        ci95_high=ci_high,
        paired=args.paired,
        foul_play_search_time_ms=args.foul_play_search_time_ms,
    )
    return summary, results


def fetch_ladder_rating(username: str, battle_format: str) -> dict[str, Optional[float]]:
    user_id = normalize_user_id(username)
    url = f"https://pokemonshowdown.com/users/{user_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"elo": None, "gxe": None}

    rating = payload.get("ratings", {}).get(battle_format, {})
    return {"elo": rating.get("elo"), "gxe": rating.get("gxe")}


async def run_ladder(args: argparse.Namespace) -> dict[str, object]:
    if not args.username:
        raise ValueError("--mode ladder requires --username")
    server_configuration = make_server_configuration(args)
    if is_foul_play(args.agent):
        with tempfile.TemporaryDirectory(prefix="phase0-ladder-") as temp_dir_name:
            proc, log_path, log_file = await start_foul_play(
                args,
                server_configuration,
                args.username,
                "search_ladder",
                None,
                Path(temp_dir_name),
            )
            output = await wait_for_foul_play(
                proc, log_path, log_file, args.game_timeout_seconds * args.n_games
            )
            result = {"agent": args.agent, "username": args.username, "output_tail": output[-4000:]}
    else:
        player = make_poke_env_player(
            args.agent, args.username, server_configuration, args.format
        )
        try:
            await asyncio.wait_for(
                player.ladder(args.n_games), timeout=args.game_timeout_seconds * args.n_games
            )
            result = {
                "agent": args.agent,
                "username": args.username,
                "finished": player.n_finished_battles,
                "wins": player.n_won_battles,
                "losses": player.n_lost_battles,
            }
        finally:
            await close_poke_env_player(player)

    result.update(fetch_ladder_rating(args.username, args.format))
    return result


def append_experiment_row(args: argparse.Namespace, summary: EvalSummary) -> None:
    path = Path(args.append_experiment_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    row = {
        "run_id": args.run_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "phase": "0",
        "format": summary.format,
        "change (ONE var)": "stock_foul_play_baseline",
        "baseline": f"{summary.agent_a}_vs_{summary.agent_b}",
        "N_games": str(summary.n_games),
        "winrate": f"{summary.winrate:.4f}",
        "CI95": f"[{summary.ci95_low:.4f}, {summary.ci95_high:.4f}]",
        "ladder_elo": "",
        "gxe": "",
        "belief_brier": "",
        "decision(advance/iterate/rollback)": "record",
        "notes": (
            f"paired={summary.paired}; decisive={summary.decisive_games}; "
            f"ties_or_unknown={summary.ties_or_unknown}; "
            f"foul_play_search_time_ms={summary.foul_play_search_time_ms}"
        ),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 Pokemon Showdown eval harness")
    parser.add_argument("--mode", choices=["h2h", "ladder"], default="h2h")
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--server", choices=["local", "live"], default="local")
    parser.add_argument("--websocket-uri", default=None)
    parser.add_argument("--authentication-uri", default=None)
    parser.add_argument("--n-games", type=int, default=2)
    parser.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-a", choices=AGENT_NAMES, default="foul_play")
    parser.add_argument("--agent-b", choices=AGENT_NAMES, default="random")
    parser.add_argument("--agent", choices=AGENT_NAMES, default="foul_play")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--game-timeout-seconds", type=int, default=900)
    parser.add_argument("--client-finish-grace-seconds", type=int, default=30)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--foul-play-python", default=str(ROOT_DIR / ".venv-foul-play" / "bin" / "python"))
    parser.add_argument("--foul-play-search-time-ms", type=int, default=100)
    parser.add_argument("--foul-play-search-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--foul-play-startup-delay-seconds", type=float, default=5.0)
    parser.add_argument("--poke-env-startup-delay-seconds", type=float, default=3.0)
    parser.add_argument("--foul-play-log-level", default="INFO")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--append-experiment-log", default=None)
    parser.add_argument(
        "--run-id",
        default=f"phase0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)
    if args.list_agents:
        print("\n".join(AGENT_NAMES))
        raise SystemExit(0)
    if args.n_games <= 0:
        raise ValueError("--n-games must be positive")
    return args


async def async_main(args: argparse.Namespace) -> None:
    if args.mode == "ladder":
        result = await run_ladder(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.json_out:
            write_json(args.json_out, result)
        return

    summary, results = await run_h2h(args)
    payload = {
        "summary": asdict(summary),
        "games": [asdict(result) for result in results],
    }
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.json_out:
        write_json(args.json_out, payload)
    if args.append_experiment_log:
        append_experiment_row(args, summary)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
