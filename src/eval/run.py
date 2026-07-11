from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
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


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FORMAT = "gen9randombattle"
LOCAL_WEBSOCKET_URI = "ws://localhost:8000/showdown/websocket"
LIVE_WEBSOCKET_URI = "wss://sim3.psim.us/showdown/websocket"
SHOWDOWN_AUTH_URI = "https://play.pokemonshowdown.com/action.php?"
AGENT_NAMES = (
    "random",
    "max_damage",
    "foul_play",
    "foul_play_learned",
    "foul_play_randbats_pool",
    "foul_play_randbats_conditional",
    "foul_play_tauros_kind",
    "foul_play_tauros_action",
    "foul_play_value_shield",
    "foul_play_belief_threat",
    "foul_play_wincon",
    "foul_play_pp_stall",
    "foul_play_opp_priors",
    "foul_play_root_priors",
    "foul_play_root_priors_opp",
)
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
    void: bool = False
    error: Optional[str] = None


@dataclass
class EvalSummary:
    mode: str
    format: str
    server: str
    agent_a: str
    agent_b: str
    n_games: int
    completed_games: int
    void_games: int
    decisive_games: int
    agent_a_wins: int
    agent_a_losses: int
    ties_or_unknown: int
    winrate: float
    ci95_low: float
    ci95_high: float
    paired: bool
    foul_play_search_time_ms: int
    agent_a_as_challenger_wins: int
    agent_a_as_challenger_games: int
    agent_a_as_acceptor_wins: int
    agent_a_as_acceptor_games: int
    voids_with_agent_a_challenger: int
    voids_with_agent_b_challenger: int
    sprt_decision: str
    sprt_llr: float
    scorer_gate_passed: bool
    scorer_gate_message: str


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


def sprt_llr(wins: int, losses: int, p0: float, p1: float) -> float:
    """Log-likelihood ratio for H1(p1) vs H0(p0) after wins/losses."""
    if wins + losses == 0:
        return 0.0
    return wins * math.log(p1 / p0) + losses * math.log((1.0 - p1) / (1.0 - p0))


def sprt_check(wins: int, losses: int, p0: float, p1: float,
               alpha: float = 0.05, beta: float = 0.05) -> str:
    """Returns 'accept_h1' (effect real), 'accept_h0' (no effect), or 'continue'."""
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    llr = sprt_llr(wins, losses, p0, p1)
    if llr >= upper:
        return "accept_h1"
    if llr <= lower:
        return "accept_h0"
    return "continue"


def scorer_gate_check(wins: int, losses: int, voids: int) -> tuple[bool, str]:
    """§6.3 powered self-play scorer gate. Returns (passed, message)."""
    n = wins + losses
    if n < 100:
        return False, f"insufficient decisive games: {n} < 100"
    wr = wins / n
    ci_low, ci_high = wilson_ci(wins, n)
    if not (0.45 <= wr <= 0.55):
        return False, f"winrate {wr:.4f} outside [0.45, 0.55]"
    if not (ci_low <= 0.50 <= ci_high):
        return False, f"CI [{ci_low:.4f}, {ci_high:.4f}] does not contain 0.50"
    if not (ci_low >= 0.40 and ci_high <= 0.60):
        return False, f"CI [{ci_low:.4f}, {ci_high:.44}] not contained in [0.40, 0.60]"
    if voids > 0:
        return False, f"{voids} void games (check for unexplained ties/unknowns)"
    return True, f"PASS: wr={wr:.4f} CI=[{ci_low:.4f}, {ci_high:.4f}] n={n} voids={voids}"


def normalize_user_id(username: str) -> str:
    return re.sub(r"[^a-z0-9]", "", username.lower())


def make_username(role: str, game_index: int) -> str:
    suffix = secrets.token_hex(2)
    return f"p0{role}{game_index:03d}{suffix}"[:18]


def is_foul_play(agent: str) -> bool:
    return agent in {
        "foul_play",
        "foul_play_learned",
        "foul_play_randbats_pool",
        "foul_play_randbats_conditional",
        "foul_play_tauros_kind",
        "foul_play_tauros_action",
        "foul_play_value_shield",
        "foul_play_belief_threat",
        "foul_play_wincon",
        "foul_play_pp_stall",
        "foul_play_opp_priors",
        "foul_play_root_priors",
        "foul_play_root_priors_opp",
    }


def is_learned_foul_play(agent: str) -> bool:
    return agent == "foul_play_learned"


def is_randbats_pool_foul_play(agent: str) -> bool:
    return agent == "foul_play_randbats_pool"


def is_randbats_conditional_foul_play(agent: str) -> bool:
    return agent == "foul_play_randbats_conditional"


def is_tauros_kind_foul_play(agent: str) -> bool:
    return agent in {"foul_play_tauros_kind", "foul_play_tauros_action"}


def is_value_shield_foul_play(agent: str) -> bool:
    return agent == "foul_play_value_shield"


def is_belief_threat_foul_play(agent: str) -> bool:
    return agent == "foul_play_belief_threat"


def is_opp_priors_foul_play(agent: str) -> bool:
    return agent == "foul_play_opp_priors"


def is_wincon_foul_play(agent: str) -> bool:
    return agent == "foul_play_wincon"


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
    slot: Optional[str] = None,
) -> list[str]:
    # Per-slot Python binary override (for A/B testing different poke-engine builds)
    if slot == "agent_a" and getattr(args, "agent_a_python", None):
        python_bin = Path(args.agent_a_python)
    elif slot == "agent_b" and getattr(args, "agent_b_python", None):
        python_bin = Path(args.agent_b_python)
    else:
        python_bin = Path(args.foul_play_python)
    # Per-slot search budget override (for budget-scaling A/B: agent_a@X ms vs agent_b@Y ms)
    if slot == "agent_a" and getattr(args, "agent_a_search_time_ms", None):
        search_time_ms = args.agent_a_search_time_ms
    elif slot == "agent_b" and getattr(args, "agent_b_search_time_ms", None):
        search_time_ms = args.agent_b_search_time_ms
    else:
        search_time_ms = args.foul_play_search_time_ms
    runner = ROOT_DIR / "src" / "scripts" / "run_foul_play.py"
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
        str(getattr(args, "n_games", 1) if bot_mode == "search_ladder" else 1),
        "--search-time-ms",
        str(search_time_ms),
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


def model_for_agent(args: argparse.Namespace, agent: str) -> Optional[str]:
    """Return the model path for the given agent, respecting per-slot overrides."""
    if agent == "foul_play_learned":
        # Per-slot overrides take priority over the shared --learned-value-model
        # They're stored as args.agent_a_model / args.agent_b_model and resolved
        # by the caller from the slot name.
        return args.learned_value_model or None
    return None


def model_for_slot(args: argparse.Namespace, slot: str) -> Optional[str]:
    """Return per-slot model override, falling back to shared --learned-value-model."""
    if slot == "agent_a" and getattr(args, "agent_a_model", None):
        return args.agent_a_model
    if slot == "agent_b" and getattr(args, "agent_b_model", None):
        return args.agent_b_model
    return args.learned_value_model


def foul_play_env(
    args: argparse.Namespace,
    agent: str,
    model_override: Optional[str] = None,
    slot: Optional[str] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    model = model_override if model_override is not None else model_for_agent(args, agent)
    if is_learned_foul_play(agent):
        if not model:
            raise ValueError("foul_play_learned requires --learned-value-model or a per-slot model override")
        env["METAGROSS_VALUE_MODEL"] = str(Path(model).resolve())
    else:
        env.pop("METAGROSS_VALUE_MODEL", None)
    if is_randbats_pool_foul_play(agent):
        if not args.randbats_belief_pool:
            raise ValueError("foul_play_randbats_pool requires --randbats-belief-pool")
        env["METAGROSS_RANDBATS_POOL"] = str(Path(args.randbats_belief_pool).resolve())
    else:
        env.pop("METAGROSS_RANDBATS_POOL", None)
    if is_randbats_conditional_foul_play(agent):
        env["METAGROSS_RANDBATS_CONDITIONAL_SCRIPT"] = str(
            Path(args.randbats_conditional_script).resolve()
        )
        env["METAGROSS_RANDBATS_CONDITIONAL_SAMPLES"] = str(args.randbats_conditional_samples)
        env["METAGROSS_RANDBATS_CONDITIONAL_MAX_TEAMS"] = str(args.randbats_conditional_max_teams)
        env["METAGROSS_RANDBATS_CONDITIONAL_MAX_MS"] = str(args.randbats_conditional_max_ms)
        env["METAGROSS_RANDBATS_CONDITIONAL_TIMEOUT_S"] = str(args.randbats_conditional_timeout_seconds)
        env["METAGROSS_RANDBATS_FORMAT"] = args.format
    else:
        env.pop("METAGROSS_RANDBATS_CONDITIONAL_SCRIPT", None)
        env.pop("METAGROSS_RANDBATS_CONDITIONAL_SAMPLES", None)
        env.pop("METAGROSS_RANDBATS_CONDITIONAL_MAX_TEAMS", None)
        env.pop("METAGROSS_RANDBATS_CONDITIONAL_MAX_MS", None)
        env.pop("METAGROSS_RANDBATS_CONDITIONAL_TIMEOUT_S", None)
        env.pop("METAGROSS_RANDBATS_FORMAT", None)
    prior_url = args.prior_server_url
    if slot == "agent_a" and getattr(args, "agent_a_prior_server_url", None):
        prior_url = args.agent_a_prior_server_url
    elif slot == "agent_b" and getattr(args, "agent_b_prior_server_url", None):
        prior_url = args.agent_b_prior_server_url
    if agent in ("foul_play_root_priors", "foul_play_root_priors_opp"):
        env["METAGROSS_PRIOR_SERVER"] = prior_url
        env["METAGROSS_CPUCT"] = str(args.cpuct)
    elif is_opp_priors_foul_play(agent):
        env["METAGROSS_PRIOR_SERVER"] = prior_url
        env["METAGROSS_CPUCT"] = str(args.cpuct)
        env["METAGROSS_OPP_PRIORS_ONLY"] = "1"
    else:
        env.pop("METAGROSS_PRIOR_SERVER", None)
        env.pop("METAGROSS_CPUCT", None)
        env.pop("METAGROSS_OPP_PRIORS_ONLY", None)

    if slot in ("agent_a", "agent_b"):
        prefix = slot.replace("agent_", "agent_")
        decision_log = getattr(args, f"{prefix}_decision_log", None)
        replay_dir = getattr(args, f"{prefix}_replay_dir", None)
        require_priors = getattr(args, f"{prefix}_require_priors", False)
        if decision_log:
            env["METAGROSS_DECISION_LOG"] = str(Path(decision_log).resolve())
        if replay_dir:
            env["METAGROSS_REPLAY_DIR"] = str(Path(replay_dir).resolve())
        if require_priors:
            env["METAGROSS_REQUIRE_PRIORS"] = "1"
        else:
            env.pop("METAGROSS_REQUIRE_PRIORS", None)
    if is_tauros_kind_foul_play(agent):
        env["METAGROSS_TAUROS_KIND_MODEL"] = str(Path(args.tauros_kind_model).resolve())
        env["METAGROSS_TAUROS_KIND_THRESHOLD"] = str(args.tauros_kind_threshold)
        env["METAGROSS_TAUROS_KIND_MIN_POLICY_FRAC"] = str(args.tauros_kind_min_policy_frac)
        env["METAGROSS_TAUROS_KIND_ALLOWED_KINDS"] = args.tauros_kind_allowed_kinds
    else:
        env.pop("METAGROSS_TAUROS_KIND_MODEL", None)
        env.pop("METAGROSS_TAUROS_KIND_THRESHOLD", None)
        env.pop("METAGROSS_TAUROS_KIND_MIN_POLICY_FRAC", None)
        env.pop("METAGROSS_TAUROS_KIND_ALLOWED_KINDS", None)
    if is_belief_threat_foul_play(agent):
        env["METAGROSS_BELIEF_EVAL"] = "1"
    elif is_wincon_foul_play(agent):
        env["METAGROSS_WINCON_EVAL"] = "1"
    else:
        env.pop("METAGROSS_BELIEF_EVAL", None)
        env.pop("METAGROSS_WINCON_EVAL", None)
    if agent == "foul_play_pp_stall":
        env["METAGROSS_PP_STALL"] = "1"
    else:
        env.pop("METAGROSS_PP_STALL", None)
    # Pass through data-capture env vars (self-play / ExIt pipeline)
    for passthrough in ("METAGROSS_REPLAY_DIR", "METAGROSS_DECISION_LOG", "METAGROSS_BELIEF_LOG"):
        val = os.environ.get(passthrough)
        if val:
            env[passthrough] = val
    if is_value_shield_foul_play(agent):
        env["METAGROSS_FP_VALUE_SHIELD"] = "1"
        env["METAGROSS_FP_VALUE_SHIELD_MARGIN"] = str(args.value_shield_margin)
        env["METAGROSS_FP_VALUE_SHIELD_MIN_SUPPORT"] = str(args.value_shield_min_support)
        env["METAGROSS_FP_VALUE_SHIELD_CLOSE_POLICY_FRAC"] = str(args.value_shield_close_policy_frac)
        if args.value_shield_log:
            env["METAGROSS_FP_VALUE_SHIELD_LOG"] = str(Path(args.value_shield_log).resolve())
        else:
            env.pop("METAGROSS_FP_VALUE_SHIELD_LOG", None)
    else:
        env.pop("METAGROSS_FP_VALUE_SHIELD", None)
        env.pop("METAGROSS_FP_VALUE_SHIELD_MARGIN", None)
        env.pop("METAGROSS_FP_VALUE_SHIELD_MIN_SUPPORT", None)
        env.pop("METAGROSS_FP_VALUE_SHIELD_CLOSE_POLICY_FRAC", None)
        env.pop("METAGROSS_FP_VALUE_SHIELD_LOG", None)
    return env


async def start_foul_play(
    args: argparse.Namespace,
    agent: str,
    server_configuration: ServerConfiguration,
    username: str,
    bot_mode: str,
    user_to_challenge: Optional[str],
    log_dir: Path,
    model_override: Optional[str] = None,
    slot: Optional[str] = None,
) -> tuple[asyncio.subprocess.Process, Path, object]:
    log_path = log_dir / f"{username}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        *foul_play_command(args, server_configuration, username, bot_mode, user_to_challenge, slot=slot),
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ROOT_DIR,
        env=foul_play_env(args, agent, model_override, slot=slot),
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


async def terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


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
        proc_task.cancel()
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
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    fp_username = make_username("f", game_index)
    challenger_username = make_username("c", game_index)
    proc, log_path, log_file = await start_foul_play(
        args,
        acceptor_agent,
        server_configuration,
        fp_username,
        "accept_challenge",
        None,
        log_dir,
        model_override=model_for_slot(args, acceptor_slot),
    )
    await asyncio.sleep(args.foul_play_startup_delay_seconds)
    await ensure_foul_play_still_running(proc, log_path, log_file)

    challenger = make_poke_env_player(
        challenger_agent, challenger_username, server_configuration, args.format
    )
    proc_task = None
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
    except Exception:
        if proc_task is not None and not proc_task.done():
            proc_task.cancel()
        await terminate_process(proc)
        if proc_task is not None:
            await asyncio.gather(proc_task, return_exceptions=True)
        raise
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
    challenger_agent = agent_for_slot(args, challenger_slot)
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
        challenger_agent,
        server_configuration,
        fp_username,
        "challenge_user",
        acceptor_username,
        log_dir,
        model_override=model_for_slot(args, challenger_slot),
    )
    proc_task = None
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
    except Exception:
        if proc_task is not None and not proc_task.done():
            proc_task.cancel()
        await terminate_process(proc)
        if proc_task is not None:
            await asyncio.gather(proc_task, return_exceptions=True)
        raise
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
    challenger_agent = agent_for_slot(args, challenger_slot)
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    challenger_username = make_username("x", game_index)
    acceptor_username = make_username("y", game_index)
    acceptor_proc, acceptor_log_path, acceptor_log_file = await start_foul_play(
        args,
        acceptor_agent,
        server_configuration,
        acceptor_username,
        "accept_challenge",
        None,
        log_dir,
        model_override=model_for_slot(args, acceptor_slot),
        slot=acceptor_slot,
    )
    await asyncio.sleep(args.foul_play_startup_delay_seconds)
    challenger_proc, challenger_log_path, challenger_log_file = await start_foul_play(
        args,
        challenger_agent,
        server_configuration,
        challenger_username,
        "challenge_user",
        acceptor_username,
        log_dir,
        model_override=model_for_slot(args, challenger_slot),
        slot=challenger_slot,
    )

    acceptor_task = asyncio.create_task(
        wait_for_foul_play(
            acceptor_proc,
            acceptor_log_path,
            acceptor_log_file,
            args.game_timeout_seconds,
        )
    )
    challenger_task = asyncio.create_task(
        wait_for_foul_play(
            challenger_proc,
            challenger_log_path,
            challenger_log_file,
            args.game_timeout_seconds,
        )
    )
    try:
        acceptor_output, challenger_output = await asyncio.gather(
            acceptor_task, challenger_task
        )
    except Exception:
        for task in (acceptor_task, challenger_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(acceptor_task, challenger_task, return_exceptions=True)
        await asyncio.gather(
            terminate_process(acceptor_proc),
            terminate_process(challenger_proc),
        )
        raise
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


def short_error(exc: Exception) -> str:
    message = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {message[:500]}"


async def run_scheduled_game(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    index: int,
    challenger: str,
    acceptor: str,
    log_dir: Path,
) -> GameResult:
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
    try:
        result = await play_one_game(
            args, server_configuration, index, challenger, acceptor, log_dir
        )
    except Exception as exc:
        if args.fail_fast:
            raise
        result = GameResult(
            index,
            args.agent_a,
            args.agent_b,
            challenger,
            acceptor,
            None,
            None,
            None,
            void=True,
            error=short_error(exc),
        )
        print(
            f"game={index} challenger={challenger} acceptor={acceptor} void=true error={result.error}",
            flush=True,
        )
        return result

    print(
        f"game={index} challenger={challenger} acceptor={acceptor} winner={result.winner}",
        flush=True,
    )
    return result


def emit_progress(args: argparse.Namespace, result: GameResult, results: list[GameResult]) -> None:
    """Append-only per-game progress so a crash mid-run still yields a usable partial."""
    decisive = [r for r in results if not r.void and r.winner in {"agent_a", "agent_b"}]
    wins = sum(1 for r in decisive if r.winner == "agent_a")
    losses = sum(1 for r in decisive if r.winner == "agent_b")
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "game_index": result.game_index,
        "winner": result.winner,
        "void": result.void,
        "error": result.error,
        "running_agent_a_wins": wins,
        "running_decisive": len(decisive),
        "running_winrate": round(wins / len(decisive), 4) if decisive else None,
    }
    if getattr(args, "sprt_h1", None):
        p0 = args.sprt_h0
        p1 = args.sprt_h1
        llr = sprt_llr(wins, losses, p0, p1)
        decision = sprt_check(wins, losses, p0, p1)
        line["sprt_llr"] = round(llr, 4)
        line["sprt_decision"] = decision
    print(f"PROGRESS {json.dumps(line, sort_keys=True)}", flush=True)
    if getattr(args, "json_out", None):
        progress_path = Path(str(args.json_out) + ".progress.jsonl")
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


async def run_h2h(args: argparse.Namespace) -> tuple[EvalSummary, list[GameResult]]:
    server_configuration = make_server_configuration(args)
    schedule = side_schedule(args.n_games, args.paired)
    use_sprt = getattr(args, "sprt_h1", None) is not None

    results: list[GameResult] = []
    max_concurrent = getattr(args, "concurrent_games", 1)
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if max_concurrent <= 1:
            for index, (challenger, acceptor) in enumerate(schedule, start=1):
                result = await run_scheduled_game(
                    args, server_configuration, index, challenger, acceptor, log_dir
                )
                results.append(result)
                emit_progress(args, result, results)
                if use_sprt:
                    decisive = [r for r in results if not r.void and r.winner in {"agent_a", "agent_b"}]
                    w = sum(1 for r in decisive if r.winner == "agent_a")
                    l = sum(1 for r in decisive if r.winner == "agent_b")
                    decision = sprt_check(w, l, args.sprt_h0, args.sprt_h1)
                    if decision != "continue":
                        print(f"SPRT STOP: {decision} after {len(decisive)} decisive games (w={w} l={l})", flush=True)
                        break
        else:
            # Concurrent game batches
            from collections import deque
            queue = deque(enumerate(schedule, start=1))
            pending = set()
            async def _run_one(idx, chal, acc):
                r = await run_scheduled_game(args, server_configuration, idx, chal, acc, log_dir)
                return idx, r
            while queue or pending:
                while queue and len(pending) < max_concurrent:
                    idx, (chal, acc) = queue.popleft()
                    pending.add(asyncio.create_task(_run_one(idx, chal, acc)))
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    idx, result = task.result()
                    results.append(result)
                    emit_progress(args, result, results)
                    if use_sprt:
                        decisive = [r for r in results if not r.void and r.winner in {"agent_a", "agent_b"}]
                        w = sum(1 for r in decisive if r.winner == "agent_a")
                        l = sum(1 for r in decisive if r.winner == "agent_b")
                        decision = sprt_check(w, l, args.sprt_h0, args.sprt_h1)
                        if decision != "continue":
                            print(f"SPRT STOP: {decision} after {len(decisive)} decisive games (w={w} l={l})", flush=True)
                            queue.clear()
                            break
    else:
        with tempfile.TemporaryDirectory(prefix="phase0-eval-") as temp_dir_name:
            log_dir = Path(temp_dir_name)
            for index, (challenger, acceptor) in enumerate(schedule, start=1):
                result = await run_scheduled_game(
                    args, server_configuration, index, challenger, acceptor, log_dir
                )
                results.append(result)
                emit_progress(args, result, results)
                if use_sprt:
                    decisive = [r for r in results if not r.void and r.winner in {"agent_a", "agent_b"}]
                    w = sum(1 for r in decisive if r.winner == "agent_a")
                    l = sum(1 for r in decisive if r.winner == "agent_b")
                    decision = sprt_check(w, l, args.sprt_h0, args.sprt_h1)
                    if decision != "continue":
                        print(f"SPRT STOP: {decision} after {len(decisive)} decisive games (w={w} l={l})", flush=True)
                        break

    completed_results = [result for result in results if not result.void]
    decisive_results = [
        result for result in completed_results if result.winner in {"agent_a", "agent_b"}
    ]
    void_games = len(results) - len(completed_results)
    agent_a_wins = sum(1 for result in decisive_results if result.winner == "agent_a")
    agent_a_losses = sum(1 for result in decisive_results if result.winner == "agent_b")
    ties_or_unknown = len(completed_results) - agent_a_wins - agent_a_losses
    decisive_games = agent_a_wins + agent_a_losses
    winrate = agent_a_wins / decisive_games if decisive_games else 0.0
    ci_low, ci_high = wilson_ci(agent_a_wins, decisive_games)
    agent_a_as_challenger_games = sum(
        1 for result in decisive_results if result.challenger == "agent_a"
    )
    agent_a_as_challenger_wins = sum(
        1
        for result in decisive_results
        if result.challenger == "agent_a" and result.winner == "agent_a"
    )
    agent_a_as_acceptor_games = sum(
        1 for result in decisive_results if result.acceptor == "agent_a"
    )
    agent_a_as_acceptor_wins = sum(
        1
        for result in decisive_results
        if result.acceptor == "agent_a" and result.winner == "agent_a"
    )
    summary = EvalSummary(
        mode="h2h",
        format=args.format,
        server=args.server,
        agent_a=args.agent_a,
        agent_b=args.agent_b,
        n_games=len(results),
        completed_games=len(completed_results),
        void_games=void_games,
        decisive_games=decisive_games,
        agent_a_wins=agent_a_wins,
        agent_a_losses=agent_a_losses,
        ties_or_unknown=ties_or_unknown,
        winrate=winrate,
        ci95_low=ci_low,
        ci95_high=ci_high,
        paired=args.paired,
        foul_play_search_time_ms=args.foul_play_search_time_ms,
        agent_a_as_challenger_wins=agent_a_as_challenger_wins,
        agent_a_as_challenger_games=agent_a_as_challenger_games,
        agent_a_as_acceptor_wins=agent_a_as_acceptor_wins,
        agent_a_as_acceptor_games=agent_a_as_acceptor_games,
        voids_with_agent_a_challenger=sum(
            1 for result in results if result.void and result.challenger == "agent_a"
        ),
        voids_with_agent_b_challenger=sum(
            1 for result in results if result.void and result.challenger == "agent_b"
        ),
        sprt_decision=sprt_check(agent_a_wins, agent_a_losses, args.sprt_h0, args.sprt_h1)
            if getattr(args, "sprt_h1", None) else "n/a",
        sprt_llr=round(sprt_llr(agent_a_wins, agent_a_losses, args.sprt_h0, args.sprt_h1), 4)
            if getattr(args, "sprt_h1", None) else 0.0,
        scorer_gate_passed=scorer_gate_check(agent_a_wins, agent_a_losses, void_games)[0]
            if getattr(args, "scorer_gate", False) else False,
        scorer_gate_message=scorer_gate_check(agent_a_wins, agent_a_losses, void_games)[1]
            if getattr(args, "scorer_gate", False) else "n/a",
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
        if args.log_dir:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            proc, log_path, log_file = await start_foul_play(
                args,
                args.agent,
                server_configuration,
                args.username,
                "search_ladder",
                None,
                log_dir,
            )
            output = await wait_for_foul_play(
                proc, log_path, log_file, args.game_timeout_seconds * args.n_games
            )
            result = {"agent": args.agent, "username": args.username, "output_tail": output[-4000:]}
            # extract W/L from output
            import re as _re
            wl_match = _re.findall(r"W:\s+(\d+)\s+L:\s+(\d+)", output)
            if wl_match:
                w, l = wl_match[-1]
                result["wins"] = int(w)
                result["losses"] = int(l)
                print(f"LADDER DONE: {args.username} W={w} L={l}", flush=True)
        else:
            with tempfile.TemporaryDirectory(prefix="phase0-ladder-") as temp_dir_name:
                proc, log_path, log_file = await start_foul_play(
                    args,
                    args.agent,
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


def _append_ladder_row(args: argparse.Namespace, result: dict) -> None:
    """Append a ladder run row to the experiment log."""
    path = Path(args.append_experiment_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fetch live ratings
    ratings = fetch_ladder_rating(args.username, args.format)
    row = {
        "run_id": args.run_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "phase": args.phase,
        "format": args.format,
        "change (ONE var)": args.change_name,
        "baseline": f"ladder_{args.agent}",
        "N_games": str(result.get("wins", 0) + result.get("losses", 0)),
        "winrate": f"{result.get('wins', 0) / max(1, result.get('wins', 0) + result.get('losses', 0)):.4f}",
        "CI95": "",
        "ladder_elo": str(ratings.get("elo", "")),
        "gxe": str(ratings.get("gxe", "")),
        "belief_brier": "",
        "decision(advance/iterate/rollback)": args.decision
        if hasattr(args, "decision") else "",
    }
    _write_csv_row(path, row)


def append_experiment_row(args: argparse.Namespace, summary: EvalSummary) -> None:
    path = Path(args.append_experiment_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    row = {
        "run_id": args.run_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "phase": args.phase,
        "format": summary.format,
        "change (ONE var)": args.change_name,
        "baseline": f"{summary.agent_a}_vs_{summary.agent_b}",
        "N_games": str(summary.n_games),
        "winrate": f"{summary.winrate:.4f}",
        "CI95": f"[{summary.ci95_low:.4f}, {summary.ci95_high:.4f}]",
        "ladder_elo": "",
        "gxe": "",
        "belief_brier": "",
        "decision(advance/iterate/rollback)": args.decision
        or ("iterate" if summary.void_games else "record"),
        "notes": (
            f"paired={summary.paired}; decisive={summary.decisive_games}; "
            f"completed={summary.completed_games}; voids={summary.void_games}; "
            f"ties_or_unknown={summary.ties_or_unknown}; "
            f"foul_play_search_time_ms={summary.foul_play_search_time_ms}; "
            f"agent_a_as_challenger={summary.agent_a_as_challenger_wins}/{summary.agent_a_as_challenger_games}; "
            f"agent_a_as_acceptor={summary.agent_a_as_acceptor_wins}/{summary.agent_a_as_acceptor_games}; "
            f"voids_agent_a_challenger={summary.voids_with_agent_a_challenger}; "
            f"voids_agent_b_challenger={summary.voids_with_agent_b_challenger}"
        ),
    }
    _write_csv_row(path, row)


def _write_csv_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
        if not path.exists() or path.stat().st_size == 0:
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
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--foul-play-python", default=str(ROOT_DIR / ".venv-foul-play" / "bin" / "python"))
    parser.add_argument("--agent-a-python", default=None,
                        help="Override Python binary for agent-a (for testing different poke-engine builds)")
    parser.add_argument("--agent-b-python", default=None,
                        help="Override Python binary for agent-b")
    parser.add_argument("--learned-value-model", default=None)
    parser.add_argument("--prior-server-url", default="http://127.0.0.1:8977")
    parser.add_argument(
        "--agent-a-prior-server-url",
        default=None,
        help="Per-side prior server URL for paired FP H2H tests.",
    )
    parser.add_argument(
        "--agent-b-prior-server-url",
        default=None,
        help="Per-side prior server URL for paired FP H2H tests.",
    )
    for slot in ("agent-a", "agent-b"):
        dest = slot.replace("-", "_")
        parser.add_argument(f"--{slot}-decision-log", default=None)
        parser.add_argument(f"--{slot}-replay-dir", default=None)
        parser.add_argument(f"--{slot}-require-priors", action="store_true")
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument(
        "--randbats-belief-pool",
        default=None,
        help="Path to a pre-sampled Showdown randbats team pool for foul_play_randbats_pool.",
    )
    parser.add_argument(
        "--randbats-conditional-script",
        default=str(ROOT_DIR / "src" / "scripts" / "sample_conditional_randbats.cjs"),
        help="Node script used by foul_play_randbats_conditional.",
    )
    parser.add_argument("--concurrent-games", type=int, default=1,
                        help="Number of games to run concurrently (for self-play data generation)")
    parser.add_argument("--randbats-conditional-max-teams", type=int, default=30000)
    parser.add_argument("--randbats-conditional-max-ms", type=int, default=250)
    parser.add_argument("--randbats-conditional-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--tauros-kind-model", default=str(ROOT_DIR / "src" / "nets" / "checkpoints" / "tauros_action_kind_n100.json"))
    parser.add_argument("--tauros-kind-threshold", type=float, default=0.70)
    parser.add_argument("--tauros-kind-min-policy-frac", type=float, default=0.10)
    parser.add_argument(
        "--tauros-kind-allowed-kinds",
        default="attack_or_other,boom,paralysis,recovery,sleep,switch",
        help="Comma-separated action kinds the Tauros gate may override toward.",
    )
    parser.add_argument("--value-shield-margin", type=float, default=0.15)
    parser.add_argument("--value-shield-min-support", type=float, default=0.10)
    parser.add_argument("--value-shield-close-policy-frac", type=float, default=0.75)
    parser.add_argument("--value-shield-log", default=None)
    parser.add_argument("--agent-a-model", default=None,
                        help="Per-slot model override for agent-a (foul_play_learned only).")
    parser.add_argument("--agent-b-model", default=None,
                        help="Per-slot model override for agent-b (foul_play_learned only).")
    parser.add_argument("--foul-play-search-time-ms", type=int, default=100)
    parser.add_argument("--agent-a-search-time-ms", type=int, default=None,
                        help="Override search budget (ms) for agent_a only; falls back to --foul-play-search-time-ms")
    parser.add_argument("--agent-b-search-time-ms", type=int, default=None,
                        help="Override search budget (ms) for agent_b only; falls back to --foul-play-search-time-ms")
    parser.add_argument("--foul-play-search-parallelism", type=int, default=1)
    parser.add_argument("--foul-play-search-threads", type=int, default=1)
    parser.add_argument("--foul-play-startup-delay-seconds", type=float, default=5.0)
    parser.add_argument("--poke-env-startup-delay-seconds", type=float, default=3.0)
    parser.add_argument("--foul-play-log-level", default="INFO")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--append-experiment-log", default=None)
    parser.add_argument("--phase", default="0")
    parser.add_argument("--sprt-h0", type=float, default=None,
                        help="SPRT null hypothesis winrate (e.g. 0.50). Activates sequential testing.")
    parser.add_argument("--sprt-h1", type=float, default=None,
                        help="SPRT alternative hypothesis winrate (e.g. 0.53). Activates sequential testing.")
    parser.add_argument("--scorer-gate", action="store_true",
                        help="Run §6.3 powered self-play scorer gate check on results.")
    parser.add_argument("--change-name", default="stock_foul_play_baseline")
    parser.add_argument("--decision", default=None)
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
        if args.append_experiment_log and "wins" in result:
            _append_ladder_row(args, result)
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
