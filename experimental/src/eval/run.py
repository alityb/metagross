from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlparse
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
    "direct_r1",
    "production_r1_search_first",
    "production_r1_independent_ensemble",
    "production_r1_shared_rm_plus",
    "production_r1_certified",
    "random",
    "max_damage",
    "foul_play",
    "foul_play_learned",
    "foul_play_learned_root_priors_opp",
    "foul_play_randbats_pool",
    "foul_play_randbats_conditional",
    "foul_play_randbats_conditional_root_priors_opp",
    "foul_play_action_belief_root_priors_opp",
    "foul_play_action_belief_moves_only_root_priors_opp",
    "foul_play_shared_root_action_belief_opp",
    "foul_play_selective_shared_root_opp",
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
    pair_id: Optional[str] = None
    pair_index: Optional[int] = None
    pair_leg: Optional[int] = None
    battle_seed: Optional[str] = None
    team_1_sha256: Optional[str] = None
    team_2_sha256: Optional[str] = None
    agent_a_team_sha256: Optional[str] = None
    agent_b_team_sha256: Optional[str] = None


@dataclass(frozen=True)
class PairPlan:
    pair_id: str
    pair_index: int
    battle_seed: str
    team_1_seed: str
    team_2_seed: str
    team_1_packed: str
    team_2_packed: str
    team_1_sha256: str
    team_2_sha256: str


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
    mirrored_pairs: bool
    completed_pairs: int
    void_pairs: int
    pair_sweeps_a: int
    pair_splits: int
    pair_sweeps_b: int
    pair_score_mean: float
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


DIRECT_R1_RUN_NAME = "randbats_exit_r1"
DIRECT_R1_CHECKPOINT = 5
DIRECT_R1_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"


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
        return False, f"CI [{ci_low:.4f}, {ci_high:.4f}] not contained in [0.40, 0.60]"
    if voids > 0:
        return False, f"{voids} void games (check for unexplained ties/unknowns)"
    return True, f"PASS: wr={wr:.4f} CI=[{ci_low:.4f}, {ci_high:.4f}] n={n} voids={voids}"


def normalize_user_id(username: str) -> str:
    return re.sub(r"[^a-z0-9]", "", username.lower())


def make_username(role: str, game_index: int, username_prefix: str = "p0") -> str:
    suffix = secrets.token_hex(2)
    return f"{username_prefix}{role}{game_index:03d}{suffix}"[:18]


def is_foul_play(agent: str) -> bool:
    return agent in {
        "production_r1_search_first",
        "production_r1_independent_ensemble",
        "production_r1_shared_rm_plus",
        "production_r1_certified",
        "foul_play",
        "foul_play_learned",
        "foul_play_learned_root_priors_opp",
        "foul_play_randbats_pool",
        "foul_play_randbats_conditional",
        "foul_play_randbats_conditional_root_priors_opp",
        "foul_play_action_belief_root_priors_opp",
        "foul_play_action_belief_moves_only_root_priors_opp",
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
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


def is_external_agent(agent: str) -> bool:
    return is_foul_play(agent) or agent == "direct_r1"


def verify_direct_r1_checkpoint(args: argparse.Namespace) -> Path:
    path = (
        Path(args.direct_r1_checkpoint_root)
        / DIRECT_R1_RUN_NAME
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{DIRECT_R1_CHECKPOINT}.pt"
    ).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen direct-r1 checkpoint not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not secrets.compare_digest(actual, DIRECT_R1_SHA256):
        raise RuntimeError(
            f"frozen direct-r1 checkpoint SHA-256 mismatch: expected {DIRECT_R1_SHA256}, got {actual}"
        )
    return path


def is_learned_foul_play(agent: str) -> bool:
    return agent in {"foul_play_learned", "foul_play_learned_root_priors_opp"}


def is_randbats_pool_foul_play(agent: str) -> bool:
    return agent in {
        "foul_play_randbats_pool",
        "foul_play_randbats_conditional_root_priors_opp",
        "foul_play_action_belief_root_priors_opp",
        "foul_play_action_belief_moves_only_root_priors_opp",
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
    }


def is_randbats_conditional_foul_play(agent: str) -> bool:
    return agent in {
        "foul_play_randbats_conditional",
        "foul_play_randbats_conditional_root_priors_opp",
        "foul_play_action_belief_root_priors_opp",
        "foul_play_action_belief_moves_only_root_priors_opp",
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
    }


def is_action_belief_foul_play(agent: str) -> bool:
    return agent in {
        "foul_play_action_belief_root_priors_opp",
        "foul_play_action_belief_moves_only_root_priors_opp",
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
    }


def is_tauros_kind_foul_play(agent: str) -> bool:
    return agent in {"foul_play_tauros_kind", "foul_play_tauros_action"}


def is_value_shield_foul_play(agent: str) -> bool:
    return agent == "foul_play_value_shield"


def is_belief_threat_foul_play(agent: str) -> bool:
    return agent == "foul_play_belief_threat"


def is_opp_priors_foul_play(agent: str) -> bool:
    return agent == "foul_play_opp_priors"


def uses_prior_server(agent: str) -> bool:
    return agent in {
        "production_r1_search_first",
        "production_r1_independent_ensemble",
        "production_r1_shared_rm_plus",
        "production_r1_certified",
        "foul_play_root_priors",
        "foul_play_root_priors_opp",
        "foul_play_learned_root_priors_opp",
        "foul_play_randbats_conditional_root_priors_opp",
        "foul_play_action_belief_root_priors_opp",
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
        "foul_play_opp_priors",
    }


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
    selected_agent = agent_for_slot(args, slot) if slot else args.agent
    production_controller = selected_agent in {
        "production_r1_search_first",
        "production_r1_independent_ensemble",
        "production_r1_shared_rm_plus",
        "production_r1_certified",
    }
    runner = ROOT_DIR / "src" / "scripts" / "run_foul_play.py"
    cmd = [str(python_bin)]
    if production_controller:
        cmd.extend(["-u", "-m", "srcs.metagross.run_foul_play"])
    else:
        cmd.append(str(runner))
    cmd.extend([
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
    ])
    if args.password:
        cmd.extend(["--ps-password", args.password])
    if bot_mode == "challenge_user":
        if not user_to_challenge:
            raise ValueError("user_to_challenge is required for challenge_user mode")
        cmd.extend(["--user-to-challenge", user_to_challenge])
    return cmd


def direct_r1_command(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    username: str,
    opponent_username: str,
    role: str,
    results_dir: Path,
) -> list[str]:
    return [
        str(Path(args.metamon_python)),
        str(ROOT_DIR / "src" / "scripts" / "run_direct_r1_challenge.py"),
        "--username",
        username,
        "--opponent-username",
        opponent_username,
        "--role",
        role,
        "--battle-format",
        args.format,
        "--websocket-uri",
        server_configuration.websocket_url,
        "--checkpoint-root",
        str(Path(args.direct_r1_checkpoint_root).expanduser().resolve()),
        "--save-results-to",
        str(results_dir.resolve()),
    ]


def model_for_agent(args: argparse.Namespace, agent: str) -> Optional[str]:
    """Return the model path for the given agent, respecting per-slot overrides."""
    if is_learned_foul_play(agent):
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
    for name in (
        "METAGROSS_CONTROLLER_MODE",
        "METAGROSS_PROTOCOL_DUMP",
        "METAGROSS_SEARCH_DUMP",
        "METAGROSS_HOLDOUT_LEDGER",
        "METAGROSS_PRIOR_NAMESPACE",
        "METAGROSS_REQUIRE_REMOTE_MCTS",
        "METAGROSS_VERIFIER_SHADOW",
        "METAGROSS_ALLOW_INSECURE_LOOPBACK",
        "METAGROSS_ROOT_SEARCH_MODE",
        "METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT",
        "METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE",
        "METAGROSS_INDEPENDENT_ENSEMBLE_REPEATS",
        "METAGROSS_SHARED_ROOT_ITERATIONS",
        "METAGROSS_SHARED_ROOT_CONTINUATION_ITERATIONS",
        "METAGROSS_SHARED_ROOT_PRIOR_STRENGTH",
    ):
        env.pop(name, None)
    if agent in {
        "production_r1_search_first",
        "production_r1_independent_ensemble",
        "production_r1_shared_rm_plus",
        "production_r1_certified",
    }:
        env["FOUL_PLAY_DIR"] = str(ROOT_DIR.parent / "srcs" / "vendor" / "foul-play")
        env["METAGROSS_CONTROLLER_MODE"] = (
            "search_first"
            if agent
            in {
                "production_r1_search_first",
                "production_r1_independent_ensemble",
                "production_r1_shared_rm_plus",
            }
            else "certified"
        )
        env["METAGROSS_ROOT_SEARCH_MODE"] = (
            "shared_rm_plus"
            if agent == "production_r1_shared_rm_plus"
            else (
                "independent_ensemble"
                if agent == "production_r1_independent_ensemble"
                else "independent_mcts"
            )
        )
        if agent == "production_r1_independent_ensemble":
            env["METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE"] = "1"
            env["METAGROSS_INDEPENDENT_ENSEMBLE_REPEATS"] = "3"
        if agent == "production_r1_shared_rm_plus":
            env["METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT"] = "1"
            env["METAGROSS_SHARED_ROOT_ITERATIONS"] = str(
                args.shared_root_iterations
            )
            env["METAGROSS_SHARED_ROOT_CONTINUATION_ITERATIONS"] = str(
                args.shared_root_continuation_iterations
            )
            env["METAGROSS_SHARED_ROOT_PRIOR_STRENGTH"] = str(
                args.shared_root_prior_strength
            )
        env["METAGROSS_VERIFIER_SHADOW"] = "0"
        env["METAGROSS_WEBSOCKET_KEEPALIVE"] = "1"
        env["METAGROSS_WEBSOCKET_PING_INTERVAL_SECONDS"] = "20"
        env["METAGROSS_WEBSOCKET_PING_TIMEOUT_SECONDS"] = "60"
        env.setdefault("METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS", "120")
        env["METAGROSS_WEBSOCKET_MAX_RECONNECTS"] = "1"
        env["METAGROSS_REMOTE_MCTS_TIMEOUT_SECONDS"] = "10"
        websocket_host = urlparse(args.websocket_uri or LOCAL_WEBSOCKET_URI).hostname
        if args.server != "local" or websocket_host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("production controller H2H requires an explicit loopback server")
        env["METAGROSS_ALLOW_INSECURE_LOOPBACK"] = "1"
        env["METAGROSS_RUN_SEED"] = args.production_run_seed
        env["METAGROSS_RNG_SCHEME"] = "hmac-sha256-length-prefixed-v1"
        if getattr(args, "production_remote_mcts", False):
            env.update(
                {
                    "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
                    "METAGROSS_REMOTE_MCTS_TRANSPORT": "modal",
                    "METAGROSS_REMOTE_MCTS_APP": args.production_remote_mcts_app,
                    "METAGROSS_REMOTE_MCTS_FUNCTION": args.production_remote_mcts_function,
                    "METAGROSS_REMOTE_ENGINE_SHA256": args.production_remote_engine_sha256,
                }
            )
    model = model_override if model_override is not None else model_for_agent(args, agent)
    if is_learned_foul_play(agent):
        if not model:
            raise ValueError("learned Foul Play requires --learned-value-model or a per-slot model override")
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
    if is_action_belief_foul_play(agent):
        env["METAGROSS_ACTION_CONDITIONED_BELIEF"] = "1"
        env.setdefault("METAGROSS_ACTION_EVIDENCE_TEMPERATURE", "0.5")
    else:
        env.pop("METAGROSS_ACTION_CONDITIONED_BELIEF", None)
        env.pop("METAGROSS_ACTION_EVIDENCE_TEMPERATURE", None)
    if agent == "foul_play_action_belief_moves_only_root_priors_opp":
        env["METAGROSS_ACTION_EVIDENCE_MOVES_ONLY"] = "1"
    else:
        env.pop("METAGROSS_ACTION_EVIDENCE_MOVES_ONLY", None)
    if agent in {
        "foul_play_shared_root_action_belief_opp",
        "foul_play_selective_shared_root_opp",
    }:
        env["METAGROSS_SHARED_ROOT_SEARCH"] = "1"
    else:
        env.pop("METAGROSS_SHARED_ROOT_SEARCH", None)
    if agent == "foul_play_selective_shared_root_opp":
        selective_mode = os.environ.get("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", "audit")
        if selective_mode not in {"audit", "override"}:
            raise ValueError("METAGROSS_SELECTIVE_SHARED_ROOT_MODE must be audit or override")
        env["METAGROSS_SELECTIVE_SHARED_ROOT_MODE"] = selective_mode
        env["METAGROSS_REQUIRE_SELECTIVE_PAIRED_EVALUATION"] = "1"
    else:
        env.pop("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", None)
        env.pop("METAGROSS_REQUIRE_SELECTIVE_PAIRED_EVALUATION", None)
    prior_url = args.prior_server_url
    if slot == "agent_a" and getattr(args, "agent_a_prior_server_url", None):
        prior_url = args.agent_a_prior_server_url
    elif slot == "agent_b" and getattr(args, "agent_b_prior_server_url", None):
        prior_url = args.agent_b_prior_server_url
    # Per-slot exploration constant: sharper (distilled) priors can need a
    # different c_puct than the baseline they are gated against.
    cpuct = args.cpuct
    if slot == "agent_a" and getattr(args, "agent_a_cpuct", None) is not None:
        cpuct = args.agent_a_cpuct
    elif slot == "agent_b" and getattr(args, "agent_b_cpuct", None) is not None:
        cpuct = args.agent_b_cpuct
    if uses_prior_server(agent) and not is_opp_priors_foul_play(agent):
        env["METAGROSS_PRIOR_SERVER"] = prior_url
        env["METAGROSS_CPUCT"] = str(cpuct)
        if agent in (
            "foul_play_learned_root_priors_opp",
        ):
            env["METAGROSS_OPP_PRIORS_ONLY"] = "1"
    elif is_opp_priors_foul_play(agent):
        env["METAGROSS_PRIOR_SERVER"] = prior_url
        env["METAGROSS_CPUCT"] = str(cpuct)
        env["METAGROSS_OPP_PRIORS_ONLY"] = "1"
    else:
        env.pop("METAGROSS_PRIOR_SERVER", None)
        env.pop("METAGROSS_CPUCT", None)
        env.pop("METAGROSS_OPP_PRIORS_ONLY", None)

    if slot in ("agent_a", "agent_b"):
        # Shard-local capture paths must win over any parent process environment.
        for name in (
            "METAGROSS_REPLAY_DIR",
            "METAGROSS_DECISION_LOG",
            "METAGROSS_BELIEF_LOG",
            "METAGROSS_TEACHER_ROOT_BUNDLE",
            "METAGROSS_TEACHER_ITERATIONS",
            "METAGROSS_TEACHER_REPEATS",
            "METAGROSS_TEACHER_DEEP_MULTIPLIER",
            "METAGROSS_TEACHER_SEED",
            "METAGROSS_TEACHER_INCLUDE_STATE",
            "METAGROSS_TEACHER_MANIFEST_SHA256",
            "METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES",
            "METAGROSS_TEACHER_DETERMINIZATION_SEED",
        ):
            env.pop(name, None)
        prefix = slot.replace("agent_", "agent_")
        env["METAGROSS_PRIOR_NAMESPACE"] = slot
        decision_log = getattr(args, f"{prefix}_decision_log", None)
        replay_dir = getattr(args, f"{prefix}_replay_dir", None)
        require_priors = getattr(args, f"{prefix}_require_priors", False)
        teacher_root_bundle = getattr(args, f"{prefix}_teacher_root_bundle", None)
        if decision_log:
            env["METAGROSS_DECISION_LOG"] = str(Path(decision_log).resolve())
        if replay_dir:
            env["METAGROSS_REPLAY_DIR"] = str(Path(replay_dir).resolve())
        if require_priors:
            env["METAGROSS_REQUIRE_PRIORS"] = "1"
        else:
            env.pop("METAGROSS_REQUIRE_PRIORS", None)
        if teacher_root_bundle:
            env["METAGROSS_TEACHER_ROOT_BUNDLE"] = str(
                Path(teacher_root_bundle).resolve()
            )
            env["METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES"] = str(
                args.teacher_determinization_schedules
            )
            env["METAGROSS_TEACHER_DETERMINIZATION_SEED"] = str(
                args.teacher_determinization_seed
            )
            env["METAGROSS_TEACHER_MANIFEST_SHA256"] = (
                args.teacher_manifest_sha256.lower()
            )
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
    env = foul_play_env(args, agent, model_override, slot=slot)
    production_controller = agent in {
        "production_r1_search_first",
        "production_r1_independent_ensemble",
        "production_r1_shared_rm_plus",
        "production_r1_certified",
    }
    if production_controller:
        env["METAGROSS_PROTOCOL_DUMP"] = str(
            (log_dir / f"{username}.protocol.jsonl").resolve()
        )
        env["METAGROSS_SEARCH_DUMP"] = str(
            (log_dir / f"{username}.search.jsonl").resolve()
        )
        env["METAGROSS_HOLDOUT_LEDGER"] = str(
            (log_dir / f"{username}.holdout.jsonl").resolve()
        )
    proc = await asyncio.create_subprocess_exec(
        *foul_play_command(args, server_configuration, username, bot_mode, user_to_challenge, slot=slot),
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ROOT_DIR.parent if production_controller else ROOT_DIR,
        env=env,
    )
    return proc, log_path, log_file


async def start_external_agent(
    args: argparse.Namespace,
    agent: str,
    server_configuration: ServerConfiguration,
    username: str,
    opponent_username: str,
    role: str,
    log_dir: Path,
    slot: str,
) -> tuple[asyncio.subprocess.Process, Path, object]:
    if is_foul_play(agent):
        bot_mode = "challenge_user" if role == "challenger" else "accept_challenge"
        return await start_foul_play(
            args,
            agent,
            server_configuration,
            username,
            bot_mode,
            opponent_username if role == "challenger" else None,
            log_dir,
            model_override=model_for_slot(args, slot),
            slot=slot,
        )
    if agent != "direct_r1":
        raise ValueError(f"unsupported external agent: {agent}")
    log_path = log_dir / f"{username}.log"
    log_file = log_path.open("w", encoding="utf-8")
    results_dir = log_dir / f"{username}-results"
    direct_environment = os.environ.copy()
    direct_environment["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        *direct_r1_command(
            args, server_configuration, username, opponent_username, role, results_dir
        ),
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        cwd=ROOT_DIR.parent,
        env=direct_environment,
    )
    return proc, log_path, log_file


async def wait_for_direct_r1_acceptor_ready(
    proc: asyncio.subprocess.Process,
    log_path: Path,
    timeout_seconds: float,
) -> None:
    """Wait for the cold-loaded direct policy to begin accepting challenges."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    marker = "Made Challenge Env (acceptor):"
    while asyncio.get_running_loop().time() < deadline:
        output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if marker in output:
            return
        if proc.returncode is not None:
            raise FoulPlayError(
                f"Direct R1 exited with code {proc.returncode} before readiness; "
                f"log={log_path}\n{output[-4000:]}"
            )
        await asyncio.sleep(0.25)
    raise FoulPlayError(
        f"Direct R1 did not become challenge-ready within {timeout_seconds:g}s; "
        f"log={log_path}"
    )


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
    challenger_username = make_username("c", game_index, args.username_prefix)
    acceptor_username = make_username("a", game_index, args.username_prefix)
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
    fp_username = make_username("f", game_index, args.username_prefix)
    challenger_username = make_username("c", game_index, args.username_prefix)
    proc, log_path, log_file = await start_foul_play(
        args,
        acceptor_agent,
        server_configuration,
        fp_username,
        "accept_challenge",
        None,
        log_dir,
        model_override=model_for_slot(args, acceptor_slot),
        slot=acceptor_slot,
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
    fp_username = make_username("f", game_index, args.username_prefix)
    acceptor_username = make_username("a", game_index, args.username_prefix)
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
        slot=challenger_slot,
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


async def play_external_vs_external(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
    pair_plan: PairPlan | None = None,
    pair_leg: int | None = None,
) -> GameResult:
    challenger_agent = agent_for_slot(args, challenger_slot)
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    challenger_username = make_username("x", game_index, args.username_prefix)
    acceptor_username = make_username("y", game_index, args.username_prefix)
    if pair_plan is not None:
        if pair_leg not in {1, 2}:
            raise ValueError("mirrored pair leg must be 1 or 2")
        write_pair_registrations(
            args,
            pair_plan,
            pair_leg,
            challenger_username,
            acceptor_username,
        )
    acceptor_proc, acceptor_log_path, acceptor_log_file = await start_external_agent(
        args,
        acceptor_agent,
        server_configuration,
        acceptor_username,
        challenger_username,
        "acceptor",
        log_dir,
        acceptor_slot,
    )
    if acceptor_agent == "direct_r1":
        await wait_for_direct_r1_acceptor_ready(
            acceptor_proc,
            acceptor_log_path,
            min(float(args.game_timeout_seconds), 600.0),
        )
    else:
        await asyncio.sleep(args.foul_play_startup_delay_seconds)
    challenger_proc, challenger_log_path, challenger_log_file = await start_external_agent(
        args,
        challenger_agent,
        server_configuration,
        challenger_username,
        acceptor_username,
        "challenger",
        log_dir,
        challenger_slot,
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

    result = GameResult(
        game_index,
        args.agent_a,
        args.agent_b,
        challenger_slot,
        acceptor_slot,
        winner,
        fp_winner,
        battle_tag,
    )
    if pair_plan is not None:
        apply_pair_metadata(result, pair_plan, pair_leg, challenger_slot)
    return result


async def play_one_game(
    args: argparse.Namespace,
    server_configuration: ServerConfiguration,
    game_index: int,
    challenger_slot: str,
    acceptor_slot: str,
    log_dir: Path,
    pair_plan: PairPlan | None = None,
    pair_leg: int | None = None,
) -> GameResult:
    challenger_agent = agent_for_slot(args, challenger_slot)
    acceptor_agent = agent_for_slot(args, acceptor_slot)
    challenger_is_fp = is_foul_play(challenger_agent)
    acceptor_is_fp = is_foul_play(acceptor_agent)
    if is_external_agent(challenger_agent) and is_external_agent(acceptor_agent):
        return await play_external_vs_external(
            args,
            server_configuration,
            game_index,
            challenger_slot,
            acceptor_slot,
            log_dir,
            pair_plan,
            pair_leg,
        )
    if pair_plan is not None:
        raise ValueError("mirrored pairs require registration-aware external agents")
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


def _derived_showdown_seed(master_seed: int, pair_index: int, label: str) -> str:
    digest = hashlib.sha256(f"{master_seed}:{pair_index}:{label}".encode("ascii")).digest()
    return ",".join(str(int.from_bytes(digest[offset:offset + 2], "big")) for offset in range(0, 8, 2))


def _showdown_commit(showdown_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=showdown_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("could not identify the pinned Showdown commit")
    return commit


def pair_manifest_path(args: argparse.Namespace) -> Path:
    return Path(str(args.json_out) + ".pairs.json")


def _generate_pair_plan(args: argparse.Namespace, pair_index: int) -> PairPlan:
    team_1_seed = _derived_showdown_seed(args.mirror_seed, pair_index, "team-1")
    team_2_seed = _derived_showdown_seed(args.mirror_seed, pair_index, "team-2")
    battle_seed = _derived_showdown_seed(args.mirror_seed, pair_index, "battle")
    helper = Path(args.mirrored_team_generator)
    completed = subprocess.run(
        ["node", str(helper), args.format, team_1_seed, team_2_seed],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    generated = json.loads(completed.stdout)
    pair_id = hashlib.sha256(
        f"{args.run_id}:{args.mirror_seed}:{pair_index}".encode("utf-8")
    ).hexdigest()[:24]
    return PairPlan(
        pair_id=pair_id,
        pair_index=pair_index,
        battle_seed=battle_seed,
        team_1_seed=team_1_seed,
        team_2_seed=team_2_seed,
        team_1_packed=generated["team_1_packed"],
        team_2_packed=generated["team_2_packed"],
        team_1_sha256=generated["team_1_sha256"],
        team_2_sha256=generated["team_2_sha256"],
    )


def load_or_create_pair_plans(args: argparse.Namespace) -> list[PairPlan]:
    path = pair_manifest_path(args)
    showdown_dir = Path(args.showdown_dir).resolve()
    expected_commit = _showdown_commit(showdown_dir)
    if path.is_file():
        expected_manifest_sha256 = getattr(args, "pair_manifest_sha256", None) or os.environ.get(
            "METAGROSS_PAIR_MANIFEST_SHA256"
        )
        if (
            expected_manifest_sha256
            and hashlib.sha256(path.read_bytes()).hexdigest() != expected_manifest_sha256
        ):
            raise ValueError("mirrored pair manifest differs from the frozen SHA-256")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported mirrored-pair manifest schema")
        if payload.get("config_sha256") != resume_config_sha256(args):
            raise ValueError("mirrored-pair manifest configuration does not match this run")
        if payload.get("showdown_commit") != expected_commit:
            raise ValueError("mirrored-pair manifest Showdown commit does not match")
        plans = [PairPlan(**record) for record in payload.get("pairs", [])]
    else:
        plans = [
            _generate_pair_plan(args, pair_index)
            for pair_index in range(1, args.n_games // 2 + 1)
        ]
        write_json(
            str(path),
            {
                "schema_version": 1,
                "config_sha256": resume_config_sha256(args),
                "showdown_commit": expected_commit,
                "pairs": [asdict(plan) for plan in plans],
            },
        )
    if len(plans) != args.n_games // 2 or [plan.pair_index for plan in plans] != list(
        range(1, args.n_games // 2 + 1)
    ):
        raise ValueError("mirrored-pair manifest has an invalid pair schedule")
    return plans


def write_pair_registrations(
    args: argparse.Namespace,
    plan: PairPlan,
    leg: int,
    challenger_username: str,
    acceptor_username: str,
) -> None:
    directory = Path(args.pair_registration_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    common = {
        "schema_version": 1,
        "pair_id": plan.pair_id,
        "leg": leg,
        "format": args.format,
        "battle_seed": plan.battle_seed,
        "team_1_sha256": plan.team_1_sha256,
        "team_2_sha256": plan.team_2_sha256,
    }
    assignments = (
        (challenger_username, plan.team_1_packed, plan.team_1_sha256),
        (acceptor_username, plan.team_2_packed, plan.team_2_sha256),
    )
    for username, packed_team, assigned_hash in assignments:
        write_json(
            str(directory / f"{normalize_user_id(username)}.json"),
            {
                **common,
                "assigned_team_sha256": assigned_hash,
                "packed_team": packed_team,
            },
        )


def apply_pair_metadata(
    result: GameResult,
    plan: PairPlan,
    leg: int | None,
    challenger_slot: str,
) -> None:
    result.pair_id = plan.pair_id
    result.pair_index = plan.pair_index
    result.pair_leg = leg
    result.battle_seed = plan.battle_seed
    result.team_1_sha256 = plan.team_1_sha256
    result.team_2_sha256 = plan.team_2_sha256
    if challenger_slot == "agent_a":
        result.agent_a_team_sha256 = plan.team_1_sha256
        result.agent_b_team_sha256 = plan.team_2_sha256
    else:
        result.agent_a_team_sha256 = plan.team_2_sha256
        result.agent_b_team_sha256 = plan.team_1_sha256


def resume_config_sha256(args: argparse.Namespace) -> str:
    fields = (
        "format",
        "server",
        "n_games",
        "paired",
        "mirrored_pairs",
        "mirror_seed",
        "showdown_dir",
        "mirrored_team_generator",
        "pair_registration_dir",
        "agent_a",
        "agent_b",
        "direct_r1_checkpoint_root",
        "metamon_python",
        "foul_play_search_time_ms",
        "agent_a_search_time_ms",
        "agent_b_search_time_ms",
        "foul_play_search_parallelism",
        "foul_play_search_threads",
        "cpuct",
        "agent_a_cpuct",
        "agent_b_cpuct",
        "prior_server_url",
        "agent_a_prior_server_url",
        "agent_b_prior_server_url",
        "agent_a_require_priors",
        "agent_b_require_priors",
        "sprt_h0",
        "sprt_h1",
        "run_id",
        "websocket_uri",
        "production_remote_mcts",
        "production_remote_mcts_app",
        "production_remote_mcts_function",
        "production_remote_engine_sha256",
        "production_run_seed",
        "strict_isolated_priors",
        "fail_fast",
        "game_timeout_seconds",
        "operational_gate_after_pairs",
        "operational_gate_request",
        "operational_gate_approval",
        "operational_gate_review",
        "operational_gate_prior_decisions",
        "operational_gate_showdown_launch",
        "operational_gate_token",
        "operational_gate_timeout_seconds",
        "foul_play_python",
        "log_dir",
        "json_out",
        "foul_play_startup_delay_seconds",
        "poke_env_startup_delay_seconds",
    )
    payload = {name: getattr(args, name, None) for name in fields}
    payload["selective_environment"] = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("METAGROSS_SELECTIVE_SHARED_ROOT_")
        or name.startswith("METAGROSS_SHARED_ROOT_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_identity(args: argparse.Namespace) -> dict[str, object]:
    return {
        "arguments": list(getattr(args, "_raw_arguments", ())),
        "python_executable": str(Path(sys.executable).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": resume_config_sha256(args),
        "environment": dict(sorted(os.environ.items())),
    }


def progress_snapshot_path(args: argparse.Namespace) -> Path:
    return Path(str(args.json_out) + ".progress.json")


def load_resume_results(
    args: argparse.Namespace,
    schedule: list[tuple[str, str]],
    pair_plans: list[PairPlan] | None = None,
) -> list[GameResult]:
    if not getattr(args, "resume", False):
        return []
    path = progress_snapshot_path(args)
    if not path.is_file():
        raise ValueError(f"resume snapshot does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = 2 if getattr(args, "mirrored_pairs", False) else 1
    if payload.get("schema_version") != expected_schema:
        raise ValueError("unsupported resume snapshot schema")
    if payload.get("config_sha256") != resume_config_sha256(args):
        raise ValueError("resume snapshot configuration does not match this run")

    results = [GameResult(**record) for record in payload.get("games", [])]
    seen: set[int] = set()
    for result in results:
        if result.game_index in seen or not 1 <= result.game_index <= len(schedule):
            raise ValueError("resume snapshot contains an invalid or duplicate game index")
        seen.add(result.game_index)
        if (result.challenger, result.acceptor) != schedule[result.game_index - 1]:
            raise ValueError("resume snapshot does not match the paired side schedule")
        if (result.agent_a, result.agent_b) != (args.agent_a, args.agent_b):
            raise ValueError("resume snapshot agents do not match this run")
    if getattr(args, "mirrored_pairs", False):
        if pair_plans is None or len(results) % 2:
            raise ValueError("mirrored resume snapshot must contain complete pairs")
        by_pair: dict[int, list[GameResult]] = {}
        for result in results:
            if result.pair_index is None:
                raise ValueError("mirrored resume result lacks pair metadata")
            by_pair.setdefault(result.pair_index, []).append(result)
        if sorted(by_pair) != list(range(1, len(by_pair) + 1)):
            raise ValueError("mirrored resume snapshot contains non-prefix pairs")
        for pair_index, pair_results in by_pair.items():
            plan = pair_plans[pair_index - 1]
            if len(pair_results) != 2 or {result.pair_leg for result in pair_results} != {1, 2}:
                raise ValueError("mirrored resume snapshot contains an incomplete pair")
            for result in pair_results:
                expected_a = (
                    plan.team_1_sha256 if result.challenger == "agent_a" else plan.team_2_sha256
                )
                if (
                    result.pair_id != plan.pair_id
                    or result.battle_seed != plan.battle_seed
                    or result.team_1_sha256 != plan.team_1_sha256
                    or result.team_2_sha256 != plan.team_2_sha256
                    or result.agent_a_team_sha256 != expected_a
                ):
                    raise ValueError("mirrored resume snapshot pair metadata does not match manifest")
    return sorted(results, key=lambda result: result.game_index)


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
    pair_plan: PairPlan | None = None,
    pair_leg: int | None = None,
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
            args,
            server_configuration,
            index,
            challenger,
            acceptor,
            log_dir,
            pair_plan,
            pair_leg,
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
        if pair_plan is not None:
            apply_pair_metadata(result, pair_plan, pair_leg, challenger)
        print(
            f"game={index} challenger={challenger} acceptor={acceptor} void=true error={result.error}",
            flush=True,
        )
        return result

    print(
        f"game={index} challenger={challenger} acceptor={acceptor} winner={result.winner}",
        flush=True,
    )
    if args.fail_fast and (
        result.void
        or result.error is not None
        or result.winner not in {"agent_a", "agent_b"}
        or not result.battle_tag
    ):
        raise FoulPlayError(f"game {index} did not produce a decisive identified result")
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
        "challenger": result.challenger,
        "acceptor": result.acceptor,
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
        write_json(
            str(progress_snapshot_path(args)),
            {
                "schema_version": 1,
                "config_sha256": resume_config_sha256(args),
                "games": [
                    asdict(record)
                    for record in sorted(results, key=lambda item: item.game_index)
                ],
            },
        )


def emit_pair_progress(
    args: argparse.Namespace,
    pair_results: list[GameResult],
    results: list[GameResult],
) -> None:
    if len(pair_results) != 2 or pair_results[0].pair_id != pair_results[1].pair_id:
        raise ValueError("pair progress requires exactly two matching legs")
    decisive = [r for r in results if not r.void and r.winner in {"agent_a", "agent_b"}]
    wins = sum(r.winner == "agent_a" for r in decisive)
    losses = sum(r.winner == "agent_b" for r in decisive)
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pair_id": pair_results[0].pair_id,
        "pair_index": pair_results[0].pair_index,
        "games": [asdict(result) for result in pair_results],
        "running_agent_a_wins": wins,
        "running_decisive": len(decisive),
        "running_winrate": round(wins / len(decisive), 4) if decisive else None,
    }
    if getattr(args, "sprt_h1", None):
        line["sprt_llr"] = round(sprt_llr(wins, losses, args.sprt_h0, args.sprt_h1), 4)
        line["sprt_decision"] = sprt_check(wins, losses, args.sprt_h0, args.sprt_h1)
    print(f"PAIR_PROGRESS {json.dumps(line, sort_keys=True)}", flush=True)
    progress_path = Path(str(args.json_out) + ".progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    write_json(
        str(progress_snapshot_path(args)),
        {
            "schema_version": 2,
            "config_sha256": resume_config_sha256(args),
            "games": [asdict(record) for record in sorted(results, key=lambda item: item.game_index)],
        },
    )


async def await_operational_gate(
    args: argparse.Namespace,
    pair_results: list[GameResult],
) -> None:
    request_path = Path(args.operational_gate_request)
    approval_path = Path(args.operational_gate_approval)
    review_path = Path(args.operational_gate_review)
    registration_entries = sorted(
        path.name for path in Path(args.pair_registration_dir).iterdir()
    )
    log_dir = Path(args.log_dir)
    protocol_logs = sorted(log_dir.glob("*.protocol.jsonl"))
    search_logs = sorted(log_dir.glob("*.search.jsonl"))
    prior_dumps = [Path(path) for path in args.operational_gate_prior_decisions]
    expected_tags = {result.battle_tag for result in pair_results}

    def jsonl(path: Path) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError(f"invalid operational JSONL evidence: {path}")
        return rows

    protocol_rows = [jsonl(path) for path in protocol_logs]
    search_rows = [jsonl(path) for path in search_logs]
    prior_rows = [jsonl(path) for path in prior_dumps]
    protocol_tags = [
        {tag for tag in expected_tags if any(tag in str(row.get("message", "")) for row in rows)}
        for rows in protocol_rows
    ]
    search_tags = [
        {(row.get("context") or {}).get("tag") for row in rows}
        for rows in search_rows
    ]
    prior_tags = [{row.get("tag") for row in rows} for rows in prior_rows]
    prior_health = [
        json.loads(urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5).read())
        for url in (args.agent_a_prior_server_url, args.agent_b_prior_server_url)
    ]
    showdown_launch = json.loads(Path(args.operational_gate_showdown_launch).read_text(encoding="utf-8"))
    try:
        os.kill(int(showdown_launch["pid"]), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError) as exc:
        raise RuntimeError("frozen Showdown process is not alive at the operational gate") from exc
    if (
        len(expected_tags) != 2
        or any(result.winner not in {"agent_a", "agent_b"} for result in pair_results)
        or registration_entries
        or len(protocol_logs) != 4
        or len(search_logs) != 4
        or any(path.stat().st_size == 0 for path in protocol_logs + search_logs)
        or len(prior_dumps) != 2
        or any(not path.is_file() or path.stat().st_size == 0 for path in prior_dumps)
        or any(len(tags) != 1 for tags in protocol_tags + search_tags)
        or set().union(*protocol_tags) != expected_tags
        or set().union(*search_tags) != expected_tags
        or any(tags != expected_tags for tags in prior_tags)
        or any(row.get("schema") != 4 for rows in prior_rows for row in rows)
        or any(
            line.startswith("|error|")
            for rows in protocol_rows
            for row in rows
            if row.get("direction") == "received"
            for line in str(row.get("message", "")).splitlines()
        )
        or any(health.get("ok") is not True for health in prior_health)
    ):
        raise RuntimeError("pair-1 operational evidence is incomplete")

    def evidence(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    request = {
        "schema_version": 1,
        "status": "awaiting_independent_approval",
        "pair_index": pair_results[0].pair_index,
        "pair_id": pair_results[0].pair_id,
        "games": [asdict(result) for result in pair_results],
        "config_sha256": resume_config_sha256(args),
        "progress_snapshot_sha256": hashlib.sha256(
            progress_snapshot_path(args).read_bytes()
        ).hexdigest(),
        "registrations_remaining": registration_entries,
        "protocol_logs": [evidence(path) for path in protocol_logs],
        "search_logs": [evidence(path) for path in search_logs],
        "prior_decisions": [evidence(path) for path in prior_dumps],
        "prior_health": prior_health,
        "showdown_launch": evidence(Path(args.operational_gate_showdown_launch)),
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with request_path.open("x", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2, sort_keys=True)
        handle.write("\n")
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    print(
        f"OPERATIONAL_GATE waiting after pair {pair_results[0].pair_index} "
        f"request_sha256={request_sha256}",
        flush=True,
    )
    deadline = asyncio.get_running_loop().time() + args.operational_gate_timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if approval_path.exists():
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
            review = json.loads(review_path.read_text(encoding="utf-8"))
            current_prior_health = [
                json.loads(urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5).read())
                for url in (args.agent_a_prior_server_url, args.agent_b_prior_server_url)
            ]
            if (
                approval != {
                    "approved": True,
                    "request_sha256": request_sha256,
                    "review_sha256": review_sha256,
                    "token": args.operational_gate_token,
                }
                or review.get("status") != "authorized"
                or review.get("request_sha256") != request_sha256
                or review.get("public_ladder_authorized") is not False
                or set(review.get("checks_passed", [])) != {
                    "two_decisive_mirrored_games",
                    "two_unique_battle_tags",
                    "registrations_consumed",
                    "protocol_semantics",
                    "search_semantics",
                    "schema4_prior_telemetry",
                    "processes_healthy",
                    "no_public_ladder",
                }
                or current_prior_health != prior_health
            ):
                raise RuntimeError("operational gate approval does not match the frozen request")
            print("OPERATIONAL_GATE approved; continuing unchanged", flush=True)
            return
        await asyncio.sleep(0.25)
    raise TimeoutError("operational gate approval timed out")


async def run_h2h(args: argparse.Namespace) -> tuple[EvalSummary, list[GameResult]]:
    if "direct_r1" in {args.agent_a, args.agent_b}:
        verify_direct_r1_checkpoint(args)
    server_configuration = make_server_configuration(args)
    schedule = side_schedule(args.n_games, args.paired)
    use_sprt = getattr(args, "sprt_h1", None) is not None
    pair_plans = load_or_create_pair_plans(args) if args.mirrored_pairs else None

    results = load_resume_results(args, schedule, pair_plans)
    completed_indices = {result.game_index for result in results}
    indexed_schedule = [
        (index, challenger, acceptor)
        for index, (challenger, acceptor) in enumerate(schedule, start=1)
        if index not in completed_indices
    ]
    if use_sprt:
        decisive = [
            result
            for result in results
            if not result.void and result.winner in {"agent_a", "agent_b"}
        ]
        wins = sum(result.winner == "agent_a" for result in decisive)
        losses = sum(result.winner == "agent_b" for result in decisive)
        if sprt_check(wins, losses, args.sprt_h0, args.sprt_h1) != "continue":
            indexed_schedule = []
    if args.mirrored_pairs and indexed_schedule:
        assert pair_plans is not None
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        completed_pairs = {result.pair_index for result in results}
        for plan in pair_plans:
            if plan.pair_index in completed_pairs:
                continue
            first_index = plan.pair_index * 2 - 1
            pair_results = []
            for pair_leg, game_index in enumerate((first_index, first_index + 1), start=1):
                challenger, acceptor = schedule[game_index - 1]
                pair_results.append(
                    await run_scheduled_game(
                        args,
                        server_configuration,
                        game_index,
                        challenger,
                        acceptor,
                        log_dir,
                        plan,
                        pair_leg,
                    )
                )
            results.extend(pair_results)
            emit_pair_progress(args, pair_results, results)
            if plan.pair_index == args.operational_gate_after_pairs:
                await await_operational_gate(args, pair_results)
            if use_sprt:
                decisive = [
                    result
                    for result in results
                    if not result.void and result.winner in {"agent_a", "agent_b"}
                ]
                wins = sum(result.winner == "agent_a" for result in decisive)
                losses = sum(result.winner == "agent_b" for result in decisive)
                decision = sprt_check(wins, losses, args.sprt_h0, args.sprt_h1)
                if decision != "continue":
                    print(
                        f"SPRT STOP: {decision} after {len(decisive)} decisive games "
                        f"and {plan.pair_index} complete pairs (w={wins} l={losses})",
                        flush=True,
                    )
                    break
        indexed_schedule = []
    max_concurrent = getattr(args, "concurrent_games", 1)
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if max_concurrent <= 1:
            for index, challenger, acceptor in indexed_schedule:
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
            queue = deque(indexed_schedule)
            pending = set()
            async def _run_one(idx, chal, acc):
                r = await run_scheduled_game(args, server_configuration, idx, chal, acc, log_dir)
                return idx, r
            while queue or pending:
                while queue and len(pending) < max_concurrent:
                    idx, chal, acc = queue.popleft()
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
            for index, challenger, acceptor in indexed_schedule:
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
    pair_groups: dict[int, list[GameResult]] = {}
    for result in results:
        if result.pair_index is not None:
            pair_groups.setdefault(result.pair_index, []).append(result)
    complete_pair_groups = [
        pair for pair in pair_groups.values()
        if len(pair) == 2 and all(not result.void and result.winner in {"agent_a", "agent_b"} for result in pair)
    ]
    void_pairs = len(pair_groups) - len(complete_pair_groups)
    pair_win_counts = [
        sum(result.winner == "agent_a" for result in pair) for pair in complete_pair_groups
    ]
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
        mirrored_pairs=args.mirrored_pairs,
        completed_pairs=len(complete_pair_groups),
        void_pairs=void_pairs,
        pair_sweeps_a=sum(wins == 2 for wins in pair_win_counts),
        pair_splits=sum(wins == 1 for wins in pair_win_counts),
        pair_sweeps_b=sum(wins == 0 for wins in pair_win_counts),
        pair_score_mean=(sum(pair_win_counts) / (2 * len(pair_win_counts)))
            if pair_win_counts else 0.0,
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
            f"paired={summary.paired}; mirrored_pairs={summary.mirrored_pairs}; "
            f"completed_pairs={summary.completed_pairs}; void_pairs={summary.void_pairs}; "
            f"pair_outcomes={summary.pair_sweeps_a}/{summary.pair_splits}/{summary.pair_sweeps_b}; "
            f"decisive={summary.decisive_games}; "
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
    fd, temp_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 Pokemon Showdown eval harness")
    parser.add_argument("--mode", choices=["h2h", "ladder"], default="h2h")
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--server", choices=["local", "live"], default="local")
    parser.add_argument("--websocket-uri", default=None)
    parser.add_argument("--authentication-uri", default=None)
    parser.add_argument("--n-games", type=int, default=2)
    parser.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirrored-pairs", action="store_true")
    parser.add_argument("--mirror-seed", type=int, default=0)
    parser.add_argument(
        "--showdown-dir",
        default=str(ROOT_DIR.parent / "external" / "pokemon-showdown"),
    )
    parser.add_argument(
        "--mirrored-team-generator",
        default=str(ROOT_DIR / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs"),
    )
    parser.add_argument("--pair-registration-dir", default=None)
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
    parser.add_argument("--production-remote-mcts", action="store_true")
    parser.add_argument("--production-remote-mcts-app", default="metagross-mcts-r1-p16")
    parser.add_argument("--production-remote-mcts-function", default="search_batch")
    parser.add_argument(
        "--production-remote-engine-sha256",
        default="d9a163c92d0371cf83a5319a7ac077c973734b9f4a816a083dcacfa0306c2f45",
    )
    parser.add_argument("--production-run-seed", default=None)
    parser.add_argument("--shared-root-iterations", type=int, default=10_000)
    parser.add_argument(
        "--shared-root-continuation-iterations", type=int, default=8
    )
    parser.add_argument("--shared-root-prior-strength", type=float, default=1.0)
    parser.add_argument(
        "--metamon-python",
        default=str(ROOT_DIR.parent / ".venv-metamon" / "bin" / "python"),
    )
    parser.add_argument(
        "--direct-r1-checkpoint-root",
        default=str(ROOT_DIR.parent / "srcs" / "models"),
    )
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
        parser.add_argument(
            f"--{slot}-teacher-root-bundle",
            default=None,
            help="Private JSONL output for deterministic same-root treatments.",
        )
    parser.add_argument("--teacher-determinization-schedules", type=int, default=1)
    parser.add_argument("--teacher-determinization-seed", type=int, default=0)
    parser.add_argument(
        "--teacher-manifest-sha256",
        default=None,
        help="Frozen input-manifest hash linked into every root bundle.",
    )
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument("--agent-a-cpuct", type=float, default=None,
                        help="Per-slot c_puct override for agent A (falls back to --cpuct).")
    parser.add_argument("--agent-b-cpuct", type=float, default=None,
                        help="Per-slot c_puct override for agent B (falls back to --cpuct).")
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
    parser.add_argument("--username-prefix", default="p0")
    parser.add_argument("--randbats-conditional-samples", type=int, default=24)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the atomic <json-out>.progress.json snapshot.",
    )
    parser.add_argument("--operational-gate-after-pairs", type=int, default=None)
    parser.add_argument("--operational-gate-request", default=None)
    parser.add_argument("--operational-gate-approval", default=None)
    parser.add_argument("--operational-gate-review", default=None)
    parser.add_argument("--operational-gate-prior-decisions", action="append", default=[])
    parser.add_argument("--operational-gate-showdown-launch", default=None)
    parser.add_argument("--operational-gate-token", default=None)
    parser.add_argument("--operational-gate-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--prepare-mirrored-pairs-only",
        action="store_true",
        help="Create and validate the complete mirrored pair manifest without starting a game.",
    )
    parser.add_argument("--pair-manifest-sha256", default=None)
    parser.add_argument(
        "--strict-isolated-priors",
        action="store_true",
        help="Require separate fail-closed prior servers for both H2H slots.",
    )
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
    production_controller_requested = any(
        agent
        in {
            "production_r1_search_first",
            "production_r1_independent_ensemble",
            "production_r1_shared_rm_plus",
            "production_r1_certified",
        }
        for agent in (args.agent_a, args.agent_b, args.agent)
    )
    if production_controller_requested and (
        not args.production_run_seed
        or not re.fullmatch(r"[0-9a-f]{64}", args.production_run_seed)
    ):
        raise ValueError("production controller agents require a 64-hex --production-run-seed")
    if args.production_remote_mcts:
        if not production_controller_requested:
            raise ValueError("--production-remote-mcts requires a production controller agent")
        if not re.fullmatch(r"[0-9a-f]{64}", args.production_remote_engine_sha256):
            raise ValueError("--production-remote-engine-sha256 must be 64 lowercase hex")
        production_agents = {
            "production_r1_search_first",
            "production_r1_independent_ensemble",
            "production_r1_shared_rm_plus",
            "production_r1_certified",
        }
        effective_budgets = {
            "agent_a": args.agent_a_search_time_ms or args.foul_play_search_time_ms,
            "agent_b": args.agent_b_search_time_ms or args.foul_play_search_time_ms,
            "agent": args.foul_play_search_time_ms,
        }
        configured_agents = {
            "agent_a": args.agent_a,
            "agent_b": args.agent_b,
            "agent": args.agent,
        }
        if any(
            effective_budgets[slot] != 500
            for slot, agent in configured_agents.items()
            if agent in production_agents
        ):
            raise ValueError("production remote MCTS requires an effective 500 ms search budget")
    if not 1 <= args.shared_root_iterations <= 1_000_000:
        raise ValueError("--shared-root-iterations must be in [1, 1000000]")
    if not 1 <= args.shared_root_continuation_iterations <= 1_000_000:
        raise ValueError(
            "--shared-root-continuation-iterations must be in [1, 1000000]"
        )
    if (
        not math.isfinite(args.shared_root_prior_strength)
        or not 0 <= args.shared_root_prior_strength <= 1_000
    ):
        raise ValueError(
            "--shared-root-prior-strength must be finite and in [0, 1000]"
        )
    if args.resume and not args.json_out:
        raise ValueError("--resume requires --json-out")
    gate_values = (
        args.operational_gate_after_pairs,
        args.operational_gate_request,
        args.operational_gate_approval,
        args.operational_gate_review,
        args.operational_gate_showdown_launch,
        args.operational_gate_token,
    )
    if any(value is not None for value in gate_values):
        if any(value is None for value in gate_values):
            raise ValueError("all operational gate arguments are required together")
        if not args.mirrored_pairs:
            raise ValueError("the operational gate requires --mirrored-pairs")
        if not 1 <= args.operational_gate_after_pairs < args.n_games // 2:
            raise ValueError("--operational-gate-after-pairs must precede the final pair")
        if args.operational_gate_timeout_seconds <= 0:
            raise ValueError("--operational-gate-timeout-seconds must be positive")
        if len(args.operational_gate_prior_decisions) != 2:
            raise ValueError("the operational gate requires exactly two prior decision paths")
        if args.resume:
            raise ValueError("--resume cannot bypass the frozen operational gate")
        for gate_path in (
            args.operational_gate_request,
            args.operational_gate_approval,
            args.operational_gate_review,
        ):
            if Path(gate_path).exists():
                raise ValueError(f"operational gate path must be fresh: {gate_path}")
    if args.mirrored_pairs:
        if args.mode != "h2h" or args.server != "local":
            raise ValueError("--mirrored-pairs requires local H2H mode")
        if not args.paired or args.n_games % 2:
            raise ValueError("--mirrored-pairs requires an even paired game count")
        if args.format != "gen9randombattle":
            raise ValueError("--mirrored-pairs currently supports only gen9randombattle")
        if not args.json_out or not args.log_dir or not args.pair_registration_dir:
            raise ValueError("--mirrored-pairs requires --json-out, --log-dir, and --pair-registration-dir")
        if args.concurrent_games != 1:
            raise ValueError("--mirrored-pairs currently requires --concurrent-games 1")
        if not is_external_agent(args.agent_a) or not is_external_agent(args.agent_b):
            raise ValueError(
                "--mirrored-pairs requires registration-aware Foul Play/direct-r1 agents"
            )
        if not args.fail_fast:
            raise ValueError("--mirrored-pairs requires --fail-fast")
        registration_dir = Path(args.pair_registration_dir).resolve()
        if not args.resume and registration_dir.exists() and any(registration_dir.iterdir()):
            raise ValueError("--pair-registration-dir must be fresh or empty")
        if not 0 <= args.mirror_seed < 2**64:
            raise ValueError("--mirror-seed must fit an unsigned 64-bit integer")
        if args.pair_manifest_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.pair_manifest_sha256):
            raise ValueError("--pair-manifest-sha256 must be 64 lowercase hex characters")
    if args.strict_isolated_priors:
        if not args.agent_a_require_priors or not args.agent_b_require_priors:
            raise ValueError(
                "--strict-isolated-priors requires both per-slot --require-priors flags"
            )
        if not args.agent_a_prior_server_url or not args.agent_b_prior_server_url:
            raise ValueError(
                "--strict-isolated-priors requires both per-slot prior server URLs"
            )
        if args.agent_a_prior_server_url.rstrip("/") == args.agent_b_prior_server_url.rstrip("/"):
            raise ValueError("--strict-isolated-priors requires distinct prior server URLs")
        if not uses_prior_server(args.agent_a) or not uses_prior_server(args.agent_b):
            raise ValueError("--strict-isolated-priors requires two prior-enabled agents")
    teacher_paths = [
        args.agent_a_teacher_root_bundle,
        args.agent_b_teacher_root_bundle,
    ]
    if any(teacher_paths):
        if args.teacher_determinization_schedules <= 0:
            raise ValueError("--teacher-determinization-schedules must be positive")
        if not 0 <= args.teacher_determinization_seed < 2**64:
            raise ValueError("--teacher-determinization-seed must fit an unsigned 64-bit integer")
        if args.teacher_manifest_sha256 is None or not re.fullmatch(
            r"[0-9a-fA-F]{64}", args.teacher_manifest_sha256
        ):
            raise ValueError("root-bundle capture requires --teacher-manifest-sha256")
        if args.concurrent_games != 1:
            raise ValueError(
                "root-bundle capture requires --concurrent-games 1 unless each game has a distinct output"
            )
        if args.agent_a_teacher_root_bundle and not args.agent_a_require_priors:
            raise ValueError("agent-a root-bundle capture requires --agent-a-require-priors")
        if args.agent_b_teacher_root_bundle and not args.agent_b_require_priors:
            raise ValueError("agent-b root-bundle capture requires --agent-b-require-priors")
        for slot, path in zip(("agent_a", "agent_b"), teacher_paths):
            if path and agent_for_slot(args, slot) not in {
                "foul_play_root_priors",
                "foul_play_root_priors_opp",
            }:
                raise ValueError(
                    "schedule-aware root capture currently supports only ordinary root-prior Foul Play agents"
                )
        resolved_paths = [str(Path(path).resolve()) for path in teacher_paths if path]
        if len(resolved_paths) != len(set(resolved_paths)):
            raise ValueError("agent root-bundle output paths must differ")
    if not re.fullmatch(r"[a-z0-9]{1,8}", args.username_prefix):
        raise ValueError("--username-prefix must contain 1-8 lowercase alphanumeric characters")
    return args


async def async_main(args: argparse.Namespace) -> None:
    if args.prepare_mirrored_pairs_only:
        if not args.mirrored_pairs:
            raise ValueError("--prepare-mirrored-pairs-only requires --mirrored-pairs")
        plans = load_or_create_pair_plans(args)
        print(json.dumps({"prepared_pairs": len(plans), "config_sha256": resume_config_sha256(args)}))
        return
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
        "execution_identity": execution_identity(args),
    }
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.json_out:
        write_json(args.json_out, payload)
    if args.append_experiment_log:
        append_experiment_row(args, summary)


def main(argv: Optional[list[str]] = None) -> None:
    raw_arguments = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_arguments)
    args._raw_arguments = list(raw_arguments)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
