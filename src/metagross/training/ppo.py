from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F

from metagross.model.checkpoint import load_checkpoint, save_checkpoint
from metagross.model.network import PokeNet
from metagross.model.state import build_vocabulary, stack_encoded
from .env_runner import RolloutPlayer, RolloutStep, compute_gae

try:
    from poke_env import AccountConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from poke_env.ps_client import ServerConfiguration
except Exception:  # pragma: no cover - poke-env is optional for local imports
    AccountConfiguration = None  # type: ignore[assignment]
    SimpleHeuristicsPlayer = None  # type: ignore[assignment]
    ServerConfiguration = None  # type: ignore[assignment]


@dataclass
class PPOConfig:
    gamma: float = 0.9999
    gae_lambda: float = 0.754
    clip_range: float = 0.083
    value_clip: float = 0.018
    entropy_coef: float = 0.059
    value_coef: float = 0.438
    max_grad_norm: float = 0.543
    minibatch_size: int = 1024
    sgd_epochs: int = 7
    rollout_steps: int = 512
    environments: int = 100


def wang_lr(update_index: int) -> float:
    return 10 ** (-4.23) / (8 * update_index + 1) ** 1.5


def ppo_loss(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_values: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    clipped_values = old_values + torch.clamp(values - old_values, -config.value_clip, config.value_clip)
    value_loss = torch.max((values - returns).pow(2), (clipped_values - returns).pow(2)).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def _server_configuration(server_url: str) -> object:
    if ServerConfiguration is None:
        raise RuntimeError("poke-env is not installed; install requirements.txt first")
    websocket_url = server_url
    if not websocket_url.startswith("ws://") and not websocket_url.startswith("wss://"):
        websocket_url = f"ws://{websocket_url}"
    if websocket_url.rstrip("/").endswith(":8000"):
        websocket_url = websocket_url.rstrip("/") + "/showdown/websocket"
    return ServerConfiguration(websocket_url=websocket_url, authentication_url="https://play.pokemonshowdown.com/action.php?")


def _server_reachable(server_url: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(server_url if "://" in server_url else f"ws://{server_url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _account(name: str) -> object | None:
    if AccountConfiguration is None:
        return None
    return AccountConfiguration(name, None)


def ppo_update(model: PokeNet, optimizer: torch.optim.Optimizer, steps: list[RolloutStep], config: PPOConfig, update: int, device: torch.device) -> dict[str, float]:
    model.train()
    indices = np.arange(len(steps))
    metrics: list[dict[str, float]] = []
    for _epoch in range(config.sgd_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch_steps = [steps[int(index)] for index in indices[start : start + config.minibatch_size]]
            batch = stack_encoded([step.state for step in batch_steps])
            actions = torch.tensor([step.action for step in batch_steps], dtype=torch.long, device=device)
            old_log_probs = torch.tensor([step.log_prob for step in batch_steps], dtype=torch.float32, device=device)
            returns = torch.tensor([step.return_ for step in batch_steps], dtype=torch.float32, device=device)
            advantages = torch.tensor([step.advantage for step in batch_steps], dtype=torch.float32, device=device)
            old_values = torch.tensor([step.value for step in batch_steps], dtype=torch.float32, device=device)
            logits, values = model(batch)
            loss, metric = ppo_loss(
                logits=logits,
                values=values,
                actions=actions,
                old_log_probs=old_log_probs,
                returns=returns,
                advantages=advantages,
                old_values=old_values,
                config=config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(metric)
    if not metrics:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    return {key: sum(metric[key] for metric in metrics) / len(metrics) for key in metrics[0]}


async def validate(model: PokeNet, vocab: object, device: torch.device, server_url: str = "ws://localhost:8000", n_games: int = 100) -> float:
    if SimpleHeuristicsPlayer is None:
        raise RuntimeError("poke-env is not installed; cannot run validation")
    server = _server_configuration(server_url)
    tag = random.randint(0, 1_000_000)
    player = RolloutPlayer(
        model,
        vocab,  # type: ignore[arg-type]
        device,
        account_configuration=_account(f"AlphaPPOVal{tag}"),
        battle_format="gen9randombattle",
        server_configuration=server,
        max_concurrent_battles=1,
    )
    opponent = SimpleHeuristicsPlayer(
        account_configuration=_account(f"SimpleVal{tag}"),
        battle_format="gen9randombattle",
        server_configuration=server,
        max_concurrent_battles=1,
    )
    await player.battle_against(opponent, n_battles=n_games)
    wins = int(getattr(player, "n_won_battles", 0))
    winrate = wins / max(1, n_games)
    print(json.dumps({"validation_games": n_games, "wins": wins, "winrate": winrate}))
    return winrate


async def run_smoke(model: PokeNet, vocab: object, device: torch.device, server_url: str, timeout: float = 90.0) -> None:
    if not _server_reachable(server_url):
        raise SystemExit("Pokemon Showdown server is not reachable. Start it with: cd ~/ps-server && node pokemon-showdown start --no-security")
    server = _server_configuration(server_url)
    tag = random.randint(0, 1_000_000)
    p1 = RolloutPlayer(
        model,
        vocab,  # type: ignore[arg-type]
        device,
        account_configuration=_account(f"AlphaPPOA{tag}"),
        battle_format="gen9randombattle",
        server_configuration=server,
        max_concurrent_battles=1,
    )
    p2 = RolloutPlayer(
        model,
        vocab,  # type: ignore[arg-type]
        device,
        account_configuration=_account(f"AlphaPPOB{tag}"),
        battle_format="gen9randombattle",
        server_configuration=server,
        max_concurrent_battles=1,
    )
    before_wins = int(getattr(p1, "n_won_battles", 0))
    await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=timeout)
    after_wins = int(getattr(p1, "n_won_battles", 0))
    outcome_p1 = 1.0 if after_wins > before_wins else -1.0
    steps_p1 = compute_gae(p1.collect_episode(outcome_p1))
    steps_p2 = compute_gae(p2.collect_episode(-outcome_p1))
    print(json.dumps({"smoke": True, "p1_steps": len(steps_p1), "p2_steps": len(steps_p2), "outcome_p1": outcome_p1}))


async def run_training(args: argparse.Namespace) -> None:
    config = PPOConfig()
    print(json.dumps({"config": asdict(config), "lr0": wang_lr(0), "lr100": wang_lr(100)}, indent=2))
    if not _server_reachable(args.server_url):
        raise SystemExit("Pokemon Showdown server is not reachable. Start it with: cd ~/ps-server && node pokemon-showdown start --no-security")
    device = torch.device(args.device)
    vocab = build_vocabulary(args.pool)
    model = load_checkpoint(args.checkpoint).to(device) if args.checkpoint else PokeNet(vocab=vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wang_lr(0))
    if args.smoke:
        await run_smoke(model, vocab, device, args.server_url, timeout=args.timeout)
        return
    if args.dry_run:
        return
    server = _server_configuration(args.server_url)
    tag = random.randint(0, 1_000_000)
    p1 = RolloutPlayer(model, vocab, device, account_configuration=_account(f"AlphaPPOA{tag}"), battle_format="gen9randombattle", server_configuration=server, max_concurrent_battles=1)
    p2 = RolloutPlayer(model, vocab, device, account_configuration=_account(f"AlphaPPOB{tag}"), battle_format="gen9randombattle", server_configuration=server, max_concurrent_battles=1)
    update = 0
    accumulated_steps: list[RolloutStep] = []
    while args.max_updates is None or update < args.max_updates:
        before_wins = int(getattr(p1, "n_won_battles", 0))
        await p1.battle_against(p2, n_battles=1)
        outcome_p1 = 1.0 if int(getattr(p1, "n_won_battles", 0)) > before_wins else -1.0
        steps_p1 = compute_gae(p1.collect_episode(outcome_p1), config.gamma, config.gae_lambda)
        steps_p2 = compute_gae(p2.collect_episode(-outcome_p1), config.gamma, config.gae_lambda)
        accumulated_steps.extend(steps_p1 + steps_p2)
        print(json.dumps({"game_steps": len(steps_p1) + len(steps_p2), "accumulated_steps": len(accumulated_steps)}))
        if len(accumulated_steps) >= config.rollout_steps * 20:
            metrics = ppo_update(model, optimizer, accumulated_steps, config, update, device)
            accumulated_steps.clear()
            update += 1
            for group in optimizer.param_groups:
                group["lr"] = wang_lr(update)
            print(json.dumps({"update": update, **metrics, "lr": wang_lr(update)}))
            if update % 100 == 0:
                await validate(model, vocab, device, args.server_url, n_games=args.validation_games)
            if args.output:
                save_checkpoint(args.output, model, optimizer, phase="phase2", update=update)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 PPO self-play trainer scaffold")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--smoke", action="store_true", help="Run exactly one self-play game and exit")
    parser.add_argument("--server-url", default="ws://localhost:8000")
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/phase2.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--validation-games", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PPOConfig()
    if args.dry_run and not args.smoke:
        print(json.dumps({"config": asdict(config), "lr0": wang_lr(0), "lr100": wang_lr(100)}, indent=2))
        return
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
