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

from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import PokeNet
from model.state import build_vocabulary, stack_encoded
from .env_runner import RolloutPlayer, RolloutStep, compute_gae

try:
    from poke_env import AccountConfiguration
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
    from poke_env.ps_client import ServerConfiguration
except Exception:  # pragma: no cover - poke-env is optional for local imports
    AccountConfiguration = None  # type: ignore[assignment]
    MaxBasePowerPlayer = None  # type: ignore[assignment]
    RandomPlayer = None  # type: ignore[assignment]
    SimpleHeuristicsPlayer = None  # type: ignore[assignment]
    ServerConfiguration = None  # type: ignore[assignment]


# Curriculum: start weak, increase difficulty as win rate rises.
# Format: list of (opponent_class_name, promote_at_winrate)
# MaxBasePowerPlayer removed — it creates a hard domain shift from Random
# (policy unlearns RandomPlayer habits against an always-max-damage opponent
# before having a chance to learn against strategic play). Go directly to
# SimpleHeuristicsPlayer which provides richer, more representative signal.
CURRICULUM = [
    ("RandomPlayer",           0.70),   # ~50% baseline → promote at 70%
    ("SimpleHeuristicsPlayer", None),   # final target — no promotion
]

_OPPONENT_CLASSES = {
    "RandomPlayer":           lambda: RandomPlayer,
    "MaxBasePowerPlayer":     lambda: MaxBasePowerPlayer,
    "SimpleHeuristicsPlayer": lambda: SimpleHeuristicsPlayer,
}


@dataclass
class PPOConfig:
    gamma: float = 0.9999
    gae_lambda: float = 0.754
    clip_range: float = 0.083
    value_clip: float = 0.018
    entropy_coef: float = 0.10   # raised from 0.059 — keeps entropy >1.8 during curriculum transition
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
    use_flat_lr = args.lr is not None
    init_lr = args.lr if use_flat_lr else wang_lr(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    if use_flat_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.max_updates or 5000), eta_min=1e-6)
    if args.smoke:
        await run_smoke(model, vocab, device, args.server_url, timeout=args.timeout)
        return
    if args.dry_run:
        return
    server = _server_configuration(args.server_url)
    n_envs = max(1, args.n_envs)
    tag = random.randint(0, 1_000_000)

    def make_envs(opponent_class_name: str, tag: int) -> list[tuple[Any, Any]]:
        """Spawn n_envs (p1, p2) pairs with the given opponent class."""
        opp_cls = _OPPONENT_CLASSES.get(opponent_class_name, lambda: SimpleHeuristicsPlayer)()
        envs = []
        for i in range(n_envs):
            p1 = RolloutPlayer(model, vocab, device,
                               account_configuration=_account(f"PPO_A{tag}_{i}"),
                               battle_format="gen9randombattle",
                               server_configuration=server,
                               max_concurrent_battles=1)
            if not args.vs_heuristic or opp_cls is None:
                p2: Any = RolloutPlayer(model, vocab, device,
                                        account_configuration=_account(f"PPO_B{tag}_{i}"),
                                        battle_format="gen9randombattle",
                                        server_configuration=server,
                                        max_concurrent_battles=1)
            else:
                p2 = opp_cls(
                    account_configuration=_account(f"PPO_O{tag}_{i}"),
                    battle_format="gen9randombattle",
                    server_configuration=server,
                    max_concurrent_battles=1)
            envs.append((p1, p2))
        return envs

    # Curriculum: start with RandomPlayer, promote when win rate threshold met.
    curriculum_idx = 0
    if not args.vs_heuristic:
        # No curriculum for self-play mode.
        curriculum_idx = len(CURRICULUM) - 1
    current_opponent = CURRICULUM[curriculum_idx][0]
    envs = make_envs(current_opponent, tag)
    print(json.dumps({"curriculum_start": current_opponent}), flush=True)

    async def run_one_game(p1: RolloutPlayer, p2: Any) -> tuple[list[RolloutStep], float]:
        before = int(getattr(p1, "n_won_battles", 0))
        await p1.battle_against(p2, n_battles=1)
        outcome = 1.0 if int(getattr(p1, "n_won_battles", 0)) > before else -1.0
        steps = compute_gae(p1.collect_episode(outcome), config.gamma, config.gae_lambda)
        if isinstance(p2, RolloutPlayer):
            steps += compute_gae(p2.collect_episode(-outcome), config.gamma, config.gae_lambda)
        return steps, outcome

    update = 0
    accumulated_steps: list[RolloutStep] = []
    recent_outcomes: list[float] = []  # rolling window for curriculum promotion
    while args.max_updates is None or update < args.max_updates:
        results = await asyncio.gather(*[run_one_game(p1, p2) for p1, p2 in envs])
        batch: list[RolloutStep] = []
        for steps, outcome in results:
            batch.extend(steps)
            recent_outcomes.append(outcome)
        accumulated_steps.extend(batch)
        print(json.dumps({"game_steps": len(batch), "accumulated_steps": len(accumulated_steps), "opponent": current_opponent}), flush=True)
        if len(accumulated_steps) >= config.rollout_steps * 20:
            metrics = ppo_update(model, optimizer, accumulated_steps, config, update, device)
            accumulated_steps.clear()
            update += 1
            if use_flat_lr:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = wang_lr(update)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            # Rolling win rate over last 50 games
            window = recent_outcomes[-50:]
            rolling_wr = sum(1 for o in window if o > 0) / max(1, len(window))
            print(json.dumps({"update": update, **metrics, "lr": current_lr,
                               "opponent": current_opponent, "rolling_wr": round(rolling_wr, 3)}), flush=True)
            # Curriculum promotion
            promote_at = CURRICULUM[curriculum_idx][1]
            if (args.vs_heuristic and promote_at is not None
                    and rolling_wr >= promote_at
                    and len(window) >= 30
                    and curriculum_idx < len(CURRICULUM) - 1):
                curriculum_idx += 1
                current_opponent = CURRICULUM[curriculum_idx][0]
                tag2 = random.randint(0, 1_000_000)
                envs = make_envs(current_opponent, tag2)
                print(json.dumps({"curriculum_promote": current_opponent, "rolling_wr": round(rolling_wr, 3)}), flush=True)
            if update % 100 == 0:
                wr = await validate(model, vocab, device, args.server_url, n_games=args.validation_games)
                if wr >= 0.80:
                    print(json.dumps({"gate": "PASSED", "winrate": wr}), flush=True)
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
    parser.add_argument("--lr", type=float, default=None,
                        help="Constant initial LR. If set, uses cosine decay to lr-min instead of Wang schedule.")
    parser.add_argument("--validation-games", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of concurrent self-play environments (asyncio.gather)")
    parser.add_argument("--vs-heuristic", action="store_true", default=True,
                        help="Train against SimpleHeuristicsPlayer (default). Use --no-vs-heuristic for self-play.")
    parser.add_argument("--no-vs-heuristic", dest="vs_heuristic", action="store_false")
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
