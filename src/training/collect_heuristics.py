"""Collect SimpleHeuristicsPlayer vs SimpleHeuristicsPlayer game trajectories.

This generates Phase 1 IL data from strategic (heuristic) play rather than
random play — fixing the core problem where the policy learns random-move
habits that don't transfer to real games.

Masking is derived deterministically from what poke-env's Battle object
exposes at each decision point (exactly what the real player could see):
- Own team: fully known via battle.team
- Opponent: only revealed pokemon via battle.opponent_team (poke-env tracks
  this automatically — unrevealed pokemon simply don't appear)

This is trivially correct for random battles (no team preview — opponent
starts as 0 revealed, grows monotonically as they switch in / use moves).

Usage:
    python -m training.collect_heuristics \\
        --pool data/all_gen_pool.json \\
        --output data/heuristics \\
        --n-games 20000 \\
        --n-envs 16 \\
        --server-url ws://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.state import Vocabulary, build_vocabulary, encode_state
from pipeline.sample_types import TrainingSample

LOGGER = logging.getLogger(__name__)

try:
    from poke_env.player import SimpleHeuristicsPlayer
    from poke_env.environment.move import Move
    from poke_env.environment.pokemon import Pokemon
except ImportError as exc:
    raise SystemExit("poke-env required: pip install poke-env") from exc


# ------------------------------------------------------------------ #
# Action index ↔ BattleOrder conversion                               #
# ------------------------------------------------------------------ #

def order_to_action_index(order: object, battle: object) -> int:
    """Map a poke-env BattleOrder back to our 0–13 integer action space.

    Action layout:
        0–3:   move slot 0–3
        4–8:   switch to bench slot 0–4
        9–12:  terastallize + move slot 0–3
        13:    terastallize + switch slot 0
    """
    target = getattr(order, "order", None)
    if target is None:
        return 0
    is_tera = bool(getattr(order, "terastallize", False))
    moves    = list(getattr(battle, "available_moves",    []) or [])
    switches = list(getattr(battle, "available_switches", []) or [])

    # Switch?
    if isinstance(target, Pokemon):
        for i, sw in enumerate(switches[:5]):
            if sw.species == target.species:
                return 13 if is_tera else 4 + i
        return 4  # fallback: first switch

    # Move (Move object or has .id)
    move_id = getattr(target, "id", None) or str(target)
    for i, mv in enumerate(moves[:4]):
        mv_id = getattr(mv, "id", None) or str(mv)
        if mv_id == move_id:
            return (9 + i) if is_tera else i
    return 0  # fallback: first move


# ------------------------------------------------------------------ #
# Recording player                                                     #
# ------------------------------------------------------------------ #

class RecordingHeuristicsPlayer(SimpleHeuristicsPlayer):  # type: ignore[misc]
    """SimpleHeuristicsPlayer that records (state, action) at each decision."""

    def __init__(self, vocab: Vocabulary, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        # Buffer: list of (battle_tag, turn, encoded_state, action_idx)
        self._buf: list[tuple[str, int, object, int]] = []

    def choose_move(self, battle):  # type: ignore[override]
        # Encode from the player's own perspective.
        # poke-env Battle already applies partial observability:
        # battle.opponent_team only contains REVEALED pokemon.
        state = encode_state(battle, vocab=self.vocab)
        order = super().choose_move(battle)
        action = order_to_action_index(order, battle)
        self._buf.append((battle.battle_tag, battle.turn, state, action))
        return order

    def flush_episode(self, battle_tag: str, outcome: float) -> list[TrainingSample]:
        """Convert buffered steps for one finished battle into TrainingSamples."""
        samples: list[TrainingSample] = []
        remaining = []
        for tag, turn, state, action in self._buf:
            if tag == battle_tag:
                samples.append(TrainingSample(
                    battle_id=tag,
                    turn=turn,
                    encoded_state=state,
                    human_action=action,
                    outcome=outcome,
                    true_opponent_team={},
                    generation=9,
                ))
            else:
                remaining.append((tag, turn, state, action))
        self._buf = remaining
        return samples


# ------------------------------------------------------------------ #
# Async game runner                                                    #
# ------------------------------------------------------------------ #

def _server_config(url: str):
    from poke_env import ServerConfiguration
    ws = url.replace("http://", "ws://").replace("https://", "wss://")
    if not ws.startswith("ws"):
        ws = f"ws://{ws}"
    host = ws.replace("ws://", "").replace("wss://", "")
    return ServerConfiguration(f"{host}", "")


def _account(name: str):
    from poke_env import AccountConfiguration
    return AccountConfiguration(name, None)


async def run_collection(
    *,
    vocab: Vocabulary,
    server_url: str,
    n_games: int,
    n_envs: int,
    output_dir: Path,
    batch_size: int = 200,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = random.randint(0, 10**6)
    server = _server_config(server_url)

    players_a: list[RecordingHeuristicsPlayer] = []
    players_b: list[RecordingHeuristicsPlayer] = []
    for i in range(n_envs):
        a = RecordingHeuristicsPlayer(
            vocab,
            account_configuration=_account(f"CollA{tag}_{i}"),
            battle_format="gen9randombattle",
            server_configuration=server,
            max_concurrent_battles=1,
        )
        b = RecordingHeuristicsPlayer(
            vocab,
            account_configuration=_account(f"CollB{tag}_{i}"),
            battle_format="gen9randombattle",
            server_configuration=server,
            max_concurrent_battles=1,
        )
        players_a.append(a)
        players_b.append(b)

    all_samples: list[TrainingSample] = []
    batch_idx = len(list(output_dir.glob("heuristics_*.pkl")))
    games_done = 0

    async def play_one(a: RecordingHeuristicsPlayer, b: RecordingHeuristicsPlayer) -> list[TrainingSample]:
        before_a = int(getattr(a, "n_won_battles", 0))
        await a.battle_against(b, n_battles=1)
        won_a = int(getattr(a, "n_won_battles", 0)) > before_a
        outcome_a = 1.0 if won_a else -1.0

        # Faint shaping (same as PPO)
        try:
            battles_a = list(a.battles.values())
            if battles_a:
                last = battles_a[-1]
                tag_ = last.battle_tag
                own_f  = 6 if not won_a else sum(1 for p in last.team.values() if p.fainted)
                opp_f  = 6 if won_a     else sum(1 for p in last.opponent_team.values() if p.fainted)
                outcome_a += 0.1 * (opp_f - own_f)
                # Flush both sides
                samples  = a.flush_episode(tag_, outcome_a)
                samples += b.flush_episode(tag_, -outcome_a)
                return samples
        except Exception as exc:
            LOGGER.debug("Faint shaping error (non-fatal): %s", exc)
        return []

    while games_done < n_games:
        remaining = n_games - games_done
        active = min(n_envs, remaining)
        results = await asyncio.gather(*[
            play_one(players_a[i], players_b[i])
            for i in range(active)
        ])
        for samples in results:
            all_samples.extend(samples)
        games_done += active

        if len(all_samples) >= batch_size * 100:
            path = output_dir / f"heuristics_{batch_idx:06d}.pkl"
            with path.open("wb") as fh:
                pickle.dump(all_samples, fh)
            print(json.dumps({"games": games_done, "samples": len(all_samples), "path": str(path)}),
                  flush=True)
            batch_idx += 1
            all_samples = []

    if all_samples:
        path = output_dir / f"heuristics_{batch_idx:06d}.pkl"
        with path.open("wb") as fh:
            pickle.dump(all_samples, fh)
        print(json.dumps({"games": games_done, "samples": len(all_samples), "path": str(path)}),
              flush=True)

    print(json.dumps({"done": True, "total_games": games_done}), flush=True)


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SimpleHeuristics IL data")
    parser.add_argument("--pool",       default="data/all_gen_pool.json")
    parser.add_argument("--output",     default="data/heuristics")
    parser.add_argument("--n-games",    type=int, default=20_000)
    parser.add_argument("--n-envs",     type=int, default=16)
    parser.add_argument("--server-url", default="ws://localhost:8000")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Games per output batch (× ~150 states/game)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    vocab = build_vocabulary(args.pool)
    asyncio.run(run_collection(
        vocab=vocab,
        server_url=args.server_url,
        n_games=args.n_games,
        n_envs=args.n_envs,
        output_dir=Path(args.output),
        batch_size=args.batch_size,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
