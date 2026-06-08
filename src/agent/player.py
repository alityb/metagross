from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from belief import BeliefStateModule
from search import MCTSEngine
from model.checkpoint import load_checkpoint, new_model_from_pool
from model.state import build_vocabulary, encode_state
from rlm import RLMStrategist
from rlm.strategist import NullRLMStrategist

try:  # poke-env is an optional runtime dependency for local smoke tests.
    from poke_env.player import Player
except Exception:  # pragma: no cover - exercised only when poke-env is missing
    Player = object  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)


class AlphaPokemonCore:
    def __init__(
        self,
        checkpoint: str | None = None,
        pool: str = "data/all_gen_pool.json",
        rlm: str = "null",         # "null" | "heuristic" | "anthropic" | "local"
        rlm_model: str | None = None,
        time_budget: float = 7.5,
    ):
        default_pool = pool if Path(pool).exists() else "data/gen9_random_pool.json"
        self.pool_path = default_pool
        self.pool = json.loads(Path(default_pool).read_text()) if Path(default_pool).exists() else {}
        self.vocab = build_vocabulary(default_pool)
        self.pokenet = load_checkpoint(checkpoint) if checkpoint else new_model_from_pool(default_pool)[0]
        # Use NullRLMStrategist by default — drops in real RLM when ready.
        if rlm == "null":
            self.rlm: Any = NullRLMStrategist()
        else:
            self.rlm = RLMStrategist.from_provider(rlm, model=rlm_model)
        self.belief = BeliefStateModule(default_pool)
        self.mcts = MCTSEngine(self.pokenet, workers=20, time_budget=time_budget)
        self.full_log = ""

    def choose_action_index(self, battle: Any) -> int:
        # 1. Rule-based Bayesian update.
        self.belief.update(battle)
        # 2. RLM strategic reasoning (uses full log for cross-turn compound inference).
        # Get a base policy first (with UNKNOWN opponent tokens) for the RLM prompt.
        state_unknown = encode_state(battle, vocab=self.vocab)
        base_policy, _value = self.pokenet.policy_value(state_unknown)
        rlm_out = self.rlm.assess(
            log=self.full_log, state=battle, pool=self.pool,
            base_policy=base_policy[0].detach().cpu().tolist()
        )
        # 3. Merge RLM posterior into belief module.
        self.belief.refine(rlm_out.refined_belief)
        # 4. Re-encode state with belief posterior filling opponent UNKNOWN tokens.
        #    This is the key fix: PokeNet sees the most-probable opponent set instead
        #    of UNKNOWN everywhere.
        belief_posterior = self.belief.posterior()  # {slot: [{set, probability}, ...]}
        state = encode_state(battle, vocab=self.vocab, belief_posterior=belief_posterior)
        # 5. Sample K=4 opponent configs for K-tree MCTS parallelization.
        # 6. Run MCTS with RLM-augmented root prior.
        return self.mcts.search(
            state=state,
            configs=self.belief.sample(k=4),
            root_prior_rlm=rlm_out.pi_rlm,
            root_value_rlm=rlm_out.v_rlm,
        )


def _safe_create_order(player: Any, target: Any, *, terastallize: bool = False) -> Any:
    try:
        return player.create_order(target, terastallize=terastallize)
    except TypeError:
        if terastallize:
            return player.create_order(target)
        raise


def decode_action_to_order(player: Any, battle: Any, action: int) -> Any:
    moves = list(getattr(battle, "available_moves", []) or [])
    switches = list(getattr(battle, "available_switches", []) or [])
    if 0 <= action <= 3 and action < len(moves):
        return _safe_create_order(player, moves[action])
    if 4 <= action <= 8 and action - 4 < len(switches):
        return _safe_create_order(player, switches[action - 4])
    if 9 <= action <= 12 and action - 9 < len(moves):
        return _safe_create_order(player, moves[action - 9], terastallize=True)
    if action == 13 and switches:
        return _safe_create_order(player, switches[0], terastallize=True)
    LOGGER.warning("Illegal action %s; falling back to first available move", action)
    if moves:
        return _safe_create_order(player, moves[0])
    if switches:
        return _safe_create_order(player, switches[0])
    if hasattr(player, "choose_random_move"):
        return player.choose_random_move(battle)
    raise RuntimeError("No legal moves or switches available")


class AlphaPokemonAgent(Player):  # type: ignore[misc,valid-type]
    def __init__(self, checkpoint: str | None = None, pool: str = "data/gen9_random_pool.json", **kwargs: Any):
        super().__init__(**kwargs)
        self.core = AlphaPokemonCore(checkpoint=checkpoint, pool=pool)

    def choose_move(self, battle: Any) -> Any:
        action = self.core.choose_action_index(battle)
        return decode_action_to_order(self, battle, action)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaPokemon Showdown agent")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--mode", choices=["smoke", "search_ladder"], default="smoke")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--format", default="gen9randombattle")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    core = AlphaPokemonCore(checkpoint=args.checkpoint, pool=args.pool)
    if args.mode == "smoke":
        action = core.choose_action_index({"turn": 1, "available_moves": [{"move": "tackle", "disabled": False}]})
        print(json.dumps({"action": action, "model_params": core.pokenet.parameter_count()}))
        return
    raise SystemExit("poke-env ladder connection is not wired yet; core decision path is available in smoke mode.")


if __name__ == "__main__":
    main()
