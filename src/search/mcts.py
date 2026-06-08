from __future__ import annotations

import time
import random
import logging
from typing import Any, Protocol

import torch

from model.network import PokeNet
from model.state import EncodedState, Vocabulary, build_vocabulary, encode_state
from model.engine_bridge import battle_to_poke_engine_state, encode_poke_engine_state
from rlm.strategist import blend_root_prior

from .node import MCTSNode


LOGGER = logging.getLogger(__name__)


class Simulator(Protocol):
    def is_terminal(self, state: Any) -> bool:
        ...

    def terminal_value(self, state: Any) -> float:
        ...

    def next_state(self, state: Any, action: int, config: dict[str, Any] | None = None) -> Any:
        ...


class MCTSEngine:
    """Root-prior neural MCTS shell.

    Without a poke-engine simulator this still performs the production-critical
    root blend and returns the highest-prior legal action. Supplying a simulator
    enables time-budgeted neural rollouts through the same interface.
    """

    def __init__(
        self,
        model: PokeNet,
        simulator: Simulator | None = None,
        workers: int = 20,
        time_budget: float = 7.5,
        c_puct: float = 1.25,
        device: str | torch.device | None = None,
        use_poke_engine: bool = True,
        vocab: Vocabulary | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab or build_vocabulary("data/gen9_random_pool.json")
        self.simulator = simulator
        self.workers = workers
        self.time_budget = time_budget
        self.c_puct = c_puct
        self._rollout_count = 0
        if device is not None:
            self.model.to(device)
        if self.simulator is None and use_poke_engine:
            try:
                self.simulator = PokeEngineSimulator(self.model, self.vocab)
            except Exception as exc:
                LOGGER.warning("poke_engine simulator unavailable; using root-prior fallback: %s", exc)
                self.simulator = None

    @torch.no_grad()
    def root_policy_value(self, state: Any, root_prior_rlm: list[float] | None = None, root_value_rlm: float | None = None) -> tuple[list[float], float]:
        encoded = self._encode_for_model(state)
        policy, value = self.model.policy_value(encoded)
        pi_net = policy[0].detach().cpu().tolist()
        root_prior = blend_root_prior(pi_net, root_prior_rlm or pi_net) if root_prior_rlm is not None else pi_net
        if root_value_rlm is not None:
            root_value = 0.5 * float(value[0].detach().cpu()) + 0.5 * float(root_value_rlm)
        else:
            root_value = float(value[0].detach().cpu())
        mask = encoded.action_mask.tolist()
        root_prior = [prior if mask[idx] else 0.0 for idx, prior in enumerate(root_prior)]
        total = sum(root_prior)
        if total <= 0:
            root_prior = [1.0 / len(root_prior)] * len(root_prior)
        else:
            root_prior = [prior / total for prior in root_prior]
        return root_prior, root_value

    def _encode_for_model(self, state: Any) -> EncodedState:
        if isinstance(state, EncodedState):
            return state
        if hasattr(state, "side_one") and hasattr(state, "side_two") and hasattr(state, "to_string"):
            return encode_poke_engine_state(state, vocab=self.vocab)
        return encode_state(state, vocab=self.vocab)

    def search(
        self,
        *,
        state: Any,
        configs: list[dict[str, Any]] | None = None,
        root_prior_rlm: list[float] | None = None,
        root_value_rlm: float | None = None,
    ) -> int:
        self._rollout_count = 0
        search_state = state
        if self.simulator is not None and not (hasattr(state, "side_one") and hasattr(state, "side_two")):
            try:
                config = configs[0] if configs else None
                search_state = battle_to_poke_engine_state(state, config)
            except Exception as exc:
                LOGGER.warning("Failed to convert state for poke_engine MCTS; using root-prior fallback: %s", exc)
                self.simulator = None
                search_state = state
        root_prior, root_value = self.root_policy_value(search_state, root_prior_rlm, root_value_rlm)
        root = MCTSNode(root_prior)
        if self.simulator is None or self.time_budget <= 0:
            return max(range(len(root_prior)), key=lambda idx: root_prior[idx])
        deadline = time.monotonic() + self.time_budget
        configs = configs or [None]
        rollout = 0
        while time.monotonic() < deadline:
            config = configs[rollout % len(configs)]
            action = root.select_action(self.c_puct)
            value = self._rollout(search_state, action, config)
            root.update(action, value)
            rollout += 1
        return max(root.stats, key=lambda action: root.stats[action].visits)

    @torch.no_grad()
    def _rollout(self, state: Any, action: int, config: dict[str, Any] | None) -> float:
        assert self.simulator is not None
        next_state = self.simulator.next_state(state, action, config)
        self._rollout_count += 1
        if self.simulator.is_terminal(next_state):
            return self.simulator.terminal_value(next_state)
        _policy, value = self.model.policy_value(self._encode_for_model(next_state))
        return float(value[0].detach().cpu())


class PokeEngineSimulator:
    def __init__(self, model: PokeNet, vocab: Vocabulary, rng: random.Random | None = None) -> None:
        self.model = model
        self.vocab = vocab
        self.rng = rng or random.Random()
        try:
            import poke_engine  # type: ignore[import-not-found]
        except Exception:
            raise RuntimeError("poke_engine is not importable")
        self.poke_engine = poke_engine

    def is_terminal(self, state: Any) -> bool:
        return not self._has_healthy_pokemon(state.side_one) or not self._has_healthy_pokemon(state.side_two)

    def terminal_value(self, state: Any) -> float:
        our_alive = self._has_healthy_pokemon(state.side_one)
        opp_alive = self._has_healthy_pokemon(state.side_two)
        if our_alive and not opp_alive:
            return 1.0
        if opp_alive and not our_alive:
            return -1.0
        return 0.0

    def next_state(self, state: Any, action: int, config: dict[str, Any] | None = None) -> Any:
        side_one_move = self._action_to_engine_choice(state.side_one, action)
        side_two_action = self._sample_player_two_action(state)
        side_two_move = self._action_to_engine_choice(state.side_two, side_two_action)
        try:
            branches = self.poke_engine.generate_instructions(state, side_one_move, side_two_move)
        except Exception:
            side_one_move = self._first_move_choice(state.side_one)
            side_two_move = self._first_move_choice(state.side_two)
            branches = self.poke_engine.generate_instructions(state, side_one_move, side_two_move)
        if not branches:
            return type(state).from_string(state.to_string())
        weights = [max(0.0, float(getattr(branch, "percentage", 0.0))) for branch in branches]
        if sum(weights) <= 0:
            branch = self.rng.choice(branches)
        else:
            branch = self.rng.choices(branches, weights=weights, k=1)[0]
        return state.apply_instructions(branch)

    @staticmethod
    def _has_healthy_pokemon(side: Any) -> bool:
        return any(getattr(mon, "hp", 0) > 0 for mon in list(getattr(side, "pokemon", []) or []))

    def _sample_player_two_action(self, state: Any) -> int:
        encoded = encode_poke_engine_state(state, vocab=self.vocab, mirror=True)
        with torch.no_grad():
            logits, _value = self.model(encoded)
            distribution = torch.distributions.Categorical(logits=logits[0])
            return int(distribution.sample().detach().cpu())

    def _action_to_engine_choice(self, side: Any, action: int) -> str:
        if 0 <= action <= 3:
            return self._move_choice(side, action)
        if 4 <= action <= 8:
            switch = self._switch_choice(side, action - 4)
            return switch or self._first_move_choice(side)
        if 9 <= action <= 12:
            return self._move_choice(side, action - 9)
        if action == 13:
            return self._switch_choice(side, 0) or self._first_move_choice(side)
        return self._first_move_choice(side)

    def _move_choice(self, side: Any, move_index: int) -> str:
        active = self._active_pokemon(side)
        moves = list(getattr(active, "moves", []) or [])
        if 0 <= move_index < len(moves):
            move = moves[move_index]
            if not getattr(move, "disabled", False) and getattr(move, "id", "none") != "none":
                return str(move.id)
        return self._first_move_choice(side)

    def _first_move_choice(self, side: Any) -> str:
        active = self._active_pokemon(side)
        for move in list(getattr(active, "moves", []) or []):
            if not getattr(move, "disabled", False) and getattr(move, "id", "none") != "none":
                return str(move.id)
        return "none"

    def _switch_choice(self, side: Any, switch_index: int) -> str | None:
        active_index = int(getattr(side, "active_index", "0") or 0)
        switches = [
            mon for idx, mon in enumerate(list(getattr(side, "pokemon", []) or []))
            if idx != active_index and getattr(mon, "hp", 0) > 0
        ]
        if 0 <= switch_index < len(switches):
            return f"switch {switches[switch_index].id}"
        return None

    @staticmethod
    def _active_pokemon(side: Any) -> Any:
        index = int(getattr(side, "active_index", "0") or 0)
        return list(getattr(side, "pokemon", []) or [])[index]
