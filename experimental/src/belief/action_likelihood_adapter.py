"""Candidate-conditioned frozen-r1 action likelihoods for offline Randbats.

This module intentionally constructs one opponent-view policy state per active
set.  It never uses ``PriorSession.compute_opponent_priors``: that method is a
public-only heuristic and is not P(action | candidate).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from belief.action_conditioned_randbats import Candidate, CandidateValidationError


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _candidate_value(candidate: Candidate, *keys: str) -> Any:
    data = candidate.public_data or {}
    for key in keys:
        if key in data:
            return data[key]
    return None


def _candidate_active(source: Any, candidate: Candidate) -> Any:
    """Copy a revealed active and replace only its generator-set attributes."""
    from metamon.interface import UniversalMove
    from metamon.backend.replay_parser.replay_state import Move as ReplayMove

    moves = _candidate_value(candidate, "moves")
    if not isinstance(moves, Sequence) or isinstance(moves, (str, bytes)) or not moves:
        raise CandidateValidationError(f"candidate {candidate.candidate_id} requires a non-empty moves list")
    if len(moves) > 4 or not all(isinstance(move, str) and move for move in moves):
        raise CandidateValidationError(f"candidate {candidate.candidate_id} has invalid moves")
    source_species = _norm(source.base_species or source.name)
    species = _candidate_value(candidate, "speciesId", "species", "base_species")
    if species is not None and _norm(species) != source_species:
        raise CandidateValidationError(f"candidate {candidate.candidate_id} does not match the active species")

    source_moves = {_norm(move.name): move for move in source.moves}
    patched_moves = []
    for name in moves:
        old = source_moves.get(_norm(name))
        if old is not None:
            patched_moves.append(copy.deepcopy(old))
            continue
        # ReplayMove gives the candidate move its real dex features before the
        # observation space tokenizes it. Its PP is the standard initial PP.
        replay_move = ReplayMove(name=name, gen=9)
        patched_moves.append(UniversalMove.from_ReplayMove(replay_move))

    item = _candidate_value(candidate, "item")
    ability = _candidate_value(candidate, "ability")
    tera_type = _candidate_value(candidate, "teraType", "tera_type")
    level = _candidate_value(candidate, "level", "lvl")
    if item is None or ability is None or tera_type is None or level is None:
        raise CandidateValidationError(
            f"candidate {candidate.candidate_id} requires item, ability, tera type, and level"
        )
    changes: dict[str, Any] = {"moves": patched_moves}
    changes["item"] = _norm(item)
    changes["ability"] = _norm(ability)
    changes["tera_type"] = _norm(tera_type)
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 100:
        raise CandidateValidationError(f"candidate {candidate.candidate_id} has invalid level")
    changes["lvl"] = level
    return replace(source, **changes)


def build_candidate_opponent_state(
    observer_state: Any,
    candidate: Candidate,
    revealed_acting_switches: Sequence[Any],
    *,
    acting_can_tera: bool,
    public_opponent_remaining: int,
) -> Any:
    """Flip an observer UniversalState without carrying its unrevealed party.

    ``revealed_acting_switches`` must contain only the acting opponent's
    publicly revealed, non-active Pokemon.  In particular, this function does
    not reuse ``observer_state.available_switches``: those are on the other
    side and would leak an unrevealed team into the policy input.
    """
    if isinstance(public_opponent_remaining, bool) or not isinstance(public_opponent_remaining, int) or not 0 <= public_opponent_remaining <= 6:
        raise CandidateValidationError("public_opponent_remaining must be an integer in [0, 6]")
    if not isinstance(acting_can_tera, bool):
        raise CandidateValidationError("acting_can_tera must be a bool")
    return replace(
        observer_state,
        player_active_pokemon=_candidate_active(observer_state.opponent_active_pokemon, candidate),
        opponent_active_pokemon=copy.deepcopy(observer_state.player_active_pokemon),
        available_switches=copy.deepcopy(list(revealed_acting_switches)),
        player_prev_move=copy.deepcopy(observer_state.opponent_prev_move),
        opponent_prev_move=copy.deepcopy(observer_state.player_prev_move),
        player_conditions=observer_state.opponent_conditions,
        opponent_conditions=observer_state.player_conditions,
        opponents_remaining=public_opponent_remaining,
        can_tera=acting_can_tera,
        # The replay POV does not supply the other side's preview here.
        opponent_teampreview=[],
    )


def candidate_state_from_replay(
    replay_state: Any,
    candidate: Candidate,
    *,
    acting_can_tera: bool,
    public_opponent_remaining: int,
) -> Any:
    """Build an acting-opponent UniversalState from a Metamon ReplayState."""
    from metamon.interface import UniversalPokemon, UniversalState
    from metamon.backend.replay_parser.pe_datatypes import PEStatus

    observer_state = UniversalState.from_ReplayState(replay_state)
    active_id = replay_state.opponent_active_pokemon.unique_id
    revealed_switches = [
        UniversalPokemon.from_ReplayPokemon(pokemon)
        for pokemon in replay_state.opponent_team
        if pokemon is not None and pokemon.unique_id != active_id and pokemon.status != PEStatus.FNT
    ]
    return build_candidate_opponent_state(
        observer_state,
        candidate,
        revealed_switches,
        acting_can_tera=acting_can_tera,
        public_opponent_remaining=public_opponent_remaining,
    )


def legal_action_mask(state: Any) -> list[bool]:
    """Return the r1 13-action illegal mask from the candidate-specific state."""
    from metamon.interface import UniversalAction

    illegal = [True] * 13
    for action in UniversalAction.maybe_valid_actions(state):
        illegal[action.action_idx] = False
    return illegal


def action_index(state: Any, observed_action: str) -> int:
    """Map canonical ``move X``, ``move X-tera``, or ``switch X`` to r1 index."""
    if not isinstance(observed_action, str) or not observed_action:
        raise CandidateValidationError("observed_action must be a non-empty canonical action string")
    from metamon.interface import consistent_move_order, consistent_pokemon_order

    action = observed_action.strip().lower()
    tera = action.startswith("move ") and action.endswith("-tera")
    if action.startswith("move "):
        name = action[5:-5] if tera else action[5:]
        for index, move in enumerate(consistent_move_order(state.player_active_pokemon.moves)):
            if _norm(move.name) == _norm(name):
                return index + (9 if tera else 0)
    elif action.startswith("switch "):
        name = action[7:]
        for index, pokemon in enumerate(consistent_pokemon_order(state.available_switches)):
            if _norm(pokemon.name) == _norm(name) or _norm(pokemon.base_species) == _norm(name):
                return 4 + index
    raise CandidateValidationError(f"observed action {observed_action!r} is unavailable for candidate state")


PolicyBatch = Callable[[Sequence[Any], Sequence[Sequence[bool]]], Sequence[Sequence[float]]]


class FrozenR1CandidatePolicyLikelihoodAdapter:
    """Evaluate the frozen r1 policy once for each candidate, in one batch.

    ``policy_batch`` is injectable so benchmark producers can test state and
    mask construction without importing torch or loading a checkpoint.  The
    production factory below supplies the frozen transformer implementation.
    """

    def __init__(self, policy_batch: PolicyBatch):
        self.policy_batch = policy_batch

    def action_likelihoods(
        self, public_state: Mapping[str, Any], candidates: Sequence[Candidate], observed_action: str
    ) -> Mapping[str, float]:
        replay_state = public_state.get("replay_state")
        if replay_state is None:
            raise CandidateValidationError("public_state requires replay_state")
        states = [candidate_state_from_replay(
            replay_state, candidate,
            acting_can_tera=public_state.get("acting_can_tera"),
            public_opponent_remaining=public_state.get("public_opponent_remaining"),
        ) for candidate in candidates]
        masks = [legal_action_mask(state) for state in states]
        indices: list[int | None] = []
        for candidate, state in zip(candidates, states):
            try:
                indices.append(action_index(state, observed_action))
            except CandidateValidationError:
                indices.append(None)
        eligible = [index for index, (mask, action) in enumerate(zip(masks, indices)) if action is not None and not mask[action]]
        probabilities = self.policy_batch([states[index] for index in eligible], [masks[index] for index in eligible]) if eligible else []
        if len(probabilities) != len(eligible):
            raise CandidateValidationError("policy batch result does not match candidate count")
        likelihoods: dict[str, float] = {}
        for candidate in candidates:
            likelihoods[candidate.candidate_id] = 0.0
        for candidate_index, probability in zip(eligible, probabilities):
            candidate = candidates[candidate_index]
            action = indices[candidate_index]
            assert action is not None
            if len(probability) != 13:
                raise CandidateValidationError("policy batch result must contain 13 action probabilities")
            value = float(probability[action])
            if not math.isfinite(value) or value <= 0.0:
                raise CandidateValidationError(f"non-positive frozen-policy likelihood for {candidate.candidate_id}")
            likelihoods[candidate.candidate_id] = value
        return likelihoods


def frozen_r1_policy_batch(agent: Any, observation_space: Any, device: Any) -> PolicyBatch:
    """Create the batched two-step r1 inference callable used by prior_server."""
    def infer(states: Sequence[Any], masks: Sequence[Sequence[bool]]) -> Sequence[Sequence[float]]:
        import numpy as np
        import torch

        observations = [dict(observation_space.state_to_obs(state)) for state in states]
        text_now = np.stack([obs["text_tokens"] for obs in observations])
        numbers_now = np.stack([obs["numbers"] for obs in observations])
        illegal_now = np.asarray(masks, dtype=bool)
        batch = {
            "text_tokens": torch.tensor(np.stack([np.zeros_like(text_now), text_now], axis=1), dtype=torch.int32, device=device),
            "numbers": torch.nan_to_num(torch.tensor(np.stack([np.zeros_like(numbers_now), numbers_now], axis=1), dtype=torch.float32, device=device)),
            "illegal_actions": torch.tensor(np.stack([np.ones_like(illegal_now), illegal_now], axis=1), device=device),
        }
        rl2s = torch.zeros((len(states), 2, 14), device=device)
        time_idxs = torch.arange(2, device=device).long().reshape(1, 2, 1).expand(len(states), -1, -1)
        with torch.no_grad():
            embeddings, _ = agent.get_state_embedding(obs=batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None)
            distributions = agent.actor(embeddings, straight_from_obs={key: batch[key][:, :embeddings.shape[1]] for key in agent.pass_obs_keys_to_actor})
            probs = distributions.probs[:, -1, -1, :].cpu().numpy()
        probs *= ~illegal_now
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.tolist()
    return infer


def make_frozen_r1_adapter(agent: Any, observation_space: Any, device: Any) -> FrozenR1CandidatePolicyLikelihoodAdapter:
    """Integration contract for a benchmark producer with a loaded frozen r1.

    Call ``adapter.action_likelihoods({"replay_state": replay_state,
    "acting_can_tera": bool, "public_opponent_remaining": int}, candidates,
    observed_action)``.  The two scalar state fields must come from public
    protocol information, not replay-finalized team data.
    """
    return FrozenR1CandidatePolicyLikelihoodAdapter(
        frozen_r1_policy_batch(agent, observation_space, device)
    )
