from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import Candidate, CandidateValidationError  # noqa: E402
from belief.action_likelihood_adapter import (  # noqa: E402
    FrozenR1CandidatePolicyLikelihoodAdapter,
    build_candidate_opponent_state,
    legal_action_mask,
)


@dataclass
class FakeMove:
    name: str


@dataclass
class FakePokemon:
    name: str
    base_species: str
    moves: list[FakeMove]
    item: str = "leftovers"
    ability: str = "pressure"
    tera_type: str = "water"
    lvl: int = 80


@dataclass
class FakeState:
    player_active_pokemon: FakePokemon
    opponent_active_pokemon: FakePokemon
    available_switches: list[FakePokemon]
    player_prev_move: FakeMove
    opponent_prev_move: FakeMove
    player_conditions: str = "player"
    opponent_conditions: str = "opponent"
    opponents_remaining: int = 6
    can_tera: bool = False
    opponent_teampreview: list[str] = field(default_factory=lambda: ["secret-a", "secret-b"])
    forced_switch: bool = False


def fake_metamon_modules():
    """Minimal lazy imports used by candidate-state construction and masks."""
    interface = types.ModuleType("metamon.interface")
    class UniversalMove:
        @staticmethod
        def from_ReplayMove(move):
            return FakeMove(move.name)

    interface.UniversalMove = UniversalMove
    interface.consistent_move_order = lambda moves: sorted(moves, key=lambda move: move.name)
    interface.consistent_pokemon_order = lambda pokemon: sorted(pokemon, key=lambda mon: mon.name)

    class Action:
        def __init__(self, action_idx):
            self.action_idx = action_idx

        @classmethod
        def maybe_valid_actions(cls, state):
            actions = [] if state.forced_switch else list(range(len(state.player_active_pokemon.moves)))
            if state.can_tera and not state.forced_switch:
                actions += list(range(9, 9 + len(state.player_active_pokemon.moves)))
            actions += list(range(4, 4 + len(state.available_switches)))
            return {cls(index) for index in actions}

    interface.UniversalAction = Action
    replay_state = types.ModuleType("metamon.backend.replay_parser.replay_state")
    replay_state.Move = lambda name, gen: FakeMove(name)
    return {"metamon.interface": interface, "metamon.backend.replay_parser.replay_state": replay_state}


class CandidateStateTests(unittest.TestCase):
    def setUp(self):
        self.observer_active = FakePokemon("observer", "observer", [FakeMove("tackle")])
        self.actor_active = FakePokemon("candidate-mon", "candidate-mon", [FakeMove("surf")])
        self.secret_switch = FakePokemon("unrevealed", "unrevealed", [FakeMove("secret")])
        self.revealed_switch = FakePokemon("revealed", "revealed", [FakeMove("growl")])
        self.state = FakeState(
            self.observer_active, self.actor_active, [self.secret_switch], FakeMove("a"), FakeMove("b")
        )

    def test_candidate_set_mutates_policy_active_without_mutating_source_or_leaking_team(self):
        candidate = Candidate("set", public_data={
            "speciesId": "candidate-mon", "moves": ["thunderbolt"], "item": "choice scarf",
            "ability": "static", "teraType": "electric", "level": 91,
        })
        with patch.dict(sys.modules, fake_metamon_modules()):
            built = build_candidate_opponent_state(
                self.state, candidate, [self.revealed_switch], acting_can_tera=True, public_opponent_remaining=4
            )
        active = built.player_active_pokemon
        self.assertEqual([move.name for move in active.moves], ["thunderbolt"])
        self.assertEqual((active.item, active.ability, active.tera_type, active.lvl), ("choicescarf", "static", "electric", 91))
        self.assertEqual([move.name for move in self.actor_active.moves], ["surf"])
        self.assertEqual([mon.name for mon in built.available_switches], ["revealed"])
        self.assertNotIn("unrevealed", built.opponent_teampreview)
        self.assertEqual(built.opponent_teampreview, [])

    def test_candidate_move_changes_legal_mask(self):
        set_fields = {"item": "leftovers", "ability": "pressure", "teraType": "water", "level": 80}
        one_move = Candidate("one", public_data={"moves": ["surf"], **set_fields})
        two_moves = Candidate("two", public_data={"moves": ["surf", "thunderbolt"], **set_fields})
        with patch.dict(sys.modules, fake_metamon_modules()):
            state_one = build_candidate_opponent_state(self.state, one_move, [], acting_can_tera=False, public_opponent_remaining=6)
            state_two = build_candidate_opponent_state(self.state, two_moves, [], acting_can_tera=False, public_opponent_remaining=6)
            self.assertTrue(legal_action_mask(state_one)[1])
            self.assertFalse(legal_action_mask(state_two)[1])

    def test_adapter_returns_zero_for_candidate_without_observed_action(self):
        candidate = Candidate("set", public_data={"moves": ["surf"]})
        adapter = FrozenR1CandidatePolicyLikelihoodAdapter(lambda states, masks: [[1.0] + [0.0] * 12])
        # The state factory is outside this unit's scope; provide states directly
        # to make the rejection check model-free and candidate-specific.
        built = replace(self.state, player_active_pokemon=self.actor_active, available_switches=[])
        with patch("belief.action_likelihood_adapter.candidate_state_from_replay", return_value=built), patch.dict(sys.modules, fake_metamon_modules()):
            likelihoods = adapter.action_likelihoods(
                {"replay_state": object(), "acting_can_tera": False, "public_opponent_remaining": 6},
                [candidate], "move thunderbolt",
            )
        self.assertEqual(likelihoods["set"], 0.0)


if __name__ == "__main__":
    unittest.main()
