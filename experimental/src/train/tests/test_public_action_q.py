from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.public_action_q import FEATURE_COUNT, action_features  # noqa: E402


def pokemon(identity: str, types: tuple[str, str], *, hp: int = 100, hidden: int = 0):
    return SimpleNamespace(
        id=identity,
        types=types,
        hp=hp,
        maxhp=100,
        status="NONE",
        attack=200 + hidden,
        defense=200 + hidden,
        special_attack=200 + hidden,
        special_defense=200 + hidden,
        speed=200 + hidden,
        level=80,
        moves=[SimpleNamespace(id="TACKLE", pp=20)],
        item=f"private-{hidden}",
        ability=f"private-{hidden}",
    )


def state(private_hidden: int = 0):
    own = [pokemon("OWN", ("NORMAL", "TYPELESS")), pokemon("BENCH", ("WATER", "TYPELESS"))]
    opponent = [
        pokemon("PUBLICFOE", ("FIRE", "TYPELESS")),
        pokemon(f"PRIVATE{private_hidden}", ("DRAGON", "TYPELESS"), hidden=private_hidden),
    ]
    return SimpleNamespace(
        side_one=SimpleNamespace(active_index="0", pokemon=own),
        side_two=SimpleNamespace(active_index="0", pokemon=opponent),
    )


class FakeEngine:
    @staticmethod
    def compute_public_value_features(_state):
        return [0.0] * 18


MOVES = {
    "tackle": {
        "accuracy": 100,
        "basePower": 40,
        "category": "physical",
        "flags": {"contact": 1},
        "pp": 35,
        "priority": 0,
        "type": "normal",
    }
}


class PublicActionQFeatureTests(unittest.TestCase):
    def test_private_opponent_reserve_does_not_change_features(self):
        first = action_features(state(1), "tackle", poke_engine=FakeEngine, move_database=MOVES)
        second = action_features(state(99), "tackle", poke_engine=FakeEngine, move_database=MOVES)
        self.assertEqual(first.shape, (FEATURE_COUNT,))
        np.testing.assert_array_equal(first, second)

    def test_action_identity_and_public_matchup_change_features(self):
        move = action_features(state(), "tackle", poke_engine=FakeEngine, move_database=MOVES)
        switch = action_features(state(), "switch bench", poke_engine=FakeEngine, move_database=MOVES)
        self.assertFalse(np.array_equal(move, switch))
        changed = state()
        changed.side_two.pokemon[0].id = "OTHERPUBLICFOE"
        matchup = action_features(changed, "tackle", poke_engine=FakeEngine, move_database=MOVES)
        self.assertFalse(np.array_equal(move, matchup))


if __name__ == "__main__":
    unittest.main()
