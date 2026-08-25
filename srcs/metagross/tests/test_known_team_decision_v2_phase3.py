from __future__ import annotations

import unittest

from srcs.metagross.known_team_decision_v2 import canonical_json
from srcs.metagross.known_team_decision_v2_phase3 import (
    ParticleGame,
    allocate_exact_compute,
    mixture_weights,
    require_phase3_authorization,
    solve_multistate_game,
)


def game(identity: str, weight: float, matrix: list[list[float]]) -> ParticleGame:
    return ParticleGame(
        identity,
        weight,
        {
            f"a{row}": {f"o{column}": value for column, value in enumerate(values)}
            for row, values in enumerate(matrix)
        },
    )


class KnownTeamDecisionV2Phase3Test(unittest.TestCase):
    def test_identical_particle_matches_ordinary_single_game(self):
        single = solve_multistate_game([game("only", 1.0, [[1, -1], [-1, 1]])], rounds=4000)
        duplicate = solve_multistate_game(
            [game("only", 0.5, [[1, -1], [-1, 1]]), game("only", 0.5, [[1, -1], [-1, 1]])],
            rounds=4000,
        )
        self.assertEqual(canonical_json(single), canonical_json(duplicate))

    def test_alpha_zero_exactly_reproduces_current_weights(self):
        current = [0.7, 0.2, 0.1]
        self.assertEqual(mixture_weights(current, [0.1, 0.2, 0.7], 0.0), current)

    def test_illegal_actions_cannot_acquire_mass(self):
        result = solve_multistate_game(
            [game("p", 1.0, [[0.0], [1.0], [2.0]])],
            allowed_actions={"a0", "a1"},
            rounds=100,
        )
        self.assertNotIn("a2", result["policy"])

    def test_matching_pennies_does_not_collapse_to_pure_action(self):
        result = solve_multistate_game([game("p", 1.0, [[1, -1], [-1, 1]])], rounds=8000)
        self.assertTrue(all(0.45 < value < 0.55 for value in result["policy"].values()))

    def test_rps_does_not_collapse_to_pure_action(self):
        result = solve_multistate_game(
            [game("p", 1.0, [[0, -1, 1], [1, 0, -1], [-1, 1, 0]])],
            rounds=8000,
        )
        self.assertTrue(all(0.28 < value < 0.39 for value in result["policy"].values()))

    def test_particle_reordering_is_byte_identical(self):
        particles = [
            game("b", 0.4, [[0.8, 0.2], [0.1, 0.9]]),
            game("a", 0.6, [[0.2, 0.7], [0.9, 0.1]]),
        ]
        left = solve_multistate_game(particles, rounds=500)
        right = solve_multistate_game(list(reversed(particles)), rounds=500)
        self.assertEqual(canonical_json(left), canonical_json(right))

    def test_player_action_policy_is_shared_across_worlds(self):
        result = solve_multistate_game(
            [game("a", 0.5, [[1], [0]]), game("b", 0.5, [[0], [1]])], rounds=1000
        )
        self.assertEqual(set(result["policy"]), {"a0", "a1"})
        self.assertNotIn("player_policy", next(iter(result["opponent_policies"].values())))

    def test_treatment_blind_compute_accounting_is_exact(self):
        allocation = allocate_exact_compute(20_003, 12)
        self.assertEqual(sum(allocation), 20_003)
        self.assertLessEqual(max(allocation) - min(allocation), 1)

    def test_phase3_requires_exact_frozen_authorization(self):
        with self.assertRaises(RuntimeError):
            require_phase3_authorization({"summary": {"phase3_authorized": False}})
        require_phase3_authorization(
            {"summary": {
                "phase3_authorized": True,
                "decision": "authorize_phase3_maple_vertical_slice",
            }}
        )


if __name__ == "__main__":
    unittest.main()
