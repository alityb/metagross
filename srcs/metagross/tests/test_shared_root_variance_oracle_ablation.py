import unittest

from srcs.metagross.shared_root_variance_oracle_ablation import (
    _expected_action,
    _smoothed,
    solve_rm_plus,
)


class SharedRootVarianceOracleAblationTest(unittest.TestCase):
    def test_matching_pennies_converges_to_uniform(self):
        result = solve_rm_plus(
            [1.0],
            [[[1.0, 0.0], [0.0, 1.0]]],
            10_000,
            None,
            [None],
            0.0,
        )
        self.assertAlmostEqual(result["player_policy"][0], 0.5, places=3)
        self.assertAlmostEqual(result["opponent_policies"][0][0], 0.5, places=3)

    def test_dominant_action_is_selected(self):
        result = solve_rm_plus(
            [1.0],
            [[[1.0, 1.0], [0.0, 0.0]]],
            1_000,
            None,
            [None],
            0.0,
        )
        self.assertGreater(result["player_policy"][0], 0.99)

    def test_zero_weight_world_does_not_change_policy(self):
        result = solve_rm_plus(
            [1.0, 0.0],
            [[[1.0], [0.0]], [[0.0], [1.0]]],
            1_000,
            None,
            [None, None],
            0.0,
        )
        self.assertGreater(result["player_policy"][0], 0.99)

    def test_expected_action_uses_opponent_policy(self):
        action, scores = _expected_action(
            ["a", "b"],
            [1.0],
            [[[1.0, 0.0], [0.4, 0.4]]],
            [[0.75, 0.25]],
        )
        self.assertEqual(action, "a")
        self.assertEqual(scores, [0.75, 0.4])

    def test_smoothing_adds_uniform_mass(self):
        self.assertEqual(_smoothed([1.0, 0.0], 0.25), [0.875, 0.125])


if __name__ == "__main__":
    unittest.main()
