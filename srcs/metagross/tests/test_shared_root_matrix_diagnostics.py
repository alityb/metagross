import unittest

from srcs.metagross.shared_root_matrix_diagnostics import (
    _policy_argmax,
    _select,
    _strategy_scores,
)


class SharedRootMatrixDiagnosticsTest(unittest.TestCase):
    def test_tie_breaking_matches_controller_order(self):
        self.assertEqual(_select({"b": 0.5, "a": 0.5}), "a")
        self.assertEqual(
            _policy_argmax(
                [
                    {"action": "b", "probability": 0.5, "counterfactual_value": 0.6},
                    {"action": "a", "probability": 0.5, "counterfactual_value": 0.6},
                ]
            ),
            "a",
        )

    def test_strategy_scores_use_weights_priors_and_worst_case(self):
        result = {
            "policy": [
                {"action": "a", "counterfactual_value": 0.4},
                {"action": "b", "counterfactual_value": 0.5},
            ],
            "replay_capture": {
                "own_action_support": ["a", "b"],
                "canonical_particles": [
                    {
                        "normalized_weight": 1.0,
                        "normalized_opponent_prior": [0.75, 0.25],
                        "payoff_matrix": [[1.0, 0.0], [0.6, 0.6]],
                    }
                ],
            },
        }
        scores = _strategy_scores(result)
        self.assertEqual(scores["opponent_prior_expected"], {"a": 0.75, "b": 0.6})
        self.assertEqual(scores["worst_case_endpoint"], {"a": 0.0, "b": 0.6})
        self.assertAlmostEqual(scores["bounded_robust_0.50"]["a"], 0.375)
        self.assertAlmostEqual(scores["bounded_robust_0.50"]["b"], 0.6)

    def test_prior_treatments_are_unavailable_without_all_priors(self):
        result = {
            "policy": [{"action": "a", "counterfactual_value": 0.5}],
            "replay_capture": {
                "own_action_support": ["a"],
                "canonical_particles": [
                    {
                        "normalized_weight": 1.0,
                        "normalized_opponent_prior": None,
                        "payoff_matrix": [[0.5]],
                    }
                ],
            },
        }
        scores = _strategy_scores(result)
        self.assertIsNone(scores["opponent_prior_expected"])
        self.assertIsNone(scores["bounded_robust_0.25"])


if __name__ == "__main__":
    unittest.main()
