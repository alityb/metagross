import unittest

from srcs.metagross.independent_mcts_ensemble_ablation import _ensemble_action


class IndependentMctsEnsembleAblationTest(unittest.TestCase):
    def test_ensemble_aggregates_normalized_visits_and_world_weights(self):
        def result(a, b):
            return {
                "total_visits": a + b,
                "side_one": [
                    {"action": "a", "visits": a},
                    {"action": "b", "visits": b},
                ],
            }

        schedule = {
            "worlds": [
                {
                    "sample_weight": 0.75,
                    "treatments": {"S-B": [{"result": result(8, 2)}]},
                },
                {
                    "sample_weight": 0.25,
                    "treatments": {"S-B": [{"result": result(0, 10)}]},
                },
            ]
        }
        action, mass = _ensemble_action(schedule, 1)
        self.assertEqual(action, "a")
        self.assertAlmostEqual(mass["a"], 0.6)
        self.assertAlmostEqual(mass["b"], 0.4)


if __name__ == "__main__":
    unittest.main()
