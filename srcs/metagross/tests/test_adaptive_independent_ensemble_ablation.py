import unittest

from srcs.metagross.adaptive_independent_ensemble_ablation import (
    adaptive_repeat_count,
    expected_teacher_mass,
    production_selection_distribution,
)
from srcs.metagross.independent_mcts_ensemble_ablation import _ensemble_action


class AdaptiveIndependentEnsembleAblationTest(unittest.TestCase):
    def test_repeat_count_uses_all_available_wire_capacity(self):
        self.assertEqual(adaptive_repeat_count(16), 3)
        self.assertEqual(adaptive_repeat_count(32), 2)
        self.assertEqual(adaptive_repeat_count(64), 1)
        with self.assertRaisesRegex(ValueError, "wire contract"):
            adaptive_repeat_count(65)

    def test_selection_matches_production_threshold_distribution(self):
        schedule = {
            "worlds": [{
                "sample_weight": 1.0,
                "treatments": {"S-B": [
                    {"result": {"total_visits": 100, "side_one": [
                        {"action": "b", "visits": 40},
                        {"action": "a", "visits": 40},
                        {"action": "c", "visits": 20},
                    ]}}
                ]},
            }]
        }
        action, mass = _ensemble_action(schedule, 1)
        self.assertEqual(action, "a")
        self.assertEqual(mass, {"a": 0.4, "b": 0.4, "c": 0.2})
        self.assertEqual(
            production_selection_distribution(schedule, 1),
            {"a": 0.5, "b": 0.5},
        )

    def test_expected_teacher_mass_uses_full_selection_distribution(self):
        schedule = {
            "worlds": [{
                "sample_weight": 1.0,
                "treatments": {"S-B": [{"result": {
                    "total_visits": 10,
                    "side_one": [
                        {"action": "a", "visits": 5},
                        {"action": "b", "visits": 5},
                    ],
                }}]},
            }],
            "aggregate_treatments": {"S-4B": [{
                "side_one_policy": [
                    {"action": "a", "probability": 0.8},
                    {"action": "b", "probability": 0.2},
                ],
            }]},
        }
        distribution = production_selection_distribution(schedule, 1)
        self.assertAlmostEqual(expected_teacher_mass(schedule, distribution), 0.5)


if __name__ == "__main__":
    unittest.main()
