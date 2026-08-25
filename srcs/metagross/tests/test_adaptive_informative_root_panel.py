import unittest

from srcs.metagross.adaptive_informative_root_panel import (
    baseline_distribution,
    support_metrics,
)


class AdaptiveInformativeRootPanelTest(unittest.TestCase):
    def test_baseline_distribution_uses_only_first_repeat(self):
        search = {
            "remote_search": {"worlds": 2},
            "samples": [
                {
                    "index": 0,
                    "sample_chance": 0.25,
                    "result": {
                        "total_visits": 10,
                        "side_one": [
                            {"move_choice": "a", "visits": 6},
                            {"move_choice": "b", "visits": 4},
                        ],
                    },
                },
                {
                    "index": 1,
                    "sample_chance": 0.25,
                    "result": {
                        "total_visits": 10,
                        "side_one": [
                            {"move_choice": "a", "visits": 4},
                            {"move_choice": "b", "visits": 6},
                        ],
                    },
                },
                {
                    "index": 2,
                    "sample_chance": 0.25,
                    "result": {
                        "total_visits": 10,
                        "side_one": [{"move_choice": "c", "visits": 10}],
                    },
                },
            ],
        }
        self.assertEqual(baseline_distribution(search), {"a": 0.5, "b": 0.5})

    def test_support_metrics_collapse_duplicate_states(self):
        metrics = support_metrics(["a", "a", "b"], [0.25, 0.25, 0.5])
        self.assertEqual(metrics["nominal_worlds"], 3)
        self.assertEqual(metrics["unique_worlds"], 2)
        self.assertAlmostEqual(metrics["state_ess"], 2.0)


if __name__ == "__main__":
    unittest.main()
