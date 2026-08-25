import unittest

from srcs.metagross.adaptive_budget_development import analyze


class AdaptiveBudgetDevelopmentTest(unittest.TestCase):
    def test_no_threshold_passes_when_saving_and_gain_trade_off(self):
        repeats = [
            {"visit_policy": {"a": 0.9, "b": 0.1}},
            {"visit_policy": {"a": 0.9, "b": 0.1}},
        ]
        root = {
            "identity": ["x", "p1", 0],
            "attribution": "finite_search_budget",
            "budgets": {
                "5000": {"repeats": repeats, "visit": {"teacher_deltas": [0.0, 0.0]}},
                "20000": {"repeats": repeats, "visit": {"teacher_deltas": [0.0, 0.0]}},
                "80000": {
                    "repeats": [{"visit_policy": {"b": 0.9, "a": 0.1}}] * 2,
                    "visit": {"teacher_deltas": [0.1, 0.1]},
                },
            },
        }
        report = analyze({"roots": [root]})
        self.assertFalse(report["decision"]["gate_passed"])
        self.assertEqual(report["decision"]["next"], "information_state_value_dataset")


if __name__ == "__main__":
    unittest.main()
