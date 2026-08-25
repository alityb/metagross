import unittest

from srcs.metagross.shared_root_prior_risk_filter import FEATURES, _applies, _select_rule


class SharedRootPriorRiskFilterTest(unittest.TestCase):
    def test_rule_does_not_apply_without_available_delta(self):
        rule = {"feature": "raw_prior_action_count", "direction": "at_least", "threshold": 2}
        self.assertFalse(_applies({"raw_prior_action_count": 3, "delta": None}, rule))

    def test_training_rejects_gain_with_severe_regression(self):
        rows = [
            {
                "identity": {"root": 1},
                "weight": 1.0,
                "delta": 0.3,
                **{feature: 1.0 for feature in FEATURES},
            },
            {
                "identity": {"root": 2},
                "weight": 1.0,
                "delta": -0.3,
                **{feature: 1.0 for feature in FEATURES},
            },
        ]
        rule, metrics = _select_rule(rows)
        self.assertIsNone(rule)
        self.assertEqual(metrics["prior_selected_roots"], 0)


if __name__ == "__main__":
    unittest.main()
