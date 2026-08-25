import math
import unittest

from srcs.metagross.stage2_loss_attribution import _action_class, _aggregate, _entropy


class Stage2LossAttributionTest(unittest.TestCase):
    def test_action_classes(self):
        self.assertEqual(_action_class("switch pikachu"), "switch")
        self.assertEqual(_action_class("thunderbolt-tera"), "tera_move")
        self.assertEqual(_action_class("thunderbolt"), "ordinary_move")

    def test_entropy(self):
        entropy, normalized = _entropy(
            [
                {"probability": 0.5},
                {"probability": 0.5},
            ]
        )
        self.assertAlmostEqual(entropy, math.log(2))
        self.assertAlmostEqual(normalized, 1.0)

    def test_aggregate_distinguishes_sampled_and_executed_actions(self):
        decision = {
            "baseline_agreement": False,
            "sampled_is_policy_argmax": False,
            "entropy": 0.5,
            "normalized_entropy": 0.25,
            "top_probability": 0.8,
            "sampled_probability": 0.2,
            "sampled_counterfactual_value": 0.6,
            "baseline_counterfactual_value": 0.5,
            "counterfactual_delta": 0.1,
            "baseline_class": "ordinary_move",
            "sampled_class": "tera_move",
            "executed_class": "switch",
            "selection_class": "deterministic_correction",
            "selection_reason": "semantic_no_progress_losing_stall",
        }
        aggregate = _aggregate([decision])
        self.assertEqual(aggregate["root_divergences"], 1)
        self.assertEqual(aggregate["sampled_non_argmax"], 1)
        self.assertEqual(aggregate["actions"]["sampled_tera"], 1)
        self.assertEqual(aggregate["actions"]["executed_switches"], 1)
        self.assertEqual(aggregate["deterministic_corrections"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
