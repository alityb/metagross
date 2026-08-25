import math
import unittest

from srcs.metagross.shared_root_offline_ablation import (
    _argmax,
    _percentile,
    _teacher_alignment,
    _tv,
)


class SharedRootOfflineAblationTest(unittest.TestCase):
    def test_policy_metrics(self):
        left = {"a": 0.75, "b": 0.25}
        right = {"a": 0.25, "b": 0.75}
        self.assertEqual(_argmax(left), "a")
        self.assertAlmostEqual(_tv(left, right), 0.5)
        self.assertAlmostEqual(_teacher_alignment(left, right), 0.375)

    def test_nearest_rank_percentile(self):
        self.assertIsNone(_percentile([], 0.95))
        self.assertEqual(_percentile([1.0, 4.0, 2.0, 3.0], 0.50), 2.0)
        self.assertEqual(_percentile([1.0, 4.0, 2.0, 3.0], 0.95), 4.0)

    def test_total_variation_handles_different_supports(self):
        self.assertTrue(math.isclose(_tv({"a": 1.0}, {"b": 1.0}), 1.0))


if __name__ == "__main__":
    unittest.main()
