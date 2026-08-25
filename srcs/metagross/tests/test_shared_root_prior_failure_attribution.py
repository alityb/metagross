import unittest

from srcs.metagross.shared_root_prior_failure_attribution import (
    _correlations,
    _pearson,
    _ranks,
)


class SharedRootPriorFailureAttributionTest(unittest.TestCase):
    def test_average_tie_ranks(self):
        self.assertEqual(_ranks([3.0, 1.0, 1.0, 2.0]), [3.0, 0.5, 0.5, 2.0])

    def test_pearson_handles_linear_and_constant_inputs(self):
        self.assertAlmostEqual(_pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertIsNone(_pearson([1.0, 1.0], [1.0, 2.0]))

    def test_correlations_drop_missing_feature_rows(self):
        result = _correlations(
            [{"feature": 1.0, "delta": 2.0}, {"feature": None, "delta": 3.0}],
            "delta",
            ["feature"],
        )
        self.assertEqual(result["feature"]["count"], 1)
        self.assertIsNone(result["feature"]["pearson"])


if __name__ == "__main__":
    unittest.main()
