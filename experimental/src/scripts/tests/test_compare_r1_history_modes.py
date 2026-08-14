from __future__ import annotations

import unittest

import numpy as np

from experimental.src.scripts import compare_r1_history_modes


class CompareHistoryModesTests(unittest.TestCase):
    def test_distribution_metrics(self):
        metrics = compare_r1_history_modes._distribution_metrics(
            np.asarray([0.75, 0.25]), np.asarray([0.25, 0.75])
        )

        self.assertAlmostEqual(metrics["total_variation"], 0.5)
        self.assertGreater(metrics["jensen_shannon"], 0.0)
        self.assertEqual(metrics["top1_changed"], 1)

    def test_history_bins(self):
        self.assertEqual(
            [compare_r1_history_modes._history_bin(value) for value in (1, 2, 5, 6, 10, 11, 20, 21)],
            ["1", "2-5", "2-5", "6-10", "6-10", "11-20", "11-20", "21+"],
        )


if __name__ == "__main__":
    unittest.main()
