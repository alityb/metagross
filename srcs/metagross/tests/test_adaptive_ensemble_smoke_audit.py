import unittest

from srcs.metagross.adaptive_ensemble_smoke_audit import _expected_modal_batch_sizes


class AdaptiveEnsembleSmokeAuditTest(unittest.TestCase):
    def test_expected_modal_batch_sizes(self):
        self.assertEqual(_expected_modal_batch_sizes(1), [1])
        self.assertEqual(_expected_modal_batch_sizes(16), [16] * 16)
        self.assertEqual(
            _expected_modal_batch_sizes(17), [16] * 16 + [1]
        )
        self.assertEqual(
            _expected_modal_batch_sizes(40), [16] * 32 + [8] * 8
        )
        self.assertEqual(_expected_modal_batch_sizes(64), [16] * 64)

    def test_search_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _expected_modal_batch_sizes(0)


if __name__ == "__main__":
    unittest.main()
