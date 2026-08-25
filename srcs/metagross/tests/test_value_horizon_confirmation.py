import unittest
from unittest import mock

from srcs.metagross.value_horizon_confirmation import _bootstrap_mean, _seed


class ValueHorizonConfirmationTest(unittest.TestCase):
    def test_confirmation_seed_is_deterministic(self):
        identity = ("battle", "p2", 9)
        self.assertEqual(_seed(identity, 0, 3), _seed(identity, 0, 3))
        self.assertNotEqual(_seed(identity, 0, 3), _seed(identity, 1, 3))

    def test_bootstrap_is_deterministic_and_ordered(self):
        first = _bootstrap_mean([-0.1, 0.0, 0.1, 0.2])
        second = _bootstrap_mean([-0.1, 0.0, 0.1, 0.2])
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])


if __name__ == "__main__":
    unittest.main()
