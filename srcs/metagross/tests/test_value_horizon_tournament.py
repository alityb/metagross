import unittest

from srcs.metagross.value_horizon_tournament import _seed, _select_action


class ValueHorizonTournamentTest(unittest.TestCase):
    def test_seed_is_candidate_independent(self):
        identity = ("battle", "p1", 3)
        self.assertEqual(_seed(identity, 4, 1, 7), _seed(identity, 4, 1, 7))
        self.assertNotEqual(_seed(identity, 4, 1, 7), _seed(identity, 8, 1, 7))

    def test_select_action_includes_baseline_zero(self):
        self.assertEqual(_select_action({"bad": -0.1}, "baseline"), "baseline")
        self.assertEqual(_select_action({"good": 0.1}, "baseline"), "good")

    def test_select_action_ties_lexically(self):
        self.assertEqual(_select_action({"z": 0.2, "a": 0.2}, "baseline"), "a")


if __name__ == "__main__":
    unittest.main()
