import unittest

from srcs.metagross.shared_root_paired_verifier_experiment import _seed


class SharedRootPairedVerifierExperimentTest(unittest.TestCase):
    def test_seed_is_stable_and_world_specific(self):
        identity = {"battle_tag": "battle-1", "decision_idx": 2}
        first = _seed(identity, 0, 0)
        self.assertEqual(first, _seed(identity, 0, 0))
        self.assertNotEqual(first, _seed(identity, 0, 1))
        self.assertNotEqual(first, _seed(identity, 1, 0))


if __name__ == "__main__":
    unittest.main()
