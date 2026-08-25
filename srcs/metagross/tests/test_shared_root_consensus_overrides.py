import unittest

from srcs.metagross.shared_root_consensus_overrides import _override


class SharedRootConsensusOverridesTest(unittest.TestCase):
    def test_consensus_requires_every_named_selector(self):
        actions = {
            "rm_policy_argmax": "a",
            "opponent_prior_expected": "a",
            "worst_case_endpoint": "b",
            "bounded_robust_0.10": "a",
            "bounded_robust_0.25": "a",
            "bounded_robust_0.50": "a",
        }
        self.assertEqual(_override("rm_prior_consensus", actions, "direct"), "a")
        self.assertEqual(_override("rm_prior_worst_consensus", actions, "direct"), "direct")

    def test_majority_must_be_strict(self):
        actions = {
            "rm_policy_argmax": "a",
            "opponent_prior_expected": "a",
            "worst_case_endpoint": "a",
            "bounded_robust_0.10": "b",
            "bounded_robust_0.25": "b",
            "bounded_robust_0.50": "b",
        }
        self.assertEqual(_override("strict_search_majority", actions, "direct"), "direct")


if __name__ == "__main__":
    unittest.main()
