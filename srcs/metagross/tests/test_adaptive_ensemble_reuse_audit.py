import unittest

from srcs.metagross.adaptive_ensemble_reuse_audit import _cluster_bootstrap


class AdaptiveEnsembleReuseAuditTest(unittest.TestCase):
    def test_cluster_bootstrap_keeps_mirrored_battles_together(self):
        rows = [
            {"identity": {"battle_tag": "a1"}, "root_delta": 0.1},
            {"identity": {"battle_tag": "a2"}, "root_delta": -0.1},
            {"identity": {"battle_tag": "b1"}, "root_delta": 0.2},
            {"identity": {"battle_tag": "b2"}, "root_delta": 0.2},
        ]
        result = _cluster_bootstrap(
            rows,
            {"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
            seed=7,
            resamples=1000,
        )
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["resamples"], 1000)
        self.assertEqual(result["seed"], 7)
        self.assertGreaterEqual(result["ci95_low"], 0.0)
        self.assertLessEqual(result["ci95_high"], 0.2)


if __name__ == "__main__":
    unittest.main()
