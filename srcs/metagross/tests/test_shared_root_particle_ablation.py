import unittest

from srcs.metagross.shared_root_particle_ablation import _average_policy


class SharedRootParticleAblationTest(unittest.TestCase):
    def test_average_policy_handles_different_supports(self):
        rows = [
            {"policy": {"a": 0.75, "b": 0.25}},
            {"policy": {"a": 0.5, "c": 0.5}},
        ]
        self.assertEqual(_average_policy(rows), {"a": 0.625, "b": 0.125, "c": 0.25})


if __name__ == "__main__":
    unittest.main()
